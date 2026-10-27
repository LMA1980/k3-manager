//! Dedicated spare recovery watchdog thread and runaway workload intervention (US-06.3, ADR-007, ADR-041).
//!
//! Supervises compute workers and control-plane tasks from a dedicated watchdog thread pinned to
//! the reserved spare recovery capacity (`spare_recovery_mask`, defaulting to `X100` core 5 / `0x0020`).
//!
//! Automatically detects stalled compute workers (missed heartbeats or pass timeouts) and runaway
//! control tasks (expired task leases), preempting, terminating (`SIGTERM` -> `SIGKILL`), reaping,
//! respawning replacement workers reattached to surviving `k3-kv` arenas via `SCM_RIGHTS`,
//! and recording structured recovery audit events without requiring a physical board reboot (US-06 Scenario 2).

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use k3_core::affinity::ResolvedControlCapacity;

use crate::domain::{OsAffinityBackend, RealOsAffinityBackend};
use crate::error::ManagerError;
use crate::runtime::ControlPlaneRuntime;
use crate::supervisor::{
    DEFAULT_HEARTBEAT_TIMEOUT, DEFAULT_PASS_TIMEOUT, ManagedWorker, WorkerSupervisor,
};

/// Target entity subject to recovery intervention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryTarget {
    /// Compute worker running an isolated model inference engine.
    ComputeWorker {
        /// Logical model identifier.
        model_id: String,
        /// OS process ID before recovery.
        pid: Option<u32>,
    },
    /// Control-plane task scheduled on an active control core.
    ControlTask {
        /// Unique task identifier.
        task_id: u64,
        /// Physical core index where task stalled.
        core_id: usize,
    },
}

impl fmt::Display for RecoveryTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ComputeWorker { model_id, pid } => {
                write!(f, "compute worker '{model_id}' (pid: {pid:?})")
            }
            Self::ControlTask { task_id, core_id } => {
                write!(f, "control task {task_id} on core {core_id}")
            }
        }
    }
}

/// Recovery action taken by the health supervisor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Stalled or crashed compute worker terminated, reaped, and replaced with surviving arena.
    WorkerTerminatedAndRespawned {
        /// OS process ID of old terminated worker.
        old_pid: Option<u32>,
        /// OS process ID of newly spawned replacement worker.
        new_pid: Option<u32>,
        /// Generation counter of replacement worker.
        generation: u64,
    },
    /// Stalled control task cooperatively cancelled and its worker thread replaced.
    ControlTaskPreempted {
        /// Preempted task identifier.
        task_id: u64,
        /// Whether worker thread on assigned core was replaced.
        thread_replaced: bool,
    },
}

impl fmt::Display for RecoveryAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkerTerminatedAndRespawned {
                old_pid,
                new_pid,
                generation,
            } => {
                write!(
                    f,
                    "worker terminated (pid {old_pid:?}) and respawned (new pid {new_pid:?}, generation {generation})"
                )
            }
            Self::ControlTaskPreempted {
                task_id,
                thread_replaced,
            } => {
                write!(
                    f,
                    "task {task_id} preempted (thread_replaced: {thread_replaced})"
                )
            }
        }
    }
}

/// Structured audit event recorded when the health supervisor executes recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryAuditEvent {
    /// Timestamp when recovery intervention occurred.
    pub timestamp: Instant,
    /// Target workload subject to recovery.
    pub target: RecoveryTarget,
    /// Diagnostic reason triggering recovery.
    pub reason: String,
    /// Specific recovery action executed.
    pub action: RecoveryAction,
}

impl fmt::Display for RecoveryAuditEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:?}] RECOVERY: {} - Reason: {} -> Action: {}",
            self.timestamp, self.target, self.reason, self.action
        )
    }
}

/// Configuration parameters for [`HealthSupervisor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthSupervisorConfig {
    /// Bitmask of dedicated spare recovery core (default `0x0020` / core 5 per ADR-007).
    pub spare_recovery_mask: u64,
    /// Watchdog polling interval.
    pub check_interval: Duration,
    /// Heartbeat probe timeout.
    pub heartbeat_timeout: Duration,
    /// Forward pass timeout.
    pub pass_timeout: Duration,
    /// Control-plane task lease deadline.
    pub task_lease_duration: Duration,
}

impl Default for HealthSupervisorConfig {
    fn default() -> Self {
        Self {
            spare_recovery_mask: 0x0020,
            check_interval: Duration::from_millis(50),
            heartbeat_timeout: DEFAULT_HEARTBEAT_TIMEOUT,
            pass_timeout: DEFAULT_PASS_TIMEOUT,
            task_lease_duration: Duration::from_secs(5),
        }
    }
}

/// Dedicated health supervisor running on reserved spare recovery capacity (ADR-007, ADR-041).
#[derive(Debug)]
pub struct HealthSupervisor {
    config: HealthSupervisorConfig,
    capacity: ResolvedControlCapacity,
    affinity_backend: Arc<dyn OsAffinityBackend>,
    worker_supervisor: Option<Arc<Mutex<WorkerSupervisor>>>,
    control_runtime: Option<Arc<ControlPlaneRuntime>>,
    audit_events: Arc<Mutex<Vec<RecoveryAuditEvent>>>,
    running: Arc<AtomicBool>,
    watchdog_handle: Option<JoinHandle<()>>,
}

impl HealthSupervisor {
    /// Creates a new [`HealthSupervisor`] bound to resolved capacity and real OS affinity.
    ///
    /// # Errors
    /// Returns [`ManagerError`] if configuration is invalid.
    pub fn new(capacity: ResolvedControlCapacity) -> Result<Self, ManagerError> {
        Self::with_affinity_backend(capacity, Arc::new(RealOsAffinityBackend))
    }

    /// Creates a new [`HealthSupervisor`] with custom [`OsAffinityBackend`].
    ///
    /// # Errors
    /// Returns [`ManagerError`] if configuration is invalid.
    pub fn with_affinity_backend(
        capacity: ResolvedControlCapacity,
        affinity_backend: Arc<dyn OsAffinityBackend>,
    ) -> Result<Self, ManagerError> {
        let spare_mask = if capacity.spare_recovery_mask != 0 {
            capacity.spare_recovery_mask
        } else {
            0x0020
        };

        let config = HealthSupervisorConfig {
            spare_recovery_mask: spare_mask,
            ..Default::default()
        };

        Ok(Self {
            config,
            capacity,
            affinity_backend,
            worker_supervisor: None,
            control_runtime: None,
            audit_events: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(AtomicBool::new(false)),
            watchdog_handle: None,
        })
    }

    /// Connects a [`WorkerSupervisor`] instance for compute worker monitoring and recovery.
    #[must_use]
    pub fn with_worker_supervisor(mut self, ws: Arc<Mutex<WorkerSupervisor>>) -> Self {
        self.worker_supervisor = Some(ws);
        self
    }

    /// Sets the [`WorkerSupervisor`] instance.
    pub fn set_worker_supervisor(&mut self, ws: Arc<Mutex<WorkerSupervisor>>) {
        self.worker_supervisor = Some(ws);
    }

    /// Connects a [`ControlPlaneRuntime`] instance for control task lease monitoring and thread replacement.
    #[must_use]
    pub fn with_control_runtime(mut self, rt: Arc<ControlPlaneRuntime>) -> Self {
        rt.set_default_lease_duration(self.config.task_lease_duration);
        self.control_runtime = Some(rt);
        self
    }

    /// Sets the [`ControlPlaneRuntime`] instance.
    pub fn set_control_runtime(&mut self, rt: Arc<ControlPlaneRuntime>) {
        rt.set_default_lease_duration(self.config.task_lease_duration);
        self.control_runtime = Some(rt);
    }

    /// Returns a reference to the active [`ResolvedControlCapacity`].
    #[must_use]
    pub const fn capacity(&self) -> &ResolvedControlCapacity {
        &self.capacity
    }

    /// Sets full supervisor configuration using builder syntax.
    #[must_use]
    pub fn with_config(mut self, config: HealthSupervisorConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets watchdog check interval using builder syntax.
    #[must_use]
    pub const fn with_check_interval(mut self, interval: Duration) -> Self {
        self.config.check_interval = interval;
        self
    }

    /// Sets heartbeat timeout using builder syntax.
    #[must_use]
    pub const fn with_heartbeat_timeout(mut self, timeout: Duration) -> Self {
        self.config.heartbeat_timeout = timeout;
        self
    }

    /// Sets forward pass timeout using builder syntax.
    #[must_use]
    pub const fn with_pass_timeout(mut self, timeout: Duration) -> Self {
        self.config.pass_timeout = timeout;
        self
    }

    /// Sets control task lease duration using builder syntax.
    #[must_use]
    pub fn with_task_lease_duration(mut self, duration: Duration) -> Self {
        self.config.task_lease_duration = duration;
        if let Some(ref rt) = self.control_runtime {
            rt.set_default_lease_duration(duration);
        }
        self
    }

    /// Returns the active supervisor configuration.
    #[must_use]
    pub const fn config(&self) -> &HealthSupervisorConfig {
        &self.config
    }

    /// Returns the active spare recovery mask.
    #[must_use]
    pub const fn spare_recovery_mask(&self) -> u64 {
        self.config.spare_recovery_mask
    }

    /// Returns the primary spare recovery core ID if configured.
    #[must_use]
    pub fn spare_recovery_core(&self) -> Option<usize> {
        let mask = self.config.spare_recovery_mask;
        if mask == 0 {
            None
        } else {
            Some(mask.trailing_zeros() as usize)
        }
    }

    /// Returns `true` if the watchdog background thread is actively running.
    #[must_use]
    pub fn is_watchdog_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
            && self
                .watchdog_handle
                .as_ref()
                .is_none_or(|h| !h.is_finished())
    }

    /// Starts the dedicated spare recovery watchdog thread pinned to `spare_recovery_mask`.
    ///
    /// # Errors
    /// Returns [`ManagerError`] if watchdog thread spawning or core affinity application fails.
    pub fn start_watchdog(&mut self) -> Result<(), ManagerError> {
        if self.is_watchdog_running() {
            return Ok(());
        }

        if let Some(old_handle) = self.watchdog_handle.take() {
            let _ = old_handle.join();
        }

        let spare_mask = self.config.spare_recovery_mask;
        let backend = Arc::clone(&self.affinity_backend);
        let running = Arc::clone(&self.running);
        running.store(true, Ordering::SeqCst);

        let ws_opt = self.worker_supervisor.clone();
        let rt_opt = self.control_runtime.clone();
        let audit_events = Arc::clone(&self.audit_events);
        let check_interval = self.config.check_interval;
        let config_clone = self.config.clone();

        let (ready_tx, ready_rx) = sync_channel::<Result<(), ManagerError>>(1);

        let handle = std::thread::Builder::new()
            .name("k3-health-watchdog".to_string())
            .spawn(move || {
                // Pin watchdog thread to spare recovery core (ADR-007, ADR-041)
                let aff_res = if spare_mask != 0 {
                    backend.set_affinity(0, spare_mask)
                } else {
                    Ok(())
                };
                let is_ok = aff_res.is_ok();
                let _ = ready_tx.send(aff_res);
                if !is_ok {
                    return;
                }

                while running.load(Ordering::Relaxed) {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        Self::perform_health_check(
                            &config_clone,
                            ws_opt.as_ref(),
                            rt_opt.as_ref(),
                            &audit_events,
                        );
                    }));
                    std::thread::sleep(check_interval);
                }
                running.store(false, Ordering::SeqCst);
            })
            .map_err(|e| {
                ManagerError::WorkerSpawn(format!(
                    "failed to spawn health watchdog thread for spare core mask 0x{spare_mask:04X}: {e}"
                ))
            })?;

        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                self.running.store(false, Ordering::SeqCst);
                let _ = handle.join();
                return Err(e);
            }
            Err(_) => {
                self.running.store(false, Ordering::SeqCst);
                let _ = handle.join();
                return Err(ManagerError::WorkerSpawn(format!(
                    "watchdog thread for spare core mask 0x{spare_mask:04X} panicked during initialization"
                )));
            }
        }
        self.watchdog_handle = Some(handle);
        Ok(())
    }

    /// Stops the watchdog thread cleanly.
    pub fn stop_watchdog(&mut self) {
        if self.running.swap(false, Ordering::SeqCst) {
            if let Some(handle) = self.watchdog_handle.take() {
                let _ = handle.join();
            }
        }
    }

    /// Executes a single synchronous health check and recovery cycle across registered workers and tasks.
    #[must_use]
    pub fn check_once(&self) -> Vec<RecoveryAuditEvent> {
        Self::perform_health_check(
            &self.config,
            self.worker_supervisor.as_ref(),
            self.control_runtime.as_ref(),
            &self.audit_events,
        )
    }

    /// Returns a copy of all recorded recovery audit events.
    #[must_use]
    pub fn audit_events(&self) -> Vec<RecoveryAuditEvent> {
        let events = self
            .audit_events
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        events.clone()
    }

    /// Clears recorded audit events.
    pub fn clear_audit_events(&self) {
        let mut events = self
            .audit_events
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        events.clear();
    }

    // ========================================================================
    // Internal Health Supervision & Recovery Engine
    // ========================================================================

    #[allow(clippy::too_many_lines)]
    fn perform_health_check(
        config: &HealthSupervisorConfig,
        worker_supervisor: Option<&Arc<Mutex<WorkerSupervisor>>>,
        control_runtime: Option<&Arc<ControlPlaneRuntime>>,
        audit_events: &Arc<Mutex<Vec<RecoveryAuditEvent>>>,
    ) -> Vec<RecoveryAuditEvent> {
        let mut new_events = Vec::new();

        // 1. Monitor Control-Plane Task Health (ADR-007, ADR-041)
        if let Some(rt) = control_runtime {
            let preempted = rt.preempt_expired_tasks();
            for (task_id, core_id) in preempted {
                let event = RecoveryAuditEvent {
                    timestamp: Instant::now(),
                    target: RecoveryTarget::ControlTask { task_id, core_id },
                    reason: format!(
                        "control task {task_id} on core {core_id} exceeded lease deadline"
                    ),
                    action: RecoveryAction::ControlTaskPreempted {
                        task_id,
                        thread_replaced: true,
                    },
                };
                new_events.push(event);
            }
        }

        // 2. Monitor Compute Worker Health (ADR-018, ADR-041)
        if let Some(ws_lock) = worker_supervisor {
            let Ok(mut ws) = ws_lock.try_lock() else {
                // Lock held by client dispatch thread; append any control task events and return
                if !new_events.is_empty() {
                    let mut log = audit_events.lock().unwrap_or_else(PoisonError::into_inner);
                    log.extend(new_events.clone());
                }
                return new_events;
            };

            let model_ids = ws.model_ids();
            for model_id in model_ids {
                let mut needs_recovery = false;
                let mut failure_reason = String::new();

                if let Some(worker) = ws.get_worker_mut(&model_id) {
                    let effective_pass_timeout = worker
                        .placement()
                        .pass_timeout
                        .unwrap_or(config.pass_timeout);
                    let effective_heartbeat_timeout = worker
                        .placement()
                        .heartbeat_timeout
                        .unwrap_or(config.heartbeat_timeout);
                    worker.set_pass_timeout(effective_pass_timeout);
                    worker.set_heartbeat_timeout(effective_heartbeat_timeout);

                    if !worker.is_alive() {
                        needs_recovery = true;
                        failure_reason = "worker process terminated unexpectedly".to_string();
                    } else if worker.has_timed_out() {
                        needs_recovery = true;
                        failure_reason = "worker socket I/O deadline breached".to_string();
                    } else if let Some(elapsed) = worker.active_pass_elapsed() {
                        if elapsed > effective_pass_timeout {
                            needs_recovery = true;
                            failure_reason = format!(
                                "forward pass deadline exceeded (elapsed {}ms > {}ms limit)",
                                elapsed.as_millis(),
                                effective_pass_timeout.as_millis()
                            );
                        }
                    } else {
                        // Forward pass is not active; send heartbeat probe
                        match worker.send_heartbeat() {
                            Ok(_) => {}
                            Err(ManagerError::WorkerUnresponsive { elapsed_ms, .. }) => {
                                needs_recovery = true;
                                failure_reason =
                                    format!("heartbeat probe timed out after {elapsed_ms}ms");
                            }
                            Err(e) => {
                                needs_recovery = true;
                                failure_reason = format!("heartbeat probe failed: {e}");
                            }
                        }
                    }
                }

                if needs_recovery {
                    let old_pid = ws.get_worker(&model_id).and_then(ManagedWorker::pid);

                    // Preempt/terminate runaway worker process: SIGTERM -> wait -> SIGKILL
                    if let Some(worker) = ws.get_worker_mut(&model_id) {
                        if worker.is_alive() {
                            let _ = worker.kill(libc::SIGTERM);
                            // Also send SIGCONT so stopped/paused workers receive SIGTERM immediately
                            let _ = worker.kill(libc::SIGCONT);
                            let start_kill = Instant::now();
                            while worker.is_alive()
                                && start_kill.elapsed() < Duration::from_millis(50)
                            {
                                std::thread::sleep(Duration::from_millis(5));
                            }
                            if worker.is_alive() {
                                let _ = worker.kill(libc::SIGKILL);
                            }
                        }
                    }

                    // Reap old worker process
                    let _ = ws.reap_worker(&model_id);

                    // Recover worker: spawn fresh process re-attached to surviving k3-kv arena via SCM_RIGHTS
                    match ws.recover_worker(&model_id) {
                        Ok(recovered) => {
                            let new_pid = recovered.pid();
                            let new_gen = recovered.generation();
                            let event = RecoveryAuditEvent {
                                timestamp: Instant::now(),
                                target: RecoveryTarget::ComputeWorker {
                                    model_id: model_id.clone(),
                                    pid: old_pid,
                                },
                                reason: failure_reason,
                                action: RecoveryAction::WorkerTerminatedAndRespawned {
                                    old_pid,
                                    new_pid,
                                    generation: new_gen,
                                },
                            };
                            new_events.push(event);

                            if let Some(rt) = control_runtime {
                                rt.update_worker_health_with_pid(&model_id, new_pid, true, new_gen);
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "[health supervisor] failed to recover worker '{model_id}': {e}"
                            );
                            if let Some(rt) = control_runtime {
                                rt.update_worker_health_with_pid(&model_id, None, false, 0);
                            }
                        }
                    }
                }
            }
        }

        if !new_events.is_empty() {
            let mut log = audit_events.lock().unwrap_or_else(PoisonError::into_inner);
            log.extend(new_events.clone());
        }

        new_events
    }
}

impl Drop for HealthSupervisor {
    fn drop(&mut self) {
        self.stop_watchdog();
    }
}
