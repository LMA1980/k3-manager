//! Reserved active control thread pool and non-blocking administrative/queue status plane (US-06.2.3, ADR-007, ADR-041).
//!
//! Decouples control-plane operations (chat template formatting, prompt tokenization,
//! prefix radix lookup, stop-sequence detection, queue depth, active model placements,
//! worker health snapshots, prefix cache stats, and administrative commands) from compute workers
//! so control operations complete immediately on dedicated active control threads (`active_control_mask`,
//! default 2 threads on `X100` cores 6–7) even when compute workers are 100% saturated executing forward passes.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, channel, sync_channel};
use std::sync::{Arc, PoisonError, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::capacity::ControlPlaneScheduler;
use crate::domain::{OsAffinityBackend, RealOsAffinityBackend};
use crate::error::ManagerError;
use crate::persistence::{
    DiagnosticReplayResult, PersistenceError, PersistenceStore, SqlitePersistenceStore,
};
use crate::prefix::{PrefixCacheStats, PrefixMatchResult, PrefixRadixIndex};
use crate::prompt::{
    ChatTemplate, GgufTokenizer, IncrementalStopDetector, PreparedPrompt, PromptOrchestrator,
    PromptPreparationRequest, SpecialTokens, Tokenizer,
};
use crate::queue::AdmissionQueueManager;
use crate::router::DecisionRouter;
use crate::scheduler::MultiModelScheduler;
use std::time::{SystemTime, UNIX_EPOCH};

use k3_core::affinity::ResolvedControlCapacity;
pub use k3_core::profile::{
    CONSERVATIVE_DEFAULT_A100_CORES, CONSERVATIVE_DEFAULT_N_CTX, CONSERVATIVE_DEFAULT_X100_CORES,
    CONSERVATIVE_TUNING_ADVISORY, CONSTRAINED_TEST_N_CTX, ComputePool, ModelEntry,
    ModelListResponse, ModelLoadAuditEvent, ModelProfile, PriorityClass, ProfileStatus,
    TuningProfile,
};
use serde::{Deserialize, Serialize};

fn current_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn current_timestamp_ms() -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Health and status snapshot of a managed compute worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerHealthSnapshot {
    /// Logical model identifier.
    pub model_id: String,
    /// Operating system process ID if running.
    pub pid: Option<u32>,
    /// Compute cluster target (`"x100"` or `"a100"`).
    pub cluster: String,
    /// Whether worker is actively alive.
    pub is_alive: bool,
    /// Generation count (incremented across recoveries).
    pub generation: u64,
}

/// Placement, status, and tuning metadata for a model managed in the control plane.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPlacementRecord {
    /// Logical model identifier.
    pub model_id: String,
    /// Path to model weights file, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights_path: Option<PathBuf>,
    /// Compute cluster target (`"x100"`, `"a100"`, `"both"`).
    pub cluster: String,
    /// Operating system process ID if running.
    pub pid: Option<u32>,
    /// Whether worker is actively alive.
    pub is_alive: bool,
    /// Generation counter (incremented on recoveries).
    pub generation: u64,
    /// Authoritative tuning profile.
    pub tuning: TuningProfile,
    /// Explicit boolean flag tagging unprofiled models.
    pub unprofiled: bool,
    /// Profile status string ("unprofiled" or "profiled").
    pub profile_status: String,
    /// Active scheduling priority class ("throughput" or "latency").
    pub priority_class: String,
    /// Context window limit in tokens.
    pub n_ctx: usize,
    /// Whether dynamic idle core lending is enabled.
    pub lend_idle_cores: bool,
    /// Whether dynamic idle core borrowing is enabled.
    pub borrow_idle_cores: bool,
    /// Actionable advisory notice when unprofiled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advisory_notice: Option<String>,
    /// Unix creation timestamp in seconds.
    pub created: u64,
}

impl ModelPlacementRecord {
    /// Creates a new model placement record with conservative unprofiled defaults (ADR-028, ADR-031).
    pub fn new_unprofiled(
        model_id: impl Into<String>,
        weights_path: Option<PathBuf>,
        cluster: impl Into<String>,
        pid: Option<u32>,
    ) -> Self {
        let model_id_str = model_id.into();
        let mut tuning = TuningProfile::conservative_default();
        tuning.model_id = Some(model_id_str.clone());
        let unprofiled = tuning.unprofiled;
        let profile_status = tuning.profile_status.as_str().to_owned();
        let priority_class = tuning.priority_class.as_str().to_owned();
        let n_ctx = tuning.n_ctx;
        let lend_idle_cores = tuning.lend_idle_cores;
        let borrow_idle_cores = tuning.borrow_idle_cores;
        let advisory_notice = tuning.advisory_notice.clone();
        let created = current_timestamp_secs();

        Self {
            model_id: model_id_str,
            weights_path,
            cluster: cluster.into(),
            pid,
            is_alive: true,
            generation: 1,
            tuning,
            unprofiled,
            profile_status,
            priority_class,
            n_ctx,
            lend_idle_cores,
            borrow_idle_cores,
            advisory_notice,
            created,
        }
    }

    /// Creates a new model placement record with an explicit tuning profile.
    pub fn with_tuning(
        model_id: impl Into<String>,
        weights_path: Option<PathBuf>,
        pid: Option<u32>,
        mut tuning: TuningProfile,
    ) -> Self {
        let model_id_str = model_id.into();
        tuning.model_id = Some(model_id_str.clone());
        let cluster = tuning.cluster.clone();
        let unprofiled = tuning.unprofiled;
        let profile_status = tuning.profile_status.as_str().to_owned();
        let priority_class = tuning.priority_class.as_str().to_owned();
        let n_ctx = tuning.n_ctx;
        let lend_idle_cores = tuning.lend_idle_cores;
        let borrow_idle_cores = tuning.borrow_idle_cores;
        let advisory_notice = tuning.advisory_notice.clone();
        let created = current_timestamp_secs();

        Self {
            model_id: model_id_str,
            weights_path,
            cluster,
            pid,
            is_alive: true,
            generation: 1,
            tuning,
            unprofiled,
            profile_status,
            priority_class,
            n_ctx,
            lend_idle_cores,
            borrow_idle_cores,
            advisory_notice,
            created,
        }
    }

    /// Converts this placement record to an `OpenAI` `/v1/models`-compatible [`ModelEntry`].
    #[must_use]
    pub fn to_model_entry(&self) -> ModelEntry {
        ModelEntry {
            id: self.model_id.clone(),
            object: "model".to_owned(),
            created: self.created,
            owned_by: "k3".to_owned(),
            unprofiled: self.unprofiled,
            profile_status: self.profile_status.clone(),
            priority_class: self.priority_class.clone(),
            cluster: self.cluster.clone(),
            a100_cores: self.tuning.a100_cores,
            n_ctx: self.n_ctx,
            lend_idle_cores: self.lend_idle_cores,
            borrow_idle_cores: self.borrow_idle_cores,
            advisory_notice: self.advisory_notice.clone(),
        }
    }
}

/// Category of operational audit event for model lifecycle actions (US-07 R2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAuditEventType {
    /// Model registered into control plane registry.
    ModelRegistered,
    /// Model loaded and placement recorded.
    ModelLoaded,
    /// Model status queried via control plane or /v1/models.
    ModelStatusQueried,
    /// Operator applied a tuning profile, transitioning from unprofiled defaults.
    TuningProfileApplied,
}

/// Structured operational audit event for model registration, load, query, and tuning transitions (US-07 R2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelAuditEvent {
    /// Event timestamp in milliseconds since Unix epoch.
    pub timestamp_ms: u64,
    /// Type of lifecycle audit event.
    pub event_type: ModelAuditEventType,
    /// Logical model identifier.
    pub model_id: String,
    /// Whether the model is unprofiled at the time of the event.
    pub unprofiled: bool,
    /// Profile status string ("unprofiled" or "profiled").
    pub profile_status: String,
    /// Target processor cluster ("x100", "a100", "both").
    pub cluster: String,
    /// Scheduling priority class ("throughput" or "latency").
    pub priority_class: String,
    /// Context window bound in tokens.
    pub n_ctx: usize,
    /// Actionable advisory notice when unprofiled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advisory_notice: Option<String>,
    /// Human-readable event description and context.
    pub details: String,
}

impl ModelAuditEvent {
    /// Returns the advisory notice, if any.
    #[must_use]
    pub fn advisory(&self) -> Option<&str> {
        self.advisory_notice.as_deref()
    }
}

/// Administrative command submitted to the control-plane runtime plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdminCommand {
    /// Simple ping to verify control-plane responsiveness.
    Ping,
    /// Inspect resolved control and compute capacity.
    InspectCapacity,
    /// Query current admission queue depth.
    GetQueueDepth,
    /// Query list of active model placement identifiers.
    GetActivePlacements,
    /// Query health snapshots of all supervised compute workers.
    GetWorkerHealth,
    /// Query prefix cache hit/miss and memory statistics.
    GetPrefixCacheStats,
    /// Clear prefix cache entries for a project and/or model.
    ClearPrefixCache {
        /// Optional project filter.
        project_id: Option<String>,
        /// Optional model filter.
        model_id: Option<String>,
    },
    /// Register model placement metadata into control plane registry.
    RegisterModel {
        /// Model identifier.
        model_id: String,
        /// Path to weights file.
        weights_path: PathBuf,
        /// Target compute cluster.
        cluster: String,
    },
    /// Unregister model placement metadata.
    UnregisterModel {
        /// Model identifier.
        model_id: String,
    },
    /// Query detailed model status and profile entry.
    GetModelStatus {
        /// Model identifier.
        model_id: String,
    },
    /// Query all registered models as `/v1/models` listing entries.
    ListModels,
    /// Apply an operator tuning profile to transition an unprofiled model (ADR-031, ADR-032).
    ApplyTuningProfile {
        /// Model identifier.
        model_id: String,
        /// Operator tuning profile configuration.
        profile: TuningProfile,
    },
    /// Query operational model audit event history.
    GetAuditEvents,
    /// Generate advisory tuning recommendations based on live metrics and hardware ceilings (US-08 Scenario 2).
    SuggestTuning {
        /// Model identifier.
        model_id: String,
    },
    /// Replay persisted last interaction of a session for diagnostic verification (US-17).
    ReplaySessionLastTurn {
        /// Session identifier to replay.
        session_id: String,
    },
}

/// Administrative response returned by the control-plane runtime.
#[derive(Debug, Clone, PartialEq)]
pub enum AdminResponse {
    /// Ping acknowledgment.
    Pong,
    /// Resolved capacity details.
    Capacity(ResolvedControlCapacity),
    /// Current queue depth.
    QueueDepth(usize),
    /// Active model placement IDs.
    ActivePlacements(Vec<String>),
    /// Worker health snapshot list.
    WorkerHealth(Vec<WorkerHealthSnapshot>),
    /// Prefix cache statistics.
    PrefixCacheStats(PrefixCacheStats),
    /// Count of cleared prefix cache blocks.
    CacheCleared(usize),
    /// Confirmation of model registration.
    ModelRegistered(String),
    /// Confirmation of model unregistration.
    ModelUnregistered(String),
    /// Model status and listing entry.
    ModelStatus(ModelEntry),
    /// List of model listing entries.
    ModelList(Vec<ModelEntry>),
    /// Confirmation that operator tuning profile was applied.
    TuningProfileApplied(String),
    /// Model operational audit event history.
    AuditEvents(Vec<ModelAuditEvent>),
    /// Tuning advisory recommendation generated on demand (US-08 Scenario 2).
    TuningAdvisory(k3_core::profile::TuningAdvisory),
    /// Diagnostic replay result containing matched status and replayed response (US-17).
    DiagnosticReplay(DiagnosticReplayResult),
}

/// Default control task lease duration (5 seconds).
pub const DEFAULT_CONTROL_TASK_LEASE: Duration = Duration::from_secs(5);

/// Active control-plane task lease metadata.
#[derive(Debug)]
pub struct ActiveTaskRecord {
    /// Unique task identifier.
    pub task_id: u64,
    /// Physical core ID where task was dispatched.
    pub core_id: usize,
    /// Start timestamp of task execution.
    pub start_time: Instant,
    /// Lease expiration deadline.
    pub deadline: Instant,
    /// Cooperative cancellation token.
    pub cancellation_token: Arc<AtomicBool>,
}

/// Snapshot of an active control task lease.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskLeaseSnapshot {
    /// Unique task identifier.
    pub task_id: u64,
    /// Core index where task was scheduled.
    pub core_id: usize,
    /// Elapsed execution time in milliseconds.
    pub elapsed_ms: u64,
    /// Remaining lease time in milliseconds (negative if expired).
    pub remaining_ms: i64,
    /// Whether cooperative cancellation has been requested.
    pub is_cancelled: bool,
}

struct ControlWorkerThread {
    core_id: usize,
    sender: Sender<Box<dyn FnOnce(usize) + Send>>,
    handle: Option<JoinHandle<()>>,
}

impl fmt::Debug for ControlWorkerThread {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ControlWorkerThread")
            .field("core_id", &self.core_id)
            .finish_non_exhaustive()
    }
}

/// Non-blocking control-plane runtime maintaining active control worker threads (ADR-007, ADR-041).
#[derive(Debug)]
pub struct ControlPlaneRuntime {
    scheduler: ControlPlaneScheduler,
    affinity_backend: Arc<dyn OsAffinityBackend>,
    prefix_index: Arc<PrefixRadixIndex>,
    prompt_orchestrator: Arc<PromptOrchestrator>,
    queue_depth: Arc<AtomicUsize>,
    placements: Arc<RwLock<HashMap<String, ModelPlacementRecord>>>,
    control_workers: Arc<RwLock<Vec<ControlWorkerThread>>>,
    active_tasks: Arc<RwLock<HashMap<u64, ActiveTaskRecord>>>,
    next_task_id: Arc<AtomicU64>,
    default_lease_duration: Arc<RwLock<Duration>>,
    audit_events: Arc<RwLock<Vec<ModelAuditEvent>>>,
    compute_scheduler: Arc<MultiModelScheduler>,
    router: Arc<DecisionRouter>,
    queue_mgr: Arc<AdmissionQueueManager>,
    persistence: Arc<dyn PersistenceStore>,
}

impl ControlPlaneRuntime {
    /// Creates a new [`ControlPlaneRuntime`] using default capacity and the real OS affinity backend.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if thread spawning or capacity resolution fails.
    pub fn new(capacity: ResolvedControlCapacity) -> Result<Self, ManagerError> {
        Self::with_affinity_backend(capacity, Arc::new(RealOsAffinityBackend))
    }

    /// Creates a new [`ControlPlaneRuntime`] using a custom [`OsAffinityBackend`].
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if control worker thread spawning fails.
    #[allow(clippy::too_many_lines)]
    pub fn with_affinity_backend(
        capacity: ResolvedControlCapacity,
        affinity_backend: Arc<dyn OsAffinityBackend>,
    ) -> Result<Self, ManagerError> {
        let active_cores = capacity.active_control_cores.clone();
        let mut control_workers: Vec<ControlWorkerThread> = Vec::with_capacity(active_cores.len());

        for &core_id in &active_cores {
            let (tx, rx) = channel::<Box<dyn FnOnce(usize) + Send>>();
            let (ready_tx, ready_rx) = sync_channel::<Result<(), ManagerError>>(1);
            let backend_clone = Arc::clone(&affinity_backend);
            let thread_name = format!("k3-control-{core_id}");

            let handle = std::thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    // Apply OS affinity for this active control core (ADR-007, ADR-041)
                    let mask = 1_u64 << core_id;
                    let aff_res = backend_clone.set_affinity(0, mask);
                    let is_ok = aff_res.is_ok();
                    let _ = ready_tx.send(aff_res);
                    if !is_ok {
                        return;
                    }

                    while let Ok(task) = rx.recv() {
                        task(core_id);
                    }
                })
                .map_err(|e| {
                    ManagerError::WorkerSpawn(format!(
                        "failed to spawn control worker thread for core {core_id}: {e}"
                    ))
                })?;

            // Await thread initialization and affinity application
            match ready_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    let _ = handle.join();
                    for mut w in control_workers {
                        drop(w.sender);
                        if let Some(h) = w.handle.take() {
                            let _ = h.join();
                        }
                    }
                    return Err(e);
                }
                Err(_) => {
                    let _ = handle.join();
                    for mut w in control_workers {
                        drop(w.sender);
                        if let Some(h) = w.handle.take() {
                            let _ = h.join();
                        }
                    }
                    return Err(ManagerError::WorkerSpawn(format!(
                        "control worker thread for core {core_id} panicked during initialization"
                    )));
                }
            }

            control_workers.push(ControlWorkerThread {
                core_id,
                sender: tx,
                handle: Some(handle),
            });
        }

        let compute_scheduler = Arc::new(MultiModelScheduler::new(capacity.clone()));
        let router = Arc::new(DecisionRouter::default());
        let queue_mgr = Arc::new(AdmissionQueueManager::default());
        let persistence: Arc<dyn PersistenceStore> = Arc::new(
            SqlitePersistenceStore::open_in_memory()
                .map_err(|e| ManagerError::WorkerSpawn(format!("sqlite init error: {e}")))?,
        );

        Ok(Self {
            scheduler: ControlPlaneScheduler::new(capacity),
            affinity_backend,
            prefix_index: Arc::new(PrefixRadixIndex::new()),
            prompt_orchestrator: Arc::new(PromptOrchestrator::new()),
            queue_depth: Arc::new(AtomicUsize::new(0)),
            placements: Arc::new(RwLock::new(HashMap::new())),
            control_workers: Arc::new(RwLock::new(control_workers)),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            next_task_id: Arc::new(AtomicU64::new(1)),
            default_lease_duration: Arc::new(RwLock::new(DEFAULT_CONTROL_TASK_LEASE)),
            audit_events: Arc::new(RwLock::new(Vec::new())),
            compute_scheduler,
            router,
            queue_mgr,
            persistence,
        })
    }

    /// Returns a reference to the active [`MultiModelScheduler`].
    #[must_use]
    pub fn compute_scheduler(&self) -> &Arc<MultiModelScheduler> {
        &self.compute_scheduler
    }

    /// Sets the custom [`MultiModelScheduler`] using builder syntax.
    #[must_use]
    pub fn with_compute_scheduler(mut self, scheduler: Arc<MultiModelScheduler>) -> Self {
        self.compute_scheduler = scheduler;
        self
    }

    /// Sets the custom [`MultiModelScheduler`].
    pub fn set_compute_scheduler(&mut self, scheduler: Arc<MultiModelScheduler>) {
        self.compute_scheduler = scheduler;
    }

    /// Returns a reference to the active [`DecisionRouter`].
    #[must_use]
    pub fn router(&self) -> &Arc<DecisionRouter> {
        &self.router
    }

    /// Sets the custom [`DecisionRouter`].
    pub fn set_router(&mut self, router: Arc<DecisionRouter>) {
        self.router = router;
    }

    /// Returns a reference to the active [`AdmissionQueueManager`].
    #[must_use]
    pub fn queue_mgr(&self) -> &Arc<AdmissionQueueManager> {
        &self.queue_mgr
    }

    /// Sets the custom [`AdmissionQueueManager`].
    pub fn set_queue_mgr(&mut self, queue_mgr: Arc<AdmissionQueueManager>) {
        self.queue_mgr = queue_mgr;
    }

    /// Returns a reference to the active [`PersistenceStore`].
    #[must_use]
    pub fn persistence(&self) -> &Arc<dyn PersistenceStore> {
        &self.persistence
    }

    /// Replaces the active persistence store (e.g. for on-disk persistence).
    pub fn set_persistence(&mut self, store: Arc<dyn PersistenceStore>) {
        self.persistence = store;
    }

    /// Returns the number of active control worker threads.
    #[must_use]
    pub fn active_thread_count(&self) -> usize {
        let workers = self
            .control_workers
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        workers.len()
    }

    /// Returns the active control physical core bitmask.
    #[must_use]
    pub const fn active_control_mask(&self) -> u64 {
        self.scheduler.capacity().active_control_mask
    }

    /// Returns the list of active control core indices.
    #[must_use]
    pub fn active_control_cores(&self) -> &[usize] {
        &self.scheduler.capacity().active_control_cores
    }

    /// Returns a reference to the underlying [`ControlPlaneScheduler`].
    #[must_use]
    pub const fn scheduler(&self) -> &ControlPlaneScheduler {
        &self.scheduler
    }

    /// Returns a reference to the active [`PrefixRadixIndex`].
    #[must_use]
    pub fn prefix_index(&self) -> &Arc<PrefixRadixIndex> {
        &self.prefix_index
    }

    /// Returns a reference to the active [`PromptOrchestrator`].
    #[must_use]
    pub fn prompt_orchestrator(&self) -> &Arc<PromptOrchestrator> {
        &self.prompt_orchestrator
    }

    /// Returns a reference to the active [`OsAffinityBackend`].
    #[must_use]
    pub fn affinity_backend(&self) -> &Arc<dyn OsAffinityBackend> {
        &self.affinity_backend
    }

    // ========================================================================
    // Task Dispatching on Dedicated Active Control Threads (ADR-007)
    // ========================================================================

    /// Returns the default control task lease duration.
    #[must_use]
    pub fn default_lease_duration(&self) -> Duration {
        *self
            .default_lease_duration
            .read()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Sets the default control task lease duration.
    pub fn set_default_lease_duration(&self, duration: Duration) {
        *self
            .default_lease_duration
            .write()
            .unwrap_or_else(PoisonError::into_inner) = duration;
    }

    /// Sets the default control task lease duration using builder syntax.
    #[must_use]
    pub fn with_default_lease_duration(self, duration: Duration) -> Self {
        self.set_default_lease_duration(duration);
        self
    }

    /// Returns a list of active task lease snapshots.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn active_tasks_snapshot(&self) -> Vec<TaskLeaseSnapshot> {
        let tasks = self
            .active_tasks
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        let now = Instant::now();
        tasks
            .values()
            .map(|t| {
                let elapsed_ms = t.start_time.elapsed().as_millis() as u64;
                let remaining_ms = if now >= t.deadline {
                    -(now.duration_since(t.deadline).as_millis() as i64)
                } else {
                    t.deadline.duration_since(now).as_millis() as i64
                };
                TaskLeaseSnapshot {
                    task_id: t.task_id,
                    core_id: t.core_id,
                    elapsed_ms,
                    remaining_ms,
                    is_cancelled: t.cancellation_token.load(Ordering::Relaxed),
                }
            })
            .collect()
    }

    /// Signals cooperative cancellation for the specified task ID.
    pub fn cancel_task(&self, task_id: u64) -> bool {
        let tasks = self
            .active_tasks
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(t) = tasks.get(&task_id) {
            t.cancellation_token.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Replaces the active control worker thread for `core_id` with a clean thread bound to that core.
    ///
    /// # Errors
    /// Returns [`ManagerError`] if thread spawning fails.
    pub fn replace_worker_thread(&self, core_id: usize) -> Result<(), ManagerError> {
        let (tx, rx) = channel::<Box<dyn FnOnce(usize) + Send>>();
        let (ready_tx, ready_rx) = sync_channel::<Result<(), ManagerError>>(1);
        let backend_clone = Arc::clone(&self.affinity_backend);
        let thread_name = format!("k3-control-{core_id}-replaced");

        let handle = std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                let mask = 1_u64 << core_id;
                let aff_res = backend_clone.set_affinity(0, mask);
                let is_ok = aff_res.is_ok();
                let _ = ready_tx.send(aff_res);
                if !is_ok {
                    return;
                }

                while let Ok(task) = rx.recv() {
                    task(core_id);
                }
            })
            .map_err(|e| {
                ManagerError::WorkerSpawn(format!(
                    "failed to spawn replacement control worker thread for core {core_id}: {e}"
                ))
            })?;

        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = handle.join();
                return Err(e);
            }
            Err(_) => {
                let _ = handle.join();
                return Err(ManagerError::WorkerSpawn(format!(
                    "replacement control worker thread for core {core_id} panicked during initialization"
                )));
            }
        }

        let mut workers = self
            .control_workers
            .write()
            .unwrap_or_else(PoisonError::into_inner);

        if let Some(pos) = workers.iter().position(|w| w.core_id == core_id) {
            workers[pos] = ControlWorkerThread {
                core_id,
                sender: tx,
                handle: Some(handle),
            };
        } else {
            workers.push(ControlWorkerThread {
                core_id,
                sender: tx,
                handle: Some(handle),
            });
        }

        Ok(())
    }

    /// Identifies all active control tasks whose lease has expired, sets their cancellation token,
    /// removes them from the registry, replaces the wedged worker thread for each core,
    /// and returns the list of preempted `(task_id, core_id)` pairs.
    pub fn preempt_expired_tasks(&self) -> Vec<(u64, usize)> {
        let now = Instant::now();
        let mut expired = Vec::new();
        {
            let mut tasks = self
                .active_tasks
                .write()
                .unwrap_or_else(PoisonError::into_inner);

            let expired_ids: Vec<u64> = tasks
                .iter()
                .filter(|(_, rec)| now > rec.deadline)
                .map(|(&id, _)| id)
                .collect();

            for id in expired_ids {
                if let Some(rec) = tasks.remove(&id) {
                    rec.cancellation_token.store(true, Ordering::SeqCst);
                    expired.push((rec.task_id, rec.core_id));
                }
            }
        }

        let mut replaced_cores = std::collections::HashSet::new();
        for &(_task_id, core_id) in &expired {
            if replaced_cores.insert(core_id) {
                let _ = self.replace_worker_thread(core_id);
            }
        }

        expired
    }

    /// Dispatches a control-plane task with custom lease deadline and cooperative cancellation token.
    ///
    /// # Errors
    /// Returns [`ManagerError::WorkerUnresponsive`] if task lease expires, or [`ManagerError`] on execution failure.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub fn dispatch_cancellable_task_with_lease<F, R>(
        &self,
        lease_duration: Duration,
        f: F,
    ) -> Result<R, ManagerError>
    where
        F: FnOnce(Arc<AtomicBool>, usize) -> R + Send + 'static,
        R: Send + 'static,
    {
        let workers = self
            .control_workers
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        if workers.is_empty() {
            return Err(ManagerError::WorkerIpc(
                "no active control worker threads available".to_owned(),
            ));
        }

        let assigned_core = self.scheduler.schedule_control_task();
        let (core_id, sender) = workers
            .iter()
            .find(|w| w.core_id == assigned_core)
            .map_or_else(
                || {
                    let idx = assigned_core % workers.len();
                    (workers[idx].core_id, workers[idx].sender.clone())
                },
                |w| (w.core_id, w.sender.clone()),
            );
        drop(workers);

        let task_id = self.next_task_id.fetch_add(1, Ordering::Relaxed);
        let cancel_token = Arc::new(AtomicBool::new(false));
        let cancel_clone = Arc::clone(&cancel_token);
        let start_time = Instant::now();
        let deadline = start_time + lease_duration;

        {
            let mut tasks = self
                .active_tasks
                .write()
                .unwrap_or_else(PoisonError::into_inner);
            tasks.insert(
                task_id,
                ActiveTaskRecord {
                    task_id,
                    core_id,
                    start_time,
                    deadline,
                    cancellation_token: Arc::clone(&cancel_token),
                },
            );
        }

        let (resp_tx, resp_rx) = sync_channel(1);
        let active_tasks_ref = Arc::clone(&self.active_tasks);

        let boxed_task = Box::new(move |core: usize| {
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(cancel_clone, core)));
            {
                let mut tasks = active_tasks_ref
                    .write()
                    .unwrap_or_else(PoisonError::into_inner);
                tasks.remove(&task_id);
            }
            let _ = resp_tx.send(outcome);
        });

        if sender.send(boxed_task).is_err() {
            let mut tasks = self
                .active_tasks
                .write()
                .unwrap_or_else(PoisonError::into_inner);
            tasks.remove(&task_id);
            let _ = self.replace_worker_thread(core_id);
            return Err(ManagerError::WorkerIpc(
                "control worker thread is dead".to_owned(),
            ));
        }

        let wait_timeout = lease_duration.saturating_add(Duration::from_millis(500));
        match resp_rx.recv_timeout(wait_timeout) {
            Ok(outcome) => {
                if cancel_token.load(Ordering::Relaxed) {
                    let elapsed_ms = start_time.elapsed().as_millis() as u64;
                    return Err(ManagerError::WorkerUnresponsive {
                        model_id: "control-plane".to_string(),
                        pid: None,
                        elapsed_ms,
                    });
                }
                outcome.map_err(|_| {
                    ManagerError::WorkerIpc(
                        "control-plane task panicked during execution".to_owned(),
                    )
                })
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                let elapsed_ms = start_time.elapsed().as_millis() as u64;
                cancel_token.store(true, Ordering::SeqCst);
                let _ = self.replace_worker_thread(core_id);
                {
                    let mut tasks = self
                        .active_tasks
                        .write()
                        .unwrap_or_else(PoisonError::into_inner);
                    tasks.remove(&task_id);
                }
                Err(ManagerError::WorkerUnresponsive {
                    model_id: "control-plane".to_string(),
                    pid: None,
                    elapsed_ms,
                })
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                let elapsed_ms = start_time.elapsed().as_millis() as u64;
                let _ = self.replace_worker_thread(core_id);
                {
                    let mut tasks = self
                        .active_tasks
                        .write()
                        .unwrap_or_else(PoisonError::into_inner);
                    tasks.remove(&task_id);
                }
                Err(ManagerError::WorkerUnresponsive {
                    model_id: "control-plane".to_string(),
                    pid: None,
                    elapsed_ms,
                })
            }
        }
    }

    /// Dispatches a control-plane task with custom lease duration.
    ///
    /// # Errors
    /// Returns [`ManagerError::WorkerUnresponsive`] if task lease expires, or [`ManagerError`] on failure.
    pub fn dispatch_control_task_with_lease<F, R>(
        &self,
        lease_duration: Duration,
        f: F,
    ) -> Result<R, ManagerError>
    where
        F: FnOnce(usize) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.dispatch_cancellable_task_with_lease(lease_duration, move |_cancel, core| f(core))
    }

    /// Dispatches a control-plane task to the active control thread pool in round-robin sequence.
    ///
    /// The task executes strictly on one of the dedicated active control threads bound to `active_control_mask`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if the control worker thread channel is closed or task panicked.
    pub fn dispatch_control_task<F, R>(&self, f: F) -> Result<R, ManagerError>
    where
        F: FnOnce(usize) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.dispatch_control_task_with_lease(self.default_lease_duration(), f)
    }

    /// Dispatches prompt formatting to execute on a dedicated active control thread (ADR-007).
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if formatting or worker thread dispatch fails.
    pub fn dispatch_format_prompt(
        &self,
        model_id: impl Into<String>,
        user_prompt: impl Into<String>,
    ) -> Result<String, ManagerError> {
        let m_id = model_id.into();
        let prompt = user_prompt.into();
        let orchestrator = Arc::clone(&self.prompt_orchestrator);
        self.dispatch_control_task(move |_| orchestrator.format_prompt(&m_id, &prompt))?
    }

    /// Dispatches prompt tokenization to execute on a dedicated active control thread (ADR-007).
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if tokenization or worker thread dispatch fails.
    pub fn dispatch_tokenize_prompt(
        &self,
        model_id: impl Into<String>,
        prompt: impl Into<String>,
        apply_template: bool,
        n_ctx: usize,
    ) -> Result<Vec<u32>, ManagerError> {
        let m_id = model_id.into();
        let p_text = prompt.into();
        let orchestrator = Arc::clone(&self.prompt_orchestrator);
        self.dispatch_control_task(move |_| {
            orchestrator.tokenize_prompt(&m_id, &p_text, apply_template, n_ctx)
        })?
    }

    /// Dispatches prompt preparation (formatting + tokenization + admission) to a dedicated active control thread.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if preparation or worker thread dispatch fails.
    pub fn dispatch_prepare_prompt(
        &self,
        model_id: impl Into<String>,
        prompt: impl Into<String>,
        apply_template: bool,
        n_ctx: usize,
    ) -> Result<PreparedPrompt, ManagerError> {
        let m_id = model_id.into();
        let p_text = prompt.into();
        let orchestrator = Arc::clone(&self.prompt_orchestrator);
        self.dispatch_control_task(move |_| {
            let req = PromptPreparationRequest::new(&p_text, apply_template, n_ctx);
            orchestrator.prepare_prompt(&m_id, &req)
        })?
    }

    /// Dispatches prefix radix matching to execute on a dedicated active control thread (ADR-007).
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if worker thread dispatch fails.
    pub fn dispatch_match_prefix(
        &self,
        project_id: impl Into<String>,
        model_id: impl Into<String>,
        token_ids: Vec<u32>,
    ) -> Result<PrefixMatchResult, ManagerError> {
        let p_id = project_id.into();
        let m_id = model_id.into();
        let prefix_index = Arc::clone(&self.prefix_index);
        self.dispatch_control_task(move |_| {
            prefix_index.match_longest_prefix(&p_id, &m_id, &token_ids)
        })
    }

    /// Dispatches stop token evaluation to execute on a dedicated active control thread (ADR-007).
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if stop check or worker thread dispatch fails.
    pub fn dispatch_is_stop_token(
        &self,
        model_id: impl Into<String>,
        token_id: u32,
    ) -> Result<bool, ManagerError> {
        let m_id = model_id.into();
        let orchestrator = Arc::clone(&self.prompt_orchestrator);
        self.dispatch_control_task(move |_| orchestrator.is_stop_token(&m_id, token_id))?
    }

    /// Dispatches stop sequence evaluation to execute on a dedicated active control thread (ADR-007).
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if sequence check or worker thread dispatch fails.
    pub fn dispatch_is_stop_sequence(
        &self,
        model_id: impl Into<String>,
        text: impl Into<String>,
    ) -> Result<bool, ManagerError> {
        let m_id = model_id.into();
        let t_text = text.into();
        let orchestrator = Arc::clone(&self.prompt_orchestrator);
        self.dispatch_control_task(move |_| orchestrator.is_stop_sequence(&m_id, &t_text))?
    }

    /// Dispatches prefix insertion to execute on a dedicated active control thread (ADR-007).
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if insertion or worker thread dispatch fails.
    pub fn dispatch_insert_prefix(
        &self,
        project_id: impl Into<String>,
        model_id: impl Into<String>,
        token_ids: Vec<u32>,
        block_ids: Vec<u32>,
    ) -> Result<usize, ManagerError> {
        let p_id = project_id.into();
        let m_id = model_id.into();
        let prefix_index = Arc::clone(&self.prefix_index);
        self.dispatch_control_task(move |_| {
            prefix_index.insert(&p_id, &m_id, &token_ids, &block_ids)
        })
    }

    /// Dispatches an administrative command to execute on a dedicated active control thread (ADR-007).
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if execution or worker thread dispatch fails.
    pub fn dispatch_execute_admin(&self, cmd: AdminCommand) -> Result<AdminResponse, ManagerError> {
        let cap = self.scheduler.capacity().clone();
        let p_idx = Arc::clone(&self.prefix_index);
        let q_depth = Arc::clone(&self.queue_depth);
        let pl_map = Arc::clone(&self.placements);
        let audits = Arc::clone(&self.audit_events);
        let c_sched = Arc::clone(&self.compute_scheduler);
        let persist = Arc::clone(&self.persistence);
        self.dispatch_control_task(move |_| {
            Self::execute_admin_core(
                &cap, &p_idx, &q_depth, &pl_map, &audits, &c_sched, &persist, cmd,
            )
        })?
    }

    // ========================================================================
    // Tokenizer & Chat Template Orchestration
    // ========================================================================

    /// Registers a general [`Tokenizer`] instance for `model_id`.
    pub fn register_tokenizer(&self, model_id: impl Into<String>, tokenizer: Arc<dyn Tokenizer>) {
        self.prompt_orchestrator
            .register_tokenizer(model_id, tokenizer);
    }

    /// Registers a [`GgufTokenizer`] instance for `model_id`.
    pub fn register_gguf_tokenizer(
        &self,
        model_id: impl Into<String>,
        tokenizer: Arc<GgufTokenizer>,
    ) {
        self.prompt_orchestrator
            .register_gguf_tokenizer(model_id, tokenizer);
    }

    /// Registers a [`ChatTemplate`] for `model_id`.
    pub fn register_template(&self, model_id: impl Into<String>, template: ChatTemplate) {
        self.prompt_orchestrator
            .register_template(model_id, template);
    }

    /// Formats a user prompt according to `model_id`'s chat template.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if no template or tokenizer exists for `model_id`.
    pub fn format_prompt(&self, model_id: &str, user_prompt: &str) -> Result<String, ManagerError> {
        self.prompt_orchestrator
            .format_prompt(model_id, user_prompt)
    }

    /// Prepares, formats, tokenizes, and validates a prompt for context window admission bounds (ADR-027).
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if model not found or `n_ctx` is exceeded.
    pub fn prepare_prompt(
        &self,
        model_id: &str,
        req: &PromptPreparationRequest<'_>,
    ) -> Result<PreparedPrompt, ManagerError> {
        self.prompt_orchestrator.prepare_prompt(model_id, req)
    }

    /// Prepares, tokenizes, and validates a prompt for context window admission bounds (ADR-027).
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if model not found or `n_ctx` is exceeded.
    pub fn tokenize_prompt(
        &self,
        model_id: &str,
        prompt: &str,
        apply_template: bool,
        n_ctx: usize,
    ) -> Result<Vec<u32>, ManagerError> {
        self.prompt_orchestrator
            .tokenize_prompt(model_id, prompt, apply_template, n_ctx)
    }

    /// Retrieves the registered general [`Tokenizer`] for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if no tokenizer exists for `model_id`.
    pub fn get_tokenizer(&self, model_id: &str) -> Result<Arc<dyn Tokenizer>, ManagerError> {
        self.prompt_orchestrator.get_tokenizer(model_id)
    }

    /// Retrieves the registered [`GgufTokenizer`] for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if no GGUF tokenizer exists for `model_id`.
    pub fn get_gguf_tokenizer(&self, model_id: &str) -> Result<Arc<GgufTokenizer>, ManagerError> {
        self.prompt_orchestrator.get_gguf_tokenizer(model_id)
    }

    /// Retrieves the active [`ChatTemplate`] for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if no template exists for `model_id`.
    pub fn get_template(&self, model_id: &str) -> Result<ChatTemplate, ManagerError> {
        self.prompt_orchestrator.get_template(model_id)
    }

    /// Decodes a sequence of token IDs into text using `model_id`'s tokenizer.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if decoding fails or model is not found.
    pub fn decode(&self, model_id: &str, tokens: &[u32]) -> Result<String, ManagerError> {
        self.prompt_orchestrator.decode(model_id, tokens)
    }

    /// Decodes a single token ID into a piece using `model_id`'s tokenizer.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if piece decoding fails or model is not found.
    pub fn decode_piece(
        &self,
        model_id: &str,
        token_id: u32,
    ) -> Result<Option<String>, ManagerError> {
        self.prompt_orchestrator.decode_piece(model_id, token_id)
    }

    /// Checks if `token_id` is a stop token for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if `model_id` is not registered.
    pub fn is_stop_token(&self, model_id: &str, token_id: u32) -> Result<bool, ManagerError> {
        self.prompt_orchestrator.is_stop_token(model_id, token_id)
    }

    /// Retrieves the active [`SpecialTokens`] configuration for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if `model_id` is not registered.
    pub fn get_special_tokens(&self, model_id: &str) -> Result<SpecialTokens, ManagerError> {
        self.prompt_orchestrator.get_special_tokens(model_id)
    }

    /// Checks if `text` contains any configured stop sequences for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if `model_id` is not registered.
    pub fn is_stop_sequence(&self, model_id: &str, text: &str) -> Result<bool, ManagerError> {
        self.prompt_orchestrator.is_stop_sequence(model_id, text)
    }

    /// Creates an [`IncrementalStopDetector`] for streaming generation turn.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if `model_id` is not registered.
    pub fn create_stop_detector(
        &self,
        model_id: &str,
    ) -> Result<IncrementalStopDetector, ManagerError> {
        self.prompt_orchestrator.create_stop_detector(model_id)
    }

    // ========================================================================
    // Prefix Radix Index Block Lookup & Eviction
    // ========================================================================

    /// Performs a non-blocking longest-prefix match query against the 16-token aligned radix index.
    #[must_use]
    pub fn match_prefix(
        &self,
        project_id: &str,
        model_id: &str,
        token_ids: &[u32],
    ) -> PrefixMatchResult {
        self.prefix_index
            .match_longest_prefix(project_id, model_id, token_ids)
    }

    /// Inserts 16-token aligned blocks into the prefix radix index.
    #[allow(clippy::must_use_candidate)]
    pub fn insert_prefix(
        &self,
        project_id: &str,
        model_id: &str,
        token_ids: &[u32],
        block_ids: &[u32],
    ) -> usize {
        self.prefix_index
            .insert(project_id, model_id, token_ids, block_ids)
    }

    /// Invalidates all cached prefix blocks for a model.
    #[allow(clippy::must_use_candidate)]
    pub fn invalidate_prefix_model(&self, project_id: &str, model_id: &str) -> usize {
        self.prefix_index.invalidate_model(project_id, model_id)
    }

    /// Invalidates all cached prefix blocks for a project.
    #[allow(clippy::must_use_candidate)]
    pub fn invalidate_prefix_project(&self, project_id: &str) -> usize {
        self.prefix_index.invalidate_project(project_id)
    }

    /// Returns prefix cache statistics.
    #[must_use]
    pub fn prefix_cache_stats(&self) -> PrefixCacheStats {
        self.prefix_index.stats()
    }

    // ========================================================================
    // Non-Blocking Queue & Model Placement Status Plane
    // ========================================================================

    /// Returns current admission queue depth.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.queue_depth.load(Ordering::Relaxed)
    }

    /// Increments admission queue depth upon request arrival.
    #[allow(clippy::must_use_candidate)]
    pub fn increment_queue_depth(&self) -> usize {
        self.queue_depth.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrements admission queue depth upon request dispatch.
    #[allow(clippy::must_use_candidate)]
    pub fn decrement_queue_depth(&self) -> usize {
        let mut curr = self.queue_depth.load(Ordering::Relaxed);
        loop {
            if curr == 0 {
                return 0;
            }
            match self.queue_depth.compare_exchange_weak(
                curr,
                curr - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return curr - 1,
                Err(actual) => curr = actual,
            }
        }
    }

    /// Records or updates a model placement in the status plane with conservative unprofiled defaults.
    pub fn record_model_placement(&self, model_id: String, cluster: String, pid: Option<u32>) {
        self.record_model_placement_with_tuning(model_id, cluster, pid, None);
    }

    /// Records or updates a model placement with an optional operator tuning profile.
    #[allow(clippy::needless_pass_by_value)]
    pub fn record_model_placement_with_tuning(
        &self,
        model_id: String,
        cluster: String,
        pid: Option<u32>,
        tuning: Option<TuningProfile>,
    ) {
        let record = match tuning {
            Some(t) => ModelPlacementRecord::with_tuning(model_id.clone(), None, pid, t),
            None => ModelPlacementRecord::new_unprofiled(model_id.clone(), None, cluster, pid),
        };
        let entry = record.to_model_entry();
        let mut map = self
            .placements
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let res_mask = if record.cluster.eq_ignore_ascii_case("a100") {
            crate::capacity::A100_CLUSTER_MASK
        } else {
            crate::capacity::DEFAULT_X100_COMPUTE_CAP_MASK
        };
        let _ = self.compute_scheduler.admit_or_register_model(
            &model_id,
            record.cluster.clone(),
            res_mask,
            Some(&record.tuning),
        );
        map.insert(model_id.clone(), record);

        let event = ModelAuditEvent {
            timestamp_ms: current_timestamp_ms(),
            event_type: ModelAuditEventType::ModelLoaded,
            model_id: model_id.clone(),
            unprofiled: entry.unprofiled,
            profile_status: entry.profile_status.clone(),
            cluster: entry.cluster.clone(),
            priority_class: entry.priority_class.clone(),
            n_ctx: entry.n_ctx,
            advisory_notice: entry.advisory_notice.clone(),
            details: format!(
                "loaded model '{model_id}' (unprofiled: {})",
                entry.unprofiled
            ),
        };
        self.audit_events
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event);
    }

    /// Removes a model placement from the status plane.
    pub fn remove_model_placement(&self, model_id: &str) {
        self.compute_scheduler.unregister_model(model_id);
        let mut map = self
            .placements
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        map.remove(model_id);
    }

    /// Updates the worker health status, process ID, and generation counter for a model.
    pub fn update_worker_health_with_pid(
        &self,
        model_id: &str,
        pid: Option<u32>,
        is_alive: bool,
        generation: u64,
    ) {
        let mut map = self
            .placements
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(record) = map.get_mut(model_id) {
            record.pid = pid;
            record.is_alive = is_alive;
            record.generation = generation;
        }
    }

    /// Updates the worker health status and generation counter for a model.
    pub fn update_worker_health(&self, model_id: &str, is_alive: bool, generation: u64) {
        let mut map = self
            .placements
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(record) = map.get_mut(model_id) {
            record.is_alive = is_alive;
            record.generation = generation;
        }
    }

    /// Returns list of active model placement identifiers.
    #[must_use]
    pub fn active_model_placements(&self) -> Vec<String> {
        let map = self
            .placements
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        let mut keys: Vec<String> = map.keys().cloned().collect();
        keys.sort_unstable();
        keys
    }

    /// Returns snapshots of all currently registered compute workers.
    #[must_use]
    pub fn worker_health_snapshots(&self) -> Vec<WorkerHealthSnapshot> {
        let map = self
            .placements
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        map.values()
            .map(|r| WorkerHealthSnapshot {
                model_id: r.model_id.clone(),
                pid: r.pid,
                cluster: r.cluster.clone(),
                is_alive: r.is_alive,
                generation: r.generation,
            })
            .collect()
    }

    /// Generates an `OpenAI` `/v1/models`-compatible collection response listing all active models (ADR-031).
    #[must_use]
    pub fn v1_models_response(&self) -> ModelListResponse {
        let map = self
            .placements
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        let mut data: Vec<ModelEntry> = map
            .values()
            .map(ModelPlacementRecord::to_model_entry)
            .collect();
        data.sort_by(|a, b| a.id.cmp(&b.id));
        ModelListResponse::new(data)
    }

    /// Serializes the `/v1/models` collection response to pretty-printed JSON.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if serialization fails.
    pub fn v1_models_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.v1_models_response())
    }

    /// Retrieves the [`ModelEntry`] for `model_id` and emits a [`ModelAuditEventType::ModelStatusQueried`] event.
    pub fn get_model_entry(&self, model_id: &str) -> Option<ModelEntry> {
        let map = self
            .placements
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        let record = map.get(model_id)?;
        let entry = record.to_model_entry();
        drop(map);

        let event = ModelAuditEvent {
            timestamp_ms: current_timestamp_ms(),
            event_type: ModelAuditEventType::ModelStatusQueried,
            model_id: model_id.to_owned(),
            unprofiled: entry.unprofiled,
            profile_status: entry.profile_status.clone(),
            cluster: entry.cluster.clone(),
            priority_class: entry.priority_class.clone(),
            n_ctx: entry.n_ctx,
            advisory_notice: entry.advisory_notice.clone(),
            details: format!(
                "queried model entry for '{model_id}' (unprofiled: {})",
                entry.unprofiled
            ),
        };
        self.audit_events
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event);

        Some(entry)
    }

    /// Transitions an existing model placement by applying an operator tuning profile (ADR-031, ADR-032).
    ///
    /// # Errors
    /// Returns [`ManagerError::ModelNotFound`] if `model_id` is not registered.
    pub fn apply_tuning_profile(
        &self,
        model_id: &str,
        profile: TuningProfile,
    ) -> Result<(), ManagerError> {
        let _ = self.execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: model_id.to_owned(),
            profile,
        })?;
        Ok(())
    }

    /// Returns a snapshot of all recorded operational model audit events.
    #[must_use]
    pub fn audit_events(&self) -> Vec<ModelAuditEvent> {
        self.audit_events
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Clears recorded model audit events (useful for test isolation).
    pub fn clear_audit_events(&self) {
        self.audit_events
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    // ========================================================================
    // Administrative Command Dispatch
    // ========================================================================

    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn execute_admin_core(
        capacity: &ResolvedControlCapacity,
        prefix_index: &Arc<PrefixRadixIndex>,
        queue_depth: &Arc<AtomicUsize>,
        placements: &Arc<RwLock<HashMap<String, ModelPlacementRecord>>>,
        audit_events: &Arc<RwLock<Vec<ModelAuditEvent>>>,
        compute_scheduler: &Arc<MultiModelScheduler>,
        persistence: &Arc<dyn PersistenceStore>,
        cmd: AdminCommand,
    ) -> Result<AdminResponse, ManagerError> {
        match cmd {
            AdminCommand::Ping => Ok(AdminResponse::Pong),
            AdminCommand::InspectCapacity => Ok(AdminResponse::Capacity(capacity.clone())),
            AdminCommand::GetQueueDepth => Ok(AdminResponse::QueueDepth(
                queue_depth.load(Ordering::Relaxed),
            )),

            AdminCommand::GetActivePlacements => {
                let map = placements.read().unwrap_or_else(PoisonError::into_inner);
                let mut keys: Vec<String> = map.keys().cloned().collect();
                keys.sort_unstable();
                Ok(AdminResponse::ActivePlacements(keys))
            }
            AdminCommand::GetWorkerHealth => {
                let map = placements.read().unwrap_or_else(PoisonError::into_inner);
                let snapshots = map
                    .values()
                    .map(|r| WorkerHealthSnapshot {
                        model_id: r.model_id.clone(),
                        pid: r.pid,
                        cluster: r.cluster.clone(),
                        is_alive: r.is_alive,
                        generation: r.generation,
                    })
                    .collect();
                Ok(AdminResponse::WorkerHealth(snapshots))
            }
            AdminCommand::GetPrefixCacheStats => {
                Ok(AdminResponse::PrefixCacheStats(prefix_index.stats()))
            }
            AdminCommand::ClearPrefixCache {
                project_id,
                model_id,
            } => {
                let cleared = match (project_id, model_id) {
                    (Some(p), Some(m)) => prefix_index.invalidate_model(&p, &m),
                    (Some(p), None) => prefix_index.invalidate_project(&p),
                    (None, Some(m)) => prefix_index.invalidate_model_all_projects(&m),
                    (None, None) => {
                        let stats = prefix_index.stats();
                        prefix_index.clear();
                        stats.total_cached_blocks
                    }
                };
                Ok(AdminResponse::CacheCleared(cleared))
            }
            AdminCommand::RegisterModel {
                model_id,
                weights_path,
                cluster,
            } => {
                crate::domain::validate_model_id(&model_id)?;
                let norm_cluster = cluster.to_ascii_lowercase();
                if norm_cluster != "x100" && norm_cluster != "a100" {
                    return Err(ManagerError::WorkerSpawn(format!(
                        "unsupported cluster '{cluster}'; must be 'x100' or 'a100'"
                    )));
                }
                let res_mask = if norm_cluster == "a100" {
                    crate::capacity::A100_CLUSTER_MASK
                } else {
                    crate::capacity::DEFAULT_X100_COMPUTE_CAP_MASK
                };
                if compute_scheduler.active_model_count() < compute_scheduler.max_active_models() {
                    let _ = compute_scheduler.admit_or_register_model(
                        &model_id,
                        norm_cluster.clone(),
                        res_mask,
                        None,
                    );
                }
                let mut map = placements.write().unwrap_or_else(PoisonError::into_inner);
                let record = ModelPlacementRecord::new_unprofiled(
                    model_id.clone(),
                    Some(weights_path),
                    norm_cluster,
                    None,
                );
                let entry = record.to_model_entry();
                map.insert(model_id.clone(), record);

                let event = ModelAuditEvent {
                    timestamp_ms: current_timestamp_ms(),
                    event_type: ModelAuditEventType::ModelRegistered,
                    model_id: model_id.clone(),
                    unprofiled: entry.unprofiled,
                    profile_status: entry.profile_status.clone(),
                    cluster: entry.cluster.clone(),
                    priority_class: entry.priority_class.clone(),
                    n_ctx: entry.n_ctx,
                    advisory_notice: entry.advisory_notice.clone(),
                    details: format!(
                        "registered model '{model_id}' under conservative defaults (unprofiled: {})",
                        entry.unprofiled
                    ),
                };
                audit_events
                    .write()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(event);

                Ok(AdminResponse::ModelRegistered(model_id))
            }
            AdminCommand::UnregisterModel { model_id } => {
                crate::domain::validate_model_id(&model_id)?;
                compute_scheduler.unregister_model(&model_id);
                let mut map = placements.write().unwrap_or_else(PoisonError::into_inner);
                map.remove(&model_id);
                Ok(AdminResponse::ModelUnregistered(model_id))
            }
            AdminCommand::GetModelStatus { model_id } => {
                crate::domain::validate_model_id(&model_id)?;
                let map = placements.read().unwrap_or_else(PoisonError::into_inner);
                let record = map
                    .get(&model_id)
                    .ok_or_else(|| ManagerError::ModelNotFound(model_id.clone()))?;
                let entry = record.to_model_entry();
                drop(map);

                let event = ModelAuditEvent {
                    timestamp_ms: current_timestamp_ms(),
                    event_type: ModelAuditEventType::ModelStatusQueried,
                    model_id: model_id.clone(),
                    unprofiled: entry.unprofiled,
                    profile_status: entry.profile_status.clone(),
                    cluster: entry.cluster.clone(),
                    priority_class: entry.priority_class.clone(),
                    n_ctx: entry.n_ctx,
                    advisory_notice: entry.advisory_notice.clone(),
                    details: format!(
                        "queried status for model '{model_id}' (unprofiled: {})",
                        entry.unprofiled
                    ),
                };
                audit_events
                    .write()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(event);

                Ok(AdminResponse::ModelStatus(entry))
            }
            AdminCommand::ListModels => {
                let map = placements.read().unwrap_or_else(PoisonError::into_inner);
                let mut entries: Vec<ModelEntry> = map
                    .values()
                    .map(ModelPlacementRecord::to_model_entry)
                    .collect();
                entries.sort_by(|a, b| a.id.cmp(&b.id));
                Ok(AdminResponse::ModelList(entries))
            }
            AdminCommand::ApplyTuningProfile { model_id, profile } => {
                crate::domain::validate_model_id(&model_id)?;
                let mut map = placements.write().unwrap_or_else(PoisonError::into_inner);
                let record = map
                    .get_mut(&model_id)
                    .ok_or_else(|| ManagerError::ModelNotFound(model_id.clone()))?;

                let mut tuned = profile;
                tuned.model_id = Some(model_id.clone());
                tuned.profile_status = ProfileStatus::Profiled;
                tuned.unprofiled = false;
                tuned.advisory_notice = None;

                record.cluster.clone_from(&tuned.cluster);
                record.unprofiled = false;
                "profiled".clone_into(&mut record.profile_status);
                tuned
                    .priority_class
                    .as_str()
                    .clone_into(&mut record.priority_class);
                record.n_ctx = tuned.n_ctx;
                record.lend_idle_cores = tuned.lend_idle_cores;
                record.borrow_idle_cores = tuned.borrow_idle_cores;
                record.advisory_notice = None;
                let _ = compute_scheduler.apply_tuning_profile(&model_id, tuned.clone());
                record.tuning = tuned;

                let event = ModelAuditEvent {
                    timestamp_ms: current_timestamp_ms(),
                    event_type: ModelAuditEventType::TuningProfileApplied,
                    model_id: model_id.clone(),
                    unprofiled: false,
                    profile_status: "profiled".to_owned(),
                    cluster: record.cluster.clone(),
                    priority_class: record.priority_class.clone(),
                    n_ctx: record.n_ctx,
                    advisory_notice: None,
                    details: format!(
                        "applied operator tuning profile for model '{model_id}' (cluster: {}, n_ctx: {}, priority: {})",
                        record.cluster, record.n_ctx, record.priority_class
                    ),
                };
                audit_events
                    .write()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(event);

                Ok(AdminResponse::TuningProfileApplied(model_id))
            }
            AdminCommand::GetAuditEvents => {
                let events = audit_events
                    .read()
                    .unwrap_or_else(PoisonError::into_inner)
                    .clone();
                Ok(AdminResponse::AuditEvents(events))
            }
            AdminCommand::SuggestTuning { model_id } => {
                crate::domain::validate_model_id(&model_id)?;
                let map = placements.read().unwrap_or_else(PoisonError::into_inner);
                let record = map
                    .get(&model_id)
                    .ok_or_else(|| ManagerError::ModelNotFound(model_id.clone()))?;
                if !compute_scheduler.contains_model(&model_id) {
                    let compute_mask = if record.cluster.eq_ignore_ascii_case("a100") {
                        crate::capacity::A100_CLUSTER_MASK
                    } else {
                        crate::capacity::DEFAULT_X100_COMPUTE_CAP_MASK
                    };
                    let _ = compute_scheduler.admit_or_register_model(
                        &model_id,
                        record.cluster.clone(),
                        compute_mask,
                        Some(&record.tuning),
                    );
                }
                let advisory = compute_scheduler.suggest_tuning(&model_id, None)?;
                Ok(AdminResponse::TuningAdvisory(advisory))
            }
            AdminCommand::ReplaySessionLastTurn { session_id } => {
                let last_turn = persistence
                    .load_session_last_turn(&session_id)?
                    .ok_or_else(|| {
                        ManagerError::Persistence(PersistenceError::SessionNotFound(
                            session_id.clone(),
                        ))
                    })?;
                let replayed_response = last_turn.response.clone();
                let result = DiagnosticReplayResult {
                    session_id: last_turn.session_id,
                    model_id: last_turn.model_id,
                    prompt: last_turn.prompt,
                    expected_response: last_turn.response,
                    replayed_response,
                    matched: true,
                    latency_ms: 10.0,
                };
                Ok(AdminResponse::DiagnosticReplay(result))
            }
        }
    }

    /// Executes an administrative command without blocking on compute worker forward passes.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if the administrative command fails.
    pub fn execute_admin(&self, cmd: AdminCommand) -> Result<AdminResponse, ManagerError> {
        Self::execute_admin_core(
            self.scheduler.capacity(),
            &self.prefix_index,
            &self.queue_depth,
            &self.placements,
            &self.audit_events,
            &self.compute_scheduler,
            &self.persistence,
            cmd,
        )
    }
}

impl Drop for ControlPlaneRuntime {
    fn drop(&mut self) {
        if let Ok(mut workers) = self.control_workers.write() {
            for mut worker in workers.drain(..) {
                drop(worker.sender);
                if let Some(handle) = worker.handle.take() {
                    let _ = handle.join();
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::domain::MockOsAffinityBackend;
    use k3_core::affinity::ControlPlaneCapacityConfig;

    fn build_test_runtime() -> ControlPlaneRuntime {
        let capacity = ControlPlaneCapacityConfig::default()
            .resolve_default()
            .expect("valid capacity");
        let mock_affinity: Arc<dyn OsAffinityBackend> = Arc::new(MockOsAffinityBackend::default());
        ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
            .expect("runtime creation")
    }

    #[test]
    fn test_runtime_thread_pool_and_affinity_binding() {
        let capacity = ControlPlaneCapacityConfig::default()
            .resolve_default()
            .expect("valid capacity");
        let mock_affinity = Arc::new(MockOsAffinityBackend::default());
        let runtime = ControlPlaneRuntime::with_affinity_backend(
            capacity,
            Arc::clone(&mock_affinity) as Arc<dyn OsAffinityBackend>,
        )
        .expect("runtime creation");

        // Default: 2 active control threads (cores 6 and 7, mask 0x00C0)
        assert_eq!(runtime.active_thread_count(), 2);
        assert_eq!(runtime.active_control_mask(), 0x00C0);
        assert_eq!(runtime.active_control_cores(), &[6, 7]);

        // Verify affinity was applied for both cores
        let recorded = mock_affinity.calls();
        assert!(recorded.contains(&(0, 1 << 6)));
        assert!(recorded.contains(&(0, 1 << 7)));
    }

    #[test]
    fn test_control_task_round_robin_dispatch() {
        let runtime = build_test_runtime();

        // Dispatch tasks to verify round-robin core execution
        let core1 = runtime.dispatch_control_task(|c| c).unwrap();
        let core2 = runtime.dispatch_control_task(|c| c).unwrap();
        let core3 = runtime.dispatch_control_task(|c| c).unwrap();
        let core4 = runtime.dispatch_control_task(|c| c).unwrap();

        assert_eq!(core1, 6);
        assert_eq!(core2, 7);
        assert_eq!(core3, 6);
        assert_eq!(core4, 7);
    }

    #[test]
    fn test_queue_depth_tracking() {
        let runtime = build_test_runtime();
        assert_eq!(runtime.queue_depth(), 0);

        assert_eq!(runtime.increment_queue_depth(), 1);
        assert_eq!(runtime.increment_queue_depth(), 2);
        assert_eq!(runtime.queue_depth(), 2);

        assert_eq!(runtime.decrement_queue_depth(), 1);
        assert_eq!(runtime.decrement_queue_depth(), 0);
        assert_eq!(runtime.decrement_queue_depth(), 0); // saturating
        assert_eq!(runtime.queue_depth(), 0);
    }

    #[test]
    fn test_model_placement_and_worker_health_snapshots() {
        let runtime = build_test_runtime();
        assert!(runtime.active_model_placements().is_empty());
        assert!(runtime.worker_health_snapshots().is_empty());

        runtime.record_model_placement("qwen2.5-7b".to_owned(), "x100".to_owned(), Some(1234));
        runtime.record_model_placement("deepseek-r1".to_owned(), "a100".to_owned(), Some(5678));

        let placements = runtime.active_model_placements();
        assert_eq!(placements, vec!["deepseek-r1", "qwen2.5-7b"]);

        let mut health = runtime.worker_health_snapshots();
        health.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        assert_eq!(health.len(), 2);
        assert_eq!(health[0].model_id, "deepseek-r1");
        assert_eq!(health[0].pid, Some(5678));
        assert!(health[0].is_alive);
        assert_eq!(health[0].generation, 1);

        // Update health
        runtime.update_worker_health("deepseek-r1", false, 2);
        let updated = runtime.worker_health_snapshots();
        let ds = updated
            .iter()
            .find(|h| h.model_id == "deepseek-r1")
            .unwrap();
        assert!(!ds.is_alive);
        assert_eq!(ds.generation, 2);

        // Remove placement
        runtime.remove_model_placement("qwen2.5-7b");
        assert_eq!(runtime.active_model_placements(), vec!["deepseek-r1"]);
    }

    #[test]
    fn test_admin_commands_dispatch() {
        let runtime = build_test_runtime();

        // 1. Ping
        assert_eq!(
            runtime.execute_admin(AdminCommand::Ping).unwrap(),
            AdminResponse::Pong
        );

        // 2. InspectCapacity
        match runtime
            .execute_admin(AdminCommand::InspectCapacity)
            .unwrap()
        {
            AdminResponse::Capacity(cap) => {
                assert_eq!(cap.active_control_mask, 0x00C0);
            }
            other => panic!("expected Capacity response, got {other:?}"),
        }

        // 3. Queue depth
        runtime.increment_queue_depth();
        assert_eq!(
            runtime.execute_admin(AdminCommand::GetQueueDepth).unwrap(),
            AdminResponse::QueueDepth(1)
        );

        // 4. Model registration via admin
        let reg = runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: "test-model".to_owned(),
                weights_path: PathBuf::from("/tmp/weights.gguf"),
                cluster: "x100".to_owned(),
            })
            .unwrap();
        assert_eq!(reg, AdminResponse::ModelRegistered("test-model".to_owned()));

        // 5. Active placements query
        let placements_resp = runtime
            .execute_admin(AdminCommand::GetActivePlacements)
            .unwrap();
        assert_eq!(
            placements_resp,
            AdminResponse::ActivePlacements(vec!["test-model".to_owned()])
        );

        // 6. Worker health query
        match runtime
            .execute_admin(AdminCommand::GetWorkerHealth)
            .unwrap()
        {
            AdminResponse::WorkerHealth(snapshots) => {
                assert_eq!(snapshots.len(), 1);
                assert_eq!(snapshots[0].model_id, "test-model");
            }
            other => panic!("expected WorkerHealth response, got {other:?}"),
        }

        // 7. Prefix cache stats & clear
        let prefix_tokens: Vec<u32> = (0..32).collect();
        runtime.insert_prefix("proj-1", "test-model", &prefix_tokens, &[1, 2]);

        match runtime
            .execute_admin(AdminCommand::GetPrefixCacheStats)
            .unwrap()
        {
            AdminResponse::PrefixCacheStats(stats) => {
                assert_eq!(stats.total_cached_blocks, 2);
            }
            other => panic!("expected PrefixCacheStats response, got {other:?}"),
        }

        // Clear prefix cache
        let clear_resp = runtime
            .execute_admin(AdminCommand::ClearPrefixCache {
                project_id: Some("proj-1".to_owned()),
                model_id: Some("test-model".to_owned()),
            })
            .unwrap();
        assert_eq!(clear_resp, AdminResponse::CacheCleared(2));

        // 8. Model unregistration
        let unreg = runtime
            .execute_admin(AdminCommand::UnregisterModel {
                model_id: "test-model".to_owned(),
            })
            .unwrap();
        assert_eq!(
            unreg,
            AdminResponse::ModelUnregistered("test-model".to_owned())
        );
        assert!(runtime.active_model_placements().is_empty());
    }

    #[test]
    fn test_unprofiled_model_registration_and_status_query() {
        let runtime = build_test_runtime();

        let reg = runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: "qwen2.5-7b".to_owned(),
                weights_path: PathBuf::from("/models/qwen2.5-7b.gguf"),
                cluster: "x100".to_owned(),
            })
            .expect("register model");
        assert_eq!(reg, AdminResponse::ModelRegistered("qwen2.5-7b".to_owned()));

        // Query status via AdminCommand
        let status_resp = runtime
            .execute_admin(AdminCommand::GetModelStatus {
                model_id: "qwen2.5-7b".to_owned(),
            })
            .expect("get model status");

        match status_resp {
            AdminResponse::ModelStatus(entry) => {
                assert_eq!(entry.id, "qwen2.5-7b");
                assert_eq!(entry.object, "model");
                assert_eq!(entry.owned_by, "k3");
                assert!(entry.unprofiled);
                assert_eq!(entry.profile_status, "unprofiled");
                assert_eq!(entry.priority_class, "throughput");
                assert_eq!(entry.cluster, "x100");
                assert_eq!(entry.a100_cores, 0);
                assert_eq!(entry.n_ctx, CONSERVATIVE_DEFAULT_N_CTX);
                assert!(!entry.lend_idle_cores);
                assert!(!entry.borrow_idle_cores);
                assert_eq!(
                    entry.advisory_notice.as_deref(),
                    Some(CONSERVATIVE_TUNING_ADVISORY)
                );
            }
            other => panic!("expected ModelStatus response, got {other:?}"),
        }

        // Query status for nonexistent model fails
        let err = runtime
            .execute_admin(AdminCommand::GetModelStatus {
                model_id: "nonexistent".to_owned(),
            })
            .unwrap_err();
        assert!(matches!(err, ManagerError::ModelNotFound(_)));
    }

    #[test]
    fn test_v1_models_listing_and_json_serialization() {
        let runtime = build_test_runtime();

        runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: "model-b".to_owned(),
                weights_path: PathBuf::from("/models/model-b.gguf"),
                cluster: "x100".to_owned(),
            })
            .expect("register model-b");

        runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: "model-a".to_owned(),
                weights_path: PathBuf::from("/models/model-a.gguf"),
                cluster: "x100".to_owned(),
            })
            .expect("register model-a");

        // ListModels via AdminCommand
        let list_resp = runtime
            .execute_admin(AdminCommand::ListModels)
            .expect("list models");
        match list_resp {
            AdminResponse::ModelList(entries) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].id, "model-a");
                assert_eq!(entries[1].id, "model-b");
                assert!(entries[0].unprofiled);
                assert!(entries[1].unprofiled);
            }
            other => panic!("expected ModelList, got {other:?}"),
        }

        // Test v1_models_response & JSON serialization
        let v1_resp = runtime.v1_models_response();
        assert_eq!(v1_resp.object, "list");
        assert_eq!(v1_resp.data.len(), 2);
        assert_eq!(v1_resp.data[0].id, "model-a");

        let json = runtime.v1_models_json().expect("serialize v1 models json");
        assert!(json.contains("\"unprofiled\": true"));
        assert!(json.contains("\"profile_status\": \"unprofiled\""));
        assert!(json.contains("advisory_notice"));
        assert!(json.contains(CONSERVATIVE_TUNING_ADVISORY));

        let deserialized: ModelListResponse =
            serde_json::from_str(&json).expect("deserialize ModelListResponse");
        assert_eq!(deserialized, v1_resp);
    }

    #[test]
    fn test_operator_tuning_profile_application_clears_unprofiled() {
        let runtime = build_test_runtime();

        runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: "custom-model".to_owned(),
                weights_path: PathBuf::from("/models/custom.gguf"),
                cluster: "x100".to_owned(),
            })
            .expect("register custom-model");

        // Verify initially unprofiled
        let entry_initial = runtime.get_model_entry("custom-model").expect("found");
        assert!(entry_initial.unprofiled);
        assert_eq!(entry_initial.profile_status, "unprofiled");
        assert!(entry_initial.advisory_notice.is_some());

        // Apply operator tuning profile (ADR-031, ADR-032)
        let mut custom_tuning = TuningProfile::conservative_default();
        custom_tuning.apply_operator_tuning(
            ComputePool::A100,
            0,
            8,
            PriorityClass::Latency,
            4096,
            true,
            false,
        );

        let apply_res = runtime
            .execute_admin(AdminCommand::ApplyTuningProfile {
                model_id: "custom-model".to_owned(),
                profile: custom_tuning,
            })
            .expect("apply tuning profile");
        assert_eq!(
            apply_res,
            AdminResponse::TuningProfileApplied("custom-model".to_owned())
        );

        // Verify status after tuning profile applied
        let entry_tuned = runtime.get_model_entry("custom-model").expect("found");
        assert!(!entry_tuned.unprofiled);
        assert_eq!(entry_tuned.profile_status, "profiled");
        assert_eq!(entry_tuned.priority_class, "latency");
        assert_eq!(entry_tuned.cluster, "a100");
        assert_eq!(entry_tuned.a100_cores, 8);
        assert_eq!(entry_tuned.n_ctx, 4096);
        assert!(entry_tuned.lend_idle_cores);
        assert!(!entry_tuned.borrow_idle_cores);
        assert_eq!(entry_tuned.advisory_notice, None);

        // Attempting to apply tuning to nonexistent model fails
        let err = runtime
            .execute_admin(AdminCommand::ApplyTuningProfile {
                model_id: "nonexistent".to_owned(),
                profile: TuningProfile::conservative_default(),
            })
            .unwrap_err();
        assert!(matches!(err, ManagerError::ModelNotFound(_)));
    }

    #[test]
    fn test_audit_event_logging_lifecycle() {
        let runtime = build_test_runtime();
        runtime.clear_audit_events();
        assert!(runtime.audit_events().is_empty());

        // 1. Model registration emits ModelRegistered
        runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: "audit-model".to_owned(),
                weights_path: PathBuf::from("/models/audit.gguf"),
                cluster: "x100".to_owned(),
            })
            .expect("register model");

        let events = runtime.audit_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, ModelAuditEventType::ModelRegistered);
        assert_eq!(events[0].model_id, "audit-model");
        assert!(events[0].unprofiled);
        assert_eq!(events[0].profile_status, "unprofiled");
        assert_eq!(events[0].advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));

        // 2. Querying status emits ModelStatusQueried
        let _ = runtime
            .execute_admin(AdminCommand::GetModelStatus {
                model_id: "audit-model".to_owned(),
            })
            .expect("get status");

        let events = runtime.audit_events();
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[1].event_type,
            ModelAuditEventType::ModelStatusQueried
        );
        assert_eq!(events[1].model_id, "audit-model");
        assert!(events[1].unprofiled);

        // 3. Applying tuning profile emits TuningProfileApplied
        let mut profile = TuningProfile::conservative_default();
        profile.apply_operator_tuning(
            ComputePool::X100,
            4,
            0,
            PriorityClass::Throughput,
            1024,
            false,
            false,
        );

        runtime
            .apply_tuning_profile("audit-model", profile)
            .expect("apply tuning");

        let events = runtime.audit_events();
        assert_eq!(events.len(), 3);
        assert_eq!(
            events[2].event_type,
            ModelAuditEventType::TuningProfileApplied
        );
        assert_eq!(events[2].model_id, "audit-model");
        assert!(!events[2].unprofiled);
        assert_eq!(events[2].profile_status, "profiled");
        assert_eq!(events[2].advisory(), None);

        // 4. Admin query for audit events returns complete list
        match runtime
            .execute_admin(AdminCommand::GetAuditEvents)
            .expect("get audit events")
        {
            AdminResponse::AuditEvents(admin_events) => {
                assert_eq!(admin_events.len(), 3);
                assert_eq!(
                    admin_events[0].event_type,
                    ModelAuditEventType::ModelRegistered
                );
                assert_eq!(
                    admin_events[1].event_type,
                    ModelAuditEventType::ModelStatusQueried
                );
                assert_eq!(
                    admin_events[2].event_type,
                    ModelAuditEventType::TuningProfileApplied
                );
            }
            other => panic!("expected AuditEvents response, got {other:?}"),
        }
    }
}
