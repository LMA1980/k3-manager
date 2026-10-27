//! Isolated compute worker supervisor and fault recovery lifecycle management.
//!
//! Supervises isolated `k3-worker` OS processes per model placement (ADR-018, ADR-041),
//! passes context memory arenas over UDS via `SCM_RIGHTS` (`AttachArenaCommand`),
//! manages shared-memory `ArenaHeader` and `KvBlockMeta` tags,
//! and orchestrates fault containment and clean recovery upon worker crash (Scenario 2).

use std::collections::HashMap;
use std::fs;
use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::capacity::ResolvedControlCapacityExt;
use k3_core::affinity::{
    ControlPlaneCapacityConfig, DEFAULT_X100_MAX_COMPUTE_CORES, ResolvedControlCapacity,
};
use k3_core::ipc::layout::{ArenaHeader, KvBlockMeta, KvBlockTag};
use k3_core::ipc::protocol::{
    AttachArenaCommand, IpcMessage, StatusCode, WorkerHeartbeatRequest, WorkerHeartbeatResponse,
    recv_frame, recv_msg_frame, send_msg_frame,
};
use k3_core::ipc::shm::SharedMemoryArena;
use k3_core::profile::TuningProfile;

use crate::dispatch::{ForwardPassRequest, ForwardPassResult, dispatch_forward_pass};
use crate::domain::{
    CgroupV2DomainController, IsolationDomainController, PlacementOutcome,
    resolve_worker_compute_mask, validate_model_id,
};
use crate::error::ManagerError;

/// Locates the `k3-worker` binary across environment variables, target outputs, and PATH.
///
/// # Errors
/// Returns [`ManagerError::WorkerSpawn`] if binary cannot be located.
pub fn find_worker_binary() -> Result<PathBuf, ManagerError> {
    if let Ok(env_bin) = std::env::var("K3_WORKER_BIN") {
        let p = PathBuf::from(env_bin);
        if p.exists() {
            return Ok(p);
        }
    }

    if let Some(bin) = option_env!("CARGO_BIN_EXE_k3-worker") {
        let p = PathBuf::from(bin);
        if p.exists() {
            return Ok(p);
        }
    }

    if let Ok(curr) = std::env::current_exe() {
        if let Some(dir) = curr.parent() {
            let candidate = dir.join("k3-worker");
            if candidate.exists() {
                return Ok(candidate);
            }
            let candidate_parent = dir.join("../k3-worker");
            if candidate_parent.exists() {
                return Ok(candidate_parent);
            }
        }
    }

    for path in &[
        "target/debug/k3-worker",
        "target/release/k3-worker",
        "../target/debug/k3-worker",
        "../target/release/k3-worker",
        "../../target/debug/k3-worker",
        "../../target/release/k3-worker",
    ] {
        let p = PathBuf::from(path);
        if p.exists() {
            return fs::canonicalize(&p).map_err(|e| {
                ManagerError::WorkerSpawn(format!("failed to canonicalize worker path: {e}"))
            });
        }
    }

    Ok(PathBuf::from("k3-worker"))
}

/// Default socket timeout for compute worker heartbeat requests (500ms).
pub const DEFAULT_HEARTBEAT_TIMEOUT: Duration = Duration::from_millis(500);

/// Default socket timeout for compute worker forward pass execution (5s).
pub const DEFAULT_PASS_TIMEOUT: Duration = Duration::from_secs(5);

/// Placement and isolation parameters for an isolated model compute worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerPlacement {
    /// Logical model identifier (e.g. `"qwen2.5-7b"`).
    pub model_id: String,
    /// Filesystem path to model weights file.
    pub weights_path: PathBuf,
    /// Unix Domain Socket path for supervisor IPC.
    pub socket_path: PathBuf,
    /// Target compute cluster (`"x100"` or `"a100"`).
    pub cluster: String,
    /// Worker thread count or parallel execution threads.
    pub threads: Option<u32>,
    /// Bounded operator memory ceiling in bytes.
    pub memlock_limit: Option<u64>,
    /// Thread affinity or core pinning specification.
    pub affinity: Option<String>,
    /// Explicit executable path override for `k3-worker`.
    pub worker_binary: Option<PathBuf>,
    /// Heartbeat timeout threshold override.
    pub heartbeat_timeout: Option<Duration>,
    /// Forward pass timeout threshold override.
    pub pass_timeout: Option<Duration>,
    /// Authoritative tuning profile configuration (ADR-031, ADR-032).
    pub tuning: Option<TuningProfile>,
}

impl WorkerPlacement {
    /// Creates a new placement specification with required parameters.
    #[must_use]
    pub fn new(
        model_id: impl Into<String>,
        weights_path: impl Into<PathBuf>,
        socket_path: impl Into<PathBuf>,
        cluster: impl Into<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            weights_path: weights_path.into(),
            socket_path: socket_path.into(),
            cluster: cluster.into(),
            threads: None,
            memlock_limit: None,
            affinity: None,
            worker_binary: None,
            heartbeat_timeout: None,
            pass_timeout: None,
            tuning: None,
        }
    }

    /// Attaches an operator tuning profile or conservative default.
    #[must_use]
    pub fn with_tuning(mut self, tuning: TuningProfile) -> Self {
        self.tuning = Some(tuning);
        self
    }

    /// Attaches a tuning profile.
    #[must_use]
    pub fn with_profile(mut self, profile: TuningProfile) -> Self {
        self.tuning = Some(profile);
        self
    }

    /// Creates an unprofiled placement with conservative defaults (ADR-028, ADR-031).
    #[must_use]
    pub fn unprofiled(
        model_id: impl Into<String>,
        weights_path: impl Into<PathBuf>,
        socket_path: impl Into<PathBuf>,
    ) -> Self {
        let model_id_str = model_id.into();
        let mut tuning = TuningProfile::conservative_default();
        tuning.model_id = Some(model_id_str.clone());
        Self {
            model_id: model_id_str,
            weights_path: weights_path.into(),
            socket_path: socket_path.into(),
            cluster: "x100".to_owned(),
            threads: Some(u32::try_from(DEFAULT_X100_MAX_COMPUTE_CORES).unwrap_or(6)),
            memlock_limit: None,
            affinity: None,
            worker_binary: None,
            heartbeat_timeout: None,
            pass_timeout: None,
            tuning: Some(tuning),
        }
    }

    /// Sets thread count.
    #[must_use]
    pub const fn with_threads(mut self, threads: u32) -> Self {
        self.threads = Some(threads);
        self
    }

    /// Sets memlock ceiling in bytes.
    #[must_use]
    pub const fn with_memlock_limit(mut self, limit: u64) -> Self {
        self.memlock_limit = Some(limit);
        self
    }

    /// Sets core pinning affinity.
    #[must_use]
    pub fn with_affinity(mut self, affinity: impl Into<String>) -> Self {
        self.affinity = Some(affinity.into());
        self
    }

    /// Sets explicit worker binary path.
    #[must_use]
    pub fn with_worker_binary(mut self, binary: impl Into<PathBuf>) -> Self {
        self.worker_binary = Some(binary.into());
        self
    }

    /// Sets explicit heartbeat timeout override.
    #[must_use]
    pub const fn with_heartbeat_timeout(mut self, timeout: Duration) -> Self {
        self.heartbeat_timeout = Some(timeout);
        self
    }

    /// Sets explicit forward pass timeout override.
    #[must_use]
    pub const fn with_pass_timeout(mut self, timeout: Duration) -> Self {
        self.pass_timeout = Some(timeout);
        self
    }
}

/// Supervised compute worker instance managing an isolated OS process.
#[derive(Debug)]
pub struct ManagedWorker {
    placement: WorkerPlacement,
    child: Option<Child>,
    stream: Option<UnixStream>,
    arena_id: u64,
    arena_bytes: usize,
    n_ctx: u32,
    max_blocks: u32,
    arena: SharedMemoryArena,
    generation: u64,
    is_alive: bool,
    last_placement_outcome: Option<PlacementOutcome>,
    heartbeat_timeout: Duration,
    pass_timeout: Duration,
    active_pass_start: Option<Instant>,
    last_heartbeat_latency: Option<Duration>,
    last_heartbeat_time: Option<Instant>,
    has_timed_out: bool,
}

impl ManagedWorker {
    /// Returns the placement configuration.
    #[must_use]
    pub const fn placement(&self) -> &WorkerPlacement {
        &self.placement
    }

    /// Returns the context window limit in tokens.
    #[must_use]
    pub const fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Returns the attached tuning profile, if any.
    #[must_use]
    pub fn tuning(&self) -> Option<&TuningProfile> {
        self.placement.tuning.as_ref()
    }

    /// Returns `true` if this worker is running under unprofiled conservative defaults.
    #[must_use]
    pub fn is_unprofiled(&self) -> bool {
        self.placement.tuning.as_ref().is_none_or(|t| t.unprofiled)
    }

    /// Returns the model identifier.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.placement.model_id
    }

    /// Returns the OS process ID if running.
    #[must_use]
    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }

    /// Returns the arena identifier.
    #[must_use]
    pub const fn arena_id(&self) -> u64 {
        self.arena_id
    }

    /// Returns a reference to the mapped shared memory arena.
    #[must_use]
    pub const fn arena(&self) -> &SharedMemoryArena {
        &self.arena
    }

    /// Returns a mutable reference to the mapped shared memory arena.
    pub fn arena_mut(&mut self) -> &mut SharedMemoryArena {
        &mut self.arena
    }

    /// Returns the generation count (incremented on each spawn/recovery).
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the most recent isolation domain placement outcome for this worker, if available.
    #[must_use]
    pub const fn last_placement_outcome(&self) -> Option<&PlacementOutcome> {
        self.last_placement_outcome.as_ref()
    }

    /// Returns the configured socket heartbeat timeout for this worker.
    #[must_use]
    pub const fn heartbeat_timeout(&self) -> Duration {
        self.heartbeat_timeout
    }

    /// Sets the socket heartbeat timeout for this worker.
    pub fn set_heartbeat_timeout(&mut self, timeout: Duration) {
        self.heartbeat_timeout = timeout;
    }

    /// Returns the configured forward pass timeout for this worker.
    #[must_use]
    pub const fn pass_timeout(&self) -> Duration {
        self.pass_timeout
    }

    /// Sets the forward pass timeout for this worker.
    pub fn set_pass_timeout(&mut self, timeout: Duration) {
        self.pass_timeout = timeout;
    }

    /// Returns the start instant of the currently executing forward pass, if any.
    #[must_use]
    pub const fn active_pass_start(&self) -> Option<Instant> {
        self.active_pass_start
    }

    /// Returns elapsed duration of the currently executing forward pass, if any.
    #[must_use]
    pub fn active_pass_elapsed(&self) -> Option<Duration> {
        self.active_pass_start.map(|t| t.elapsed())
    }

    /// Returns `true` if a forward pass is currently in flight.
    #[must_use]
    pub const fn is_pass_in_flight(&self) -> bool {
        self.active_pass_start.is_some()
    }

    /// Returns the measured roundtrip latency of the most recent heartbeat.
    #[must_use]
    pub const fn last_heartbeat_latency(&self) -> Option<Duration> {
        self.last_heartbeat_latency
    }

    /// Returns the timestamp of the most recent successful heartbeat.
    #[must_use]
    pub const fn last_heartbeat_time(&self) -> Option<Instant> {
        self.last_heartbeat_time
    }

    /// Returns `true` if this worker has timed out or breached a deadline.
    #[must_use]
    pub const fn has_timed_out(&self) -> bool {
        self.has_timed_out
    }

    /// Sets the timed out flag on this worker.
    pub fn set_timed_out(&mut self, timed_out: bool) {
        self.has_timed_out = timed_out;
    }

    /// Constructs a simulated [`ManagedWorker`] for testing deadline-bounded IPC and fault recovery.
    #[doc(hidden)]
    #[must_use]
    pub fn create_test_worker(
        placement: WorkerPlacement,
        stream: UnixStream,
        arena: SharedMemoryArena,
        arena_id: u64,
        arena_bytes: usize,
        n_ctx: u32,
        max_blocks: u32,
    ) -> Self {
        let heartbeat_timeout = placement
            .heartbeat_timeout
            .unwrap_or(DEFAULT_HEARTBEAT_TIMEOUT);
        let pass_timeout = placement.pass_timeout.unwrap_or(DEFAULT_PASS_TIMEOUT);
        Self {
            placement,
            child: None,
            stream: Some(stream),
            arena_id,
            arena_bytes,
            n_ctx,
            max_blocks,
            arena,
            generation: 1,
            is_alive: true,
            last_placement_outcome: None,
            heartbeat_timeout,
            pass_timeout,
            active_pass_start: None,
            last_heartbeat_latency: None,
            last_heartbeat_time: None,
            has_timed_out: false,
        }
    }

    /// Checks if the worker process is currently running and healthy.
    pub fn is_alive(&mut self) -> bool {
        if let Some(child) = &mut self.child {
            match child.try_wait() {
                Ok(None) => {
                    self.is_alive = true;
                    true
                }
                Ok(Some(_status)) => {
                    self.is_alive = false;
                    false
                }
                Err(_) => {
                    self.is_alive = false;
                    false
                }
            }
        } else {
            self.is_alive
        }
    }

    /// Sends an OS signal to the worker process (for fault containment testing).
    ///
    /// # Errors
    /// Returns [`ManagerError::WorkerIpc`] if sending signal fails.
    pub fn kill(&mut self, signal: libc::c_int) -> Result<(), ManagerError> {
        let is_non_terminating = matches!(
            signal,
            libc::SIGSTOP
                | libc::SIGCONT
                | libc::SIGTSTP
                | libc::SIGTTIN
                | libc::SIGTTOU
                | libc::SIGCHLD
                | libc::SIGWINCH
                | libc::SIGURG
        );

        if let Some(child) = &mut self.child {
            let pid = child.id();
            // SAFETY: kill sends standard signal to supervised child pid.
            let ret = unsafe { libc::kill(pid as libc::pid_t, signal) };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                return Err(ManagerError::WorkerIpc(format!(
                    "failed to send signal {signal} to pid {pid}: {err}"
                )));
            }

            if is_non_terminating {
                // Non-terminating signal: process remains alive in OS; stream remains open
                return Ok(());
            }

            // Wait briefly for kernel to deliver signal and record exit status
            let start = Instant::now();
            while start.elapsed() < Duration::from_millis(50) {
                if let Ok(Some(_)) = child.try_wait() {
                    self.is_alive = false;
                    self.stream = None;
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(5));
            }

            if let Ok(Some(_)) = child.try_wait() {
                self.is_alive = false;
                self.stream = None;
            }

            Ok(())
        } else {
            if is_non_terminating {
                return Ok(());
            }
            self.is_alive = false;
            self.stream = None;
            Ok(())
        }
    }

    /// Reaps the worker process if it has exited.
    ///
    /// # Errors
    /// Returns [`ManagerError::Io`] on wait failure.
    pub fn reap(&mut self) -> Result<Option<std::process::ExitStatus>, ManagerError> {
        if let Some(child) = &mut self.child {
            let status = child.try_wait()?;
            if status.is_some() {
                self.child = None;
                self.is_alive = false;
                self.stream = None;
            }
            Ok(status)
        } else {
            Ok(None)
        }
    }

    /// Waits synchronously for the child process to exit.
    ///
    /// # Errors
    /// Returns [`ManagerError::Io`] on wait failure.
    pub fn wait(&mut self) -> Result<std::process::ExitStatus, ManagerError> {
        if let Some(mut child) = self.child.take() {
            let status = child.wait()?;
            self.is_alive = false;
            self.stream = None;
            Ok(status)
        } else {
            Err(ManagerError::WorkerIpc(
                "no child process to wait on".to_owned(),
            ))
        }
    }

    /// Sends a heartbeat request to the worker over UDS with deadline-bounded timeout.
    ///
    /// # Errors
    /// Returns [`ManagerError::WorkerUnresponsive`] if heartbeat times out, or [`ManagerError`] on communication failure.
    #[allow(clippy::cast_possible_truncation)]
    pub fn send_heartbeat(&mut self) -> Result<WorkerHeartbeatResponse, ManagerError> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| ManagerError::WorkerIpc("worker stream is closed".to_owned()))?;

        let _ = stream.set_read_timeout(Some(self.heartbeat_timeout));
        let _ = stream.set_write_timeout(Some(self.heartbeat_timeout));

        let t0 = Instant::now();
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let req = IpcMessage::WorkerHeartbeatRequest(WorkerHeartbeatRequest { timestamp_ns });

        if let Err(e) = send_msg_frame(stream, &req, None) {
            let elapsed_ms = t0.elapsed().as_millis() as u64;
            let is_timeout = e.kind() == std::io::ErrorKind::WouldBlock
                || e.kind() == std::io::ErrorKind::TimedOut
                || e.raw_os_error() == Some(libc::EAGAIN)
                || e.raw_os_error() == Some(libc::ETIMEDOUT);
            if is_timeout {
                self.has_timed_out = true;
                return Err(ManagerError::WorkerUnresponsive {
                    model_id: self.placement.model_id.clone(),
                    pid: self.pid(),
                    elapsed_ms,
                });
            }
            return Err(ManagerError::WorkerIpc(format!(
                "failed to send WorkerHeartbeatRequest: {e}"
            )));
        }

        let (reply, _) = match recv_msg_frame(stream) {
            Ok(r) => r,
            Err(e) => {
                let elapsed_ms = t0.elapsed().as_millis() as u64;
                let is_timeout = e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut
                    || e.raw_os_error() == Some(libc::EAGAIN)
                    || e.raw_os_error() == Some(libc::ETIMEDOUT);
                if is_timeout {
                    self.has_timed_out = true;
                    return Err(ManagerError::WorkerUnresponsive {
                        model_id: self.placement.model_id.clone(),
                        pid: self.pid(),
                        elapsed_ms,
                    });
                }
                return Err(ManagerError::WorkerIpc(format!(
                    "failed to receive WorkerHeartbeatResponse: {e}"
                )));
            }
        };

        match reply {
            IpcMessage::WorkerHeartbeatResponse(resp) => {
                let latency = t0.elapsed();
                self.last_heartbeat_latency = Some(latency);
                self.last_heartbeat_time = Some(Instant::now());
                self.has_timed_out = false;
                Ok(resp)
            }
            other => Err(ManagerError::Protocol(format!(
                "expected WorkerHeartbeatResponse, got {other:?}"
            ))),
        }
    }

    /// Instructs the worker to shut down cleanly (`ShutdownCommand` / Type ID 206).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on failure.
    pub fn shutdown(&mut self, graceful: bool) -> Result<(), ManagerError> {
        if graceful {
            if let Some(stream) = &self.stream {
                let cmd = IpcMessage::ShutdownCommand(k3_core::ipc::protocol::ShutdownCommand {
                    graceful: true,
                });
                let _ = send_msg_frame(stream, &cmd, None);
            }
        }
        self.stream = None;
        if let Some(mut child) = self.child.take() {
            if graceful {
                let start = Instant::now();
                let mut exited = false;
                while start.elapsed() < Duration::from_millis(500) {
                    if let Ok(Some(_)) = child.try_wait() {
                        exited = true;
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                if !exited {
                    let _ = child.kill();
                }
            } else {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
        self.is_alive = false;
        let _ = fs::remove_file(&self.placement.socket_path);
        Ok(())
    }

    /// Dispatches a zero-copy forward pass to this worker with deadline-bounded timeout.
    ///
    /// # Errors
    /// Returns [`ManagerError::WorkerUnresponsive`] if pass times out, or [`ManagerError`] on failure.
    #[allow(clippy::cast_possible_truncation)]
    pub fn execute_forward_pass(
        &mut self,
        req: &ForwardPassRequest,
    ) -> Result<ForwardPassResult, ManagerError> {
        if !self.is_alive() {
            let status = self
                .child
                .as_mut()
                .and_then(|c| c.try_wait().ok().flatten());
            let exit_code = status.as_ref().and_then(std::process::ExitStatus::code);
            let signal = status.as_ref().and_then(ExitStatusExt::signal);
            return Err(ManagerError::WorkerCrashed {
                model_id: self.placement.model_id.clone(),
                exit_code,
                signal,
            });
        }

        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| ManagerError::WorkerIpc("worker UDS connection is dead".to_owned()))?;

        let _ = stream.set_read_timeout(Some(self.pass_timeout));
        let _ = stream.set_write_timeout(Some(self.pass_timeout));

        let t0 = Instant::now();
        self.active_pass_start = Some(t0);

        let res = dispatch_forward_pass(stream, &mut self.arena, req);
        self.active_pass_start = None;

        match res {
            Ok(val) => {
                self.has_timed_out = false;
                Ok(val)
            }
            Err(e) => {
                let elapsed_ms = t0.elapsed().as_millis() as u64;
                let is_timeout = match &e {
                    ManagerError::Io(io_err) => {
                        io_err.kind() == std::io::ErrorKind::WouldBlock
                            || io_err.kind() == std::io::ErrorKind::TimedOut
                            || io_err.raw_os_error() == Some(libc::EAGAIN)
                            || io_err.raw_os_error() == Some(libc::ETIMEDOUT)
                    }
                    ManagerError::WorkerIpc(msg) => {
                        msg.contains("timed out")
                            || msg.contains("Resource temporarily unavailable")
                            || msg.contains("Operation timed out")
                    }
                    _ => false,
                };

                if is_timeout {
                    self.has_timed_out = true;
                    return Err(ManagerError::WorkerUnresponsive {
                        model_id: self.placement.model_id.clone(),
                        pid: self.pid(),
                        elapsed_ms,
                    });
                }

                // Check if process crashed mid-pass
                if !self.is_alive() {
                    let status = self
                        .child
                        .as_mut()
                        .and_then(|c| c.try_wait().ok().flatten());
                    let exit_code = status.as_ref().and_then(std::process::ExitStatus::code);
                    let signal = status.as_ref().and_then(ExitStatusExt::signal);
                    return Err(ManagerError::WorkerCrashed {
                        model_id: self.placement.model_id.clone(),
                        exit_code,
                        signal,
                    });
                }

                Err(e)
            }
        }
    }

    /// Dispatches a zero-copy forward pass to this worker with deadline-bounded timeout.
    ///
    /// Alias for [`execute_forward_pass`](Self::execute_forward_pass) fulfilling US-06.3 (#75) requirements.
    ///
    /// # Errors
    /// Returns [`ManagerError::WorkerUnresponsive`] if pass times out, or [`ManagerError`] on failure.
    pub fn dispatch_forward_pass(
        &mut self,
        req: &ForwardPassRequest,
    ) -> Result<ForwardPassResult, ManagerError> {
        self.execute_forward_pass(req)
    }

    // ========================================================================
    // Shared Memory Arena & KV Block Meta Management (ADR-018, ADR-042)
    // ========================================================================

    /// Reads the [`ArenaHeader`] at offset 0 of the shared memory arena.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on bounds check failure.
    pub fn arena_header(&self) -> Result<ArenaHeader, ManagerError> {
        let hdr = *self.arena.header()?;
        Ok(hdr)
    }

    /// Sets retention tag on the specified KV cache block index (`purge`, `persist`, `volatile`).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on bounds check failure.
    pub fn set_block_tag(&mut self, block_idx: usize, tag: KvBlockTag) -> Result<(), ManagerError> {
        let block = self.arena.block_meta_mut(block_idx)?;
        block.set_tag(tag);
        Ok(())
    }

    /// Returns the metadata record for the specified KV cache block index.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on bounds check failure.
    pub fn get_block_meta(&self, block_idx: usize) -> Result<KvBlockMeta, ManagerError> {
        let block = *self.arena.block_meta(block_idx)?;
        Ok(block)
    }

    /// Updates the committed token count on the specified KV cache block index.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on bounds check failure.
    pub fn set_block_committed_tokens(
        &mut self,
        block_idx: usize,
        tokens: u32,
    ) -> Result<(), ManagerError> {
        let block = self.arena.block_meta_mut(block_idx)?;
        block.set_committed_tokens(tokens);
        Ok(())
    }

    /// Updates the 256-bit sampler PRNG state on the specified KV cache block index.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on bounds check failure.
    pub fn set_block_rng_state(
        &mut self,
        block_idx: usize,
        rng_state: [u64; 4],
    ) -> Result<(), ManagerError> {
        let block = self.arena.block_meta_mut(block_idx)?;
        block.sampler_rng_state = rng_state;
        Ok(())
    }

    /// Marks a block for purging and zeroing (`KvBlockTag::Purge`).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on bounds check failure.
    pub fn purge_block(&mut self, block_idx: usize) -> Result<(), ManagerError> {
        self.set_block_tag(block_idx, KvBlockTag::Purge)?;
        self.set_block_committed_tokens(block_idx, 0)?;
        Ok(())
    }

    /// Marks a block as persistent checkpoint (`KvBlockTag::Persist`).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on bounds check failure.
    pub fn persist_block(&mut self, block_idx: usize) -> Result<(), ManagerError> {
        self.set_block_tag(block_idx, KvBlockTag::Persist)
    }

    /// Marks a block as volatile context (`KvBlockTag::Volatile`).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on bounds check failure.
    pub fn mark_volatile_block(&mut self, block_idx: usize) -> Result<(), ManagerError> {
        self.set_block_tag(block_idx, KvBlockTag::Volatile)
    }
}

impl Drop for ManagedWorker {
    fn drop(&mut self) {
        let _ = self.shutdown(false);
    }
}

/// Spawns an isolated `k3-worker` process and establishes UDS IPC with arena transfer.
#[allow(clippy::too_many_lines, clippy::similar_names)]
fn spawn_and_attach_process(
    placement: &WorkerPlacement,
    arena: &SharedMemoryArena,
    arena_id: u64,
    arena_bytes: usize,
    n_ctx: u32,
    max_blocks: u32,
) -> Result<(Child, UnixStream), ManagerError> {
    // 1. Prepare socket path and bind listener
    let socket_path = &placement.socket_path;
    if socket_path.exists() {
        let _ = fs::remove_file(socket_path);
    }
    if let Some(parent) = socket_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    let listener = UnixListener::bind(socket_path).map_err(|e| {
        ManagerError::WorkerSpawn(format!(
            "failed to bind supervisor listener on '{}': {e}",
            socket_path.display()
        ))
    })?;

    listener.set_nonblocking(true).map_err(|e| {
        ManagerError::WorkerSpawn(format!("failed to set nonblocking listener: {e}"))
    })?;

    // 2. Resolve worker binary
    let bin_path = if let Some(bin) = &placement.worker_binary {
        bin.clone()
    } else {
        find_worker_binary()?
    };

    // 3. Spawn isolated compute worker process
    let mut cmd = Command::new(&bin_path);
    cmd.arg("--model").arg(&placement.weights_path);
    cmd.arg("--socket").arg(&placement.socket_path);
    cmd.arg("--cluster").arg(&placement.cluster);

    if placement.affinity.is_some() {
        let compute_mask = resolve_worker_compute_mask(placement)?;
        cmd.arg("--affinity").arg(format!("0x{compute_mask:04X}"));
    }
    if let Some(threads) = placement.threads {
        cmd.arg("--threads").arg(threads.to_string());
    }
    if let Some(limit) = placement.memlock_limit {
        cmd.arg("--memlock-limit").arg(limit.to_string());
    }
    cmd.arg("--model-id").arg(&placement.model_id);

    let mut child = cmd.spawn().map_err(|e| {
        let _ = fs::remove_file(socket_path);
        ManagerError::WorkerSpawn(format!(
            "failed to spawn worker process '{}': {e}",
            bin_path.display()
        ))
    })?;

    // 4. Accept incoming connection from worker with 5-second timeout and fast exit detection
    let start_wait = Instant::now();
    let timeout = Duration::from_secs(5);
    let mut stream: Option<UnixStream> = None;

    while start_wait.elapsed() < timeout {
        match listener.accept() {
            Ok((s, _)) => {
                stream = Some(s);
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Proactively detect premature child exit without waiting full timeout
                if let Ok(Some(status)) = child.try_wait() {
                    let _ = fs::remove_file(socket_path);
                    return Err(ManagerError::WorkerSpawn(format!(
                        "worker process exited prematurely with status {status}"
                    )));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = fs::remove_file(socket_path);
                return Err(ManagerError::WorkerSpawn(format!("accept error: {e}")));
            }
        }
    }

    let Some(stream) = stream else {
        let _ = child.kill();
        let _ = child.wait();
        let _ = fs::remove_file(socket_path);
        return Err(ManagerError::WorkerSpawn(format!(
            "worker failed to connect to socket '{}' within timeout",
            socket_path.display()
        )));
    };

    if let Err(e) = stream.set_nonblocking(false) {
        let _ = child.kill();
        let _ = child.wait();
        let _ = fs::remove_file(socket_path);
        return Err(ManagerError::WorkerSpawn(format!(
            "failed to set stream blocking: {e}"
        )));
    }

    let attach_timeout = Duration::from_secs(5);
    let _ = stream.set_read_timeout(Some(attach_timeout));
    let _ = stream.set_write_timeout(Some(attach_timeout));

    // 5. Transfer shared-memory arena FD via SCM_RIGHTS (AttachArenaCommand / Type ID 201)
    let arena_fd = match arena.duplicate_fd() {
        Ok(fd) => fd,
        Err(e) => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_file(socket_path);
            return Err(ManagerError::Shm(e));
        }
    };

    let attach_cmd = IpcMessage::AttachArenaCommand(AttachArenaCommand {
        arena_id,
        arena_bytes: arena_bytes as u64,
        n_ctx,
        max_blocks,
    });

    if let Err(e) = send_msg_frame(&stream, &attach_cmd, Some(arena_fd.as_raw_fd())) {
        let _ = child.kill();
        let _ = child.wait();
        let _ = fs::remove_file(socket_path);
        return Err(ManagerError::WorkerIpc(format!(
            "failed to send AttachArenaCommand via SCM_RIGHTS: {e}"
        )));
    }

    // 6. Await confirmation frame from worker (Type ID 206 with StatusCode::Success)
    let (header, payload, _) = match recv_frame(stream.as_raw_fd()) {
        Ok(f) => f,
        Err(e) => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_file(socket_path);
            return Err(ManagerError::WorkerIpc(format!(
                "failed to receive AttachArena response: {e}"
            )));
        }
    };

    if header.type_id != 206 {
        let _ = child.kill();
        let _ = child.wait();
        let _ = fs::remove_file(socket_path);
        return Err(ManagerError::Protocol(format!(
            "expected AttachArena response type 206, got {}",
            header.type_id
        )));
    }

    if payload.len() >= 4 {
        let status_code = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        let status = StatusCode::from_u32(status_code);
        if !status.is_success() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_file(socket_path);
            return Err(ManagerError::Status {
                status,
                message: format!("AttachArena returned failure status: {status:?}"),
            });
        }
    }

    Ok((child, stream))
}

/// Control-plane supervisor managing isolated compute workers and orchestrating recovery.
#[derive(Debug)]
pub struct WorkerSupervisor {
    workers: HashMap<String, ManagedWorker>,
    domain_controller: Arc<dyn IsolationDomainController>,
    capacity: ResolvedControlCapacity,
    default_heartbeat_timeout: Duration,
    default_pass_timeout: Duration,
}

impl Default for WorkerSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerSupervisor {
    /// Constructs a new [`WorkerSupervisor`] using the default cgroup v2 domain controller
    /// and default resolved control plane capacity.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn new() -> Self {
        let capacity = ControlPlaneCapacityConfig::default()
            .resolve_default()
            .expect("default control plane capacity resolution is guaranteed valid");
        Self {
            workers: HashMap::new(),
            domain_controller: Arc::new(CgroupV2DomainController::default()),
            capacity,
            default_heartbeat_timeout: DEFAULT_HEARTBEAT_TIMEOUT,
            default_pass_timeout: DEFAULT_PASS_TIMEOUT,
        }
    }

    /// Sets the custom [`IsolationDomainController`] for compute worker and control domain placement.
    pub fn set_domain_controller(&mut self, controller: Arc<dyn IsolationDomainController>) {
        self.domain_controller = controller;
    }

    /// Sets the custom [`IsolationDomainController`] using builder syntax.
    #[must_use]
    pub fn with_domain_controller(
        mut self,
        controller: Arc<dyn IsolationDomainController>,
    ) -> Self {
        self.domain_controller = controller;
        self
    }

    /// Sets the custom [`ResolvedControlCapacity`] for compute worker placement admission.
    pub fn set_capacity(&mut self, capacity: ResolvedControlCapacity) {
        self.capacity = capacity;
    }

    /// Sets the custom [`ResolvedControlCapacity`] using builder syntax.
    #[must_use]
    pub fn with_capacity(mut self, capacity: ResolvedControlCapacity) -> Self {
        self.capacity = capacity;
        self
    }

    /// Returns a reference to the active [`ResolvedControlCapacity`].
    #[must_use]
    pub const fn capacity(&self) -> &ResolvedControlCapacity {
        &self.capacity
    }

    /// Returns the default heartbeat timeout.
    #[must_use]
    pub const fn default_heartbeat_timeout(&self) -> Duration {
        self.default_heartbeat_timeout
    }

    /// Sets the default heartbeat timeout.
    pub fn set_default_heartbeat_timeout(&mut self, timeout: Duration) {
        self.default_heartbeat_timeout = timeout;
    }

    /// Sets the default heartbeat timeout using builder syntax.
    #[must_use]
    pub const fn with_default_heartbeat_timeout(mut self, timeout: Duration) -> Self {
        self.default_heartbeat_timeout = timeout;
        self
    }

    /// Returns the default forward pass timeout.
    #[must_use]
    pub const fn default_pass_timeout(&self) -> Duration {
        self.default_pass_timeout
    }

    /// Sets the default forward pass timeout.
    pub fn set_default_pass_timeout(&mut self, timeout: Duration) {
        self.default_pass_timeout = timeout;
    }

    /// Sets the default forward pass timeout using builder syntax.
    #[must_use]
    pub const fn with_default_pass_timeout(mut self, timeout: Duration) -> Self {
        self.default_pass_timeout = timeout;
        self
    }

    /// Returns a reference to the active [`IsolationDomainController`].
    #[must_use]
    pub fn domain_controller(&self) -> &Arc<dyn IsolationDomainController> {
        &self.domain_controller
    }

    /// Places the control plane into its isolated cgroup v2 domain (`.../control`) or OS affinity fallback.
    ///
    /// If `target_pid` is 0, targets the current process ID (`std::process::id()`).
    ///
    /// # Errors
    /// Returns [`ManagerError`] if domain placement fails.
    pub fn place_control_plane(
        &self,
        target_pid: u32,
        capacity: &ResolvedControlCapacity,
    ) -> Result<PlacementOutcome, ManagerError> {
        self.domain_controller
            .place_control_plane(target_pid, capacity)
    }

    /// Returns the number of managed workers.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Returns a sorted list of registered model identifiers.
    #[must_use]
    pub fn model_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.workers.keys().cloned().collect();
        ids.sort_unstable();
        ids
    }

    /// Returns `true` if a worker for `model_id` is registered.
    #[must_use]
    pub fn contains_model(&self, model_id: &str) -> bool {
        self.workers.contains_key(model_id)
    }

    /// Returns a reference to the [`ManagedWorker`] for `model_id`.
    #[must_use]
    pub fn get_worker(&self, model_id: &str) -> Option<&ManagedWorker> {
        self.workers.get(model_id)
    }

    /// Returns a mutable reference to the [`ManagedWorker`] for `model_id`.
    pub fn get_worker_mut(&mut self, model_id: &str) -> Option<&mut ManagedWorker> {
        self.workers.get_mut(model_id)
    }

    /// Registers a pre-constructed [`ManagedWorker`] into the supervisor.
    pub fn register_worker(&mut self, worker: ManagedWorker) {
        self.workers.insert(worker.model_id().to_owned(), worker);
    }

    /// Spawns a new isolated compute worker for the specified model placement,
    /// attaches its arena, and enforces cgroup v2 or OS affinity isolation.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on spawn, IPC, or isolation domain placement failure.
    #[allow(clippy::missing_panics_doc)]
    pub fn spawn_worker(
        &mut self,
        mut placement: WorkerPlacement,
        arena: SharedMemoryArena,
        arena_id: u64,
        arena_bytes: usize,
        n_ctx: u32,
        max_blocks: u32,
    ) -> Result<&mut ManagedWorker, ManagerError> {
        // Validate model identifier and placement capacity upfront before spawning process
        validate_model_id(&placement.model_id)?;
        self.capacity.validate_worker_placement(&placement)?;
        let compute_mask = resolve_worker_compute_mask(&placement)?;

        // Enforce unprofiled conservative defaults and operator tuning invariants (US-07 Scenario 1, ADR-028, ADR-030, ADR-031)
        let is_unprofiled = placement.tuning.as_ref().is_none_or(|t| t.unprofiled);

        if is_unprofiled {
            // Unprofiled models must reside on X100 only (a100_cores == 0 per ADR-028)
            if !placement.cluster.eq_ignore_ascii_case("x100") {
                return Err(ManagerError::Affinity(k3_core::K3Error::AffinityError {
                    requested_mask: compute_mask,
                    requested_threads: placement.threads.unwrap_or(0) as usize,
                    reason: format!(
                        "unprofiled model '{}' cannot be placed on cluster '{}'; unprofiled defaults require X100 compute pool only (ADR-028, ADR-031)",
                        placement.model_id, placement.cluster
                    ),
                }));
            }

            // Unprofiled models cannot enable dynamic idle core lending or borrowing (ADR-030) or allocate A100 cores
            if let Some(ref t) = placement.tuning {
                if t.a100_cores > 0 {
                    return Err(ManagerError::Affinity(k3_core::K3Error::AffinityError {
                        requested_mask: compute_mask,
                        requested_threads: placement.threads.unwrap_or(0) as usize,
                        reason: format!(
                            "unprofiled model '{}' cannot allocate A100 cores (a100_cores: {}); unprofiled defaults require a100_cores == 0 (ADR-028, ADR-031)",
                            placement.model_id, t.a100_cores
                        ),
                    }));
                }
                if t.lend_idle_cores || t.borrow_idle_cores {
                    return Err(ManagerError::WorkerSpawn(format!(
                        "unprofiled model '{}' cannot enable dynamic idle core lending or borrowing (ADR-030)",
                        placement.model_id
                    )));
                }
            }

            // Attach conservative defaults if not explicitly set
            if placement.tuning.is_none() {
                let mut def = TuningProfile::conservative_default_with_n_ctx(n_ctx as usize);
                def.model_id = Some(placement.model_id.clone());
                placement.tuning = Some(def);
            }
        }

        // If a worker with this model_id already exists, cleanly tear it down first
        if let Some(mut existing) = self.workers.remove(&placement.model_id) {
            let _ = existing.shutdown(false);
        }

        let (child, stream) =
            spawn_and_attach_process(&placement, &arena, arena_id, arena_bytes, n_ctx, max_blocks)?;

        let pid = child.id();
        let placement_outcome =
            match self
                .domain_controller
                .place_worker(pid, &placement.model_id, compute_mask)
            {
                Ok(outcome) => outcome,
                Err(e) => {
                    let mut dead_child = child;
                    let _ = dead_child.kill();
                    let _ = dead_child.wait();
                    let _ = fs::remove_file(&placement.socket_path);
                    return Err(e);
                }
            };

        let model_id = placement.model_id.clone();
        let heartbeat_timeout = placement
            .heartbeat_timeout
            .unwrap_or(self.default_heartbeat_timeout);
        let pass_timeout = placement.pass_timeout.unwrap_or(self.default_pass_timeout);

        let worker = ManagedWorker {
            placement,
            child: Some(child),
            stream: Some(stream),
            arena_id,
            arena_bytes,
            n_ctx,
            max_blocks,
            arena,
            generation: 1,
            is_alive: true,
            last_placement_outcome: Some(placement_outcome),
            heartbeat_timeout,
            pass_timeout,
            active_pass_start: None,
            last_heartbeat_latency: None,
            last_heartbeat_time: None,
            has_timed_out: false,
        };

        self.workers.insert(model_id.clone(), worker);
        Ok(self.workers.get_mut(&model_id).expect("worker inserted"))
    }

    /// Checks if a worker for `model_id` is alive.
    pub fn is_worker_alive(&mut self, model_id: &str) -> bool {
        self.workers
            .get_mut(model_id)
            .is_some_and(ManagedWorker::is_alive)
    }

    /// Dispatches a forward pass to the worker for `model_id`.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on failure.
    pub fn forward_pass(
        &mut self,
        model_id: &str,
        req: &ForwardPassRequest,
    ) -> Result<ForwardPassResult, ManagerError> {
        let worker = self
            .workers
            .get_mut(model_id)
            .ok_or_else(|| ManagerError::ModelNotFound(model_id.to_owned()))?;

        worker.execute_forward_pass(req)
    }

    /// Dispatches a forward pass to the worker for `model_id`.
    ///
    /// Alias for [`forward_pass`](Self::forward_pass) fulfilling US-06.3 (#75) requirements.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on failure.
    pub fn dispatch_forward_pass(
        &mut self,
        model_id: &str,
        req: &ForwardPassRequest,
    ) -> Result<ForwardPassResult, ManagerError> {
        self.forward_pass(model_id, req)
    }

    /// Reaps a crashed or exited worker process for `model_id`.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on failure.
    pub fn reap_worker(
        &mut self,
        model_id: &str,
    ) -> Result<Option<std::process::ExitStatus>, ManagerError> {
        let worker = self
            .workers
            .get_mut(model_id)
            .ok_or_else(|| ManagerError::ModelNotFound(model_id.to_owned()))?;
        worker.reap()
    }

    /// Recovers a failed worker process: reaps the old child, spawns a replacement worker process,
    /// re-attaches it to the surviving context arena via `SCM_RIGHTS`, and increments the generation counter.
    ///
    /// The shared-memory arena, committed KV blocks, retention tags, and PRNG states remain intact.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on recovery failure.
    #[allow(clippy::missing_panics_doc)]
    pub fn recover_worker(&mut self, model_id: &str) -> Result<&mut ManagedWorker, ManagerError> {
        let worker = self
            .workers
            .get_mut(model_id)
            .ok_or_else(|| ManagerError::ModelNotFound(model_id.to_owned()))?;

        // 1. Cleanly ensure previous child process is reaped/terminated
        if let Some(mut child) = worker.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        worker.is_alive = false;
        worker.stream = None;

        validate_model_id(&worker.placement.model_id)?;
        self.capacity.validate_worker_placement(&worker.placement)?;
        let compute_mask = resolve_worker_compute_mask(&worker.placement)?;

        // 2. Spawn replacement worker and re-attach surviving arena via SCM_RIGHTS
        let (child, stream) = spawn_and_attach_process(
            &worker.placement,
            &worker.arena,
            worker.arena_id,
            worker.arena_bytes,
            worker.n_ctx,
            worker.max_blocks,
        )?;

        let pid = child.id();
        let placement_outcome =
            match self
                .domain_controller
                .place_worker(pid, &worker.placement.model_id, compute_mask)
            {
                Ok(outcome) => outcome,
                Err(e) => {
                    let mut dead_child = child;
                    let _ = dead_child.kill();
                    let _ = dead_child.wait();
                    let _ = fs::remove_file(&worker.placement.socket_path);
                    return Err(e);
                }
            };

        worker.child = Some(child);
        worker.stream = Some(stream);
        worker.generation += 1;
        worker.is_alive = true;
        worker.last_placement_outcome = Some(placement_outcome);
        worker.active_pass_start = None;
        worker.has_timed_out = false;

        Ok(self.workers.get_mut(model_id).expect("worker exists"))
    }

    /// Shuts down and unregisters all supervised workers cleanly.
    pub fn shutdown_all(&mut self) {
        for worker in self.workers.values_mut() {
            let _ = worker.shutdown(true);
        }
        self.workers.clear();
    }
}

impl Drop for WorkerSupervisor {
    fn drop(&mut self) {
        self.shutdown_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k3_core::profile::PriorityClass;

    #[test]
    fn test_worker_placement_unprofiled_constructor() {
        let placement =
            WorkerPlacement::unprofiled("unprofiled-model", "/weights/test.gguf", "/tmp/test.sock");
        assert_eq!(placement.model_id, "unprofiled-model");
        assert_eq!(placement.cluster, "x100");
        assert_eq!(placement.threads, Some(6));
        assert!(placement.affinity.is_none());
        assert!(placement.tuning.is_some());

        let tuning = placement.tuning.as_ref().unwrap();
        assert!(tuning.unprofiled);
        assert_eq!(tuning.cluster, "x100");
        assert_eq!(tuning.a100_cores, 0);
        assert_eq!(tuning.x100_cores, 6);
        assert_eq!(tuning.priority_class, PriorityClass::Throughput);
        assert!(!tuning.lend_idle_cores);
        assert!(!tuning.borrow_idle_cores);
    }

    #[test]
    fn test_spawn_worker_rejects_unprofiled_on_a100() {
        let dummy_weights = PathBuf::from("/tmp/weights.gguf");
        let dummy_socket = PathBuf::from("/tmp/socket.sock");
        let arena = SharedMemoryArena::create("a100_unprofiled_arena", 4096).expect("create arena");
        let mut supervisor = WorkerSupervisor::new();

        // 8 threads on A100 passes capacity check, but unprofiled defaults require X100 only (ADR-028, ADR-031)
        let placement =
            WorkerPlacement::new("unprofiled_a100", &dummy_weights, &dummy_socket, "a100")
                .with_threads(8);

        let dup_arena = arena
            .duplicate_fd()
            .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
            .expect("dup arena");

        let err = supervisor
            .spawn_worker(placement, dup_arena, 1, 4096, 64, 4)
            .unwrap_err();

        assert!(matches!(err, ManagerError::Affinity(_)));
        let err_str = format!("{err}");
        assert!(err_str.contains("unprofiled defaults require X100 compute pool only"));
    }

    #[test]
    fn test_spawn_worker_rejects_unprofiled_with_core_lending() {
        let dummy_weights = PathBuf::from("/tmp/weights.gguf");
        let dummy_socket = PathBuf::from("/tmp/socket.sock");
        let arena =
            SharedMemoryArena::create("lending_unprofiled_arena", 4096).expect("create arena");
        let mut supervisor = WorkerSupervisor::new();

        let mut unprofiled_tuning = TuningProfile::conservative_default();
        unprofiled_tuning.lend_idle_cores = true;

        let placement =
            WorkerPlacement::new("unprofiled_lend", &dummy_weights, &dummy_socket, "x100")
                .with_threads(4)
                .with_tuning(unprofiled_tuning);

        let dup_arena = arena
            .duplicate_fd()
            .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
            .expect("dup arena");

        let err = supervisor
            .spawn_worker(placement, dup_arena, 1, 4096, 64, 4)
            .unwrap_err();

        assert!(matches!(err, ManagerError::WorkerSpawn(_)));
        let err_str = format!("{err}");
        assert!(err_str.contains("cannot enable dynamic idle core lending or borrowing"));
    }
}
