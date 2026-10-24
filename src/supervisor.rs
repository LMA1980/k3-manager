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
use std::time::{Duration, Instant};

use k3_core::ipc::layout::{ArenaHeader, KvBlockMeta, KvBlockTag};
use k3_core::ipc::protocol::{
    AttachArenaCommand, IpcMessage, StatusCode, WorkerHeartbeatRequest, WorkerHeartbeatResponse,
    recv_frame, recv_msg_frame, send_msg_frame,
};
use k3_core::ipc::shm::SharedMemoryArena;

use crate::dispatch::{ForwardPassRequest, ForwardPassResult, dispatch_forward_pass};
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
}

impl ManagedWorker {
    /// Returns the placement configuration.
    #[must_use]
    pub const fn placement(&self) -> &WorkerPlacement {
        &self.placement
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

    /// Checks if the worker process is currently running and healthy.
    pub fn is_alive(&mut self) -> bool {
        if let Some(child) = &mut self.child {
            match child.try_wait() {
                Ok(None) => self.is_alive,
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
            false
        }
    }

    /// Sends an OS signal to the worker process (for fault containment testing).
    ///
    /// # Errors
    /// Returns [`ManagerError::WorkerIpc`] if sending signal fails.
    pub fn kill(&mut self, signal: libc::c_int) -> Result<(), ManagerError> {
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
            self.is_alive = false;
            // Close stream immediately to mirror socket termination
            self.stream = None;

            // Wait briefly for kernel to deliver signal and record exit status
            let start = Instant::now();
            while start.elapsed() < Duration::from_millis(200) {
                if let Ok(Some(_)) = child.try_wait() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(())
        } else {
            Err(ManagerError::WorkerIpc(
                "worker child process is not running".to_owned(),
            ))
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

    /// Sends a heartbeat request to the worker over UDS.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on communication failure.
    pub fn send_heartbeat(&mut self) -> Result<WorkerHeartbeatResponse, ManagerError> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| ManagerError::WorkerIpc("worker stream is closed".to_owned()))?;

        let req = IpcMessage::WorkerHeartbeatRequest(WorkerHeartbeatRequest { timestamp_ns: 1 });

        send_msg_frame(stream, &req, None).map_err(|e| {
            ManagerError::WorkerIpc(format!("failed to send WorkerHeartbeatRequest: {e}"))
        })?;

        let (reply, _) = recv_msg_frame(stream).map_err(|e| {
            ManagerError::WorkerIpc(format!("failed to receive WorkerHeartbeatResponse: {e}"))
        })?;

        match reply {
            IpcMessage::WorkerHeartbeatResponse(resp) => Ok(resp),
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

    /// Dispatches a zero-copy forward pass to this worker.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on failure.
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

        let res = dispatch_forward_pass(stream, &mut self.arena, req);
        if res.is_err() {
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
        }
        res
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

    if let Some(aff) = &placement.affinity {
        cmd.arg("--affinity").arg(aff);
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
#[derive(Debug, Default)]
pub struct WorkerSupervisor {
    workers: HashMap<String, ManagedWorker>,
}

impl WorkerSupervisor {
    /// Constructs a new empty [`WorkerSupervisor`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            workers: HashMap::new(),
        }
    }

    /// Returns the number of managed workers.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
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

    /// Spawns a new isolated compute worker for the specified model placement and attaches its arena.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on spawn or IPC failure.
    #[allow(clippy::missing_panics_doc)]
    pub fn spawn_worker(
        &mut self,
        placement: WorkerPlacement,
        arena: SharedMemoryArena,
        arena_id: u64,
        arena_bytes: usize,
        n_ctx: u32,
        max_blocks: u32,
    ) -> Result<&mut ManagedWorker, ManagerError> {
        // If a worker with this model_id already exists, cleanly tear it down first
        if let Some(mut existing) = self.workers.remove(&placement.model_id) {
            let _ = existing.shutdown(false);
        }

        let (child, stream) =
            spawn_and_attach_process(&placement, &arena, arena_id, arena_bytes, n_ctx, max_blocks)?;

        let model_id = placement.model_id.clone();
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

        // 2. Spawn replacement worker and re-attach surviving arena via SCM_RIGHTS
        let (child, stream) = spawn_and_attach_process(
            &worker.placement,
            &worker.arena,
            worker.arena_id,
            worker.arena_bytes,
            worker.n_ctx,
            worker.max_blocks,
        )?;

        worker.child = Some(child);
        worker.stream = Some(stream);
        worker.generation += 1;
        worker.is_alive = true;

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
