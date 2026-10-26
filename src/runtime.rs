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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, channel, sync_channel};
use std::sync::{Arc, PoisonError, RwLock};
use std::thread::JoinHandle;

use crate::capacity::ControlPlaneScheduler;
use crate::domain::{OsAffinityBackend, RealOsAffinityBackend};
use crate::error::ManagerError;
use crate::prefix::{PrefixCacheStats, PrefixMatchResult, PrefixRadixIndex};
use crate::prompt::{
    ChatTemplate, GgufTokenizer, IncrementalStopDetector, PromptOrchestrator, Tokenizer,
};
use k3_core::affinity::ResolvedControlCapacity;

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelPlacementRecord {
    model_id: String,
    cluster: String,
    pid: Option<u32>,
    is_alive: bool,
    generation: u64,
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
}

/// Administrative response returned by the control-plane runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    queue_depth: AtomicUsize,
    placements: RwLock<HashMap<String, ModelPlacementRecord>>,
    control_workers: Vec<ControlWorkerThread>,
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
        let mut control_workers = Vec::with_capacity(active_cores.len());

        for &core_id in &active_cores {
            let (tx, rx) = channel::<Box<dyn FnOnce(usize) + Send>>();
            let (ready_tx, ready_rx) = sync_channel::<()>(1);
            let backend_clone = Arc::clone(&affinity_backend);
            let thread_name = format!("k3-control-{core_id}");

            let handle = std::thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    // Apply OS affinity for this active control core (ADR-007, ADR-041)
                    let mask = 1_u64 << core_id;
                    let _ = backend_clone.set_affinity(0, mask);
                    let _ = ready_tx.send(());

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
            let _ = ready_rx.recv();

            control_workers.push(ControlWorkerThread {
                core_id,
                sender: tx,
                handle: Some(handle),
            });
        }

        Ok(Self {
            scheduler: ControlPlaneScheduler::new(capacity),
            affinity_backend,
            prefix_index: Arc::new(PrefixRadixIndex::new()),
            prompt_orchestrator: Arc::new(PromptOrchestrator::new()),
            queue_depth: AtomicUsize::new(0),
            placements: RwLock::new(HashMap::new()),
            control_workers,
        })
    }

    /// Returns the number of active control worker threads.
    #[must_use]
    pub fn active_thread_count(&self) -> usize {
        self.control_workers.len()
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

    /// Dispatches a control-plane task to the active control thread pool in round-robin sequence.
    ///
    /// The task executes strictly on one of the dedicated active control threads bound to `active_control_mask`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::WorkerIpc`] if the control worker thread channel is closed.
    pub fn dispatch_control_task<F, R>(&self, f: F) -> Result<R, ManagerError>
    where
        F: FnOnce(usize) -> R + Send + 'static,
        R: Send + 'static,
    {
        if self.control_workers.is_empty() {
            return Err(ManagerError::WorkerIpc(
                "no active control worker threads available".to_owned(),
            ));
        }

        let assigned_core = self.scheduler.schedule_control_task();
        let worker = self
            .control_workers
            .iter()
            .find(|w| w.core_id == assigned_core)
            .unwrap_or_else(|| {
                let idx = assigned_core % self.control_workers.len();
                &self.control_workers[idx]
            });

        let (resp_tx, resp_rx) = sync_channel(1);
        let boxed_task = Box::new(move |core: usize| {
            let res = f(core);
            let _ = resp_tx.send(res);
        });

        worker
            .sender
            .send(boxed_task)
            .map_err(|_| ManagerError::WorkerIpc("control worker thread is dead".to_owned()))?;

        resp_rx.recv().map_err(|_| {
            ManagerError::WorkerIpc("control worker thread failed to respond".to_owned())
        })
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

    /// Checks if `token_id` is a stop token for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if `model_id` is not registered.
    pub fn is_stop_token(&self, model_id: &str, token_id: u32) -> Result<bool, ManagerError> {
        self.prompt_orchestrator.is_stop_token(model_id, token_id)
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
    pub fn invalidate_prefix_model(&self, project_id: &str, model_id: &str) -> usize {
        self.prefix_index.invalidate_model(project_id, model_id)
    }

    /// Invalidates all cached prefix blocks for a project.
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
    pub fn increment_queue_depth(&self) -> usize {
        self.queue_depth.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrements admission queue depth upon request dispatch.
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

    /// Records or updates a model placement in the status plane.
    pub fn record_model_placement(&self, model_id: String, cluster: String, pid: Option<u32>) {
        let mut map = self
            .placements
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        map.insert(
            model_id.clone(),
            ModelPlacementRecord {
                model_id,
                cluster,
                pid,
                is_alive: true,
                generation: 1,
            },
        );
    }

    /// Removes a model placement from the status plane.
    pub fn remove_model_placement(&self, model_id: &str) {
        let mut map = self
            .placements
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        map.remove(model_id);
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

    // ========================================================================
    // Administrative Command Dispatch
    // ========================================================================

    /// Executes an administrative command without blocking on compute worker forward passes.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if the administrative command fails.
    pub fn execute_admin(&self, cmd: AdminCommand) -> Result<AdminResponse, ManagerError> {
        match cmd {
            AdminCommand::Ping => Ok(AdminResponse::Pong),
            AdminCommand::InspectCapacity => {
                Ok(AdminResponse::Capacity(self.scheduler.capacity().clone()))
            }
            AdminCommand::GetQueueDepth => Ok(AdminResponse::QueueDepth(self.queue_depth())),
            AdminCommand::GetActivePlacements => Ok(AdminResponse::ActivePlacements(
                self.active_model_placements(),
            )),
            AdminCommand::GetWorkerHealth => {
                Ok(AdminResponse::WorkerHealth(self.worker_health_snapshots()))
            }
            AdminCommand::GetPrefixCacheStats => {
                Ok(AdminResponse::PrefixCacheStats(self.prefix_cache_stats()))
            }
            AdminCommand::ClearPrefixCache {
                project_id,
                model_id,
            } => {
                let cleared = match (project_id, model_id) {
                    (Some(p), Some(m)) => self.invalidate_prefix_model(&p, &m),
                    (Some(p), None) => self.invalidate_prefix_project(&p),
                    (None, Some(m)) => {
                        let mut total = 0;
                        let projects: Vec<String> = {
                            let map = self
                                .placements
                                .read()
                                .unwrap_or_else(PoisonError::into_inner);
                            map.keys().cloned().collect()
                        };
                        for p in projects {
                            total += self.invalidate_prefix_model(&p, &m);
                        }
                        total
                    }
                    (None, None) => {
                        let stats = self.prefix_cache_stats();
                        self.prefix_index.clear();
                        stats.total_cached_blocks
                    }
                };
                Ok(AdminResponse::CacheCleared(cleared))
            }
            AdminCommand::RegisterModel {
                model_id,
                weights_path: _,
                cluster,
            } => {
                self.record_model_placement(model_id.clone(), cluster, None);
                Ok(AdminResponse::ModelRegistered(model_id))
            }
            AdminCommand::UnregisterModel { model_id } => {
                self.remove_model_placement(&model_id);
                Ok(AdminResponse::ModelUnregistered(model_id))
            }
        }
    }
}

impl Drop for ControlPlaneRuntime {
    fn drop(&mut self) {
        let mut workers = std::mem::take(&mut self.control_workers);
        for mut worker in workers.drain(..) {
            drop(worker.sender);
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
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
}
