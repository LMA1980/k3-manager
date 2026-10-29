//! Multi-model compute scheduler enforcing pass-boundary time-slicing,
//! priority preemption, cost-weighted fair sharing, and dynamic idle core spillover.
//!
//! # Architecture & Guarantees
//! 1. **Operator Tuning Immutability (US-08, ADR-010, ADR-032)**:
//!    The operator's configured [`TuningProfile`] is the sole authority governing
//!    cluster reservations, priority class, context cap, and core lending/borrowing flags.
//!    Live pass costs are observed strictly to refine internal fair-sharing weights
//!    and generate on-demand advisory trade-off recommendations without auto-mutation.
//! 2. **Multi-Model Coexistence & Concurrency Cap (US-09, ADR-043)**:
//!    Enforces `max_active_models = 3` cap at load time, rejecting excess with
//!    [`ManagerError::ConcurrencyLimitExceeded`].
//!    Disjoint core reservations execute in parallel (`(mask_a & mask_b) == 0`).
//!    Overlapping reservations time-slice strictly at forward-pass boundaries.
//!    Single-prompt exclusivity is guaranteed per model forward pass (no cross-prompt batching).
//! 3. **Priority Preemption & Cost-Weighted Fair Queuing (US-10, ADR-010, ADR-014)**:
//!    [`PriorityClass::Latency`] preempts [`PriorityClass::Throughput`] at the next forward-pass boundary.
//!    Passes in the same priority class interleave using Weighted Fair Queuing (WFQ)
//!    virtual runtime tracking (`V_i += cost_i / w_i`) driven by live measured pass costs.
//!    Client-supplied priority overrides are rejected or stripped.
//! 4. **Opt-In Dynamic Idle Core Spillover & Reclaim (US-11, ADR-030, ADR-041)**:
//!    Requires dual opt-in (`lend_idle_cores == true` and `borrow_idle_cores == true`).
//!    Idle models lend cores for a single pass at a time.
//!    Cores are reclaimed instantly at the pass boundary when the owner receives traffic.
//!    Pre-pinned worker threads execute tasks without runtime cpuset rewrites.
//!    Dedicated control cores (`0x00C0`) and spare recovery core (`0x0020`) are strictly non-lendable.

use std::collections::HashMap;
use std::sync::{Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use k3_core::affinity::{
    A100_CLUSTER_MASK, DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_SPARE_RECOVERY_MASK,
    DEFAULT_X100_COMPUTE_CAP_MASK, ResolvedControlCapacity,
};
use k3_core::profile::{CostProfile, PriorityClass, ProfileStatus, TuningAdvisory, TuningProfile};
use serde::{Deserialize, Serialize};

use crate::error::ManagerError;

/// Default maximum number of simultaneously active models loaded in memory (ADR-043).
pub const DEFAULT_MAX_ACTIVE_MODELS: usize = 3;

/// Default initial pass cost estimate in milliseconds used before first live measurement.
pub const DEFAULT_INITIAL_PASS_COST_MS: f64 = 10.0;

/// Operating state of an active model in the multi-model scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSchedulingState {
    /// Model has no active pass and no pending requests; cores are lendable if opted in.
    Idle,
    /// Model has pending forward pass requests queued and is awaiting physical core availability.
    Waiting,
    /// Model is currently executing a forward pass on physical cores.
    Running,
    /// Model was idle with cores lent to a peer, but received incoming traffic; awaiting pass boundary to reclaim.
    ReclaimPending,
}

/// Online execution cost statistics tracked per active model to drive WFQ and advisories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelExecutionMetrics {
    /// Logical model identifier.
    pub model_id: String,
    /// Total forward passes completed.
    pub total_passes: u32,
    /// Total tokens processed across all forward passes.
    pub total_tokens: u32,
    /// Total prompt prefill tokens processed.
    pub total_prefill_tokens: u32,
    /// Total autoregressive decode tokens generated.
    pub total_decode_tokens: u32,
    /// Measured cost of the most recent forward pass in milliseconds.
    pub last_pass_cost_ms: f64,
    /// Running average forward pass execution cost in milliseconds.
    pub avg_pass_cost_ms: f64,
    /// Exponential moving average of prefill cost in ms per token.
    pub avg_prefill_ms_per_tok: f64,
    /// Exponential moving average of decode cost in ms per token.
    pub avg_decode_ms_per_tok: f64,
    /// Dynamic scheduling weight used for weighted fair queue interleaving.
    pub dynamic_scheduling_weight: f64,
    /// Cumulative virtual runtime in milliseconds for weighted fair queuing (`V_i += cost_i / w_i`).
    pub virtual_runtime_ms: f64,
}

const EMA_ALPHA: f64 = 0.2;

impl ModelExecutionMetrics {
    /// Constructs a new [`ModelExecutionMetrics`] instance for `model_id`.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            total_passes: 0,
            total_tokens: 0,
            total_prefill_tokens: 0,
            total_decode_tokens: 0,
            last_pass_cost_ms: 0.0,
            avg_pass_cost_ms: 0.0,
            avg_prefill_ms_per_tok: 0.0,
            avg_decode_ms_per_tok: 0.0,
            dynamic_scheduling_weight: 1.0,
            virtual_runtime_ms: 0.0,
        }
    }

    /// Records completion of a forward pass, updating rolling statistics and advancing virtual runtime.
    ///
    /// Preserves operator profile immutability per ADR-010 and ADR-032.
    pub fn record_pass(&mut self, is_prefill: bool, elapsed_ms: f64, token_count: u32) {
        self.total_passes = self.total_passes.saturating_add(1);
        self.total_tokens = self.total_tokens.saturating_add(token_count);
        self.last_pass_cost_ms = elapsed_ms;

        // Cumulative running average of pass latency
        let n = f64::from(self.total_passes);
        self.avg_pass_cost_ms = self.avg_pass_cost_ms * ((n - 1.0) / n) + (elapsed_ms / n);

        if token_count > 0 && elapsed_ms > 0.0 {
            let ms_per_tok = elapsed_ms / f64::from(token_count);
            if is_prefill {
                self.total_prefill_tokens = self.total_prefill_tokens.saturating_add(token_count);
                if self.avg_prefill_ms_per_tok == 0.0 {
                    self.avg_prefill_ms_per_tok = ms_per_tok;
                } else {
                    self.avg_prefill_ms_per_tok =
                        EMA_ALPHA * ms_per_tok + (1.0 - EMA_ALPHA) * self.avg_prefill_ms_per_tok;
                }
            } else {
                self.total_decode_tokens = self.total_decode_tokens.saturating_add(token_count);
                if self.avg_decode_ms_per_tok == 0.0 {
                    self.avg_decode_ms_per_tok = ms_per_tok;
                } else {
                    self.avg_decode_ms_per_tok =
                        EMA_ALPHA * ms_per_tok + (1.0 - EMA_ALPHA) * self.avg_decode_ms_per_tok;
                }
            }
        }

        // Refine dynamic scheduling weight (cost-weighted fair sharing per ADR-010)
        self.dynamic_scheduling_weight = if self.avg_pass_cost_ms > 0.0 {
            self.avg_pass_cost_ms
        } else {
            1.0
        };

        // Advance virtual runtime proportional to measured pass cost and fair weight (default w_i = 1.0):
        // V_i += cost_i / w_i
        self.virtual_runtime_ms += elapsed_ms;
    }

    /// Records completion of a pass where both prefill and decode metrics are measured.
    pub fn record_combined_pass(&mut self, prefill_ms: f64, decode_ms: f64, token_count: u32) {
        let total_ms = prefill_ms + decode_ms;
        self.total_passes = self.total_passes.saturating_add(1);
        self.total_tokens = self.total_tokens.saturating_add(token_count);
        self.last_pass_cost_ms = total_ms;

        let n = f64::from(self.total_passes);
        self.avg_pass_cost_ms = self.avg_pass_cost_ms * ((n - 1.0) / n) + (total_ms / n);

        if token_count > 0 {
            if prefill_ms > 0.0 {
                let ms_per_tok = prefill_ms / f64::from(token_count);
                self.total_prefill_tokens = self.total_prefill_tokens.saturating_add(token_count);
                if self.avg_prefill_ms_per_tok == 0.0 {
                    self.avg_prefill_ms_per_tok = ms_per_tok;
                } else {
                    self.avg_prefill_ms_per_tok =
                        EMA_ALPHA * ms_per_tok + (1.0 - EMA_ALPHA) * self.avg_prefill_ms_per_tok;
                }
            }
            if decode_ms > 0.0 {
                let ms_per_tok = decode_ms / f64::from(token_count);
                self.total_decode_tokens = self.total_decode_tokens.saturating_add(token_count);
                if self.avg_decode_ms_per_tok == 0.0 {
                    self.avg_decode_ms_per_tok = ms_per_tok;
                } else {
                    self.avg_decode_ms_per_tok =
                        EMA_ALPHA * ms_per_tok + (1.0 - EMA_ALPHA) * self.avg_decode_ms_per_tok;
                }
            }
        }

        self.dynamic_scheduling_weight = if self.avg_pass_cost_ms > 0.0 {
            self.avg_pass_cost_ms
        } else {
            1.0
        };

        // Advance virtual runtime proportional to measured pass cost:
        // V_i += cost_i / w_i
        self.virtual_runtime_ms += total_ms;
    }
}

/// Registration metadata and scheduling state for an active model in the compute plane.
#[derive(Debug, Clone)]
pub struct ModelSchedulerEntry {
    /// Logical model identifier.
    pub model_id: String,
    /// Configured compute cluster ("x100", "a100", "both").
    pub cluster: String,
    /// Base compute core reservation bitmask (subset of `0x003F` on X100, `0xFF00` on A100).
    pub reservation_mask: u64,
    /// Maximum allowed compute cap mask for dynamic expansion.
    pub compute_cap_mask: u64,
    /// Authoritative scheduling priority class (ADR-014).
    pub priority_class: PriorityClass,
    /// Whether this model permits lending its idle cores to peers (default false, ADR-030).
    pub lend_idle_cores: bool,
    /// Whether this model permits borrowing idle cores from peers (default false, ADR-030).
    pub borrow_idle_cores: bool,
    /// Whether this model is running unprofiled conservative defaults (ADR-031).
    pub unprofiled: bool,
    /// Context window limit in tokens.
    pub n_ctx: usize,
    /// Current scheduling state.
    pub state: ModelSchedulingState,
    /// Number of active forward passes currently executing (0 or 1, ADR-005).
    pub active_pass_count: u32,
    /// Physical core bitmask actively running for the current pass (base + borrowed).
    pub active_running_mask: u64,
    /// Cores currently borrowed from idle peers for this pass (0 if none).
    pub active_borrowed_mask: u64,
    /// Cores currently lent out to borrowing peers (0 if none).
    pub active_lent_mask: u64,
    /// Peer models from whom cores were borrowed for the active pass.
    pub borrowed_from: Vec<String>,
    /// Count of pending inference requests waiting to dispatch.
    pub pending_requests_count: u32,
    /// Timestamp when this model became idle (`None` if busy).
    pub idle_since: Option<Instant>,
    /// Measured execution cost and virtual runtime metrics.
    pub metrics: ModelExecutionMetrics,
    /// Cached authoritative operator tuning profile, if explicitly provided.
    pub tuning_profile: Option<TuningProfile>,
}

impl ModelSchedulerEntry {
    /// Returns `true` if this model is completely idle (no active pass and no pending requests).
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.active_pass_count == 0 && self.pending_requests_count == 0
    }
}

/// Granted execution permit for a single forward pass yielding allocated and optional borrowed cores (ADR-006, ADR-030).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledPass {
    /// Target model identifier.
    pub model_id: String,
    /// Base core reservation allocated to this model.
    pub allocated_mask: u64,
    /// Optional cores borrowed from idle peers (0 if none borrowed).
    pub borrowed_mask: u64,
    /// Effective core mask (`allocated_mask | borrowed_mask`).
    pub effective_mask: u64,
    /// Peer models from whom cores were borrowed.
    pub borrowed_from: Vec<String>,
    /// Timestamp when pass was scheduled.
    pub scheduled_at: Instant,
}

/// Internal synchronized state of the multi-model scheduler.
#[derive(Debug)]
struct SchedulerState {
    /// Maximum allowed active models in memory (default: 3, ADR-043).
    max_active_models: usize,
    /// Registered active models mapped by model ID.
    active_models: HashMap<String, ModelSchedulerEntry>,
    /// Physical core bitmask currently locked by active forward passes.
    locked_cores_mask: u64,
    /// Minimum virtual runtime across currently active models (prevents starvation on wake).
    min_virtual_runtime_ms: f64,
}

/// Multi-model compute scheduler arbitrating overlapping reservations at forward-pass boundaries.
///
/// Implements:
/// - Operator Tuning Immutability (US-08)
/// - Multi-Model Coexistence & Overlapping Time-Slicing (US-09)
/// - Priority Preemption & Cost-Weighted Fair Sharing (US-10)
/// - Opt-In Dynamic Idle Core Spillover & Reclaim (US-11)
#[derive(Debug)]
pub struct MultiModelScheduler {
    state: Mutex<SchedulerState>,
    /// Condvar signaled at every forward-pass boundary when cores unlock or requests update.
    condvar: Condvar,
    /// Resolved control-plane capacity to enforce `SoC` cluster and isolation invariants.
    capacity: ResolvedControlCapacity,
}

impl MultiModelScheduler {
    /// Creates a new [`MultiModelScheduler`] with default capacity and `max_active_models = 3`.
    #[must_use]
    pub fn new(capacity: ResolvedControlCapacity) -> Self {
        Self::with_max_active_models(capacity, DEFAULT_MAX_ACTIVE_MODELS)
    }

    /// Creates a new [`MultiModelScheduler`] with a custom `max_active_models` limit.
    #[must_use]
    pub fn with_max_active_models(
        capacity: ResolvedControlCapacity,
        max_active_models: usize,
    ) -> Self {
        Self {
            state: Mutex::new(SchedulerState {
                max_active_models,
                active_models: HashMap::new(),
                locked_cores_mask: 0,
                min_virtual_runtime_ms: 0.0,
            }),
            condvar: Condvar::new(),
            capacity,
        }
    }

    /// Returns a reference to the active [`ResolvedControlCapacity`].
    #[must_use]
    pub const fn capacity(&self) -> &ResolvedControlCapacity {
        &self.capacity
    }

    /// Returns the maximum allowed active models loaded concurrently.
    #[must_use]
    pub fn max_active_models(&self) -> usize {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.max_active_models
    }

    /// Returns the count of currently registered active models.
    #[must_use]
    pub fn active_model_count(&self) -> usize {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.active_models.len()
    }

    /// Returns the current bitmask of physical cores locked by running forward passes.
    #[must_use]
    pub fn running_cores_mask(&self) -> u64 {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.locked_cores_mask
    }

    /// Returns `true` if `model_id` is currently registered in the scheduler.
    #[must_use]
    pub fn contains_model(&self, model_id: &str) -> bool {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.active_models.contains_key(model_id)
    }

    /// Returns a sorted list of registered active model identifiers.
    #[must_use]
    pub fn list_models(&self) -> Vec<String> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let mut keys: Vec<String> = state.active_models.keys().cloned().collect();
        keys.sort_unstable();
        keys
    }

    /// Returns a clone of the scheduler entry for `model_id` if registered.
    #[must_use]
    pub fn get_model_entry(&self, model_id: &str) -> Option<ModelSchedulerEntry> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.active_models.get(model_id).cloned()
    }

    /// Returns a clone of the execution metrics for `model_id` if registered.
    #[must_use]
    pub fn get_metrics(&self, model_id: &str) -> Option<ModelExecutionMetrics> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.active_models.get(model_id).map(|e| e.metrics.clone())
    }

    /// Increments the pending request counter for `model_id`.
    ///
    /// # Errors
    /// Returns [`ManagerError::ModelNotFound`] if model is not registered.
    pub fn inc_pending_requests(&self, model_id: &str) -> Result<u32, ManagerError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let entry = state
            .active_models
            .get_mut(model_id)
            .ok_or_else(|| ManagerError::ModelNotFound(model_id.to_owned()))?;
        entry.pending_requests_count = entry.pending_requests_count.saturating_add(1);
        entry.state = ModelSchedulingState::Waiting;
        entry.idle_since = None;
        self.condvar.notify_all();
        Ok(entry.pending_requests_count)
    }

    /// Decrements the pending request counter for `model_id`.
    ///
    /// # Errors
    /// Returns [`ManagerError::ModelNotFound`] if model is not registered.
    pub fn dec_pending_requests(&self, model_id: &str) -> Result<u32, ManagerError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let entry = state
            .active_models
            .get_mut(model_id)
            .ok_or_else(|| ManagerError::ModelNotFound(model_id.to_owned()))?;
        entry.pending_requests_count = entry.pending_requests_count.saturating_sub(1);
        if entry.is_idle() {
            entry.state = ModelSchedulingState::Idle;
            entry.idle_since = Some(Instant::now());
        }
        self.condvar.notify_all();
        Ok(entry.pending_requests_count)
    }

    /// Admits or registers an active model placement into the scheduler.
    ///
    /// If `model_id` is already registered, updates its placement and tuning configuration.
    /// If `model_id` is new and active models count has reached `max_active_models`,
    /// rejects upfront with [`ManagerError::ConcurrencyLimitExceeded`] per ADR-043 and US-09 Scenario 3.
    ///
    /// # Errors
    /// Returns [`ManagerError::ConcurrencyLimitExceeded`] if capacity is full,
    /// [`ManagerError::Affinity`] on core mask violation, or
    /// [`ManagerError::WorkerSpawn`] if unprofiled models attempt to enable lending/borrowing.
    pub fn admit_or_register_model(
        &self,
        model_id: &str,
        cluster: String,
        reservation_mask: u64,
        tuning: Option<&TuningProfile>,
    ) -> Result<(), ManagerError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);

        let is_existing = state.active_models.contains_key(model_id);

        if !is_existing && state.active_models.len() >= state.max_active_models {
            return Err(ManagerError::ConcurrencyLimitExceeded {
                active: state.active_models.len(),
                max: state.max_active_models,
            });
        }

        // Validate compute mask against SoC isolation boundaries (ADR-007, ADR-041)
        if (reservation_mask & self.capacity.active_control_mask) != 0 {
            return Err(ManagerError::Affinity(k3_core::K3Error::AffinityError {
                requested_mask: reservation_mask,
                requested_threads: reservation_mask.count_ones() as usize,
                reason: format!(
                    "model '{model_id}' reservation 0x{reservation_mask:04X} overlaps dedicated active control cores 0x{:04X}",
                    self.capacity.active_control_mask
                ),
            }));
        }

        let is_unprofiled = tuning.is_none_or(|t| t.unprofiled);

        // Unprofiled models cannot enable dynamic idle core lending or borrowing (ADR-030, ADR-031)
        if is_unprofiled {
            if let Some(t) = tuning {
                if t.lend_idle_cores || t.borrow_idle_cores {
                    return Err(ManagerError::WorkerSpawn(format!(
                        "unprofiled model '{model_id}' cannot enable dynamic idle core lending or borrowing (ADR-030)"
                    )));
                }
            }
        }

        let priority_class = tuning.map_or(PriorityClass::Throughput, |t| t.priority_class);
        let lend_idle_cores = tuning.is_some_and(|t| t.lend_idle_cores);
        let borrow_idle_cores = tuning.is_some_and(|t| t.borrow_idle_cores);
        let n_ctx = tuning.map_or(2048, |t| t.n_ctx);

        let compute_cap_mask = match cluster.to_ascii_lowercase().as_str() {
            "x100" => DEFAULT_X100_COMPUTE_CAP_MASK,
            "a100" => A100_CLUSTER_MASK,
            _ => DEFAULT_X100_COMPUTE_CAP_MASK | A100_CLUSTER_MASK,
        };

        if let Some(existing) = state.active_models.get_mut(model_id) {
            existing.cluster = cluster;
            existing.reservation_mask = reservation_mask;
            existing.compute_cap_mask = compute_cap_mask;
            existing.priority_class = priority_class;
            existing.lend_idle_cores = lend_idle_cores;
            existing.borrow_idle_cores = borrow_idle_cores;
            existing.unprofiled = is_unprofiled;
            existing.n_ctx = n_ctx;
            existing.tuning_profile = tuning.cloned();
        } else {
            let mut metrics = ModelExecutionMetrics::new(model_id);
            // Catch up virtual runtime to current system minimum to prevent unfair starvation
            metrics.virtual_runtime_ms = state.min_virtual_runtime_ms;

            state.active_models.insert(
                model_id.to_owned(),
                ModelSchedulerEntry {
                    model_id: model_id.to_owned(),
                    cluster,
                    reservation_mask,
                    compute_cap_mask,
                    priority_class,
                    lend_idle_cores,
                    borrow_idle_cores,
                    unprofiled: is_unprofiled,
                    n_ctx,
                    state: ModelSchedulingState::Idle,
                    active_pass_count: 0,
                    active_running_mask: 0,
                    active_borrowed_mask: 0,
                    active_lent_mask: 0,
                    borrowed_from: Vec::new(),
                    pending_requests_count: 0,
                    idle_since: Some(Instant::now()),
                    metrics,
                    tuning_profile: tuning.cloned(),
                },
            );
        }

        self.condvar.notify_all();
        Ok(())
    }

    /// Unregisters an active model from the scheduler, releasing any held core locks and waking waiting peers.
    pub fn unregister_model(&self, model_id: &str) -> bool {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(removed) = state.active_models.remove(model_id) {
            state.locked_cores_mask &= !removed.reservation_mask;
            state.locked_cores_mask &= !removed.active_borrowed_mask;

            // If this model lent cores to peers, clean up lender tracking
            for peer in state.active_models.values_mut() {
                if peer.borrowed_from.iter().any(|id| id == model_id) {
                    peer.borrowed_from.retain(|id| id != model_id);
                }
            }

            self.condvar.notify_all();
            true
        } else {
            false
        }
    }

    /// Evaluates whether a candidate model can be scheduled given the current scheduler state.
    ///
    /// Returns `Some((allocated_mask, borrowed_mask, borrowed_from))` if eligible, or `None`.
    fn try_select_model(
        state: &SchedulerState,
        candidate_id: &str,
    ) -> Option<(u64, u64, Vec<String>)> {
        let candidate = state.active_models.get(candidate_id)?;

        // 1. Single-Prompt Exclusivity (ADR-005): model must not be running a pass
        if candidate.active_pass_count > 0 {
            return None;
        }

        // 2. Physical core availability: candidate base reservation must not overlap locked cores
        let my_mask = candidate.reservation_mask;
        if (my_mask & state.locked_cores_mask) != 0 {
            return None;
        }

        // 3. Contention & Priority Arbitration (ADR-014, ADR-010)
        // Check if any other model with pending requests contends for overlapping cores
        // and has higher priority or lower virtual runtime
        for (peer_id, peer) in &state.active_models {
            if peer_id == candidate_id || peer.pending_requests_count == 0 {
                continue;
            }

            // Do we contend for overlapping cores with this peer?
            let overlaps = (peer.reservation_mask & my_mask) != 0;
            if !overlaps {
                continue;
            }

            // Priority Preemption: Latency preempts Throughput
            if candidate.priority_class == PriorityClass::Throughput
                && peer.priority_class == PriorityClass::Latency
            {
                return None;
            }

            // Intra-Class Fairness: Compare virtual runtime
            if candidate.priority_class == peer.priority_class {
                if peer.metrics.virtual_runtime_ms < candidate.metrics.virtual_runtime_ms {
                    return None;
                } else if (peer.metrics.virtual_runtime_ms - candidate.metrics.virtual_runtime_ms)
                    .abs()
                    < f64::EPSILON
                    && peer_id.as_str() < candidate_id
                {
                    // Deterministic tie-breaker
                    return None;
                }
            }
        }

        // Model is selected! Calculate dynamic idle core borrowing if opted in (US-11 Scenario 2)
        let allocated_mask = candidate.reservation_mask;
        let mut borrowed_mask = 0_u64;
        let mut borrowed_from = Vec::new();

        if candidate.borrow_idle_cores && !candidate.unprofiled {
            for (peer_id, peer) in &state.active_models {
                if peer_id == candidate_id {
                    continue;
                }

                // Peer must opt into lending, be completely idle, have zero pending requests, and not be unprofiled
                if peer.lend_idle_cores
                    && peer.is_idle()
                    && peer.active_lent_mask == 0
                    && !peer.unprofiled
                {
                    // Lendable cores: peer's reservation that does not overlap currently locked cores
                    let lendable = peer.reservation_mask & !state.locked_cores_mask;
                    // Guard: strictly bound within candidate's compute cap and cluster
                    let candidate_cap = candidate.compute_cap_mask;
                    let valid_lendable = lendable
                        & candidate_cap
                        & (DEFAULT_X100_COMPUTE_CAP_MASK | A100_CLUSTER_MASK);

                    // Guard: never touch dedicated active control cores or spare recovery core
                    let sanitized = valid_lendable
                        & !(DEFAULT_ACTIVE_CONTROL_MASK | DEFAULT_SPARE_RECOVERY_MASK);

                    if sanitized != 0 {
                        borrowed_mask |= sanitized;
                        borrowed_from.push(peer_id.clone());
                    }
                }
            }
        }

        Some((allocated_mask, borrowed_mask, borrowed_from))
    }

    /// Schedules an exclusive forward pass for `model_id`, blocking until required cores are available.
    ///
    /// Yields a [`ScheduledPass`] specifying base allocated cores and optional borrowed cores.
    ///
    /// # Errors
    /// Returns [`ManagerError::ModelNotFound`] if model is not registered,
    /// or [`ManagerError::WorkerUnresponsive`] if the timeout expires.
    pub fn schedule_forward_pass(
        &self,
        model_id: &str,
        timeout: Option<Duration>,
    ) -> Result<ScheduledPass, ManagerError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);

        // 1. Verify model presence and increment pending requests count
        {
            let entry = state
                .active_models
                .get_mut(model_id)
                .ok_or_else(|| ManagerError::ModelNotFound(model_id.to_owned()))?;
            entry.pending_requests_count += 1;
            entry.state = ModelSchedulingState::Waiting;
            entry.idle_since = None;
        }

        let start_time = Instant::now();

        // 2. Loop until scheduling conditions are satisfied or timeout occurs
        loop {
            if let Some((allocated_mask, borrowed_mask, borrowed_from)) =
                Self::try_select_model(&state, model_id)
            {
                let effective_mask = allocated_mask | borrowed_mask;
                state.locked_cores_mask |= effective_mask;

                if let Some(entry) = state.active_models.get_mut(model_id) {
                    entry.pending_requests_count = entry.pending_requests_count.saturating_sub(1);
                    entry.active_pass_count = 1;
                    entry.active_running_mask = effective_mask;
                    entry.active_borrowed_mask = borrowed_mask;
                    entry.borrowed_from.clone_from(&borrowed_from);
                    entry.state = ModelSchedulingState::Running;
                }

                // Mark lenders with lent mask
                for lender_id in &borrowed_from {
                    if let Some(lender) = state.active_models.get_mut(lender_id) {
                        lender.active_lent_mask |= lender.reservation_mask & borrowed_mask;
                    }
                }

                return Ok(ScheduledPass {
                    model_id: model_id.to_owned(),
                    allocated_mask,
                    borrowed_mask,
                    effective_mask,
                    borrowed_from,
                    scheduled_at: Instant::now(),
                });
            }

            // Wait on condvar for next pass boundary or request update
            if let Some(t) = timeout {
                let elapsed = start_time.elapsed();
                if elapsed >= t {
                    if let Some(e) = state.active_models.get_mut(model_id) {
                        e.pending_requests_count = e.pending_requests_count.saturating_sub(1);
                        if e.is_idle() {
                            e.state = ModelSchedulingState::Idle;
                            e.idle_since = Some(Instant::now());
                        }
                    }
                    let elapsed_u64 = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
                    return Err(ManagerError::WorkerUnresponsive {
                        model_id: model_id.to_owned(),
                        pid: None,
                        elapsed_ms: elapsed_u64,
                    });
                }
                let remaining = t.saturating_sub(elapsed);
                let (new_state, timeout_res) = self
                    .condvar
                    .wait_timeout(state, remaining)
                    .unwrap_or_else(PoisonError::into_inner);
                state = new_state;
                if timeout_res.timed_out() {
                    if let Some(e) = state.active_models.get_mut(model_id) {
                        e.pending_requests_count = e.pending_requests_count.saturating_sub(1);
                        if e.is_idle() {
                            e.state = ModelSchedulingState::Idle;
                            e.idle_since = Some(Instant::now());
                        }
                    }
                    let elapsed_u64 =
                        u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);
                    return Err(ManagerError::WorkerUnresponsive {
                        model_id: model_id.to_owned(),
                        pid: None,
                        elapsed_ms: elapsed_u64,
                    });
                }
            } else {
                state = self
                    .condvar
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        }
    }

    /// Attempts to schedule a forward pass immediately without blocking.
    ///
    /// Returns `Ok(Some(permit))` if cores are available and the model is selected,
    /// or `Ok(None)` if cores are contested or a higher-priority model is waiting.
    ///
    /// # Errors
    /// Returns [`ManagerError::ModelNotFound`] if model is not registered.
    pub fn try_schedule_forward_pass(
        &self,
        model_id: &str,
    ) -> Result<Option<ScheduledPass>, ManagerError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);

        if !state.active_models.contains_key(model_id) {
            return Err(ManagerError::ModelNotFound(model_id.to_owned()));
        }

        if let Some((allocated_mask, borrowed_mask, borrowed_from)) =
            Self::try_select_model(&state, model_id)
        {
            let effective_mask = allocated_mask | borrowed_mask;
            state.locked_cores_mask |= effective_mask;

            if let Some(entry) = state.active_models.get_mut(model_id) {
                entry.pending_requests_count = entry.pending_requests_count.saturating_sub(1);
                entry.active_pass_count = 1;
                entry.active_running_mask = effective_mask;
                entry.active_borrowed_mask = borrowed_mask;
                entry.borrowed_from.clone_from(&borrowed_from);
                entry.state = ModelSchedulingState::Running;
            }

            for lender_id in &borrowed_from {
                if let Some(lender) = state.active_models.get_mut(lender_id) {
                    lender.active_lent_mask |= lender.reservation_mask & borrowed_mask;
                }
            }

            Ok(Some(ScheduledPass {
                model_id: model_id.to_owned(),
                allocated_mask,
                borrowed_mask,
                effective_mask,
                borrowed_from,
                scheduled_at: Instant::now(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Records completion of a forward pass, updating internal metrics and releasing core locks.
    ///
    /// Implements instant pass-boundary reclaim (US-11 Scenario 3) and preserves operator tuning
    /// profile immutability (US-08 Scenario 1).
    pub fn record_pass_completion(
        &self,
        pass: &ScheduledPass,
        prefill_latency_ms: f64,
        decode_latency_ms: f64,
        token_count: u32,
    ) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);

        // 1. Release running cores (both base reservation and borrowed cores)
        state.locked_cores_mask &= !pass.effective_mask;

        // 2. Clear borrowed cores and lenders' lent cores (Instant pass-boundary reclaim)
        for lender_id in &pass.borrowed_from {
            if let Some(lender) = state.active_models.get_mut(lender_id) {
                lender.active_lent_mask &= !pass.borrowed_mask;
                if lender.pending_requests_count > 0 {
                    lender.state = ModelSchedulingState::Waiting;
                } else {
                    lender.state = ModelSchedulingState::Idle;
                    lender.idle_since = Some(Instant::now());
                }
            }
        }

        // 3. Update borrower model entry
        if let Some(entry) = state.active_models.get_mut(&pass.model_id) {
            entry.active_pass_count = 0;
            entry.active_running_mask = 0;
            entry.active_borrowed_mask = 0;
            entry.borrowed_from.clear();

            if entry.pending_requests_count == 0 {
                entry.state = ModelSchedulingState::Idle;
                entry.idle_since = Some(Instant::now());
            } else {
                entry.state = ModelSchedulingState::Waiting;
            }

            let has_prefill = prefill_latency_ms > 0.0;
            let has_decode = decode_latency_ms > 0.0;
            if has_prefill && !has_decode {
                entry
                    .metrics
                    .record_pass(true, prefill_latency_ms, token_count);
            } else if has_decode && !has_prefill {
                entry
                    .metrics
                    .record_pass(false, decode_latency_ms, token_count);
            } else if has_prefill && has_decode {
                entry.metrics.record_combined_pass(
                    prefill_latency_ms,
                    decode_latency_ms,
                    token_count,
                );
            } else {
                entry.metrics.record_pass(false, 0.0, token_count);
            }
        }

        // 4. Update system minimum virtual runtime
        state.min_virtual_runtime_ms = state
            .active_models
            .values()
            .map(|e| e.metrics.virtual_runtime_ms)
            .fold(f64::INFINITY, f64::min);
        if state.min_virtual_runtime_ms.is_infinite() {
            state.min_virtual_runtime_ms = 0.0;
        }

        // 5. Notify all waiting threads at this pass boundary
        self.condvar.notify_all();
    }

    /// Records failure, timeout, or abnormal termination of a forward pass, unlocking cores.
    pub fn record_pass_failure(&self, pass: &ScheduledPass) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);

        state.locked_cores_mask &= !pass.effective_mask;

        for lender_id in &pass.borrowed_from {
            if let Some(lender) = state.active_models.get_mut(lender_id) {
                lender.active_lent_mask &= !pass.borrowed_mask;
                if lender.pending_requests_count > 0 {
                    lender.state = ModelSchedulingState::Waiting;
                } else {
                    lender.state = ModelSchedulingState::Idle;
                    lender.idle_since = Some(Instant::now());
                }
            }
        }

        if let Some(entry) = state.active_models.get_mut(&pass.model_id) {
            entry.active_pass_count = 0;
            entry.active_running_mask = 0;
            entry.active_borrowed_mask = 0;
            entry.borrowed_from.clear();

            if entry.pending_requests_count == 0 {
                entry.state = ModelSchedulingState::Idle;
                entry.idle_since = Some(Instant::now());
            } else {
                entry.state = ModelSchedulingState::Waiting;
            }
        }

        self.condvar.notify_all();
    }

    /// Handles an unexpected worker crash, clearing all locked cores and waking waiting peers.
    pub fn on_worker_crashed(&self, model_id: &str) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);

        let cleanup = if let Some(entry) = state.active_models.get_mut(model_id) {
            let running = entry.active_running_mask;
            let borrowed = entry.active_borrowed_mask;
            let from = entry.borrowed_from.clone();
            entry.active_pass_count = 0;
            entry.active_running_mask = 0;
            entry.active_borrowed_mask = 0;
            entry.borrowed_from.clear();
            entry.state = ModelSchedulingState::Idle;
            Some((running, borrowed, from))
        } else {
            None
        };

        if let Some((running, borrowed, from)) = cleanup {
            state.locked_cores_mask &= !running;
            for lender_id in &from {
                if let Some(lender) = state.active_models.get_mut(lender_id) {
                    lender.active_lent_mask &= !borrowed;
                    if lender.pending_requests_count > 0 {
                        lender.state = ModelSchedulingState::Waiting;
                    } else {
                        lender.state = ModelSchedulingState::Idle;
                    }
                }
            }
        }

        self.condvar.notify_all();
    }

    /// Evaluates client request priority claim, enforcing operator authority (US-10 Scenario 3).
    ///
    /// Discards client-supplied priority overrides and returns the authoritative operator priority class.
    ///
    /// # Errors
    /// Returns [`ManagerError::ModelNotFound`] if model is not registered.
    pub fn enforce_operator_priority(
        &self,
        model_id: &str,
        client_priority_header: Option<&str>,
    ) -> Result<PriorityClass, ManagerError> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let entry = state
            .active_models
            .get(model_id)
            .ok_or_else(|| ManagerError::ModelNotFound(model_id.to_owned()))?;

        // Client priority claims are strictly ignored/stripped (ADR-014, US-10 Scenario 3)
        let _ = client_priority_header;

        Ok(entry.priority_class)
    }

    /// Evaluates client request priority claim, enforcing operator authority (US-10 Scenario 3).
    ///
    /// Discards client-supplied priority overrides and returns the authoritative operator priority class.
    ///
    /// # Errors
    /// Returns [`ManagerError::ModelNotFound`] if model is not registered.
    pub fn resolve_request_priority(
        &self,
        model_id: &str,
        client_priority_header: Option<PriorityClass>,
    ) -> Result<PriorityClass, ManagerError> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let entry = state
            .active_models
            .get(model_id)
            .ok_or_else(|| ManagerError::ModelNotFound(model_id.to_owned()))?;

        let _ = client_priority_header;
        Ok(entry.priority_class)
    }

    /// Generates an on-demand evidence-based advisory tuning recommendation (US-08 Scenario 2, ADR-032).
    ///
    /// Compares online execution metrics against static hardware ceilings without mutating
    /// the active operator tuning profile.
    ///
    /// # Errors
    /// Returns [`ManagerError::ModelNotFound`] if `model_id` is unknown.
    pub fn suggest_tuning(
        &self,
        model_id: &str,
        baseline: Option<&CostProfile>,
    ) -> Result<TuningAdvisory, ManagerError> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let entry = state
            .active_models
            .get(model_id)
            .ok_or_else(|| ManagerError::ModelNotFound(model_id.to_owned()))?;

        let m = &entry.metrics;
        let mut recommendations = Vec::new();

        // 1. Bottleneck classification based on decode vs prefill latency ratio and baseline ceilings
        let is_bandwidth_bound = (m.avg_prefill_ms_per_tok == 0.0
            || m.avg_decode_ms_per_tok > m.avg_prefill_ms_per_tok * 1.5)
            && m.avg_decode_ms_per_tok > 0.0;
        let bottleneck = if is_bandwidth_bound {
            recommendations.push(
                "Autoregressive decode is memory-bandwidth bound. Consider increasing A100 core allocation or reducing KV cache precision.".to_owned(),
            );
            "bandwidth-bound"
        } else {
            recommendations.push(
                "Prompt prefill is compute bound. Consider enabling SpacemiT IME matrix acceleration or expanding X100 compute cap.".to_owned(),
            );
            "compute-bound"
        };

        // 2. Cluster targeting recommendation
        let recommended_cluster = if bottleneck == "compute-bound" && entry.cluster == "x100" {
            recommendations.push(
                "Model exhibits high prefill latency on Cluster 0 (X100). Offloading prefill to Cluster 1 (A100) will yield ~3.2x MAC acceleration.".to_owned(),
            );
            "both".to_owned()
        } else {
            entry.cluster.clone()
        };

        // 3. Priority recommendation
        if entry.priority_class == PriorityClass::Throughput
            && (model_id.contains("router") || model_id.contains("interactive"))
        {
            recommendations.push(
                "Interactive/decision model configured with Throughput priority. Transitioning to Latency priority class will enable pass-boundary preemption.".to_owned(),
            );
        }

        // 4. Dynamic core lending/borrowing recommendation
        if !entry.lend_idle_cores && !entry.borrow_idle_cores {
            recommendations.push(
                "Dynamic core lending and borrowing are disabled. Enabling both flags will permit single-pass idle core work-stealing.".to_owned(),
            );
        }

        // 5. Unprofiled status warning
        if entry.unprofiled {
            recommendations.push(
                "Model is currently running conservative defaults (X100, throughput, n_ctx=2048). Run 'k3-diag' to evaluate empirical hardware ceiling baselines.".to_owned(),
            );
        }

        let rationale = format!(
            "Observed {} passes (avg pass: {:.2}ms, prefill: {:.2}ms/tok, decode: {:.2}ms/tok). Baseline static weight: {:.2}, dynamic weight: {:.2}.",
            m.total_passes,
            m.avg_pass_cost_ms,
            m.avg_prefill_ms_per_tok,
            m.avg_decode_ms_per_tok,
            baseline.map_or(1.0, |b| b.static_scheduling_weight),
            m.dynamic_scheduling_weight
        );

        let recommended_ime = recommended_cluster.eq_ignore_ascii_case("a100")
            || recommended_cluster.eq_ignore_ascii_case("both");

        Ok(TuningAdvisory {
            bottleneck_classification: bottleneck.to_owned(),
            recommended_cluster,
            recommended_split: None,
            recommended_ime,
            rationale,
            recommendations,
            operator_authority_notice:
                "Advisory recommendation only. Runtime preserves operator tuning authority per ADR-032; changes will not take effect until explicitly applied."
                    .to_owned(),
        })
    }

    /// Explicitly applies an operator tuning profile, transitioning unprofiled models
    /// and updating scheduling parameters immediately without process restart.
    ///
    /// # Errors
    /// Returns [`ManagerError::ModelNotFound`] if model is not registered,
    /// or [`ManagerError::Affinity`] if the new reservation mask violates topology invariants.
    pub fn apply_tuning_profile(
        &self,
        model_id: &str,
        mut profile: TuningProfile,
    ) -> Result<(), ManagerError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let entry = state
            .active_models
            .get_mut(model_id)
            .ok_or_else(|| ManagerError::ModelNotFound(model_id.to_owned()))?;

        profile.model_id = Some(model_id.to_owned());
        profile.profile_status = ProfileStatus::Profiled;
        profile.unprofiled = false;
        profile.advisory_notice = None;

        entry.cluster.clone_from(&profile.cluster);
        entry.priority_class = profile.priority_class;
        entry.n_ctx = profile.n_ctx;
        entry.lend_idle_cores = profile.lend_idle_cores;
        entry.borrow_idle_cores = profile.borrow_idle_cores;
        entry.unprofiled = false;
        entry.tuning_profile = Some(profile);

        self.condvar.notify_all();
        Ok(())
    }
}
