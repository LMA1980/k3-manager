//! Patient admission queuing & SSE keep-alive streaming (US-13, Features 29, 30, 31, ADR-015).
//!
//! Provides per-model FIFO admission queues (default depth 64), wait timeouts disabled by default,
//! zero usage/token quotas, periodic SSE keep-alives (`: keep-alive\n\n`), advancing queue-position
//! updates (`: queue-position <pos>\n\n` and `: position=<pos>\n\n`), client disconnect pruning,
//! and graceful saturation rejection (`HTTP 429/529` + `Retry-After: <sec>`).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use k3_core::error::K3Error;

/// Default maximum queue capacity per model (ADR-015).
pub const DEFAULT_MAX_QUEUE_DEPTH: usize = 64;
/// Default interval between SSE keep-alive frames (1s).
pub const DEFAULT_KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(1);
/// Default minimum retry-after interval in seconds returned upon saturation.
pub const DEFAULT_MIN_RETRY_AFTER_SECS: u32 = 5;
/// Default initial assumed pass execution latency in ms if unmeasured (25.0ms).
pub const DEFAULT_ESTIMATED_PASS_LATENCY_MS: f64 = 25.0;
/// Default estimated forward passes required per request turn (16).
pub const DEFAULT_PASSES_PER_REQUEST: u32 = 16;

/// Configuration parameters governing per-model admission queue behavior (US-13, ADR-015).
#[derive(Debug, Clone, PartialEq)]
pub struct AdmissionQueueConfig {
    /// Maximum allowed queued requests per model (default 64).
    pub max_depth: usize,
    /// Periodic SSE heartbeat interval for streaming connections (default 1s).
    pub keep_alive_interval: Duration,
    /// Optional queue wait timeout (disabled / `None` by default per US-13).
    pub wait_timeout: Option<Duration>,
    /// Default retry-after interval in seconds returned upon saturation (default 5s).
    pub default_retry_after_secs: u32,
    /// Initial assumed pass execution latency in ms if unmeasured (default 25.0ms).
    pub estimated_pass_latency_ms: f64,
    /// Estimated forward passes required per turn (default 16).
    pub estimated_passes_per_request: u32,
}

impl Default for AdmissionQueueConfig {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_MAX_QUEUE_DEPTH,
            keep_alive_interval: DEFAULT_KEEP_ALIVE_INTERVAL,
            wait_timeout: None,
            default_retry_after_secs: DEFAULT_MIN_RETRY_AFTER_SECS,
            estimated_pass_latency_ms: DEFAULT_ESTIMATED_PASS_LATENCY_MS,
            estimated_passes_per_request: DEFAULT_PASSES_PER_REQUEST,
        }
    }
}

/// Server-Sent Events (SSE) keep-alive and queue progression frames (US-13, ADR-015).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueSseFrame {
    /// Generic SSE keep-alive comment: `: keep-alive\n\n`.
    KeepAlive,
    /// Advancing queue position frame: `: queue-position <pos>\n\n`.
    QueuePosition(u32),
}

impl QueueSseFrame {
    /// Formats this frame as an SSE comment string.
    #[must_use]
    pub fn format_frame(&self) -> String {
        match self {
            Self::KeepAlive => ": keep-alive\n\n".to_owned(),
            Self::QueuePosition(pos) => format!(": queue-position {pos}\n\n"),
        }
    }

    /// Formats legacy position comment string: `: position=<pos>\n\n`.
    #[must_use]
    pub fn format_position_comment(pos: u32) -> String {
        format!(": position={pos}\n\n")
    }

    /// Formats combined keep-alive and position comment string.
    #[must_use]
    pub fn format_combined_keep_alive(pos: u32) -> String {
        format!(": keep-alive position={pos}\n\n")
    }
}

/// Execution permit granted to a dequeued request allowing single-prompt forward-pass dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuePermit {
    /// Request identifier matching the admitted request.
    pub request_id: String,
    /// Logical model identifier.
    pub model_id: String,
    /// Elapsed duration spent waiting in the queue.
    pub queue_wait_duration: Duration,
}

/// Internal tracked representation of an active client request waiting in the admission queue.
#[derive(Debug)]
pub struct QueuedRequest {
    /// Unique identifier for this request.
    pub request_id: String,
    /// Model identifier requested.
    pub model_id: String,
    /// Wall-clock instant when the request entered the queue.
    pub arrived_at: Instant,
    /// Whether the client requested streaming (SSE).
    pub is_streaming: bool,
    /// Optional client-specified request timeout.
    pub client_timeout: Option<Duration>,
    /// Cancellation flag set when client disconnects or abandons.
    pub cancel_token: Arc<AtomicBool>,
    /// Optional permit sender channel.
    pub permit_sender: Option<tokio::sync::oneshot::Sender<QueuePermit>>,
    /// Optional SSE comment sender channel.
    pub sse_sender: Option<tokio::sync::mpsc::Sender<String>>,
}

/// Queue operational metrics tracked per model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueueMetrics {
    /// Total requests successfully admitted to the queue.
    pub total_admitted: u64,
    /// Total requests dequeued and granted execution permits.
    pub total_dispatched: u64,
    /// Total requests rejected due to saturation or timeout.
    pub total_rejected: u64,
    /// Total requests cancelled while queued (e.g. client disconnect).
    pub total_cancelled: u64,
    /// Total requests timed out waiting in queue.
    pub total_timed_out: u64,
}

/// Thread-safe per-model FIFO admission queue enforcing single-prompt exclusivity and depth bounds.
#[derive(Debug)]
pub struct ModelAdmissionQueue {
    model_id: String,
    config: AdmissionQueueConfig,
    queue: VecDeque<QueuedRequest>,
    is_executing: bool,
    metrics: QueueMetrics,
}

impl ModelAdmissionQueue {
    /// Creates a new per-model admission queue.
    #[must_use]
    pub fn new(model_id: impl Into<String>, config: AdmissionQueueConfig) -> Self {
        Self {
            model_id: model_id.into(),
            config,
            queue: VecDeque::new(),
            is_executing: false,
            metrics: QueueMetrics::default(),
        }
    }

    /// Current number of waiting requests in the queue (usize per AGENTS.md §9).
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` if no requests are waiting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Configured maximum depth ceiling.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.config.max_depth
    }

    /// Whether the model is currently executing a forward pass.
    #[must_use]
    pub fn is_executing(&self) -> bool {
        self.is_executing
    }

    /// Sets the executing state of the model.
    pub fn set_executing(&mut self, executing: bool) {
        self.is_executing = executing;
    }

    /// Returns a copy of the queue metrics.
    #[must_use]
    pub fn metrics(&self) -> QueueMetrics {
        self.metrics.clone()
    }

    /// Dynamic wait-time estimation in milliseconds:
    /// `wait_ms = (current_depth + is_executing) * (avg_pass_latency * passes_per_request)`
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimate_wait_ms(&self, avg_pass_latency_ms: Option<f64>) -> f64 {
        let pass_lat = avg_pass_latency_ms
            .filter(|&l| l.is_finite() && l > 0.0)
            .unwrap_or(self.config.estimated_pass_latency_ms);
        let request_lat = pass_lat * f64::from(self.config.estimated_passes_per_request);
        let pending_slots = self.queue.len() + usize::from(self.is_executing);
        (pending_slots as f64) * request_lat
    }

    /// Calculates recommended `Retry-After` seconds (u32 per AGENTS.md §9).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[must_use]
    pub fn calculate_retry_after_secs(&self, avg_pass_latency_ms: Option<f64>) -> u32 {
        let wait_ms = self.estimate_wait_ms(avg_pass_latency_ms);
        let secs = (wait_ms / 1000.0).ceil();
        if secs < 1.0 {
            self.config.default_retry_after_secs
        } else {
            u32::try_from(secs as u64).unwrap_or(self.config.default_retry_after_secs)
        }
    }

    /// Admits an incoming request to the queue.
    ///
    /// # Errors
    /// Returns [`K3Error::QueueSaturated`] if:
    /// 1. `queue.len() >= max_depth` (or `max_depth == 0`)
    /// 2. `client_timeout` is specified and `estimated_wait > client_timeout`
    #[allow(
        clippy::too_many_arguments,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn admit(
        &mut self,
        request_id: String,
        is_streaming: bool,
        client_timeout: Option<Duration>,
        avg_pass_latency_ms: Option<f64>,
        cancel_token: Arc<AtomicBool>,
        permit_sender: Option<tokio::sync::oneshot::Sender<QueuePermit>>,
        sse_sender: Option<tokio::sync::mpsc::Sender<String>>,
    ) -> Result<u32, K3Error> {
        // 1. Queue capacity saturation check
        if self.config.max_depth == 0 || self.queue.len() >= self.config.max_depth {
            self.metrics.total_rejected += 1;
            let retry_after_secs = self.calculate_retry_after_secs(avg_pass_latency_ms);
            return Err(K3Error::QueueSaturated {
                model_id: self.model_id.clone(),
                current_depth: self.queue.len(),
                max_depth: self.config.max_depth,
                retry_after_secs,
            });
        }

        // 2. Dynamic wait-time vs client timeout check
        if let Some(timeout) = client_timeout {
            let est_wait_ms = self.estimate_wait_ms(avg_pass_latency_ms);
            let est_wait = Duration::from_millis(est_wait_ms as u64);
            if est_wait > timeout {
                self.metrics.total_rejected += 1;
                let retry_after_secs = self.calculate_retry_after_secs(avg_pass_latency_ms);
                return Err(K3Error::QueueSaturated {
                    model_id: self.model_id.clone(),
                    current_depth: self.queue.len(),
                    max_depth: self.config.max_depth,
                    retry_after_secs,
                });
            }
        }

        // 3. FIFO Enqueue (1-indexed position)
        let position = u32::try_from(self.queue.len() + 1).unwrap_or(u32::MAX);
        let req = QueuedRequest {
            request_id,
            model_id: self.model_id.clone(),
            arrived_at: Instant::now(),
            is_streaming,
            client_timeout,
            cancel_token,
            permit_sender,
            sse_sender,
        };
        self.queue.push_back(req);
        self.metrics.total_admitted += 1;

        Ok(position)
    }

    /// Convenience admission method without channels.
    ///
    /// # Errors
    /// Returns [`K3Error::QueueSaturated`] if the queue is saturated.
    pub fn admit_simple(&mut self, request_id: String, is_streaming: bool) -> Result<u32, K3Error> {
        self.admit(
            request_id,
            is_streaming,
            None,
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            None,
        )
    }

    /// Cancels a queued request on client socket disconnect / drop.
    /// Prunes the entry and immediately notifies remaining requests of their advanced positions.
    pub fn cancel(&mut self, request_id: &str) -> bool {
        let before_len = self.queue.len();
        self.queue.retain(|req| req.request_id != request_id);
        let removed = self.queue.len() < before_len;
        if removed {
            self.metrics.total_cancelled += 1;
            self.broadcast_advancing_positions();
        }
        removed
    }

    /// Dispatches the next request in FIFO order when execution finishes.
    pub fn pop_next(&mut self) -> Option<QueuePermit> {
        while let Some(req) = self.queue.pop_front() {
            if req.cancel_token.load(Ordering::Relaxed) {
                self.metrics.total_cancelled += 1;
                continue;
            }
            self.is_executing = true;
            self.metrics.total_dispatched += 1;
            let wait_dur = req.arrived_at.elapsed();
            let permit = QueuePermit {
                request_id: req.request_id,
                model_id: req.model_id,
                queue_wait_duration: wait_dur,
            };
            if let Some(tx) = req.permit_sender {
                let _ = tx.send(permit.clone());
            }
            self.broadcast_advancing_positions();
            return Some(permit);
        }
        self.is_executing = false;
        None
    }

    /// Marks the current execution pass completed.
    pub fn mark_execution_completed(&mut self) {
        if self.is_executing {
            self.is_executing = false;
        } else {
            self.queue.pop_front();
        }
    }

    /// Broadcasts advancing queue positions to all queued streaming requests.
    pub fn broadcast_advancing_positions(&self) {
        for (idx, req) in self.queue.iter().enumerate() {
            if req.is_streaming {
                let pos = u32::try_from(idx + 1).unwrap_or(u32::MAX);
                if let Some(sse_tx) = &req.sse_sender {
                    let frame = QueueSseFrame::QueuePosition(pos).format_frame();
                    let _ = sse_tx.try_send(frame);
                    let legacy_frame = QueueSseFrame::format_position_comment(pos);
                    let _ = sse_tx.try_send(legacy_frame);
                }
            }
        }
    }

    /// Emits periodic keep-alive heartbeats to all queued streaming requests.
    pub fn emit_keep_alives(&self) {
        for req in &self.queue {
            if req.is_streaming {
                if let Some(sse_tx) = &req.sse_sender {
                    let frame = QueueSseFrame::KeepAlive.format_frame();
                    let _ = sse_tx.try_send(frame);
                }
            }
        }
    }
}

/// Coordinates admission queues across all registered model placements (US-13, ADR-015).
#[derive(Debug, Default)]
pub struct AdmissionQueueManager {
    queues: Mutex<HashMap<String, ModelAdmissionQueue>>,
    default_config: AdmissionQueueConfig,
}

impl AdmissionQueueManager {
    /// Creates a new manager with default per-model configuration.
    #[must_use]
    pub fn new(default_config: AdmissionQueueConfig) -> Self {
        Self {
            queues: Mutex::new(HashMap::new()),
            default_config,
        }
    }

    /// Registers an admission queue for a model.
    pub fn register_model(&self, model_id: &str, custom_config: Option<AdmissionQueueConfig>) {
        let mut q_map = self.queues.lock().unwrap_or_else(PoisonError::into_inner);
        let cfg = custom_config.unwrap_or_else(|| self.default_config.clone());
        q_map.insert(model_id.to_owned(), ModelAdmissionQueue::new(model_id, cfg));
    }

    /// Unregisters an admission queue for a model.
    pub fn unregister_model(&self, model_id: &str) -> bool {
        let mut q_map = self.queues.lock().unwrap_or_else(PoisonError::into_inner);
        q_map.remove(model_id).is_some()
    }

    /// Returns the queue depth for a specific model (usize per AGENTS.md §9).
    #[must_use]
    pub fn queue_depth(&self, model_id: &str) -> usize {
        let q_map = self.queues.lock().unwrap_or_else(PoisonError::into_inner);
        q_map.get(model_id).map_or(0, ModelAdmissionQueue::len)
    }

    /// Returns the total aggregate queue depth across all registered models.
    #[must_use]
    pub fn total_queue_depth(&self) -> usize {
        let q_map = self.queues.lock().unwrap_or_else(PoisonError::into_inner);
        q_map.values().map(ModelAdmissionQueue::len).sum()
    }

    /// Admits a request to the target model queue.
    ///
    /// # Errors
    /// Returns [`K3Error::QueueSaturated`] if the target queue is full,
    /// or registers a new default queue if not previously registered.
    #[allow(clippy::too_many_arguments)]
    pub fn admit_to_model(
        &self,
        model_id: &str,
        request_id: String,
        is_streaming: bool,
        client_timeout: Option<Duration>,
        avg_pass_latency_ms: Option<f64>,
        cancel_token: Arc<AtomicBool>,
        permit_sender: Option<tokio::sync::oneshot::Sender<QueuePermit>>,
        sse_sender: Option<tokio::sync::mpsc::Sender<String>>,
    ) -> Result<u32, K3Error> {
        let mut q_map = self.queues.lock().unwrap_or_else(PoisonError::into_inner);
        let queue = q_map
            .entry(model_id.to_owned())
            .or_insert_with(|| ModelAdmissionQueue::new(model_id, self.default_config.clone()));
        queue.admit(
            request_id,
            is_streaming,
            client_timeout,
            avg_pass_latency_ms,
            cancel_token,
            permit_sender,
            sse_sender,
        )
    }

    /// Dispatches the next request for the target model.
    pub fn pop_next_for_model(&self, model_id: &str) -> Option<QueuePermit> {
        let mut q_map = self.queues.lock().unwrap_or_else(PoisonError::into_inner);
        q_map
            .get_mut(model_id)
            .and_then(ModelAdmissionQueue::pop_next)
    }

    /// Cancels a request in the target model queue.
    pub fn cancel_in_model(&self, model_id: &str, request_id: &str) -> bool {
        let mut q_map = self.queues.lock().unwrap_or_else(PoisonError::into_inner);
        q_map
            .get_mut(model_id)
            .is_some_and(|q| q.cancel(request_id))
    }

    /// Marks the execution for a model completed.
    pub fn mark_execution_completed(&self, model_id: &str) {
        let mut q_map = self.queues.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(q) = q_map.get_mut(model_id) {
            q.mark_execution_completed();
        }
    }

    /// Emits keep-alive frames across all model queues.
    pub fn emit_all_keep_alives(&self) {
        let q_map = self.queues.lock().unwrap_or_else(PoisonError::into_inner);
        for q in q_map.values() {
            q.emit_keep_alives();
        }
    }
}
