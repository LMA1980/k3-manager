//! Production session retention, catchup stream buffering, and reconnection registry (US-16, ADR-026, `features/kv-cache.md`).
//!
//! Provides `SessionState` and thread-safe `SessionRegistry` for managing client generation streams,
//! tracking 30-second disconnect retention windows, buffering token generation history for client catchup,
//! and distributing live stream tokens to multi-client subscribers via broadcast channels.

use std::collections::HashMap;
use std::fmt;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use tokio::sync::broadcast;

/// Default client reconnection retention window: 30 seconds (US-16).
pub const DEFAULT_RETENTION_WINDOW: Duration = Duration::from_secs(30);

/// Channel capacity for multi-subscriber broadcast streams.
pub const DEFAULT_BROADCAST_CAPACITY: usize = 4096;

/// Error conditions encountered during session management and reconnection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// Requested session ID was not found in the registry.
    SessionNotFound(String),
    /// Reconnection attempt occurred after the retention window expired.
    RetentionWindowExpired {
        /// Logical session identifier.
        session_id: String,
        /// Elapsed time in milliseconds since client disconnection.
        elapsed_ms: u64,
        /// Configured retention window in milliseconds.
        retention_window_ms: u64,
    },
    /// Internal synchronization lock poisoned.
    LockError(String),
    /// Broadcast stream distribution error.
    BroadcastError(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionNotFound(id) => write!(f, "session '{id}' not found in registry"),
            Self::RetentionWindowExpired {
                session_id,
                elapsed_ms,
                retention_window_ms,
            } => write!(
                f,
                "session '{session_id}' retention window expired ({elapsed_ms}ms elapsed > {retention_window_ms}ms window)"
            ),
            Self::LockError(msg) => write!(f, "session registry lock error: {msg}"),
            Self::BroadcastError(msg) => write!(f, "session broadcast error: {msg}"),
        }
    }
}

impl std::error::Error for SessionError {}

/// Tracked state and streaming buffer for an active or disconnected generation session (US-16).
#[derive(Debug)]
pub struct SessionState {
    /// Unique session identifier.
    pub session_id: String,
    /// Associated project identifier for isolation.
    pub project_id: String,
    /// Model identifier running this generation session.
    pub model_id: String,
    /// Bounded in-memory buffer of all tokens generated for this session turn.
    pub tokens: Vec<u32>,
    /// Timestamp of most recent token generation or client activity.
    pub last_activity: Instant,
    /// Configured retention window before expired sessions are pruned (default 30s).
    pub retention_window: Duration,
    /// Disconnection timestamp, or `None` if client is actively connected.
    pub disconnected_at: Option<Instant>,
    /// Broadcast channel sender for live lockstep streaming to multiple subscribers.
    sender: broadcast::Sender<u32>,
}

impl SessionState {
    /// Creates a new `SessionState` with default 30-second retention window.
    #[must_use]
    pub fn new(session_id: &str, project_id: &str, model_id: &str) -> Self {
        Self::with_retention_window(session_id, project_id, model_id, DEFAULT_RETENTION_WINDOW)
    }

    /// Creates a new `SessionState` with an explicit retention window.
    #[must_use]
    pub fn with_retention_window(
        session_id: &str,
        project_id: &str,
        model_id: &str,
        retention_window: Duration,
    ) -> Self {
        let (sender, _) = broadcast::channel(DEFAULT_BROADCAST_CAPACITY);
        Self {
            session_id: session_id.to_owned(),
            project_id: project_id.to_owned(),
            model_id: model_id.to_owned(),
            tokens: Vec::new(),
            last_activity: Instant::now(),
            retention_window,
            disconnected_at: None,
            sender,
        }
    }

    /// Appends a newly generated token ID to the session catchup buffer
    /// and broadcasts it to all connected subscribers in lockstep.
    pub fn append_token(&mut self, token_id: u32) {
        self.tokens.push(token_id);
        self.last_activity = Instant::now();
        // Ignore SendError if there are currently no active receivers
        let _ = self.sender.send(token_id);
    }

    /// Marks the session as disconnected at `Instant::now()`.
    pub fn record_disconnect(&mut self) {
        self.disconnected_at = Some(Instant::now());
    }

    /// Marks the session as disconnected at a specified timestamp.
    pub fn record_disconnect_at(&mut self, at_time: Instant) {
        self.disconnected_at = Some(at_time);
    }

    /// Subscribes to the live token broadcast stream.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<u32> {
        self.sender.subscribe()
    }

    /// Returns `true` if the session's disconnect retention window has expired at `at_time`.
    #[must_use]
    pub fn is_expired_at(&self, at_time: Instant) -> bool {
        match self.disconnected_at {
            Some(disc) => at_time.saturating_duration_since(disc) > self.retention_window,
            None => false,
        }
    }

    /// Returns `true` if the session's disconnect retention window has expired now.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.is_expired_at(Instant::now())
    }

    /// Computes catchup tokens starting from `last_acked_token_idx + 1` up to the current end.
    ///
    /// If `last_acked_token_idx` is `None`, returns all accumulated tokens from index 0.
    #[must_use]
    pub fn catchup_tokens(&self, last_acked_token_idx: Option<usize>) -> Vec<u32> {
        match last_acked_token_idx {
            None => self.tokens.clone(),
            Some(idx) => {
                let start = idx.saturating_add(1);
                if start < self.tokens.len() {
                    self.tokens[start..].to_vec()
                } else {
                    Vec::new()
                }
            }
        }
    }
}

/// Thread-safe concurrent registry managing active and recoverable client generation sessions (US-16).
#[derive(Debug, Default)]
pub struct SessionRegistry {
    sessions: RwLock<HashMap<String, SessionState>>,
}

impl SessionRegistry {
    /// Creates a new, empty `SessionRegistry`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }

    /// Registers a new generation session with default 30-second retention window.
    ///
    /// # Errors
    /// Returns [`SessionError::LockError`] if lock acquisition fails.
    pub fn register_session(
        &self,
        session_id: &str,
        project_id: &str,
        model_id: &str,
    ) -> Result<(), SessionError> {
        self.register_session_with_retention(
            session_id,
            project_id,
            model_id,
            DEFAULT_RETENTION_WINDOW,
        )
    }

    /// Registers a new generation session with an explicit retention window.
    ///
    /// # Errors
    /// Returns [`SessionError::LockError`] if lock acquisition fails.
    pub fn register_session_with_retention(
        &self,
        session_id: &str,
        project_id: &str,
        model_id: &str,
        retention_window: Duration,
    ) -> Result<(), SessionError> {
        let mut guard = self
            .sessions
            .write()
            .map_err(|e| SessionError::LockError(e.to_string()))?;
        guard.insert(
            session_id.to_owned(),
            SessionState::with_retention_window(session_id, project_id, model_id, retention_window),
        );
        Ok(())
    }

    /// Appends a generated token to the specified session buffer and broadcasts to subscribers.
    ///
    /// # Errors
    /// Returns [`SessionError::SessionNotFound`] if the session ID is not registered,
    /// or [`SessionError::LockError`] if lock acquisition fails.
    pub fn append_token(&self, session_id: &str, token_id: u32) -> Result<(), SessionError> {
        let mut guard = self
            .sessions
            .write()
            .map_err(|e| SessionError::LockError(e.to_string()))?;
        let session = guard
            .get_mut(session_id)
            .ok_or_else(|| SessionError::SessionNotFound(session_id.to_owned()))?;
        session.append_token(token_id);
        Ok(())
    }

    /// Records client disconnection at `Instant::now()`, initiating the retention countdown.
    ///
    /// # Errors
    /// Returns [`SessionError::SessionNotFound`] if the session ID is not registered,
    /// or [`SessionError::LockError`] if lock acquisition fails.
    pub fn record_disconnect(&self, session_id: &str) -> Result<(), SessionError> {
        self.record_disconnect_at(session_id, Instant::now())
    }

    /// Records client disconnection at a specific timestamp.
    ///
    /// # Errors
    /// Returns [`SessionError::SessionNotFound`] if the session ID is not registered,
    /// or [`SessionError::LockError`] if lock acquisition fails.
    pub fn record_disconnect_at(
        &self,
        session_id: &str,
        at_time: Instant,
    ) -> Result<(), SessionError> {
        let mut guard = self
            .sessions
            .write()
            .map_err(|e| SessionError::LockError(e.to_string()))?;
        let session = guard
            .get_mut(session_id)
            .ok_or_else(|| SessionError::SessionNotFound(session_id.to_owned()))?;
        session.record_disconnect_at(at_time);
        Ok(())
    }

    /// Reconnects a client within the retention window, returning the catchup tokens
    /// generated since `last_acked_token_idx` and clearing the disconnect state.
    ///
    /// # Errors
    /// - Returns [`SessionError::SessionNotFound`] if the session was pruned or never existed.
    /// - Returns [`SessionError::RetentionWindowExpired`] if the elapsed disconnect time exceeds the retention window.
    /// - Returns [`SessionError::LockError`] if lock acquisition fails.
    pub fn reconnect_client(
        &self,
        session_id: &str,
        last_acked_token_idx: usize,
    ) -> Result<Vec<u32>, SessionError> {
        self.reconnect_client_opt(session_id, Some(last_acked_token_idx))
    }

    /// Reconnects a client with optional `last_acked_token_idx` (None returns all tokens).
    ///
    /// # Errors
    /// Same as `reconnect_client`.
    pub fn reconnect_client_opt(
        &self,
        session_id: &str,
        last_acked_token_idx: Option<usize>,
    ) -> Result<Vec<u32>, SessionError> {
        self.reconnect_client_at(session_id, last_acked_token_idx, Instant::now())
    }

    /// Reconnects a client at a specific evaluation timestamp (useful for deterministic tests).
    ///
    /// If the retention window has expired, the session is automatically pruned from memory.
    ///
    /// # Errors
    /// Same as `reconnect_client`.
    pub fn reconnect_client_at(
        &self,
        session_id: &str,
        last_acked_token_idx: Option<usize>,
        at_time: Instant,
    ) -> Result<Vec<u32>, SessionError> {
        let mut guard = self
            .sessions
            .write()
            .map_err(|e| SessionError::LockError(e.to_string()))?;

        let session = guard
            .get_mut(session_id)
            .ok_or_else(|| SessionError::SessionNotFound(session_id.to_owned()))?;

        if let Some(disc) = session.disconnected_at {
            let elapsed = at_time.saturating_duration_since(disc);
            if elapsed > session.retention_window {
                let retention_window = session.retention_window;
                // Auto-prune expired session from active sessions
                guard.remove(session_id);
                return Err(SessionError::RetentionWindowExpired {
                    session_id: session_id.to_owned(),
                    elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
                    retention_window_ms: u64::try_from(retention_window.as_millis())
                        .unwrap_or(u64::MAX),
                });
            }
        }

        // Reconnection successful: clear disconnect timestamp and update activity
        session.disconnected_at = None;
        session.last_activity = at_time;
        Ok(session.catchup_tokens(last_acked_token_idx))
    }

    /// Subscribes a client to live stream tokens for the given session.
    ///
    /// # Errors
    /// Returns [`SessionError::SessionNotFound`] if the session ID does not exist,
    /// or [`SessionError::LockError`] if lock acquisition fails.
    pub fn subscribe(&self, session_id: &str) -> Result<broadcast::Receiver<u32>, SessionError> {
        let guard = self
            .sessions
            .read()
            .map_err(|e| SessionError::LockError(e.to_string()))?;
        let session = guard
            .get(session_id)
            .ok_or_else(|| SessionError::SessionNotFound(session_id.to_owned()))?;
        Ok(session.subscribe())
    }

    /// Prunes all disconnected sessions whose retention window has expired as of `Instant::now()`.
    ///
    /// Returns the number of pruned sessions.
    pub fn prune_expired(&self) -> usize {
        self.prune_expired_at(Instant::now())
    }

    /// Prunes all disconnected sessions whose retention window has expired as of `at_time`.
    ///
    /// Returns the number of pruned sessions.
    pub fn prune_expired_at(&self, at_time: Instant) -> usize {
        if let Ok(mut guard) = self.sessions.write() {
            let initial_count = guard.len();
            guard.retain(|_, session| !session.is_expired_at(at_time));
            initial_count.saturating_sub(guard.len())
        } else {
            0
        }
    }

    /// Returns a copy of all accumulated tokens for a session.
    ///
    /// # Errors
    /// Returns [`SessionError::SessionNotFound`] if not found,
    /// or [`SessionError::LockError`] if lock acquisition fails.
    pub fn get_tokens(&self, session_id: &str) -> Result<Vec<u32>, SessionError> {
        let guard = self
            .sessions
            .read()
            .map_err(|e| SessionError::LockError(e.to_string()))?;
        let session = guard
            .get(session_id)
            .ok_or_else(|| SessionError::SessionNotFound(session_id.to_owned()))?;
        Ok(session.tokens.clone())
    }

    /// Returns `true` if the session ID is currently present in the registry.
    #[must_use]
    pub fn contains_session(&self, session_id: &str) -> bool {
        self.sessions
            .read()
            .is_ok_and(|guard| guard.contains_key(session_id))
    }

    /// Returns the total number of active sessions currently registered.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions.read().map_or(0, |guard| guard.len())
    }

    /// Removes a session from the registry, returning its state if present.
    pub fn remove_session(&self, session_id: &str) -> Option<SessionState> {
        self.sessions.write().ok()?.remove(session_id)
    }
}
