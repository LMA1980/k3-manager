//! Standardized error types for `k3-manager`.

use std::fmt;
use std::io;

use k3_core::K3Error;
use k3_core::ipc::protocol::StatusCode;
use k3_core::ipc::shm::IpcError;

/// Error type encompassing control plane, KV client, worker supervisor, and dispatch failures.
#[derive(Debug)]
pub enum ManagerError {
    /// KV holder communication or protocol error.
    Kv(String),
    /// Failure spawning worker OS process.
    WorkerSpawn(String),
    /// Worker IPC communication error.
    WorkerIpc(String),
    /// Compute worker process terminated unexpectedly or crashed.
    WorkerCrashed {
        /// Logical model identifier.
        model_id: String,
        /// Process exit code if available.
        exit_code: Option<i32>,
        /// Terminating signal if killed.
        signal: Option<i32>,
    },
    /// Compute worker or control task timed out or failed to respond within deadline bounds.
    WorkerUnresponsive {
        /// Logical model identifier or task name.
        model_id: String,
        /// OS process ID if known.
        pid: Option<u32>,
        /// Elapsed time in milliseconds when timeout occurred.
        elapsed_ms: u64,
    },
    /// Shared memory arena or `SCM_RIGHTS` transfer error.
    Shm(IpcError),
    /// Core affinity configuration, topology validation, or reservation error.
    Affinity(K3Error),
    /// Tensor bounds or layout error.
    TensorBounds(K3Error),
    /// Protocol framing or type error.
    Protocol(String),
    /// Non-success status returned from peer.
    Status {
        /// Protocol status code.
        status: StatusCode,
        /// Diagnostic message.
        message: String,
    },
    /// Model identifier was not found in registry.
    ModelNotFound(String),
    /// Active model concurrency limit exceeded (ADR-043, US-09 Scenario 3).
    ConcurrencyLimitExceeded {
        /// Current number of active loaded models.
        active: usize,
        /// Maximum allowed active models.
        max: usize,
    },
    /// Client-supplied priority override rejected (ADR-014, US-10 Scenario 3).
    PriorityOverrideRejected {
        /// Logical model identifier.
        model_id: String,
    },
    /// Resumption requested before prefill was fully committed in shared memory (US-16).
    IncompletePrefillResumptionNotSupported {
        /// Committed token count in arena metadata.
        committed_tokens: u32,
        /// Full prompt length.
        prompt_len: u32,
    },
    /// Automatic decision routing unavailable (US-12).
    RoutingUnavailable {
        /// Diagnostic explanation of the routing failure.
        reason: String,
    },
    /// Admission queue saturated or estimated wait exceeded (US-13).
    QueueSaturated(K3Error),
    /// Operational metadata or session persistence failure (US-17).
    Persistence(crate::persistence::PersistenceError),
    /// General I/O failure.
    Io(io::Error),
}

impl fmt::Display for ManagerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kv(msg) => write!(f, "[k3-manager kv error] {msg}"),
            Self::WorkerSpawn(msg) => write!(f, "[k3-manager worker spawn error] {msg}"),
            Self::WorkerIpc(msg) => write!(f, "[k3-manager worker ipc error] {msg}"),
            Self::WorkerCrashed {
                model_id,
                exit_code,
                signal,
            } => {
                write!(
                    f,
                    "[k3-manager worker crash] model '{model_id}' terminated abnormally (exit_code: {exit_code:?}, signal: {signal:?})"
                )
            }
            Self::WorkerUnresponsive {
                model_id,
                pid,
                elapsed_ms,
            } => {
                write!(
                    f,
                    "[k3-manager worker unresponsive] model '{model_id}' (pid: {pid:?}) timed out after {elapsed_ms}ms"
                )
            }
            Self::Shm(e) => write!(f, "[k3-manager shm error] {e}"),
            Self::Affinity(e) => write!(f, "[k3-manager affinity error] {e}"),
            Self::TensorBounds(e) => write!(f, "[k3-manager tensor bounds error] {e}"),
            Self::Protocol(msg) => write!(f, "[k3-manager protocol error] {msg}"),
            Self::Status { status, message } => {
                write!(f, "[k3-manager status {status:?}] {message}")
            }
            Self::ModelNotFound(id) => write!(f, "[k3-manager model not found] '{id}'"),
            Self::ConcurrencyLimitExceeded { active, max } => {
                write!(
                    f,
                    "[k3-manager concurrency limit exceeded] active models ({active}) reached maximum limit ({max})"
                )
            }
            Self::PriorityOverrideRejected { model_id } => {
                write!(
                    f,
                    "[k3-manager priority override rejected] client priority override rejected for model '{model_id}'; operator tuning profile is authoritative"
                )
            }
            Self::IncompletePrefillResumptionNotSupported {
                committed_tokens,
                prompt_len,
            } => {
                write!(
                    f,
                    "[k3-manager resumption error] incomplete prefill resumption not supported (committed: {committed_tokens}, prompt_len: {prompt_len})"
                )
            }
            Self::RoutingUnavailable { reason } => {
                write!(
                    f,
                    "[k3-manager routing unavailable] routing unavailable: {reason}"
                )
            }
            Self::QueueSaturated(e) => write!(f, "[k3-manager queue saturated] {e}"),
            Self::Persistence(e) => write!(f, "[k3-manager persistence error] {e}"),
            Self::Io(e) => write!(f, "[k3-manager io error] {e}"),
        }
    }
}

impl std::error::Error for ManagerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Shm(e) => Some(e),
            Self::Affinity(e) | Self::TensorBounds(e) | Self::QueueSaturated(e) => Some(e),
            Self::Persistence(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for ManagerError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<IpcError> for ManagerError {
    fn from(err: IpcError) -> Self {
        Self::Shm(err)
    }
}

impl From<K3Error> for ManagerError {
    fn from(err: K3Error) -> Self {
        match err {
            e @ K3Error::AffinityError { .. } => Self::Affinity(e),
            K3Error::RoutingUnavailable { reason } => Self::RoutingUnavailable { reason },
            e @ K3Error::QueueSaturated { .. } => Self::QueueSaturated(e),
            other => Self::TensorBounds(other),
        }
    }
}

impl From<crate::persistence::PersistenceError> for ManagerError {
    fn from(err: crate::persistence::PersistenceError) -> Self {
        Self::Persistence(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_from_affinity_error_discriminates_to_affinity_variant() {
        let k3_err = K3Error::AffinityError {
            requested_mask: 0x00FF,
            requested_threads: 8,
            reason: "control cores cannot exceed available online cores".to_string(),
        };

        let mgr_err: ManagerError = k3_err.clone().into();
        match mgr_err {
            ManagerError::Affinity(ref inner) => {
                assert_eq!(inner, &k3_err);
            }
            other => panic!("expected ManagerError::Affinity, got {other:?}"),
        }

        let display_str = format!("{mgr_err}");
        assert!(display_str.contains("[k3-manager affinity error]"));
        assert!(display_str.contains("[affinity_error]"));
        assert!(display_str.contains("0x00FF"));
        assert!(display_str.contains("threads: 8"));

        let source = mgr_err.source().expect("source should be Some");
        assert_eq!(format!("{source}"), format!("{k3_err}"));
    }

    #[test]
    fn test_from_tensor_bounds_error_discriminates_to_tensor_bounds_variant() {
        let k3_err = K3Error::TensorBoundsError {
            tensor_name: "weight".to_string(),
            offset: 0,
            byte_len: 128,
            buffer_len: 64,
            reason: "exceeds buffer".to_string(),
        };

        let mgr_err: ManagerError = k3_err.clone().into();
        match mgr_err {
            ManagerError::TensorBounds(ref inner) => {
                assert_eq!(inner, &k3_err);
            }
            other => panic!("expected ManagerError::TensorBounds, got {other:?}"),
        }

        let display_str = format!("{mgr_err}");
        assert!(display_str.contains("[k3-manager tensor bounds error]"));
        assert!(display_str.contains("[tensor_bounds_error]"));

        let source = mgr_err.source().expect("source should be Some");
        assert_eq!(format!("{source}"), format!("{k3_err}"));
    }

    #[test]
    fn test_source_implementation_for_other_variants() {
        let io_err = ManagerError::Io(io::Error::new(io::ErrorKind::NotFound, "file not found"));
        assert!(io_err.source().is_some());

        let proto_err = ManagerError::Protocol("invalid frame".to_string());
        assert!(proto_err.source().is_none());
    }

    #[test]
    fn test_worker_unresponsive_error_display() {
        let err = ManagerError::WorkerUnresponsive {
            model_id: "qwen2.5-7b".to_string(),
            pid: Some(1234),
            elapsed_ms: 250,
        };
        let disp = format!("{err}");
        assert!(disp.contains("[k3-manager worker unresponsive]"));
        assert!(disp.contains("qwen2.5-7b"));
        assert!(disp.contains("1234"));
        assert!(disp.contains("250ms"));
    }

    #[test]
    fn test_concurrency_limit_exceeded_display() {
        let err = ManagerError::ConcurrencyLimitExceeded { active: 3, max: 3 };
        let disp = format!("{err}");
        assert!(disp.contains("[k3-manager concurrency limit exceeded]"));
        assert!(disp.contains("active models (3) reached maximum limit (3)"));
    }

    #[test]
    fn test_priority_override_rejected_display() {
        let err = ManagerError::PriorityOverrideRejected {
            model_id: "batch-gen".to_string(),
        };
        let disp = format!("{err}");
        assert!(disp.contains("[k3-manager priority override rejected]"));
        assert!(disp.contains("batch-gen"));
    }
}
