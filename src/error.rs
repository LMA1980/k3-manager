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
    /// Shared memory arena or `SCM_RIGHTS` transfer error.
    Shm(IpcError),
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
            Self::Shm(e) => write!(f, "[k3-manager shm error] {e}"),
            Self::TensorBounds(e) => write!(f, "[k3-manager tensor bounds error] {e}"),
            Self::Protocol(msg) => write!(f, "[k3-manager protocol error] {msg}"),
            Self::Status { status, message } => {
                write!(f, "[k3-manager status {status:?}] {message}")
            }
            Self::ModelNotFound(id) => write!(f, "[k3-manager model not found] '{id}'"),
            Self::Io(e) => write!(f, "[k3-manager io error] {e}"),
        }
    }
}

impl std::error::Error for ManagerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Shm(e) => Some(e),
            Self::TensorBounds(e) => Some(e),
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
        Self::TensorBounds(err)
    }
}
