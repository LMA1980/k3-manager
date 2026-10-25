//! `k3-manager`: Control plane and IPC daemon for the `SpacemiT` K3 AI platform.
//!
//! Supervises isolated `k3-worker` execution domains (ADR-018, ADR-041),
//! communicates with `k3-kv` context memory holder (ADR-042),
//! and orchestrates zero-copy forward pass dispatch and crash recovery (US-05).

pub mod capacity;
pub mod client;
pub mod dispatch;
pub mod error;
pub mod supervisor;

pub use capacity::{
    A100_CLUSTER_MASK, A100_DEFAULT_MAX_COMPUTE_THREADS, ActiveControlRotator,
    ControlPlaneCapacityConfig, ControlPlaneScheduler, DEFAULT_A100_MAX_COMPUTE_CORES,
    DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_MAX_A100_COMPUTE_CORES, DEFAULT_MAX_X100_COMPUTE_CORES,
    DEFAULT_SPARE_RECOVERY_MASK, DEFAULT_X100_COMPUTE_CAP_MASK, DEFAULT_X100_MAX_COMPUTE_CORES,
    ResolvedControlCapacity, ResolvedControlCapacityExt, SOC_CLUSTER_MASK, X100_CLUSTER_MASK,
    X100_DEFAULT_MAX_COMPUTE_THREADS,
};
pub use client::KvClient;
pub use dispatch::{
    ForwardPassRequest, ForwardPassResult, dispatch_forward_pass, wrap_input_tokens_as_arrow,
    wrap_logits_as_float32_array,
};
pub use error::ManagerError;
pub use supervisor::{ManagedWorker, WorkerPlacement, WorkerSupervisor, find_worker_binary};
