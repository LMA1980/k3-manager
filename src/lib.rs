//! `k3-manager`: Control plane and IPC daemon for the `SpacemiT` K3 AI platform.
//!
//! Supervises isolated `k3-worker` execution domains (ADR-018, ADR-041),
//! communicates with `k3-kv` context memory holder (ADR-042),
//! and orchestrates zero-copy forward pass dispatch and crash recovery (US-05).

pub mod client;
pub mod dispatch;
pub mod error;
pub mod supervisor;

pub use client::KvClient;
pub use dispatch::{
    dispatch_forward_pass, wrap_input_tokens_as_arrow, wrap_logits_as_float32_array,
    ForwardPassRequest, ForwardPassResult,
};
pub use error::ManagerError;
pub use supervisor::{find_worker_binary, ManagedWorker, WorkerPlacement, WorkerSupervisor};

