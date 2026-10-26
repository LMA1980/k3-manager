//! `k3-manager`: Control plane and IPC daemon for the `SpacemiT` K3 AI platform.
//!
//! Supervises isolated `k3-worker` execution domains (ADR-018, ADR-041),
//! communicates with `k3-kv` context memory holder (ADR-042),
//! and orchestrates zero-copy forward pass dispatch and crash recovery (US-05).

pub mod capacity;
pub mod client;
pub mod dispatch;
pub mod domain;
pub mod error;
pub mod prefix;
pub mod prompt;
pub mod runtime;
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
pub use domain::{
    CgroupFs, CgroupV2DomainController, DomainPlacementMode, IsolationDomainController,
    MockCgroupFs, MockOsAffinityBackend, OsAffinityBackend, PlacementOutcome, RealCgroupFs,
    RealOsAffinityBackend, format_cpuset_list, parse_cpuset_list, resolve_worker_compute_mask,
    validate_model_id,
};
pub use error::ManagerError;
pub use prefix::{
    BLOCK_SIZE_TOKENS, BlockTokens, PrefixCacheStats, PrefixMatchResult, PrefixRadixIndex,
};
pub use prompt::{
    ChatTemplate, DetokenizeStep, GgufTokenizer, IncrementalStopDetector, PreparedPrompt,
    PromptOrchestrator, PromptPreparationRequest, SimpleTokenizer, SpecialTokens,
    StreamingDetokenizer, TokenType, Tokenizer, evaluate_stop_token, format_chat_prompt,
    validate_and_tokenize,
};
pub use runtime::{AdminCommand, AdminResponse, ControlPlaneRuntime, WorkerHealthSnapshot};
pub use supervisor::{ManagedWorker, WorkerPlacement, WorkerSupervisor, find_worker_binary};
