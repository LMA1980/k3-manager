//! Isolated Model Ingestion & Source Adapters (US-19, ADR-019, ADR-040, ADR-041).
//!
//! Exposes external repository lifecycle methods (`auth`, `search`, `list`, `pull`)
//! with credential isolation, streaming SHA-256 verification, `zerocopy` header validation,
//! and atomic promotion into content-addressed storage.

pub mod adapter;
pub mod provider;
pub mod staging;

pub use adapter::{
    AuthStatus, IngestionError, ModelFormat, ModelRevision, ModelVersionListing,
    PullArtifactSummary, PullProgress, PullRequest, RemoteModelSummary, SearchFilter,
    SourceAdapter, SourceCredentials, StagingDownload,
};
pub use provider::{HuggingFaceAdapter, LocalFileAdapter, OllamaAdapter, SourceAdapterRegistry};
pub use staging::{GgufHeaderPrefix, SafeTensorsHeaderPrefix, StagingPipeline};
