//! Content-Addressed Store, Lifecycle Management, and Safe Garbage Collection (US-18, US-20, US-21, US-22).

pub mod cas;
pub mod derived;
pub mod filter;
pub mod gc;
pub mod sha256;

pub use cas::{
    ContentAddressedStore, DEFAULT_STORE_ROOT, ENV_STORE_ROOT, MAX_PATH_COMPONENT_LEN,
    ModelManifest, ResolvedAlias, parse_alias_id, validate_path_component, validate_sha256_digest,
};
pub use derived::{
    DerivedBlobManager, DerivedResolution, RetentionPolicy, StaleRecipeOutcome,
    check_derived_memory_admission,
};
pub use filter::{
    AlignedPromptEnvironment, CandidateSuitability, CapabilityFilter, RecipeDescriptor,
};
pub use gc::{
    BlobUsageInfo, GcReport, HighWaterMarkWarning, MockProcMapsScanner, ProcessMappingInfo,
    ProcessMapsScanner, RealProcMapsScanner, StorageGarbageCollector,
};
pub use sha256::{Sha256, hex_encode};
