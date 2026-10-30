//! Source Adapter Contract & Credential Isolation (US-19, ADR-019, ADR-041).
//!
//! Defines the standard lifecycle contract (`auth`, `search`, `list`, `pull`) for
//! interacting with external model repositories (Hugging Face, Ollama, private registries)
//! with credential isolation and fail-closed secret resolution.

#![allow(
    clippy::pedantic,
    clippy::doc_markdown,
    clippy::should_implement_trait,
    clippy::needless_pass_by_value,
    clippy::missing_errors_doc
)]

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use k3_core::K3Error;
use serde::{Deserialize, Serialize};

/// Supported model weight format types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFormat {
    /// GGUF binary format (v2/v3).
    Gguf,
    /// Hugging Face SafeTensors binary format.
    SafeTensors,
    /// Apache Arrow tensor batch format.
    ArrowTensor,
    /// Custom or proprietary format.
    Custom(String),
}

impl ModelFormat {
    /// Returns the canonical format name string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Gguf => "gguf",
            Self::SafeTensors => "safetensors",
            Self::ArrowTensor => "arrow",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Parses a model format from a string identifier.
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "gguf" => Self::Gguf,
            "safetensors" | "safetensor" => Self::SafeTensors,
            "arrow" | "arrowtensor" => Self::ArrowTensor,
            other => Self::Custom(other.to_owned()),
        }
    }
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Authentication credentials for external model repositories.
///
/// Follows the fail-closed secret resolution invariant:
/// file paths are never used as literal token values if reading fails.
#[derive(Clone, PartialEq, Eq)]
pub enum SourceCredentials {
    /// Anonymous unauthenticated access.
    Anonymous,
    /// Explicit API token in memory.
    Token(String),
    /// HTTP Bearer authentication token.
    BearerToken(String),
    /// Path to a protected credential file (e.g., mode 0600 `/etc/k3/credentials/<source>.env`).
    SecretFile(PathBuf),
    /// HTTP Basic username and password/token pair.
    BasicAuth {
        /// Repository username.
        username: String,
        /// Authentication secret or personal access token.
        password_or_token: String,
    },
}

impl SourceCredentials {
    /// Resolves the raw secret string, failing closed if file reading fails.
    ///
    /// # Invariant
    /// If a `SecretFile` does not exist, cannot be read, or is empty, this method returns
    /// [`IngestionError::CredentialResolutionFailed`] rather than falling back to
    /// using the file path string literal as a secret token.
    ///
    /// # Errors
    /// Returns [`IngestionError::CredentialResolutionFailed`] if reading the secret file fails.
    pub fn resolve_secret(&self) -> Result<Option<String>, IngestionError> {
        match self {
            Self::Anonymous => Ok(None),
            Self::Token(tok) | Self::BearerToken(tok) => {
                let trimmed = tok.trim();
                if trimmed.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(trimmed.to_owned()))
                }
            }
            Self::SecretFile(path) => {
                if !path.exists() {
                    return Err(IngestionError::CredentialResolutionFailed(format!(
                        "credential file '{}' does not exist",
                        path.display()
                    )));
                }

                match fs::read_to_string(path) {
                    Ok(content) => {
                        let trimmed = content.trim();
                        if trimmed.is_empty() {
                            Err(IngestionError::CredentialResolutionFailed(format!(
                                "credential file '{}' is empty",
                                path.display()
                            )))
                        } else {
                            Ok(Some(trimmed.to_owned()))
                        }
                    }
                    Err(e) => Err(IngestionError::CredentialResolutionFailed(format!(
                        "failed to read credential file '{}': {e}",
                        path.display()
                    ))),
                }
            }
            Self::BasicAuth {
                password_or_token, ..
            } => {
                let trimmed = password_or_token.trim();
                if trimmed.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(trimmed.to_owned()))
                }
            }
        }
    }
}

impl fmt::Debug for SourceCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Anonymous => write!(f, "SourceCredentials::Anonymous"),
            Self::Token(_) => write!(f, "SourceCredentials::Token([REDACTED])"),
            Self::BearerToken(_) => write!(f, "SourceCredentials::BearerToken([REDACTED])"),
            Self::SecretFile(p) => write!(
                f,
                "SourceCredentials::SecretFile({})",
                p.file_name().unwrap_or_default().to_string_lossy()
            ),
            Self::BasicAuth { username, .. } => write!(
                f,
                "SourceCredentials::BasicAuth {{ username: {username}, password: [REDACTED] }}"
            ),
        }
    }
}

/// Authentication outcome status returned by [`SourceAdapter::auth`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthStatus {
    /// Successfully authenticated with remote provider.
    Authenticated {
        /// Authenticated user or service account identity.
        identity: String,
        /// Granted permissions or scopes.
        permissions: Vec<String>,
    },
    /// Connected anonymously without credentials.
    Anonymous,
    /// Authentication rejected by remote repository.
    Failed {
        /// Diagnostic reason for authentication rejection.
        reason: String,
    },
}

/// Query filter options for repository search operations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchFilter {
    /// Filter by model architecture (e.g., "qwen2", "llama").
    pub architecture: Option<String>,
    /// Filter by quantization format (e.g., "Q4_K_M", "Q8_0").
    pub quantization: Option<String>,
    /// Maximum number of results to return.
    pub limit: Option<usize>,
}

/// Summary description of a model discovered in an external repository.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteModelSummary {
    /// Full repository identifier (e.g., "Qwen/Qwen2.5-7B-Instruct").
    pub id: String,
    /// Model name segment.
    pub name: String,
    /// Model namespace or organization.
    pub namespace: String,
    /// Brief description or summary.
    pub description: String,
    /// Repository tags and labels.
    pub tags: Vec<String>,
    /// Approximate download count.
    pub downloads: u64,
}

/// Available versions, revisions, and downloadable artifacts for a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelVersionListing {
    /// Repository namespace.
    pub namespace: String,
    /// Model name.
    pub model: String,
    /// List of available revisions or tags.
    pub revisions: Vec<ModelRevision>,
}

/// Specific downloadable revision of a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRevision {
    /// Revision tag (e.g., "q8_0", "main", "v1.0").
    pub tag: String,
    /// Expected SHA-256 digest hex string.
    pub digest: String,
    /// Total artifact size in bytes.
    pub size_bytes: u64,
    /// Format of the model artifact.
    pub format: ModelFormat,
}

/// Request parameters for initiating a model pull into the staging area.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullRequest {
    /// Source provider identifier (e.g., "huggingface", "ollama", "local").
    pub source: String,
    /// Repository namespace (e.g., "Qwen").
    pub namespace: String,
    /// Model name (e.g., "Qwen2.5-7B-Instruct").
    pub model: String,
    /// Tag or quantization identifier (e.g., "q8_0").
    pub tag: String,
    /// Expected cryptographic digest (optional; verified if provided).
    pub expected_digest: Option<String>,
    /// Target model format if known.
    pub format: Option<ModelFormat>,
}

/// Live streaming progress update for ongoing model pulls.
#[derive(Debug, Clone, PartialEq)]
pub struct PullProgress {
    /// Number of bytes downloaded so far.
    pub bytes_downloaded: u64,
    /// Total expected bytes if known from headers.
    pub total_bytes: Option<u64>,
    /// Completion fraction (0.0 to 1.0).
    pub percentage: f32,
}

/// Summary of a completed model pull in the staging directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullArtifactSummary {
    /// Model identifier.
    pub model_id: String,
    /// Revision tag.
    pub tag: String,
    /// Model binary format.
    pub format: ModelFormat,
    /// Total bytes downloaded and written.
    pub bytes_downloaded: u64,
    /// Cryptographic SHA-256 digest computed during streaming.
    pub computed_digest: String,
    /// Staging file path (`<store_root>/staging/<uuid>.part`).
    pub staging_path: PathBuf,
}

/// Download artifact metadata matching the project interface contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagingDownload {
    /// Path to the downloaded staging file.
    pub staging_path: PathBuf,
    /// Declared digest if specified in pull request.
    pub declared_digest: Option<String>,
    /// Computed cryptographic SHA-256 digest.
    pub computed_digest: String,
    /// Total size in bytes.
    pub size_bytes: u64,
    /// Format of the downloaded model.
    pub format: ModelFormat,
}

/// Standardized error type covering ingestion, staging, verification, and promotion.
#[derive(Debug)]
pub enum IngestionError {
    /// Remote repository authentication failed.
    AuthFailed(String),
    /// Network connection or transfer failure.
    NetworkError(String),
    /// Requested model, namespace, or tag not found.
    ModelNotFound(String),
    /// Model binary format header failed validation.
    CorruptedFormat {
        /// Format identifier.
        format: String,
        /// Specific reason for header corruption.
        reason: String,
    },
    /// Downloaded bytes failed cryptographic checksum verification.
    DigestMismatch {
        /// Expected SHA-256 digest.
        expected: String,
        /// Computed SHA-256 digest.
        actual: String,
    },
    /// Secret file could not be read or failed closed.
    CredentialResolutionFailed(String),
    /// Failure during staging file creation or writing.
    StagingError(String),
    /// Atomic promotion into content-addressed storage failed.
    PromotionError(String),
    /// Standard I/O error.
    Io(io::Error),
    /// Underlying platform error from `k3-core`.
    K3(K3Error),
}

impl fmt::Display for IngestionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthFailed(msg) => write!(f, "[ingestion auth failed] {msg}"),
            Self::NetworkError(msg) => write!(f, "[ingestion network error] {msg}"),
            Self::ModelNotFound(msg) => write!(f, "[ingestion model not found] {msg}"),
            Self::CorruptedFormat { format, reason } => {
                write!(
                    f,
                    "[ingestion format error] format '{format}' invalid: {reason}"
                )
            }
            Self::DigestMismatch { expected, actual } => write!(
                f,
                "[ingestion digest mismatch] expected digest '{expected}', computed '{actual}'"
            ),
            Self::CredentialResolutionFailed(msg) => {
                write!(f, "[ingestion credential error] {msg}")
            }
            Self::StagingError(msg) => write!(f, "[ingestion staging error] {msg}"),
            Self::PromotionError(msg) => write!(f, "[ingestion promotion error] {msg}"),
            Self::Io(e) => write!(f, "[ingestion io error] {e}"),
            Self::K3(e) => write!(f, "[ingestion k3 error] {e}"),
        }
    }
}

impl std::error::Error for IngestionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for IngestionError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<K3Error> for IngestionError {
    fn from(err: K3Error) -> Self {
        Self::K3(err)
    }
}

impl From<IngestionError> for K3Error {
    fn from(err: IngestionError) -> Self {
        match err {
            IngestionError::K3(k) => k,
            IngestionError::CorruptedFormat { format, reason } => K3Error::CapabilityMismatch {
                architecture: format,
                reason,
            },
            IngestionError::DigestMismatch { expected, actual } => K3Error::CapabilityMismatch {
                architecture: "sha256".to_owned(),
                reason: format!("digest mismatch: expected {expected}, got {actual}"),
            },
            other => K3Error::CapabilityMismatch {
                architecture: "ingestion".to_owned(),
                reason: other.to_string(),
            },
        }
    }
}

/// Standard lifecycle contract for external model source adapters (US-19).
pub trait SourceAdapter: Send + Sync + fmt::Debug {
    /// Returns the provider name identifier (e.g. "huggingface", "ollama", "local").
    fn source_name(&self) -> &str;

    /// Authenticates with the remote repository using provided credentials.
    ///
    /// # Errors
    /// Returns [`IngestionError::AuthFailed`] on rejected authentication.
    fn auth(&self, creds: &SourceCredentials) -> Result<AuthStatus, IngestionError>;

    /// Searches remote repository for models matching query and filter criteria.
    ///
    /// # Errors
    /// Returns [`IngestionError::NetworkError`] or [`IngestionError::AuthFailed`] on failure.
    fn search(
        &self,
        query: &str,
        filter: &SearchFilter,
    ) -> Result<Vec<RemoteModelSummary>, IngestionError>;

    /// Lists available revisions, tags, and downloadable files for a specific model.
    ///
    /// # Errors
    /// Returns [`IngestionError::ModelNotFound`] if model does not exist.
    fn list(&self, namespace: &str, model: &str) -> Result<ModelVersionListing, IngestionError>;

    /// Pulls model weights into a staging path with progress updates.
    ///
    /// # Errors
    /// Returns [`IngestionError`] on transfer, digest, or staging failure.
    fn pull(
        &self,
        request: &PullRequest,
        staging_path: &Path,
        progress_tx: Option<tokio::sync::mpsc::Sender<PullProgress>>,
    ) -> Result<PullArtifactSummary, IngestionError>;

    /// Convenience pull matching the `PROJECT.md` interface contract:
    /// `SourceAdapter::pull(source, model_id, tag, staging_dir) -> Result<StagingDownload, K3Error>`
    ///
    /// # Errors
    /// Returns [`K3Error`] on failure.
    fn pull_to_staging(
        &self,
        source: &str,
        model_id: &str,
        tag: &str,
        staging_dir: &Path,
    ) -> Result<StagingDownload, K3Error> {
        let (namespace, model) = if let Some((ns, m)) = model_id.split_once('/') {
            (ns.to_owned(), m.to_owned())
        } else {
            ("default".to_owned(), model_id.to_owned())
        };

        // Sanitize model and tag components to prevent relative path traversal
        let sanitized_model = model.replace(['/', '\\'], "_").replace("..", "_");
        let sanitized_tag = tag.replace(['/', '\\'], "_").replace("..", "_");

        let req = PullRequest {
            source: source.to_owned(),
            namespace,
            model: sanitized_model,
            tag: sanitized_tag,
            expected_digest: None,
            format: None,
        };

        // Use UUID v4 filename strictly confined within staging_dir per PROJECT.md
        let part_filename = super::staging::StagingPipeline::generate_uuid_part_filename();
        let part_path = staging_dir.join(part_filename);

        let summary = self.pull(&req, &part_path, None).map_err(K3Error::from)?;

        Ok(StagingDownload {
            staging_path: summary.staging_path,
            declared_digest: req.expected_digest,
            computed_digest: summary.computed_digest,
            size_bytes: summary.bytes_downloaded,
            format: summary.format,
        })
    }
}
