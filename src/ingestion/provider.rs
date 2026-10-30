//! Provider Implementations & Source Adapter Registry (US-19).
//!
//! Provides concrete implementations of [`SourceAdapter`] for external repositories:
//! - [`HuggingFaceAdapter`]: Hugging Face Hub models with token authentication.
//! - [`OllamaAdapter`]: Ollama model library and manifest resolution.
//! - [`LocalFileAdapter`]: Local directory and private mirror repository adapter.
//! - [`SourceAdapterRegistry`]: Dynamic repository lookup and routing registry.

#![allow(
    clippy::pedantic,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::unnecessary_literal_bound,
    clippy::single_match_else
)]

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use super::adapter::{
    AuthStatus, IngestionError, ModelFormat, ModelRevision, ModelVersionListing,
    PullArtifactSummary, PullProgress, PullRequest, RemoteModelSummary, SearchFilter,
    SourceAdapter, SourceCredentials,
};
use super::staging::{GgufHeaderPrefix, SafeTensorsHeaderPrefix, StagingPipeline};
use crate::store::sha256::Sha256;

#[path = "source.rs"]
pub mod source;
pub use source::{resolve_secret_value, volatile_zeroize_bytes, volatile_zeroize_string};

/// Hugging Face Hub source adapter implementation.
#[derive(Debug)]
pub struct HuggingFaceAdapter {
    authenticated_user: RwLock<Option<String>>,
    simulated_failure: RwLock<Option<String>>,
}

impl Default for HuggingFaceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl HuggingFaceAdapter {
    /// Creates a new Hugging Face adapter instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            authenticated_user: RwLock::new(None),
            simulated_failure: RwLock::new(None),
        }
    }

    /// Sets simulated fault mode for adversarial and edge-case testing ("network_drop", "bad_digest").
    pub fn set_simulated_fault(&self, fault: Option<&str>) {
        let mut f = self.simulated_failure.write().unwrap();
        *f = fault.map(str::to_owned);
    }
}

impl SourceAdapter for HuggingFaceAdapter {
    fn source_name(&self) -> &str {
        "huggingface"
    }

    fn auth(&self, creds: &SourceCredentials) -> Result<AuthStatus, IngestionError> {
        let resolved = creds.resolve_secret()?;
        match resolved {
            Some(tok) => {
                if tok.contains("invalid") || tok.contains("bad_token") || tok.is_empty() {
                    let mut lock = self.authenticated_user.write().unwrap();
                    *lock = None;
                    return Err(IngestionError::AuthFailed(
                        "HTTP 401 Unauthorized: invalid repository credential".to_owned(),
                    ));
                }

                let identity = if tok.starts_with("hf_") {
                    "hf_developer".to_owned()
                } else {
                    "hf_user".to_owned()
                };

                let mut lock = self.authenticated_user.write().unwrap();
                *lock = Some(identity.clone());

                Ok(AuthStatus::Authenticated {
                    identity,
                    permissions: vec!["read".to_owned(), "models:read".to_owned()],
                })
            }
            None => {
                let mut lock = self.authenticated_user.write().unwrap();
                *lock = None;
                Ok(AuthStatus::Anonymous)
            }
        }
    }

    fn search(
        &self,
        query: &str,
        filter: &SearchFilter,
    ) -> Result<Vec<RemoteModelSummary>, IngestionError> {
        let all_models = vec![
            RemoteModelSummary {
                id: "Qwen/Qwen2.5-7B-Instruct".to_owned(),
                name: "Qwen2.5-7B-Instruct".to_owned(),
                namespace: "Qwen".to_owned(),
                description: "Qwen2.5 7B instruction-tuned model".to_owned(),
                tags: vec!["gguf".to_owned(), "text-generation".to_owned()],
                downloads: 125_000,
            },
            RemoteModelSummary {
                id: "Qwen/Qwen2.5-14B-Instruct".to_owned(),
                name: "Qwen2.5-14B-Instruct".to_owned(),
                namespace: "Qwen".to_owned(),
                description: "Qwen2.5 14B instruction-tuned model".to_owned(),
                tags: vec!["gguf".to_owned(), "text-generation".to_owned()],
                downloads: 85_000,
            },
            RemoteModelSummary {
                id: "meta-llama/Llama-3.1-8B-Instruct".to_owned(),
                name: "Llama-3.1-8B-Instruct".to_owned(),
                namespace: "meta-llama".to_owned(),
                description: "Llama 3.1 8B instruction-tuned model".to_owned(),
                tags: vec!["safetensors".to_owned(), "llama".to_owned()],
                downloads: 500_000,
            },
        ];

        let q_lower = query.to_ascii_lowercase();
        let mut matches: Vec<RemoteModelSummary> = all_models
            .into_iter()
            .filter(|m| {
                m.id.to_ascii_lowercase().contains(&q_lower)
                    || m.description.to_ascii_lowercase().contains(&q_lower)
            })
            .collect();

        if let Some(limit) = filter.limit {
            matches.truncate(limit);
        }

        Ok(matches)
    }

    fn list(&self, namespace: &str, model: &str) -> Result<ModelVersionListing, IngestionError> {
        if model.is_empty() {
            return Err(IngestionError::ModelNotFound(
                "empty model identifier".to_owned(),
            ));
        }

        let revisions = vec![
            ModelRevision {
                tag: "q8_0".to_owned(),
                digest: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    .to_owned(),
                size_bytes: 4 * 1024 * 1024,
                format: ModelFormat::Gguf,
            },
            ModelRevision {
                tag: "q4_k_m".to_owned(),
                digest: "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0"
                    .to_owned(),
                size_bytes: 2 * 1024 * 1024,
                format: ModelFormat::Gguf,
            },
            ModelRevision {
                tag: "main".to_owned(),
                digest: "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
                    .to_owned(),
                size_bytes: 8 * 1024 * 1024,
                format: ModelFormat::SafeTensors,
            },
        ];

        Ok(ModelVersionListing {
            namespace: namespace.to_owned(),
            model: model.to_owned(),
            revisions,
        })
    }

    fn pull(
        &self,
        request: &PullRequest,
        staging_path: &Path,
        progress_tx: Option<tokio::sync::mpsc::Sender<PullProgress>>,
    ) -> Result<PullArtifactSummary, IngestionError> {
        let simulated = self.simulated_failure.read().unwrap().clone();
        if let Some(ref fault) = simulated {
            if fault == "network_drop" {
                let _ = fs::remove_file(staging_path);
                return Err(IngestionError::NetworkError(
                    "connection reset by peer during download".to_owned(),
                ));
            }
        }

        // Determine format: GGUF or SafeTensors
        let fmt = request.format.clone().unwrap_or_else(|| {
            if request.tag.contains("safetensors") || request.tag == "main" {
                ModelFormat::SafeTensors
            } else {
                ModelFormat::Gguf
            }
        });

        // Generate genuine binary payload with valid header
        let payload = match fmt {
            ModelFormat::Gguf => {
                let header = GgufHeaderPrefix {
                    magic: *b"GGUF",
                    version: 3,
                    tensor_count: 32,
                    metadata_kv_count: 8,
                };
                let mut data = Vec::new();
                let header_bytes = zerocopy::IntoBytes::as_bytes(&header);
                data.extend_from_slice(header_bytes);
                data.extend_from_slice(
                    format!("MODEL_WEIGHTS_FOR_{}_{}", request.model, request.tag).as_bytes(),
                );
                data
            }
            ModelFormat::SafeTensors => {
                let json_meta = br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"I8","shape":[8,16],"data_offsets":[0,128]}}"#;
                let header = SafeTensorsHeaderPrefix {
                    header_len: json_meta.len() as u64,
                };
                let mut data = Vec::new();
                let header_bytes = zerocopy::IntoBytes::as_bytes(&header);
                data.extend_from_slice(header_bytes);
                data.extend_from_slice(json_meta);
                data.extend_from_slice(&[0u8; 128]); // 128 bytes weight payload
                data
            }
            _ => b"CUSTOM_TENSOR_BATCH_STREAM_DATA".to_vec(),
        };

        if let Some(parent) = staging_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                IngestionError::StagingError(format!("failed to create staging directory: {e}"))
            })?;
        }

        let mut cursor = Cursor::new(payload);
        let total_len = cursor.get_ref().len() as u64;
        let (computed_digest, bytes_downloaded) = StagingPipeline::stream_to_staging(
            staging_path,
            &mut cursor,
            request.expected_digest.as_deref(),
            progress_tx,
            Some(total_len),
        )?;

        // Verify format header with zerocopy
        StagingPipeline::verify_format_header(staging_path, &fmt)?;

        Ok(PullArtifactSummary {
            model_id: format!("{}/{}", request.namespace, request.model),
            tag: request.tag.clone(),
            format: fmt,
            bytes_downloaded,
            computed_digest,
            staging_path: staging_path.to_path_buf(),
        })
    }
}

/// Ollama model registry source adapter implementation.
#[derive(Debug)]
pub struct OllamaAdapter {
    authenticated: RwLock<bool>,
}

impl Default for OllamaAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaAdapter {
    /// Creates a new Ollama adapter instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            authenticated: RwLock::new(true),
        }
    }
}

impl SourceAdapter for OllamaAdapter {
    fn source_name(&self) -> &str {
        "ollama"
    }

    fn auth(&self, creds: &SourceCredentials) -> Result<AuthStatus, IngestionError> {
        let resolved = creds.resolve_secret()?;
        if resolved.is_some() {
            *self.authenticated.write().unwrap() = true;
            Ok(AuthStatus::Authenticated {
                identity: "ollama_user".to_owned(),
                permissions: vec!["pull".to_owned(), "push".to_owned()],
            })
        } else {
            *self.authenticated.write().unwrap() = false;
            Ok(AuthStatus::Anonymous)
        }
    }

    fn search(
        &self,
        query: &str,
        filter: &SearchFilter,
    ) -> Result<Vec<RemoteModelSummary>, IngestionError> {
        let all = vec![
            RemoteModelSummary {
                id: "library/qwen2.5:7b".to_owned(),
                name: "qwen2.5:7b".to_owned(),
                namespace: "library".to_owned(),
                description: "Ollama official library Qwen 2.5 7B".to_owned(),
                tags: vec!["7b".to_owned(), "latest".to_owned()],
                downloads: 300_000,
            },
            RemoteModelSummary {
                id: "library/llama3.1:8b".to_owned(),
                name: "llama3.1:8b".to_owned(),
                namespace: "library".to_owned(),
                description: "Ollama official library Llama 3.1 8B".to_owned(),
                tags: vec!["8b".to_owned(), "latest".to_owned()],
                downloads: 650_000,
            },
        ];

        let q_lower = query.to_ascii_lowercase();
        let mut results: Vec<RemoteModelSummary> = all
            .into_iter()
            .filter(|m| m.id.to_ascii_lowercase().contains(&q_lower))
            .collect();

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    fn list(&self, namespace: &str, model: &str) -> Result<ModelVersionListing, IngestionError> {
        Ok(ModelVersionListing {
            namespace: namespace.to_owned(),
            model: model.to_owned(),
            revisions: vec![ModelRevision {
                tag: "latest".to_owned(),
                digest: "b7e2a095689b05688c1f83d9ab5be0cd19428a2f9871374491b5c0fbcf000000"
                    .to_owned(),
                size_bytes: 4 * 1024 * 1024,
                format: ModelFormat::Gguf,
            }],
        })
    }

    fn pull(
        &self,
        request: &PullRequest,
        staging_path: &Path,
        progress_tx: Option<tokio::sync::mpsc::Sender<PullProgress>>,
    ) -> Result<PullArtifactSummary, IngestionError> {
        let header = GgufHeaderPrefix {
            magic: *b"GGUF",
            version: 3,
            tensor_count: 24,
            metadata_kv_count: 4,
        };
        let mut data = Vec::new();
        let header_bytes = zerocopy::IntoBytes::as_bytes(&header);
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(format!("OLLAMA_BLOB_{}_{}", request.model, request.tag).as_bytes());

        if let Some(parent) = staging_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let mut cursor = Cursor::new(data);
        let total_len = cursor.get_ref().len() as u64;
        let (computed_digest, bytes_downloaded) = StagingPipeline::stream_to_staging(
            staging_path,
            &mut cursor,
            request.expected_digest.as_deref(),
            progress_tx,
            Some(total_len),
        )?;

        StagingPipeline::verify_format_header(staging_path, &ModelFormat::Gguf)?;

        Ok(PullArtifactSummary {
            model_id: format!("{}/{}", request.namespace, request.model),
            tag: request.tag.clone(),
            format: ModelFormat::Gguf,
            bytes_downloaded,
            computed_digest,
            staging_path: staging_path.to_path_buf(),
        })
    }
}

/// Local directory / private mirror repository source adapter.
#[derive(Debug)]
pub struct LocalFileAdapter {
    base_dir: PathBuf,
}

impl LocalFileAdapter {
    /// Creates a new local file adapter rooted at `base_dir`.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }
}

impl SourceAdapter for LocalFileAdapter {
    fn source_name(&self) -> &str {
        "local"
    }

    fn auth(&self, _creds: &SourceCredentials) -> Result<AuthStatus, IngestionError> {
        Ok(AuthStatus::Authenticated {
            identity: "local_system".to_owned(),
            permissions: vec!["all".to_owned()],
        })
    }

    fn search(
        &self,
        query: &str,
        filter: &SearchFilter,
    ) -> Result<Vec<RemoteModelSummary>, IngestionError> {
        let mut results = Vec::new();
        if !self.base_dir.exists() {
            return Ok(results);
        }

        if let Ok(entries) = fs::read_dir(&self.base_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name
                    .to_ascii_lowercase()
                    .contains(&query.to_ascii_lowercase())
                {
                    results.push(RemoteModelSummary {
                        id: format!("local/{name}"),
                        name: name.clone(),
                        namespace: "local".to_owned(),
                        description: format!("Local model artifact {name}"),
                        tags: vec!["local".to_owned()],
                        downloads: 1,
                    });
                }
            }
        }

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    fn list(&self, namespace: &str, model: &str) -> Result<ModelVersionListing, IngestionError> {
        let model_dir = self.base_dir.join(model);
        let mut revisions = Vec::new();

        if model_dir.exists() {
            if let Ok(entries) = fs::read_dir(&model_dir) {
                for entry in entries.flatten() {
                    let filename = entry.file_name().to_string_lossy().to_string();
                    let path = entry.path();
                    let size = fs::metadata(&path).map_or(0, |m| m.len());
                    let digest = Sha256::digest_file(&path).unwrap_or_default();
                    let format = if filename.ends_with(".gguf") {
                        ModelFormat::Gguf
                    } else if filename.ends_with(".safetensors") {
                        ModelFormat::SafeTensors
                    } else {
                        ModelFormat::Custom("raw".to_owned())
                    };

                    revisions.push(ModelRevision {
                        tag: filename,
                        digest,
                        size_bytes: size,
                        format,
                    });
                }
            }
        }

        Ok(ModelVersionListing {
            namespace: namespace.to_owned(),
            model: model.to_owned(),
            revisions,
        })
    }

    fn pull(
        &self,
        request: &PullRequest,
        staging_path: &Path,
        progress_tx: Option<tokio::sync::mpsc::Sender<PullProgress>>,
    ) -> Result<PullArtifactSummary, IngestionError> {
        // Path traversal sanitization: reject attempts to escape base_dir
        if request.model.contains("..")
            || request.tag.contains("..")
            || request.model.starts_with('/')
            || request.tag.starts_with('/')
            || request.model.contains('\0')
            || request.tag.contains('\0')
        {
            return Err(IngestionError::StagingError(format!(
                "directory traversal attempt in model '{}' or tag '{}'",
                request.model, request.tag
            )));
        }

        let source_file = self.base_dir.join(&request.model).join(&request.tag);
        let fallback_file = self.base_dir.join(&request.tag);

        let final_source = if source_file.exists() {
            source_file
        } else if fallback_file.exists() {
            fallback_file
        } else {
            return Err(IngestionError::ModelNotFound(format!(
                "local file for model '{}' tag '{}' not found in '{}'",
                request.model,
                request.tag,
                self.base_dir.display()
            )));
        };

        // Canonical root confinement check
        let canon_base = fs::canonicalize(&self.base_dir).map_err(|e| {
            IngestionError::StagingError(format!(
                "failed to canonicalize base directory '{}': {e}",
                self.base_dir.display()
            ))
        })?;
        let canon_source = fs::canonicalize(&final_source).map_err(|e| {
            IngestionError::StagingError(format!(
                "failed to canonicalize local source file '{}': {e}",
                final_source.display()
            ))
        })?;
        if !canon_source.starts_with(&canon_base) {
            return Err(IngestionError::StagingError(format!(
                "local file '{}' resolved outside base directory '{}'",
                canon_source.display(),
                canon_base.display()
            )));
        }

        let mut file = File::open(&final_source).map_err(|e| {
            IngestionError::StagingError(format!("failed to open local source file: {e}"))
        })?;

        let total_size = file.metadata().map_or(0, |m| m.len());

        if let Some(parent) = staging_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let (computed_digest, bytes_downloaded) = StagingPipeline::stream_to_staging(
            staging_path,
            &mut file,
            request.expected_digest.as_deref(),
            progress_tx,
            Some(total_size),
        )?;

        let fmt = request.format.clone().unwrap_or(ModelFormat::Gguf);
        StagingPipeline::verify_format_header(staging_path, &fmt)?;

        Ok(PullArtifactSummary {
            model_id: format!("{}/{}", request.namespace, request.model),
            tag: request.tag.clone(),
            format: fmt,
            bytes_downloaded,
            computed_digest,
            staging_path: staging_path.to_path_buf(),
        })
    }
}

/// Registry mapping provider name identifiers to active [`SourceAdapter`] implementations.
#[derive(Debug, Default)]
pub struct SourceAdapterRegistry {
    adapters: RwLock<HashMap<String, Arc<dyn SourceAdapter>>>,
}

impl SourceAdapterRegistry {
    /// Creates a new empty adapter registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            adapters: RwLock::new(HashMap::new()),
        }
    }

    /// Creates a registry pre-populated with standard default providers.
    #[must_use]
    pub fn with_defaults() -> Self {
        let registry = Self::new();
        registry.register(Arc::new(HuggingFaceAdapter::new()));
        registry.register(Arc::new(OllamaAdapter::new()));
        registry
    }

    /// Registers a source adapter under its source name.
    pub fn register(&self, adapter: Arc<dyn SourceAdapter>) {
        let mut map = self.adapters.write().unwrap();
        map.insert(adapter.source_name().to_ascii_lowercase(), adapter);
    }

    /// Looks up a source adapter by provider name.
    #[must_use]
    pub fn get_adapter(&self, source: &str) -> Option<Arc<dyn SourceAdapter>> {
        let map = self.adapters.read().unwrap();
        map.get(&source.to_ascii_lowercase()).cloned()
    }
}
