//! Content-Addressed Store (CAS) & Human-Browsable Model Registry (US-18, ADR-035, ADR-036).
//!
//! Organizes model weights by cryptographic SHA-256 digest (`blobs/sha256/<digest>`),
//! structured canonical manifests (`manifests/sha256/<manifest_digest>.json`),
//! and a human-browsable relative symlink alias tree (`models/<source>/<namespace>/<name>/<tag>`).

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use k3_core::K3Error;
use serde::{Deserialize, Serialize};

use super::sha256::Sha256;
use crate::persistence::{
    BlobReferenceRecord, CasCatalog, ModelAliasRecord, ModelBlobRecord, ModelManifestRecord,
};

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Default system store root when none is configured via CLI or environment (ADR-036).
pub const DEFAULT_STORE_ROOT: &str = "/var/lib/k3";

/// Environment variable for overriding the content-addressed store root (US-18 Scenario 2).
pub const ENV_STORE_ROOT: &str = "K3_STORE_ROOT";

/// Maximum allowed character length for any single path segment (source, namespace, name, tag).
pub const MAX_PATH_COMPONENT_LEN: usize = 128;

/// Validates that a path component does not contain directory traversal sequences,
/// path separators, null bytes, or dangerous control characters (Security R3).
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] if validation fails.
pub fn validate_path_component(component: &str) -> Result<(), K3Error> {
    if component.is_empty() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_path".to_owned(),
            reason: "path component cannot be empty".to_owned(),
        });
    }

    if component.len() > MAX_PATH_COMPONENT_LEN {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_path".to_owned(),
            reason: format!(
                "path component exceeds maximum length of {MAX_PATH_COMPONENT_LEN} characters"
            ),
        });
    }

    if component == "." || component == ".." {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_path".to_owned(),
            reason: format!("path traversal sequence '{component}' is prohibited"),
        });
    }

    if component.starts_with('-') {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_path".to_owned(),
            reason: format!("leading dash in path component '{component}' is prohibited"),
        });
    }

    for ch in component.chars() {
        if ch == '/' || ch == '\\' || ch == '\0' || ch.is_control() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_path".to_owned(),
                reason: format!("illegal character '{ch}' in path component '{component}'"),
            });
        }
        if !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-' && ch != '.' {
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_path".to_owned(),
                reason: format!("unpermitted character '{ch}' in path component '{component}'"),
            });
        }
    }

    Ok(())
}

/// Parses and validates a 4-part alias identifier into `(source, namespace, name, tag)`.
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] if the format is invalid or any component fails validation.
pub fn parse_alias_id(alias_id: &str) -> Result<(String, String, String, String), K3Error> {
    let clean = alias_id.strip_prefix("models/").unwrap_or(alias_id);
    let parts: Vec<&str> = clean.split('/').collect();
    if parts.len() != 4 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_alias".to_owned(),
            reason: format!(
                "alias identifier must be in the format '<source>/<namespace>/<name>/<tag>'; got '{alias_id}'"
            ),
        });
    }

    validate_path_component(parts[0])?;
    validate_path_component(parts[1])?;
    validate_path_component(parts[2])?;
    validate_path_component(parts[3])?;

    Ok((
        parts[0].to_owned(),
        parts[1].to_owned(),
        parts[2].to_owned(),
        parts[3].to_owned(),
    ))
}

/// Validates that a string is a strict 64-character lowercase hexadecimal SHA-256 digest (`^[a-f0-9]{64}$`).
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] if length is not 64 or any character is outside `[a-f0-9]`.
pub fn validate_sha256_digest(digest: &str) -> Result<(), K3Error> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_digest".to_owned(),
            reason: format!(
                "invalid SHA-256 digest '{digest}': must be exactly 64 lowercase hexadecimal characters (^[a-f0-9]{{64}}$)"
            ),
        });
    }
    Ok(())
}

/// Canonical metadata manifest describing a registered model build (US-18, ADR-035).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Manifest schema format version (default 1).
    pub schema_version: u32,
    /// Model format: "gguf", "safetensors", etc.
    pub format: String,
    /// Architecture identifier (e.g., "synthetic", "qwen3").
    pub architecture_id: String,
    /// Architecture recipe version (e.g., "1.0.0").
    pub recipe_version: String,
    /// Quantization format (e.g., `"Q4_K_M"`, `"F32"`).
    pub quantization: String,
    /// Context length trained into weights.
    pub context_length: u64,
    /// Optional license string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Optional provenance or download URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// Cryptographic SHA-256 digest of underlying model weights blob.
    pub weight_digest: String,
    /// Size of weights blob in bytes.
    pub weight_size_bytes: u64,
    /// Manifest creation timestamp in epoch seconds.
    pub created_at: u64,
}

impl ModelManifest {
    /// Serializes this manifest to a canonical JSON string.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if serialization fails.
    pub fn to_canonical_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Computes the SHA-256 digest of this manifest's canonical JSON.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if serialization fails.
    pub fn manifest_digest(&self) -> Result<String, serde_json::Error> {
        let json = self.to_canonical_json()?;
        Ok(Sha256::digest(json.as_bytes()))
    }
}

/// Resolved model alias information linking to manifest and weights on disk (US-18 Scenario 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAlias {
    /// Canonical alias identifier (e.g. "huggingface/Qwen/Qwen2.5-7B/latest").
    pub alias_id: String,
    /// Deserialized manifest metadata.
    pub manifest: ModelManifest,
    /// Absolute filesystem path to canonical manifest file.
    pub manifest_path: PathBuf,
    /// Absolute filesystem path to deduplicated weights blob.
    pub weight_path: PathBuf,
}

/// Content-Addressed Store (CAS) management instance.
#[derive(Debug, Clone)]
pub struct ContentAddressedStore {
    root: PathBuf,
    catalog: Arc<dyn CasCatalog>,
}

impl ContentAddressedStore {
    /// Opens or initializes a Content-Addressed Store under `root`.
    ///
    /// Creates the directory hierarchy (`blobs/sha256`, `manifests/sha256`, `models`, `staging`)
    /// if it does not already exist.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if directory creation fails.
    pub fn open(root: impl AsRef<Path>, catalog: Arc<dyn CasCatalog>) -> Result<Self, K3Error> {
        let root_buf = root.as_ref().to_path_buf();
        let blobs_dir = root_buf.join("blobs").join("sha256");
        let manifests_dir = root_buf.join("manifests").join("sha256");
        let models_dir = root_buf.join("models");
        let staging_dir = root_buf.join("staging");

        fs::create_dir_all(&blobs_dir).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_init".to_owned(),
            reason: format!("failed to create blobs dir '{}': {e}", blobs_dir.display()),
        })?;

        fs::create_dir_all(&manifests_dir).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_init".to_owned(),
            reason: format!(
                "failed to create manifests dir '{}': {e}",
                manifests_dir.display()
            ),
        })?;

        fs::create_dir_all(&models_dir).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_init".to_owned(),
            reason: format!(
                "failed to create models dir '{}': {e}",
                models_dir.display()
            ),
        })?;

        fs::create_dir_all(&staging_dir).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_init".to_owned(),
            reason: format!(
                "failed to create staging dir '{}': {e}",
                staging_dir.display()
            ),
        })?;

        Ok(Self {
            root: root_buf,
            catalog,
        })
    }

    /// Opens the store at the configured default location, respecting `K3_STORE_ROOT` override.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on initialization failure.
    pub fn open_default(catalog: Arc<dyn CasCatalog>) -> Result<Self, K3Error> {
        let path = std::env::var(ENV_STORE_ROOT).unwrap_or_else(|_| DEFAULT_STORE_ROOT.to_owned());
        Self::open(Path::new(&path), catalog)
    }

    /// Returns the root path of the store.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the blobs directory (`<root>/blobs/sha256`).
    #[must_use]
    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs").join("sha256")
    }

    /// Returns the manifests directory (`<root>/manifests/sha256`).
    #[must_use]
    pub fn manifests_dir(&self) -> PathBuf {
        self.root.join("manifests").join("sha256")
    }

    /// Returns the models alias directory (`<root>/models`).
    #[must_use]
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    /// Returns the staging directory (`<root>/staging`).
    #[must_use]
    pub fn staging_dir(&self) -> PathBuf {
        self.root.join("staging")
    }

    /// Returns a reference to the embedded `DuckDB` catalog.
    #[must_use]
    pub fn catalog(&self) -> &Arc<dyn CasCatalog> {
        &self.catalog
    }

    /// Creates a unique temporary file in `<root>/staging/<uuid>.part`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if file creation fails.
    pub fn create_staging_file(&self) -> Result<(PathBuf, File), K3Error> {
        let seq = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let pid = std::process::id();

        let mut hasher = Sha256::new();
        hasher.update(&nanos.to_le_bytes());
        hasher.update(&pid.to_le_bytes());
        hasher.update(&seq.to_le_bytes());
        let digest = hasher.finalize();

        let mut uuid_bytes = [0u8; 16];
        uuid_bytes.copy_from_slice(&digest[0..16]);
        uuid_bytes[6] = (uuid_bytes[6] & 0x0f) | 0x40; // UUID Version 4
        uuid_bytes[8] = (uuid_bytes[8] & 0x3f) | 0x80; // Variant RFC 4122

        let unique_name = format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}.part",
            uuid_bytes[0],
            uuid_bytes[1],
            uuid_bytes[2],
            uuid_bytes[3],
            uuid_bytes[4],
            uuid_bytes[5],
            uuid_bytes[6],
            uuid_bytes[7],
            uuid_bytes[8],
            uuid_bytes[9],
            uuid_bytes[10],
            uuid_bytes[11],
            uuid_bytes[12],
            uuid_bytes[13],
            uuid_bytes[14],
            uuid_bytes[15]
        );
        let path = self.staging_dir().join(unique_name);
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_staging".to_owned(),
                reason: format!("failed to create staging file '{}': {e}", path.display()),
            })?;

        // Best effort: set restrictive 0600 permissions on unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
        }

        Ok((path, file))
    }

    /// Verifies the cryptographic digest and format header of a staged file, then
    /// atomically promotes it into `blobs/sha256/<digest>` (US-18, US-19, ADR-040, Feature 26).
    ///
    /// If an identical blob already exists, the staging file is unlinked and the existing
    /// blob is deduplicated without duplicating bytes on storage (US-18 Scenario 1).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if digest verification or format checking fails.
    #[allow(clippy::too_many_lines)]
    pub fn promote_staged_blob(
        &self,
        staging_path: &Path,
        expected_digest: &str,
        format: &str,
    ) -> Result<PathBuf, K3Error> {
        validate_sha256_digest(expected_digest)?;

        // 1. Open staging file with read/write access and take exclusive non-blocking advisory lock
        let mut file = File::options()
            .read(true)
            .write(true)
            .open(staging_path)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_promote".to_owned(),
                reason: format!(
                    "failed to open staging file '{}': {e}",
                    staging_path.display()
                ),
            })?;

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            // SAFETY: `fd` is an open, valid file descriptor owned by promote_staged_blob.
            let lock_res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
            if lock_res != 0 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "cas_promote".to_owned(),
                    reason: format!(
                        "staging file '{}' is locked by another process (flock contention)",
                        staging_path.display()
                    ),
                });
            }
        }

        // 2. Stream and compute SHA-256 over open descriptor
        file.seek(SeekFrom::Start(0)).map_err(|e| {
            let _ = fs::remove_file(staging_path);
            K3Error::CapabilityMismatch {
                architecture: "cas_promote".to_owned(),
                reason: format!("failed to seek staging file: {e}"),
            }
        })?;

        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 64 * 1024];
        let mut total_bytes = 0u64;
        loop {
            match file.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    hasher.update(&buffer[..n]);
                    total_bytes = total_bytes.saturating_add(n as u64);
                }
                Err(e) => {
                    let _ = fs::remove_file(staging_path);
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "cas_integrity".to_owned(),
                        reason: format!(
                            "failed to read staging file '{}': {e}",
                            staging_path.display()
                        ),
                    });
                }
            }
        }
        let actual_digest = hasher.finalize_hex();

        if actual_digest != expected_digest {
            let _ = fs::remove_file(staging_path);
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_integrity".to_owned(),
                reason: format!(
                    "integrity verification failed for staging blob: expected digest '{expected_digest}', got '{actual_digest}'"
                ),
            });
        }

        // 3. Format header verification from open descriptor
        file.seek(SeekFrom::Start(0)).map_err(|e| {
            let _ = fs::remove_file(staging_path);
            K3Error::CapabilityMismatch {
                architecture: "cas_promote".to_owned(),
                reason: format!("failed to seek staging file: {e}"),
            }
        })?;

        if let Err(e) = Self::verify_format_header_from_file(&mut file, format, total_bytes) {
            let _ = fs::remove_file(staging_path);
            return Err(e);
        }

        // 4. Target blob destination and atomic promotion
        let final_path = self.blobs_dir().join(&actual_digest);
        let size_bytes = total_bytes;

        if final_path.exists() {
            // Deduplication: unlinked staging file and reuse existing blob
            let _ = fs::remove_file(staging_path);
        } else {
            // Atomic rename into blobs/sha256/<digest>
            fs::rename(staging_path, &final_path).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_promote".to_owned(),
                reason: format!(
                    "failed to atomically promote staging file to '{}': {e}",
                    final_path.display()
                ),
            })?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(&final_path, fs::Permissions::from_mode(0o444));
            }
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // SAFETY: `file.as_raw_fd()` is a valid open file descriptor.
            let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
        }
        drop(file);

        // 5. Register in DuckDB catalog if not already tracked
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let blob_rec = ModelBlobRecord {
            digest: actual_digest,
            size_bytes,
            blob_type: "raw_weights".to_owned(),
            created_at: now,
            last_accessed_at: now,
            ref_count: 0,
        };

        self.catalog
            .save_blob(&blob_rec)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_catalog".to_owned(),
                reason: format!("failed to register blob in catalog: {e}"),
            })?;

        Ok(final_path)
    }

    /// Verifies format header directly from an open, locked file descriptor.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on invalid header format or bounds violation.
    #[allow(clippy::too_many_lines)]
    pub fn verify_format_header_from_file(
        file: &mut File,
        format: &str,
        file_len: u64,
    ) -> Result<(), K3Error> {
        let fmt_lower = format.to_ascii_lowercase();
        if fmt_lower == "gguf" {
            if file_len < 24 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("file is truncated for GGUF: length {file_len} < 24"),
                });
            }
            let mut magic = [0u8; 4];
            file.read_exact(&mut magic)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("failed to read GGUF magic: {e}"),
                })?;
            if &magic != b"GGUF" {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("file does not contain valid GGUF magic (found: {magic:?})"),
                });
            }
            let mut ver_bytes = [0u8; 4];
            file.read_exact(&mut ver_bytes)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("failed to read GGUF version: {e}"),
                })?;
            let version = u32::from_le_bytes(ver_bytes);
            if !(1..=3).contains(&version) {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("unsupported GGUF version {version}; supported: 1, 2, 3"),
                });
            }
            let mut tc_bytes = [0u8; 8];
            file.read_exact(&mut tc_bytes)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("failed to read GGUF tensor_count: {e}"),
                })?;
            let tensor_count = u64::from_le_bytes(tc_bytes);
            if tensor_count > 65_536 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!(
                        "GGUF tensor_count {tensor_count} exceeds maximum sanity limit of 65536"
                    ),
                });
            }
            let mut kc_bytes = [0u8; 8];
            file.read_exact(&mut kc_bytes)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("failed to read GGUF kv_count: {e}"),
                })?;
            let kv_count = u64::from_le_bytes(kc_bytes);
            if kv_count > 65_536 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!(
                        "GGUF kv_count {kv_count} exceeds maximum sanity limit of 65536"
                    ),
                });
            }
        } else if fmt_lower == "safetensors" {
            if file_len < 8 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!("file is truncated for SafeTensors: length {file_len} < 8"),
                });
            }
            let mut len_bytes = [0u8; 8];
            file.read_exact(&mut len_bytes)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!("failed to read SafeTensors header length: {e}"),
                })?;
            let header_len = u64::from_le_bytes(len_bytes);
            if header_len == 0 || header_len > 100 * 1024 * 1024 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!("invalid SafeTensors header length: {header_len}"),
                });
            }
            let total_min_len = (8u64).saturating_add(header_len);
            if file_len < total_min_len {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!(
                        "file is truncated: requires at least {total_min_len} bytes for SafeTensors header, got {file_len}"
                    ),
                });
            }
            let header_len_usize =
                usize::try_from(header_len).map_err(|_| K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: "SafeTensors header length exceeds pointer address space".to_owned(),
                })?;
            let mut json_buf = vec![0u8; header_len_usize];
            file.read_exact(&mut json_buf)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!("failed to read SafeTensors JSON header: {e}"),
                })?;
            let value: serde_json::Value =
                serde_json::from_slice(&json_buf).map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!("invalid SafeTensors JSON metadata: {e}"),
                })?;
            if let Some(map) = value.as_object() {
                for (name, tensor_val) in map {
                    if name == "__metadata__" {
                        continue;
                    }
                    if let Some(offsets) = tensor_val.get("data_offsets").and_then(|v| v.as_array())
                    {
                        if offsets.len() != 2 {
                            return Err(K3Error::CapabilityMismatch {
                                architecture: "safetensors".to_owned(),
                                reason: format!(
                                    "tensor '{name}' data_offsets must be a 2-element array"
                                ),
                            });
                        }
                        let begin =
                            offsets[0]
                                .as_u64()
                                .ok_or_else(|| K3Error::CapabilityMismatch {
                                    architecture: "safetensors".to_owned(),
                                    reason: format!(
                                        "tensor '{name}' begin offset is not a valid u64"
                                    ),
                                })?;
                        let end =
                            offsets[1]
                                .as_u64()
                                .ok_or_else(|| K3Error::CapabilityMismatch {
                                    architecture: "safetensors".to_owned(),
                                    reason: format!(
                                        "tensor '{name}' end offset is not a valid u64"
                                    ),
                                })?;
                        if begin > end {
                            return Err(K3Error::CapabilityMismatch {
                                architecture: "safetensors".to_owned(),
                                reason: format!(
                                    "tensor '{name}' inverted data_offsets: begin ({begin}) > end ({end})"
                                ),
                            });
                        }
                        let total_extent = (8u64)
                            .checked_add(header_len)
                            .and_then(|h| h.checked_add(end));
                        match total_extent {
                            Some(ext) if ext <= file_len => {}
                            _ => {
                                return Err(K3Error::CapabilityMismatch {
                                    architecture: "safetensors".to_owned(),
                                    reason: format!(
                                        "tensor '{name}' offset {end} exceeds file length {file_len}"
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Verifies the format header of a weight file (GGUF magic / `SafeTensors` header).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on invalid header format.
    pub fn verify_format_header(path: &Path, format: &str) -> Result<(), K3Error> {
        let mut file = File::open(path).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_header".to_owned(),
            reason: format!("failed to open file '{}': {e}", path.display()),
        })?;
        let file_len = file.metadata().map_or(0, |m| m.len());
        Self::verify_format_header_from_file(&mut file, format, file_len)
    }

    /// Registers a model manifest in the store and `DuckDB` catalog (US-18).
    ///
    /// Writes the canonical manifest to `<root>/manifests/sha256/<manifest_digest>.json`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on serialization, disk write, or catalog failure.
    pub fn register_manifest(
        &self,
        manifest: &ModelManifest,
    ) -> Result<ModelManifestRecord, K3Error> {
        validate_sha256_digest(&manifest.weight_digest)?;

        let canonical_json =
            manifest
                .to_canonical_json()
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "cas_manifest".to_owned(),
                    reason: format!("failed to serialize manifest to canonical JSON: {e}"),
                })?;

        let manifest_digest = Sha256::digest(canonical_json.as_bytes());
        let manifest_path = self.manifests_dir().join(format!("{manifest_digest}.json"));

        // Write manifest atomically via temporary file
        let temp_manifest = self.manifests_dir().join(format!("{manifest_digest}.tmp"));
        let mut f = File::create(&temp_manifest).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_manifest".to_owned(),
            reason: format!(
                "failed to create temp manifest file '{}': {e}",
                temp_manifest.display()
            ),
        })?;

        f.write_all(canonical_json.as_bytes())
            .and_then(|()| f.flush())
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_manifest".to_owned(),
                reason: format!("failed to write manifest: {e}"),
            })?;

        fs::rename(&temp_manifest, &manifest_path).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_manifest".to_owned(),
            reason: format!(
                "failed to atomically promote manifest to '{}': {e}",
                manifest_path.display()
            ),
        })?;

        let record = ModelManifestRecord {
            manifest_digest: manifest_digest.clone(),
            weight_digest: manifest.weight_digest.clone(),
            format: manifest.format.clone(),
            architecture_id: manifest.architecture_id.clone(),
            recipe_version: manifest.recipe_version.clone(),
            quantization: manifest.quantization.clone(),
            context_length: manifest.context_length,
            license: manifest.license.clone(),
            origin: manifest.origin.clone(),
            manifest_json: canonical_json,
            created_at: manifest.created_at,
        };

        self.catalog
            .save_manifest(&record)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_catalog".to_owned(),
                reason: format!("failed to save manifest in DuckDB catalog: {e}"),
            })?;

        // Record granular reference in DuckDB
        let ref_rec = BlobReferenceRecord {
            ref_id: format!("manifest:{manifest_digest}"),
            blob_digest: manifest.weight_digest.clone(),
            ref_type: "manifest".to_owned(),
            ref_source: manifest_digest,
            created_at: manifest.created_at,
        };
        self.catalog
            .add_blob_reference(&ref_rec)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_catalog".to_owned(),
                reason: format!("failed to record blob reference: {e}"),
            })?;

        Ok(record)
    }

    /// Creates a human-browsable alias directory in `models/<source>/<namespace>/<name>/<tag>`
    /// containing relative symlinks to the manifest and weight blob (US-18 Scenario 3).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if path validation, directory creation, symlinking,
    /// or catalog recording fails.
    #[allow(clippy::too_many_lines)]
    pub fn create_alias(
        &self,
        source: &str,
        namespace: &str,
        name: &str,
        tag: &str,
        manifest_digest: &str,
    ) -> Result<ModelAliasRecord, K3Error> {
        validate_path_component(source)?;
        validate_path_component(namespace)?;
        validate_path_component(name)?;
        validate_path_component(tag)?;
        validate_sha256_digest(manifest_digest)?;

        let manifest = self
            .catalog
            .get_manifest(manifest_digest)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_alias".to_owned(),
                reason: format!("failed to lookup manifest '{manifest_digest}': {e}"),
            })?
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "cas_alias".to_owned(),
                reason: format!("manifest '{manifest_digest}' not found in catalog"),
            })?;

        let alias_id = format!("{source}/{namespace}/{name}/{tag}");
        let alias_dir = self
            .models_dir()
            .join(source)
            .join(namespace)
            .join(name)
            .join(tag);

        fs::create_dir_all(&alias_dir).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_alias".to_owned(),
            reason: format!(
                "failed to create alias directory '{}': {e}",
                alias_dir.display()
            ),
        })?;

        // Create relative symlinks:
        // models/<source>/<namespace>/<name>/<tag>/manifest.json -> ../../../../../manifests/sha256/<digest>.json
        // models/<source>/<namespace>/<name>/<tag>/weights -> ../../../../../blobs/sha256/<digest>
        let manifest_target = PathBuf::from("../../../../../manifests/sha256")
            .join(format!("{manifest_digest}.json"));
        let weights_target =
            PathBuf::from("../../../../../blobs/sha256").join(&manifest.weight_digest);

        let manifest_link = alias_dir.join("manifest.json");
        let weights_link = alias_dir.join("weights");

        let _ = fs::remove_file(&manifest_link);
        let _ = fs::remove_file(&weights_link);

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            symlink(&manifest_target, &manifest_link).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_alias".to_owned(),
                reason: format!(
                    "failed to create manifest symlink '{}' -> '{}': {e}",
                    manifest_link.display(),
                    manifest_target.display()
                ),
            })?;

            symlink(&weights_target, &weights_link).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_alias".to_owned(),
                reason: format!(
                    "failed to create weights symlink '{}' -> '{}': {e}",
                    weights_link.display(),
                    weights_target.display()
                ),
            })?;
        }

        #[cfg(not(unix))]
        {
            // Fallback for non-unix platforms
            fs::write(&manifest_link, manifest_target.to_string_lossy().as_bytes()).map_err(
                |e| K3Error::CapabilityMismatch {
                    architecture: "cas_alias".to_owned(),
                    reason: format!("failed to write fallback manifest link: {e}"),
                },
            )?;
            fs::write(&weights_link, weights_target.to_string_lossy().as_bytes()).map_err(|e| {
                K3Error::CapabilityMismatch {
                    architecture: "cas_alias".to_owned(),
                    reason: format!("failed to write fallback weights link: {e}"),
                }
            })?;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let alias_rel_path = format!("models/{source}/{namespace}/{name}/{tag}");

        let record = ModelAliasRecord {
            alias_id: alias_id.clone(),
            source: source.to_owned(),
            namespace: namespace.to_owned(),
            name: name.to_owned(),
            tag: tag.to_owned(),
            manifest_digest: manifest_digest.to_owned(),
            weight_digest: manifest.weight_digest.clone(),
            alias_path: alias_rel_path,
            created_at: now,
        };

        self.catalog
            .save_alias(&record)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_catalog".to_owned(),
                reason: format!("failed to save alias in DuckDB catalog: {e}"),
            })?;

        // Add reference in DuckDB
        let ref_rec = BlobReferenceRecord {
            ref_id: format!("alias:{alias_id}"),
            blob_digest: manifest.weight_digest,
            ref_type: "alias".to_owned(),
            ref_source: alias_id,
            created_at: now,
        };
        self.catalog
            .add_blob_reference(&ref_rec)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_catalog".to_owned(),
                reason: format!("failed to record alias blob reference: {e}"),
            })?;

        Ok(record)
    }

    /// Resolves a human-browsable alias into its manifest and canonical weights path (US-18 Scenario 3).
    ///
    /// Verifies symlinks resolve strictly within the store root to prevent path traversal attacks.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if alias resolution fails or points outside the store.
    #[allow(clippy::too_many_lines)]
    pub fn resolve_alias(&self, alias_str: &str) -> Result<ResolvedAlias, K3Error> {
        let (source, namespace, name, tag) = parse_alias_id(alias_str)?;
        let alias_id = format!("{source}/{namespace}/{name}/{tag}");

        let alias_dir = self
            .models_dir()
            .join(&source)
            .join(&namespace)
            .join(&name)
            .join(&tag);
        let canon_root = fs::canonicalize(&self.root).unwrap_or_else(|_| self.root.clone());

        // Symlink escape defense on physical alias directory
        if alias_dir.exists() {
            let weights_link = alias_dir.join("weights");
            if weights_link.is_symlink() || weights_link.exists() {
                if let Ok(canon_weights) = fs::canonicalize(&weights_link) {
                    if !canon_weights.starts_with(&canon_root) {
                        return Err(K3Error::CapabilityMismatch {
                            architecture: "cas_security".to_owned(),
                            reason: format!(
                                "alias '{alias_id}' resolved outside configured store root"
                            ),
                        });
                    }
                }
            }
            let manifest_link = alias_dir.join("manifest.json");
            if manifest_link.is_symlink() || manifest_link.exists() {
                if let Ok(canon_manifest) = fs::canonicalize(&manifest_link) {
                    if !canon_manifest.starts_with(&canon_root) {
                        return Err(K3Error::CapabilityMismatch {
                            architecture: "cas_security".to_owned(),
                            reason: format!(
                                "alias '{alias_id}' resolved outside configured store root"
                            ),
                        });
                    }
                }
            }
        }

        let alias_record = self
            .catalog
            .find_alias(&source, &namespace, &name, &tag)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!("catalog query failed for alias '{alias_id}': {e}"),
            })?
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!("model alias '{alias_id}' is not registered"),
            })?;

        let manifest_record = self
            .catalog
            .get_manifest(&alias_record.manifest_digest)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!(
                    "catalog query failed for manifest '{}': {e}",
                    alias_record.manifest_digest
                ),
            })?
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!(
                    "manifest '{}' for alias '{alias_id}' not found in catalog",
                    alias_record.manifest_digest
                ),
            })?;

        let manifest: ModelManifest = serde_json::from_str(&manifest_record.manifest_json)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!("corrupt manifest JSON in catalog: {e}"),
            })?;

        let weight_path = self.blobs_dir().join(&manifest.weight_digest);
        let manifest_path = self
            .manifests_dir()
            .join(format!("{}.json", alias_record.manifest_digest));

        if !weight_path.exists() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!(
                    "weight blob '{}' referenced by alias '{alias_id}' is missing from storage",
                    weight_path.display()
                ),
            });
        }

        // Symlink traversal defense: verify resolved paths reside under canonical store root
        let canon_weight =
            fs::canonicalize(&weight_path).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_security".to_owned(),
                reason: format!(
                    "failed to canonicalize weight path '{}': {e}",
                    weight_path.display()
                ),
            })?;
        let canon_manifest =
            fs::canonicalize(&manifest_path).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_security".to_owned(),
                reason: format!(
                    "failed to canonicalize manifest path '{}': {e}",
                    manifest_path.display()
                ),
            })?;

        if !canon_weight.starts_with(&canon_root) || !canon_manifest.starts_with(&canon_root) {
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_security".to_owned(),
                reason: format!("alias '{alias_id}' resolved outside configured store root"),
            });
        }

        // Update access timestamp
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let _ = self
            .catalog
            .update_blob_access(&manifest.weight_digest, now);

        Ok(ResolvedAlias {
            alias_id,
            manifest,
            manifest_path,
            weight_path,
        })
    }

    /// Removes a model alias and unlinks its relative directory, decrementing reference counts
    /// in `DuckDB` without immediately deleting the underlying weight blob (US-22 Scenario 1).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on failure.
    pub fn remove_alias(
        &self,
        source: &str,
        namespace: &str,
        name: &str,
        tag: &str,
    ) -> Result<bool, K3Error> {
        validate_path_component(source)?;
        validate_path_component(namespace)?;
        validate_path_component(name)?;
        validate_path_component(tag)?;

        let alias_id = format!("{source}/{namespace}/{name}/{tag}");
        let alias_dir = self
            .models_dir()
            .join(source)
            .join(namespace)
            .join(name)
            .join(tag);

        // Remove from DuckDB catalog
        let alias_opt =
            self.catalog
                .get_alias(&alias_id)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "cas_rm".to_owned(),
                    reason: format!("failed to query alias '{alias_id}': {e}"),
                })?;

        let Some(alias_rec) = alias_opt else {
            return Ok(false);
        };

        // Remove alias reference from blob_references (this decrements ref_count in DuckDB)
        let ref_id = format!("alias:{alias_id}");
        let _ = self.catalog.remove_blob_reference(&ref_id);

        // Delete from model_aliases table
        self.catalog
            .delete_alias(&alias_id)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_rm".to_owned(),
                reason: format!("failed to delete alias from catalog: {e}"),
            })?;

        // Remove directory and symlinks from filesystem
        if alias_dir.exists() {
            let _ = fs::remove_dir_all(&alias_dir);
        }

        // Check if manifest is still referenced by any other alias
        let all_aliases = self.catalog.list_aliases().unwrap_or_default();
        let manifest_has_aliases = all_aliases
            .iter()
            .any(|a| a.manifest_digest == alias_rec.manifest_digest);

        if !manifest_has_aliases {
            // Unlink manifest file and remove manifest reference
            let manifest_path = self
                .manifests_dir()
                .join(format!("{}.json", alias_rec.manifest_digest));
            let _ = fs::remove_file(&manifest_path);

            let manifest_ref_id = format!("manifest:{}", alias_rec.manifest_digest);
            let _ = self.catalog.remove_blob_reference(&manifest_ref_id);
            let _ = self.catalog.delete_manifest(&alias_rec.manifest_digest);
        }

        // Clean up empty parent directories under models/
        let name_dir = self.models_dir().join(source).join(namespace).join(name);
        let _ = fs::remove_dir(&name_dir);
        let ns_dir = self.models_dir().join(source).join(namespace);
        let _ = fs::remove_dir(&ns_dir);
        let s_dir = self.models_dir().join(source);
        let _ = fs::remove_dir(&s_dir);

        Ok(true)
    }
}
