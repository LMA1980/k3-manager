//! Staging & Atomic Promotion Pipeline (US-19 Scenario 2, ADR-040).
//!
//! Downloads model weight streams into temporary staging files (`<store_root>/staging/<uuid>.part`)
//! with restricted `0600` permissions, computes streaming SHA-256 digests on the fly, verifies binary
//! format headers with `zerocopy`, and atomically promotes verified files into content-addressed
//! storage (`blobs/sha256/<digest>`) with deduplication.

#![allow(
    clippy::pedantic,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use k3_core::K3Error;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::adapter::{IngestionError, ModelFormat, PullProgress, StagingDownload};
use crate::persistence::{CasCatalog, ModelBlobRecord};
use crate::store::sha256::Sha256;

static STAGING_SEQ: AtomicU64 = AtomicU64::new(1);

/// GGUF binary format header prefix for `zerocopy` verification.
#[repr(C)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufHeaderPrefix {
    /// Format magic bytes: must be `b"GGUF"`.
    pub magic: [u8; 4],
    /// GGUF file format version (e.g. 2, 3).
    pub version: u32,
    /// Number of tensors stored in this file.
    pub tensor_count: u64,
    /// Number of metadata key-value pairs stored in this file.
    pub metadata_kv_count: u64,
}

/// SafeTensors binary header prefix for `zerocopy` length validation.
#[repr(C)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafeTensorsHeaderPrefix {
    /// 8-byte little-endian header byte length.
    pub header_len: u64,
}

/// Pipeline managing staging downloads, checksums, header validation, and atomic promotion.
#[derive(Debug, Clone)]
pub struct StagingPipeline {
    store_root: PathBuf,
}

impl StagingPipeline {
    /// Creates a new staging pipeline anchored under `store_root`.
    pub fn new(store_root: impl Into<PathBuf>) -> Self {
        Self {
            store_root: store_root.into(),
        }
    }

    /// Returns the staging directory (`<store_root>/staging`).
    #[must_use]
    pub fn staging_dir(&self) -> PathBuf {
        self.store_root.join("staging")
    }

    /// Returns the blobs directory (`<store_root>/blobs/sha256`).
    #[must_use]
    pub fn blobs_dir(&self) -> PathBuf {
        self.store_root.join("blobs").join("sha256")
    }

    /// Generates a unique UUID v4 filename (`<uuid>.part`) for staging downloads per PROJECT.md.
    #[must_use]
    pub fn generate_uuid_part_filename() -> String {
        let seq = STAGING_SEQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let mut hasher = Sha256::new();
        hasher.update(&nanos.to_le_bytes());
        hasher.update(&pid.to_le_bytes());
        hasher.update(&seq.to_le_bytes());
        let digest = hasher.finalize();

        let mut uuid_bytes = [0u8; 16];
        uuid_bytes.copy_from_slice(&digest[0..16]);
        uuid_bytes[6] = (uuid_bytes[6] & 0x0f) | 0x40; // UUID v4
        uuid_bytes[8] = (uuid_bytes[8] & 0x3f) | 0x80; // RFC 4122 variant

        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}.part",
            uuid_bytes[0], uuid_bytes[1], uuid_bytes[2], uuid_bytes[3],
            uuid_bytes[4], uuid_bytes[5],
            uuid_bytes[6], uuid_bytes[7],
            uuid_bytes[8], uuid_bytes[9],
            uuid_bytes[10], uuid_bytes[11], uuid_bytes[12], uuid_bytes[13], uuid_bytes[14], uuid_bytes[15]
        )
    }

    /// Creates a new temporary staging file (`<staging>/<uuid>.part`) with restricted 0600 permissions.
    ///
    /// # Errors
    /// Returns [`IngestionError::StagingError`] if directory or file creation fails.
    pub fn create_staging_file(staging_dir: &Path) -> Result<(PathBuf, File), IngestionError> {
        fs::create_dir_all(staging_dir).map_err(|e| {
            IngestionError::StagingError(format!(
                "failed to create staging dir '{}': {e}",
                staging_dir.display()
            ))
        })?;

        let filename = Self::generate_uuid_part_filename();
        let part_path = staging_dir.join(filename);
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&part_path)
            .map_err(|e| {
                IngestionError::StagingError(format!(
                    "failed to open staging file '{}': {e}",
                    part_path.display()
                ))
            })?;

        // Restrictive file permissions: 0600 on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&part_path, fs::Permissions::from_mode(0o600));
        }

        Ok((part_path, file))
    }

    /// Streams incoming reader bytes into `staging_path`, calculating the SHA-256 digest on the fly.
    ///
    /// If streaming fails, an I/O error occurs, or the computed digest does not match
    /// `expected_digest`, `staging_path` is unlinked immediately to keep the store uncontaminated.
    ///
    /// # Errors
    /// Returns [`IngestionError::DigestMismatch`] or [`IngestionError::StagingError`] on failure.
    pub fn stream_to_staging<R: Read>(
        staging_path: &Path,
        reader: &mut R,
        expected_digest: Option<&str>,
        progress_tx: Option<tokio::sync::mpsc::Sender<PullProgress>>,
        total_bytes: Option<u64>,
    ) -> Result<(String, u64), IngestionError> {
        if let Some(parent) = staging_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        if staging_path.is_symlink() {
            let _ = fs::remove_file(staging_path);
            return Err(IngestionError::StagingError(format!(
                "staging path '{}' is a symlink; rejecting to prevent symlink truncation",
                staging_path.display()
            )));
        }

        let mut file = match File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(staging_path)
        {
            Ok(f) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = fs::set_permissions(staging_path, fs::Permissions::from_mode(0o600));
                }
                f
            }
            Err(e) => {
                let _ = fs::remove_file(staging_path);
                return Err(IngestionError::StagingError(format!(
                    "failed to open staging file for writing '{}': {e}",
                    staging_path.display()
                )));
            }
        };

        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 64 * 1024]; // 64KB chunks
        let mut bytes_downloaded: u64 = 0;

        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    hasher.update(&buffer[..n]);
                    if let Err(e) = file.write_all(&buffer[..n]) {
                        let _ = fs::remove_file(staging_path);
                        return Err(IngestionError::StagingError(format!(
                            "write failed on staging file '{}': {e}",
                            staging_path.display()
                        )));
                    }

                    bytes_downloaded = bytes_downloaded.saturating_add(n as u64);

                    if let Some(ref tx) = progress_tx {
                        let percentage = total_bytes.map_or(0.0, |tot| {
                            if tot == 0 {
                                1.0
                            } else {
                                (bytes_downloaded as f32 / tot as f32).min(1.0)
                            }
                        });
                        let _ = tx.try_send(PullProgress {
                            bytes_downloaded,
                            total_bytes,
                            percentage,
                        });
                    }
                }
                Err(e) => {
                    // Drop/network interruption: unlink immediately
                    let _ = fs::remove_file(staging_path);
                    return Err(IngestionError::NetworkError(format!(
                        "download stream interrupted on '{}': {e}",
                        staging_path.display()
                    )));
                }
            }
        }

        if let Err(e) = file.flush() {
            let _ = fs::remove_file(staging_path);
            return Err(IngestionError::StagingError(format!(
                "failed to flush staging file: {e}"
            )));
        }

        let computed_digest = hasher.finalize_hex();

        // Checksum verification
        if let Some(expected) = expected_digest {
            let clean_expected = expected.trim().to_ascii_lowercase();
            if clean_expected != computed_digest {
                let _ = fs::remove_file(staging_path);
                return Err(IngestionError::DigestMismatch {
                    expected: clean_expected,
                    actual: computed_digest,
                });
            }
        }

        Ok((computed_digest, bytes_downloaded))
    }

    /// Validates binary format headers using `zerocopy` before promotion.
    ///
    /// # Errors
    /// Returns [`IngestionError::CorruptedFormat`] if magic or header structure is invalid.
    pub fn verify_format_header(path: &Path, format: &ModelFormat) -> Result<(), IngestionError> {
        let mut file = File::open(path).map_err(|e| {
            IngestionError::StagingError(format!("failed to open file '{}': {e}", path.display()))
        })?;

        let file_len = file.metadata().map_or(0, |m| m.len());

        match format {
            ModelFormat::Gguf => {
                if file_len < size_of::<GgufHeaderPrefix>() as u64 {
                    let _ = fs::remove_file(path);
                    return Err(IngestionError::CorruptedFormat {
                        format: "gguf".to_owned(),
                        reason: format!(
                            "file size ({file_len} bytes) too small for GGUF header ({} bytes required)",
                            size_of::<GgufHeaderPrefix>()
                        ),
                    });
                }

                let mut prefix_bytes = [0u8; size_of::<GgufHeaderPrefix>()];
                file.read_exact(&mut prefix_bytes).map_err(|e| {
                    let _ = fs::remove_file(path);
                    IngestionError::CorruptedFormat {
                        format: "gguf".to_owned(),
                        reason: format!("failed to read GGUF header prefix: {e}"),
                    }
                })?;

                let (prefix, _) =
                    GgufHeaderPrefix::read_from_prefix(&prefix_bytes).map_err(|e| {
                        let _ = fs::remove_file(path);
                        IngestionError::CorruptedFormat {
                            format: "gguf".to_owned(),
                            reason: format!("failed to parse GGUF header with zerocopy: {e}"),
                        }
                    })?;

                if &prefix.magic != b"GGUF" {
                    let _ = fs::remove_file(path);
                    return Err(IngestionError::CorruptedFormat {
                        format: "gguf".to_owned(),
                        reason: format!(
                            "invalid GGUF magic bytes: expected [b'G', b'G', b'U', b'F'], found {:?}",
                            prefix.magic
                        ),
                    });
                }

                if prefix.version < 1 || prefix.version > 4 {
                    let _ = fs::remove_file(path);
                    return Err(IngestionError::CorruptedFormat {
                        format: "gguf".to_owned(),
                        reason: format!("unsupported GGUF version {}", prefix.version),
                    });
                }

                if prefix.tensor_count > 65_536 {
                    let _ = fs::remove_file(path);
                    return Err(IngestionError::CorruptedFormat {
                        format: "gguf".to_owned(),
                        reason: format!(
                            "GGUF tensor_count {} exceeds maximum sanity limit of 65536",
                            prefix.tensor_count
                        ),
                    });
                }

                if prefix.metadata_kv_count > 65_536 {
                    let _ = fs::remove_file(path);
                    return Err(IngestionError::CorruptedFormat {
                        format: "gguf".to_owned(),
                        reason: format!(
                            "GGUF metadata_kv_count {} exceeds maximum sanity limit of 65536",
                            prefix.metadata_kv_count
                        ),
                    });
                }
            }
            ModelFormat::SafeTensors => {
                if file_len < size_of::<SafeTensorsHeaderPrefix>() as u64 {
                    let _ = fs::remove_file(path);
                    return Err(IngestionError::CorruptedFormat {
                        format: "safetensors".to_owned(),
                        reason: format!(
                            "file size ({file_len} bytes) too small for SafeTensors length prefix"
                        ),
                    });
                }

                let mut prefix_bytes = [0u8; size_of::<SafeTensorsHeaderPrefix>()];
                file.read_exact(&mut prefix_bytes).map_err(|e| {
                    let _ = fs::remove_file(path);
                    IngestionError::CorruptedFormat {
                        format: "safetensors".to_owned(),
                        reason: format!("failed to read SafeTensors header length: {e}"),
                    }
                })?;

                let (prefix, _) = SafeTensorsHeaderPrefix::read_from_prefix(&prefix_bytes)
                    .map_err(|e| {
                        let _ = fs::remove_file(path);
                        IngestionError::CorruptedFormat {
                            format: "safetensors".to_owned(),
                            reason: format!(
                                "failed to parse SafeTensors prefix with zerocopy: {e}"
                            ),
                        }
                    })?;

                let header_len = prefix.header_len;
                if header_len == 0 || header_len > 100 * 1024 * 1024 {
                    let _ = fs::remove_file(path);
                    return Err(IngestionError::CorruptedFormat {
                        format: "safetensors".to_owned(),
                        reason: format!("invalid SafeTensors header length: {header_len}"),
                    });
                }

                let total_min_len =
                    (size_of::<SafeTensorsHeaderPrefix>() as u64).saturating_add(header_len);
                if file_len < total_min_len {
                    let _ = fs::remove_file(path);
                    return Err(IngestionError::CorruptedFormat {
                        format: "safetensors".to_owned(),
                        reason: format!(
                            "file truncated: requires at least {total_min_len} bytes for header, got {file_len}"
                        ),
                    });
                }

                let mut json_buf = vec![0u8; header_len as usize];
                file.read_exact(&mut json_buf).map_err(|e| {
                    let _ = fs::remove_file(path);
                    IngestionError::CorruptedFormat {
                        format: "safetensors".to_owned(),
                        reason: format!("failed to read SafeTensors JSON header: {e}"),
                    }
                })?;

                let value: serde_json::Value = serde_json::from_slice(&json_buf).map_err(|e| {
                    let _ = fs::remove_file(path);
                    IngestionError::CorruptedFormat {
                        format: "safetensors".to_owned(),
                        reason: format!("invalid SafeTensors JSON metadata: {e}"),
                    }
                })?;

                if let Some(map) = value.as_object() {
                    for (name, tensor_val) in map {
                        if name == "__metadata__" {
                            continue;
                        }
                        if let Some(offsets) =
                            tensor_val.get("data_offsets").and_then(|v| v.as_array())
                        {
                            if offsets.len() != 2 {
                                let _ = fs::remove_file(path);
                                return Err(IngestionError::CorruptedFormat {
                                    format: "safetensors".to_owned(),
                                    reason: format!(
                                        "tensor '{name}' data_offsets must be a 2-element array"
                                    ),
                                });
                            }
                            let begin = offsets[0].as_u64().ok_or_else(|| {
                                let _ = fs::remove_file(path);
                                IngestionError::CorruptedFormat {
                                    format: "safetensors".to_owned(),
                                    reason: format!(
                                        "tensor '{name}' begin offset is not a valid u64"
                                    ),
                                }
                            })?;
                            let end = offsets[1].as_u64().ok_or_else(|| {
                                let _ = fs::remove_file(path);
                                IngestionError::CorruptedFormat {
                                    format: "safetensors".to_owned(),
                                    reason: format!(
                                        "tensor '{name}' end offset is not a valid u64"
                                    ),
                                }
                            })?;
                            if begin > end {
                                let _ = fs::remove_file(path);
                                return Err(IngestionError::CorruptedFormat {
                                    format: "safetensors".to_owned(),
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
                                    let _ = fs::remove_file(path);
                                    return Err(IngestionError::CorruptedFormat {
                                        format: "safetensors".to_owned(),
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
            ModelFormat::ArrowTensor | ModelFormat::Custom(_) => {
                // Header check not strictly enforced for custom / generic arrow formats
            }
        }

        Ok(())
    }

    /// Atomically promotes a staged file into `<blobs_dir>/<computed_digest>`.
    ///
    /// Handles automatic deduplication: if the blob already exists in the store,
    /// the staging file is unlinked, catalog ref count updated, and the existing
    /// blob path is returned without duplicating bytes (US-18 Scenario 1, US-19).
    ///
    /// # Errors
    /// Returns [`IngestionError::PromotionError`] if atomic promotion or catalog update fails.
    pub fn promote_to_cas(
        staging_path: &Path,
        computed_digest: &str,
        blobs_dir: &Path,
        catalog: Option<&Arc<dyn CasCatalog>>,
    ) -> Result<PathBuf, IngestionError> {
        // Open staging file with read/write access and take exclusive non-blocking advisory lock
        let file = File::options()
            .read(true)
            .write(true)
            .open(staging_path)
            .map_err(|e| {
                let _ = fs::remove_file(staging_path);
                IngestionError::PromotionError(format!(
                    "failed to open staging file '{}': {e}",
                    staging_path.display()
                ))
            })?;

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            // SAFETY: `fd` is an open, valid file descriptor owned by StagingPipeline.
            let lock_res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
            if lock_res != 0 {
                return Err(IngestionError::PromotionError(format!(
                    "staging file '{}' is locked by another process (flock contention)",
                    staging_path.display()
                )));
            }
        }

        fs::create_dir_all(blobs_dir).map_err(|e| {
            let _ = fs::remove_file(staging_path);
            IngestionError::PromotionError(format!(
                "failed to create blobs directory '{}': {e}",
                blobs_dir.display()
            ))
        })?;

        let final_path = blobs_dir.join(computed_digest);
        let size_bytes = fs::metadata(staging_path).map_or(0, |m| m.len());

        if final_path.exists() {
            // Deduplication: unlinked staging file and reuse existing blob
            let _ = fs::remove_file(staging_path);
            #[cfg(unix)]
            {
                use std::os::unix::io::AsRawFd;
                // SAFETY: `file.as_raw_fd()` is a valid open file descriptor.
                let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
            }
            drop(file);
            if let Some(cat) = catalog {
                let _ = cat.increment_blob_ref(computed_digest);
            }
            return Ok(final_path);
        }

        // Atomic POSIX rename into blobs/sha256/<digest>
        fs::rename(staging_path, &final_path).map_err(|e| {
            let _ = fs::remove_file(staging_path);
            IngestionError::PromotionError(format!(
                "failed to atomically rename staging file to '{}': {e}",
                final_path.display()
            ))
        })?;

        // Restrictive read-only permissions: 0444 on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&final_path, fs::Permissions::from_mode(0o444));
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // SAFETY: `file.as_raw_fd()` is a valid open file descriptor.
            let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
        }
        drop(file);

        // Transactional registration in DuckDB CAS catalog
        if let Some(cat) = catalog {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let record = ModelBlobRecord {
                digest: computed_digest.to_owned(),
                size_bytes,
                blob_type: "raw_weights".to_owned(),
                created_at: now,
                last_accessed_at: now,
                ref_count: 1,
            };

            if cat.save_blob(&record).is_err() {
                let _ = cat.increment_blob_ref(computed_digest);
            }
        }

        Ok(final_path)
    }

    /// Verifies and atomically promotes a staging file per the `PROJECT.md` interface contract:
    /// `StagingPipeline::verify_and_promote(part_path: &Path, declared_digest: &str, format: ModelFormat) -> Result<PathBuf, K3Error>`
    ///
    /// # Errors
    /// Returns [`K3Error`] if verification or promotion fails.
    pub fn verify_and_promote(
        part_path: &Path,
        declared_digest: &str,
        format: ModelFormat,
    ) -> Result<PathBuf, K3Error> {
        let clean_declared = declared_digest.trim().to_ascii_lowercase();
        crate::store::cas::validate_sha256_digest(&clean_declared)?;

        // 1. Digest calculation
        let actual_digest = match Sha256::digest_file(part_path) {
            Ok(d) => d,
            Err(e) => {
                let _ = fs::remove_file(part_path);
                return Err(K3Error::CapabilityMismatch {
                    architecture: "sha256".to_owned(),
                    reason: format!("failed to compute staging file digest: {e}"),
                });
            }
        };

        if actual_digest != clean_declared {
            let _ = fs::remove_file(part_path);
            return Err(K3Error::CapabilityMismatch {
                architecture: "sha256".to_owned(),
                reason: format!(
                    "digest verification failed: expected '{clean_declared}', calculated '{actual_digest}'"
                ),
            });
        }

        // 2. Zerocopy format header verification
        Self::verify_format_header(part_path, &format).map_err(K3Error::from)?;

        // 3. Resolve parent blobs directory
        let blobs_dir = if let Some(parent) = part_path.parent() {
            if let Some(grandparent) = parent.parent() {
                grandparent.join("blobs").join("sha256")
            } else {
                parent.join("blobs").join("sha256")
            }
        } else {
            PathBuf::from("/var/lib/k3/blobs/sha256")
        };

        Self::promote_to_cas(part_path, &actual_digest, &blobs_dir, None).map_err(K3Error::from)
    }

    /// Complete helper to stream from reader, verify, and promote.
    ///
    /// # Errors
    /// Returns [`IngestionError`] on failure.
    pub fn ingest_and_promote<R: Read>(
        &self,
        reader: &mut R,
        declared_digest: Option<&str>,
        format: ModelFormat,
        catalog: Option<&Arc<dyn CasCatalog>>,
    ) -> Result<StagingDownload, IngestionError> {
        let (part_path, _) = Self::create_staging_file(&self.staging_dir())?;

        let (computed_digest, size_bytes) =
            Self::stream_to_staging(&part_path, reader, declared_digest, None, None)?;

        Self::verify_format_header(&part_path, &format)?;

        let promoted_path =
            Self::promote_to_cas(&part_path, &computed_digest, &self.blobs_dir(), catalog)?;

        Ok(StagingDownload {
            staging_path: promoted_path,
            declared_digest: declared_digest.map(str::to_owned),
            computed_digest,
            size_bytes,
            format,
        })
    }
}
