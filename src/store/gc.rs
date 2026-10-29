//! Safe Storage Garbage Collection Dual-Gate & `who-uses` Inspection (US-22, ADR-039, ADR-040).
//!
//! Enforces decoupled, reference-counted storage reclamation with a mandatory two-gate safety protocol:
//! 1. Gate 1: Non-blocking exclusive advisory lock (`flock(LOCK_EX | LOCK_NB)`).
//! 2. Gate 2: Running process memory mapping scan (`/proc/[pid]/maps`) verifying zero active mappings.
//!
//! Works offline without requiring `k3-manager` daemon. Emits high-water mark warnings without unapproved deletion.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use k3_core::K3Error;
use serde::{Deserialize, Serialize};

use super::cas::ContentAddressedStore;
use crate::persistence::CasCatalog;

/// Details of an active process virtual memory mapping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessMappingInfo {
    /// Process PID.
    pub pid: u32,
    /// Process executable name or command.
    pub comm: String,
    /// Virtual address range (e.g., "0x7f8a10000000-0x7f8a90000000").
    pub address_range: String,
    /// Virtual memory permissions (e.g., "r--p").
    pub permissions: String,
    /// Backing file path as reported by kernel maps.
    pub mapped_path: String,
}

/// Administrative usage report inspecting active locks and memory mappings for a blob (US-22 Scenario 3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobUsageInfo {
    /// SHA-256 digest of inspected blob.
    pub digest: String,
    /// Absolute filesystem path to blob on disk.
    pub file_path: PathBuf,
    /// Size of blob in bytes.
    pub size_bytes: u64,
    /// Active reference count in `DuckDB` catalog (u32 per AGENTS.md §9).
    pub ref_count: u32,
    /// Referencing sources (manifests, aliases, derivations).
    pub referencing_sources: Vec<String>,
    /// Whether an active compute worker holds a shared advisory lock (`flock(LOCK_SH)`).
    pub gate1_shared_lock_held: bool,
    /// Active processes currently mapping this blob into memory.
    pub gate2_active_mappings: Vec<ProcessMappingInfo>,
}

impl BlobUsageInfo {
    /// Returns `true` if the blob is actively in use by any process.
    #[must_use]
    pub fn is_in_use(&self) -> bool {
        self.gate1_shared_lock_held || !self.gate2_active_mappings.is_empty()
    }
}

/// Storage usage warning emitted when consumption exceeds configured thresholds (US-22 Scenario 2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HighWaterMarkWarning {
    /// Current storage consumption in bytes.
    pub current_bytes: u64,
    /// Configured high-water mark ceiling in bytes.
    pub threshold_bytes: u64,
    /// Usage percentage of configured threshold (0.0 .. 1.0+).
    pub usage_ratio: f64,
    /// Actionable operator warning message.
    pub warning_message: String,
}

/// Detailed execution report for a garbage collection sweep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GcReport {
    /// Number of candidate blobs examined.
    pub examined_candidates: usize,
    /// Digests of blobs safely reclaimed.
    pub reclaimed_blobs: Vec<String>,
    /// Aggregate bytes reclaimed from storage.
    pub reclaimed_bytes: u64,
    /// Digests of candidate blobs skipped because Gate 1 (flock) failed.
    pub skipped_in_use_gate1: Vec<String>,
    /// Digests of candidate blobs skipped because Gate 2 (maps scan) found active mappings.
    pub skipped_mapped_gate2: Vec<String>,
    /// Whether this execution was a dry run.
    pub dry_run: bool,
}

/// Trait abstracting kernel process virtual memory mapping inspection.
pub trait ProcessMapsScanner: Send + Sync + std::fmt::Debug {
    /// Scans running processes for virtual memory mappings referencing `canonical_path`.
    ///
    /// # Errors
    /// Returns [`io::Error`] if scanning fails.
    fn scan_mappings_for_path(
        &self,
        canonical_path: &Path,
    ) -> Result<Vec<ProcessMappingInfo>, io::Error>;
}

/// Real Linux `/proc/[pid]/maps` scanner implementation (ADR-040).
#[derive(Debug, Clone, Default)]
pub struct RealProcMapsScanner;

impl ProcessMapsScanner for RealProcMapsScanner {
    fn scan_mappings_for_path(
        &self,
        canonical_path: &Path,
    ) -> Result<Vec<ProcessMappingInfo>, io::Error> {
        let path_str = canonical_path.to_string_lossy();
        let mut results = Vec::new();

        let proc_dir = Path::new("/proc");
        if !proc_dir.exists() {
            // Non-Linux or unmounted /proc; returns empty list
            return Ok(results);
        }

        let Ok(entries) = fs::read_dir(proc_dir) else {
            return Ok(results);
        };

        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let name_str = file_name.to_string_lossy();
            if let Ok(pid) = name_str.parse::<u32>() {
                let maps_path = entry.path().join("maps");
                if let Ok(file) = File::open(&maps_path) {
                    let reader = BufReader::new(file);
                    for line_res in reader.lines().map_while(Result::ok) {
                        let parts: Vec<&str> = line_res.split_whitespace().collect();
                        if parts.len() >= 6 {
                            let mapped = parts[5..].join(" ");
                            let mapped_clean = mapped.strip_suffix(" (deleted)").unwrap_or(&mapped);
                            if mapped_clean == path_str.as_ref()
                                || mapped == path_str.as_ref()
                                || line_res.contains(path_str.as_ref())
                            {
                                let comm_path = entry.path().join("comm");
                                let comm = fs::read_to_string(&comm_path)
                                    .map_or_else(|_| "unknown".to_owned(), |s| s.trim().to_owned());

                                results.push(ProcessMappingInfo {
                                    pid,
                                    comm,
                                    address_range: parts[0].to_owned(),
                                    permissions: parts[1].to_owned(),
                                    mapped_path: mapped,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(results)
    }
}

/// Mock `/proc/[pid]/maps` scanner for deterministic cross-platform tests.
#[derive(Debug, Default)]
pub struct MockProcMapsScanner {
    simulated_mappings: Mutex<HashMap<PathBuf, Vec<ProcessMappingInfo>>>,
}

impl MockProcMapsScanner {
    /// Constructs a new mock maps scanner.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a simulated memory mapping for `path`.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    pub fn add_mapping(&self, path: impl AsRef<Path>, mapping: ProcessMappingInfo) {
        let mut map = self.simulated_mappings.lock().unwrap();
        let p = path.as_ref().to_path_buf();
        if let Ok(canon) = fs::canonicalize(&p) {
            if canon != p {
                map.entry(canon).or_default().push(mapping.clone());
            }
        }
        map.entry(p).or_default().push(mapping);
    }

    /// Clears all simulated mappings.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    pub fn clear(&self) {
        let mut map = self.simulated_mappings.lock().unwrap();
        map.clear();
    }
}

impl ProcessMapsScanner for MockProcMapsScanner {
    fn scan_mappings_for_path(
        &self,
        canonical_path: &Path,
    ) -> Result<Vec<ProcessMappingInfo>, io::Error> {
        let map = self.simulated_mappings.lock().unwrap();
        if let Some(res) = map.get(canonical_path) {
            return Ok(res.clone());
        }
        let canon_in =
            fs::canonicalize(canonical_path).unwrap_or_else(|_| canonical_path.to_path_buf());
        if let Some(res) = map.get(&canon_in) {
            return Ok(res.clone());
        }
        for (k, v) in map.iter() {
            if k == canonical_path || k == &canon_in {
                return Ok(v.clone());
            }
            if let Ok(canon_k) = fs::canonicalize(k) {
                if canon_k == canonical_path || canon_k == canon_in {
                    return Ok(v.clone());
                }
            }
        }
        Ok(Vec::new())
    }
}

/// Storage Garbage Collector enforcing the dual-gate safety check (ADR-040).
#[derive(Debug, Clone)]
pub struct StorageGarbageCollector {
    store: Arc<ContentAddressedStore>,
    catalog: Arc<dyn CasCatalog>,
    scanner: Arc<dyn ProcessMapsScanner>,
    high_watermark_bytes: Option<u64>,
}

impl StorageGarbageCollector {
    /// Constructs a new garbage collector with the real `/proc/[pid]/maps` scanner.
    #[must_use]
    pub fn new(store: Arc<ContentAddressedStore>) -> Self {
        let catalog = store.catalog().clone();
        Self {
            store,
            catalog,
            scanner: Arc::new(RealProcMapsScanner),
            high_watermark_bytes: None,
        }
    }

    /// Constructs a garbage collector with a custom or mock process mapping scanner.
    #[must_use]
    pub fn with_scanner(
        store: Arc<ContentAddressedStore>,
        scanner: Arc<dyn ProcessMapsScanner>,
    ) -> Self {
        let catalog = store.catalog().clone();
        Self {
            store,
            catalog,
            scanner,
            high_watermark_bytes: None,
        }
    }

    /// Sets the high-water mark storage threshold in bytes.
    pub fn set_high_watermark_bytes(&mut self, threshold_bytes: Option<u64>) {
        self.high_watermark_bytes = threshold_bytes;
    }

    /// Checks current storage usage against the configured high-water mark threshold (US-22 Scenario 2).
    ///
    /// Emits a structured warning if usage exceeds threshold, without performing unapproved deletion.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on catalog query failure.
    #[allow(clippy::cast_precision_loss)]
    pub fn check_watermark(&self) -> Result<Option<HighWaterMarkWarning>, K3Error> {
        let Some(threshold) = self.high_watermark_bytes else {
            return Ok(None);
        };

        let current_bytes =
            self.catalog
                .total_blob_bytes()
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "cas_watermark".to_owned(),
                    reason: format!("failed to calculate total blob bytes: {e}"),
                })?;

        if current_bytes >= threshold {
            let usage_ratio = if threshold > 0 {
                current_bytes as f64 / threshold as f64
            } else {
                1.0
            };
            let warning = HighWaterMarkWarning {
                current_bytes,
                threshold_bytes: threshold,
                usage_ratio,
                warning_message: format!(
                    "Model storage consumption ({current_bytes} bytes) exceeds configured high-water mark \
                     ({threshold} bytes; {:.1}%). Explicit garbage collection is recommended.",
                    usage_ratio * 100.0
                ),
            };
            Ok(Some(warning))
        } else {
            Ok(None)
        }
    }

    /// Administrative query inspecting active holders and mappings for a given blob (US-22 Scenario 3).
    ///
    /// Returns lock status, active memory mappings, and referencing sources even if the daemon is offline.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if blob does not exist or inspection fails.
    pub fn who_uses(&self, digest: &str) -> Result<BlobUsageInfo, K3Error> {
        super::cas::validate_sha256_digest(digest)?;
        let clean = digest.to_owned();

        let blobs_dir = self.store.blobs_dir();
        let blob_path = blobs_dir.join(digest);
        if !blob_path.exists() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_gc".to_owned(),
                reason: format!("blob '{}' not found in store", blob_path.display()),
            });
        }

        let canon_blobs = fs::canonicalize(&blobs_dir).unwrap_or(blobs_dir);
        let canon_blob_path = fs::canonicalize(&blob_path).unwrap_or_else(|_| blob_path.clone());
        if !canon_blob_path.starts_with(&canon_blobs) {
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_gc".to_owned(),
                reason: format!("path traversal detected in digest '{digest}'"),
            });
        }

        let size_bytes = fs::metadata(&blob_path).map_or(0, |m| m.len());

        // Fetch catalog metadata
        let (ref_count, referencing_sources) = match self.catalog.get_blob(&clean) {
            Ok(Some(b)) => {
                let refs = self
                    .catalog
                    .list_blob_references(&clean)
                    .unwrap_or_default();
                let sources = refs.into_iter().map(|r| r.ref_source).collect();
                (b.ref_count, sources)
            }
            _ => (0, Vec::new()),
        };

        // Gate 1 check: probe non-blocking exclusive lock
        let gate1_shared_lock_held = match File::open(&blob_path) {
            Ok(file) => {
                let fd = file.as_raw_fd();
                // SAFETY: flock on valid open file descriptor with LOCK_EX | LOCK_NB
                let res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
                if res != 0 {
                    true
                } else {
                    // SAFETY: Releasing lock on valid open file descriptor
                    unsafe { libc::flock(fd, libc::LOCK_UN) };
                    false
                }
            }
            Err(_) => false,
        };

        // Gate 2 check: scan /proc/[pid]/maps
        let canonical_blob_path =
            fs::canonicalize(&blob_path).unwrap_or_else(|_| blob_path.clone());
        let gate2_active_mappings = self
            .scanner
            .scan_mappings_for_path(&canonical_blob_path)
            .unwrap_or_default();

        Ok(BlobUsageInfo {
            digest: clean,
            file_path: blob_path,
            size_bytes,
            ref_count,
            referencing_sources,
            gate1_shared_lock_held,
            gate2_active_mappings,
        })
    }

    /// Executes safe garbage collection over unreferenced (`ref_count == 0`) weight blobs (US-22).
    ///
    /// Verifies both Gate 1 (`flock`) and Gate 2 (`/proc/*/maps`) before unlinking any file.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on catalog or filesystem error.
    pub fn run_gc(&self, dry_run: bool) -> Result<GcReport, K3Error> {
        let mut report = GcReport {
            dry_run,
            ..Default::default()
        };

        // 1. Recalculate and query candidates with ref_count == 0
        let _ = self.catalog.recalculate_ref_counts();
        let candidates =
            self.catalog
                .list_unreferenced_blobs()
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "cas_gc".to_owned(),
                    reason: format!("failed to list unreferenced blobs: {e}"),
                })?;

        report.examined_candidates = candidates.len();

        for candidate in candidates {
            if super::cas::validate_sha256_digest(&candidate.digest).is_err() {
                continue;
            }

            let blob_path = self.store.blobs_dir().join(&candidate.digest);
            let canonical_blob_path =
                fs::canonicalize(&blob_path).unwrap_or_else(|_| blob_path.clone());
            if !blob_path.exists() {
                // File already gone from disk, clean up catalog row
                if !dry_run {
                    let _ = self.catalog.delete_blob(&candidate.digest);
                }
                continue;
            }

            // Gate 1: Advisory File Lock Check
            let Ok(file) = File::open(&blob_path) else {
                report.skipped_in_use_gate1.push(candidate.digest.clone());
                continue;
            };

            let fd = file.as_raw_fd();
            // SAFETY: flock on open fd with LOCK_EX | LOCK_NB
            let flock_res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
            if flock_res != 0 {
                // Active compute worker holds shared lock; skip blob
                report.skipped_in_use_gate1.push(candidate.digest.clone());
                continue;
            }

            // While holding LOCK_EX on `file`, verify Gate 2 using canonical path:
            let mappings = self
                .scanner
                .scan_mappings_for_path(&canonical_blob_path)
                .unwrap_or_default();

            if !mappings.is_empty() {
                // Actively mapped in process memory; skip blob
                // SAFETY: Release lock on open fd
                unsafe { libc::flock(fd, libc::LOCK_UN) };
                report.skipped_mapped_gate2.push(candidate.digest.clone());
                continue;
            }

            // Both gates passed! Safe to delete.
            if dry_run {
                // SAFETY: Release lock on open fd
                unsafe { libc::flock(fd, libc::LOCK_UN) };
            } else {
                // Inode check before unlinking to eliminate TOCTOU file replacement
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    if let (Ok(m_fd), Ok(m_path)) = (file.metadata(), fs::metadata(&blob_path)) {
                        if m_fd.dev() != m_path.dev() || m_fd.ino() != m_path.ino() {
                            // SAFETY: Release lock on open fd
                            unsafe { libc::flock(fd, libc::LOCK_UN) };
                            continue;
                        }
                    }
                }

                // Unlink file while holding open descriptor (eliminating TOCTOU race)
                if fs::remove_file(&blob_path).is_err() {
                    // SAFETY: Release lock on open fd
                    unsafe { libc::flock(fd, libc::LOCK_UN) };
                    continue;
                }
                // Release lock and remove row from catalog
                // SAFETY: Release lock on open fd
                unsafe { libc::flock(fd, libc::LOCK_UN) };
                let _ = self.catalog.delete_blob(&candidate.digest);
            }

            report.reclaimed_blobs.push(candidate.digest.clone());
            report.reclaimed_bytes = report.reclaimed_bytes.saturating_add(candidate.size_bytes);
        }

        Ok(report)
    }
}
