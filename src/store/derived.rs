//! Derived Weight Layout Caching & Stale Recipe Detection (US-21, ADR-037, ADR-038).
//!
//! Caches hardware-repacked (Cluster 1 A100 `8x16` `i8` IME tiles) or format-converted weights
//! as derived blobs keyed by `(source_digest, recipe_version, target_layout)`. Enforces
//! operator-controlled retention policies, memory admission accounting, and auto-rebuilding
//! of stale derivations upon runtime upgrades.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use k3_core::K3Error;
use serde::{Deserialize, Serialize};

use super::cas::ContentAddressedStore;
use super::sha256::Sha256;
use crate::persistence::{BlobReferenceRecord, CasCatalog, DerivedBlobRecord, ModelBlobRecord};

/// Operator-configurable retention policy for original source weights after a derived layout is generated (US-21 Scenario 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RetentionPolicy {
    /// Retains original source weights indefinitely in storage alongside derived layouts (default).
    #[default]
    Keep,
    /// Immediately marks original source weights for reclamation once the derived layout is promoted.
    PurgeImmediate,
    /// Retains original source weights for an operator-configured grace period before marking for reclamation.
    PurgeAfterDelay {
        /// Grace period in seconds (e.g. 86400 for 24 hours).
        delay_secs: u64,
    },
}

impl RetentionPolicy {
    /// Returns the canonical policy string identifier.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::PurgeImmediate => "purge_immediate",
            Self::PurgeAfterDelay { .. } => "purge_after_delay",
        }
    }
}

/// Outcome of derived weight resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DerivedResolution {
    /// The derived weight layout was already cached on disk and in `DuckDB` catalog (US-21 Scenario 1).
    Cached {
        /// SHA-256 digest of derived layout blob.
        derived_digest: String,
        /// Filesystem path to cached derived blob.
        derived_path: PathBuf,
    },
    /// A new derived layout was freshly converted and registered.
    Created {
        /// SHA-256 digest of derived layout blob.
        derived_digest: String,
        /// Filesystem path to newly created derived blob.
        derived_path: PathBuf,
    },
}

/// Outcome of evaluating potentially stale derived layouts upon breaking runtime upgrades (US-21 Scenario 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaleRecipeOutcome {
    /// Layout was already valid and up-to-date with active recipe version.
    UpToDate { derived_digest: String },
    /// Layout was stale, but original source weights were present and layout was auto-rebuilt.
    Rebuilt {
        new_derived_digest: String,
        new_derived_path: PathBuf,
    },
    /// Layout was stale and original source weights were purged; requires operator advisory.
    SourcePurgedAdvisory {
        source_digest: String,
        stale_recipe_version: String,
        active_recipe_version: String,
        advisory_message: String,
    },
}

/// Enforces memory admission accounting for simultaneously loaded weight layouts (US-21, ADR-037).
///
/// Refuses load with [`K3Error::InsufficientLockableMemory`] if total required locked bytes
/// exceed available RAM or `RLIMIT_MEMLOCK` bound.
///
/// # Errors
/// Returns [`K3Error::InsufficientLockableMemory`] on ceiling violation.
pub fn check_derived_memory_admission(
    current_locked_bytes: u64,
    additional_layout_bytes: u64,
    kv_arena_bytes: u64,
    available_ram_bytes: u64,
    rlimit_memlock_bytes: u64,
) -> Result<(), K3Error> {
    let total_required = current_locked_bytes
        .saturating_add(additional_layout_bytes)
        .saturating_add(kv_arena_bytes);

    if (rlimit_memlock_bytes > 0 && total_required > rlimit_memlock_bytes)
        || (available_ram_bytes > 0 && total_required > available_ram_bytes)
    {
        return Err(K3Error::InsufficientLockableMemory {
            required_bytes: total_required,
            weight_bytes: current_locked_bytes.saturating_add(additional_layout_bytes),
            kv_bytes: kv_arena_bytes,
            available_bytes: available_ram_bytes,
            rlimit_memlock_bytes,
        });
    }

    Ok(())
}

/// Manager for derived weight layout caching, retention lifecycle, and stale recipe detection.
#[derive(Debug, Clone)]
pub struct DerivedBlobManager {
    store: Arc<ContentAddressedStore>,
    catalog: Arc<dyn CasCatalog>,
}

impl DerivedBlobManager {
    /// Constructs a new derived blob manager bound to a content-addressed store.
    #[must_use]
    pub fn new(store: Arc<ContentAddressedStore>) -> Self {
        let catalog = store.catalog().clone();
        Self { store, catalog }
    }

    /// Resolves or builds a hardware-repacked or format-converted derived blob (US-21 Scenario 1 & 2).
    ///
    /// Keyed by `(source_digest, recipe_version, target_layout)`.
    ///
    /// If already cached and valid, directly returns [`DerivedResolution::Cached`].
    /// Otherwise, invokes `converter` to populate a staging file, promotes it to `blobs/sha256/<derived_digest>`,
    /// registers in `DuckDB`, and applies the operator's configured `retention` policy to the source weights.
    ///
    /// # Errors
    /// Returns [`K3Error`] on conversion failure, catalog error, or missing source weights.
    #[allow(clippy::too_many_lines)]
    pub fn get_or_create_derived_layout<F>(
        &self,
        source_digest: &str,
        recipe_version: &str,
        target_layout: &str,
        retention: RetentionPolicy,
        converter: F,
    ) -> Result<DerivedResolution, K3Error>
    where
        F: FnOnce(&Path, &Path) -> Result<u64, K3Error>,
    {
        super::cas::validate_sha256_digest(source_digest)?;
        let source_clean = source_digest.to_owned();

        // 1. Check if valid derived blob already exists in DuckDB catalog and filesystem (Scenario 1)
        if let Ok(Some(existing)) =
            self.catalog
                .get_derived_blob(source_digest, recipe_version, target_layout)
        {
            if existing.status == "valid" {
                let derived_path = self.store.blobs_dir().join(&existing.derived_digest);
                if derived_path.exists() {
                    // Update access timestamp
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let _ = self
                        .catalog
                        .update_blob_access(&existing.derived_digest, now);

                    return Ok(DerivedResolution::Cached {
                        derived_digest: existing.derived_digest,
                        derived_path,
                    });
                }
            }
        }

        // 2. Validate that original source weights blob exists
        super::cas::validate_sha256_digest(source_digest)?;
        let blobs_dir = self.store.blobs_dir();
        let source_path = blobs_dir.join(source_digest);
        if !source_path.exists() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "derived_weights".to_owned(),
                reason: format!(
                    "original source weights blob '{}' not found in store; cannot derive layout",
                    source_path.display()
                ),
            });
        }
        let canon_blobs = fs::canonicalize(&blobs_dir).unwrap_or(blobs_dir);
        let canon_source = fs::canonicalize(&source_path).unwrap_or_else(|_| source_path.clone());
        if !canon_source.starts_with(&canon_blobs) {
            return Err(K3Error::CapabilityMismatch {
                architecture: "derived_weights".to_owned(),
                reason: format!("path traversal detected in source_digest '{source_digest}'"),
            });
        }

        // 3. Create staging file for conversion
        let (staging_path, mut staging_file) = self.store.create_staging_file()?;

        // 4. Run conversion
        let derived_bytes = match converter(&source_path, &staging_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                let _ = fs::remove_file(&staging_path);
                return Err(e);
            }
        };

        // Flush staging file
        staging_file
            .flush()
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "derived_weights".to_owned(),
                reason: format!("failed to flush staging derived weights: {e}"),
            })?;
        drop(staging_file);

        // 5. Compute SHA-256 digest of converted derived weights
        let derived_digest = Sha256::digest_file(&staging_path).map_err(|e| {
            let _ = fs::remove_file(&staging_path);
            K3Error::CapabilityMismatch {
                architecture: "derived_weights".to_owned(),
                reason: format!("failed to hash derived staging weights: {e}"),
            }
        })?;

        // 6. Atomically promote staging file to blobs/sha256/<derived_digest>
        let final_derived_path = self.store.blobs_dir().join(&derived_digest);
        if final_derived_path.exists() {
            let _ = fs::remove_file(&staging_path);
        } else {
            fs::rename(&staging_path, &final_derived_path).map_err(|e| {
                let _ = fs::remove_file(&staging_path);
                K3Error::CapabilityMismatch {
                    architecture: "derived_weights".to_owned(),
                    reason: format!(
                        "failed to atomically promote derived weights to '{}': {e}",
                        final_derived_path.display()
                    ),
                }
            })?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(&final_derived_path, fs::Permissions::from_mode(0o444));
            }
        }

        // 7. Register derived blob in DuckDB catalog
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let blob_rec = ModelBlobRecord {
            digest: derived_digest.clone(),
            size_bytes: derived_bytes,
            blob_type: "derived_weights".to_owned(),
            created_at: now,
            last_accessed_at: now,
            ref_count: 1,
        };
        self.catalog
            .save_blob(&blob_rec)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "derived_catalog".to_owned(),
                reason: format!("failed to save derived blob record: {e}"),
            })?;

        let derived_rec = DerivedBlobRecord {
            derived_digest: derived_digest.clone(),
            source_digest: source_clean.clone(),
            recipe_version: recipe_version.to_owned(),
            target_layout: target_layout.to_owned(),
            status: "valid".to_owned(),
            created_at: now,
        };
        self.catalog
            .save_derived_blob(&derived_rec)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "derived_catalog".to_owned(),
                reason: format!("failed to save derived association: {e}"),
            })?;

        // Add reference linking derived blob to source
        let ref_rec = BlobReferenceRecord {
            ref_id: format!("derived:{derived_digest}"),
            blob_digest: derived_digest.clone(),
            ref_type: "derived".to_owned(),
            ref_source: source_clean.clone(),
            created_at: now,
        };
        let _ = self.catalog.add_blob_reference(&ref_rec);

        // 8. Apply operator retention policy (Scenario 2)
        match retention {
            RetentionPolicy::Keep => {
                // Keep source blob indefinitely; no action needed
            }
            RetentionPolicy::PurgeImmediate => {
                // Mark source blob for immediate GC reclamation if unreferenced by other manifests
                let purge_key = format!("purge_immediate:{source_clean}");
                let _ = self.catalog.set_storage_config(&purge_key, "true");

                // Release manifest and alias references from blob_references
                if let Ok(refs) = self.catalog.list_blob_references(&source_clean) {
                    for r in refs {
                        if r.ref_type == "manifest" || r.ref_type == "alias" {
                            let _ = self.catalog.remove_blob_reference(&r.ref_id);
                        }
                    }
                }
                let _ = self.catalog.decrement_blob_ref(&source_clean);
            }
            RetentionPolicy::PurgeAfterDelay { delay_secs } => {
                let purge_timestamp = now.saturating_add(delay_secs);
                let purge_key = format!("purge_after:{source_clean}");
                let _ = self
                    .catalog
                    .set_storage_config(&purge_key, &purge_timestamp.to_string());
            }
        }

        Ok(DerivedResolution::Created {
            derived_digest,
            derived_path: final_derived_path,
        })
    }

    /// Evaluates existing derived weight layouts against the active runtime recipe version (US-21 Scenario 3).
    ///
    /// Upon breaking recipe upgrades:
    /// - Flags outdated derivations as `status = 'stale'`.
    /// - If original source weights are still present, automatically rebuilds the layout.
    /// - If original source weights were purged, returns [`StaleRecipeOutcome::SourcePurgedAdvisory`]
    ///   with an actionable operator advisory notice.
    ///
    /// # Errors
    /// Returns [`K3Error`] on rebuild error or catalog failure.
    pub fn detect_and_handle_stale_recipe<F>(
        &self,
        source_digest: &str,
        active_recipe_version: &str,
        target_layout: &str,
        model_name: &str,
        rebuilder: F,
    ) -> Result<StaleRecipeOutcome, K3Error>
    where
        F: FnOnce(&Path, &Path) -> Result<u64, K3Error>,
    {
        super::cas::validate_sha256_digest(source_digest)?;
        let source_clean = source_digest.to_owned();

        // Query existing derived blobs for this source and layout
        let all_derived = self
            .catalog
            .list_derived_for_source(source_digest)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "derived_stale".to_owned(),
                reason: format!("failed to list derived blobs for source: {e}"),
            })?;

        let matching_layout: Vec<DerivedBlobRecord> = all_derived
            .into_iter()
            .filter(|d| d.target_layout == target_layout)
            .collect();

        // If a valid derivation exists for the active recipe version, we are up to date
        if let Some(up_to_date) = matching_layout
            .iter()
            .find(|d| d.recipe_version == active_recipe_version && d.status == "valid")
        {
            return Ok(StaleRecipeOutcome::UpToDate {
                derived_digest: up_to_date.derived_digest.clone(),
            });
        }

        // Check if there is an outdated derivation
        let Some(stale_rec) = matching_layout
            .iter()
            .find(|d| d.recipe_version != active_recipe_version)
        else {
            // No prior derivation at all: not stale, simply unbuilt
            return Ok(StaleRecipeOutcome::UpToDate {
                derived_digest: String::new(),
            });
        };

        // Flag existing derived record as stale in DuckDB
        let _ = self
            .catalog
            .update_derived_status(&stale_rec.derived_digest, "stale");

        // Check whether the original source weights are still present on disk
        let source_path = self.store.blobs_dir().join(&source_clean);
        if !source_path.starts_with(self.store.blobs_dir()) {
            return Err(K3Error::CapabilityMismatch {
                architecture: "derived_stale".to_owned(),
                reason: format!("path traversal detected in source_digest '{source_digest}'"),
            });
        }
        if source_path.exists() {
            // Auto-rebuild the derived blob for the new recipe version (US-21 Scenario 3)
            let resolution = self.get_or_create_derived_layout(
                &source_clean,
                active_recipe_version,
                target_layout,
                RetentionPolicy::Keep,
                rebuilder,
            )?;

            match resolution {
                DerivedResolution::Cached {
                    derived_digest,
                    derived_path,
                }
                | DerivedResolution::Created {
                    derived_digest,
                    derived_path,
                } => Ok(StaleRecipeOutcome::Rebuilt {
                    new_derived_digest: derived_digest,
                    new_derived_path: derived_path,
                }),
            }
        } else {
            // Original source weights were purged; prompt operator to re-acquire (US-21 Scenario 3)
            let advisory = format!(
                "Derived weight layout for model '{model_name}' (source: {source_clean}) is stale \
                 (built for recipe version {}, current runtime is {active_recipe_version}), \
                 and original source weights have been purged. Please re-acquire the source model.",
                stale_rec.recipe_version
            );

            Ok(StaleRecipeOutcome::SourcePurgedAdvisory {
                source_digest: source_clean,
                stale_recipe_version: stale_rec.recipe_version.clone(),
                active_recipe_version: active_recipe_version.to_owned(),
                advisory_message: advisory,
            })
        }
    }
}
