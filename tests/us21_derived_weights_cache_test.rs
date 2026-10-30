//! Scenario and unit verification for US-21: Derived Weight Layout Caching & Stale Recipe Detection.
//!
//! Verifies:
//! - Scenario 1: Reuse of cached derived weight layouts across loads without repeating conversion.
//! - Scenario 2: Operator-controlled retention of original source weights (`Keep`, `PurgeImmediate`, `PurgeAfterDelay`).
//! - Scenario 3: Stale derived blob detection and auto-rebuild or operator advisory notification.
//! - Memory admission accounting for simultaneously loaded weight layouts.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use k3_core::K3Error;
use k3_manager::persistence::{CasCatalog, DuckDbPersistenceStore};
use k3_manager::store::{
    ContentAddressedStore, DerivedBlobManager, DerivedResolution, RetentionPolicy, Sha256,
    StaleRecipeOutcome, check_derived_memory_admission,
};

static GLOBAL_SEQ: AtomicU64 = AtomicU64::new(100);

struct TestTempDir {
    path: PathBuf,
}

impl TestTempDir {
    fn new(prefix: &str) -> Self {
        let id = GLOBAL_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{id}", std::process::id()));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn create_store_with_source_blob(
    source_content: &[u8],
) -> (
    Arc<ContentAddressedStore>,
    DerivedBlobManager,
    String,
    TestTempDir,
) {
    let temp_dir = TestTempDir::new("k3_test_us21");
    let catalog = DuckDbPersistenceStore::open_in_memory().expect("open catalog");
    let catalog_arc: Arc<dyn CasCatalog> = Arc::new(catalog);

    let store = Arc::new(
        ContentAddressedStore::open(temp_dir.path(), catalog_arc.clone()).expect("open store"),
    );
    let manager = DerivedBlobManager::new(store.clone());

    // Ingest original source weights blob
    let source_digest = Sha256::digest(source_content);
    let (staging, mut f) = store.create_staging_file().expect("staging");
    f.write_all(source_content).expect("write");
    drop(f);

    store
        .promote_staged_blob(&staging, &source_digest, "raw")
        .expect("promote source");

    (store, manager, source_digest, temp_dir)
}

#[test]
fn test_scenario1_reuse_cached_derived_layout_across_loads() {
    let source_bytes = b"original_raw_weights_12345";
    let (store, manager, source_digest, _tmp) = create_store_with_source_blob(source_bytes);

    let mut conversion_count = 0;
    let converter = |src: &Path, dst: &Path| -> Result<u64, K3Error> {
        conversion_count += 1;
        let original = fs::read(src).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "test".to_owned(),
            reason: e.to_string(),
        })?;
        let mut repacked = Vec::new();
        repacked.extend_from_slice(b"IME_REPACKED_TILE_8x16:");
        repacked.extend_from_slice(&original);
        fs::write(dst, &repacked).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "test".to_owned(),
            reason: e.to_string(),
        })?;
        Ok(repacked.len() as u64)
    };

    // 1. First Load: triggers conversion and caches the derived blob
    let resolution1 = manager
        .get_or_create_derived_layout(
            &source_digest,
            "1.0.0",
            "ime_tile_8x16",
            RetentionPolicy::Keep,
            converter,
        )
        .expect("first load creates derived layout");

    assert_eq!(conversion_count, 1, "conversion must run on first load");
    let (derived_digest1, derived_path1) = match resolution1 {
        DerivedResolution::Created {
            derived_digest,
            derived_path,
        } => (derived_digest, derived_path),
        DerivedResolution::Cached { .. } => {
            panic!("first load must return DerivedResolution::Created")
        }
    };
    assert!(derived_path1.exists());

    // 2. Second Load on same runtime version: directly reuses cached derived blob
    let mut second_conversion_ran = false;
    let dummy_converter = |_src: &Path, _dst: &Path| -> Result<u64, K3Error> {
        second_conversion_ran = true;
        panic!("converter must not be invoked on cached reload!");
    };

    let resolution2 = manager
        .get_or_create_derived_layout(
            &source_digest,
            "1.0.0",
            "ime_tile_8x16",
            RetentionPolicy::Keep,
            dummy_converter,
        )
        .expect("second load reuses cached layout");

    assert!(
        !second_conversion_ran,
        "converter was not run on second load"
    );
    match resolution2 {
        DerivedResolution::Cached {
            derived_digest,
            derived_path,
        } => {
            assert_eq!(derived_digest, derived_digest1);
            assert_eq!(derived_path, derived_path1);
        }
        DerivedResolution::Created { .. } => {
            panic!("second load must return DerivedResolution::Cached")
        }
    }

    // Assert exactly 2 physical blobs exist: 1 source + 1 derived
    let physical_blobs: Vec<_> = fs::read_dir(store.blobs_dir())
        .expect("read dir")
        .flatten()
        .collect();
    assert_eq!(physical_blobs.len(), 2);
}

#[test]
fn test_scenario2_operator_controlled_retention_policies() {
    let source_bytes = b"source_weights_for_retention_policy_test";

    // 1. RetentionPolicy::Keep
    {
        let (store, manager, source_digest, _tmp) = create_store_with_source_blob(source_bytes);
        let res = manager
            .get_or_create_derived_layout(
                &source_digest,
                "1.0.0",
                "ime_tile_8x16",
                RetentionPolicy::Keep,
                |_src, dst| {
                    fs::write(dst, b"tiled").unwrap();
                    Ok(5)
                },
            )
            .expect("derive keep");
        assert!(matches!(res, DerivedResolution::Created { .. }));

        // Original source weights still exist
        assert!(store.blobs_dir().join(&source_digest).exists());
        let purge_flag = store
            .catalog()
            .get_storage_config(&format!("purge_immediate:{source_digest}"))
            .unwrap();
        assert!(
            purge_flag.is_none(),
            "Keep policy must not mark for immediate purge"
        );
    }

    // 2. RetentionPolicy::PurgeImmediate
    {
        let (store, manager, source_digest, _tmp) = create_store_with_source_blob(source_bytes);
        let res = manager
            .get_or_create_derived_layout(
                &source_digest,
                "1.0.0",
                "ime_tile_8x16",
                RetentionPolicy::PurgeImmediate,
                |_src, dst| {
                    fs::write(dst, b"tiled").unwrap();
                    Ok(5)
                },
            )
            .expect("derive purge_immediate");
        assert!(matches!(res, DerivedResolution::Created { .. }));

        // Marked in storage_config for immediate GC reclamation
        let purge_flag = store
            .catalog()
            .get_storage_config(&format!("purge_immediate:{source_digest}"))
            .unwrap();
        assert_eq!(purge_flag, Some("true".to_owned()));
    }

    // 3. RetentionPolicy::PurgeAfterDelay
    {
        let (store, manager, source_digest, _tmp) = create_store_with_source_blob(source_bytes);
        let delay = 3600; // 1 hour
        let res = manager
            .get_or_create_derived_layout(
                &source_digest,
                "1.0.0",
                "ime_tile_8x16",
                RetentionPolicy::PurgeAfterDelay { delay_secs: delay },
                |_src, dst| {
                    fs::write(dst, b"tiled").unwrap();
                    Ok(5)
                },
            )
            .expect("derive purge_after_delay");
        assert!(matches!(res, DerivedResolution::Created { .. }));

        let purge_ts_str = store
            .catalog()
            .get_storage_config(&format!("purge_after:{source_digest}"))
            .unwrap()
            .expect("must record purge_after timestamp");
        let purge_ts: u64 = purge_ts_str.parse().expect("valid timestamp");
        assert!(purge_ts > 0);
    }
}

#[test]
fn test_scenario3_stale_recipe_detection_and_auto_rebuild() {
    let source_bytes = b"source_weights_for_stale_recipe_detection";
    let (store, manager, source_digest, _tmp) = create_store_with_source_blob(source_bytes);

    // Initial build for recipe "1.0.0"
    let _initial = manager
        .get_or_create_derived_layout(
            &source_digest,
            "1.0.0",
            "ime_tile_8x16",
            RetentionPolicy::Keep,
            |_src, dst| {
                fs::write(dst, b"layout_v1_payload").unwrap();
                Ok(17)
            },
        )
        .expect("initial build");

    // Case A: Runtime upgrade introduces recipe "2.0.0", and source weights are present -> Auto-Rebuild
    let mut rebuild_ran = false;
    let rebuilder = |_src: &Path, dst: &Path| -> Result<u64, K3Error> {
        rebuild_ran = true;
        fs::write(dst, b"layout_v2_new_kernel_payload").unwrap();
        Ok(29)
    };

    let outcome_rebuilt = manager
        .detect_and_handle_stale_recipe(
            &source_digest,
            "2.0.0", // upgraded recipe version
            "ime_tile_8x16",
            "qwen3-test",
            rebuilder,
        )
        .expect("handle stale recipe");

    assert!(
        rebuild_ran,
        "auto-rebuild must run when source weights are present"
    );
    match outcome_rebuilt {
        StaleRecipeOutcome::Rebuilt {
            new_derived_digest,
            new_derived_path,
        } => {
            assert!(new_derived_path.exists());
            assert_eq!(
                new_derived_digest,
                Sha256::digest(b"layout_v2_new_kernel_payload")
            );
        }
        _ => panic!("must return StaleRecipeOutcome::Rebuilt"),
    }

    // Case B: Breaking runtime upgrade to "3.0.0", but source weights were purged -> Operator Advisory
    // Remove original source blob to simulate purged source
    let source_path = store.blobs_dir().join(&source_digest);
    fs::remove_file(&source_path).expect("remove source to simulate purged");

    let outcome_purged = manager
        .detect_and_handle_stale_recipe(
            &source_digest,
            "3.0.0", // upgraded recipe version
            "ime_tile_8x16",
            "qwen3-test",
            |_src, _dst| panic!("rebuilder must not be called when source is purged"),
        )
        .expect("handle stale recipe with purged source");

    match outcome_purged {
        StaleRecipeOutcome::SourcePurgedAdvisory {
            source_digest: s_dig,
            stale_recipe_version,
            active_recipe_version,
            advisory_message,
        } => {
            assert_eq!(s_dig, source_digest);
            assert_eq!(stale_recipe_version, "1.0.0");
            assert_eq!(active_recipe_version, "3.0.0");
            assert!(advisory_message.contains("Please re-acquire the source model"));
            assert!(advisory_message.contains("qwen3-test"));
        }
        _ => panic!("must return StaleRecipeOutcome::SourcePurgedAdvisory"),
    }
}

#[test]
fn test_memory_admission_accounting_for_derived_layouts() {
    let current_locked = 10 * 1024 * 1024 * 1024; // 10 GiB
    let additional_layout = 15 * 1024 * 1024 * 1024; // 15 GiB
    let kv_arena = 2 * 1024 * 1024 * 1024; // 2 GiB
    let rlimit_memlock = 28 * 1024 * 1024 * 1024; // 28 GiB ceiling on 32 GiB hardware
    let available_ram = 30 * 1024 * 1024 * 1024; // 30 GiB

    // Total: 10 + 15 + 2 = 27 GiB <= 28 GiB -> Succeeded
    assert!(
        check_derived_memory_admission(
            current_locked,
            additional_layout,
            kv_arena,
            available_ram,
            rlimit_memlock
        )
        .is_ok()
    );

    // Total exceeds ceiling: 10 + 18 + 2 = 30 GiB > 28 GiB -> Refused with InsufficientLockableMemory
    let oversized_layout = 18 * 1024 * 1024 * 1024;
    let err = check_derived_memory_admission(
        current_locked,
        oversized_layout,
        kv_arena,
        available_ram,
        rlimit_memlock,
    )
    .expect_err("must refuse load when aggregate exceeds rlimit");

    assert!(matches!(err, K3Error::InsufficientLockableMemory { .. }));
    assert_eq!(err.diagnostic_code(), "insufficient_lockable_memory");
}
