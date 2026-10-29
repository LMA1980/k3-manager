//! Scenario and unit verification for US-22: Operator-Driven Storage Lifecycle & Safe Garbage Collection.
//!
//! Verifies:
//! - Scenario 1: Reference-counted manifest removal without immediate blob destruction.
//! - Scenario 2: High-water mark warning without unapproved automatic blob deletion.
//! - Scenario 3: Dual-gate safety check (Gate 1 flock + Gate 2 maps scan) preventing deletion of mapped blobs (online or offline).

use std::fs::{self, File};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use k3_manager::persistence::{CasCatalog, DuckDbPersistenceStore};
use k3_manager::store::{
    ContentAddressedStore, MockProcMapsScanner, ModelManifest, ProcessMappingInfo,
    ProcessMapsScanner, Sha256, StorageGarbageCollector,
};

static GLOBAL_SEQ: AtomicU64 = AtomicU64::new(200);

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

fn create_store_and_gc() -> (
    Arc<ContentAddressedStore>,
    StorageGarbageCollector,
    Arc<MockProcMapsScanner>,
    TestTempDir,
) {
    let temp_dir = TestTempDir::new("k3_test_us22");
    let catalog = DuckDbPersistenceStore::open_in_memory().expect("open catalog");
    let catalog_arc: Arc<dyn CasCatalog> = Arc::new(catalog);

    let store = Arc::new(
        ContentAddressedStore::open(temp_dir.path(), catalog_arc)
            .expect("open store"),
    );

    let mock_scanner = Arc::new(MockProcMapsScanner::new());
    let gc = StorageGarbageCollector::with_scanner(
        store.clone(),
        mock_scanner.clone() as Arc<dyn ProcessMapsScanner>,
    );

    (store, gc, mock_scanner, temp_dir)
}

fn create_dummy_gguf(content: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(content);
    bytes
}

#[test]
fn test_scenario1_reference_counted_manifest_removal_without_blob_destruction() {
    let (store, gc, _scanner, _tmp) = create_store_and_gc();

    // 1. Ingest a shared weight blob referenced by two tags (v1.0 and v1.1)
    let payload = create_dummy_gguf(b"shared_weights_between_tags_v1_and_v2");
    let digest = Sha256::digest(&payload);

    let (staging, mut f) = store.create_staging_file().expect("staging");
    f.write_all(&payload).expect("write");
    drop(f);
    store.promote_staged_blob(&staging, &digest, "gguf").expect("promote");

    // Tag 1: v1.0
    let man1 = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: None,
        origin: None,
        weight_digest: digest.clone(),
        weight_size_bytes: payload.len() as u64,
        created_at: 1000,
    };
    let man_rec1 = store.register_manifest(&man1).expect("man1");
    store
        .create_alias("local", "team", "model-x", "v1.0", &man_rec1.manifest_digest)
        .expect("alias1");

    // Tag 2: v1.1
    let man2 = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: None,
        origin: None,
        weight_digest: digest.clone(),
        weight_size_bytes: payload.len() as u64,
        created_at: 2000,
    };
    let man_rec2 = store.register_manifest(&man2).expect("man2");
    store
        .create_alias("local", "team", "model-x", "v1.1", &man_rec2.manifest_digest)
        .expect("alias2");

    let blob_before = store.catalog().get_blob(&digest).unwrap().unwrap();
    assert_eq!(blob_before.ref_count, 4); // 2 manifests + 2 aliases

    // 2. Remove Tag 1 (v1.0)
    let removed = store
        .remove_alias("local", "team", "model-x", "v1.0")
        .expect("remove alias v1.0");
    assert!(removed);

    // Assert manifest 1 and alias 1 are gone
    let alias1_opt = store.catalog().get_alias("local/team/model-x/v1.0").unwrap();
    assert!(alias1_opt.is_none());
    assert!(!store.models_dir().join("local/team/model-x/v1.0").exists());

    // Run garbage collection
    let gc_report = gc.run_gc(false).expect("run gc");
    assert_eq!(gc_report.reclaimed_blobs.len(), 0, "shared blob must not be deleted");

    // 3. Shared weight blob remains intact on storage for Tag 2 (v1.1)
    let weight_blob_path = store.blobs_dir().join(&digest);
    assert!(
        weight_blob_path.exists(),
        "underlying shared weight blob must remain intact on storage"
    );

    let blob_after = store.catalog().get_blob(&digest).unwrap().unwrap();
    assert_eq!(blob_after.ref_count, 2); // still referenced by manifest 2 and alias 2

    // Tag 2 resolves cleanly
    let resolved = store
        .resolve_alias("local/team/model-x/v1.1")
        .expect("resolve tag 2");
    assert_eq!(resolved.weight_path, weight_blob_path);
}

#[test]
fn test_scenario2_high_water_mark_warning_without_auto_deletion() {
    let (store, mut gc, _scanner, _tmp) = create_store_and_gc();

    // Configure high-water mark ceiling of 5,000 bytes
    let threshold = 5_000;
    gc.set_high_watermark_bytes(Some(threshold));

    // Store a blob of 8,000 bytes (exceeding the 5,000 threshold)
    let payload = create_dummy_gguf(&vec![0xAA; 8_000]);
    let digest = Sha256::digest(&payload);

    let (staging, mut f) = store.create_staging_file().expect("staging");
    f.write_all(&payload).expect("write");
    drop(f);
    store.promote_staged_blob(&staging, &digest, "gguf").expect("promote");

    // 1. Check watermark: emits operational warning
    let warning_opt = gc.check_watermark().expect("check watermark");
    assert!(warning_opt.is_some(), "must emit high-water mark warning");
    let warning = warning_opt.unwrap();
    assert_eq!(warning.threshold_bytes, threshold);
    assert!(warning.current_bytes >= 8000);
    assert!(warning.usage_ratio > 1.0);
    assert!(warning.warning_message.contains("exceeds configured high-water mark"));

    // 2. Crucial Invariant: The system NEVER automatically deletes model blobs on watermark crossing
    let blob_path = store.blobs_dir().join(&digest);
    assert!(
        blob_path.exists(),
        "crossing high-water mark must never trigger unapproved automatic blob deletion"
    );
}

#[test]
fn test_scenario3_dual_gate_safety_check_online_or_offline() {
    let (store, gc, mock_scanner, _tmp) = create_store_and_gc();

    let payload = create_dummy_gguf(b"isolated_model_for_dual_gate_test");
    let digest = Sha256::digest(&payload);

    let (staging, mut f) = store.create_staging_file().expect("staging");
    f.write_all(&payload).expect("write");
    drop(f);
    let blob_path = store.promote_staged_blob(&staging, &digest, "gguf").expect("promote");

    let manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: None,
        origin: None,
        weight_digest: digest.clone(),
        weight_size_bytes: payload.len() as u64,
        created_at: 1000,
    };
    let man_rec = store.register_manifest(&manifest).expect("manifest");
    store
        .create_alias("local", "app", "worker-model", "active", &man_rec.manifest_digest)
        .expect("alias");

    // Manifest removed by operator: alias unlinked, ref_count decrements to 0
    let removed = store
        .remove_alias("local", "app", "worker-model", "active")
        .expect("remove alias");
    assert!(removed);

    let unreferenced = store.catalog().list_unreferenced_blobs().expect("unreferenced");
    assert_eq!(unreferenced.len(), 1);
    assert_eq!(unreferenced[0].digest, digest);

    // =========================================================================
    // Gate 1 Test: Active compute worker holds shared flock(LOCK_SH) on the blob
    // =========================================================================
    {
        let worker_file = File::open(&blob_path).expect("worker open file");
        let worker_fd = worker_file.as_raw_fd();
        // Worker acquires shared advisory lock
        // SAFETY: Calling flock with valid open file descriptor
        let flock_res = unsafe { libc::flock(worker_fd, libc::LOCK_SH | libc::LOCK_NB) };
        assert_eq!(flock_res, 0, "worker acquired LOCK_SH");

        // Inspect who_uses query
        let usage = gc.who_uses(&digest).expect("who-uses query");
        assert!(usage.gate1_shared_lock_held, "who-uses detects shared flock held");
        assert!(usage.is_in_use(), "blob flagged as in-use");

        // GC runs while worker holds lock: must skip deleting
        let gc_report1 = gc.run_gc(false).expect("gc run 1");
        assert_eq!(gc_report1.reclaimed_blobs.len(), 0);
        assert!(
            gc_report1.skipped_in_use_gate1.contains(&digest),
            "Gate 1 must skip blob held by active shared lock"
        );
        assert!(blob_path.exists(), "blob must not be deleted while Gate 1 is locked");

        // Worker drops file and releases lock
        // SAFETY: Calling flock to release lock on valid open file descriptor
        unsafe { libc::flock(worker_fd, libc::LOCK_UN) };
        drop(worker_file);
    }

    // =========================================================================
    // Gate 2 Test: Active worker process maps the blob into virtual memory (/proc/*/maps)
    // =========================================================================
    {
        // Simulate a running worker process (PID 4242) mapping the blob into memory
        mock_scanner.add_mapping(
            &blob_path,
            ProcessMappingInfo {
                pid: 4242,
                comm: "k3-worker".to_owned(),
                address_range: "0x7f8a10000000-0x7f8a90000000".to_owned(),
                permissions: "r--p".to_owned(),
                mapped_path: blob_path.to_string_lossy().to_string(),
            },
        );

        // Inspect who_uses query
        let usage2 = gc.who_uses(&digest).expect("who-uses query");
        assert!(!usage2.gate1_shared_lock_held, "flock is free");
        assert_eq!(usage2.gate2_active_mappings.len(), 1, "Gate 2 detects active mapping");
        assert_eq!(usage2.gate2_active_mappings[0].pid, 4242);
        assert_eq!(usage2.gate2_active_mappings[0].comm, "k3-worker");
        assert!(usage2.is_in_use(), "blob flagged in-use due to Gate 2 mapping");

        // GC runs while memory mapping exists: must skip deleting
        let gc_report2 = gc.run_gc(false).expect("gc run 2");
        assert_eq!(gc_report2.reclaimed_blobs.len(), 0);
        assert!(
            gc_report2.skipped_mapped_gate2.contains(&digest),
            "Gate 2 must skip blob mapped in process virtual memory"
        );
        assert!(blob_path.exists(), "blob must not be deleted while mapped in memory");

        // Process exits: clear memory mapping
        mock_scanner.clear();
    }

    // =========================================================================
    // Both Gates Pass: Safe reclamation executed
    // =========================================================================
    let usage_idle = gc.who_uses(&digest).expect("who-uses idle");
    assert!(!usage_idle.is_in_use(), "blob is now completely idle");

    let gc_report_final = gc.run_gc(false).expect("final gc");
    assert_eq!(gc_report_final.reclaimed_blobs.len(), 1);
    assert_eq!(gc_report_final.reclaimed_blobs[0], digest);
    assert!(gc_report_final.reclaimed_bytes > 0);

    // Physically unlinked from storage
    assert!(
        !blob_path.exists(),
        "unreferenced blob must be unlinked once both gates pass"
    );

    // Removed from DuckDB catalog
    let blob_in_catalog = store.catalog().get_blob(&digest).unwrap();
    assert!(blob_in_catalog.is_none());
}
