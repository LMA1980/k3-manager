//! Empirical Adversarial Challenge Suite for Milestone M4 Iteration 3:
//! StorageGarbageCollector::who_uses Robustness, Security & Concurrency Verification.
//!
//! Specifically validates:
//! 1. Missing Blobs: Non-existent digests, phantom catalog entries, broken symlinks, untracked disk files.
//! 2. Boundary Digest Matrix: Lengths 0..65536, casing, forbidden chars, null bytes, unicode, path traversal, injection payloads.
//! 3. Symlink Attacks: Symlinks pointing outside store root, symlinks pointing to sister store dirs (manifests/models),
//!    and legitimate store root under symlinked directories (e.g. macOS /var -> /private/var).
//! 4. Concurrency: High-throughput concurrent who_uses while run_gc is actively sweeping, locking, and unlinking blobs.

#![allow(
    clippy::pedantic,
    clippy::undocumented_unsafe_blocks,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::needless_pass_by_value
)]

use std::fs::{self, File};
use std::io::Write;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use k3_core::K3Error;
use k3_manager::persistence::{CasCatalog, DuckDbPersistenceStore, ModelBlobRecord};
use k3_manager::store::{
    ContentAddressedStore, MockProcMapsScanner, ProcessMappingInfo, ProcessMapsScanner, Sha256,
    StorageGarbageCollector,
};

static SUITE_SEQ: AtomicU64 = AtomicU64::new(40_000);

struct TempSandbox {
    path: PathBuf,
}

impl TempSandbox {
    fn new(name: &str) -> Self {
        let seq = SUITE_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "m4_iter3_who_uses_{name}_{}_{seq}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempSandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn setup_test_cas() -> (
    ContentAddressedStore,
    Arc<DuckDbPersistenceStore>,
    StorageGarbageCollector,
    Arc<MockProcMapsScanner>,
    TempSandbox,
) {
    let sandbox = TempSandbox::new("cas");
    let catalog = Arc::new(DuckDbPersistenceStore::open_in_memory().expect("open catalog"));
    let store = ContentAddressedStore::open(sandbox.path(), catalog.clone()).expect("open store");
    let scanner = Arc::new(MockProcMapsScanner::new());
    let gc = StorageGarbageCollector::with_scanner(
        Arc::new(store.clone()),
        scanner.clone() as Arc<dyn ProcessMapsScanner>,
    );
    (store, catalog, gc, scanner, sandbox)
}

// =========================================================================
// 1. MISSING BLOBS & BROKEN SYMLINKS
// =========================================================================

#[test]
fn test_who_uses_missing_blob_empty_store() {
    let (_store, _catalog, gc, _scanner, _sandbox) = setup_test_cas();

    let missing = "0000000000000000000000000000000000000000000000000000000000000000";
    let res = gc.who_uses(missing);
    assert!(res.is_err());
    match res.unwrap_err() {
        K3Error::CapabilityMismatch {
            architecture,
            reason,
        } => {
            assert_eq!(architecture, "cas_gc");
            assert!(
                reason.contains("not found in store"),
                "Must report 'not found in store', got: '{reason}'"
            );
            assert!(
                !reason.contains("path traversal"),
                "Must NOT falsely claim path traversal for missing blob: '{reason}'"
            );
        }
        other => panic!("Expected CapabilityMismatch, got {other:?}"),
    }
}

#[test]
fn test_who_uses_missing_blob_phantom_catalog_row() {
    let (_store, catalog, gc, _scanner, _sandbox) = setup_test_cas();

    let digest = "1111111111111111111111111111111111111111111111111111111111111111";

    // Insert catalog record without creating file on disk
    let phantom_record = ModelBlobRecord {
        digest: digest.to_owned(),
        blob_type: "raw_weights".to_owned(),
        size_bytes: 4096,
        created_at: 1000,
        last_accessed_at: 1000,
        ref_count: 3,
    };
    catalog.save_blob(&phantom_record).expect("save phantom");

    // Query who_uses: must detect missing file on disk rather than trusting phantom catalog row
    let res = gc.who_uses(digest);
    assert!(res.is_err());
    match res.unwrap_err() {
        K3Error::CapabilityMismatch {
            architecture,
            reason,
        } => {
            assert_eq!(architecture, "cas_gc");
            assert!(
                reason.contains("not found in store"),
                "Must detect file missing on disk despite catalog row: '{reason}'"
            );
            assert!(!reason.contains("path traversal"));
        }
        other => panic!("Expected CapabilityMismatch, got {other:?}"),
    }
}

#[test]
fn test_who_uses_broken_symlink_in_blobs_dir() {
    let (store, _catalog, gc, _scanner, sandbox) = setup_test_cas();

    let digest = "2222222222222222222222222222222222222222222222222222222222222222";
    let link_target = sandbox.path().join("non_existent_target.bin");
    let blob_link = store.blobs_dir().join(digest);

    std::os::unix::fs::symlink(&link_target, &blob_link).expect("create broken symlink");

    // Since symlink target does not exist, blob_path.exists() is false
    let res = gc.who_uses(digest);
    assert!(res.is_err());
    match res.unwrap_err() {
        K3Error::CapabilityMismatch {
            architecture,
            reason,
        } => {
            assert_eq!(architecture, "cas_gc");
            assert!(
                reason.contains("not found in store"),
                "Broken symlink must report not found: '{reason}'"
            );
        }
        other => panic!("Expected CapabilityMismatch, got {other:?}"),
    }
}

#[test]
fn test_who_uses_untracked_disk_blob() {
    let (store, catalog, gc, _scanner, _sandbox) = setup_test_cas();

    let payload = b"UNTRACKED_DISK_BLOB_PAYLOAD";
    let digest = Sha256::digest(payload);
    let blob_path = store.blobs_dir().join(&digest);
    fs::write(&blob_path, payload).expect("write untracked blob");

    // Untracked blob has no row in catalog
    assert!(catalog.get_blob(&digest).unwrap().is_none());

    let info = gc.who_uses(&digest).expect("who_uses succeeds");
    assert_eq!(info.digest, digest);
    assert_eq!(info.size_bytes, payload.len() as u64);
    assert_eq!(info.ref_count, 0);
    assert!(info.referencing_sources.is_empty());
    assert!(!info.gate1_shared_lock_held);
    assert!(info.gate2_active_mappings.is_empty());
    assert!(!info.is_in_use());
}

// =========================================================================
// 2. BOUNDARY DIGEST MATRIX & INJECTION ATTACKS
// =========================================================================

#[test]
fn test_who_uses_digest_length_boundaries() {
    let (_store, _catalog, gc, _scanner, _sandbox) = setup_test_cas();

    let test_lengths = [
        0, 1, 2, 10, 31, 32, 40, 63, // below 64
        65, 66, 70, 128, 256, 1024, 65536, // above 64
    ];

    for len in test_lengths {
        let input = "a".repeat(len);
        let res = gc.who_uses(&input);
        assert!(res.is_err(), "Length {len} must be rejected");
        match res.unwrap_err() {
            K3Error::CapabilityMismatch {
                architecture,
                reason,
            } => {
                assert_eq!(architecture, "cas_digest");
                assert!(
                    reason.contains("must be exactly 64 lowercase hexadecimal characters"),
                    "Error for len {len} must specify 64 chars requirement: '{reason}'"
                );
            }
            other => panic!("Expected CapabilityMismatch for len {len}, got {other:?}"),
        }
    }
}

#[test]
fn test_who_uses_digest_casing_and_invalid_chars() {
    let (_store, _catalog, gc, _scanner, _sandbox) = setup_test_cas();

    let invalid_digests = [
        // Uppercase
        "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85A",
        "Ae3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        // Non-hex
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85g",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85z",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85-",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85_",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85.",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85/",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85 ",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85\t",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85\n",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85\0",
        // Path traversal patterns
        "../../../../../../../../../../../../../../../../etc/passwd00000000",
        "../blobs/sha256/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca4",
        "e3b0c44298fc1c149afbf4c8996fb924/../../../../../../../../etc/passwd",
        // SQL injection / format strings
        "' OR '1'='1' -- 00000000000000000000000000000000000000000000000000",
        "%s%s%s%s%n0000000000000000000000000000000000000000000000000000000",
        // Unicode Cyrillic homoglyph (looks like hex 'a', but is 2-byte U+0430)
        "а3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ];

    for bad in invalid_digests {
        let res = gc.who_uses(bad);
        assert!(
            res.is_err(),
            "Digest payload '{bad:?}' must be rejected by who_uses"
        );
    }
}

// =========================================================================
// 3. SYMLINK ATTACKS & SYMLINKED ROOT RESOLUTION
// =========================================================================

#[test]
fn test_who_uses_symlink_escape_outside_store_root() {
    let (store, _catalog, gc, _scanner, sandbox) = setup_test_cas();

    // Create target sensitive file outside store root
    let outside_secret = sandbox.path().join("secret_external.key");
    fs::write(&outside_secret, b"TOP_SECRET_EXTERNAL_KEY").expect("write secret");

    // Create symlink inside blobs_dir named as a valid 64-hex digest pointing to outside secret
    let digest = "3333333333333333333333333333333333333333333333333333333333333333";
    let evil_link = store.blobs_dir().join(digest);
    std::os::unix::fs::symlink(&outside_secret, &evil_link).expect("create evil symlink");

    // Call who_uses: must detect path traversal because target is outside blobs_dir
    let res = gc.who_uses(digest);
    assert!(res.is_err(), "Symlink escape outside store root must fail");
    match res.unwrap_err() {
        K3Error::CapabilityMismatch {
            architecture,
            reason,
        } => {
            assert_eq!(architecture, "cas_gc");
            assert!(
                reason.contains("path traversal detected"),
                "Must report 'path traversal detected', got: '{reason}'"
            );
        }
        other => panic!("Expected CapabilityMismatch, got {other:?}"),
    }
}

#[test]
fn test_who_uses_symlink_escape_to_sister_store_directory() {
    let (store, _catalog, gc, _scanner, _sandbox) = setup_test_cas();

    // Create a file in manifests/ dir
    let manifest_file = store.manifests_dir().join("some_manifest.json");
    fs::write(&manifest_file, b"{\"schema_version\": 1}").expect("write manifest");

    // Create symlink in blobs_dir pointing to the manifest file
    let digest = "4444444444444444444444444444444444444444444444444444444444444444";
    let evil_link = store.blobs_dir().join(digest);
    std::os::unix::fs::symlink(&manifest_file, &evil_link).expect("create link to manifests");

    // Even though manifests_dir is under store root, it is outside blobs_dir!
    let res = gc.who_uses(digest);
    assert!(res.is_err(), "Symlink into sister directory must fail");
    match res.unwrap_err() {
        K3Error::CapabilityMismatch {
            architecture,
            reason,
        } => {
            assert_eq!(architecture, "cas_gc");
            assert!(
                reason.contains("path traversal detected"),
                "Must report 'path traversal detected' when escaping blobs_dir, got: '{reason}'"
            );
        }
        other => panic!("Expected CapabilityMismatch, got {other:?}"),
    }
}

#[test]
fn test_who_uses_under_symlinked_store_root() {
    let sandbox = TempSandbox::new("symlink_root");
    let real_root = sandbox.path().join("real_store_root");
    fs::create_dir_all(&real_root).expect("create real root");

    // Create symlink to root
    let symlinked_root = sandbox.path().join("symlinked_store_root");
    std::os::unix::fs::symlink(&real_root, &symlinked_root).expect("create root symlink");

    let catalog = Arc::new(DuckDbPersistenceStore::open_in_memory().expect("open catalog"));
    let store = Arc::new(
        ContentAddressedStore::open(&symlinked_root, catalog.clone()).expect("open store"),
    );
    let scanner = Arc::new(MockProcMapsScanner::new());
    let gc = StorageGarbageCollector::with_scanner(store.clone(), scanner.clone());

    // Ingest a legitimate blob
    let payload = b"AUTHENTIC_BLOB_UNDER_SYMLINKED_ROOT";
    let digest = Sha256::digest(payload);
    let (staging, mut f) = store.create_staging_file().unwrap();
    f.write_all(payload).unwrap();
    drop(f);
    let blob_path = store
        .promote_staged_blob(&staging, &digest, "raw")
        .expect("promote");

    // Mock an active process mapping
    scanner.add_mapping(
        &blob_path,
        ProcessMappingInfo {
            pid: 8888,
            comm: "k3-engine".into(),
            mapped_path: blob_path.display().to_string(),
            address_range: "0x4000-0x5000".into(),
            permissions: "r--p".into(),
        },
    );

    // who_uses must succeed without false path traversal
    let info = gc
        .who_uses(&digest)
        .expect("who_uses succeeds under symlinked root");
    assert_eq!(info.digest, digest);
    assert_eq!(info.size_bytes, payload.len() as u64);
    assert_eq!(info.gate2_active_mappings.len(), 1);
    assert_eq!(info.gate2_active_mappings[0].pid, 8888);

    // Also query non-existent blob under symlinked root
    let missing_digest = "5555555555555555555555555555555555555555555555555555555555555555";
    let res_missing = gc.who_uses(missing_digest);
    assert!(res_missing.is_err());
    match res_missing.unwrap_err() {
        K3Error::CapabilityMismatch { reason, .. } => {
            assert!(
                reason.contains("not found in store"),
                "Must report 'not found in store', got: '{reason}'"
            );
            assert!(
                !reason.contains("path traversal"),
                "Must NOT falsely claim path traversal: '{reason}'"
            );
        }
        other => panic!("Expected CapabilityMismatch, got {other:?}"),
    }
}

// =========================================================================
// 4. CONCURRENCY: WHO_USES CONCURRENT WITH RUN_GC & ACTIVE WORKERS
// =========================================================================

#[test]
fn test_who_uses_concurrent_with_active_gc_sweep() {
    let (store, catalog, gc, scanner, _sandbox) = setup_test_cas();
    let store_arc = Arc::new(store);
    let gc_arc = Arc::new(gc);

    // Ingest 3 distinct blobs:
    // Blob A: Unreferenced, eligible for GC reclamation.
    // Blob B: Unreferenced, but held by active flock(LOCK_SH) (Gate 1).
    // Blob C: Unreferenced, but mapped in process virtual memory (Gate 2).

    let payload_a = b"GC_CONCURRENT_SWEEP_PAYLOAD_A";
    let digest_a = Sha256::digest(payload_a);
    let (s_a, mut f_a) = store_arc.create_staging_file().unwrap();
    f_a.write_all(payload_a).unwrap();
    drop(f_a);
    let path_a = store_arc.promote_staged_blob(&s_a, &digest_a, "raw").unwrap();

    let payload_b = b"GC_CONCURRENT_SWEEP_PAYLOAD_B";
    let digest_b = Sha256::digest(payload_b);
    let (s_b, mut f_b) = store_arc.create_staging_file().unwrap();
    f_b.write_all(payload_b).unwrap();
    drop(f_b);
    let path_b = store_arc.promote_staged_blob(&s_b, &digest_b, "raw").unwrap();

    let payload_c = b"GC_CONCURRENT_SWEEP_PAYLOAD_C";
    let digest_c = Sha256::digest(payload_c);
    let (s_c, mut f_c) = store_arc.create_staging_file().unwrap();
    f_c.write_all(payload_c).unwrap();
    drop(f_c);
    let path_c = store_arc.promote_staged_blob(&s_c, &digest_c, "raw").unwrap();

    // Reset catalog ref_counts to 0 so all 3 are GC candidates
    for dig in [&digest_a, &digest_b, &digest_c] {
        let mut rec = catalog.get_blob(dig).unwrap().unwrap();
        rec.ref_count = 0;
        catalog.save_blob(&rec).unwrap();
    }

    // Blob B: Lock with shared advisory lock
    let worker_file_b = File::open(&path_b).unwrap();
    let lock_res = unsafe { libc::flock(worker_file_b.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
    assert_eq!(lock_res, 0);

    // Blob C: Add process memory mapping
    scanner.add_mapping(
        &path_c,
        ProcessMappingInfo {
            pid: 7777,
            comm: "k3-worker-c".into(),
            mapped_path: path_c.display().to_string(),
            address_range: "0x8000-0x9000".into(),
            permissions: "r--p".into(),
        },
    );

    // Launch concurrent reader threads continually querying who_uses across all 3 blobs
    // plus a non-existent blob, while the main thread runs GC!
    let barrier = Arc::new(Barrier::new(5));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut reader_handles = Vec::new();

    let missing_digest = "6666666666666666666666666666666666666666666666666666666666666666".to_string();

    for thread_id in 0..4 {
        let gc_c = gc_arc.clone();
        let bar_c = barrier.clone();
        let stop_c = stop_flag.clone();
        let dig_a = digest_a.clone();
        let dig_b = digest_b.clone();
        let dig_c_str = digest_c.clone();
        let dig_miss = missing_digest.clone();

        reader_handles.push(thread::spawn(move || {
            bar_c.wait();
            let mut queries = 0usize;
            while !stop_c.load(Ordering::Relaxed) && queries < 500 {
                queries += 1;

                // Query Blob A: May succeed initially, then return 'not found in store' after GC unlinks it
                match gc_c.who_uses(&dig_a) {
                    Ok(info) => {
                        assert_eq!(info.digest, dig_a);
                    }
                    Err(K3Error::CapabilityMismatch { reason, .. }) => {
                        assert!(
                            reason.contains("not found in store"),
                            "Thread {thread_id} got unexpected error for Blob A: {reason}"
                        );
                    }
                    Err(e) => panic!("Thread {thread_id} unexpected error variant: {e:?}"),
                }

                // Query Blob B: Must always succeed and report gate1_shared_lock_held
                let res_b = gc_c.who_uses(&dig_b).unwrap();
                assert_eq!(res_b.digest, dig_b);
                assert!(res_b.gate1_shared_lock_held);

                // Query Blob C: Must always succeed and report gate2_active_mappings
                let res_c = gc_c.who_uses(&dig_c_str).unwrap();
                assert_eq!(res_c.digest, dig_c_str);
                assert_eq!(res_c.gate2_active_mappings.len(), 1);

                // Query Missing: Must always return 'not found in store'
                let res_m = gc_c.who_uses(&dig_miss);
                assert!(res_m.is_err());
                match res_m.unwrap_err() {
                    K3Error::CapabilityMismatch { reason, .. } => {
                        assert!(reason.contains("not found in store"));
                        assert!(!reason.contains("path traversal"));
                    }
                    other => panic!("Unexpected error for missing: {other:?}"),
                }
            }
            queries
        }));
    }

    // Coordinator thread: wait for barrier and execute run_gc
    barrier.wait();

    let gc_report = gc_arc.run_gc(false).expect("run_gc succeeds");
    assert!(
        gc_report.reclaimed_blobs.contains(&digest_a),
        "Blob A must be reclaimed by GC"
    );
    assert!(
        gc_report.skipped_in_use_gate1.contains(&digest_b),
        "Blob B must be skipped by Gate 1"
    );
    assert!(
        gc_report.skipped_mapped_gate2.contains(&digest_c),
        "Blob C must be skipped by Gate 2"
    );

    // Let readers run a bit longer then signal stop
    thread::sleep(Duration::from_millis(50));
    stop_flag.store(true, Ordering::Relaxed);

    for h in reader_handles {
        let count = h.join().expect("reader joined cleanly");
        assert!(count > 0, "Reader must have executed queries");
    }

    // Clean up flock on B
    unsafe { libc::flock(worker_file_b.as_raw_fd(), libc::LOCK_UN) };
    drop(worker_file_b);

    // Verify disk states:
    assert!(!path_a.exists(), "Reclaimed Blob A must be unlinked");
    assert!(path_b.exists(), "Blob B must remain on disk");
    assert!(path_c.exists(), "Blob C must remain on disk");

    // After GC, who_uses on Blob A must return 'not found in store'
    let post_gc_res = gc_arc.who_uses(&digest_a);
    assert!(post_gc_res.is_err());
    match post_gc_res.unwrap_err() {
        K3Error::CapabilityMismatch { reason, .. } => {
            assert!(
                reason.contains("not found in store"),
                "Post-GC Blob A must report not found, got: '{reason}'"
            );
            assert!(!reason.contains("path traversal"));
        }
        other => panic!("Expected CapabilityMismatch, got {other:?}"),
    }
}

#[test]
fn test_who_uses_when_blob_path_is_directory() {
    let (store, _catalog, gc, _scanner, _sandbox) = setup_test_cas();

    // Create a directory named as a valid 64-hex SHA-256 digest
    let digest = "7777777777777777777777777777777777777777777777777777777777777777";
    let dir_blob = store.blobs_dir().join(digest);
    fs::create_dir_all(&dir_blob).expect("create dir named as digest");

    // who_uses on a directory should handle it without panicking
    let info_res = gc.who_uses(digest);
    assert!(info_res.is_ok(), "who_uses on directory handles open gracefully without panic");
    let info = info_res.unwrap();
    assert_eq!(info.digest, digest);
}

#[test]
fn test_who_uses_unreadable_permissions_safety() {
    use std::os::unix::fs::PermissionsExt;
    let (store, _catalog, gc, _scanner, _sandbox) = setup_test_cas();

    let payload = b"UNREADABLE_FILE_PERMISSIONS_PAYLOAD";
    let digest = Sha256::digest(payload);
    let blob_path = store.blobs_dir().join(&digest);
    fs::write(&blob_path, payload).expect("write blob");

    // Set file permissions to 0000 (no read/write)
    fs::set_permissions(&blob_path, fs::Permissions::from_mode(0o000)).expect("chmod 0000");

    // who_uses must not panic even if File::open fails with PermissionDenied
    let res = gc.who_uses(&digest);
    assert!(res.is_ok(), "who_uses must not panic on unreadable file");
    let info = res.unwrap();
    assert!(!info.gate1_shared_lock_held);

    // Restore permissions so cleanup doesn't fail
    let _ = fs::set_permissions(&blob_path, fs::Permissions::from_mode(0o644));
}

#[test]
fn test_who_uses_multiple_referencing_manifests_and_aliases() {
    let (store, _catalog, gc, _scanner, _sandbox) = setup_test_cas();

    let payload = b"MULTI_REFERENCED_SHARED_MODEL_WEIGHTS";
    let digest = Sha256::digest(payload);
    let (staging, mut f) = store.create_staging_file().unwrap();
    f.write_all(payload).unwrap();
    drop(f);
    store.promote_staged_blob(&staging, &digest, "raw").unwrap();

    // Create 3 manifests referencing this same digest
    for i in 1..=3 {
        let manifest = k3_manager::store::ModelManifest {
            schema_version: 1,
            format: "raw".to_owned(),
            architecture_id: "synthetic".to_owned(),
            recipe_version: "1.0.0".to_owned(),
            quantization: "none".to_owned(),
            context_length: 2048,
            license: None,
            origin: None,
            weight_digest: digest.clone(),
            weight_size_bytes: payload.len() as u64,
            created_at: 1000 * i,
        };
        let man_rec = store.register_manifest(&manifest).unwrap();
        store
            .create_alias("local", "ns", &format!("model-{i}"), "v1", &man_rec.manifest_digest)
            .unwrap();
    }

    let info = gc.who_uses(&digest).expect("who_uses on multi-referenced blob");
    assert_eq!(info.digest, digest);
    // 3 manifests + 3 aliases = ref_count of 6
    assert_eq!(info.ref_count, 6);
    assert_eq!(info.referencing_sources.len(), 6);
}

#[test]
fn test_who_uses_parallel_probe_stress_under_worker_shared_flock() {
    let (store, _catalog, gc, _scanner, _sandbox) = setup_test_cas();
    let gc_arc = Arc::new(gc);

    let payload = b"PARALLEL_PROBE_STRESS_UNDER_SHARED_FLOCK";
    let digest = Sha256::digest(payload);
    let (staging, mut f) = store.create_staging_file().unwrap();
    f.write_all(payload).unwrap();
    drop(f);
    let blob_path = store.promote_staged_blob(&staging, &digest, "raw").unwrap();

    // Simulated worker holds flock(LOCK_SH)
    let worker_file = File::open(&blob_path).unwrap();
    let lock_res = unsafe { libc::flock(worker_file.as_raw_fd(), libc::LOCK_SH) };
    assert_eq!(lock_res, 0);

    // 8 threads hammer who_uses concurrently on the locked blob
    let mut handles = Vec::new();
    for _ in 0..8 {
        let gc_c = gc_arc.clone();
        let dig = digest.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let info = gc_c.who_uses(&dig).expect("who_uses succeeds");
                assert!(info.gate1_shared_lock_held);
                assert!(info.is_in_use());
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }

    // Release lock
    unsafe { libc::flock(worker_file.as_raw_fd(), libc::LOCK_UN) };
    drop(worker_file);

    // Subsequent who_uses sees lock free
    let info_after = gc_arc.who_uses(&digest).unwrap();
    assert!(!info_after.gate1_shared_lock_held);
    assert!(!info_after.is_in_use());
}

