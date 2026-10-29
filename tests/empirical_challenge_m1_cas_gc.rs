//! Empirical Adversarial Challenge Test Suite for Milestone M1 (CAS Store & Safe GC).
//!
//! Evaluates:
//! 1. Challenge 1: Path Traversal & Symlink Escape Defenses.
//!    - Component & alias path validation (empty, long, traversal, special chars).
//!    - Path traversal in `DerivedBlobManager` and `who_uses`.
//! 2. Challenge 2: Boundary Conditions & Format Verification.
//!    - Empty blobs (0 bytes) under raw vs GGUF vs SafeTensors.
//!    - Corrupt format headers (bad GGUF magic, invalid SafeTensors header length).
//!    - Staging file lifecycle and orphaned file leakage on format rejection.
//! 3. Challenge 3: Concurrent Staging & Staging Filename Collisions.
//!    - Non-atomic/colliding staging filenames in `create_staging_file`.
//!    - Concurrent promotion with distinct staging files.
//! 4. Challenge 4: Dual-Gate Safe GC & `who-uses` Inspection.
//!    - Gate 1 active flock(LOCK_SH) protection.
//!    - Gate 2 active process memory mapping protection.
//!    - Dual-gate combined contention.
//!    - Watermark warning threshold immutability (zero auto-deletion).

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::undocumented_unsafe_blocks
)]

use std::fs::{self, File};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use k3_manager::persistence::{CasCatalog, DuckDbPersistenceStore};
use k3_manager::store::{
    ContentAddressedStore, DerivedBlobManager, MockProcMapsScanner, ModelManifest,
    ProcessMappingInfo, ProcessMapsScanner, RetentionPolicy, Sha256, StorageGarbageCollector,
    parse_alias_id, validate_path_component,
};

static GLOBAL_SEQ: AtomicU64 = AtomicU64::new(6000);

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

fn create_test_env() -> (
    Arc<ContentAddressedStore>,
    Arc<DuckDbPersistenceStore>,
    StorageGarbageCollector,
    Arc<MockProcMapsScanner>,
    TestTempDir,
) {
    let temp_dir = TestTempDir::new("k3_m1_challenger");
    let catalog = Arc::new(DuckDbPersistenceStore::open_in_memory().expect("open catalog"));
    let store = Arc::new(
        ContentAddressedStore::open(temp_dir.path(), catalog.clone()).expect("open store"),
    );
    let mock_scanner = Arc::new(MockProcMapsScanner::new());
    let gc = StorageGarbageCollector::with_scanner(
        store.clone(),
        mock_scanner.clone() as Arc<dyn ProcessMapsScanner>,
    );
    (store, catalog, gc, mock_scanner, temp_dir)
}

fn make_dummy_gguf(payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes()); // version 3
    bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata kv count
    bytes.extend_from_slice(payload);
    bytes
}

// ============================================================================
// CHALLENGE 1: Path Traversal & Symlink Escape Defenses
// ============================================================================

#[test]
fn test_challenge1a_path_component_validation_boundaries() {
    // 1. Length boundaries: 128 chars is maximum allowed; 129 must fail
    let name_128 = "a".repeat(128);
    let name_129 = "a".repeat(129);
    assert!(
        validate_path_component(&name_128).is_ok(),
        "128-char component must be valid"
    );
    assert!(
        validate_path_component(&name_129).is_err(),
        "129-char component must be rejected"
    );

    // 2. Traversal patterns and forbidden characters
    let forbidden = [
        "",                     // Empty
        ".",                    // Single dot
        "..",                   // Parent directory
        "../",                  // Relative parent
        "/etc/passwd",          // Absolute path
        "foo/bar",              // Slashes
        "foo\\bar",             // Backslashes
        "foo\0bar",             // Null byte
        "-flag",                // Leading dash
        "--opt",                // Double leading dash
        "foo bar",              // Space
        "foo\tbar",             // Tab
        "foo\nbar",             // Newline
        "foo\rbar",             // Carriage return
        "foo\x1b[31mbar",        // ANSI escape
        "café",                 // Non-ASCII unicode
        "foo%2e%2e%2fbar",      // URL-encoded percent
    ];

    for case in forbidden {
        let res = validate_path_component(case);
        assert!(
            res.is_err(),
            "Component '{case:?}' must be rejected by validate_path_component"
        );
    }
}

#[test]
fn test_challenge1b_alias_id_parser_defense() {
    let invalid_aliases = [
        "huggingface/Qwen/Qwen2.5-7B",                    // Only 3 parts
        "huggingface/Qwen/Qwen2.5-7B/latest/extra",      // 5 parts
        "",                                              // Empty
        "models/huggingface/Qwen",                       // 2 parts after models/
        "models//Qwen/Qwen2.5-7B/latest",                // Empty part
        "models/huggingface/../Qwen2.5-7B/latest",        // Traversal part
        "/etc/passwd/root/model/v1",                     // Leading slash creates empty segment
        "models/huggingface/Qwen/name/tag\0injection",   // Null byte in tag
    ];

    for alias in invalid_aliases {
        let res = parse_alias_id(alias);
        assert!(
            res.is_err(),
            "Alias '{alias}' must be rejected by parse_alias_id"
        );
    }

    // Valid alias with models/ prefix and without
    let (s, ns, n, t) = parse_alias_id("models/hf/org/model/tag").expect("valid alias with prefix");
    assert_eq!((s.as_str(), ns.as_str(), n.as_str(), t.as_str()), ("hf", "org", "model", "tag"));

    let (s, ns, n, t) = parse_alias_id("hf/org/model/tag").expect("valid alias without prefix");
    assert_eq!((s.as_str(), ns.as_str(), n.as_str(), t.as_str()), ("hf", "org", "model", "tag"));
}

#[test]
fn test_challenge1c_derived_manager_path_traversal_gap() {
    let (store, _catalog, _gc, _scanner, _tmp) = create_test_env();
    let derived_mgr = DerivedBlobManager::new(store.clone());

    // Create an external sensitive file outside the store root
    let external_dir = TestTempDir::new("k3_external_leak");
    let secret_file = external_dir.path().join("secret_data.txt");
    fs::write(&secret_file, b"CONFIDENTIAL_SYSTEM_SECRET_KEY_12345").expect("write secret");

    // Compute relative path traversal from blobs_dir to secret_file:
    // blobs_dir is <temp_dir>/blobs/sha256/
    // Path traversal: ../../../<external_dir>/secret_data.txt
    let secret_path_str = secret_file.to_str().unwrap();

    // Call get_or_create_derived_layout with path traversal in source_digest
    // Because get_or_create_derived_layout does blobs_dir().join(source_clean),
    // an absolute path or relative path with `..` escapes blobs_dir!
    let mut leak_captured = Vec::new();
    let res = derived_mgr.get_or_create_derived_layout(
        secret_path_str,
        "1.0.0",
        "native",
        RetentionPolicy::Keep,
        |src, dst| {
            // Converter is called with source path!
            let data = fs::read(src).expect("read source file");
            leak_captured.extend_from_slice(&data);
            fs::write(dst, &data).expect("write dst");
            Ok(data.len() as u64)
        },
    );

    println!("get_or_create_derived_layout with traversal source path: {res:?}");
    assert!(res.is_err(), "path traversal in source_digest must be rejected");
    assert!(
        leak_captured.is_empty(),
        "converter must never be executed on traversal inputs"
    );
}

#[test]
fn test_challenge1d_who_uses_path_traversal_gap() {
    let (_store, _catalog, gc, _scanner, _tmp) = create_test_env();

    // Create an external file outside the store root
    let external_dir = TestTempDir::new("k3_external_who_uses");
    let target_file = external_dir.path().join("external_target.bin");
    fs::write(&target_file, b"some external file").expect("write target");

    let target_path_str = target_file.to_str().unwrap();

    // Call who_uses with external target path
    let res = gc.who_uses(target_path_str);
    println!("who_uses with external path: {res:?}");
    assert!(res.is_err(), "who_uses must reject external target paths");
}

// ============================================================================
// CHALLENGE 2: Boundary Conditions & Format Verification
// ============================================================================

#[test]
fn test_challenge2a_empty_blob_handling() {
    let (store, _catalog, _gc, _scanner, _tmp) = create_test_env();

    let empty_payload = b"";
    let empty_digest = Sha256::digest(empty_payload);
    assert_eq!(
        empty_digest,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );

    // 1. Raw format empty blob: should be promoted cleanly
    let (staging_path, mut f) = store.create_staging_file().expect("staging");
    f.write_all(empty_payload).expect("write");
    drop(f);

    let promoted = store.promote_staged_blob(&staging_path, &empty_digest, "raw");
    assert!(promoted.is_ok(), "empty raw blob promotion must succeed");
    let blob_path = promoted.unwrap();
    assert!(blob_path.exists());
    assert_eq!(fs::metadata(&blob_path).unwrap().len(), 0);

    // 2. GGUF format empty blob: must fail header verification
    let (staging_path2, mut f2) = store.create_staging_file().expect("staging 2");
    f2.write_all(empty_payload).expect("write");
    drop(f2);

    let gguf_res = store.promote_staged_blob(&staging_path2, &empty_digest, "gguf");
    assert!(
        gguf_res.is_err(),
        "empty blob cannot be valid GGUF; must fail header check"
    );

    // 3. SafeTensors format empty blob: must fail header verification
    let (staging_path3, mut f3) = store.create_staging_file().expect("staging 3");
    f3.write_all(empty_payload).expect("write");
    drop(f3);

    let st_res = store.promote_staged_blob(&staging_path3, &empty_digest, "safetensors");
    assert!(
        st_res.is_err(),
        "empty blob cannot be valid safetensors; must fail header check"
    );
}

#[test]
fn test_challenge2b_corrupt_headers_and_staging_leak() {
    let (store, _catalog, _gc, _scanner, _tmp) = create_test_env();

    // 1. Corrupt GGUF magic: "NOTG" instead of "GGUF"
    let bad_magic_payload = b"NOTG\x03\0\0\0dummy";
    let bad_magic_digest = Sha256::digest(bad_magic_payload);

    let (staging_path, mut f) = store.create_staging_file().expect("staging");
    f.write_all(bad_magic_payload).expect("write");
    drop(f);

    let res = store.promote_staged_blob(&staging_path, &bad_magic_digest, "gguf");
    assert!(res.is_err(), "bad GGUF magic must fail header check");
    assert!(
        !staging_path.exists(),
        "staging file must be removed on GGUF header failure"
    );

    // 2. SafeTensors with zero header length
    let zero_len_header = 0u64.to_le_bytes();
    let zero_len_digest = Sha256::digest(&zero_len_header);

    let (staging_path2, mut f2) = store.create_staging_file().expect("staging 2");
    f2.write_all(&zero_len_header).expect("write");
    drop(f2);

    let res2 = store.promote_staged_blob(&staging_path2, &zero_len_digest, "safetensors");
    assert!(res2.is_err(), "zero-length safetensors header must fail");
    assert!(
        !staging_path2.exists(),
        "staging file 2 must be removed on SafeTensors header failure"
    );
}

// ============================================================================
// CHALLENGE 3: Concurrent Staging & Staging Filename Collisions
// ============================================================================

#[test]
fn test_challenge3a_staging_filename_collision_empirical_proof() {
    let (store, _catalog, _gc, _scanner, _tmp) = create_test_env();

    // Verify collision-free UUID v4 staging files when called rapidly across concurrent threads
    let attempts = 100;
    let collisions = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..8 {
        let store_clone = store.clone();
        let collisions_clone = collisions.clone();
        handles.push(thread::spawn(move || {
            let mut paths = std::collections::HashSet::new();
            for _ in 0..attempts {
                if let Ok((path, _file)) = store_clone.create_staging_file() {
                    if !paths.insert(path) {
                        collisions_clone.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let total_collisions = collisions.load(Ordering::Relaxed);
    println!("Intra-thread duplicate staging path generations: {total_collisions}");
    assert_eq!(
        total_collisions, 0,
        "create_staging_file must have 0 collisions under concurrency"
    );
}

#[test]
fn test_challenge3b_concurrent_promotion_with_distinct_staging_paths() {
    let (store, catalog, _gc, _scanner, _tmp) = create_test_env();

    let shared_payload = Arc::new(make_dummy_gguf(b"concurrent_distinct_staging_payload"));
    let expected_digest = Sha256::digest(&shared_payload);

    let num_threads = 8;
    let mut handles = Vec::new();

    for thread_idx in 0..num_threads {
        let store_clone = store.clone();
        let payload_clone = shared_payload.clone();
        let digest_clone = expected_digest.clone();

        handles.push(thread::spawn(move || {
            // Create uniquely suffixed staging file to bypass create_staging_file timestamp collision
            let unique_part_name = format!("thread_{thread_idx}_test.part");
            let part_path = store_clone.staging_dir().join(unique_part_name);
            let mut f = File::create(&part_path).expect("create distinct part");
            f.write_all(&payload_clone).expect("write");
            drop(f);

            store_clone
                .promote_staged_blob(&part_path, &digest_clone, "gguf")
                .unwrap_or_else(|e| panic!("thread {thread_idx} promote blob: {e}"))
        }));
    }

    let mut promoted_paths = Vec::new();
    for handle in handles {
        promoted_paths.push(handle.join().expect("thread completed"));
    }

    let canonical_blob = store.blobs_dir().join(&expected_digest);
    for p in promoted_paths {
        assert_eq!(p, canonical_blob);
    }

    assert!(canonical_blob.exists());
    assert_eq!(
        fs::metadata(&canonical_blob).unwrap().len(),
        shared_payload.len() as u64
    );

    // Staging directory must be clean
    let staging_entries: Vec<_> = fs::read_dir(store.staging_dir())
        .expect("read staging")
        .flatten()
        .collect();
    assert_eq!(
        staging_entries.len(),
        0,
        "all staging files must be cleaned up / promoted"
    );

    let blob_rec = catalog.get_blob(&expected_digest).unwrap();
    assert!(blob_rec.is_some());
    assert_eq!(blob_rec.unwrap().digest, expected_digest);
}

// ============================================================================
// CHALLENGE 4: Dual-Gate Safe GC & `who-uses` Inspection
// ============================================================================

#[test]
fn test_challenge4a_dual_gate_lock_and_map_stress() {
    let (store, catalog, gc, mock_scanner, _tmp) = create_test_env();

    // 1. Ingest blob and create alias
    let payload = make_dummy_gguf(b"dual_gate_stress_model");
    let digest = Sha256::digest(&payload);

    let (staging, mut f) = store.create_staging_file().unwrap();
    f.write_all(&payload).unwrap();
    drop(f);
    let blob_path = store.promote_staged_blob(&staging, &digest, "gguf").unwrap();

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
    let man_rec = store.register_manifest(&manifest).unwrap();
    store
        .create_alias("local", "app", "model-z", "active", &man_rec.manifest_digest)
        .unwrap();

    // Verify ref_count > 0 prevents GC
    let report_initial = gc.run_gc(false).unwrap();
    assert_eq!(report_initial.reclaimed_blobs.len(), 0);
    assert!(blob_path.exists());

    // Operator deletes alias: ref_count becomes 0
    store.remove_alias("local", "app", "model-z", "active").unwrap();
    let unref = catalog.list_unreferenced_blobs().unwrap();
    assert_eq!(unref.len(), 1);
    assert_eq!(unref[0].digest, digest);

    // ------------------------------------------------------------------------
    // SUB-TEST A: Worker process 1 holds flock(LOCK_SH)
    // ------------------------------------------------------------------------
    let file1 = File::open(&blob_path).unwrap();
    let fd1 = file1.as_raw_fd();
    let res = unsafe { libc::flock(fd1, libc::LOCK_SH | libc::LOCK_NB) };
    assert_eq!(res, 0);

    let usage = gc.who_uses(&digest).unwrap();
    assert!(usage.gate1_shared_lock_held);
    assert!(usage.is_in_use());

    let rep1 = gc.run_gc(false).unwrap();
    assert_eq!(rep1.reclaimed_blobs.len(), 0);
    assert!(rep1.skipped_in_use_gate1.contains(&digest));
    assert!(blob_path.exists());

    // ------------------------------------------------------------------------
    // SUB-TEST B: Worker process 2 ALSO maps into virtual memory
    // ------------------------------------------------------------------------
    mock_scanner.add_mapping(
        &blob_path,
        ProcessMappingInfo {
            pid: 9999,
            comm: "k3-worker-2".to_owned(),
            address_range: "0x10000000-0x20000000".to_owned(),
            permissions: "r--p".to_owned(),
            mapped_path: blob_path.to_string_lossy().to_string(),
        },
    );

    let usage2 = gc.who_uses(&digest).unwrap();
    assert!(usage2.gate1_shared_lock_held);
    assert_eq!(usage2.gate2_active_mappings.len(), 1);
    assert!(usage2.is_in_use());

    let rep2 = gc.run_gc(false).unwrap();
    assert_eq!(rep2.reclaimed_blobs.len(), 0);
    assert!(blob_path.exists());

    // ------------------------------------------------------------------------
    // SUB-TEST C: Worker 1 releases flock, but Worker 2 memory mapping persists
    // ------------------------------------------------------------------------
    unsafe { libc::flock(fd1, libc::LOCK_UN) };
    drop(file1);

    let usage3 = gc.who_uses(&digest).unwrap();
    assert!(!usage3.gate1_shared_lock_held);
    assert_eq!(usage3.gate2_active_mappings.len(), 1);
    assert!(usage3.is_in_use());

    let rep3 = gc.run_gc(false).unwrap();
    assert_eq!(rep3.reclaimed_blobs.len(), 0);
    assert!(rep3.skipped_mapped_gate2.contains(&digest));
    assert!(blob_path.exists());

    // ------------------------------------------------------------------------
    // SUB-TEST D: Dry run when both gates pass
    // ------------------------------------------------------------------------
    mock_scanner.clear();
    let usage_clear = gc.who_uses(&digest).unwrap();
    assert!(!usage_clear.is_in_use());

    let dry_run_rep = gc.run_gc(true).unwrap();
    assert_eq!(dry_run_rep.reclaimed_blobs.len(), 1);
    assert_eq!(dry_run_rep.reclaimed_blobs[0], digest);
    assert!(blob_path.exists(), "dry run must not unlink blob");

    // ------------------------------------------------------------------------
    // SUB-TEST E: Real GC execution when both gates pass
    // ------------------------------------------------------------------------
    let real_rep = gc.run_gc(false).unwrap();
    assert_eq!(real_rep.reclaimed_blobs.len(), 1);
    assert_eq!(real_rep.reclaimed_blobs[0], digest);
    assert!(!blob_path.exists(), "real GC must unlink blob");

    assert!(catalog.get_blob(&digest).unwrap().is_none());
}

#[test]
fn test_challenge4b_watermark_warning_non_destructive() {
    let (store, _catalog, mut gc, _scanner, _tmp) = create_test_env();

    gc.set_high_watermark_bytes(Some(10_000));

    let payload = make_dummy_gguf(&vec![0x42; 25_000]);
    let digest = Sha256::digest(&payload);

    let (staging, mut f) = store.create_staging_file().unwrap();
    f.write_all(&payload).unwrap();
    drop(f);
    let blob_path = store.promote_staged_blob(&staging, &digest, "gguf").unwrap();

    let warning = gc.check_watermark().unwrap().expect("warning emitted");
    assert_eq!(warning.threshold_bytes, 10_000);
    assert!(warning.current_bytes >= 25_000);
    assert!(warning.usage_ratio > 2.0);

    for _ in 0..5 {
        let _ = gc.check_watermark().unwrap();
        assert!(blob_path.exists(), "watermark check must be read-only");
    }
}
