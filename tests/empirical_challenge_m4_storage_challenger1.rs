//! Empirical Adversarial Challenge Suite 1 for Milestone M4 Storage Hardening.
//!
//! Specifically validates:
//! - Feature 25: Path traversal sequences, null bytes, symlink escapes, uppercase or non-64-hex SHA-256 digests.
//! - Feature 26: TOCTOU staging race conditions, concurrent promotion flock collision, safe GC inode validation,
//!   and macOS /var vs /private/var symlink handling in MockProcMapsScanner and who_uses.
//! - Feature 27: SafeTensors header length boundary attacks (>100MB, header_len + 8 > file_len,
//!   tensor offset begin > end, tensor offset end > file_len, u64 overflow) and GGUF count bounds (>65536)
//!   including 24-byte header mock format.
//! - Derived layout: Non-existent source blob error reporting ("not found in store; cannot derive layout").

#![allow(
    clippy::pedantic,
    clippy::undocumented_unsafe_blocks,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::needless_pass_by_value
)]

use std::fs::{self, File};
use std::io::{self, Write};
use std::os::unix::fs::MetadataExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use k3_core::K3Error;
use k3_manager::ingestion::{
    LocalFileAdapter, ModelFormat, PullRequest, SourceAdapter, StagingPipeline,
};
use k3_manager::persistence::{CasCatalog, DuckDbPersistenceStore};
use k3_manager::store::{
    parse_alias_id, validate_path_component, validate_sha256_digest, ContentAddressedStore,
    DerivedBlobManager, DerivedResolution, MockProcMapsScanner, ProcessMappingInfo,
    ProcessMapsScanner, RetentionPolicy, Sha256, StorageGarbageCollector,
};

static CHALLENGER1_SEQ: AtomicU64 = AtomicU64::new(20_000);

struct TempSandbox {
    path: PathBuf,
}

impl TempSandbox {
    fn new(name: &str) -> Self {
        let seq = CHALLENGER1_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "m4_challenger1_{name}_{}_{seq}",
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

fn setup_cas() -> (
    ContentAddressedStore,
    Arc<DuckDbPersistenceStore>,
    TempSandbox,
) {
    let sandbox = TempSandbox::new("cas");
    let catalog = Arc::new(DuckDbPersistenceStore::open_in_memory().expect("open catalog"));
    let store = ContentAddressedStore::open(sandbox.path(), catalog.clone()).expect("open store");
    (store, catalog, sandbox)
}

// =========================================================================
// 1. FEATURE 25: Strict SHA-256 Digest & Path Traversal / Escape Hardening
// =========================================================================

#[test]
fn test_challenge_f25_uppercase_and_non_hex_digest_matrix() {
    let uppercase_digests = [
        "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855", // all upper
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85A", // single upper at end
        "Ae3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85", // single upper at start
        "e3b0c44298fc1c149afbf4c8996fb92427Ae41e4649b934ca495991b7852b855", // single upper in middle
        "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef", // mixed segment
    ];

    for bad in uppercase_digests {
        let res = validate_sha256_digest(bad);
        assert!(
            res.is_err(),
            "Uppercase digest '{bad}' must be rejected by validate_sha256_digest"
        );
        let err = res.unwrap_err().to_string();
        assert!(
            err.contains("must be exactly 64 lowercase hexadecimal characters"),
            "Error must specify requirement: {err}"
        );
    }

    let non_hex_digests = [
        "",                                                                    // empty
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85",     // 63 chars (short)
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b8555",   // 65 chars (long)
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85g",    // 'g'
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85z",    // 'z'
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85-",    // '-'
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85_",    // '_'
        "e3b0c44298fc1c149afbf4c8996fb924\x007ae41e4649b934ca495991b7852b855", // embedded null
        "e3b0c44298fc1c149afbf4c8996fb924\n7ae41e4649b934ca495991b7852b855",   // embedded newline
        " e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",   // leading space
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 ",   // trailing space
    ];

    for bad in non_hex_digests {
        let res = validate_sha256_digest(bad);
        assert!(
            res.is_err(),
            "Non-hex digest '{bad:?}' must be rejected by validate_sha256_digest"
        );
    }

    // Valid 64 lowercase hex must pass
    let valid = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    assert!(validate_sha256_digest(valid).is_ok());
}

#[test]
fn test_challenge_f25_path_traversal_null_bytes_and_symlink_escapes() {
    let bad_components = [
        "",
        ".",
        "..",
        "... ",
        "../",
        "..\\",
        "./",
        "-leading-dash",
        "--double-dash",
        "foo/bar",
        "foo\\bar",
        "foo\0bar",
        "foo\x01bar",
        "foo\tbar",
        "foo\nbar",
        "foo\rbar",
        "foo:bar",
        "foo*bar",
        "foo?bar",
        "foo\"bar",
        "foo<bar",
        "foo>bar",
        "foo|bar",
    ];

    for comp in bad_components {
        assert!(
            validate_path_component(comp).is_err(),
            "Path component '{comp:?}' must be rejected"
        );
    }

    // Aliases with traversal or bad components
    let bad_aliases = [
        "huggingface/Qwen/Qwen2.5-7B",             // only 3 segments
        "huggingface/Qwen/Qwen2.5-7B/v1/extra",    // 5 segments
        "huggingface/Qwen/Qwen2.5-7B/v1/",         // trailing slash
        "/huggingface/Qwen/Qwen2.5-7B/v1",         // leading slash
        "huggingface/../model/v1",                 // traversal in namespace
        "huggingface/Qwen/../../v1",               // traversal in model
        "huggingface/Qwen/model/\0v1",             // null byte
        "models/huggingface/Qwen/Qwen2.5-7B",      // 3 segments with prefix
    ];

    for alias in bad_aliases {
        assert!(
            parse_alias_id(alias).is_err(),
            "Alias '{alias}' must be rejected by parse_alias_id"
        );
    }

    // CAS alias symlink escape: pointing outside store root
    let (store, _catalog, sandbox) = setup_cas();
    let outside_secret = sandbox.path().join("external_secret.txt");
    fs::write(&outside_secret, b"EXTERNAL_SECRET").unwrap();

    let evil_alias_dir = store
        .models_dir()
        .join("escape")
        .join("ns")
        .join("model")
        .join("v1");
    fs::create_dir_all(&evil_alias_dir).unwrap();
    let evil_link = evil_alias_dir.join("weights");
    std::os::unix::fs::symlink(&outside_secret, &evil_link).unwrap();

    let res = store.resolve_alias("escape/ns/model/v1");
    assert!(
        res.is_err(),
        "resolve_alias must reject symlinks pointing outside store root"
    );

    // LocalFileAdapter symlink escape
    let local_base = sandbox.path().join("local_base");
    let outside_local = sandbox.path().join("outside_local");
    fs::create_dir_all(&local_base).unwrap();
    fs::create_dir_all(&outside_local).unwrap();

    let secret_weights = outside_local.join("secret.bin");
    fs::write(&secret_weights, b"SECRET_LOCAL_WEIGHTS").unwrap();

    let evil_sym = local_base.join("evil_link.bin");
    std::os::unix::fs::symlink(&secret_weights, &evil_sym).unwrap();

    let adapter = LocalFileAdapter::new(&local_base);
    let staging_dir = sandbox.path().join("staging");
    let (staging_path, _) = StagingPipeline::create_staging_file(&staging_dir).unwrap();

    let pull_req = PullRequest {
        source: "local".into(),
        namespace: "local".into(),
        model: "safe".into(),
        tag: "evil_link.bin".into(),
        expected_digest: None,
        format: Some(ModelFormat::Custom("raw".into())),
    };

    let pull_res = adapter.pull(&pull_req, &staging_path, None);
    assert!(
        pull_res.is_err(),
        "LocalFileAdapter must reject pulling files via symlink escaping base directory"
    );
}

// =========================================================================
// 2. FEATURE 26: TOCTOU Races, Concurrent Promotion & Safe GC Inode Validation
// =========================================================================

#[test]
fn test_challenge_f26_concurrent_staging_promotion_and_flock_contention() {
    let (store, _catalog, _sandbox) = setup_cas();
    let (staging_path, mut f) = store.create_staging_file().unwrap();
    let payload = b"CONCURRENT_FLOCK_CHALLENGER_PAYLOAD";
    f.write_all(payload).unwrap();

    // Lock file with flock(LOCK_EX | LOCK_NB)
    let fd = f.as_raw_fd();
    let lock_res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(lock_res, 0, "Initial lock must succeed");

    let digest = Sha256::digest(payload);
    let store_clone = store.clone();
    let staging_clone = staging_path.clone();
    let digest_clone = digest.clone();

    // Attempt promotion while locked
    let handle = thread::spawn(move || {
        store_clone.promote_staged_blob(&staging_clone, &digest_clone, "raw")
    });

    let res = handle.join().expect("thread join");
    assert!(
        res.is_err(),
        "Promotion of an exclusively locked staging file must fail with flock contention"
    );
    let err = res.unwrap_err().to_string();
    assert!(
        err.contains("flock contention") || err.contains("locked by another process"),
        "Error must mention flock: {err}"
    );

    // Release lock and promote
    unsafe { libc::flock(fd, libc::LOCK_UN) };
    drop(f);

    let promote_res = store.promote_staged_blob(&staging_path, &digest, "raw");
    assert!(promote_res.is_ok(), "Promotion after unlock must succeed");

    // Stress test: 8 threads concurrently promote 8 separate staging files containing identical payload
    let store_arc = Arc::new(store);
    let mut race_handles = Vec::new();
    let num_threads = 8;

    for _ in 0..num_threads {
        let st = store_arc.clone();
        let dig = digest.clone();
        let p_bytes = payload.to_vec();
        race_handles.push(thread::spawn(move || {
            let (part_path, mut part_f) = st.create_staging_file().unwrap();
            part_f.write_all(&p_bytes).unwrap();
            drop(part_f);
            st.promote_staged_blob(&part_path, &dig, "raw")
        }));
    }

    let mut promoted_paths = Vec::new();
    for h in race_handles {
        let r = h.join().expect("join thread");
        assert!(r.is_ok(), "Concurrent deduplicated promotion must succeed");
        promoted_paths.push(r.unwrap());
    }

    // Verify all returned paths are identical and inode was preserved (no destructive overwriting)
    let first_path = &promoted_paths[0];
    let first_inode = fs::metadata(first_path).unwrap().ino();
    for p in &promoted_paths[1..] {
        assert_eq!(p, first_path);
        assert_eq!(fs::metadata(p).unwrap().ino(), first_inode);
    }
}

/// Adversarial scanner that substitutes the blob file on disk with a brand new file (different inode)
/// during the maps scan phase before GC unlinking!
#[derive(Debug)]
struct AttackerSubstitutionScanner {
    blob_path: PathBuf,
    attack_executed: AtomicBool,
    substituted_inode: AtomicU64,
}

impl ProcessMapsScanner for AttackerSubstitutionScanner {
    fn scan_mappings_for_path(
        &self,
        _canonical_path: &Path,
    ) -> Result<Vec<ProcessMappingInfo>, io::Error> {
        // Substitute the file under blob_path with a completely new file
        let _ = fs::remove_file(&self.blob_path);
        let mut f = File::create(&self.blob_path)?;
        f.write_all(b"ATTACKER_SUBSTITUTED_INODE_FILE_CONTENT")?;
        f.flush()?;
        drop(f);

        let new_ino = fs::metadata(&self.blob_path)?.ino();
        self.substituted_inode.store(new_ino, Ordering::SeqCst);
        self.attack_executed.store(true, Ordering::SeqCst);

        // Report 0 active memory mappings
        Ok(Vec::new())
    }
}

#[test]
fn test_challenge_f26_safe_gc_inode_validation_substitution_attack() {
    let (store, catalog, _sandbox) = setup_cas();
    let store_arc = Arc::new(store);

    let payload = b"VICTIM_ORIGINAL_BLOB_PAYLOAD";
    let digest = Sha256::digest(payload);
    let (staging, mut f) = store_arc.create_staging_file().unwrap();
    f.write_all(payload).unwrap();
    drop(f);

    let blob_path = store_arc
        .promote_staged_blob(&staging, &digest, "raw")
        .unwrap();
    let original_inode = fs::metadata(&blob_path).unwrap().ino();

    // Catalog has ref_count = 0
    let mut blob_rec = catalog.get_blob(&digest).unwrap().unwrap();
    blob_rec.ref_count = 0;
    catalog.save_blob(&blob_rec).unwrap();

    let attack_scanner = Arc::new(AttackerSubstitutionScanner {
        blob_path: blob_path.clone(),
        attack_executed: AtomicBool::new(false),
        substituted_inode: AtomicU64::new(0),
    });

    let gc = StorageGarbageCollector::with_scanner(store_arc.clone(), attack_scanner.clone());

    // Execute GC sweep
    let report = gc.run_gc(false).unwrap();

    // Attack was executed
    assert!(attack_scanner.attack_executed.load(Ordering::SeqCst));
    let new_ino = attack_scanner.substituted_inode.load(Ordering::SeqCst);
    assert_ne!(original_inode, new_ino);

    // Inode check must have detected mismatch between open FD and disk file!
    assert!(
        blob_path.exists(),
        "Substituted file must NOT be deleted because inode changed before unlink!"
    );
    assert_eq!(
        fs::metadata(&blob_path).unwrap().ino(),
        new_ino,
        "The newly substituted file must remain untouched"
    );
    assert!(
        !report.reclaimed_blobs.contains(&digest),
        "Blob must NOT be marked as reclaimed"
    );
}

#[test]
fn test_challenge_f26_macos_var_symlink_handling_in_mock_scanner_and_who_uses() {
    let (store, _catalog, _sandbox) = setup_cas();
    let store_arc = Arc::new(store);

    let payload = b"WHO_USES_TEST_BLOB";
    let digest = Sha256::digest(payload);
    let (staging, mut f) = store_arc.create_staging_file().unwrap();
    f.write_all(payload).unwrap();
    drop(f);

    let blob_path = store_arc
        .promote_staged_blob(&staging, &digest, "raw")
        .unwrap();

    // On macOS, blob_path in tempdir starts with /var/... and canonical path starts with /private/var/...
    let canonical_blob_path = fs::canonicalize(&blob_path).unwrap();

    let mock_scanner = MockProcMapsScanner::new();
    mock_scanner.add_mapping(
        &blob_path,
        ProcessMappingInfo {
            pid: 1234,
            comm: "k3-worker".into(),
            mapped_path: blob_path.display().to_string(),
            address_range: "0x1000-0x2000".into(),
            permissions: "r--p".into(),
        },
    );

    // Query with raw path
    let res_raw = mock_scanner.scan_mappings_for_path(&blob_path).unwrap();
    assert_eq!(res_raw.len(), 1);
    assert_eq!(res_raw[0].pid, 1234);

    // Query with canonical path (/private/var/... on macOS)
    let res_canon = mock_scanner.scan_mappings_for_path(&canonical_blob_path).unwrap();
    assert_eq!(res_canon.len(), 1, "Mapping must be found when queried via canonical path");
    assert_eq!(res_canon[0].pid, 1234);

    let test_scanner = Arc::new(MockProcMapsScanner::new());
    test_scanner.add_mapping(
        &blob_path,
        ProcessMappingInfo {
            pid: 5678,
            comm: "k3-worker".into(),
            mapped_path: blob_path.display().to_string(),
            address_range: "0x3000-0x4000".into(),
            permissions: "r--p".into(),
        },
    );

    let gc = StorageGarbageCollector::with_scanner(store_arc.clone(), test_scanner.clone());

    // Call who_uses without lock
    let info = gc.who_uses(&digest).unwrap();
    assert_eq!(info.digest, digest);
    assert!(!info.gate1_shared_lock_held);
    assert_eq!(info.gate2_active_mappings.len(), 1);
    assert_eq!(info.gate2_active_mappings[0].pid, 5678);

    // Now acquire a shared flock on blob_path (simulating active compute worker)
    let open_file = File::open(&blob_path).unwrap();
    let lock_res = unsafe { libc::flock(open_file.as_raw_fd(), libc::LOCK_SH) };
    assert_eq!(lock_res, 0);

    let info_locked = gc.who_uses(&digest).unwrap();
    assert!(info_locked.gate1_shared_lock_held);
    assert_eq!(info_locked.gate2_active_mappings.len(), 1);

    unsafe { libc::flock(open_file.as_raw_fd(), libc::LOCK_UN) };
    drop(open_file);

    // Call who_uses for a non-existent digest
    let non_existent_digest = "0000000000000000000000000000000000000000000000000000000000000000";
    let res_missing = gc.who_uses(non_existent_digest);
    assert!(res_missing.is_err(), "who_uses on non-existent digest must return error");
    let err = res_missing.unwrap_err();
    match err {
        K3Error::CapabilityMismatch { architecture, reason } => {
            assert_eq!(architecture, "cas_gc");
            assert!(
                reason.contains("not found in store"),
                "who_uses on non-existent digest must report 'not found in store', but got false-positive path traversal: '{reason}'"
            );
        }
        other => panic!("Expected CapabilityMismatch, got {other:?}"),
    }
}

// =========================================================================
// 3. FEATURE 27: SafeTensors & GGUF Header Boundary Attacks
// =========================================================================

#[test]
fn test_challenge_f27_safetensors_header_length_boundary_attacks() {
    let (store, _catalog, _sandbox) = setup_cas();

    // 1. Truncated file (< 8 bytes)
    for len in [0, 1, 4, 7] {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(&vec![0u8; len]).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(res.is_err(), "Must reject <8 bytes file");
        assert!(res.unwrap_err().to_string().contains("truncated"));
    }

    // 2. header_len == 0
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(res.is_err(), "Must reject header_len == 0");
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("invalid SafeTensors header length"));
    }

    // 3. header_len > 100MB ceiling (100 * 1024 * 1024 + 1)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        let ceiling_plus_1 = 100 * 1024 * 1024 + 1u64;
        f.write_all(&ceiling_plus_1.to_le_bytes()).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(res.is_err(), "Must reject header_len > 100MB");
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("invalid SafeTensors header length"));
    }

    // 4. header_len + 8 > file_len (truncated JSON header)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(&1000u64.to_le_bytes()).unwrap(); // claims 1000 bytes
        f.write_all(b"{\"small\":1}").unwrap();        // only 11 bytes
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(res.is_err(), "Must reject truncated header");
        assert!(res.unwrap_err().to_string().contains("truncated"));
    }

    // 5. Inverted tensor data_offsets (begin > end)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        let header_json = serde_json::json!({
            "tensor_inverted": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [100, 50] // begin 100 > end 50!
            }
        })
        .to_string();
        let header_len = header_json.len() as u64;
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_json.as_bytes()).unwrap();
        f.write_all(&[0u8; 200]).unwrap();
        drop(f);

        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(res.is_err(), "Must reject inverted data_offsets");
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("inverted data_offsets"));
    }

    // 6. Tensor data_offsets extending past file length (8 + header_len + end > file_len)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        let header_json = serde_json::json!({
            "tensor_oob": {
                "dtype": "I8",
                "shape": [10, 10],
                "data_offsets": [0, 500] // 500 bytes needed, but file only has 50 payload bytes
            }
        })
        .to_string();
        let header_len = header_json.len() as u64;
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_json.as_bytes()).unwrap();
        f.write_all(&[0u8; 50]).unwrap();
        drop(f);

        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(res.is_err(), "Must reject offsets exceeding file length");
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("exceeds file length"));
    }

    // 7. Tensor offset integer overflow (end = u64::MAX)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        let header_json = serde_json::json!({
            "tensor_overflow": {
                "dtype": "F32",
                "shape": [1, 1],
                "data_offsets": [0, u64::MAX]
            }
        })
        .to_string();
        let header_len = header_json.len() as u64;
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_json.as_bytes()).unwrap();
        f.write_all(&[0u8; 50]).unwrap();
        drop(f);

        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(res.is_err(), "Must reject u64 overflow in data_offsets");
    }

    // 8. Exact boundary SafeTensors (8 + header_len + end == file_len)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        let header_json = serde_json::json!({
            "__metadata__": { "framework": "pt" },
            "tensor_boundary": {
                "dtype": "I8",
                "shape": [2, 4],
                "data_offsets": [0, 8]
            }
        })
        .to_string();
        let header_len = header_json.len() as u64;
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_json.as_bytes()).unwrap();
        f.write_all(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap(); // exactly 8 bytes
        drop(f);

        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(res.is_ok(), "Valid boundary SafeTensors must succeed: {res:?}");
    }
}

#[test]
fn test_challenge_f27_gguf_count_bounds_and_24byte_mock_format() {
    let (store, _catalog, _sandbox) = setup_cas();

    // 1. Truncated GGUF file (< 24 bytes)
    for len in [0, 4, 10, 23] {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(&vec![0u8; len]).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_err(), "Must reject <24 bytes GGUF");
        assert!(res.unwrap_err().to_string().contains("truncated"));
    }

    // 2. Wrong magic
    let bad_magics: [&[u8]; 4] = [b"GGML", b"GGSN", b"gguf", b"\0\0\0\0"];
    for magic in bad_magics {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(magic).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_err(), "Must reject non-GGUF magic {magic:?}");
        assert!(res.unwrap_err().to_string().contains("valid GGUF magic"));
    }

    // 3. Version bounds: versions 1, 2, 3 accepted; 0, 4, 99 rejected
    for v in [0u32, 4u32, 99u32] {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&v.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_err(), "Must reject unsupported GGUF version {v}");
        assert!(res.unwrap_err().to_string().contains("unsupported GGUF version"));
    }

    for v in [1u32, 2u32, 3u32] {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&v.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_ok(), "Supported GGUF version {v} must be accepted");
    }

    // 4. tensor_count bounds: 65536 vs 65537 vs u64::MAX
    {
        // 65537 -> reject
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&65_537u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("tensor_count"));

        // 65536 -> accept
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&65_536u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_ok(), "tensor_count 65536 must be accepted");
    }

    // 5. kv_count bounds: 65536 vs 65537 vs u64::MAX
    {
        // 65537 -> reject
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&65_537u64.to_le_bytes()).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("kv_count"));

        // 65536 -> accept
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&65_536u64.to_le_bytes()).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_ok(), "kv_count 65536 must be accepted");
    }

    // 6. Authentic 24-byte mock GGUF format:
    // Magic (4B) + Version 2 (4B) + tensor_count 0 (8B) + kv_count 0 (8B)
    {
        let mock_header = b"GGUF\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(mock_header.len(), 24);

        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(mock_header).unwrap();
        drop(f);

        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_ok(), "24-byte mock GGUF must pass header verification");

        // Verify it can be promoted into CAS
        let digest = Sha256::digest(mock_header);
        let promoted = store.promote_staged_blob(&staging, &digest, "gguf");
        assert!(promoted.is_ok(), "Promotion of 24-byte mock GGUF must succeed");
    }
}

// =========================================================================
// 4. DERIVED LAYOUT: Non-Existent Source Blob Error Reporting
// =========================================================================

#[test]
fn test_challenge_derived_layout_nonexistent_source_blob_error_reporting() {
    let (store, _catalog, _sandbox) = setup_cas();
    let store_arc = Arc::new(store);
    let manager = DerivedBlobManager::new(store_arc.clone());

    // 1. Non-existent 64-hex SHA-256 digest
    let nonexistent_digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    let res = manager.get_or_create_derived_layout(
        nonexistent_digest,
        "1.0.0",
        "ime_tiled",
        RetentionPolicy::Keep,
        |_src, _dst| Ok(100),
    );

    assert!(res.is_err(), "Must error on non-existent source blob");
    let err = res.unwrap_err();
    match err {
        K3Error::CapabilityMismatch { architecture, reason } => {
            assert_eq!(architecture, "derived_weights");
            assert!(
                reason.contains("not found in store; cannot derive layout"),
                "Reason must state 'not found in store; cannot derive layout', got: '{reason}'"
            );
            assert!(
                !reason.contains("path traversal"),
                "Must NOT erroneously report path traversal on non-existent blob, got: '{reason}'"
            );
        }
        other => panic!("Expected CapabilityMismatch, got {other:?}"),
    }

    // 2. Traversal or invalid characters in source_digest are rejected upfront
    let bad_digest = "../../../../../etc/passwd00000000000000000000000000000000000000000000";
    let res_bad = manager.get_or_create_derived_layout(
        bad_digest,
        "1.0.0",
        "ime_tiled",
        RetentionPolicy::Keep,
        |_src, _dst| Ok(100),
    );
    assert!(res_bad.is_err(), "Must reject traversal digest upfront");
    let err_str = res_bad.unwrap_err().to_string();
    assert!(err_str.contains("invalid SHA-256 digest"));

    // 3. When source blob exists, conversion succeeds and is cached
    let source_payload = b"ORIGINAL_SOURCE_WEIGHTS_FOR_DERIVATION";
    let source_digest = Sha256::digest(source_payload);
    let (staging, mut f) = store_arc.create_staging_file().unwrap();
    f.write_all(source_payload).unwrap();
    drop(f);
    let _ = store_arc
        .promote_staged_blob(&staging, &source_digest, "raw")
        .unwrap();

    let conv_res = manager.get_or_create_derived_layout(
        &source_digest,
        "1.0.0",
        "ime_tiled",
        RetentionPolicy::Keep,
        |_src, dst| {
            let mut out = File::create(dst).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "test".into(),
                reason: e.to_string(),
            })?;
            out.write_all(b"DERIVED_IME_TILED_WEIGHTS").map_err(|e| K3Error::CapabilityMismatch {
                architecture: "test".into(),
                reason: e.to_string(),
            })?;
            Ok(25)
        },
    );
    assert!(conv_res.is_ok(), "Conversion with existing source blob must succeed");
    let res_enum = conv_res.unwrap();
    assert!(matches!(res_enum, DerivedResolution::Created { .. }));

    // Subsequent resolution should be Cached
    let cached_res = manager.get_or_create_derived_layout(
        &source_digest,
        "1.0.0",
        "ime_tiled",
        RetentionPolicy::Keep,
        |_src, _dst| panic!("Converter must not be invoked on cached derived layout"),
    );
    assert!(cached_res.is_ok());
    assert!(matches!(cached_res.unwrap(), DerivedResolution::Cached { .. }));
}
