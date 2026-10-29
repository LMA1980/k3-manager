//! Empirical Adversarial Challenge Suite for Milestone M4 Storage Hardening (Features 25, 26, 27).
//!
//! Written by challenger_m4_iter1_1 to stress-test:
//! - Feature 25: Path traversal sequences, null bytes, symlink escapes, uppercase or non-64-hex SHA-256 digests.
//! - Feature 26: TOCTOU staging race conditions, concurrent promotion flock collision, safe GC inode validation.
//! - Feature 27: SafeTensors header length boundary attacks (>100MB, header_len + 8 > file_len, tensor offset begin > end, tensor offset end > file_len, u64 overflow) and GGUF count bounds (>65536).

#![allow(
    clippy::pedantic,
    clippy::undocumented_unsafe_blocks,
    clippy::too_many_lines,
    clippy::similar_names
)]

use std::fs::{self, File};
use std::io::{self, Write};
use std::os::unix::fs::MetadataExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use k3_manager::ingestion::{
    LocalFileAdapter, ModelFormat, PullRequest, SourceAdapter, StagingPipeline,
};
use k3_manager::persistence::{CasCatalog, DuckDbPersistenceStore};
use k3_manager::store::{
    parse_alias_id, validate_path_component, validate_sha256_digest, ContentAddressedStore,
    ProcessMappingInfo, ProcessMapsScanner, Sha256, StorageGarbageCollector,
};

static CHALLENGE_SEQ: AtomicU64 = AtomicU64::new(10_000);

struct TempSandbox {
    path: PathBuf,
}

impl TempSandbox {
    fn new(name: &str) -> Self {
        let seq = CHALLENGE_SEQ.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("m4_challenge_{name}_{}_{seq}", std::process::id()));
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
// FEATURE 25: Path Traversal, Null Bytes, Symlink Escapes & Strict Hex Digests
// =========================================================================

#[test]
fn challenge_f25_sha256_digest_strictness_adversarial_matrix() {
    let invalid_digests = [
        "",                                                                    // Empty
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85",     // 63 chars (short)
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b8555",   // 65 chars (long)
        "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855",    // Uppercase
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852B855",    // Mixed case
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85g",    // 'g' non-hex
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85z",    // 'z' non-hex
        " e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",   // Leading space
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 ",   // Trailing space
        "e3b0c44298fc1c149afbf4c8996fb924\x007ae41e4649b934ca495991b7852b855", // Embedded null
        "e3b0c44298fc1c149afbf4c8996fb924\n7ae41e4649b934ca495991b7852b855",   // Embedded newline
        "../../../../../../../../../../../../../../../../../../../../../../a", // Traversal path
        "/etc/passwd00000000000000000000000000000000000000000000000000000",    // Traversal prefix
    ];

    for bad in invalid_digests {
        let res = validate_sha256_digest(bad);
        assert!(
            res.is_err(),
            "Digest '{bad}' must be rejected by validate_sha256_digest"
        );
    }

    // Valid 64 lowercase hex must pass
    let valid = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    assert!(validate_sha256_digest(valid).is_ok());
}

#[test]
fn challenge_f25_path_component_and_alias_traversal_matrix() {
    let too_long = "a".repeat(129);
    let bad_components = [
        "",
        ".",
        "..",
        "... ",
        "-flag",
        "--option",
        "foo/bar",
        "foo\\bar",
        "foo\0bar",
        "foo\x01bar",
        "foo\nbar",
        "foo\rbar",
        "foo\tbar",
        "foo:bar",
        "foo*bar",
        "foo?bar",
        "foo\"bar",
        "foo<bar",
        "foo>bar",
        "foo|bar",
        too_long.as_str(),
    ];

    for comp in bad_components {
        assert!(
            validate_path_component(comp).is_err(),
            "Component '{comp:?}' must be rejected by validate_path_component"
        );
    }

    let bad_aliases = [
        "huggingface/Qwen/Qwen2.5-7B",             // 3 parts
        "huggingface/Qwen/Qwen2.5-7B/v1/extra",    // 5 parts
        "huggingface/Qwen/Qwen2.5-7B/v1/",         // Trailing slash (empty 5th part)
        "/huggingface/Qwen/Qwen2.5-7B/v1",         // Leading slash (empty 1st part)
        "huggingface//Qwen2.5-7B/v1",              // Empty namespace
        "huggingface/../root/secret/v1",           // Traversal in namespace
        "huggingface/Qwen/../../etc/v1",           // Traversal in name
        "huggingface/Qwen/Qwen2.5-7B/\0malicious", // Null byte in tag
    ];

    for alias in bad_aliases {
        assert!(
            parse_alias_id(alias).is_err(),
            "Alias '{alias}' must be rejected by parse_alias_id"
        );
    }
}

#[test]
fn challenge_f25_local_file_adapter_symlink_escape_rejection() {
    let sandbox = TempSandbox::new("local_symlink");
    let base_dir = sandbox.path().join("allowed_base");
    let outside_dir = sandbox.path().join("forbidden_outside");
    fs::create_dir_all(&base_dir).unwrap();
    fs::create_dir_all(&outside_dir).unwrap();

    // Create a target file outside the base_dir
    let secret_file = outside_dir.join("confidential.gguf");
    fs::write(&secret_file, b"CONFIDENTIAL_DATA").unwrap();

    // Create a symlink inside base_dir pointing to the outside secret file
    let evil_symlink = base_dir.join("symlink_escape.gguf");
    std::os::unix::fs::symlink(&secret_file, &evil_symlink).unwrap();

    let adapter = LocalFileAdapter::new(&base_dir);
    let staging_dir = sandbox.path().join("staging");
    let (staging_path, _) = StagingPipeline::create_staging_file(&staging_dir).unwrap();

    let req = PullRequest {
        source: "local".into(),
        namespace: "local".into(),
        model: "safe_dir".into(),
        tag: "symlink_escape.gguf".into(),
        expected_digest: None,
        format: Some(ModelFormat::Custom("raw".into())),
    };

    let res = adapter.pull(&req, &staging_path, None);
    assert!(
        res.is_err(),
        "LocalFileAdapter must reject pulling via a symlink pointing outside base_dir"
    );
    let err_str = res.unwrap_err().to_string();
    assert!(
        err_str.contains("resolved outside base directory")
            || err_str.contains("directory traversal"),
        "Error must clearly indicate escape detection: {err_str}"
    );
}

#[test]
fn challenge_f25_store_alias_symlink_escape_rejection() {
    let (store, _catalog, sandbox) = setup_cas();

    let outside_dir = sandbox.path().join("outside");
    fs::create_dir_all(&outside_dir).unwrap();
    let victim_weights = outside_dir.join("secret_weights");
    fs::write(&victim_weights, b"VICTIM_WEIGHTS").unwrap();

    // Construct an alias directory with an evil symlink pointing outside CAS root
    let alias_dir = store
        .models_dir()
        .join("evil")
        .join("ns")
        .join("model")
        .join("v1");
    fs::create_dir_all(&alias_dir).unwrap();
    let weights_link = alias_dir.join("weights");
    std::os::unix::fs::symlink(&victim_weights, &weights_link).unwrap();

    let res = store.resolve_alias("evil/ns/model/v1");
    assert!(res.is_err());
    let err = res.unwrap_err().to_string();
    assert!(
        err.contains("outside configured store root") || err.contains("not registered"),
        "Must reject symlink escape: {err}"
    );
}

// =========================================================================
// FEATURE 26: TOCTOU Staging Races, Inode Preservation & Concurrent Flocks
// =========================================================================

#[test]
fn challenge_f26_concurrent_promotion_flock_contention_stress() {
    let (store, _catalog, _sandbox) = setup_cas();
    let (staging_path, mut f) = store.create_staging_file().unwrap();
    let payload = b"CONCURRENT_FLOCK_TEST_PAYLOAD";
    f.write_all(payload).unwrap();

    // Thread A locks the staging file
    let fd = f.as_raw_fd();
    let lock_res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(lock_res, 0, "Initial lock acquisition must succeed");

    // Thread B attempts promotion while Thread A holds the exclusive lock
    let digest = Sha256::digest(payload);
    let store_clone = store.clone();
    let staging_clone = staging_path.clone();
    let digest_clone = digest.clone();

    let handle = thread::spawn(move || {
        store_clone.promote_staged_blob(&staging_clone, &digest_clone, "raw")
    });

    let res = handle.join().expect("thread join");
    assert!(
        res.is_err(),
        "Concurrent promotion must fail immediately with flock contention"
    );
    let err = res.unwrap_err().to_string();
    assert!(
        err.contains("flock contention") || err.contains("locked by another process"),
        "Error must specify flock contention: {err}"
    );

    // Thread A releases lock
    unsafe { libc::flock(fd, libc::LOCK_UN) };
    drop(f);

    // Now promotion should succeed cleanly
    let promote_res = store.promote_staged_blob(&staging_path, &digest, "raw");
    assert!(
        promote_res.is_ok(),
        "Promotion after lock release must succeed"
    );
}

#[test]
fn challenge_f26_multi_thread_concurrent_identical_promotions_preserve_inode() {
    let (store, _catalog, _sandbox) = setup_cas();
    let payload = b"IDENTICAL_BLOB_MULTI_THREAD_DEDUP";
    let digest = Sha256::digest(payload);

    let store_arc = Arc::new(store);
    let num_threads = 6;
    let mut handles = Vec::new();

    for _ in 0..num_threads {
        let st = store_arc.clone();
        let dig = digest.clone();
        let p_bytes = payload.to_vec();
        handles.push(thread::spawn(move || {
            let (staging_path, mut f) = st.create_staging_file().unwrap();
            f.write_all(&p_bytes).unwrap();
            drop(f);
            st.promote_staged_blob(&staging_path, &dig, "raw")
        }));
    }

    let mut promoted_paths = Vec::new();
    for h in handles {
        let res = h.join().expect("thread join");
        assert!(
            res.is_ok(),
            "Every concurrent promote must succeed (either primary or dedup)"
        );
        promoted_paths.push(res.unwrap());
    }

    // Verify all returned paths are identical and point to a single invariant inode
    let first_path = &promoted_paths[0];
    let first_inode = fs::metadata(first_path).unwrap().ino();
    for p in &promoted_paths[1..] {
        assert_eq!(p, first_path);
        assert_eq!(
            fs::metadata(p).unwrap().ino(),
            first_inode,
            "Inode must be strictly preserved across all deduplications"
        );
    }
}

/// Attack scanner simulating a TOCTOU race: replaces the blob file on disk with a brand new file
/// (new inode) during the Gate 2 inspection hook!
#[derive(Debug)]
struct AttackToctouScanner {
    blob_path: PathBuf,
    attack_triggered: AtomicBool,
    new_inode: AtomicU64,
}

impl ProcessMapsScanner for AttackToctouScanner {
    fn scan_mappings_for_path(
        &self,
        _canonical_path: &Path,
    ) -> Result<Vec<ProcessMappingInfo>, io::Error> {
        // Trigger TOCTOU file substitution attack
        let _ = fs::remove_file(&self.blob_path);
        let mut f = File::create(&self.blob_path)?;
        f.write_all(b"BRAND_NEW_ATTACKER_INODE_FILE")?;
        f.flush()?;
        drop(f);

        let new_ino = fs::metadata(&self.blob_path)?.ino();
        self.new_inode.store(new_ino, Ordering::SeqCst);
        self.attack_triggered.store(true, Ordering::SeqCst);

        // Report zero memory mappings so Gate 2 would normally pass
        Ok(Vec::new())
    }
}

#[test]
fn challenge_f26_safe_gc_detects_inode_substitution_race() {
    let (store, catalog, _sandbox) = setup_cas();
    let store_arc = Arc::new(store);

    // Ingest candidate blob
    let payload = b"ORIGINAL_INODE_PAYLOAD";
    let digest = Sha256::digest(payload);
    let (staging, mut f) = store_arc.create_staging_file().unwrap();
    f.write_all(payload).unwrap();
    drop(f);
    let blob_path = store_arc
        .promote_staged_blob(&staging, &digest, "raw")
        .unwrap();
    let orig_inode = fs::metadata(&blob_path).unwrap().ino();

    // Catalog has ref_count = 0
    let mut blob_rec = catalog.get_blob(&digest).unwrap().unwrap();
    blob_rec.ref_count = 0;
    catalog.save_blob(&blob_rec).unwrap();

    // Configure attack scanner that replaces blob_path during the scan
    let attack_scanner = Arc::new(AttackToctouScanner {
        blob_path: blob_path.clone(),
        attack_triggered: AtomicBool::new(false),
        new_inode: AtomicU64::new(0),
    });
    let gc = StorageGarbageCollector::with_scanner(store_arc.clone(), attack_scanner.clone());

    // Run GC sweep
    let report = gc.run_gc(false).unwrap();

    // Verification:
    // 1. Attack must have been triggered
    assert!(attack_scanner.attack_triggered.load(Ordering::SeqCst));
    let substituted_inode = attack_scanner.new_inode.load(Ordering::SeqCst);
    assert_ne!(orig_inode, substituted_inode);

    // 2. Inode validation check must have detected inode mismatch (m_fd.ino() != m_path.ino())
    // and skipped unlinking!
    assert!(
        blob_path.exists(),
        "Substituted file must NOT be unlinked because inode changed between open and unlink!"
    );
    assert_eq!(
        fs::metadata(&blob_path).unwrap().ino(),
        substituted_inode,
        "The substituted inode must remain intact on storage"
    );
    assert!(
        !report.reclaimed_blobs.contains(&digest),
        "Blob must NOT be marked as reclaimed in GC report"
    );
}

// =========================================================================
// FEATURE 27: SafeTensors & GGUF Header Boundary Attacks
// =========================================================================

#[test]
fn challenge_f27_safetensors_header_boundary_attacks() {
    let (store, _catalog, _sandbox) = setup_cas();

    // Attack 1: file_len < 8 bytes
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(&[1, 2, 3, 4, 5, 6, 7]).unwrap(); // 7 bytes
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(res.is_err(), "Must reject <8 bytes file");
        assert!(res.unwrap_err().to_string().contains("truncated"));
    }

    // Attack 2: header_len == 0
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

    // Attack 3: header_len > 100MB ceiling (100MB + 1 byte)
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

    // Attack 4: header_len + 8 > file_len (claim 50 bytes JSON, but only 10 provided)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(&50u64.to_le_bytes()).unwrap();
        f.write_all(b"{\"a\":1}").unwrap(); // 7 bytes
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(
            res.is_err(),
            "Must reject header extending past file length"
        );
        assert!(res.unwrap_err().to_string().contains("truncated"));
    }

    // Attack 5: Tensor offset integer overflow attack (end = u64::MAX)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        let header_json = serde_json::json!({
            "tensor_overflow": {
                "dtype": "F32",
                "shape": [1, 2],
                "data_offsets": [0, u64::MAX]
            }
        })
        .to_string();
        let header_len = header_json.len() as u64;
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_json.as_bytes()).unwrap();
        f.write_all(&[0u8; 100]).unwrap();
        drop(f);

        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(
            res.is_err(),
            "Must reject tensor offset causing u64 overflow"
        );
        let err = res.unwrap_err().to_string();
        assert!(err.contains("exceeds file length") || err.contains("overflow"));
    }

    // Attack 6: Inverted data offsets (begin = 50, end = 20)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        let header_json = serde_json::json!({
            "tensor_inv": {
                "dtype": "I8",
                "shape": [4, 4],
                "data_offsets": [50, 20]
            }
        })
        .to_string();
        let header_len = header_json.len() as u64;
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_json.as_bytes()).unwrap();
        f.write_all(&[0u8; 100]).unwrap();
        drop(f);

        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(res.is_err(), "Must reject inverted data_offsets");
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("inverted data_offsets"));
    }

    // Attack 7: Valid SafeTensors exactly on boundary (8 + header_len + end == file_len)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        let header_json = serde_json::json!({
            "__metadata__": { "format": "pt" },
            "tensor_valid": {
                "dtype": "I8",
                "shape": [2, 2],
                "data_offsets": [0, 4]
            }
        })
        .to_string();
        let header_len = header_json.len() as u64;
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_json.as_bytes()).unwrap();
        f.write_all(&[1, 2, 3, 4]).unwrap(); // exactly 4 bytes
        drop(f);

        let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
        assert!(
            res.is_ok(),
            "Valid boundary SafeTensors must succeed: {res:?}"
        );
    }
}

#[test]
fn challenge_f27_gguf_count_and_magic_boundary_attacks() {
    let (store, _catalog, _sandbox) = setup_cas();

    // Attack 1: file_len < 24 bytes
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&[0u8; 10]).unwrap(); // 14 bytes total (< 24)
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_err(), "Must reject truncated GGUF file (<24 bytes)");
        assert!(res.unwrap_err().to_string().contains("truncated"));
    }

    // Attack 2: Wrong magic (e.g. GGML or GGSN)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(b"GGML").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&10u64.to_le_bytes()).unwrap();
        f.write_all(&10u64.to_le_bytes()).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_err(), "Must reject non-GGUF magic");
        assert!(res.unwrap_err().to_string().contains("valid GGUF magic"));
    }

    // Attack 3: Unsupported version (version 0, or version 99)
    {
        let (staging, mut f) = store.create_staging_file().unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&99u32.to_le_bytes()).unwrap(); // Version 99
        f.write_all(&10u64.to_le_bytes()).unwrap();
        f.write_all(&10u64.to_le_bytes()).unwrap();
        drop(f);
        let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
        assert!(res.is_err(), "Must reject unsupported version 99");
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("unsupported GGUF version"));
    }

    // Attack 4: tensor_count boundary: 65536 vs 65537
    {
        // 65537 must fail
        let (staging_bad, mut f_bad) = store.create_staging_file().unwrap();
        f_bad.write_all(b"GGUF").unwrap();
        f_bad.write_all(&3u32.to_le_bytes()).unwrap();
        f_bad.write_all(&65_537u64.to_le_bytes()).unwrap();
        f_bad.write_all(&10u64.to_le_bytes()).unwrap();
        drop(f_bad);
        let res_bad = ContentAddressedStore::verify_format_header(&staging_bad, "gguf");
        assert!(res_bad.is_err(), "Must reject tensor_count == 65537");
        assert!(res_bad.unwrap_err().to_string().contains("tensor_count"));

        // 65536 must pass
        let (staging_ok, mut f_ok) = store.create_staging_file().unwrap();
        f_ok.write_all(b"GGUF").unwrap();
        f_ok.write_all(&3u32.to_le_bytes()).unwrap();
        f_ok.write_all(&65_536u64.to_le_bytes()).unwrap();
        f_ok.write_all(&10u64.to_le_bytes()).unwrap();
        drop(f_ok);
        let res_ok = ContentAddressedStore::verify_format_header(&staging_ok, "gguf");
        assert!(res_ok.is_ok(), "Must accept tensor_count == 65536 limit");
    }

    // Attack 5: metadata_kv_count boundary: 65536 vs 65537
    {
        // 65537 must fail
        let (staging_bad, mut f_bad) = store.create_staging_file().unwrap();
        f_bad.write_all(b"GGUF").unwrap();
        f_bad.write_all(&3u32.to_le_bytes()).unwrap();
        f_bad.write_all(&10u64.to_le_bytes()).unwrap();
        f_bad.write_all(&65_537u64.to_le_bytes()).unwrap();
        drop(f_bad);
        let res_bad = ContentAddressedStore::verify_format_header(&staging_bad, "gguf");
        assert!(res_bad.is_err(), "Must reject metadata_kv_count == 65537");
        assert!(res_bad.unwrap_err().to_string().contains("kv_count"));

        // 65536 must pass
        let (staging_ok, mut f_ok) = store.create_staging_file().unwrap();
        f_ok.write_all(b"GGUF").unwrap();
        f_ok.write_all(&3u32.to_le_bytes()).unwrap();
        f_ok.write_all(&10u64.to_le_bytes()).unwrap();
        f_ok.write_all(&65_536u64.to_le_bytes()).unwrap();
        drop(f_ok);
        let res_ok = ContentAddressedStore::verify_format_header(&staging_ok, "gguf");
        assert!(
            res_ok.is_ok(),
            "Must accept metadata_kv_count == 65536 limit"
        );
    }
}
