//! Comprehensive Stack-Wide Security Hardening Test Suite (Milestone M4, Features 25–30, AGENTS.md §3, §5, §9).
//!
//! Verifies:
//! - Feature 25: Canonical path resolution, symlink escape rejection, strict 64-hex manifest validation, LocalFileAdapter sanitization, safe staging file creation.
//! - Feature 26: TOCTOU open FD locking (`flock(LOCK_EX | LOCK_NB)`), atomic deduplication without inode replacement, Safe GC canonical path maps scanning.
//! - Feature 27: SafeTensors & GGUF header ceilings, SafeTensors tensor offset bounds checks (`(8 + header_len) + end <= file_len`, inverted offsets), GGUF header sanity checks.
//! - Feature 28: ProtocolServer DoS defenses (10MB body limit, 64KB headers, timeouts, queue backpressure).
//! - Feature 29: DuckDB parameterized queries negative tests across tenants, projects, session_last_turns, tuning_profiles, manifests, storage_config.
//! - Feature 30: SourceCredentials fail-closed secret path resolution (AGENTS.md §3) and memory zeroization (.wipe()).

#![allow(
    clippy::pedantic,
    clippy::doc_markdown,
    clippy::undocumented_unsafe_blocks,
    clippy::field_reassign_with_default,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::needless_pass_by_value
)]

use std::fs;
use std::io::Write;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use k3_core::K3Error;
use k3_core::profile::TuningProfile;
use k3_manager::ingestion::provider::resolve_secret_value;
use k3_manager::ingestion::{
    IngestionError, LocalFileAdapter, PullRequest, SourceAdapter, SourceCredentials,
    StagingPipeline,
};
use k3_manager::persistence::{
    CasCatalog, DuckDbPersistenceStore, ModelAliasRecord, ModelBlobRecord, ModelManifestRecord,
    PersistenceStore, Project, SessionLastTurn, Tenant,
};
use k3_manager::protocol::ProtocolServerConfig;
use k3_manager::queue::AdmissionQueueManager;
use k3_manager::store::{
    ContentAddressedStore, MockProcMapsScanner, ModelManifest, ProcessMappingInfo, Sha256,
    StorageGarbageCollector, validate_sha256_digest,
};

static TEST_SEQ: AtomicU64 = AtomicU64::new(500);

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{seq}", std::process::id()));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn create_test_store() -> (
    ContentAddressedStore,
    Arc<DuckDbPersistenceStore>,
    TempDirGuard,
) {
    let temp_dir = TempDirGuard::new("k3_m4_sec_test");
    let catalog = Arc::new(DuckDbPersistenceStore::open_in_memory().expect("open catalog"));
    let store = ContentAddressedStore::open(temp_dir.path(), catalog.clone()).expect("open store");
    (store, catalog, temp_dir)
}

// =========================================================================
// FEATURE 25: Canonical Path Resolution & Symlink Confinement
// =========================================================================

#[test]
fn test_feature25_symlink_escape_attack_rejection() {
    let (store, _catalog, _tmp) = create_test_store();

    // Create an external secret file outside store root
    let secret_dir = TempDirGuard::new("outside_secret");
    let secret_file = secret_dir.path().join("secret.key");
    fs::write(&secret_file, b"super_secret_payload").unwrap();

    // Create an alias directory whose weights link points to the external secret
    let alias_dir = store
        .models_dir()
        .join("evil")
        .join("pkg")
        .join("escape")
        .join("v1");
    fs::create_dir_all(&alias_dir).unwrap();
    let evil_link = alias_dir.join("weights");
    std::os::unix::fs::symlink(&secret_file, &evil_link).unwrap();

    // resolve_alias MUST fail with confinement error
    let res = store.resolve_alias("evil/pkg/escape/v1");
    assert!(
        res.is_err(),
        "resolve_alias must reject symlink escaping store root"
    );
    let err = res.unwrap_err();
    assert!(
        err.to_string().contains("outside configured store root")
            || err.to_string().contains("not registered")
    );
}

#[test]
fn test_feature25_manifest_weight_digest_traversal_rejected() {
    let (store, _catalog, _tmp) = create_test_store();
    let malicious_manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: None,
        origin: None,
        weight_digest: "../../../../etc/passwd".to_owned(),
        weight_size_bytes: 1024,
        created_at: 1000,
    };
    let res = store.register_manifest(&malicious_manifest);
    assert!(
        res.is_err(),
        "register_manifest must reject path traversal in weight_digest"
    );
}

#[test]
fn test_feature25_validate_sha256_digest_strictness() {
    // Uppercase hex must fail
    assert!(
        validate_sha256_digest("E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855")
            .is_err()
    );
    // Leading/trailing whitespace must fail
    assert!(
        validate_sha256_digest(
            " e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 "
        )
        .is_err()
    );
    // 63 chars (too short)
    assert!(
        validate_sha256_digest("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85")
            .is_err()
    );
    // 65 chars (too long)
    assert!(
        validate_sha256_digest("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855a")
            .is_err()
    );
    // Illegal characters
    assert!(
        validate_sha256_digest("g3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            .is_err()
    );
    // Valid 64 lowercase hex
    assert!(
        validate_sha256_digest("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            .is_ok()
    );
}

#[test]
fn test_feature25_local_file_adapter_rejects_path_traversal() {
    let temp = TempDirGuard::new("local_adapter_sec");
    let adapter = LocalFileAdapter::new(temp.path());
    let staging_dir = temp.path().join("staging");
    let (staging_path, _) = StagingPipeline::create_staging_file(&staging_dir).unwrap();

    let req_traversal_tag = PullRequest {
        source: "local".into(),
        namespace: "local".into(),
        model: "safe_model".into(),
        tag: "../../etc/passwd".into(),
        expected_digest: None,
        format: None,
    };
    let res = adapter.pull(&req_traversal_tag, &staging_path, None);
    assert!(
        res.is_err(),
        "LocalFileAdapter::pull must reject directory traversal in tag"
    );

    let req_traversal_model = PullRequest {
        source: "local".into(),
        namespace: "local".into(),
        model: "../../../secret".into(),
        tag: "v1".into(),
        expected_digest: None,
        format: None,
    };
    let res2 = adapter.pull(&req_traversal_model, &staging_path, None);
    assert!(
        res2.is_err(),
        "LocalFileAdapter::pull must reject directory traversal in model"
    );
}

#[test]
fn test_feature25_staging_file_creation_symlink_safety() {
    let temp = TempDirGuard::new("staging_symlink_attack");
    let secret_file = temp.path().join("victim.txt");
    fs::write(&secret_file, b"unmodified_secret_bytes").unwrap();

    // Create a symlink in staging pointing to victim file
    let evil_staging = temp.path().join("evil.part");
    std::os::unix::fs::symlink(&secret_file, &evil_staging).unwrap();

    // Attempting to stream to the evil staging symlink must fail and not truncate victim file
    let mut reader = std::io::Cursor::new(b"attacker_payload");
    let res = StagingPipeline::stream_to_staging(&evil_staging, &mut reader, None, None, None);
    assert!(
        res.is_err(),
        "stream_to_staging must reject symlinked staging path"
    );

    // Victim file must remain intact
    let content = fs::read(&secret_file).unwrap();
    assert_eq!(
        content, b"unmodified_secret_bytes",
        "victim file must not be truncated by symlink attack"
    );
}

// =========================================================================
// FEATURE 26: TOCTOU Elimination & Open FD Locking
// =========================================================================

#[test]
fn test_feature26_staging_promotion_locked_part_file_rejected() {
    let (store, _catalog, _tmp) = create_test_store();
    let (staging_path, mut f) = store.create_staging_file().unwrap();
    f.write_all(b"valid_payload_bytes").unwrap();

    // External process takes exclusive lock on staging file
    let fd = f.as_raw_fd();
    let lock_res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(lock_res, 0);

    // Concurrent promote must detect locked staging file and refuse promotion
    let digest = Sha256::digest(b"valid_payload_bytes");
    let res = store.promote_staged_blob(&staging_path, &digest, "raw");
    assert!(
        res.is_err(),
        "promote_staged_blob must fail when staging file is locked"
    );

    // Release lock
    unsafe { libc::flock(fd, libc::LOCK_UN) };
}

#[test]
fn test_feature26_atomic_deduplication_without_overwriting_inode() {
    use std::os::unix::fs::MetadataExt;

    let (store, _catalog, _tmp) = create_test_store();
    let payload = b"shared_dedup_bytes";
    let digest = Sha256::digest(payload);

    // Ingest first blob
    let (staging1, mut f1) = store.create_staging_file().unwrap();
    f1.write_all(payload).unwrap();
    drop(f1);
    let blob_path1 = store
        .promote_staged_blob(&staging1, &digest, "raw")
        .unwrap();
    let inode1 = fs::metadata(&blob_path1).unwrap().ino();

    // Ingest identical blob a second time (concurrent promotion simulation)
    let (staging2, mut f2) = store.create_staging_file().unwrap();
    f2.write_all(payload).unwrap();
    drop(f2);
    let blob_path2 = store
        .promote_staged_blob(&staging2, &digest, "raw")
        .unwrap();
    let inode2 = fs::metadata(&blob_path2).unwrap().ino();

    // Must return the exact same blob and maintain unchanged inode (non-replacing dedup)
    assert_eq!(blob_path1, blob_path2);
    assert_eq!(
        inode1, inode2,
        "atomic deduplication must preserve original inode without replacement"
    );
    assert!(!staging2.exists(), "second staging file must be unlinked");
}

#[test]
fn test_feature26_safe_gc_canonical_path_matching_under_symlinked_root() {
    let root_temp = TempDirGuard::new("real_store_root");
    let symlink_temp = TempDirGuard::new("symlink_dir");
    let symlinked_root = symlink_temp.path().join("store_link");
    std::os::unix::fs::symlink(root_temp.path(), &symlinked_root).unwrap();

    let catalog = Arc::new(DuckDbPersistenceStore::open_in_memory().unwrap());
    let store = Arc::new(ContentAddressedStore::open(&symlinked_root, catalog).unwrap());
    let mock_scanner = Arc::new(MockProcMapsScanner::new());
    let gc = StorageGarbageCollector::with_scanner(store.clone(), mock_scanner.clone());

    // Ingest blob
    let payload = b"mapped_under_symlink";
    let digest = Sha256::digest(payload);
    let (staging, mut f) = store.create_staging_file().unwrap();
    f.write_all(payload).unwrap();
    drop(f);
    let blob_path = store.promote_staged_blob(&staging, &digest, "raw").unwrap();

    // Kernel maps report the CANONICAL path
    let canonical_blob_path = fs::canonicalize(&blob_path).unwrap();
    mock_scanner.add_mapping(
        &canonical_blob_path,
        ProcessMappingInfo {
            pid: 9999,
            comm: "k3-engine".into(),
            address_range: "0x1000-0x2000".into(),
            permissions: "r--p".into(),
            mapped_path: canonical_blob_path.to_string_lossy().to_string(),
        },
    );

    // GC must match mapping under canonical path and refuse to delete
    let report = gc.run_gc(false).unwrap();
    assert_eq!(report.reclaimed_blobs.len(), 0);
    assert!(
        blob_path.exists(),
        "blob must not be unlinked when mapped under canonical path"
    );
}

#[test]
fn test_feature26_who_uses_nonexistent_and_symlinked_root_regression() {
    // 1. Setup store under a symlinked root to rigorously simulate macOS /var -> /private/var
    let root_temp = TempDirGuard::new("real_store_root_who_uses");
    let symlink_temp = TempDirGuard::new("symlink_dir_who_uses");
    let symlinked_root = symlink_temp.path().join("store_link");
    std::os::unix::fs::symlink(root_temp.path(), &symlinked_root).unwrap();

    let catalog = Arc::new(DuckDbPersistenceStore::open_in_memory().unwrap());
    let store = Arc::new(ContentAddressedStore::open(&symlinked_root, catalog).unwrap());
    let mock_scanner = Arc::new(MockProcMapsScanner::new());
    let gc = StorageGarbageCollector::with_scanner(store.clone(), mock_scanner);

    // 2. Query non-existent 64-hex digest: must return "not found in store", NEVER false-positive path traversal
    let non_existent_digest = "0000000000000000000000000000000000000000000000000000000000000000";
    let res_missing = gc.who_uses(non_existent_digest);
    assert!(
        res_missing.is_err(),
        "who_uses on non-existent digest must return error"
    );
    match res_missing.unwrap_err() {
        K3Error::CapabilityMismatch {
            architecture,
            reason,
        } => {
            assert_eq!(architecture, "cas_gc");
            assert!(
                reason.contains("not found in store"),
                "who_uses on missing digest must report 'not found in store', got: '{reason}'"
            );
            assert!(
                !reason.contains("path traversal detected"),
                "who_uses must not trigger false-positive path traversal: '{reason}'"
            );
        }
        other => panic!("Expected CapabilityMismatch, got {other:?}"),
    }

    // 3. Ingest an authentic blob and verify who_uses succeeds under symlinked root
    let payload = b"authenticated_weights_data";
    let digest = Sha256::digest(payload);
    let (staging, mut f) = store.create_staging_file().unwrap();
    f.write_all(payload).unwrap();
    drop(f);
    let _blob_path = store.promote_staged_blob(&staging, &digest, "raw").unwrap();

    let info = gc
        .who_uses(&digest)
        .expect("who_uses on existing blob must succeed");
    assert_eq!(info.digest, digest);
    assert_eq!(info.size_bytes, payload.len() as u64);
    assert!(!info.gate1_shared_lock_held);
    assert_eq!(info.gate2_active_mappings.len(), 0);

    // 4. Verify genuine symlink escape attack is still detected and rejected
    let secret_temp = TempDirGuard::new("secret_outside");
    let secret_file = secret_temp.path().join("external_secret.bin");
    fs::write(&secret_file, b"top_secret").unwrap();

    // Create a symlink in blobs_dir with a valid 64-hex name pointing outside
    let malicious_digest = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    let malicious_symlink = store.blobs_dir().join(malicious_digest);
    std::os::unix::fs::symlink(&secret_file, &malicious_symlink).unwrap();

    let res_escape = gc.who_uses(malicious_digest);
    assert!(
        res_escape.is_err(),
        "who_uses must detect symlink escape attack"
    );
    match res_escape.unwrap_err() {
        K3Error::CapabilityMismatch {
            architecture,
            reason,
        } => {
            assert_eq!(architecture, "cas_gc");
            assert!(
                reason.contains("path traversal detected in digest"),
                "expected path traversal detection for external symlink, got: '{reason}'"
            );
        }
        other => panic!("Expected CapabilityMismatch, got {other:?}"),
    }
}

// =========================================================================
// FEATURE 27: Bounds Checking & Format Header Verification
// =========================================================================

#[test]
fn test_feature27_safetensors_header_ceiling_rejected() {
    let (store, _catalog, _tmp) = create_test_store();
    let (staging, mut f) = store.create_staging_file().unwrap();

    // Header length 150 MB (exceeds 100 MB ceiling)
    let bad_len = (150 * 1024 * 1024u64).to_le_bytes();
    f.write_all(&bad_len).unwrap();
    f.write_all(&[0u8; 100]).unwrap();
    drop(f);

    let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
    assert!(
        res.is_err(),
        "verify_format_header must reject SafeTensors header > 100MB"
    );
    let err = res.unwrap_err();
    assert!(
        err.to_string()
            .contains("invalid SafeTensors header length")
    );
}

#[test]
fn test_feature27_safetensors_truncated_cas_header() {
    let (store, _catalog, _tmp) = create_test_store();
    let (staging, mut f) = store.create_staging_file().unwrap();

    // Claim header_len = 5000, but file only has 8 bytes total
    f.write_all(&5000u64.to_le_bytes()).unwrap();
    drop(f);

    let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
    assert!(
        res.is_err(),
        "verify_format_header must reject truncated SafeTensors file"
    );
    let err = res.unwrap_err();
    assert!(err.to_string().contains("truncated"));
}

#[test]
fn test_feature27_safetensors_oob_data_offsets_rejected() {
    let (store, _catalog, _tmp) = create_test_store();
    let (staging, mut f) = store.create_staging_file().unwrap();

    // Valid header length with out-of-bounds offset in JSON metadata
    let header_json = serde_json::json!({
        "tensor_1": {
            "dtype": "F32",
            "shape": [1, 10],
            "data_offsets": [0, 99999999] // Exceeds file size
        }
    })
    .to_string();

    let header_len = header_json.len() as u64;
    f.write_all(&header_len.to_le_bytes()).unwrap();
    f.write_all(header_json.as_bytes()).unwrap();
    // Add 40 bytes of payload
    f.write_all(&[0u8; 40]).unwrap();
    drop(f);

    let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
    assert!(
        res.is_err(),
        "verify_format_header must reject SafeTensors with data_offsets exceeding file size"
    );
    let err = res.unwrap_err();
    assert!(err.to_string().contains("exceeds file length"));
}

#[test]
fn test_feature27_safetensors_inverted_data_offsets_rejected() {
    let (store, _catalog, _tmp) = create_test_store();
    let (staging, mut f) = store.create_staging_file().unwrap();

    let header_json = serde_json::json!({
        "tensor_inverted": {
            "dtype": "F32",
            "shape": [1, 10],
            "data_offsets": [100, 50] // Inverted offsets: begin > end
        }
    })
    .to_string();

    let header_len = header_json.len() as u64;
    f.write_all(&header_len.to_le_bytes()).unwrap();
    f.write_all(header_json.as_bytes()).unwrap();
    f.write_all(&[0u8; 200]).unwrap();
    drop(f);

    let res = ContentAddressedStore::verify_format_header(&staging, "safetensors");
    assert!(
        res.is_err(),
        "verify_format_header must reject SafeTensors with inverted data_offsets"
    );
    let err = res.unwrap_err();
    assert!(err.to_string().contains("inverted data_offsets"));
}

#[test]
fn test_feature27_gguf_header_sanity_checks() {
    let (store, _catalog, _tmp) = create_test_store();
    let (staging, mut f) = store.create_staging_file().unwrap();

    // Write magic GGUF, version 3, but tensor_count = 1_000_000 (> 65536)
    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&1_000_000u64.to_le_bytes()).unwrap(); // tensor_count
    f.write_all(&10u64.to_le_bytes()).unwrap(); // kv_count
    drop(f);

    let res = ContentAddressedStore::verify_format_header(&staging, "gguf");
    assert!(
        res.is_err(),
        "verify_format_header must reject GGUF tensor_count > 65536"
    );
    let err = res.unwrap_err();
    assert!(err.to_string().contains("tensor_count"));
}

// =========================================================================
// FEATURE 28: HTTP/SSE Denial-of-Service Defenses
// =========================================================================

#[test]
fn test_feature28_protocol_server_config_bounds() {
    let config = ProtocolServerConfig::default();
    assert_eq!(
        config.max_request_body_bytes,
        10 * 1024 * 1024,
        "default body limit must be exactly 10MB"
    );
    assert_eq!(
        config.max_headers_bytes,
        64 * 1024,
        "default headers limit must be 64KB"
    );
    assert!(
        config.socket_timeout >= Duration::from_secs(10),
        "socket timeout must be reasonable"
    );
}

#[test]
fn test_feature28_queue_saturation_retry_after_bounds() {
    let queue_mgr = AdmissionQueueManager::default();
    queue_mgr.register_model("model-a", None);

    // Fill queue to capacity (default depth is 64)
    for i in 1..=64 {
        let res = queue_mgr.admit_to_model(
            "model-a",
            format!("req_{i}"),
            false,
            None,
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            None,
        );
        assert!(res.is_ok(), "request within capacity must be admitted");
    }

    // 65th request breaches capacity -> returns Err(QueueSaturated)
    let overflow = queue_mgr.admit_to_model(
        "model-a",
        "overflow_req".to_string(),
        false,
        None,
        None,
        Arc::new(AtomicBool::new(false)),
        None,
        None,
    );
    assert!(overflow.is_err(), "overflow request must be rejected");

    if let Err(K3Error::QueueSaturated {
        retry_after_secs, ..
    }) = overflow
    {
        assert!(
            (1..=86400).contains(&retry_after_secs),
            "retry_after_secs must be clamped within [1, 86400]"
        );
    } else {
        panic!("expected QueueSaturated error variant");
    }
}

// =========================================================================
// FEATURE 29: DuckDB Parameterized Query Enforcement
// =========================================================================

#[test]
fn test_feature29_duckdb_injection_resistance_across_all_schema_tables() {
    let (_store, catalog, _tmp) = create_test_store();

    let payloads = [
        "'; DROP TABLE tenants; --",
        "' OR '1'='1",
        "1; DELETE FROM model_manifests; --",
        "admin'--",
        "UNION SELECT * FROM sqlite_master --",
        "payload' OR 1=1; SELECT pg_sleep(10); --",
    ];

    let base_blob = ModelBlobRecord {
        digest: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_owned(),
        size_bytes: 1024,
        blob_type: "raw_weights".to_owned(),
        created_at: 100,
        last_accessed_at: 100,
        ref_count: 0,
    };
    catalog.save_blob(&base_blob).unwrap();

    for (i, payload) in payloads.iter().enumerate() {
        // 1. Tenants
        let tenant_id = format!("t_sec_{i}");
        let tenant = Tenant {
            id: tenant_id.clone(),
            name: (*payload).to_owned(),
            created_at: 100,
        };
        assert!(catalog.save_tenant(&tenant).is_ok());
        let got_tenant = catalog.get_tenant(&tenant_id).unwrap().unwrap();
        assert_eq!(got_tenant.name, *payload);

        // 2. Projects
        let proj_id = format!("p_sec_{i}");
        let project = Project {
            id: proj_id.clone(),
            tenant_id: tenant_id.clone(),
            name: (*payload).to_owned(),
            token_budget: Some(1000),
            created_at: 100,
        };
        assert!(catalog.save_project(&project).is_ok());
        let got_proj = catalog.get_project(&proj_id).unwrap().unwrap();
        assert_eq!(got_proj.name, *payload);

        // 3. Session last turns
        let sess_id = format!("sess_sec_{i}");
        let turn = SessionLastTurn {
            session_id: sess_id.clone(),
            project_id: proj_id.clone(),
            tenant_id: tenant_id.clone(),
            model_id: (*payload).to_owned(),
            prompt: (*payload).to_owned(),
            response: (*payload).to_owned(),
            prompt_tokens: 10,
            completion_tokens: 20,
            parameters_json: (*payload).to_owned(),
            updated_at: 100,
        };
        assert!(catalog.save_session_last_turn(&turn).is_ok());
        let got_turn = catalog.load_session_last_turn(&sess_id).unwrap().unwrap();
        assert_eq!(got_turn.prompt, *payload);
        assert_eq!(got_turn.response, *payload);

        // 4. Tuning profiles
        let model_key = format!("model_sec_{i}");
        let mut prof = TuningProfile::default();
        prof.cluster = (*payload).to_owned();
        prof.advisory_notice = Some((*payload).to_owned());
        assert!(catalog.save_tuning_profile(&model_key, &prof).is_ok());
        let got_prof = catalog.load_tuning_profile(&model_key).unwrap().unwrap();
        assert_eq!(got_prof.cluster, *payload);

        // 5. Storage config
        let cfg_key = format!("config_sec_{i}");
        assert!(catalog.set_storage_config(&cfg_key, payload).is_ok());
        let got_cfg = catalog.get_storage_config(&cfg_key).unwrap().unwrap();
        assert_eq!(got_cfg, *payload);

        // 6. Model manifests
        let manifest_rec = ModelManifestRecord {
            manifest_digest: format!("man_sec_{i}"),
            weight_digest: base_blob.digest.clone(),
            format: (*payload).to_owned(),
            architecture_id: (*payload).to_owned(),
            recipe_version: (*payload).to_owned(),
            quantization: (*payload).to_owned(),
            context_length: 2048,
            license: Some((*payload).to_owned()),
            origin: Some((*payload).to_owned()),
            manifest_json: (*payload).to_owned(),
            created_at: 100,
        };
        assert!(catalog.save_manifest(&manifest_rec).is_ok());
        let got_man = catalog
            .get_manifest(&manifest_rec.manifest_digest)
            .unwrap()
            .unwrap();
        assert_eq!(got_man.architecture_id, *payload);

        // 7. Model aliases
        let alias_rec = ModelAliasRecord {
            alias_id: format!("alias_sec_{i}"),
            source: "local".to_owned(),
            namespace: "ns".to_owned(),
            name: format!("model_{i}"),
            tag: format!("tag_{i}"),
            manifest_digest: manifest_rec.manifest_digest.clone(),
            weight_digest: base_blob.digest.clone(),
            alias_path: (*payload).to_owned(),
            created_at: 100,
        };
        assert!(catalog.save_alias(&alias_rec).is_ok());
        let got_alias = catalog.get_alias(&alias_rec.alias_id).unwrap().unwrap();
        assert_eq!(got_alias.alias_path, *payload);
    }
}

// =========================================================================
// FEATURE 30: Secret Resolution Failing Closed & Memory Zeroization
// =========================================================================

#[test]
fn test_feature30_resolve_secret_value_fail_closed_on_missing_or_unreadable() {
    let temp = TempDirGuard::new("secret_fail_closed");

    // 1. Missing absolute path -> None
    assert_eq!(
        resolve_secret_value("/nonexistent/k3_token.env"),
        None,
        "missing absolute path must fail closed"
    );

    // 2. Missing relative path -> None
    assert_eq!(
        resolve_secret_value("./missing_token.env"),
        None,
        "missing relative path must fail closed"
    );
    assert_eq!(
        resolve_secret_value("../missing_token.env"),
        None,
        "missing parent traversal path must fail closed"
    );

    // 3. Directory path -> None
    assert_eq!(
        resolve_secret_value(temp.path().to_str().unwrap()),
        None,
        "directory path passed as secret must fail closed"
    );

    // 3b. Bare directory traversal components -> None
    assert_eq!(
        resolve_secret_value("."),
        None,
        "bare '.' current directory must fail closed"
    );
    assert_eq!(
        resolve_secret_value(".."),
        None,
        "bare '..' parent directory must fail closed"
    );

    // 4. Empty file -> None
    let empty_file = temp.path().join("empty.key");
    fs::write(&empty_file, b"").unwrap();
    assert_eq!(
        resolve_secret_value(empty_file.to_str().unwrap()),
        None,
        "empty secret file must fail closed"
    );

    // 5. Whitespace-only file -> None
    let whitespace_file = temp.path().join("whitespace.key");
    fs::write(&whitespace_file, b"   \r\n\t   ").unwrap();
    assert_eq!(
        resolve_secret_value(whitespace_file.to_str().unwrap()),
        None,
        "whitespace-only secret file must fail closed"
    );

    // 6. Valid file with whitespace -> Some(trimmed)
    let valid_file = temp.path().join("valid.key");
    fs::write(&valid_file, b"  my_secret_token_12345\n").unwrap();
    assert_eq!(
        resolve_secret_value(valid_file.to_str().unwrap()),
        Some("my_secret_token_12345".to_string()),
        "valid secret file must resolve trimmed content"
    );

    // 7. Literal non-path string -> Some(val)
    assert_eq!(
        resolve_secret_value("literal_secret_key_abc"),
        Some("literal_secret_key_abc".to_string())
    );

    // 8. Empty or whitespace string -> None
    assert_eq!(resolve_secret_value(""), None);
    assert_eq!(resolve_secret_value("   \t  "), None);
}

#[test]
fn test_feature30_source_credentials_from_secret_string() {
    let temp = TempDirGuard::new("source_creds_test");

    // Fail closed on invalid path
    let bad_res = SourceCredentials::from_secret_string("/nonexistent/file.env");
    assert!(
        bad_res.is_err(),
        "must fail closed with error on unreadable path"
    );
    assert!(matches!(
        bad_res.unwrap_err(),
        IngestionError::CredentialResolutionFailed(_)
    ));

    // Valid path resolves secret file contents
    let secret_file = temp.path().join("token.txt");
    fs::write(&secret_file, "hf_file_secret_key\n").unwrap();
    let file_res = SourceCredentials::from_secret_string(secret_file.to_str().unwrap());
    assert!(file_res.is_ok(), "must resolve valid secret file");
    assert_eq!(
        file_res.unwrap(),
        SourceCredentials::Token("hf_file_secret_key".to_string())
    );

    // Literal token succeeds
    let ok_res = SourceCredentials::from_secret_string("hf_api_key_valid");
    assert!(ok_res.is_ok());
    assert_eq!(
        ok_res.unwrap(),
        SourceCredentials::Token("hf_api_key_valid".to_string())
    );

    // Empty string yields Anonymous
    let anon_res = SourceCredentials::from_secret_string("   ");
    assert!(anon_res.is_ok());
    assert_eq!(anon_res.unwrap(), SourceCredentials::Anonymous);
}

#[test]
fn test_feature30_credential_zeroization_and_wipe() {
    // 1. Token wipe
    let mut token_creds = SourceCredentials::Token("sensitive_key_to_wipe".to_string());
    token_creds.wipe();
    assert_eq!(
        token_creds,
        SourceCredentials::Anonymous,
        "wipe must reset token credentials to Anonymous"
    );

    // 2. BearerToken wipe
    let mut bearer_creds = SourceCredentials::BearerToken("bearer_secret_123".to_string());
    bearer_creds.wipe();
    assert_eq!(bearer_creds, SourceCredentials::Anonymous);

    // 3. BasicAuth wipe
    let mut basic_creds = SourceCredentials::BasicAuth {
        username: "admin".to_string(),
        password_or_token: "super_secret_pw".to_string(),
    };
    basic_creds.wipe();
    assert_eq!(basic_creds, SourceCredentials::Anonymous);
}

#[test]
fn test_feature30_debug_format_redaction() {
    let secret = "leaked_super_token_999";
    let token_creds = SourceCredentials::Token(secret.to_string());
    let debug_str = format!("{token_creds:?}");
    assert!(
        debug_str.contains("[REDACTED]"),
        "debug output must redact secret token"
    );
    assert!(
        !debug_str.contains(secret),
        "debug output must never contain raw plaintext secret"
    );

    let bearer_creds = SourceCredentials::BearerToken(secret.to_string());
    let debug_bearer = format!("{bearer_creds:?}");
    assert!(debug_bearer.contains("[REDACTED]"));
    assert!(!debug_bearer.contains(secret));
}

#[test]
fn test_feature30_authentic_volatile_zeroization() {
    let mut secret = "super_secret_token_to_verify_volatile_wipe_12345".to_string();
    let ptr = secret.as_ptr();
    let len = secret.len();
    assert_ne!(unsafe { *ptr }, 0);

    k3_manager::ingestion::provider::volatile_zeroize_string(&mut secret);

    assert_eq!(secret.len(), 0);
    // Directly verify that all bytes in the backing buffer were overwritten with 0
    for i in 0..len {
        assert_eq!(
            unsafe { *ptr.add(i) },
            0,
            "byte at offset {i} was not overwritten with volatile zero"
        );
    }
}
