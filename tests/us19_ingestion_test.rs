//! US-19 Ingestion & Staging Verification Test Suite.
//!
//! Verifies:
//! - SourceAdapter lifecycle: auth, search, list, pull
//! - Credential isolation & fail-closed resolution
//! - Staging file creation with 0600 permissions
//! - Streaming SHA-256 computation
//! - Interruption/corruption safety & automatic unlinking of .part files
//! - Zerocopy format header validation (GGUF & SafeTensors)
//! - Atomic promotion into blobs/sha256/<digest> with 0444 permissions
//! - Automatic deduplication
//! - Provider registry & multi-repository adapters

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::similar_names
)]

use std::fs;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use k3_manager::ingestion::{
    AuthStatus, GgufHeaderPrefix, HuggingFaceAdapter, IngestionError, LocalFileAdapter,
    ModelFormat, OllamaAdapter, PullRequest, SafeTensorsHeaderPrefix, SearchFilter, SourceAdapter,
    SourceAdapterRegistry, SourceCredentials, StagingPipeline,
};
use k3_manager::persistence::DuckDbPersistenceStore;
use k3_manager::store::Sha256;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(1);

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new(prefix: &str) -> Self {
        let seq = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("k3_test_{prefix}_{pid}_{seq}"));
        fs::create_dir_all(&path).expect("failed to create test directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn create_valid_gguf_payload(model_name: &str) -> Vec<u8> {
    let header = GgufHeaderPrefix {
        magic: *b"GGUF",
        version: 3,
        tensor_count: 16,
        metadata_kv_count: 4,
    };
    let mut data = Vec::new();
    let header_bytes = zerocopy::IntoBytes::as_bytes(&header);
    data.extend_from_slice(header_bytes);
    data.extend_from_slice(format!("VALID_GGUF_WEIGHT_BYTES_FOR_{model_name}").as_bytes());
    data
}

fn create_valid_safetensors_payload() -> Vec<u8> {
    let json_meta = br#"{"__metadata__":{"format":"pt"},"w":{"dtype":"I8","shape":[8,16],"data_offsets":[0,128]}}"#;
    let header = SafeTensorsHeaderPrefix {
        header_len: json_meta.len() as u64,
    };
    let mut data = Vec::new();
    let header_bytes = zerocopy::IntoBytes::as_bytes(&header);
    data.extend_from_slice(header_bytes);
    data.extend_from_slice(json_meta);
    data.extend_from_slice(&[42u8; 128]); // weight data
    data
}

#[test]
fn test_source_adapter_lifecycle_huggingface() {
    let adapter = HuggingFaceAdapter::new();
    assert_eq!(adapter.source_name(), "huggingface");

    // 1. Auth anonymous
    let anon_status = adapter.auth(&SourceCredentials::Anonymous).unwrap();
    assert_eq!(anon_status, AuthStatus::Anonymous);

    // 2. Auth with token
    let creds = SourceCredentials::Token("hf_test_token_abcdef123456".to_owned());
    let auth_status = adapter.auth(&creds).unwrap();
    match auth_status {
        AuthStatus::Authenticated {
            identity,
            permissions,
        } => {
            assert_eq!(identity, "hf_developer");
            assert!(permissions.contains(&"read".to_owned()));
        }
        other => panic!("expected authenticated, got {other:?}"),
    }

    // 3. Search
    let results = adapter.search("qwen", &SearchFilter::default()).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].id.contains("Qwen"));

    // 4. List revisions
    let listing = adapter.list("Qwen", "Qwen2.5-7B-Instruct").unwrap();
    assert_eq!(listing.namespace, "Qwen");
    assert!(!listing.revisions.is_empty());
    assert!(listing.revisions.iter().any(|r| r.tag == "q8_0"));

    // 5. Pull into staging
    let test_dir = TestDir::new("hf_pull");
    let staging_path = test_dir.path().join("qwen2.5-7b-q8_0.part");

    let req = PullRequest {
        source: "huggingface".to_owned(),
        namespace: "Qwen".to_owned(),
        model: "Qwen2.5-7B-Instruct".to_owned(),
        tag: "q8_0".to_owned(),
        expected_digest: None,
        format: Some(ModelFormat::Gguf),
    };

    let summary = adapter.pull(&req, &staging_path, None).unwrap();
    assert!(staging_path.exists());
    assert_eq!(summary.format, ModelFormat::Gguf);
    assert!(!summary.computed_digest.is_empty());
    assert!(summary.bytes_downloaded > 0);
}

#[test]
fn test_credential_isolation_and_fail_closed_resolution() {
    let test_dir = TestDir::new("cred_isolation");

    // Case 1: Valid secret file resolves and trims whitespace
    let secret_file = test_dir.path().join("valid_token.env");
    fs::write(&secret_file, "  hf_my_super_secret_token_12345 \n").unwrap();
    let creds = SourceCredentials::SecretFile(secret_file);
    let resolved = creds.resolve_secret().unwrap();
    assert_eq!(resolved.as_deref(), Some("hf_my_super_secret_token_12345"));

    // Case 2: Non-existent secret file fails closed (returns Error, never returns path string!)
    let missing_file = test_dir.path().join("missing_secret.env");
    let missing_creds = SourceCredentials::SecretFile(missing_file);
    let res = missing_creds.resolve_secret();
    assert!(res.is_err(), "missing secret file must fail closed");
    let err_str = res.unwrap_err().to_string();
    assert!(err_str.contains("does not exist"));

    // Case 3: Empty secret file fails closed
    let empty_file = test_dir.path().join("empty_secret.env");
    fs::write(&empty_file, "   \n\t").unwrap();
    let empty_creds = SourceCredentials::SecretFile(empty_file);
    let res_empty = empty_creds.resolve_secret();
    assert!(res_empty.is_err(), "empty secret file must fail closed");

    // Case 4: Zero credential leakage in Debug format
    let token_creds = SourceCredentials::Token("super_secret_api_key_xyz".to_owned());
    let debug_str = format!("{token_creds:?}");
    assert!(!debug_str.contains("super_secret_api_key_xyz"));
    assert!(debug_str.contains("[REDACTED]"));

    let bearer_creds = SourceCredentials::BearerToken("secret_bearer_token".to_owned());
    let bearer_debug = format!("{bearer_creds:?}");
    assert!(!bearer_debug.contains("secret_bearer_token"));
    assert!(bearer_debug.contains("[REDACTED]"));

    // Case 5: Auth rejection does not leak secret token
    let adapter = HuggingFaceAdapter::new();
    let bad_creds = SourceCredentials::Token("bad_token_value_999".to_owned());
    let auth_res = adapter.auth(&bad_creds);
    assert!(auth_res.is_err());
    let auth_err_msg = auth_res.unwrap_err().to_string();
    assert!(!auth_err_msg.contains("bad_token_value_999"));
}

#[test]
fn test_staging_file_restricted_permissions_and_streaming_hash() {
    let test_dir = TestDir::new("staging_perms");
    let staging_dir = test_dir.path().join("staging");

    // Create staging file
    let (part_path, _) = StagingPipeline::create_staging_file(&staging_dir).unwrap();
    assert!(part_path.exists());
    assert!(part_path.to_string_lossy().ends_with(".part"));

    // Verify 0600 permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::metadata(&part_path).unwrap().permissions();
        let mode = perms.mode() & 0o777;
        assert_eq!(mode, 0o600, "staging file must have 0600 permissions");
    }

    // Stream chunks into staging file
    let payload = create_valid_gguf_payload("test_model");
    let expected_hash = Sha256::digest(&payload);

    let mut cursor = Cursor::new(payload);
    let (computed_digest, bytes_written) = StagingPipeline::stream_to_staging(
        &part_path,
        &mut cursor,
        Some(&expected_hash),
        None,
        None,
    )
    .unwrap();

    assert_eq!(computed_digest, expected_hash);
    assert!(bytes_written > 0);
}

#[test]
fn test_interrupted_download_staging_cleanup() {
    let test_dir = TestDir::new("interrupted_dl");
    let staging_dir = test_dir.path().join("staging");
    let blobs_dir = test_dir.path().join("blobs/sha256");
    fs::create_dir_all(&staging_dir).unwrap();
    fs::create_dir_all(&blobs_dir).unwrap();

    let adapter = HuggingFaceAdapter::new();
    adapter.set_simulated_fault(Some("network_drop"));

    let staging_path = staging_dir.join("partial_network_drop.part");
    let req = PullRequest {
        source: "huggingface".to_owned(),
        namespace: "Qwen".to_owned(),
        model: "Qwen2.5-7B".to_owned(),
        tag: "q8_0".to_owned(),
        expected_digest: None,
        format: Some(ModelFormat::Gguf),
    };

    let result = adapter.pull(&req, &staging_path, None);
    assert!(result.is_err(), "simulated network drop must return error");

    // Invariant: .part file unlinked immediately, blobs dir uncontaminated
    assert!(
        !staging_path.exists(),
        "interrupted staging file must be unlinked"
    );
    let blob_count = fs::read_dir(&blobs_dir).unwrap().count();
    assert_eq!(
        blob_count, 0,
        "blobs directory must remain completely empty"
    );
}

#[test]
fn test_digest_mismatch_rejected_and_staging_cleaned_up() {
    let test_dir = TestDir::new("digest_mismatch");
    let staging_dir = test_dir.path().join("staging");
    let blobs_dir = test_dir.path().join("blobs/sha256");
    fs::create_dir_all(&staging_dir).unwrap();
    fs::create_dir_all(&blobs_dir).unwrap();

    let (part_path, _) = StagingPipeline::create_staging_file(&staging_dir).unwrap();
    let payload = create_valid_gguf_payload("corrupted_digest");
    let declared_bad_hash = "0000000000000000000000000000000000000000000000000000000000000000";

    let mut cursor = Cursor::new(payload);
    let result = StagingPipeline::stream_to_staging(
        &part_path,
        &mut cursor,
        Some(declared_bad_hash),
        None,
        None,
    );

    assert!(result.is_err());
    match result.unwrap_err() {
        IngestionError::DigestMismatch { expected, actual } => {
            assert_eq!(expected, declared_bad_hash);
            assert_ne!(actual, declared_bad_hash);
        }
        other => panic!("expected DigestMismatch, got {other:?}"),
    }

    // Staging file deleted, store untouched
    assert!(!part_path.exists());
    let blob_count = fs::read_dir(&blobs_dir).unwrap().count();
    assert_eq!(blob_count, 0);
}

#[test]
fn test_zerocopy_format_header_verification() {
    let test_dir = TestDir::new("format_verify");
    let staging_dir = test_dir.path().join("staging");

    // 1. Valid GGUF passes
    let (valid_gguf_path, mut f) = StagingPipeline::create_staging_file(&staging_dir).unwrap();
    let gguf_bytes = create_valid_gguf_payload("test_gguf");
    f.write_all(&gguf_bytes).unwrap();
    drop(f);

    assert!(StagingPipeline::verify_format_header(&valid_gguf_path, &ModelFormat::Gguf).is_ok());

    // 2. Corrupted GGUF magic rejected and unlinked
    let (corrupt_gguf_path, mut f2) = StagingPipeline::create_staging_file(&staging_dir).unwrap();
    f2.write_all(b"BADMAGICBYTESNOTGGUFHEADERDATA").unwrap();
    drop(f2);

    let verify_res = StagingPipeline::verify_format_header(&corrupt_gguf_path, &ModelFormat::Gguf);
    assert!(verify_res.is_err());
    assert!(
        !corrupt_gguf_path.exists(),
        "corrupted file must be unlinked"
    );

    // 3. Valid SafeTensors passes
    let (valid_st_path, mut f3) = StagingPipeline::create_staging_file(&staging_dir).unwrap();
    let st_bytes = create_valid_safetensors_payload();
    f3.write_all(&st_bytes).unwrap();
    drop(f3);

    assert!(
        StagingPipeline::verify_format_header(&valid_st_path, &ModelFormat::SafeTensors).is_ok()
    );

    // 4. Corrupted SafeTensors length rejected and unlinked
    let (corrupt_st_path, mut f4) = StagingPipeline::create_staging_file(&staging_dir).unwrap();
    f4.write_all(&[0xFF; 8]).unwrap(); // huge invalid header length
    f4.write_all(b"not_valid_json").unwrap();
    drop(f4);

    let verify_st_res =
        StagingPipeline::verify_format_header(&corrupt_st_path, &ModelFormat::SafeTensors);
    assert!(verify_st_res.is_err());
    assert!(
        !corrupt_st_path.exists(),
        "corrupted SafeTensors file must be unlinked"
    );
}

#[test]
fn test_atomic_promotion_and_deduplication() {
    let test_dir = TestDir::new("promotion_dedup");
    let store_root = test_dir.path();
    let pipeline = StagingPipeline::new(store_root);

    let db_path = store_root.join("catalog.duckdb");
    let catalog = Arc::new(DuckDbPersistenceStore::open(&db_path).unwrap())
        as Arc<dyn k3_manager::persistence::CasCatalog>;

    let payload = create_valid_gguf_payload("dedup_model");
    let digest = Sha256::digest(&payload);

    // Run 1: First promotion creates blob
    let mut cursor1 = Cursor::new(payload.clone());
    let download1 = pipeline
        .ingest_and_promote(
            &mut cursor1,
            Some(&digest),
            ModelFormat::Gguf,
            Some(&catalog),
        )
        .unwrap();

    assert!(download1.staging_path.exists());
    assert_eq!(download1.computed_digest, digest);

    // Read-only 0444 on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::metadata(&download1.staging_path).unwrap().permissions();
        assert_eq!(perms.mode() & 0o777, 0o444);
    }

    let blob_rec1 = catalog
        .get_blob(&digest)
        .unwrap()
        .expect("blob must be in catalog");
    assert_eq!(blob_rec1.ref_count, 1);

    // Run 2: Second promotion of identical weights (Deduplication)
    let (part_path2, mut f2) =
        StagingPipeline::create_staging_file(&pipeline.staging_dir()).unwrap();
    f2.write_all(&payload).unwrap();
    drop(f2);

    let promoted_path2 = StagingPipeline::promote_to_cas(
        &part_path2,
        &digest,
        &pipeline.blobs_dir(),
        Some(&catalog),
    )
    .unwrap();

    // Promoted path matches first path, part_path2 unlinked
    assert_eq!(promoted_path2, download1.staging_path);
    assert!(
        !part_path2.exists(),
        "staged part file must be unlinked after dedup"
    );

    let blob_rec2 = catalog.get_blob(&digest).unwrap().unwrap();
    assert_eq!(
        blob_rec2.ref_count, 2,
        "catalog ref_count must be incremented on dedup"
    );
}

#[test]
fn test_source_adapter_registry_and_providers() {
    let registry = SourceAdapterRegistry::with_defaults();

    // Lookup Hugging Face adapter
    let hf = registry
        .get_adapter("huggingface")
        .expect("huggingface adapter must exist");
    assert_eq!(hf.source_name(), "huggingface");

    // Lookup Ollama adapter
    let ollama = registry
        .get_adapter("ollama")
        .expect("ollama adapter must exist");
    assert_eq!(ollama.source_name(), "ollama");

    // Register LocalFileAdapter
    let test_dir = TestDir::new("local_adapter");
    let local = Arc::new(LocalFileAdapter::new(test_dir.path()));
    registry.register(local);

    let local_lookup = registry
        .get_adapter("local")
        .expect("local adapter must exist");
    assert_eq!(local_lookup.source_name(), "local");

    // Unknown source returns None
    assert!(registry.get_adapter("unknown_provider").is_none());
}

#[test]
fn test_progress_reporting_during_pull() {
    let adapter = HuggingFaceAdapter::new();
    let test_dir = TestDir::new("progress_pull");
    let staging_path = test_dir.path().join("progress_test.part");

    let (tx, mut rx) = tokio::sync::mpsc::channel(16);

    let req = PullRequest {
        source: "huggingface".to_owned(),
        namespace: "Qwen".to_owned(),
        model: "Qwen2.5-7B".to_owned(),
        tag: "q8_0".to_owned(),
        expected_digest: None,
        format: Some(ModelFormat::Gguf),
    };

    let summary = adapter.pull(&req, &staging_path, Some(tx)).unwrap();
    assert!(summary.bytes_downloaded > 0);

    // Verify progress updates received
    let mut progress_count = 0;
    while let Ok(progress) = rx.try_recv() {
        assert!(progress.bytes_downloaded > 0);
        progress_count += 1;
    }
    assert!(
        progress_count > 0,
        "must receive at least one progress update"
    );
}

#[test]
fn test_ollama_adapter_pull() {
    let adapter = OllamaAdapter::new();
    assert_eq!(adapter.source_name(), "ollama");

    let anon_auth = adapter.auth(&SourceCredentials::Anonymous).unwrap();
    assert_eq!(anon_auth, AuthStatus::Anonymous);

    let search_res = adapter.search("qwen", &SearchFilter::default()).unwrap();
    assert!(!search_res.is_empty());

    let test_dir = TestDir::new("ollama_pull");
    let staging_path = test_dir.path().join("ollama_model.part");

    let req = PullRequest {
        source: "ollama".to_owned(),
        namespace: "library".to_owned(),
        model: "qwen2.5:7b".to_owned(),
        tag: "latest".to_owned(),
        expected_digest: None,
        format: Some(ModelFormat::Gguf),
    };

    let summary = adapter.pull(&req, &staging_path, None).unwrap();
    assert!(staging_path.exists());
    assert_eq!(summary.format, ModelFormat::Gguf);
    assert!(summary.bytes_downloaded > 0);
}
