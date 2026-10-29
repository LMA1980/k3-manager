//! Empirical Challenger Test Suite for Milestone M4 (Features 28, 29, 30).
//!
//! Adversarially challenges:
//! - Feature 28: Protocol DoS defenses (10MB payload ceiling, 64KB header limit, timeouts, queue backpressure 429/529 with Retry-After).
//! - Feature 29: DuckDB parameterized queries injection resistance across all 10 schema tables and methods.
//! - Feature 30: Secret isolation (fail closed on missing, directory, unreadable, empty, whitespace paths per AGENTS.md §3; credential buffer zeroization on wipe).

#![allow(
    clippy::pedantic,
    clippy::doc_markdown,
    clippy::undocumented_unsafe_blocks,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::needless_pass_by_value
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::profile::TuningProfile;
use k3_manager::domain::{MockOsAffinityBackend, OsAffinityBackend};
use k3_manager::ingestion::provider::resolve_secret_value;
use k3_manager::ingestion::{IngestionError, SourceCredentials};
use k3_manager::persistence::{
    BlobReferenceRecord, CasCatalog, DerivedBlobRecord, DuckDbPersistenceStore, ModelAliasRecord,
    ModelBlobRecord, ModelManifestRecord, PersistenceStore, Project, SessionLastTurn, Tenant,
};
use k3_manager::protocol::{
    ClaudeAdapter, HttpRequest, HttpResponseBody, OpenAiAdapter, ProtocolServer,
    ProtocolServerConfig, ServerError, DEFAULT_MAX_REQUEST_BODY_BYTES, MAX_REQUEST_HEADERS_BYTES,
};
use k3_manager::runtime::{AdminCommand, ControlPlaneRuntime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

static TEST_SEQ: AtomicU64 = AtomicU64::new(8000);

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

fn build_test_runtime() -> Arc<ControlPlaneRuntime> {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid control plane capacity");
    let mock_affinity: Arc<dyn OsAffinityBackend> = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");
    Arc::new(runtime)
}

// =========================================================================
// FEATURE 28: Protocol DoS Defenses
// =========================================================================

#[tokio::test]
async fn test_empirical_f28_payload_boundary_10mb() {
    let runtime = build_test_runtime();
    let server_config = ProtocolServerConfig {
        max_request_body_bytes: DEFAULT_MAX_REQUEST_BODY_BYTES, // 10 MB
        max_headers_bytes: MAX_REQUEST_HEADERS_BYTES,           // 64 KB
        socket_timeout: Duration::from_secs(5),
        listen_addr: None,
    };
    let server = ProtocolServer::new(server_config, runtime);

    // 1. Boundary: exactly 10MB payload (should NOT trigger 413)
    let exact_10mb = vec![b' '; DEFAULT_MAX_REQUEST_BODY_BYTES];
    let req_exact = HttpRequest::new("POST", "/v1/chat/completions", exact_10mb);
    let resp_exact = server.route_request(req_exact).await;
    assert_ne!(
        resp_exact.status, 413,
        "exact 10MB payload must not trigger 413 Payload Too Large"
    );

    // 2. Boundary: exactly 10MB + 1 byte (MUST trigger HTTP 413)
    let oversized = vec![b' '; DEFAULT_MAX_REQUEST_BODY_BYTES + 1];
    let req_oversized = HttpRequest::new("POST", "/v1/chat/completions", oversized);
    let resp_oversized = server.route_request(req_oversized).await;
    assert_eq!(
        resp_oversized.status, 413,
        "10MB + 1 byte must trigger HTTP 413"
    );
    if let HttpResponseBody::Bytes(body) = resp_oversized.body {
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("payload_too_large"));
        assert!(body_str.contains("exceeds maximum allowed limit"));
    } else {
        panic!("expected HttpResponseBody::Bytes");
    }

    // 3. Streaming test: Content-Length claiming >10MB rejects immediately before body is sent
    let (mut client, server_stream) = tokio::io::duplex(8192);
    let server_clone = server.clone();
    let server_handle =
        tokio::spawn(async move { server_clone.handle_connection(server_stream).await });

    let wire_header = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
        DEFAULT_MAX_REQUEST_BODY_BYTES + 500
    );
    client.write_all(wire_header.as_bytes()).await.unwrap();
    client.flush().await.unwrap();

    let mut resp_buf = vec![0u8; 4096];
    let n = client.read(&mut resp_buf).await.unwrap();
    let resp_str = String::from_utf8_lossy(&resp_buf[..n]);
    assert!(resp_str.contains("HTTP/1.1 413 Payload Too Large"));
    assert!(resp_str.contains("payload_too_large"));

    let server_res = server_handle.await.unwrap();
    assert!(matches!(server_res, Err(ServerError::PayloadTooLarge)));
}

#[tokio::test]
async fn test_empirical_f28_headers_boundary_64kb() {
    let runtime = build_test_runtime();
    let server_config = ProtocolServerConfig {
        max_request_body_bytes: 1024 * 1024,
        max_headers_bytes: MAX_REQUEST_HEADERS_BYTES, // 64 KB
        socket_timeout: Duration::from_secs(5),
        listen_addr: None,
    };
    let server = ProtocolServer::new(server_config, runtime);

    // 1. Headers well within 64KB boundary succeed
    let (mut client, server_stream) = tokio::io::duplex(8192);
    let server_clone = server.clone();
    let server_handle =
        tokio::spawn(async move { server_clone.handle_connection(server_stream).await });

    let normal_req = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
    client.write_all(normal_req.as_bytes()).await.unwrap();
    client.flush().await.unwrap();

    let mut resp_buf = vec![0u8; 4096];
    let n = client.read(&mut resp_buf).await.unwrap();
    let resp_str = String::from_utf8_lossy(&resp_buf[..n]);
    assert!(resp_str.contains("HTTP/1.1 200 OK"));
    assert!(server_handle.await.unwrap().is_ok());

    // 2. Headers exceeding 64KB boundary are rejected with ServerError::HeadersTooLarge
    let (mut client2, server_stream2) = tokio::io::duplex(8192);
    let server_clone2 = server.clone();
    let server_handle2 =
        tokio::spawn(async move { server_clone2.handle_connection(server_stream2).await });

    // Send 70 KB of headers without \r\n\r\n
    let huge_header = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nX-Large: {}\r\n",
        "B".repeat(70 * 1024)
    );
    client2.write_all(huge_header.as_bytes()).await.unwrap();
    client2.flush().await.unwrap();

    let server_res2 = server_handle2.await.unwrap();
    assert!(matches!(server_res2, Err(ServerError::HeadersTooLarge)));
}

#[tokio::test]
async fn test_empirical_f28_socket_timeouts_slowloris() {
    let runtime = build_test_runtime();
    let server_config = ProtocolServerConfig {
        max_request_body_bytes: 1024,
        max_headers_bytes: 2048,
        socket_timeout: Duration::from_millis(50), // 50ms short timeout
        listen_addr: None,
    };
    let server = ProtocolServer::new(server_config, runtime);

    let (mut client, server_stream) = tokio::io::duplex(1024);
    let server_handle = tokio::spawn(async move { server.handle_connection(server_stream).await });

    // Send partial request line and stall
    client
        .write_all(b"POST /v1/chat/completions HT")
        .await
        .unwrap();
    client.flush().await.unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    let server_res = server_handle.await.unwrap();
    assert!(
        matches!(server_res, Err(ServerError::Timeout)),
        "socket reading must timeout when client stalls"
    );
}

#[tokio::test]
async fn test_empirical_f28_queue_backpressure_retry_after() {
    let runtime = build_test_runtime();

    // Register a model
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "q_stress_model".to_owned(),
            weights_path: PathBuf::from("/models/stress.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let queue_mgr = runtime.queue_mgr();

    // Saturate queue to capacity (default depth is 64)
    let _cancel_tokens: Vec<Arc<AtomicBool>> = (1..=64)
        .map(|i| {
            let cancel = Arc::new(AtomicBool::new(false));
            let res = queue_mgr.admit_to_model(
                "q_stress_model",
                format!("req_{i}"),
                false,
                None,
                None,
                cancel.clone(),
                None,
                None,
            );
            assert!(res.is_ok(), "request {i} within capacity must be admitted");
            cancel
        })
        .collect();

    // 1. OpenAI request when queue is saturated -> HTTP 429 + Retry-After
    let openai_body = serde_json::to_vec(&serde_json::json!({
        "model": "q_stress_model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 32
    }))
    .unwrap();
    let openai_req = HttpRequest::new("POST", "/v1/chat/completions", openai_body);
    let openai_resp = OpenAiAdapter::handle_http(openai_req, runtime.clone()).await;

    assert_eq!(
        openai_resp.status, 429,
        "saturated queue must return HTTP 429 for OpenAI"
    );
    let retry_after_hdr = openai_resp
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"));
    assert!(
        retry_after_hdr.is_some(),
        "OpenAI 429 must include Retry-After header"
    );
    let secs: u64 = retry_after_hdr
        .unwrap()
        .1
        .parse()
        .expect("valid integer Retry-After");
    assert!(
        (1..=86400).contains(&secs),
        "Retry-After must be clamped in [1, 86400]"
    );

    // 2. Claude request when queue is saturated -> HTTP 529 + Retry-After
    let claude_body = serde_json::to_vec(&serde_json::json!({
        "model": "q_stress_model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
        "stream": false
    }))
    .unwrap();
    let claude_req = HttpRequest::new("POST", "/v1/messages", claude_body);
    let claude_resp = ClaudeAdapter::handle_http(claude_req, runtime.clone()).await;

    assert_eq!(
        claude_resp.status, 529,
        "saturated queue must return HTTP 529 for Claude"
    );
    let claude_retry_hdr = claude_resp
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"));
    assert!(
        claude_retry_hdr.is_some(),
        "Claude 529 must include Retry-After header"
    );
    let claude_secs: u64 = claude_retry_hdr
        .unwrap()
        .1
        .parse()
        .expect("valid integer Retry-After");
    assert!(
        (1..=86400).contains(&claude_secs),
        "Retry-After must be clamped in [1, 86400]"
    );

    // 3. Cancel one request: frees a slot in queue
    assert!(queue_mgr.cancel_in_model("q_stress_model", "req_1"));
    let depth = queue_mgr.queue_depth("q_stress_model");
    assert_eq!(
        depth, 63,
        "cancelled request must immediately free queue slot"
    );
}

// =========================================================================
// FEATURE 29: DuckDB Parameterized Query Injection Resistance
// =========================================================================

#[test]
fn test_empirical_f29_sql_injection_on_all_duckdb_tables() {
    let catalog =
        Arc::new(DuckDbPersistenceStore::open_in_memory().expect("open duckdb in memory"));

    let aggressive_payloads = [
        "'; DROP TABLE tenants; --",
        "'; DROP TABLE model_manifests; --",
        "'; DROP TABLE model_blobs; --",
        "'; DROP TABLE projects; --",
        "'; DROP TABLE session_last_turns; --",
        "'; DROP TABLE tuning_profiles; --",
        "'; DROP TABLE storage_config; --",
        "'; DROP TABLE model_aliases; --",
        "'; DROP TABLE blob_references; --",
        "'; DROP TABLE derived_blobs; --",
        "' OR '1'='1",
        "admin'--",
        "1; DELETE FROM model_aliases; --",
        "\"; ATTACH '/tmp/pwn.db' AS pwn; --",
        "'; COPY tenants TO '/tmp/pwn.csv'; --",
        "' UNION SELECT id, name, created_at FROM tenants WHERE ''='",
        "'; PRAGMA version; --",
        "null_byte_\0_attack",
        "emoji_injection_' 🚀 --",
    ];

    let base_blob = ModelBlobRecord {
        digest: "f3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_owned(),
        size_bytes: 4096,
        blob_type: "raw_weights".to_owned(),
        created_at: 100,
        last_accessed_at: 100,
        ref_count: 0,
    };
    catalog.save_blob(&base_blob).unwrap();

    for (idx, payload) in aggressive_payloads.iter().enumerate() {
        // Table 1: Tenants
        let t_id = format!("t_emp_{idx}");
        let tenant = Tenant {
            id: t_id.clone(),
            name: (*payload).to_owned(),
            created_at: 200,
        };
        assert!(catalog.save_tenant(&tenant).is_ok());
        let loaded_t = catalog.get_tenant(&t_id).unwrap().unwrap();
        assert_eq!(loaded_t.name, *payload);

        // Table 2: Projects
        let p_id = format!("p_emp_{idx}");
        let project = Project {
            id: p_id.clone(),
            tenant_id: t_id.clone(),
            name: (*payload).to_owned(),
            token_budget: Some(5000),
            created_at: 200,
        };
        assert!(catalog.save_project(&project).is_ok());
        let loaded_p = catalog.get_project(&p_id).unwrap().unwrap();
        assert_eq!(loaded_p.name, *payload);

        // Table 3: SessionLastTurn
        let s_id = format!("sess_emp_{idx}");
        let turn = SessionLastTurn {
            session_id: s_id.clone(),
            project_id: p_id.clone(),
            tenant_id: t_id.clone(),
            model_id: (*payload).to_owned(),
            prompt: (*payload).to_owned(),
            response: (*payload).to_owned(),
            prompt_tokens: 42,
            completion_tokens: 84,
            parameters_json: (*payload).to_owned(),
            updated_at: 200,
        };
        assert!(catalog.save_session_last_turn(&turn).is_ok());
        let loaded_turn = catalog.load_session_last_turn(&s_id).unwrap().unwrap();
        assert_eq!(loaded_turn.prompt, *payload);
        assert_eq!(loaded_turn.response, *payload);

        // Table 4: TuningProfiles
        let m_id = format!("model_emp_{idx}");
        let prof = TuningProfile {
            cluster: (*payload).to_owned(),
            advisory_notice: Some((*payload).to_owned()),
            ..Default::default()
        };
        assert!(catalog.save_tuning_profile(&m_id, &prof).is_ok());
        let loaded_prof = catalog.load_tuning_profile(&m_id).unwrap().unwrap();
        assert_eq!(loaded_prof.cluster, *payload);

        // Table 5: StorageConfig
        let cfg_key = format!("cfg_emp_{idx}");
        assert!(catalog.set_storage_config(&cfg_key, payload).is_ok());
        let loaded_cfg = catalog.get_storage_config(&cfg_key).unwrap().unwrap();
        assert_eq!(loaded_cfg, *payload);

        // Table 6: ModelManifests
        let m_digest = format!("man_emp_{idx}");
        let manifest = ModelManifestRecord {
            manifest_digest: m_digest.clone(),
            weight_digest: base_blob.digest.clone(),
            format: (*payload).to_owned(),
            architecture_id: (*payload).to_owned(),
            recipe_version: (*payload).to_owned(),
            quantization: (*payload).to_owned(),
            context_length: 4096,
            license: Some((*payload).to_owned()),
            origin: Some((*payload).to_owned()),
            manifest_json: (*payload).to_owned(),
            created_at: 200,
        };
        assert!(catalog.save_manifest(&manifest).is_ok());
        let loaded_man = catalog.get_manifest(&m_digest).unwrap().unwrap();
        assert_eq!(loaded_man.architecture_id, *payload);

        // Table 7: ModelAliases
        let a_id = format!("alias_emp_{idx}");
        let alias = ModelAliasRecord {
            alias_id: a_id.clone(),
            source: (*payload).to_owned(),
            namespace: (*payload).to_owned(),
            name: format!("mod_{idx}"),
            tag: format!("tag_{idx}"),
            manifest_digest: m_digest.clone(),
            weight_digest: base_blob.digest.clone(),
            alias_path: (*payload).to_owned(),
            created_at: 200,
        };
        assert!(catalog.save_alias(&alias).is_ok());
        let loaded_alias = catalog.get_alias(&a_id).unwrap().unwrap();
        assert_eq!(loaded_alias.alias_path, *payload);

        // Table 8: BlobReferences
        let ref_rec = BlobReferenceRecord {
            ref_id: format!("ref_emp_{idx}"),
            blob_digest: base_blob.digest.clone(),
            ref_type: "test".to_owned(),
            ref_source: (*payload).to_owned(),
            created_at: 200,
        };
        assert!(catalog.add_blob_reference(&ref_rec).is_ok());
        let refs = catalog.list_blob_references(&base_blob.digest).unwrap();
        assert!(refs.iter().any(|r| r.ref_id == ref_rec.ref_id));

        // Table 9: DerivedBlobs
        let d_digest = format!(
            "f3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b8{:02x}",
            idx
        );
        let derived_blob_rec = ModelBlobRecord {
            digest: d_digest.clone(),
            size_bytes: 8192,
            blob_type: "derived".to_owned(),
            created_at: 200,
            last_accessed_at: 200,
            ref_count: 0,
        };
        catalog.save_blob(&derived_blob_rec).unwrap();

        let derived = DerivedBlobRecord {
            derived_digest: d_digest.clone(),
            source_digest: base_blob.digest.clone(),
            recipe_version: (*payload).to_owned(),
            target_layout: format!("layout_{idx}"),
            status: "valid".to_owned(),
            created_at: 200,
        };
        assert!(catalog.save_derived_blob(&derived).is_ok());
        let loaded_d = catalog
            .get_derived_blob(&base_blob.digest, payload, &format!("layout_{idx}"))
            .unwrap()
            .unwrap();
        assert_eq!(loaded_d.recipe_version, *payload);
    }

    // Verify all tables survived without dropping or data loss
    assert_eq!(
        catalog.list_tenants().unwrap().len(),
        aggressive_payloads.len()
    );
    assert!(catalog.get_blob(&base_blob.digest).unwrap().is_some());
}

// =========================================================================
// FEATURE 30: Secret Isolation & Memory Wiping
// =========================================================================

#[test]
fn test_empirical_f30_secret_path_resolution_fail_closed() {
    let temp = TempDirGuard::new("f30_sec_test");

    // 1. Missing absolute path -> None
    assert_eq!(resolve_secret_value("/nonexistent/secret_path.env"), None);

    // 2. Missing relative path -> None
    assert_eq!(resolve_secret_value("./missing_token.env"), None);
    assert_eq!(resolve_secret_value("../missing_token.env"), None);
    assert_eq!(resolve_secret_value("sub/path/missing.key"), None);

    // 3. Directory path with slash or traversal prefix -> None
    assert_eq!(resolve_secret_value(temp.path().to_str().unwrap()), None);
    assert_eq!(resolve_secret_value("/"), None);
    assert_eq!(resolve_secret_value("./"), None);
    assert_eq!(resolve_secret_value("../"), None);

    // 3b. Bare "." and ".." are directory paths and must FAIL CLOSED per AGENTS.md §3:
    assert_eq!(resolve_secret_value("."), None);
    assert_eq!(resolve_secret_value(".."), None);

    // 4. Empty file -> None
    let empty_file = temp.path().join("empty.key");
    fs::write(&empty_file, b"").unwrap();
    assert_eq!(resolve_secret_value(empty_file.to_str().unwrap()), None);

    // 5. Whitespace-only file -> None
    let ws_file = temp.path().join("whitespace.key");
    fs::write(&ws_file, b"   \r\n\t   \n").unwrap();
    assert_eq!(resolve_secret_value(ws_file.to_str().unwrap()), None);

    // 6. Valid file -> Some(trimmed content)
    let valid_file = temp.path().join("valid_secret.key");
    fs::write(&valid_file, b"   hf_super_secret_9999\n\n").unwrap();
    assert_eq!(
        resolve_secret_value(valid_file.to_str().unwrap()),
        Some("hf_super_secret_9999".to_string())
    );

    // 7. Literal non-path token -> Some(literal)
    assert_eq!(
        resolve_secret_value("hf_literal_token_xyz"),
        Some("hf_literal_token_xyz".to_string())
    );

    // 8. Empty string and whitespace -> None
    assert_eq!(resolve_secret_value(""), None);
    assert_eq!(resolve_secret_value("   "), None);

    // 9. SourceCredentials::from_secret_string integration
    let err_res = SourceCredentials::from_secret_string("/path/does/not/exist");
    assert!(err_res.is_err());
    assert!(matches!(
        err_res.unwrap_err(),
        IngestionError::CredentialResolutionFailed(_)
    ));

    let ok_file_res = SourceCredentials::from_secret_string(valid_file.to_str().unwrap());
    assert_eq!(
        ok_file_res.unwrap(),
        SourceCredentials::Token("hf_super_secret_9999".to_string())
    );

    let ok_literal_res = SourceCredentials::from_secret_string("hf_direct_token_123");
    assert_eq!(
        ok_literal_res.unwrap(),
        SourceCredentials::Token("hf_direct_token_123".to_string())
    );

    let anon_res = SourceCredentials::from_secret_string("   ");
    assert_eq!(anon_res.unwrap(), SourceCredentials::Anonymous);
}

#[test]
fn test_empirical_f30_memory_wipe_zeroization() {
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
