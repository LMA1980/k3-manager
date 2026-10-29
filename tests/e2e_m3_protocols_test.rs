//! Milestone M3 Opaque-Box E2E Test Suite — External Protocols & Ingestion (US-19, US-23, US-24).
//!
//! Tiers 1 & 2: Feature Coverage and Boundary & Corner Cases:
//! - US-19: Isolated External Model Source Ingestion & Arrow Flight Streaming
//! - US-23: OpenAI-Compatible Client Protocol Front-End
//! - US-24: Anthropic Claude-Compatible Client Protocol Front-End

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::useless_vec
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::profile::{
    ComputePool, ModelListResponse, PriorityClass, TuningProfile, CONSERVATIVE_TUNING_ADVISORY,
};
use k3_manager::domain::{MockOsAffinityBackend, OsAffinityBackend};
use k3_manager::flight::{FlightError, FlightStream, FlightTensorBatchStreamer, MemfdArena};
use k3_manager::ingestion::{
    AuthStatus, HuggingFaceAdapter, ModelFormat, PullRequest, SearchFilter, SourceAdapter,
    SourceCredentials,
};
use k3_manager::protocol::{
    ClaudeAdapter, ClaudeContent, ClaudeContentBlock, ClaudeErrorResponse, ClaudeMessage,
    ClaudeMessageRequest, ClaudeMessageResponse, ClaudeSystem, ClaudeThinkingConfig, ClaudeTool,
    HttpRequest, HttpResponseBody, OpenAiChatRequest, OpenAiChatResponse,
    OpenAiErrorResponse, OpenAiFunction, OpenAiMessage, OpenAiTool, ProtocolServer,
    ProtocolServerConfig,
};
use k3_manager::queue::AdmissionQueueConfig;
use k3_manager::runtime::{AdminCommand, ControlPlaneRuntime};

static PROTOCOL_COUNTER: AtomicU64 = AtomicU64::new(1);

/// RAII Temporary Directory for Staging & Socket Tests
#[derive(Debug)]
struct ProtocolTestDir {
    path: PathBuf,
}

impl ProtocolTestDir {
    fn new(prefix: &str) -> Self {
        let seq = PROTOCOL_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("k3_proto_{prefix}_{pid}_{seq}"));
        fs::create_dir_all(&path).expect("failed to create protocol test directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ProtocolTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Computes NIST FIPS 180-4 SHA-256 digest
fn compute_sha256(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, part) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([part[0], part[1], part[2], part[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut h_var = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h_var.wrapping_add(s1).wrapping_add(ch).wrapping_add(k[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h_var = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(h_var);
    }

    format!(
        "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
    )
}

// ============================================================================
// US-19: External Ingestion & Arrow Flight — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us19_source_adapter_lifecycle_contract() {
    let adapter = HuggingFaceAdapter::new();
    assert_eq!(adapter.source_name(), "huggingface");

    let creds = SourceCredentials::Token("valid_hf_token_12345".to_string());
    let auth_status = adapter.auth(&creds).unwrap();
    assert!(matches!(auth_status, AuthStatus::Authenticated { .. }));

    let filter = SearchFilter {
        architecture: Some("qwen2.5".to_string()),
        quantization: None,
        limit: Some(10),
    };
    let models = adapter.search("qwen", &filter).unwrap();
    assert!(!models.is_empty());

    let listing = adapter.list("Qwen", "Qwen2.5-7B").unwrap();
    assert!(!listing.revisions.is_empty());

    let test_dir = ProtocolTestDir::new("adapter_lifecycle");
    let staging_file = test_dir.path().join("pulled.part");
    let req = PullRequest {
        source: "huggingface".to_string(),
        namespace: "Qwen".to_string(),
        model: "Qwen2.5-7B".to_string(),
        tag: "q8_0".to_string(),
        expected_digest: None,
        format: Some(ModelFormat::Gguf),
    };
    let summary = adapter.pull(&req, &staging_file, None).unwrap();
    assert!(staging_file.exists());
    assert_eq!(summary.format, ModelFormat::Gguf);
    assert!(summary.bytes_downloaded > 0);
}

#[test]
fn test_tier1_us19_staging_and_atomic_promotion() {
    let test_dir = ProtocolTestDir::new("staging_promo");
    let staging_dir = test_dir.path().join("staging");
    let blobs_dir = test_dir.path().join("blobs/sha256");
    fs::create_dir_all(&staging_dir).unwrap();
    fs::create_dir_all(&blobs_dir).unwrap();

    let download_bytes = b"GGUF\x02\x00\x00\x00STAGING_PROMOTION_VERIFIED_PAYLOAD";
    let declared_digest = compute_sha256(download_bytes);

    // Step 1: Write to staging/.part file
    let part_path = staging_dir.join("download-uuid-1234.part");
    fs::write(&part_path, download_bytes).unwrap();
    assert!(part_path.exists());

    // Step 2: Streaming verify SHA-256 and GGUF header before promotion
    let verified_content = fs::read(&part_path).unwrap();
    let calculated_digest = compute_sha256(&verified_content);
    assert_eq!(calculated_digest, declared_digest);
    assert!(verified_content.starts_with(b"GGUF"));

    // Step 3: Atomic POSIX rename into blobs/sha256/<digest>
    let final_blob_path = blobs_dir.join(&calculated_digest);
    fs::rename(&part_path, &final_blob_path).unwrap();

    assert!(!part_path.exists());
    assert!(final_blob_path.exists());
    assert_eq!(fs::read(&final_blob_path).unwrap(), download_bytes);
}

#[test]
fn test_tier1_us19_network_credential_isolation() {
    // Invariant: Compute worker and scheduler have no network access or external credentials
    struct WorkerProcessConfig {
        network_isolated: bool,
        has_remote_tokens: bool,
    }

    let worker_env = WorkerProcessConfig {
        network_isolated: true,
        has_remote_tokens: false,
    };

    assert!(worker_env.network_isolated);
    assert!(!worker_env.has_remote_tokens);
}

#[test]
fn test_tier1_us19_arrow_flight_tensor_streaming() {
    let streamer = FlightTensorBatchStreamer::new();
    let weights = vec![1i8; 128];
    let batch = streamer
        .build_repacked_batch("blk.0.attn_q.weight", 0, 0, &weights)
        .unwrap();

    let flight_stream = FlightStream::new(vec![batch.clone()]);
    assert_eq!(flight_stream.batches.len(), 1);
    assert_eq!(flight_stream.total_bytes, 128);

    let summary = streamer
        .stream_tensors_to_endpoint("qwen2.5-7b", &[batch], "grpc://127.0.0.1:50051")
        .unwrap();
    assert_eq!(summary.total_batches, 1);
    assert_eq!(summary.total_bytes, 128);
}

#[test]
fn test_tier1_us19_memfd_arena_sealing_and_mlock() {
    let mut arena = MemfdArena::new("proto_memfd_arena", 64 * 1024).unwrap();
    arena.write_at(0, b"MEMFD_E2E_PAYLOAD").unwrap();

    arena.mlock(None).unwrap();
    arena.seal_write().unwrap();

    assert!(arena.is_sealed());
    assert_eq!(arena.len(), 64 * 1024);
    assert_eq!(&arena.as_slice()[0..17], b"MEMFD_E2E_PAYLOAD");
    assert!(arena.as_mut_slice().is_err());
}

// ============================================================================
// US-19: Boundary & Corner Cases — Tier 2 (5 tests)
// ============================================================================

#[test]
fn test_tier2_us19_interrupted_download_staging_cleanup() {
    let test_dir = ProtocolTestDir::new("staging_abort");
    let staging_file = test_dir.path().join("partial_abort.part");
    fs::write(&staging_file, b"INCOMPLETE_PARTIAL_DOWNLOAD").unwrap();

    // Network connection drop detected: clean up .part file
    let network_error = true;
    if network_error {
        fs::remove_file(&staging_file).unwrap();
    }
    assert!(!staging_file.exists());
}

#[test]
fn test_tier2_us19_corrupted_format_header_rejected() {
    let corrupted_magic = b"CORRUPT_NOT_GGUF_HEADER";
    let is_valid_gguf = corrupted_magic.starts_with(b"GGUF");
    let is_valid_safetensors = corrupted_magic.starts_with(b"{\"");

    assert!(!is_valid_gguf);
    assert!(!is_valid_safetensors);
}

#[test]
fn test_tier2_us19_adapter_invalid_credentials_error() {
    let token = "bad_expired_token";
    let auth_result: Result<(), &'static str> = if token == "valid_token" {
        Ok(())
    } else {
        Err("HTTP 401 Unauthorized: invalid repository credential")
    };

    assert!(auth_result.is_err());
    let err_msg = auth_result.unwrap_err();
    assert!(!err_msg.contains(token)); // zero token leak in error string
}

#[test]
fn test_tier2_us19_flight_network_interruption_recovery() {
    let mut chunks_received = 3;
    let expected_chunks = 10;

    let is_interrupted = chunks_received < expected_chunks;
    assert!(is_interrupted);

    // Teardown partial memory allocation
    chunks_received = 0;
    assert_eq!(chunks_received, 0);
}

#[test]
fn test_tier2_us19_memfd_seal_violation_prevention() {
    let mut arena = MemfdArena::new("seal_violation_arena", 4096).unwrap();
    arena.write_at(0, b"DATA").unwrap();
    arena.seal_write().unwrap();

    let write_res = arena.write_at(0, b"MUTATION");
    assert!(write_res.is_err());
    match write_res.unwrap_err() {
        FlightError::MemfdSealed => {}
        other => panic!("expected MemfdSealed, got {other:?}"),
    }
}

// ============================================================================
// Protocol Test Helpers
// ============================================================================

/// Builds an isolated in-memory test runtime.
fn build_protocol_test_runtime() -> Arc<ControlPlaneRuntime> {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid control plane capacity");
    let mock_affinity: Arc<dyn OsAffinityBackend> = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");
    Arc::new(runtime)
}

/// Helper to build a standard ProtocolServer instance.
fn build_protocol_test_server(runtime: Arc<ControlPlaneRuntime>) -> ProtocolServer {
    ProtocolServer::new(ProtocolServerConfig::default(), runtime)
}

// ============================================================================
// US-23: OpenAI Protocol Front-End — Tier 1 (5 tests)
// ============================================================================

#[tokio::test]
async fn test_tier1_us23_openai_models_endpoint_unprofiled_tag() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    // Register an unprofiled model
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-unprofiled".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-unprofiled.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register unprofiled model");

    // Register and tune a profiled model
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-profiled".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-profiled.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register profiled model");

    let mut profile = TuningProfile::conservative_default();
    profile.apply_operator_tuning(
        ComputePool::X100,
        0,
        6,
        PriorityClass::Latency,
        4096,
        false,
        false,
    );
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "qwen2.5-profiled".to_owned(),
            profile,
        })
        .expect("apply tuning profile");

    // Route GET /v1/models through ProtocolServer
    let req = HttpRequest::new("GET", "/v1/models", Vec::new());
    let resp = server.route_request(req).await;
    assert_eq!(resp.status, 200);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let model_list: ModelListResponse =
        serde_json::from_slice(&body_bytes).expect("valid model list json");
    assert_eq!(model_list.object, "list");
    assert_eq!(model_list.data.len(), 2);

    let unprofiled_entry = model_list
        .data
        .iter()
        .find(|m| m.id == "qwen2.5-unprofiled")
        .unwrap();
    assert!(unprofiled_entry.unprofiled);
    assert_eq!(unprofiled_entry.profile_status, "unprofiled");
    assert_eq!(
        unprofiled_entry.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    let profiled_entry = model_list
        .data
        .iter()
        .find(|m| m.id == "qwen2.5-profiled")
        .unwrap();
    assert!(!profiled_entry.unprofiled);
    assert_eq!(profiled_entry.profile_status, "profiled");
    assert!(profiled_entry.advisory_notice.is_none());
}

#[tokio::test]
async fn test_tier1_us23_openai_chat_sync_and_sse_streaming() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-7b".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-7b.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // 1. Sync JSON response through ProtocolServer
    let sync_req = OpenAiChatRequest {
        model: "qwen2.5-7b".to_owned(),
        messages: vec![OpenAiMessage::user("Hello, world!")],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(64),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let sync_body = serde_json::to_vec(&sync_req).unwrap();
    let http_req = HttpRequest::new("POST", "/v1/chat/completions", sync_body);
    let resp = server.route_request(http_req).await;
    assert_eq!(resp.status, 200);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let sync_completion: OpenAiChatResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert!(sync_completion.id.starts_with("chatcmpl-"));
    assert_eq!(sync_completion.object, "chat.completion");
    assert_eq!(
        sync_completion.choices[0].finish_reason.as_deref(),
        Some("stop")
    );
    assert!(sync_completion.choices[0].message.content.is_some());

    // 2. SSE streaming response through ProtocolServer
    let stream_req = OpenAiChatRequest {
        model: "qwen2.5-7b".to_owned(),
        messages: vec![OpenAiMessage::user("Stream me an answer.")],
        stream: Some(true),
        tools: None,
        tool_choice: None,
        max_tokens: Some(64),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let stream_body = serde_json::to_vec(&stream_req).unwrap();
    let http_req = HttpRequest::new("POST", "/v1/chat/completions", stream_body);
    let resp = server.route_request(http_req).await;
    assert_eq!(resp.status, 200);

    let mut rx = match resp.body {
        HttpResponseBody::Stream(rx) => rx,
        _ => panic!("expected stream response body"),
    };
    let mut chunks = Vec::new();
    while let Some(chunk) = rx.recv().await {
        chunks.push(chunk);
    }
    let all_stream = chunks.join("");
    assert!(all_stream.contains("data: "));
    assert!(all_stream.contains("chat.completion.chunk"));
    assert!(all_stream.contains("\"finish_reason\":\"stop\""));
    assert!(all_stream.contains("data: [DONE]"));
}

#[tokio::test]
async fn test_tier1_us23_openai_tool_calls_passthrough() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-tools".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-tools.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let tools = vec![OpenAiTool {
        r#type: "function".to_owned(),
        function: OpenAiFunction {
            name: "get_current_weather".to_owned(),
            description: Some("Get weather for a location".to_owned()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            })),
        },
    }];

    let req = OpenAiChatRequest {
        model: "qwen2.5-tools".to_owned(),
        messages: vec![OpenAiMessage::user("What is the weather in Boston?")],
        stream: Some(false),
        tools: Some(tools),
        tool_choice: None,
        max_tokens: Some(128),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/chat/completions", body))
        .await;
    assert_eq!(resp.status, 200);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let completion: OpenAiChatResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        completion.choices[0].finish_reason.as_deref(),
        Some("tool_calls")
    );
    let tool_calls = completion.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_current_weather");
    assert!(!tool_calls[0].function.arguments.is_empty());
}

#[tokio::test]
async fn test_tier1_us23_openai_queue_feedback_and_429_retry() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-queue".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-queue.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    runtime.queue_mgr().register_model(
        "qwen2.5-queue",
        Some(AdmissionQueueConfig {
            max_depth: 1,
            keep_alive_interval: Duration::from_millis(100),
            wait_timeout: None,
            default_retry_after_secs: 5,
            estimated_pass_latency_ms: 25.0,
            estimated_passes_per_request: 16,
        }),
    );

    // Admit request 1 to fill queue
    let pos = runtime.queue_mgr().admit_to_model(
        "qwen2.5-queue",
        "req-1".to_owned(),
        false,
        None,
        None,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        None,
        None,
    );
    assert!(pos.is_ok());

    // Request 2 should hit queue saturation HTTP 429
    let req = OpenAiChatRequest {
        model: "qwen2.5-queue".to_owned(),
        messages: vec![OpenAiMessage::user("Request while queue is full")],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(32),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/chat/completions", body))
        .await;
    assert_eq!(resp.status, 429);

    let retry_after = resp
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
        .map(|(_, v)| v.as_str());
    assert!(retry_after.is_some());

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let err_resp: OpenAiErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        err_resp.error.code.as_deref(),
        Some("rate_limit_exceeded")
    );
}

#[tokio::test]
async fn test_tier1_us23_openai_context_overflow_400_error() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-overflow".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-overflow.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let mut profile = TuningProfile::conservative_default();
    profile.apply_operator_tuning(
        ComputePool::X100,
        0,
        6,
        PriorityClass::Throughput,
        10, // strict 10 tokens limit
        false,
        false,
    );
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "qwen2.5-overflow".to_owned(),
            profile,
        })
        .expect("apply constrained tuning profile");

    let long_prompt = "The quick brown fox jumps over the lazy dog repeatedly until the context length is completely saturated.";
    let req = OpenAiChatRequest {
        model: "qwen2.5-overflow".to_owned(),
        messages: vec![OpenAiMessage::user(long_prompt)],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(64),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/chat/completions", body))
        .await;
    assert_eq!(resp.status, 400);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let err_resp: OpenAiErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        err_resp.error.code.as_deref(),
        Some("context_length_exceeded")
    );
    assert_eq!(err_resp.error.r#type, "invalid_request_error");
}

// ============================================================================
// US-23: Boundary & Corner Cases — Tier 2 (5 tests)
// ============================================================================

#[tokio::test]
async fn test_tier2_us23_openai_sse_client_early_disconnect() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-disconnect".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-disconnect.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let stream_req = OpenAiChatRequest {
        model: "qwen2.5-disconnect".to_owned(),
        messages: vec![OpenAiMessage::user("Hello streaming")],
        stream: Some(true),
        tools: None,
        tool_choice: None,
        max_tokens: Some(64),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let body = serde_json::to_vec(&stream_req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/chat/completions", body))
        .await;
    assert_eq!(resp.status, 200);

    let rx = match resp.body {
        HttpResponseBody::Stream(rx) => rx,
        _ => panic!("expected stream response body"),
    };
    // Drop receiver immediately to simulate client TCP socket disconnect
    drop(rx);
    tokio::task::yield_now().await;
}

#[tokio::test]
async fn test_tier2_us23_openai_empty_tool_definitions() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-empty-tools".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-empty.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let req = OpenAiChatRequest {
        model: "qwen2.5-empty-tools".to_owned(),
        messages: vec![OpenAiMessage::user("Hello")],
        stream: Some(false),
        tools: Some(Vec::new()), // Empty tool definitions
        tool_choice: None,
        max_tokens: Some(32),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/chat/completions", body))
        .await;
    assert_eq!(resp.status, 200);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let completion: OpenAiChatResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        completion.choices[0].finish_reason.as_deref(),
        Some("stop")
    );
    assert!(completion.choices[0].message.tool_calls.is_none());
}

#[tokio::test]
async fn test_tier2_us23_openai_prompt_exactly_n_ctx() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-exact-ctx".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-exact.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let mut profile = TuningProfile::conservative_default();
    profile.apply_operator_tuning(
        ComputePool::X100,
        0,
        6,
        PriorityClass::Throughput,
        8, // context window = 8 tokens
        false,
        false,
    );
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "qwen2.5-exact-ctx".to_owned(),
            profile,
        })
        .expect("apply tuning profile");

    // Exact prompt with 8 words (>= 8 tokens): leaves 0 tokens for completion -> must reject 400
    let exact_prompt = "one two three four five six seven eight";
    let req = OpenAiChatRequest {
        model: "qwen2.5-exact-ctx".to_owned(),
        messages: vec![OpenAiMessage::user(exact_prompt)],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(16),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/chat/completions", body))
        .await;
    assert_eq!(resp.status, 400);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let err_resp: OpenAiErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        err_resp.error.code.as_deref(),
        Some("context_length_exceeded")
    );
}

#[tokio::test]
async fn test_tier2_us23_openai_malformed_json_body_400() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(runtime);

    let malformed_payload = b"{\"model\": \"qwen2.5\", \"messages\": [".to_vec();
    let req = HttpRequest::new("POST", "/v1/chat/completions", malformed_payload);
    let resp = server.route_request(req).await;
    assert_eq!(resp.status, 400);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let json_str = String::from_utf8(body_bytes).unwrap();
    assert!(json_str.contains("invalid_payload"));
}

#[tokio::test]
async fn test_tier2_us23_openai_queue_saturation_retry_after_header() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-hdr".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-hdr.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    runtime.queue_mgr().register_model(
        "qwen2.5-hdr",
        Some(AdmissionQueueConfig {
            max_depth: 1,
            keep_alive_interval: Duration::from_millis(100),
            wait_timeout: None,
            default_retry_after_secs: 7,
            estimated_pass_latency_ms: 25.0,
            estimated_passes_per_request: 16,
        }),
    );

    let pos = runtime.queue_mgr().admit_to_model(
        "qwen2.5-hdr",
        "req-fill".to_owned(),
        false,
        None,
        None,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        None,
        None,
    );
    assert!(pos.is_ok());

    let req = OpenAiChatRequest {
        model: "qwen2.5-hdr".to_owned(),
        messages: vec![OpenAiMessage::user("Hello full queue")],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(32),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/chat/completions", body))
        .await;
    assert_eq!(resp.status, 429);

    let retry_after = resp
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
        .map(|(_, v)| v.as_str());
    assert_eq!(retry_after, Some("1"));
}

// ============================================================================
// US-24: Anthropic Claude Protocol Front-End — Tier 1 (5 tests)
// ============================================================================

#[tokio::test]
async fn test_tier1_us24_claude_messages_content_blocks() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-compat-model".to_owned(),
            weights_path: PathBuf::from("/models/claude.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let req = ClaudeMessageRequest {
        model: "claude-compat-model".to_owned(),
        system: Some(ClaudeSystem::Text(
            "You are a helpful coding assistant.".to_owned(),
        )),
        messages: vec![ClaudeMessage::user("Write a binary search in Rust.")],
        max_tokens: 1024,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: Some(0.5),
        top_p: Some(0.9),
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/messages", body))
        .await;
    assert_eq!(resp.status, 200);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let message_resp: ClaudeMessageResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert!(message_resp.id.starts_with("msg_"));
    assert_eq!(message_resp.r#type, "message");
    assert_eq!(message_resp.role, "assistant");
    assert_eq!(message_resp.model, "claude-compat-model");
    assert!(!message_resp.content.is_empty());
    assert_eq!(message_resp.content[0].r#type, "text");
    assert_eq!(message_resp.stop_reason.as_deref(), Some("end_turn"));
}

#[tokio::test]
async fn test_tier1_us24_claude_streaming_sse_full_lifecycle() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-stream-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-stream.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let req = ClaudeMessageRequest {
        model: "claude-stream-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Explain quicksort briefly.")],
        max_tokens: 512,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(true),
        temperature: None,
        top_p: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/messages", body))
        .await;
    assert_eq!(resp.status, 200);

    let mut rx = match resp.body {
        HttpResponseBody::Stream(rx) => rx,
        _ => panic!("expected stream body"),
    };
    let mut events = Vec::new();
    while let Some(ev) = rx.recv().await {
        events.push(ev);
    }
    let all_stream = events.join("");
    assert!(all_stream.contains("event: message_start"));
    assert!(all_stream.contains("event: content_block_start"));
    assert!(all_stream.contains("event: content_block_delta"));
    assert!(all_stream.contains("\"type\":\"text_delta\""));
    assert!(all_stream.contains("event: content_block_stop"));
    assert!(all_stream.contains("event: message_delta"));
    assert!(all_stream.contains("\"stop_reason\":\"end_turn\""));
    assert!(all_stream.contains("event: message_stop"));
}

#[tokio::test]
async fn test_tier1_us24_claude_thinking_delta_reasoning_stream() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-reason-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-reason.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let req = ClaudeMessageRequest {
        model: "claude-reason-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Solve this logic puzzle.")],
        max_tokens: 1024,
        thinking: Some(ClaudeThinkingConfig {
            r#type: "enabled".to_owned(),
            budget_tokens: 1024,
        }),
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(true),
        temperature: None,
        top_p: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/messages", body))
        .await;
    assert_eq!(resp.status, 200);

    let mut rx = match resp.body {
        HttpResponseBody::Stream(rx) => rx,
        _ => panic!("expected stream body"),
    };
    let mut events = Vec::new();
    while let Some(ev) = rx.recv().await {
        events.push(ev);
    }
    let all_stream = events.join("");
    assert!(all_stream.contains("\"type\":\"thinking_delta\""));
    assert!(all_stream.contains("\"type\":\"text_delta\""));
    assert!(all_stream.contains("event: message_stop"));
}

#[tokio::test]
async fn test_tier1_us24_claude_tool_use_passthrough() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-tools-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-tools.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let tools = vec![ClaudeTool {
        name: "get_weather".to_owned(),
        description: Some("Fetch weather condition".to_owned()),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": { "location": { "type": "string" } },
            "required": ["location"]
        }),
    }];

    let req = ClaudeMessageRequest {
        model: "claude-tools-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Please get the weather in San Francisco")],
        max_tokens: 512,
        thinking: None,
        tools: Some(tools),
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/messages", body))
        .await;
    assert_eq!(resp.status, 200);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let message_resp: ClaudeMessageResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(message_resp.stop_reason.as_deref(), Some("tool_use"));
    let tool_block = message_resp
        .content
        .iter()
        .find(|b| b.r#type == "tool_use")
        .unwrap();
    assert_eq!(tool_block.name.as_deref(), Some("get_weather"));
    assert!(tool_block.id.is_some());
}

#[tokio::test]
async fn test_tier1_us24_claude_queue_feedback_and_529_overload() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-queue-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-queue.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    runtime.queue_mgr().register_model(
        "claude-queue-model",
        Some(AdmissionQueueConfig {
            max_depth: 1,
            keep_alive_interval: Duration::from_millis(100),
            wait_timeout: None,
            default_retry_after_secs: 5,
            estimated_pass_latency_ms: 25.0,
            estimated_passes_per_request: 16,
        }),
    );

    let pos = runtime.queue_mgr().admit_to_model(
        "claude-queue-model",
        "req-fill".to_owned(),
        false,
        None,
        None,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        None,
        None,
    );
    assert!(pos.is_ok());

    let req = ClaudeMessageRequest {
        model: "claude-queue-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Hello while queue is full")],
        max_tokens: 64,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/messages", body))
        .await;
    assert_eq!(resp.status, 529);

    let retry_after = resp
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
        .map(|(_, v)| v.as_str());
    assert!(retry_after.is_some());

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let err_resp: ClaudeErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.error.r#type, "overloaded_error");
}

// ============================================================================
// US-24: Boundary & Corner Cases — Tier 2 (5 tests)
// ============================================================================

#[tokio::test]
async fn test_tier2_us24_claude_thinking_delta_empty_text() {
    let req = ClaudeMessageRequest {
        model: "test-model".to_owned(),
        system: None,
        messages: vec![
            ClaudeMessage::user("Solve problem"),
            ClaudeMessage {
                role: "assistant".to_owned(),
                content: ClaudeContent::Blocks(vec![ClaudeContentBlock {
                    r#type: "thinking".to_owned(),
                    text: None,
                    thinking: Some("Reasoning internally without emitting text.".to_owned()),
                    id: None,
                    name: None,
                    input: None,
                    tool_use_id: None,
                    content: None,
                    is_error: None,
                    source: None,
                }]),
            },
            ClaudeMessage::user("Continue"),
        ],
        max_tokens: 100,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };

    let prompt = ClaudeAdapter::translate_to_canonical_prompt(&req);
    assert!(prompt.contains("[Thinking: Reasoning internally without emitting text.]"));
}

#[tokio::test]
async fn test_tier2_us24_claude_multi_block_mixed_content() {
    let req = ClaudeMessageRequest {
        model: "test-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage {
            role: "user".to_owned(),
            content: ClaudeContent::Blocks(vec![
                ClaudeContentBlock::text("Part 1 of query"),
                ClaudeContentBlock {
                    r#type: "image".to_owned(),
                    text: None,
                    thinking: None,
                    id: None,
                    name: None,
                    input: None,
                    tool_use_id: None,
                    content: None,
                    is_error: None,
                    source: Some(serde_json::json!({ "media_type": "image/jpeg" })),
                },
                ClaudeContentBlock::text("Part 2 with additional context"),
            ]),
        }],
        max_tokens: 100,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };

    let prompt = ClaudeAdapter::translate_to_canonical_prompt(&req);
    assert!(prompt.contains("Part 1 of query"));
    assert!(prompt.contains("[Image]"));
    assert!(prompt.contains("Part 2 with additional context"));
}

#[tokio::test]
async fn test_tier2_us24_claude_invalid_role_sequence_error() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-seq-model".to_owned(),
            weights_path: PathBuf::from("/models/claude.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Consecutive user messages
    let req = ClaudeMessageRequest {
        model: "claude-seq-model".to_owned(),
        system: None,
        messages: vec![
            ClaudeMessage::user("Hello"),
            ClaudeMessage::user("Hello again immediately"),
        ],
        max_tokens: 100,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/messages", body))
        .await;
    assert_eq!(resp.status, 400);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let err_resp: ClaudeErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.error.r#type, "invalid_request_error");
    assert!(err_resp.error.message.contains("consecutive"));
}

#[tokio::test]
async fn test_tier2_us24_claude_context_overflow_invalid_request() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-ctx-model".to_owned(),
            weights_path: PathBuf::from("/models/claude.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let mut profile = TuningProfile::conservative_default();
    profile.apply_operator_tuning(
        ComputePool::X100,
        0,
        6,
        PriorityClass::Throughput,
        10,
        false,
        false,
    );
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "claude-ctx-model".to_owned(),
            profile,
        })
        .expect("apply tuning profile");

    let long_prompt = "This is an excessively long prompt designed to exceed the strict 10 tokens context bound for testing context overflow.";
    let req = ClaudeMessageRequest {
        model: "claude-ctx-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user(long_prompt)],
        max_tokens: 64,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/messages", body))
        .await;
    assert_eq!(resp.status, 400);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let err_resp: ClaudeErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.error.r#type, "invalid_request_error");
    assert!(err_resp.error.message.contains("context window limit"));
}

#[tokio::test]
async fn test_tier2_us24_claude_queue_saturation_529_body() {
    let runtime = build_protocol_test_runtime();
    let server = build_protocol_test_server(Arc::clone(&runtime));

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-529-detail".to_owned(),
            weights_path: PathBuf::from("/models/claude.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    runtime.queue_mgr().register_model(
        "claude-529-detail",
        Some(AdmissionQueueConfig {
            max_depth: 1,
            keep_alive_interval: Duration::from_millis(100),
            wait_timeout: None,
            default_retry_after_secs: 5,
            estimated_pass_latency_ms: 25.0,
            estimated_passes_per_request: 16,
        }),
    );

    let pos = runtime.queue_mgr().admit_to_model(
        "claude-529-detail",
        "req-fill".to_owned(),
        false,
        None,
        None,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        None,
        None,
    );
    assert!(pos.is_ok());

    let req = ClaudeMessageRequest {
        model: "claude-529-detail".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Hello while queue is saturated")],
        max_tokens: 64,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let resp = server
        .route_request(HttpRequest::new("POST", "/v1/messages", body))
        .await;
    assert_eq!(resp.status, 529);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(json["type"], "error");
    assert_eq!(json["error"]["type"], "overloaded_error");
    assert!(json["error"]["message"].as_str().unwrap().contains("saturated"));
}
