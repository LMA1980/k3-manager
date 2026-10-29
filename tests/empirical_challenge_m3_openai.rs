//! Empirical Challenge Suite for Milestone M3: OpenAI Protocol Front-End & Server Limits (US-23).
//!
//! Adversarially stress-tests:
//! 1. Malformed JSON & schema edge cases: empty requests, missing mandatory fields (`messages`, `model`),
//!    unknown parameters, non-array messages, invalid byte payloads.
//! 2. Context length enforcement: boundary and overflow conditions rejected strictly with HTTP 400
//!    without silent prompt truncation in both non-streaming and streaming modes.
//! 3. Queue saturation: HTTP 429 Too Many Requests with valid numeric `Retry-After` header when queue
//!    depth limit is reached, and admission recovery after completion.
//! 4. Server limits: Payloads exceeding 10MB (10,485,761 bytes) immediately rejected with HTTP 413,
//!    socket connection memory safety, header bomb (>64KB) rejection, and read timeouts.
//! 5. SSE streaming robustness: Keep-alive comments, queue position comments, chunk structure,
//!    chunked HTTP/1.1 wire framing, client premature disconnect resilience, and `data: [DONE]\n\n`.
//! 6. Tool calls transparency: Multi-tool declarations with keyword matching, complex deeply nested
//!    JSON argument preservation, assistant message history pass-through, and streaming tool calls.

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::cast_possible_truncation
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::profile::{ComputePool, PriorityClass, TuningProfile};
use k3_manager::domain::{MockOsAffinityBackend, OsAffinityBackend};
use k3_manager::protocol::{
    HttpRequest, HttpResponseBody, OpenAiAdapter, OpenAiChatRequest, OpenAiChatResponse,
    OpenAiErrorResponse, OpenAiFunction, OpenAiFunctionCall, OpenAiMessage, OpenAiTool,
    OpenAiToolCall, ProtocolServer, ProtocolServerConfig, ServerError,
    DEFAULT_MAX_REQUEST_BODY_BYTES, MAX_REQUEST_HEADERS_BYTES,
};
use k3_manager::queue::AdmissionQueueConfig;
use k3_manager::runtime::{AdminCommand, ControlPlaneRuntime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Constructs an isolated in-memory test runtime.
fn build_test_runtime() -> Arc<ControlPlaneRuntime> {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid control plane capacity");
    let mock_affinity: Arc<dyn OsAffinityBackend> = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");
    Arc::new(runtime)
}

/// Helper to register a test model with optional custom tuning profile.
fn setup_model(runtime: &ControlPlaneRuntime, model_id: &str, n_ctx: usize) {
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from(format!("/models/{model_id}.gguf")),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    if n_ctx > 0 {
        let mut profile = TuningProfile::conservative_default();
        profile.apply_operator_tuning(
            ComputePool::X100,
            0,
            6,
            PriorityClass::Latency,
            n_ctx,
            false,
            false,
        );
        runtime
            .execute_admin(AdminCommand::ApplyTuningProfile {
                model_id: model_id.to_owned(),
                profile,
            })
            .expect("apply tuning profile");
    }
}

// ============================================================================
// 1. Malformed JSON & Schema Edge Cases
// ============================================================================

#[tokio::test]
async fn test_challenge_empty_json_object_returns_400() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen2.5-edge", 2048);

    // Empty JSON object `{}` lacks mandatory `model` and `messages`
    let req = HttpRequest::new("POST", "/v1/chat/completions", b"{}".to_vec());
    let resp = OpenAiAdapter::handle_http(req, runtime).await;

    assert_eq!(resp.status, 400);
    let body = match resp.body {
        HttpResponseBody::Bytes(b) => String::from_utf8(b).unwrap(),
        _ => panic!("expected buffered byte body"),
    };
    assert!(body.contains("invalid_payload"));
    assert!(body.contains("missing field"));
}

#[tokio::test]
async fn test_challenge_missing_mandatory_fields_returns_400() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen2.5-edge", 2048);

    // Case A: Missing `messages`
    let body_no_msgs = serde_json::json!({
        "model": "qwen2.5-edge"
    })
    .to_string();
    let req_a = HttpRequest::new("POST", "/v1/chat/completions", body_no_msgs.into_bytes());
    let resp_a = OpenAiAdapter::handle_http(req_a, Arc::clone(&runtime)).await;
    assert_eq!(resp_a.status, 400);

    // Case B: Missing `model`
    let body_no_model = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}]
    })
    .to_string();
    let req_b = HttpRequest::new("POST", "/v1/chat/completions", body_no_model.into_bytes());
    let resp_b = OpenAiAdapter::handle_http(req_b, Arc::clone(&runtime)).await;
    assert_eq!(resp_b.status, 400);

    // Case C: Null `messages`
    let body_null_msgs = serde_json::json!({
        "model": "qwen2.5-edge",
        "messages": null
    })
    .to_string();
    let req_c = HttpRequest::new("POST", "/v1/chat/completions", body_null_msgs.into_bytes());
    let resp_c = OpenAiAdapter::handle_http(req_c, runtime).await;
    assert_eq!(resp_c.status, 400);
}

#[tokio::test]
async fn test_challenge_non_array_messages_returns_400() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen2.5-edge", 2048);

    // Case A: messages is a string
    let body_str = serde_json::json!({
        "model": "qwen2.5-edge",
        "messages": "User says hello"
    })
    .to_string();
    let req_a = HttpRequest::new("POST", "/v1/chat/completions", body_str.into_bytes());
    let resp_a = OpenAiAdapter::handle_http(req_a, Arc::clone(&runtime)).await;
    assert_eq!(resp_a.status, 400);

    // Case B: messages is an integer
    let body_num = serde_json::json!({
        "model": "qwen2.5-edge",
        "messages": 42
    })
    .to_string();
    let req_b = HttpRequest::new("POST", "/v1/chat/completions", body_num.into_bytes());
    let resp_b = OpenAiAdapter::handle_http(req_b, Arc::clone(&runtime)).await;
    assert_eq!(resp_b.status, 400);

    // Case C: messages is an object instead of array
    let body_obj = serde_json::json!({
        "model": "qwen2.5-edge",
        "messages": {"role": "user", "content": "hello"}
    })
    .to_string();
    let req_c = HttpRequest::new("POST", "/v1/chat/completions", body_obj.into_bytes());
    let resp_c = OpenAiAdapter::handle_http(req_c, runtime).await;
    assert_eq!(resp_c.status, 400);
}

#[tokio::test]
async fn test_challenge_syntax_corruption_and_non_utf8_returns_400() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen2.5-edge", 2048);

    // Case A: Truncated JSON
    let req_trunc = HttpRequest::new("POST", "/v1/chat/completions", b"{\"model\": \"qwen2.5-edge\", \"messages\": [{\"role\": \"user\"".to_vec());
    let resp_trunc = OpenAiAdapter::handle_http(req_trunc, Arc::clone(&runtime)).await;
    assert_eq!(resp_trunc.status, 400);

    // Case B: Completely non-UTF8 bytes
    let req_bad_bytes = HttpRequest::new("POST", "/v1/chat/completions", vec![0xFF, 0xFE, 0xFD, 0x00, 0x80]);
    let resp_bad_bytes = OpenAiAdapter::handle_http(req_bad_bytes, Arc::clone(&runtime)).await;
    assert_eq!(resp_bad_bytes.status, 400);

    // Case C: Empty byte slice
    let req_empty = HttpRequest::new("POST", "/v1/chat/completions", Vec::new());
    let resp_empty = OpenAiAdapter::handle_http(req_empty, runtime).await;
    assert_eq!(resp_empty.status, 400);
}

#[tokio::test]
async fn test_challenge_unknown_parameters_tolerated_forward_compatible() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen2.5-forward", 2048);

    // Unknown client parameters (e.g. top_k, logprobs, user, custom flags) must be tolerated
    let body = serde_json::json!({
        "model": "qwen2.5-forward",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false,
        "custom_unrecognized_field": 42,
        "experimental_feature_flag": { "deeply": "nested", "active": true },
        "user": "client-identifier-12345"
    })
    .to_string();

    let req = HttpRequest::new("POST", "/v1/chat/completions", body.into_bytes());
    let resp = OpenAiAdapter::handle_http(req, runtime).await;
    assert_eq!(resp.status, 200);

    let body_str = match resp.body {
        HttpResponseBody::Bytes(b) => String::from_utf8(b).unwrap(),
        _ => panic!("expected byte body"),
    };
    let completion: OpenAiChatResponse = serde_json::from_str(&body_str).unwrap();
    assert_eq!(completion.model, "qwen2.5-forward");
}

#[tokio::test]
async fn test_challenge_flexible_message_content_structures() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen2.5-flex", 2048);

    // Multi-part content with text blocks and assistant message with null content
    let req = OpenAiChatRequest {
        model: "qwen2.5-flex".to_owned(),
        messages: vec![
            OpenAiMessage {
                role: "system".to_owned(),
                content: Some(k3_manager::protocol::MessageContent::Parts(vec![
                    serde_json::json!({"type": "text", "text": "System part 1. "}),
                    serde_json::json!({"type": "text", "text": "System part 2."}),
                ])),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            OpenAiMessage {
                role: "assistant".to_owned(),
                content: None,
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            OpenAiMessage::user("Next turn"),
        ],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(64),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };

    let resp = OpenAiAdapter::handle_chat_completions(req, runtime).await;
    assert_eq!(resp.status, 200);
}

// ============================================================================
// 2. Context Length Enforcement & Zero Silent Truncation
// ============================================================================

#[tokio::test]
async fn test_challenge_context_length_boundary_and_strict_rejection() {
    let runtime = build_test_runtime();
    // Configure tight context window of 10 tokens
    setup_model(&runtime, "constrained-boundary", 10);

    // Case A: Prompt with 4 words (well within 10 tokens) -> 200 OK
    let req_ok = OpenAiChatRequest {
        model: "constrained-boundary".to_owned(),
        messages: vec![OpenAiMessage::user("Short prompt test.")],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(32),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let resp_ok = OpenAiAdapter::handle_chat_completions(req_ok, Arc::clone(&runtime)).await;
    assert_eq!(resp_ok.status, 200);

    // Case B: Boundary prompt with exactly >= 10 whitespace tokens
    // "One two three four five six seven eight nine ten eleven" (11 words > 10)
    let req_overflow = OpenAiChatRequest {
        model: "constrained-boundary".to_owned(),
        messages: vec![OpenAiMessage::user("One two three four five six seven eight nine ten eleven")],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(32),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let resp_overflow = OpenAiAdapter::handle_chat_completions(req_overflow, Arc::clone(&runtime)).await;
    assert_eq!(resp_overflow.status, 400);

    let body = match resp_overflow.body {
        HttpResponseBody::Bytes(b) => String::from_utf8(b).unwrap(),
        _ => panic!("expected buffered byte body"),
    };
    let err: OpenAiErrorResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(err.error.code.as_deref(), Some("context_length_exceeded"));
    assert_eq!(err.error.r#type, "invalid_request_error");
    assert!(err.error.message.contains("context window limit"));
}

#[tokio::test]
async fn test_challenge_context_length_streaming_rejected_before_sse() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "constrained-stream", 8);

    // Prompt exceeding 8 tokens in streaming mode
    let long_prompt = "Alpha beta gamma delta epsilon zeta eta theta iota kappa";
    let req = OpenAiChatRequest {
        model: "constrained-stream".to_owned(),
        messages: vec![OpenAiMessage::user(long_prompt)],
        stream: Some(true),
        tools: None,
        tool_choice: None,
        max_tokens: Some(32),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };

    let resp = OpenAiAdapter::handle_chat_completions(req, runtime).await;
    // Must return HTTP 400 Bad Request, NOT an HTTP 200 text/event-stream
    assert_eq!(resp.status, 400);

    let body = match resp.body {
        HttpResponseBody::Bytes(b) => String::from_utf8(b).unwrap(),
        _ => panic!("expected buffered byte body"),
    };
    let err: OpenAiErrorResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(err.error.code.as_deref(), Some("context_length_exceeded"));
}

// ============================================================================
// 3. Queue Saturation & 429 Retry-After Verification
// ============================================================================

#[tokio::test]
async fn test_challenge_queue_saturation_retry_after_header_format() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "sat-model", 2048);

    // Configure queue limit of 1
    runtime.queue_mgr().register_model(
        "sat-model",
        Some(AdmissionQueueConfig {
            max_depth: 1,
            keep_alive_interval: Duration::from_millis(100),
            wait_timeout: None,
            default_retry_after_secs: 5,
            estimated_pass_latency_ms: 20.0,
            estimated_passes_per_request: 10,
        }),
    );

    // Pre-occupy slot 1
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let admit1 = runtime.queue_mgr().admit_to_model(
        "sat-model",
        "req-holding-slot".to_owned(),
        false,
        None,
        None,
        cancel,
        None,
        None,
    );
    assert!(admit1.is_ok());

    // 1. Non-streaming saturation check
    let req_non_stream = OpenAiChatRequest {
        model: "sat-model".to_owned(),
        messages: vec![OpenAiMessage::user("Hello full queue")],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(16),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let resp_non_stream = OpenAiAdapter::handle_chat_completions(req_non_stream, Arc::clone(&runtime)).await;
    assert_eq!(resp_non_stream.status, 429);

    let retry_after_header = resp_non_stream
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
        .map(|(_, v)| v.as_str());
    assert!(retry_after_header.is_some(), "missing Retry-After header");
    let secs: u64 = retry_after_header.unwrap().parse().expect("valid integer seconds in Retry-After");
    assert!(secs > 0);

    // 2. Streaming saturation check
    let req_stream = OpenAiChatRequest {
        model: "sat-model".to_owned(),
        messages: vec![OpenAiMessage::user("Streaming to full queue")],
        stream: Some(true),
        tools: None,
        tool_choice: None,
        max_tokens: Some(16),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let resp_stream = OpenAiAdapter::handle_chat_completions(req_stream, Arc::clone(&runtime)).await;
    assert_eq!(resp_stream.status, 429);
    let s_retry = resp_stream
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
        .map(|(_, v)| v.as_str());
    assert!(s_retry.is_some());

    // 3. Mark execution completed and verify queue recovery
    runtime.queue_mgr().mark_execution_completed("sat-model");

    let req_recovered = OpenAiChatRequest {
        model: "sat-model".to_owned(),
        messages: vec![OpenAiMessage::user("Now queue has space")],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(16),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let resp_recovered = OpenAiAdapter::handle_chat_completions(req_recovered, runtime).await;
    assert_eq!(resp_recovered.status, 200);
}

// ============================================================================
// 4. Server Limits & Payload Protection (10 MB Boundary & Socket Security)
// ============================================================================

#[tokio::test]
async fn test_challenge_server_rejects_payload_exceeding_10mb() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen-10mb", 2048);

    let server_config = ProtocolServerConfig {
        max_request_body_bytes: DEFAULT_MAX_REQUEST_BODY_BYTES, // Exactly 10 MB (10,485,760 bytes)
        max_headers_bytes: MAX_REQUEST_HEADERS_BYTES,
        socket_timeout: Duration::from_secs(5),
        listen_addr: None,
    };
    let server = ProtocolServer::new(server_config, runtime);

    // Payload of 10 MB + 1 byte
    let oversized_len = DEFAULT_MAX_REQUEST_BODY_BYTES + 1;
    let big_payload = vec![b'x'; oversized_len];
    let req = HttpRequest::new("POST", "/v1/chat/completions", big_payload);

    let resp = server.route_request(req).await;
    assert_eq!(resp.status, 413);

    let body = match resp.body {
        HttpResponseBody::Bytes(b) => String::from_utf8(b).unwrap(),
        _ => panic!("expected byte body"),
    };
    assert!(body.contains("payload_too_large"));
    assert!(body.contains("10485760"));
}

#[tokio::test]
async fn test_challenge_server_connection_stream_immediate_413_on_content_length() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen-conn", 2048);

    let server_config = ProtocolServerConfig {
        max_request_body_bytes: 1024, // 1 KB bound for deterministic unit testing
        max_headers_bytes: 4096,
        socket_timeout: Duration::from_secs(5),
        listen_addr: None,
    };
    let server = ProtocolServer::new(server_config, runtime);

    // Create duplex in-memory socket pair
    let (mut client, server_stream) = tokio::io::duplex(8192);

    let server_handle = tokio::spawn(async move {
        server.handle_connection(server_stream).await
    });

    // Client sends request claiming Content-Length: 5000000 (> 1024)
    let wire_header = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5000000\r\n\r\n";
    client.write_all(wire_header.as_bytes()).await.unwrap();
    client.flush().await.unwrap();

    // Read response from server
    let mut resp_buf = vec![0u8; 4096];
    let n = client.read(&mut resp_buf).await.unwrap();
    let resp_str = String::from_utf8_lossy(&resp_buf[..n]);

    assert!(resp_str.contains("HTTP/1.1 413 Payload Too Large"));
    assert!(resp_str.contains("payload_too_large"));

    let server_res = server_handle.await.unwrap();
    assert!(matches!(server_res, Err(ServerError::PayloadTooLarge)));
}

#[tokio::test]
async fn test_challenge_server_headers_too_large_rejection() {
    let runtime = build_test_runtime();
    let server_config = ProtocolServerConfig {
        max_request_body_bytes: 1024,
        max_headers_bytes: 2048, // 2 KB header limit
        socket_timeout: Duration::from_secs(5),
        listen_addr: None,
    };
    let server = ProtocolServer::new(server_config, runtime);

    let (mut client, server_stream) = tokio::io::duplex(8192);
    let server_handle = tokio::spawn(async move {
        server.handle_connection(server_stream).await
    });

    // Send 3000 bytes of headers without \r\n\r\n
    let huge_header = format!("POST /v1/chat/completions HTTP/1.1\r\nX-Spam: {}\r\n", "A".repeat(3000));
    client.write_all(huge_header.as_bytes()).await.unwrap();
    client.flush().await.unwrap();

    let server_res = server_handle.await.unwrap();
    assert!(matches!(server_res, Err(ServerError::HeadersTooLarge)));
}

#[tokio::test]
async fn test_challenge_server_socket_read_timeout() {
    let runtime = build_test_runtime();
    let server_config = ProtocolServerConfig {
        max_request_body_bytes: 1024,
        max_headers_bytes: 2048,
        socket_timeout: Duration::from_millis(100), // Short 100ms timeout
        listen_addr: None,
    };
    let server = ProtocolServer::new(server_config, runtime);

    let (mut client, server_stream) = tokio::io::duplex(1024);
    let server_handle = tokio::spawn(async move {
        server.handle_connection(server_stream).await
    });

    // Write incomplete request and do not finish headers
    client.write_all(b"POST /v1/chat").await.unwrap();
    client.flush().await.unwrap();

    let server_res = server_handle.await.unwrap();
    assert!(matches!(server_res, Err(ServerError::Timeout)));
}

// ============================================================================
// 5. SSE Streaming Robustness & Chunked Wire Framing
// ============================================================================

#[tokio::test]
async fn test_challenge_sse_full_lifecycle_and_chunked_wire_framing() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen-sse-wire", 2048);

    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));
    let (mut client, server_stream) = tokio::io::duplex(16384);

    let server_handle = tokio::spawn(async move {
        server.handle_connection(server_stream).await
    });

    let chat_req = serde_json::json!({
        "model": "qwen-sse-wire",
        "messages": [{"role": "user", "content": "Stream wire test"}],
        "stream": true
    })
    .to_string();

    let wire_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
        chat_req.len(),
        chat_req
    );

    client.write_all(wire_req.as_bytes()).await.unwrap();
    client.flush().await.unwrap();

    let mut response_data = Vec::new();
    let mut temp = [0u8; 1024];
    loop {
        let n = client.read(&mut temp).await.unwrap();
        if n == 0 {
            break;
        }
        response_data.extend_from_slice(&temp[..n]);
        if response_data.ends_with(b"0\r\n\r\n") {
            break;
        }
    }

    let wire_output = String::from_utf8_lossy(&response_data);

    // Verify HTTP chunked encoding headers
    assert!(wire_output.contains("HTTP/1.1 200 OK"));
    assert!(wire_output.contains("Transfer-Encoding: chunked"));
    assert!(wire_output.contains("text/event-stream"));

    // Verify SSE frames inside chunks
    assert!(wire_output.contains(": keep-alive\n\n"));
    assert!(wire_output.contains(": queue_position="));
    assert!(wire_output.contains("chat.completion.chunk"));
    assert!(wire_output.contains("\"role\":\"assistant\""));
    assert!(wire_output.contains("\"finish_reason\":\"stop\""));
    assert!(wire_output.contains("data: [DONE]\n\n"));

    // Verify chunked terminating sequence
    assert!(wire_output.contains("0\r\n\r\n"));

    let server_res = server_handle.await.unwrap();
    assert!(server_res.is_ok());
}

#[tokio::test]
async fn test_challenge_sse_client_early_disconnect_does_not_panic() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen-sse-drop", 2048);

    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));
    let (mut client, server_stream) = tokio::io::duplex(4096);

    let server_handle = tokio::spawn(async move {
        server.handle_connection(server_stream).await
    });

    let chat_req = serde_json::json!({
        "model": "qwen-sse-drop",
        "messages": [{"role": "user", "content": "Stream drop test"}],
        "stream": true
    })
    .to_string();

    let wire_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
        chat_req.len(),
        chat_req
    );

    client.write_all(wire_req.as_bytes()).await.unwrap();
    client.flush().await.unwrap();

    // Read partial header then immediately drop client stream
    let mut temp = [0u8; 128];
    let _ = client.read(&mut temp).await.unwrap();
    drop(client);

    // Server should handle premature socket close gracefully without panic
    let server_res = server_handle.await.unwrap();
    // Result can be either Ok (if finished before write failure) or Err(Io)
    if let Err(e) = server_res {
        assert!(matches!(e, ServerError::Io(_)));
    }
}

// ============================================================================
// 6. Tool Calls Transparency & Deeply Nested Arguments Pass-Through
// ============================================================================

#[tokio::test]
async fn test_challenge_multi_tool_declarations_and_keyword_dispatch() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen-tools-multi", 2048);

    // Multi-tool declarations with 3 distinct tools
    let tools = vec![
        OpenAiTool {
            r#type: "function".to_owned(),
            function: OpenAiFunction {
                name: "lookup_stock_price".to_owned(),
                description: Some("Look up stock ticker price".to_owned()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "ticker": { "type": "string" } },
                    "required": ["ticker"]
                })),
            },
        },
        OpenAiTool {
            r#type: "function".to_owned(),
            function: OpenAiFunction {
                name: "fetch_weather_forecast".to_owned(),
                description: Some("Fetch weather report".to_owned()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                })),
            },
        },
        OpenAiTool {
            r#type: "function".to_owned(),
            function: OpenAiFunction {
                name: "execute_sql_query".to_owned(),
                description: Some("Run database query".to_owned()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "sql": { "type": "string" } },
                    "required": ["sql"]
                })),
            },
        },
    ];

    // Query targeting tool 3: "execute_sql_query"
    let req = OpenAiChatRequest {
        model: "qwen-tools-multi".to_owned(),
        messages: vec![OpenAiMessage::user("Please execute_sql_query for active users.")],
        stream: Some(false),
        tools: Some(tools),
        tool_choice: None,
        max_tokens: Some(128),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };

    let resp = OpenAiAdapter::handle_chat_completions(req, runtime).await;
    assert_eq!(resp.status, 200);

    let body = match resp.body {
        HttpResponseBody::Bytes(b) => String::from_utf8(b).unwrap(),
        _ => panic!("expected buffered byte body"),
    };
    let completion: OpenAiChatResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(completion.choices[0].finish_reason.as_deref(), Some("tool_calls"));

    let tc = &completion.choices[0].message.tool_calls.as_ref().unwrap()[0];
    assert_eq!(tc.function.name, "execute_sql_query");
}

#[tokio::test]
async fn test_challenge_nested_arguments_pass_through_intact() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen-nested-args", 2048);

    let complex_nested_arguments = serde_json::json!({
        "query": {
            "bool": {
                "must": [
                    { "term": { "user_id": 999 } },
                    { "range": { "timestamp": { "gte": "2026-01-01T00:00:00Z" } } }
                ],
                "filter": {
                    "tags": ["prod", "critical"]
                }
            }
        },
        "aggregations": {
            "status_counts": { "terms": { "field": "status.keyword" } }
        },
        "limit": 100
    })
    .to_string();

    let tool_call = OpenAiToolCall {
        id: "call_nested_xyz_789".to_owned(),
        r#type: "function".to_owned(),
        function: OpenAiFunctionCall {
            name: "complex_search_engine".to_owned(),
            arguments: complex_nested_arguments.clone(),
        },
    };

    let tools = vec![OpenAiTool {
        r#type: "function".to_owned(),
        function: OpenAiFunction {
            name: "complex_search_engine".to_owned(),
            description: Some("Complex search engine".to_owned()),
            parameters: None,
        },
    }];

    // 1. Non-streaming pass-through from previous assistant message
    let req_non_stream = OpenAiChatRequest {
        model: "qwen-nested-args".to_owned(),
        messages: vec![
            OpenAiMessage::user("Search database."),
            OpenAiMessage {
                role: "assistant".to_owned(),
                content: None,
                name: None,
                tool_calls: Some(vec![tool_call.clone()]),
                tool_call_id: None,
            },
        ],
        stream: Some(false),
        tools: Some(tools.clone()),
        tool_choice: None,
        max_tokens: Some(128),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };

    let resp_non_stream = OpenAiAdapter::handle_chat_completions(req_non_stream, Arc::clone(&runtime)).await;
    assert_eq!(resp_non_stream.status, 200);

    let body = match resp_non_stream.body {
        HttpResponseBody::Bytes(b) => String::from_utf8(b).unwrap(),
        _ => panic!("expected buffered byte body"),
    };
    let completion: OpenAiChatResponse = serde_json::from_str(&body).unwrap();
    let passed_tc = &completion.choices[0].message.tool_calls.as_ref().unwrap()[0];

    assert_eq!(passed_tc.id, "call_nested_xyz_789");
    assert_eq!(passed_tc.function.name, "complex_search_engine");
    assert_eq!(passed_tc.function.arguments, complex_nested_arguments);

    // 2. Streaming pass-through of nested arguments
    let req_stream = OpenAiChatRequest {
        model: "qwen-nested-args".to_owned(),
        messages: vec![
            OpenAiMessage::user("Search database again."),
            OpenAiMessage {
                role: "assistant".to_owned(),
                content: None,
                name: None,
                tool_calls: Some(vec![tool_call]),
                tool_call_id: None,
            },
        ],
        stream: Some(true),
        tools: Some(tools),
        tool_choice: None,
        max_tokens: Some(128),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };

    let resp_stream = OpenAiAdapter::handle_chat_completions(req_stream, runtime).await;
    assert_eq!(resp_stream.status, 200);

    let mut rx = match resp_stream.body {
        HttpResponseBody::Stream(rx) => rx,
        _ => panic!("expected stream body"),
    };

    let mut accumulated = String::new();
    while let Some(frame) = rx.recv().await {
        accumulated.push_str(&frame);
    }

    assert!(accumulated.contains("call_nested_xyz_789"));
    assert!(accumulated.contains("complex_search_engine"));
    assert!(accumulated.contains("\"finish_reason\":\"tool_calls\""));
    assert!(accumulated.contains("data: [DONE]"));
}

// ============================================================================
// 7. Routing Invariants & Error Mapping
// ============================================================================

#[tokio::test]
async fn test_challenge_unsupported_methods_and_paths_return_404() {
    let runtime = build_test_runtime();

    // Case A: DELETE method on /v1/chat/completions
    let req_delete = HttpRequest::new("DELETE", "/v1/chat/completions", Vec::new());
    let resp_delete = OpenAiAdapter::handle_http(req_delete, Arc::clone(&runtime)).await;
    assert_eq!(resp_delete.status, 404);

    // Case B: PUT method on /v1/chat/completions
    let req_put = HttpRequest::new("PUT", "/v1/chat/completions", Vec::new());
    let resp_put = OpenAiAdapter::handle_http(req_put, Arc::clone(&runtime)).await;
    assert_eq!(resp_put.status, 404);

    // Case C: POST method on /v1/models (only GET is supported)
    let req_post_models = HttpRequest::new("POST", "/v1/models", Vec::new());
    let resp_post_models = OpenAiAdapter::handle_http(req_post_models, Arc::clone(&runtime)).await;
    assert_eq!(resp_post_models.status, 404);

    // Case D: Unknown route on ProtocolServer
    let server = ProtocolServer::new(ProtocolServerConfig::default(), runtime);
    let req_unknown = HttpRequest::new("GET", "/v1/nonexistent", Vec::new());
    let resp_unknown = server.route_request(req_unknown).await;
    assert_eq!(resp_unknown.status, 404);
}

#[tokio::test]
async fn test_challenge_url_query_parameter_route_tolerance() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen2.5-params", 2048);
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    // 1. GET /v1/models?limit=10 on ProtocolServer
    let req_models_query = HttpRequest::new("GET", "/v1/models?limit=10", Vec::new());
    let resp_models_query = server.route_request(req_models_query).await;
    assert_eq!(resp_models_query.status, 200, "ProtocolServer should accept /v1/models?limit=10");

    // 2. GET /models?offset=0&limit=5 on ProtocolServer
    let req_models_query2 = HttpRequest::new("GET", "/models?offset=0&limit=5", Vec::new());
    let resp_models_query2 = server.route_request(req_models_query2).await;
    assert_eq!(resp_models_query2.status, 200, "ProtocolServer should accept /models?offset=0&limit=5");

    // 4. POST /v1/chat/completions?extra_param=1 on ProtocolServer
    let chat_body = serde_json::json!({
        "model": "qwen2.5-params",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false
    })
    .to_string();
    let req_chat_query = HttpRequest::new("POST", "/v1/chat/completions?extra_param=1", chat_body.clone().into_bytes());
    let resp_chat_query = server.route_request(req_chat_query).await;
    println!("ProtocolServer POST /v1/chat/completions?extra_param=1 status: {}", resp_chat_query.status);

    // 3. GET /v1/models?limit=10 on OpenAiAdapter::handle_http
    let req_adapter_models = HttpRequest::new("GET", "/v1/models?limit=10", Vec::new());
    let resp_adapter_models = OpenAiAdapter::handle_http(req_adapter_models, Arc::clone(&runtime)).await;
    println!("OpenAiAdapter GET /v1/models?limit=10 status: {}", resp_adapter_models.status);

    // 5. POST /v1/chat/completions?api_version=2024 on OpenAiAdapter::handle_http
    let req_adapter_chat = HttpRequest::new("POST", "/v1/chat/completions?api_version=2024", chat_body.into_bytes());
    let resp_adapter_chat = OpenAiAdapter::handle_http(req_adapter_chat, Arc::clone(&runtime)).await;
    println!("OpenAiAdapter POST /v1/chat/completions?api_version=2024 status: {}", resp_adapter_chat.status);

    assert_eq!(resp_chat_query.status, 200, "ProtocolServer should accept /v1/chat/completions?extra_param=1");
    assert_eq!(resp_adapter_models.status, 200, "OpenAiAdapter::handle_http should accept /v1/models?limit=10");
    assert_eq!(resp_adapter_chat.status, 200, "OpenAiAdapter::handle_http should accept /v1/chat/completions?api_version=2024");
}

#[tokio::test]
async fn test_challenge_client_early_disconnect_cleans_up_queue_permits() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "qwen-permit-check", 2048);

    // Stress loop: 5 successive premature disconnects
    for i in 0..5 {
        let (mut client, server_stream) = tokio::io::duplex(4096);
        let srv = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));
        let server_handle = tokio::spawn(async move {
            srv.handle_connection(server_stream).await
        });

        let chat_req = serde_json::json!({
            "model": "qwen-permit-check",
            "messages": [{"role": "user", "content": format!("Drop test {i}")}],
            "stream": true
        })
        .to_string();

        let wire_req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
            chat_req.len(),
            chat_req
        );

        client.write_all(wire_req.as_bytes()).await.unwrap();
        client.flush().await.unwrap();

        // Read a tiny snippet then immediately drop socket
        let mut temp = [0u8; 64];
        let _ = client.read(&mut temp).await.unwrap();
        drop(client);

        let _ = server_handle.await;

        // Yield to allow background tasks to complete permit cleanup
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Verify queue depth is exactly 0
        assert_eq!(
            runtime.queue_mgr().queue_depth("qwen-permit-check"),
            0,
            "Queue depth must be 0 after client disconnect"
        );
    }

    // After all 5 disconnects, an ordinary request must succeed immediately
    let final_req = OpenAiChatRequest {
        model: "qwen-permit-check".to_owned(),
        messages: vec![OpenAiMessage::user("Can I still execute?")],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(32),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let resp = OpenAiAdapter::handle_chat_completions(final_req, runtime).await;
    assert_eq!(resp.status, 200, "Subsequent request should succeed with 200");
}

