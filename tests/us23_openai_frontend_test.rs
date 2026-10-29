//! Unit and integration test suite for US-23: OpenAI-Compatible Protocol Front-End.
//!
//! Verifies:
//! - `GET /v1/models` discovery with explicit `unprofiled: true` tagging and tuning advisories.
//! - `POST /v1/chat/completions` non-streaming JSON responses.
//! - `POST /v1/chat/completions` real-time SSE streaming with `: keep-alive`, `: queue_position=<n>`,
//!   `chat.completion.chunk` events, and `data: [DONE]`.
//! - Transparent tool calling pass-through without server execution.
//! - Protocol-native error mappings: HTTP 400 on context length exceeded (zero silent truncation),
//!   HTTP 429 with `Retry-After` on queue saturation, HTTP 404 on missing model, and HTTP 413 on payload limit.

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
use k3_core::profile::{
    ComputePool, CONSERVATIVE_TUNING_ADVISORY, ModelListResponse, PriorityClass, TuningProfile,
};
use k3_manager::domain::{MockOsAffinityBackend, OsAffinityBackend};
use k3_manager::protocol::{
    HttpRequest, HttpResponseBody, OpenAiAdapter, OpenAiChatRequest, OpenAiChatResponse,
    OpenAiErrorResponse, OpenAiFunction, OpenAiMessage, OpenAiTool, ProtocolServer,
    ProtocolServerConfig,
};
use k3_manager::queue::AdmissionQueueConfig;
use k3_manager::runtime::{AdminCommand, ControlPlaneRuntime};

/// Builds an isolated in-memory test runtime.
fn build_test_runtime() -> Arc<ControlPlaneRuntime> {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid control plane capacity");
    let mock_affinity: Arc<dyn OsAffinityBackend> = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");
    Arc::new(runtime)
}

#[tokio::test]
async fn test_us23_get_models_surfaces_unprofiled_and_profiled_status() {
    let runtime = build_test_runtime();

    // 1. Register an unprofiled model
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-unprofiled".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register unprofiled model");

    // 2. Register and tune a profiled model
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-tuned".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-tuned.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model to tune");

    let mut custom_tuning = TuningProfile::conservative_default();
    custom_tuning.apply_operator_tuning(
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
            model_id: "qwen2.5-tuned".to_owned(),
            profile: custom_tuning,
        })
        .expect("apply tuning profile");

    // Query /v1/models via OpenAiAdapter
    let resp = OpenAiAdapter::handle_models(&runtime);
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
        .expect("unprofiled model in listing");
    assert!(unprofiled_entry.unprofiled);
    assert_eq!(unprofiled_entry.profile_status, "unprofiled");
    assert_eq!(
        unprofiled_entry.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    let tuned_entry = model_list
        .data
        .iter()
        .find(|m| m.id == "qwen2.5-tuned")
        .expect("tuned model in listing");
    assert!(!tuned_entry.unprofiled);
    assert_eq!(tuned_entry.profile_status, "profiled");
    assert!(tuned_entry.advisory_notice.is_none());
}

#[tokio::test]
async fn test_us23_chat_completions_non_streaming_json() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-7b".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-7b.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let req = OpenAiChatRequest {
        model: "qwen2.5-7b".to_owned(),
        messages: vec![
            OpenAiMessage::system("You are a helpful assistant."),
            OpenAiMessage::user("Hello! What is 2 + 2?"),
        ],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(128),
        max_completion_tokens: None,
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: None,
    };

    let resp = OpenAiAdapter::handle_chat_completions(req, runtime).await;
    assert_eq!(resp.status, 200);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let completion: OpenAiChatResponse =
        serde_json::from_slice(&body_bytes).expect("valid completion response json");

    assert!(completion.id.starts_with("chatcmpl-"));
    assert_eq!(completion.object, "chat.completion");
    assert_eq!(completion.model, "qwen2.5-7b");
    assert_eq!(completion.choices.len(), 1);
    assert_eq!(
        completion.choices[0].finish_reason.as_deref(),
        Some("stop")
    );
    assert!(completion.choices[0].message.content.is_some());
    assert!(completion.usage.prompt_tokens > 0);
    assert!(completion.usage.completion_tokens > 0);
    assert_eq!(
        completion.usage.total_tokens,
        completion.usage.prompt_tokens + completion.usage.completion_tokens
    );
}

#[tokio::test]
async fn test_us23_chat_completions_streaming_sse_lifecycle() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-stream".to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-stream.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let req = OpenAiChatRequest {
        model: "qwen2.5-stream".to_owned(),
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

    let resp = OpenAiAdapter::handle_chat_completions(req, runtime).await;
    assert_eq!(resp.status, 200);

    // Verify SSE headers
    let content_type = resp
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        .map(|(_, v)| v.as_str());
    assert!(
        content_type
            .unwrap_or_default()
            .contains("text/event-stream")
    );

    let mut rx = match resp.body {
        HttpResponseBody::Stream(rx) => rx,
        _ => panic!("expected stream response body"),
    };

    let mut received_frames = Vec::new();
    while let Some(frame) = rx.recv().await {
        received_frames.push(frame);
    }

    assert!(!received_frames.is_empty());

    // Verify queue feedback comments
    let has_keep_alive = received_frames.iter().any(|f| f.contains(": keep-alive"));
    let has_queue_pos = received_frames
        .iter()
        .any(|f| f.contains("queue_position=") || f.contains("queue-position"));
    assert!(has_keep_alive);
    assert!(has_queue_pos);

    // Verify chat.completion.chunk and [DONE]
    let has_chunk = received_frames
        .iter()
        .any(|f| f.contains("chat.completion.chunk"));
    let has_done = received_frames.iter().any(|f| f.contains("data: [DONE]"));
    assert!(has_chunk);
    assert!(has_done);
}

#[tokio::test]
async fn test_us23_tool_calls_passthrough_without_server_execution() {
    let runtime = build_test_runtime();
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
            description: Some("Get the current weather for a location".to_owned()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            })),
        },
    }];

    // 1. Non-streaming tool calls pass-through
    let req_non_stream = OpenAiChatRequest {
        model: "qwen2.5-tools".to_owned(),
        messages: vec![OpenAiMessage::user("What is the weather in Boston?")],
        stream: Some(false),
        tools: Some(tools.clone()),
        tool_choice: None,
        max_tokens: Some(128),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };

    let resp_non_stream =
        OpenAiAdapter::handle_chat_completions(req_non_stream, Arc::clone(&runtime)).await;
    assert_eq!(resp_non_stream.status, 200);

    let body_bytes = match resp_non_stream.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let completion: OpenAiChatResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        completion.choices[0].finish_reason.as_deref(),
        Some("tool_calls")
    );
    let tool_calls = completion.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls present");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_current_weather");
    assert!(tool_calls[0].function.arguments.contains("Boston"));

    // 2. Streaming tool calls pass-through
    let req_stream = OpenAiChatRequest {
        model: "qwen2.5-tools".to_owned(),
        messages: vec![OpenAiMessage::user("Check the weather please.")],
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
        _ => panic!("expected stream response body"),
    };

    let mut all_frames = String::new();
    while let Some(frame) = rx.recv().await {
        all_frames.push_str(&frame);
    }

    assert!(all_frames.contains("get_current_weather"));
    assert!(all_frames.contains("tool_calls"));
    assert!(all_frames.contains("data: [DONE]"));
}

#[tokio::test]
async fn test_us23_context_length_exceeded_returns_400_without_silent_truncation() {
    let runtime = build_test_runtime();

    // Register model and apply constrained context limit (n_ctx = 10)
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "constrained-model".to_owned(),
            weights_path: PathBuf::from("/models/constrained.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let mut constrained_tuning = TuningProfile::conservative_default();
    constrained_tuning.apply_operator_tuning(
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
            model_id: "constrained-model".to_owned(),
            profile: constrained_tuning,
        })
        .expect("apply constrained tuning profile");

    // Construct oversized prompt with 25 words (> 10 tokens)
    let long_prompt =
        "The quick brown fox jumps over the lazy dog repeatedly until the context length is completely saturated and breached by the user.";
    let req = OpenAiChatRequest {
        model: "constrained-model".to_owned(),
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

    let resp = OpenAiAdapter::handle_chat_completions(req, runtime).await;
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
    assert!(err_resp.error.message.contains("context window limit"));
}

#[tokio::test]
async fn test_us23_queue_saturation_returns_429_with_retry_after() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "saturated-model".to_owned(),
            weights_path: PathBuf::from("/models/saturated.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Configure queue capacity with max_depth = 1
    runtime.queue_mgr().register_model(
        "saturated-model",
        Some(AdmissionQueueConfig {
            max_depth: 1,
            keep_alive_interval: Duration::from_millis(100),
            wait_timeout: None,
            default_retry_after_secs: 7,
            estimated_pass_latency_ms: 25.0,
            estimated_passes_per_request: 16,
        }),
    );

    // Admit request 1 to fill queue
    let pos = runtime.queue_mgr().admit_to_model(
        "saturated-model",
        "req-1".to_owned(),
        false,
        None,
        None,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        None,
        None,
    );
    assert!(pos.is_ok());

    // Request 2 should hit queue saturation
    let req = OpenAiChatRequest {
        model: "saturated-model".to_owned(),
        messages: vec![OpenAiMessage::user("Another request while queue is full.")],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(32),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };

    let resp = OpenAiAdapter::handle_chat_completions(req, runtime).await;
    assert_eq!(resp.status, 429);

    let retry_after = resp
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
        .map(|(_, v)| v.as_str());
    assert_eq!(retry_after, Some("1"));

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let err_resp: OpenAiErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.error.code.as_deref(), Some("rate_limit_exceeded"));
}

#[tokio::test]
async fn test_us23_unknown_model_returns_404_not_found() {
    let runtime = build_test_runtime();

    let req = OpenAiChatRequest {
        model: "nonexistent-model-xyz".to_owned(),
        messages: vec![OpenAiMessage::user("Hello?")],
        stream: Some(false),
        tools: None,
        tool_choice: None,
        max_tokens: Some(32),
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };

    let resp = OpenAiAdapter::handle_chat_completions(req, runtime).await;
    assert_eq!(resp.status, 404);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let err_resp: OpenAiErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.error.code.as_deref(), Some("model_not_found"));
}

#[tokio::test]
async fn test_us23_protocol_server_enforces_payload_limits_413() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen-server".to_owned(),
            weights_path: PathBuf::from("/models/qwen.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let server_config = ProtocolServerConfig {
        max_request_body_bytes: 512, // Strict 512 byte limit
        max_headers_bytes: 4096,
        socket_timeout: Duration::from_secs(5),
        listen_addr: None,
    };
    let server = ProtocolServer::new(server_config, runtime);

    // Send payload of 1024 bytes (> 512)
    let big_body = vec![b'a'; 1024];
    let req = HttpRequest::new("POST", "/v1/chat/completions", big_body);

    let resp = server.route_request(req).await;
    assert_eq!(resp.status, 413);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let json_str = String::from_utf8(body_bytes).unwrap();
    assert!(json_str.contains("payload_too_large"));
}

#[tokio::test]
async fn test_us23_malformed_json_returns_400() {
    let runtime = build_test_runtime();
    let malformed_bytes = b"{\"model\": \"qwen\", \"messages\": [".to_vec();
    let req = HttpRequest::new("POST", "/v1/chat/completions", malformed_bytes);

    let resp = OpenAiAdapter::handle_http(req, runtime).await;
    assert_eq!(resp.status, 400);

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected buffered byte body"),
    };
    let json_str = String::from_utf8(body_bytes).unwrap();
    assert!(json_str.contains("invalid_payload"));
}
