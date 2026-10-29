//! Unit and integration test suite for US-24: Anthropic Claude-Compatible Protocol Front-End.
//!
//! Verifies:
//! - `POST /v1/messages` non-streaming JSON responses with multi-part content blocks.
//! - `POST /v1/messages` streaming SSE with full Anthropic event lifecycle:
//!   `message_start`, `content_block_start`, `thinking_delta`, `text_delta`, `content_block_stop`,
//!   `message_delta`, and `message_stop`.
//! - Transparent `tool_use` content blocks pass-through without server execution.
//! - Canonical prompt translation (`translate_to_canonical_prompt`) for system, text, thinking, and tools.
//! - Protocol-native error mappings: HTTP 400 on context length exceeded (zero silent truncation),
//!   HTTP 529 (`overloaded_error` with `Retry-After`) on queue saturation, HTTP 400 on invalid role sequence
//!   or zero max_tokens, and HTTP 404 on missing model.

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
    ClaudeAdapter, ClaudeContent, ClaudeContentBlock, ClaudeErrorResponse, ClaudeMessage,
    ClaudeMessageRequest, ClaudeMessageResponse, ClaudeSystem, ClaudeThinkingConfig, ClaudeTool,
    HttpRequest, HttpResponseBody,
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
async fn test_us24_claude_messages_non_streaming_json() {
    let runtime = build_test_runtime();
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

    let resp = ClaudeAdapter::handle_messages(req, runtime).await;
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
    assert!(message_resp.content[0].text.is_some());
    assert_eq!(message_resp.stop_reason.as_deref(), Some("end_turn"));
    assert!(message_resp.usage.input_tokens > 0);
    assert!(message_resp.usage.output_tokens > 0);
}

#[tokio::test]
async fn test_us24_claude_streaming_sse_full_lifecycle() {
    let runtime = build_test_runtime();
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

    let resp = ClaudeAdapter::handle_messages(req, runtime).await;
    assert_eq!(resp.status, 200);

    let mut rx = match resp.body {
        HttpResponseBody::Stream(rx) => rx,
        _ => panic!("expected stream body"),
    };

    let mut events = Vec::new();
    while let Some(ev) = rx.recv().await {
        events.push(ev);
    }

    assert!(!events.is_empty());

    let all_stream = events.join("");
    // 1. Keep-alive comment while queued
    assert!(all_stream.contains(": keep-alive\n\n"));
    // 2. message_start
    assert!(all_stream.contains("event: message_start"));
    assert!(all_stream.contains("\"type\":\"message_start\""));
    // 3. content_block_start
    assert!(all_stream.contains("event: content_block_start"));
    assert!(all_stream.contains("\"type\":\"content_block_start\""));
    // 4. content_block_delta with text_delta
    assert!(all_stream.contains("event: content_block_delta"));
    assert!(all_stream.contains("\"type\":\"text_delta\""));
    // 5. content_block_stop
    assert!(all_stream.contains("event: content_block_stop"));
    // 6. message_delta
    assert!(all_stream.contains("event: message_delta"));
    assert!(all_stream.contains("\"stop_reason\":\"end_turn\""));
    // 7. message_stop
    assert!(all_stream.contains("event: message_stop"));
}

#[tokio::test]
async fn test_us24_claude_thinking_delta_reasoning_stream() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-reasoning-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-reasoning.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let req = ClaudeMessageRequest {
        model: "claude-reasoning-model".to_owned(),
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

    let resp = ClaudeAdapter::handle_messages(req, runtime).await;
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
    // Both thinking_delta and text_delta must be emitted in correct order
    assert!(all_stream.contains("\"type\":\"thinking_delta\""));
    assert!(all_stream.contains("step by step"));
    assert!(all_stream.contains("\"type\":\"text_delta\""));
    assert!(all_stream.contains("message_stop"));
}

#[tokio::test]
async fn test_us24_claude_tool_use_content_block_passthrough() {
    let runtime = build_test_runtime();
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
            "properties": {
                "location": { "type": "string" }
            },
            "required": ["location"]
        }),
    }];

    // 1. Non-streaming tool use pass-through
    let req_non_stream = ClaudeMessageRequest {
        model: "claude-tools-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Please get the weather in San Francisco")],
        max_tokens: 512,
        thinking: None,
        tools: Some(tools.clone()),
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };

    let resp = ClaudeAdapter::handle_messages(req_non_stream, Arc::clone(&runtime)).await;
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
        .expect("tool_use block in response");
    assert_eq!(tool_block.name.as_deref(), Some("get_weather"));
    assert!(tool_block.id.is_some());
    assert!(tool_block.input.is_some());

    // 2. Streaming tool use pass-through
    let req_stream = ClaudeMessageRequest {
        model: "claude-tools-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Please get weather for me.")],
        max_tokens: 512,
        thinking: None,
        tools: Some(tools),
        tool_choice: None,
        stop_sequences: None,
        stream: Some(true),
        temperature: None,
        top_p: None,
    };

    let resp_stream = ClaudeAdapter::handle_messages(req_stream, runtime).await;
    assert_eq!(resp_stream.status, 200);

    let mut rx = match resp_stream.body {
        HttpResponseBody::Stream(rx) => rx,
        _ => panic!("expected stream body"),
    };

    let mut all_stream = String::new();
    while let Some(ev) = rx.recv().await {
        all_stream.push_str(&ev);
    }

    assert!(all_stream.contains("\"type\":\"tool_use\""));
    assert!(all_stream.contains("\"type\":\"input_json_delta\""));
    assert!(all_stream.contains("tool_use"));
}

#[tokio::test]
async fn test_us24_claude_context_length_exceeded_returns_400() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-small-ctx".to_owned(),
            weights_path: PathBuf::from("/models/claude-small.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let mut tuning = TuningProfile::conservative_default();
    tuning.apply_operator_tuning(
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
            model_id: "claude-small-ctx".to_owned(),
            profile: tuning,
        })
        .expect("apply tuning profile");

    let long_prompt =
        "This is an excessively long prompt designed to exceed the strict 10 tokens context bound for testing context overflow.";
    let req = ClaudeMessageRequest {
        model: "claude-small-ctx".to_owned(),
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

    let resp = ClaudeAdapter::handle_messages(req, runtime).await;
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
async fn test_us24_claude_queue_saturation_returns_529_with_retry_after() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-saturated".to_owned(),
            weights_path: PathBuf::from("/models/claude-saturated.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    runtime.queue_mgr().register_model(
        "claude-saturated",
        Some(AdmissionQueueConfig {
            max_depth: 1,
            keep_alive_interval: Duration::from_millis(100),
            wait_timeout: None,
            default_retry_after_secs: 5,
            estimated_pass_latency_ms: 25.0,
            estimated_passes_per_request: 16,
        }),
    );

    // Fill queue to capacity (1 request)
    let pos = runtime.queue_mgr().admit_to_model(
        "claude-saturated",
        "req-1".to_owned(),
        false,
        None,
        None,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        None,
        None,
    );
    assert!(pos.is_ok());

    // Next request triggers HTTP 529
    let req = ClaudeMessageRequest {
        model: "claude-saturated".to_owned(),
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

    let resp = ClaudeAdapter::handle_messages(req, runtime).await;
    assert_eq!(resp.status, 529);

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
    let err_resp: ClaudeErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.error.r#type, "overloaded_error");
}

#[tokio::test]
async fn test_us24_claude_invalid_max_tokens_and_consecutive_roles_400() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-val-model".to_owned(),
            weights_path: PathBuf::from("/models/claude.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // 1. max_tokens = 0
    let req_zero_max = ClaudeMessageRequest {
        model: "claude-val-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Hi")],
        max_tokens: 0,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp1 = ClaudeAdapter::handle_messages(req_zero_max, Arc::clone(&runtime)).await;
    assert_eq!(resp1.status, 400);

    // 2. Consecutive user messages
    let req_consecutive = ClaudeMessageRequest {
        model: "claude-val-model".to_owned(),
        system: None,
        messages: vec![
            ClaudeMessage::user("First user message"),
            ClaudeMessage::user("Second user message immediately"),
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
    let resp2 = ClaudeAdapter::handle_messages(req_consecutive, runtime).await;
    assert_eq!(resp2.status, 400);
}

#[tokio::test]
async fn test_us24_claude_canonical_prompt_translation_structure() {
    let req = ClaudeMessageRequest {
        model: "test-model".to_owned(),
        system: Some(ClaudeSystem::Text("You are Claude.".to_owned())),
        messages: vec![
            ClaudeMessage {
                role: "user".to_owned(),
                content: ClaudeContent::Blocks(vec![
                    ClaudeContentBlock {
                        r#type: "text".to_owned(),
                        text: Some("Check database status.".to_owned()),
                        thinking: None,
                        id: None,
                        name: None,
                        input: None,
                        tool_use_id: None,
                        content: None,
                        is_error: None,
                        source: None,
                    },
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
                ]),
            },
            ClaudeMessage {
                role: "assistant".to_owned(),
                content: ClaudeContent::Blocks(vec![ClaudeContentBlock {
                    r#type: "tool_use".to_owned(),
                    text: None,
                    thinking: None,
                    id: Some("call_1".to_owned()),
                    name: Some("db_query".to_owned()),
                    input: Some(serde_json::json!({ "table": "users" })),
                    tool_use_id: None,
                    content: None,
                    is_error: None,
                    source: None,
                }]),
            },
            ClaudeMessage {
                role: "user".to_owned(),
                content: ClaudeContent::Blocks(vec![ClaudeContentBlock {
                    r#type: "tool_result".to_owned(),
                    text: None,
                    thinking: None,
                    id: None,
                    name: None,
                    input: None,
                    tool_use_id: Some("call_1".to_owned()),
                    content: Some(serde_json::json!({ "rows": 42 })),
                    is_error: None,
                    source: None,
                }]),
            },
        ],
        max_tokens: 100,
        thinking: None,
        tools: Some(vec![ClaudeTool {
            name: "db_query".to_owned(),
            description: Some("Query database".to_owned()),
            input_schema: serde_json::json!({ "type": "object" }),
        }]),
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };

    let prompt = ClaudeAdapter::translate_to_canonical_prompt(&req);
    assert!(prompt.contains("[System: You are Claude.]"));
    assert!(prompt.contains("[Tools Available:"));
    assert!(prompt.contains("db_query: Query database"));
    assert!(prompt.contains("User: Check database status.[Image]"));
    assert!(prompt.contains("<tool_call>"));
    assert!(prompt.contains("<tool_response>"));
}

#[tokio::test]
async fn test_us24_claude_http_handler_dispatch_and_404() {
    let runtime = build_test_runtime();

    let invalid_route = HttpRequest::new("POST", "/v1/invalid", b"{}".to_vec());
    let resp = ClaudeAdapter::handle_http(invalid_route, runtime).await;
    assert_eq!(resp.status, 404);
}
