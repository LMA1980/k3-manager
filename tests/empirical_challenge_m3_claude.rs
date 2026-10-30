//! Empirical challenge test suite for Milestone M3: Anthropic Claude Protocol Front-End (US-24).
//!
//! Adversarial and empirical stress tests covering:
//! 1. Anthropic message invariants:
//!    - Consecutive user turns rejection (HTTP 400).
//!    - Empty messages array rejection (HTTP 400).
//!    - Invalid message roles ("system", "tool", "function", empty) rejection (HTTP 400).
//!    - Zero or negative `max_tokens` rejection (HTTP 400).
//!    - System prompt handling: top-level string vs structured blocks vs rejected in-turn system message.
//! 2. Complex multi-part content blocks:
//!    - Mixed text, images (base64 and URL sources), `tool_use`, and `tool_result` blocks.
//!    - Intact round-trip preservation of deeply nested tool input parameters without JSON corruption.
//! 3. Anthropic SSE event lifecycle:
//!    - Strict ordering: `message_start` -> `content_block_start` -> `thinking_delta` / `text_delta` / `input_json_delta` -> `content_block_stop` -> `message_delta` -> `message_stop`.
//!    - Correct SSE framing with `event:`, `data:`, and trailing `\n\n`.
//!    - Keep-alive comments (`: keep-alive\n\n`, `: queue-position <n>\n\n`) during queue wait.
//! 4. Error handling:
//!    - Admission queue saturation: HTTP 529 with `overloaded_error` and `Retry-After` header.
//!    - Context length overflow: HTTP 400 with `invalid_request_error` without silent truncation.
//!    - Missing model: HTTP 404 with `not_found_error`.
//! 5. ProtocolServer router and security bounds:
//!    - 10MB payload size limit enforcement (HTTP 413).
//!    - Malformed JSON payload rejection (HTTP 400).
//!    - Method and route dispatching (`POST /v1/messages`, `POST /messages`, `GET` rejection).

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
    ClaudeMessageRequest, ClaudeMessageResponse, ClaudeSystem, ClaudeSystemBlock,
    ClaudeThinkingConfig, ClaudeTool, HttpRequest, HttpResponseBody, ProtocolServer,
    ProtocolServerConfig,
};
use k3_manager::queue::AdmissionQueueConfig;
use k3_manager::runtime::{AdminCommand, ControlPlaneRuntime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

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

// ============================================================================
// Challenge 1: Anthropic Message Invariants
// ============================================================================

#[tokio::test]
async fn test_empirical_consecutive_user_turns_and_role_invariants() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-invariants-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-invariants.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Case 1a: Two consecutive user messages at the beginning -> HTTP 400
    let req_consec_start = ClaudeMessageRequest {
        model: "claude-invariants-model".to_owned(),
        system: None,
        messages: vec![
            ClaudeMessage::user("Hello first turn"),
            ClaudeMessage::user("Hello second consecutive user turn"),
        ],
        max_tokens: 256,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp = ClaudeAdapter::handle_messages(req_consec_start, Arc::clone(&runtime)).await;
    assert_eq!(
        resp.status, 400,
        "Consecutive user messages at start must be rejected with HTTP 400"
    );
    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected bytes"),
    };
    let err: ClaudeErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err.error.r#type, "invalid_request_error");
    assert!(err.error.message.contains("consecutive user messages"));

    // Case 1b: Consecutive user messages in the middle of a multi-turn conversation -> HTTP 400
    let req_consec_middle = ClaudeMessageRequest {
        model: "claude-invariants-model".to_owned(),
        system: None,
        messages: vec![
            ClaudeMessage::user("Turn 1 user"),
            ClaudeMessage::assistant("Turn 2 assistant"),
            ClaudeMessage::user("Turn 3 user"),
            ClaudeMessage::user("Turn 4 user consecutive violation"),
            ClaudeMessage::assistant("Turn 5 assistant"),
        ],
        max_tokens: 256,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp = ClaudeAdapter::handle_messages(req_consec_middle, Arc::clone(&runtime)).await;
    assert_eq!(
        resp.status, 400,
        "Consecutive user messages in middle must return 400"
    );

    // Case 1c: Empty messages array -> HTTP 400
    let req_empty_messages = ClaudeMessageRequest {
        model: "claude-invariants-model".to_owned(),
        system: None,
        messages: Vec::new(),
        max_tokens: 256,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp = ClaudeAdapter::handle_messages(req_empty_messages, Arc::clone(&runtime)).await;
    assert_eq!(resp.status, 400, "Empty messages array must return 400");

    // Case 1d: Invalid role "system" in messages array -> HTTP 400
    let req_invalid_role_system = ClaudeMessageRequest {
        model: "claude-invariants-model".to_owned(),
        system: None,
        messages: vec![
            ClaudeMessage {
                role: "system".to_owned(),
                content: ClaudeContent::Text("System instructions in messages array".to_owned()),
            },
            ClaudeMessage::user("Hello"),
        ],
        max_tokens: 256,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp = ClaudeAdapter::handle_messages(req_invalid_role_system, Arc::clone(&runtime)).await;
    assert_eq!(
        resp.status, 400,
        "System role in messages array must return 400"
    );

    // Case 1e: Invalid role "tool" or "function" in messages array -> HTTP 400
    let req_invalid_role_tool = ClaudeMessageRequest {
        model: "claude-invariants-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage {
            role: "tool".to_owned(),
            content: ClaudeContent::Text("Tool result".to_owned()),
        }],
        max_tokens: 256,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp = ClaudeAdapter::handle_messages(req_invalid_role_tool, Arc::clone(&runtime)).await;
    assert_eq!(resp.status, 400, "Non-standard role must return 400");

    // Case 1f: First message with assistant role -> HTTP 400
    let req_first_assistant = ClaudeMessageRequest {
        model: "claude-invariants-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::assistant("I am speaking first")],
        max_tokens: 256,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp = ClaudeAdapter::handle_messages(req_first_assistant, Arc::clone(&runtime)).await;
    assert_eq!(
        resp.status, 400,
        "First message as assistant must return 400"
    );
    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected bytes"),
    };
    let err: ClaudeErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err.error.r#type, "invalid_request_error");
    assert!(
        err.error
            .message
            .contains("first message must have role 'user'")
    );

    // Case 1g: Consecutive assistant messages -> HTTP 400
    let req_consec_assistant = ClaudeMessageRequest {
        model: "claude-invariants-model".to_owned(),
        system: None,
        messages: vec![
            ClaudeMessage::user("Hello user"),
            ClaudeMessage::assistant("Assistant reply 1"),
            ClaudeMessage::assistant("Assistant reply 2 consecutive violation"),
        ],
        max_tokens: 256,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp = ClaudeAdapter::handle_messages(req_consec_assistant, Arc::clone(&runtime)).await;
    assert_eq!(
        resp.status, 400,
        "Consecutive assistant messages must return 400"
    );
    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected bytes"),
    };
    let err: ClaudeErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err.error.r#type, "invalid_request_error");
    assert!(err.error.message.contains("consecutive assistant messages"));

    // Case 1h: Valid alternating multi-turn conversation -> HTTP 200
    let req_valid_alternating = ClaudeMessageRequest {
        model: "claude-invariants-model".to_owned(),
        system: None,
        messages: vec![
            ClaudeMessage::user("User turn 1"),
            ClaudeMessage::assistant("Assistant turn 2"),
            ClaudeMessage::user("User turn 3"),
        ],
        max_tokens: 256,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp = ClaudeAdapter::handle_messages(req_valid_alternating, runtime).await;
    assert_eq!(
        resp.status, 200,
        "Alternating user/assistant turns must succeed with 200"
    );
}

#[tokio::test]
async fn test_empirical_max_tokens_zero_and_negative_rejection() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-tokens-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-tokens.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Case 2a: max_tokens = 0 in Rust struct -> HTTP 400
    let req_zero = ClaudeMessageRequest {
        model: "claude-tokens-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Hello")],
        max_tokens: 0,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp = ClaudeAdapter::handle_messages(req_zero, Arc::clone(&runtime)).await;
    assert_eq!(resp.status, 400, "max_tokens = 0 must return 400");
    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected bytes"),
    };
    let err: ClaudeErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err.error.r#type, "invalid_request_error");
    assert!(err.error.message.contains("max_tokens must be > 0"));

    // Case 2b: Negative max_tokens in HTTP JSON body -> HTTP 400
    let raw_negative_json = serde_json::json!({
        "model": "claude-tokens-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": -50
    })
    .to_string();
    let http_req_neg = HttpRequest::new("POST", "/v1/messages", raw_negative_json.into_bytes());
    let resp_neg = ClaudeAdapter::handle_http(http_req_neg, Arc::clone(&runtime)).await;
    assert_eq!(
        resp_neg.status, 400,
        "Negative max_tokens in JSON must return 400"
    );

    // Case 2c: Missing max_tokens in HTTP JSON body -> HTTP 400
    let raw_missing_json = serde_json::json!({
        "model": "claude-tokens-model",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    .to_string();
    let http_req_missing = HttpRequest::new("POST", "/v1/messages", raw_missing_json.into_bytes());
    let resp_missing = ClaudeAdapter::handle_http(http_req_missing, Arc::clone(&runtime)).await;
    assert_eq!(
        resp_missing.status, 400,
        "Missing required max_tokens must return 400"
    );

    // Case 2d: Non-integer string max_tokens -> HTTP 400
    let raw_string_json = serde_json::json!({
        "model": "claude-tokens-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": "unlimited"
    })
    .to_string();
    let http_req_str = HttpRequest::new("POST", "/v1/messages", raw_string_json.into_bytes());
    let resp_str = ClaudeAdapter::handle_http(http_req_str, Arc::clone(&runtime)).await;
    assert_eq!(resp_str.status, 400, "String max_tokens must return 400");

    // Case 2e: Valid minimal positive max_tokens = 1 -> HTTP 200
    let req_one = ClaudeMessageRequest {
        model: "claude-tokens-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Hello")],
        max_tokens: 1,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp_one = ClaudeAdapter::handle_messages(req_one, runtime).await;
    assert_eq!(resp_one.status, 200, "max_tokens = 1 must return 200");
}

#[tokio::test]
async fn test_empirical_system_prompt_handling_invariants() {
    // 1. System prompt as a single string
    let req_text = ClaudeMessageRequest {
        model: "test-model".to_owned(),
        system: Some(ClaudeSystem::Text(
            "You are an expert Rust compiler engineer.".to_owned(),
        )),
        messages: vec![ClaudeMessage::user("Compile this code.")],
        max_tokens: 100,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let prompt_text = ClaudeAdapter::translate_to_canonical_prompt(&req_text);
    assert!(prompt_text.starts_with("[System: You are an expert Rust compiler engineer.]\n\n"));
    assert!(prompt_text.contains("User: Compile this code."));

    // 2. System prompt as an array of structured text blocks
    let req_blocks = ClaudeMessageRequest {
        model: "test-model".to_owned(),
        system: Some(ClaudeSystem::Blocks(vec![
            ClaudeSystemBlock {
                r#type: "text".to_owned(),
                text: "Primary directive: Be concise and accurate.".to_owned(),
            },
            ClaudeSystemBlock {
                r#type: "text".to_owned(),
                text: "Secondary directive: Never fabricate syntax.".to_owned(),
            },
        ])),
        messages: vec![ClaudeMessage::user("Hi")],
        max_tokens: 100,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let prompt_blocks = ClaudeAdapter::translate_to_canonical_prompt(&req_blocks);
    assert!(prompt_blocks.contains("Primary directive: Be concise and accurate."));
    assert!(prompt_blocks.contains("Secondary directive: Never fabricate syntax."));
    assert!(prompt_blocks.contains("[System: Primary directive: Be concise and accurate.\n\nSecondary directive: Never fabricate syntax.]\n\n"));

    // 3. System prompt omitted (None)
    let req_none = ClaudeMessageRequest {
        model: "test-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Hi without system prompt")],
        max_tokens: 100,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let prompt_none = ClaudeAdapter::translate_to_canonical_prompt(&req_none);
    assert!(!prompt_none.contains("[System:"));
    assert!(prompt_none.starts_with("User: Hi without system prompt"));
}

// ============================================================================
// Challenge 2: Complex Multi-Part Content Blocks & Nested JSON Preservation
// ============================================================================

#[tokio::test]
async fn test_empirical_multipart_content_blocks_text_images_tools() {
    let req = ClaudeMessageRequest {
        model: "test-model".to_owned(),
        system: Some(ClaudeSystem::Text("Multi-modal assistant".to_owned())),
        messages: vec![
            // User turn with mixed text and images (base64 and URL sources)
            ClaudeMessage {
                role: "user".to_owned(),
                content: ClaudeContent::Blocks(vec![
                    ClaudeContentBlock {
                        r#type: "text".to_owned(),
                        text: Some("Inspect diagram A:".to_owned()),
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
                        source: Some(serde_json::json!({
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY44YAAAAASUVORK5CYII="
                        })),
                    },
                    ClaudeContentBlock {
                        r#type: "text".to_owned(),
                        text: Some("And compare with diagram B:".to_owned()),
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
                        source: Some(serde_json::json!({
                            "type": "url",
                            "url": "https://example.org/diagram_b.jpg"
                        })),
                    },
                ]),
            },
            // Assistant turn with tool_use
            ClaudeMessage {
                role: "assistant".to_owned(),
                content: ClaudeContent::Blocks(vec![
                    ClaudeContentBlock {
                        r#type: "thinking".to_owned(),
                        text: None,
                        thinking: Some("Need to query database for image metadata.".to_owned()),
                        id: None,
                        name: None,
                        input: None,
                        tool_use_id: None,
                        content: None,
                        is_error: None,
                        source: None,
                    },
                    ClaudeContentBlock {
                        r#type: "tool_use".to_owned(),
                        text: None,
                        thinking: None,
                        id: Some("toolu_query_001".to_owned()),
                        name: Some("db_lookup".to_owned()),
                        input: Some(serde_json::json!({
                            "table": "images",
                            "keys": ["diagram_a", "diagram_b"]
                        })),
                        tool_use_id: None,
                        content: None,
                        is_error: None,
                        source: None,
                    },
                ]),
            },
            // User turn with tool_result
            ClaudeMessage {
                role: "user".to_owned(),
                content: ClaudeContent::Blocks(vec![ClaudeContentBlock {
                    r#type: "tool_result".to_owned(),
                    text: None,
                    thinking: None,
                    id: None,
                    name: None,
                    input: None,
                    tool_use_id: Some("toolu_query_001".to_owned()),
                    content: Some(serde_json::json!({
                        "matches": 2,
                        "hashes": ["sha256-a", "sha256-b"]
                    })),
                    is_error: Some(false),
                    source: None,
                }]),
            },
        ],
        max_tokens: 512,
        thinking: None,
        tools: Some(vec![ClaudeTool {
            name: "db_lookup".to_owned(),
            description: Some("Lookup database entities".to_owned()),
            input_schema: serde_json::json!({ "type": "object" }),
        }]),
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };

    let prompt = ClaudeAdapter::translate_to_canonical_prompt(&req);

    // Verify all multi-part elements are represented in canonical prompt
    assert!(prompt.contains("[System: Multi-modal assistant]"));
    assert!(prompt.contains("User: Inspect diagram A:[Image]And compare with diagram B:[Image]"));
    assert!(
        prompt.contains(
            "Assistant: [Thinking: Need to query database for image metadata.]<tool_call>"
        )
    );
    assert!(prompt.contains("\"table\":\"images\""));
    assert!(prompt.contains("</tool_call>"));
    assert!(prompt.contains("User: <tool_response>"));
    assert!(prompt.contains("\"matches\":2"));
    assert!(prompt.contains("</tool_response>"));
}

#[tokio::test]
async fn test_empirical_preserve_deeply_nested_tool_input_json_roundtrip() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-nested-tools".to_owned(),
            weights_path: PathBuf::from("/models/claude-nested.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Deeply nested input with arrays, booleans, floats, null, escaped quotes, newlines, and unicode
    let complex_nested_input = serde_json::json!({
        "query": {
            "filters": [
                { "field": "status", "op": "in", "values": ["active", "pending", "archived"] },
                { "field": "score", "op": "gte", "values": [99.5, -3.75] },
                { "field": "optional", "op": "eq", "values": [null] }
            ],
            "paging": {
                "offset": 0,
                "limit": 50,
                "sort_by": ["created_at:desc", "id:asc"]
            }
        },
        "flags": {
            "dry_run": false,
            "enable_tracing": true
        },
        "annotations": {
            "description": "Quotes \"and\" \\escapes\\ \n newlines \t tabs",
            "unicode": "SpacemiT 进迭时空 🚀 RVV 1.0",
            "empty_obj": {},
            "empty_arr": []
        }
    });

    let tools = vec![ClaudeTool {
        name: "complex_pipeline_executor".to_owned(),
        description: Some("Execute data pipeline with nested schema".to_owned()),
        input_schema: serde_json::json!({ "type": "object" }),
    }];

    // 1. Non-streaming roundtrip verification
    let req_non_stream = ClaudeMessageRequest {
        model: "claude-nested-tools".to_owned(),
        system: None,
        messages: vec![
            ClaudeMessage::user("Execute complex pipeline query"),
            ClaudeMessage {
                role: "assistant".to_owned(),
                content: ClaudeContent::Blocks(vec![ClaudeContentBlock {
                    r#type: "tool_use".to_owned(),
                    text: None,
                    thinking: None,
                    id: Some("toolu_nested_999".to_owned()),
                    name: Some("complex_pipeline_executor".to_owned()),
                    input: Some(complex_nested_input.clone()),
                    tool_use_id: None,
                    content: None,
                    is_error: None,
                    source: None,
                }]),
            },
            ClaudeMessage::user("Follow-up question on pipeline result"),
        ],
        max_tokens: 512,
        thinking: None,
        tools: Some(tools.clone()),
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };

    let resp_non_stream =
        ClaudeAdapter::handle_messages(req_non_stream, Arc::clone(&runtime)).await;
    assert_eq!(resp_non_stream.status, 200);

    let body_bytes = match resp_non_stream.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected bytes"),
    };
    let message_resp: ClaudeMessageResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(message_resp.stop_reason.as_deref(), Some("tool_use"));

    let tool_block = message_resp
        .content
        .iter()
        .find(|b| b.r#type == "tool_use")
        .expect("tool_use content block must be present");
    assert_eq!(tool_block.id.as_deref(), Some("toolu_nested_999"));
    assert_eq!(
        tool_block.name.as_deref(),
        Some("complex_pipeline_executor")
    );
    assert_eq!(
        tool_block.input.as_ref().unwrap(),
        &complex_nested_input,
        "Non-streaming nested tool input must match original JSON without any corruption"
    );

    // 2. Streaming SSE roundtrip verification
    let req_stream = ClaudeMessageRequest {
        model: "claude-nested-tools".to_owned(),
        system: None,
        messages: vec![
            ClaudeMessage::user("Execute complex pipeline query"),
            ClaudeMessage {
                role: "assistant".to_owned(),
                content: ClaudeContent::Blocks(vec![ClaudeContentBlock {
                    r#type: "tool_use".to_owned(),
                    text: None,
                    thinking: None,
                    id: Some("toolu_nested_999".to_owned()),
                    name: Some("complex_pipeline_executor".to_owned()),
                    input: Some(complex_nested_input.clone()),
                    tool_use_id: None,
                    content: None,
                    is_error: None,
                    source: None,
                }]),
            },
            ClaudeMessage::user("Follow-up question"),
        ],
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

    let mut all_events = Vec::new();
    while let Some(ev) = rx.recv().await {
        all_events.push(ev);
    }

    // Find content_block_delta with input_json_delta
    let delta_frame = all_events
        .iter()
        .find(|f| f.contains("\"input_json_delta\""))
        .expect("input_json_delta frame must be emitted in SSE stream");

    // Extract data line JSON
    let data_line = delta_frame
        .lines()
        .find(|l| l.starts_with("data: "))
        .expect("data: line in SSE frame");
    let json_text = data_line.strip_prefix("data: ").unwrap();
    let parsed_frame: serde_json::Value = serde_json::from_str(json_text).unwrap();

    let partial_json_str = parsed_frame["delta"]["partial_json"]
        .as_str()
        .expect("partial_json must be a JSON string");

    let reconstructed_input: serde_json::Value = serde_json::from_str(partial_json_str)
        .expect("partial_json string must parse back to valid JSON");

    assert_eq!(
        reconstructed_input, complex_nested_input,
        "Streaming input_json_delta must preserve nested input structure identically"
    );
}

// ============================================================================
// Challenge 3: Anthropic SSE Event Lifecycle Ordering & Keep-Alive
// ============================================================================

#[tokio::test]
async fn test_empirical_sse_lifecycle_strict_ordering_and_framing() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-lifecycle-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-lifecycle.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let req = ClaudeMessageRequest {
        model: "claude-lifecycle-model".to_owned(),
        system: Some(ClaudeSystem::Text("Test system".to_owned())),
        messages: vec![ClaudeMessage::user("Test reasoning and answer")],
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

    let mut raw_frames = Vec::new();
    while let Some(frame) = rx.recv().await {
        raw_frames.push(frame);
    }

    assert!(!raw_frames.is_empty(), "Must receive SSE frames");

    // Verify each frame ends with \n\n per SSE specification
    for (i, frame) in raw_frames.iter().enumerate() {
        assert!(
            frame.ends_with("\n\n"),
            "Frame {i} must end with standard SSE double newline: {:?}",
            frame
        );
    }

    // Frame 0: Keep-alive comment while queued
    assert_eq!(
        raw_frames[0], ": keep-alive\n\n",
        "Initial frame must be : keep-alive comment"
    );

    // Extract all structured event names and payloads in order
    let mut parsed_events: Vec<(String, serde_json::Value)> = Vec::new();
    for frame in &raw_frames[1..] {
        let mut event_type = None;
        let mut data_val = None;
        for line in frame.lines() {
            if let Some(ev) = line.strip_prefix("event: ") {
                event_type = Some(ev.trim().to_owned());
            } else if let Some(dt) = line.strip_prefix("data: ") {
                data_val = Some(
                    serde_json::from_str::<serde_json::Value>(dt.trim())
                        .expect("valid JSON in data:"),
                );
            }
        }
        if let (Some(ev), Some(dt)) = (event_type, data_val) {
            parsed_events.push((ev, dt));
        }
    }

    // Expected lifecycle sequence:
    // 0: message_start
    // 1: content_block_start (index: 0, thinking)
    // 2: content_block_delta (index: 0, thinking_delta)
    // 3: content_block_stop (index: 0)
    // 4: content_block_start (index: 1, text)
    // 5: content_block_delta (index: 1, text_delta)
    // 6: content_block_stop (index: 1)
    // 7: message_delta (stop_reason: end_turn)
    // 8: message_stop
    assert_eq!(
        parsed_events.len(),
        9,
        "Expected exactly 9 structured lifecycle events"
    );

    // 1. message_start
    assert_eq!(parsed_events[0].0, "message_start");
    assert_eq!(parsed_events[0].1["type"], "message_start");
    assert_eq!(parsed_events[0].1["message"]["role"], "assistant");
    assert_eq!(
        parsed_events[0].1["message"]["model"],
        "claude-lifecycle-model"
    );
    assert!(
        parsed_events[0].1["message"]["usage"]["input_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );

    // 2. content_block_start (thinking, index 0)
    assert_eq!(parsed_events[1].0, "content_block_start");
    assert_eq!(parsed_events[1].1["index"], 0);
    assert_eq!(parsed_events[1].1["content_block"]["type"], "thinking");

    // 3. content_block_delta (thinking_delta, index 0)
    assert_eq!(parsed_events[2].0, "content_block_delta");
    assert_eq!(parsed_events[2].1["index"], 0);
    assert_eq!(parsed_events[2].1["delta"]["type"], "thinking_delta");
    assert!(
        !parsed_events[2].1["delta"]["thinking"]
            .as_str()
            .unwrap()
            .is_empty()
    );

    // 4. content_block_stop (index 0)
    assert_eq!(parsed_events[3].0, "content_block_stop");
    assert_eq!(parsed_events[3].1["index"], 0);

    // 5. content_block_start (text, index 1)
    assert_eq!(parsed_events[4].0, "content_block_start");
    assert_eq!(parsed_events[4].1["index"], 1);
    assert_eq!(parsed_events[4].1["content_block"]["type"], "text");

    // 6. content_block_delta (text_delta, index 1)
    assert_eq!(parsed_events[5].0, "content_block_delta");
    assert_eq!(parsed_events[5].1["index"], 1);
    assert_eq!(parsed_events[5].1["delta"]["type"], "text_delta");
    assert!(
        !parsed_events[5].1["delta"]["text"]
            .as_str()
            .unwrap()
            .is_empty()
    );

    // 7. content_block_stop (index 1)
    assert_eq!(parsed_events[6].0, "content_block_stop");
    assert_eq!(parsed_events[6].1["index"], 1);

    // 8. message_delta
    assert_eq!(parsed_events[7].0, "message_delta");
    assert_eq!(parsed_events[7].1["delta"]["stop_reason"], "end_turn");
    assert!(
        parsed_events[7].1["usage"]["output_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );

    // 9. message_stop
    assert_eq!(parsed_events[8].0, "message_stop");
    assert_eq!(parsed_events[8].1["type"], "message_stop");
}

#[tokio::test]
async fn test_empirical_sse_keepalive_comments_in_queue() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-queue-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-queue.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Configure queue manager
    runtime.queue_mgr().register_model(
        "claude-queue-model",
        Some(AdmissionQueueConfig {
            max_depth: 10,
            keep_alive_interval: Duration::from_millis(50),
            wait_timeout: None,
            default_retry_after_secs: 5,
            estimated_pass_latency_ms: 20.0,
            estimated_passes_per_request: 8,
        }),
    );

    // Fill slot 1 with a placeholder request
    let first_admit = runtime.queue_mgr().admit_to_model(
        "claude-queue-model",
        "existing-req-1".to_owned(),
        false,
        None,
        None,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        None,
        None,
    );
    assert!(first_admit.is_ok());

    // Admit streaming request 2 behind request 1 (initial_pos == 2)
    let req_streaming = ClaudeMessageRequest {
        model: "claude-queue-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Queued streaming request")],
        max_tokens: 128,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(true),
        temperature: None,
        top_p: None,
    };

    let resp = ClaudeAdapter::handle_messages(req_streaming, runtime).await;
    assert_eq!(resp.status, 200);

    let mut rx = match resp.body {
        HttpResponseBody::Stream(rx) => rx,
        _ => panic!("expected stream body"),
    };

    let mut frames = Vec::new();
    while let Some(frame) = rx.recv().await {
        frames.push(frame);
    }

    let all_stream = frames.join("");
    // Both keep-alive comment and advancing queue position comment must be present
    assert!(
        all_stream.contains(": keep-alive\n\n"),
        "Must contain initial keep-alive comment"
    );
    assert!(
        all_stream.contains(": queue-position 2\n\n"),
        "Must contain queue-position comment when pos > 1"
    );
}

// ============================================================================
// Challenge 4: Error Handling (Overload 529, Context Overflow 400, Not Found 404)
// ============================================================================

#[tokio::test]
async fn test_empirical_queue_saturation_http_529_and_retry_after_header() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-capacity-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-capacity.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Queue depth cap = 2
    runtime.queue_mgr().register_model(
        "claude-capacity-model",
        Some(AdmissionQueueConfig {
            max_depth: 2,
            keep_alive_interval: Duration::from_millis(50),
            wait_timeout: None,
            default_retry_after_secs: 7,
            estimated_pass_latency_ms: 10.0,
            estimated_passes_per_request: 10,
        }),
    );

    // Admit 2 requests to fill queue
    assert!(
        runtime
            .queue_mgr()
            .admit_to_model(
                "claude-capacity-model",
                "fill-1".to_owned(),
                false,
                None,
                None,
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                None,
                None,
            )
            .is_ok()
    );

    assert!(
        runtime
            .queue_mgr()
            .admit_to_model(
                "claude-capacity-model",
                "fill-2".to_owned(),
                false,
                None,
                None,
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                None,
                None,
            )
            .is_ok()
    );

    // 1. Non-streaming request arriving while saturated -> HTTP 529
    let req_non_stream = ClaudeMessageRequest {
        model: "claude-capacity-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Overflow attempt non-stream")],
        max_tokens: 64,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };

    let resp_529 = ClaudeAdapter::handle_messages(req_non_stream, Arc::clone(&runtime)).await;
    assert_eq!(resp_529.status, 529, "Saturated queue must return HTTP 529");

    let retry_after_hdr = resp_529
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
        .map(|(_, v)| v.as_str());
    assert!(
        retry_after_hdr.is_some(),
        "Retry-After header must be present on 529 response"
    );
    assert!(
        retry_after_hdr.unwrap().parse::<u64>().is_ok(),
        "Retry-After must be numeric"
    );

    let body_bytes = match resp_529.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected bytes"),
    };
    let err_resp: ClaudeErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.r#type, "error");
    assert_eq!(err_resp.error.r#type, "overloaded_error");
    assert!(err_resp.error.message.contains("saturated"));

    // 2. Streaming request arriving while saturated -> HTTP 529 (NOT a streaming 200 with error event)
    let req_stream = ClaudeMessageRequest {
        model: "claude-capacity-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Overflow attempt streaming")],
        max_tokens: 64,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(true),
        temperature: None,
        top_p: None,
    };

    let resp_stream_529 = ClaudeAdapter::handle_messages(req_stream, runtime).await;
    assert_eq!(
        resp_stream_529.status, 529,
        "Saturated queue on streaming request must return HTTP 529"
    );
    let stream_retry = resp_stream_529
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"));
    assert!(
        stream_retry.is_some(),
        "Retry-After header must be present on streaming 529"
    );
}

#[tokio::test]
async fn test_empirical_context_overflow_rejection_without_silent_truncation() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-ctx-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-ctx.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Enforce strict 8 tokens context bound
    let mut profile = TuningProfile::conservative_default();
    profile.apply_operator_tuning(
        ComputePool::X100,
        0,
        6,
        PriorityClass::Throughput,
        8, // n_ctx = 8
        false,
        false,
    );
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "claude-ctx-model".to_owned(),
            profile,
        })
        .expect("apply tuning profile");

    // Prompt containing >20 tokens
    let excessive_prompt = "The quick brown fox jumps over the lazy dog and runs across the wide open fields in autumn.";

    // 1. Non-streaming context overflow
    let req_non_stream = ClaudeMessageRequest {
        model: "claude-ctx-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user(excessive_prompt)],
        max_tokens: 64,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };

    let resp_non_stream =
        ClaudeAdapter::handle_messages(req_non_stream, Arc::clone(&runtime)).await;
    assert_eq!(
        resp_non_stream.status, 400,
        "Context length overflow must return HTTP 400"
    );

    let body_bytes = match resp_non_stream.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected bytes"),
    };
    let err_resp: ClaudeErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.r#type, "error");
    assert_eq!(err_resp.error.r#type, "invalid_request_error");
    assert!(
        err_resp.error.message.contains("context window limit")
            || err_resp.error.message.contains("n_ctx"),
        "Error message must specify context window violation"
    );

    // 2. Streaming context overflow -> HTTP 400 (NOT a stream)
    let req_stream = ClaudeMessageRequest {
        model: "claude-ctx-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user(excessive_prompt)],
        max_tokens: 64,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(true),
        temperature: None,
        top_p: None,
    };

    let resp_stream = ClaudeAdapter::handle_messages(req_stream, runtime).await;
    assert_eq!(
        resp_stream.status, 400,
        "Streaming request with context overflow must return HTTP 400 immediately"
    );
}

#[tokio::test]
async fn test_empirical_missing_model_returns_404_not_found() {
    let runtime = build_test_runtime();

    let req = ClaudeMessageRequest {
        model: "non-existent-claude-model".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Hello missing model")],
        max_tokens: 128,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };

    let resp = ClaudeAdapter::handle_messages(req, runtime).await;
    assert_eq!(resp.status, 404, "Missing model must return HTTP 404");

    let body_bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        _ => panic!("expected bytes"),
    };
    let err_resp: ClaudeErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.error.r#type, "not_found_error");
    assert!(err_resp.error.message.contains("non-existent-claude-model"));
}

// ============================================================================
// Challenge 5: ProtocolServer Integration, Limits & Security Defenses
// ============================================================================

#[tokio::test]
async fn test_empirical_protocol_server_routes_and_security_bounds() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-server-test-model".to_owned(),
            weights_path: PathBuf::from("/models/claude-server.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    // 1. Valid POST /v1/messages
    let valid_payload = serde_json::json!({
        "model": "claude-server-test-model",
        "messages": [{"role": "user", "content": "Hello via ProtocolServer"}],
        "max_tokens": 100
    })
    .to_string();
    let req_v1 = HttpRequest::new("POST", "/v1/messages", valid_payload.clone().into_bytes());
    let resp_v1 = server.route_request(req_v1).await;
    assert_eq!(resp_v1.status, 200, "POST /v1/messages must succeed");

    // 2. Valid POST /messages (alias)
    let req_alias = HttpRequest::new("POST", "/messages", valid_payload.into_bytes());
    let resp_alias = server.route_request(req_alias).await;
    assert_eq!(resp_alias.status, 200, "POST /messages must succeed");

    // 3. Payload size exceeding 10MB limit -> HTTP 413
    let oversized_body = vec![b'x'; 10 * 1024 * 1024 + 1024]; // 10MB + 1KB
    let req_oversized = HttpRequest::new("POST", "/v1/messages", oversized_body);
    let resp_oversized = server.route_request(req_oversized).await;
    assert_eq!(
        resp_oversized.status, 413,
        "Oversized request body must return HTTP 413"
    );

    // 4. Malformed JSON body -> HTTP 400
    let malformed_req = HttpRequest::new(
        "POST",
        "/v1/messages",
        b"{\"messages\": [ incomplete json".to_vec(),
    );
    let resp_malformed = server.route_request(malformed_req).await;
    assert_eq!(
        resp_malformed.status, 400,
        "Malformed JSON request must return HTTP 400"
    );

    // 5. Wrong HTTP method: GET /v1/messages -> HTTP 404
    let req_get = HttpRequest::new("GET", "/v1/messages", Vec::new());
    let resp_get = server.route_request(req_get).await;
    assert_eq!(
        resp_get.status, 404,
        "GET /v1/messages must return HTTP 404"
    );

    // 6. Non-existent route -> HTTP 404
    let req_unknown = HttpRequest::new("POST", "/v1/unknown_route", Vec::new());
    let resp_unknown = server.route_request(req_unknown).await;
    assert_eq!(
        resp_unknown.status, 404,
        "Unknown route must return HTTP 404"
    );
}

#[tokio::test]
async fn test_challenge_claude_url_query_parameter_route_tolerance() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-param-test".to_owned(),
            weights_path: PathBuf::from("/models/claude-param.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    let body = serde_json::json!({
        "model": "claude-param-test",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    })
    .to_string();

    // 1. POST /v1/messages?beta=true on ProtocolServer
    let req_v1 = HttpRequest::new("POST", "/v1/messages?beta=true", body.clone().into_bytes());
    let resp_v1 = server.route_request(req_v1).await;
    assert_eq!(
        resp_v1.status, 200,
        "ProtocolServer should accept /v1/messages?beta=true"
    );

    // 2. POST /messages?version=2023-06-01 on ProtocolServer
    let req_alias = HttpRequest::new(
        "POST",
        "/messages?version=2023-06-01",
        body.clone().into_bytes(),
    );
    let resp_alias = server.route_request(req_alias).await;
    assert_eq!(
        resp_alias.status, 200,
        "ProtocolServer should accept /messages?version=2023-06-01"
    );

    // 3. POST /v1/messages?anthropic_version=2023-06-01 on ClaudeAdapter::handle_http directly
    let req_direct = HttpRequest::new(
        "POST",
        "/v1/messages?anthropic_version=2023-06-01",
        body.into_bytes(),
    );
    let resp_direct = ClaudeAdapter::handle_http(req_direct, runtime).await;
    assert_eq!(
        resp_direct.status, 200,
        "ClaudeAdapter::handle_http should accept /v1/messages?anthropic_version=2023-06-01"
    );
}

#[tokio::test]
async fn test_challenge_claude_client_early_disconnect_cleans_up_queue_permits() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "claude-permit-check".to_owned(),
            weights_path: PathBuf::from("/models/claude-permit-check.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Stress loop: 5 successive premature client disconnects on Claude streaming
    for i in 0..5 {
        let (mut client, server_stream) = tokio::io::duplex(4096);
        let srv = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));
        let server_handle = tokio::spawn(async move { srv.handle_connection(server_stream).await });

        let msg_req = serde_json::json!({
            "model": "claude-permit-check",
            "messages": [{"role": "user", "content": format!("Drop test {i}")}],
            "max_tokens": 100,
            "stream": true
        })
        .to_string();

        let wire_req = format!(
            "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
            msg_req.len(),
            msg_req
        );

        client.write_all(wire_req.as_bytes()).await.unwrap();
        client.flush().await.unwrap();

        let mut temp = [0u8; 64];
        let _ = client.read(&mut temp).await.unwrap();
        drop(client);

        let _ = server_handle.await;

        tokio::time::sleep(Duration::from_millis(20)).await;

        assert_eq!(
            runtime.queue_mgr().queue_depth("claude-permit-check"),
            0,
            "Queue depth must be 0 after client disconnect"
        );
    }

    let final_req = ClaudeMessageRequest {
        model: "claude-permit-check".to_owned(),
        system: None,
        messages: vec![ClaudeMessage::user("Can I still execute?")],
        max_tokens: 100,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };
    let resp = ClaudeAdapter::handle_messages(final_req, runtime).await;
    assert_eq!(
        resp.status, 200,
        "Subsequent request should succeed with 200"
    );
}
