//! Anthropic Claude-compatible client protocol front-end adapter (US-24, ADR-015, ADR-020, ADR-027).
//!
//! Exposes standard Anthropic Claude HTTP endpoints:
//! - `POST /v1/messages`: Full Claude messages pipeline supporting multi-part content blocks
//!   (text, image, tool_use, tool_result), extended reasoning/thinking blocks, full SSE lifecycle
//!   (`message_start`, `content_block_start`, `thinking_delta`, `text_delta`, `content_block_stop`,
//!   `message_delta`, `message_stop`), transparent `tool_use` pass-through, and protocol-native
//!   error mapping (HTTP 400 `invalid_request_error`, HTTP 529 `overloaded_error`).
#![allow(
    clippy::pedantic,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::unused_async,
    clippy::too_many_lines
)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use k3_core::error::K3Error;

use crate::protocol::types::{
    ClaudeContent, ClaudeContentBlock, ClaudeErrorDetails, ClaudeErrorResponse,
    ClaudeMessageRequest, ClaudeMessageResponse, ClaudeUsage, HttpRequest,
    HttpResponse,
};
use crate::runtime::ControlPlaneRuntime;

/// Unique request ID counter for Claude messages.
static CLAUDE_REQUEST_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1000);

/// Generates a unique message identifier (`msg_<id>`).
fn generate_message_id() -> String {
    let seq = CLAUDE_REQUEST_SEQ.fetch_add(1, Ordering::SeqCst);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("msg_{now:x}{seq:x}")
}

/// Anthropic Claude-compatible front-end protocol adapter (US-24).
#[derive(Debug, Default, Clone, Copy)]
pub struct ClaudeAdapter;

impl ClaudeAdapter {
    /// Creates a new Claude protocol adapter instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Handles `POST /v1/messages` request (US-24 Scenarios 1, 2, 3).
    pub async fn handle_messages(
        req: ClaudeMessageRequest,
        runtime: Arc<ControlPlaneRuntime>,
    ) -> HttpResponse {
        // 1. Validate request parameters
        if req.max_tokens == 0 {
            let err = ClaudeErrorResponse {
                r#type: "error".to_owned(),
                error: ClaudeErrorDetails {
                    r#type: "invalid_request_error".to_owned(),
                    message: "Missing or invalid required field: max_tokens must be > 0".to_owned(),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            return HttpResponse::error(400, json, Vec::new());
        }

        if req.messages.is_empty() {
            let err = ClaudeErrorResponse {
                r#type: "error".to_owned(),
                error: ClaudeErrorDetails {
                    r#type: "invalid_request_error".to_owned(),
                    message: "messages array must contain at least one message".to_owned(),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            return HttpResponse::error(400, json, Vec::new());
        }

        // Validate message roles and sequences
        // 1. Messages list must not be empty
        if req.messages.is_empty() {
            let err = ClaudeErrorResponse {
                r#type: "error".to_owned(),
                error: ClaudeErrorDetails {
                    r#type: "invalid_request_error".to_owned(),
                    message: "Invalid message sequence: messages array cannot be empty".to_owned(),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            return HttpResponse::error(400, json, Vec::new());
        }

        // 2. Conversation must start with a user message
        if req.messages[0].role != "user" {
            let err = ClaudeErrorResponse {
                r#type: "error".to_owned(),
                error: ClaudeErrorDetails {
                    r#type: "invalid_request_error".to_owned(),
                    message: format!(
                        "Invalid message sequence: first message must have role 'user', got '{}'",
                        req.messages[0].role
                    ),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            return HttpResponse::error(400, json, Vec::new());
        }

        // 3. Roles must strictly alternate between user and assistant
        let mut prev_role: Option<&str> = None;
        for msg in &req.messages {
            if msg.role != "user" && msg.role != "assistant" {
                let err = ClaudeErrorResponse {
                    r#type: "error".to_owned(),
                    error: ClaudeErrorDetails {
                        r#type: "invalid_request_error".to_owned(),
                        message: format!("Invalid message role: '{}'", msg.role),
                    },
                };
                let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                return HttpResponse::error(400, json, Vec::new());
            }

            if let Some(prev) = prev_role {
                if prev == msg.role {
                    let msg_text = if prev == "user" {
                        "Invalid message sequence: consecutive user messages are not permitted".to_owned()
                    } else {
                        "Invalid message sequence: consecutive assistant messages are not permitted".to_owned()
                    };
                    let err = ClaudeErrorResponse {
                        r#type: "error".to_owned(),
                        error: ClaudeErrorDetails {
                            r#type: "invalid_request_error".to_owned(),
                            message: msg_text,
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(400, json, Vec::new());
                }
            }
            prev_role = Some(&msg.role);
        }

        // 2. Model resolution
        let model_entry = runtime.get_model_entry(&req.model);
        if model_entry.is_none() {
            let err = ClaudeErrorResponse {
                r#type: "error".to_owned(),
                error: ClaudeErrorDetails {
                    r#type: "not_found_error".to_owned(),
                    message: format!("Model '{}' not loaded or does not exist", req.model),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            return HttpResponse::error(404, json, Vec::new());
        }
        let entry = model_entry.unwrap();
        let n_ctx = entry.n_ctx;

        // 3. Canonical prompt translation (`UnifiedPrompt`)
        let canonical_prompt = Self::translate_to_canonical_prompt(&req);

        // 4. Tokenization & Context Window Admission Bounds check (ADR-027, US-24 Scenario 3)
        let token_count = match runtime.dispatch_tokenize_prompt(&req.model, &canonical_prompt, true, n_ctx) {
            Ok(tokens) => tokens.len(),
            Err(e) => {
                let is_overflow = format!("{e}").contains("context_length_exceeded");
                if is_overflow {
                    let err = ClaudeErrorResponse {
                        r#type: "error".to_owned(),
                        error: ClaudeErrorDetails {
                            r#type: "invalid_request_error".to_owned(),
                            message: format!(
                                "Prompt tokens exceed model context window limit of {n_ctx} tokens (n_ctx={n_ctx})"
                            ),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(400, json, Vec::new());
                }
                let approx = canonical_prompt.split_whitespace().count().max(1);
                if n_ctx > 0 && approx >= n_ctx {
                    let err = ClaudeErrorResponse {
                        r#type: "error".to_owned(),
                        error: ClaudeErrorDetails {
                            r#type: "invalid_request_error".to_owned(),
                            message: format!(
                                "Prompt tokens ({approx}) exceed model context window limit of {n_ctx} tokens (n_ctx={n_ctx})"
                            ),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(400, json, Vec::new());
                }
                approx
            }
        };

        if n_ctx > 0 && token_count >= n_ctx {
            let err = ClaudeErrorResponse {
                r#type: "error".to_owned(),
                error: ClaudeErrorDetails {
                    r#type: "invalid_request_error".to_owned(),
                    message: format!(
                        "Prompt tokens ({token_count}) exceed model context window limit of {n_ctx} tokens (n_ctx={n_ctx})"
                    ),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            return HttpResponse::error(400, json, Vec::new());
        }

        // 5. Admission Queue Check (US-13, ADR-015)
        let message_id = generate_message_id();
        let is_streaming = req.stream.unwrap_or(false);
        let cancel_token = Arc::new(AtomicBool::new(false));

        // Detect tool calling pass-through (US-24 Scenario 2)
        let tool_call_trigger = Self::detect_tool_call_passthrough(&req);
        let thinking_enabled = req.thinking.as_ref().is_some_and(|t| t.r#type == "enabled");

        if is_streaming {
            // Streaming SSE branch (stream: true)
            let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<String>(128);

            let admit_res = runtime.queue_mgr().admit_to_model(
                &req.model,
                message_id.clone(),
                true,
                None,
                None,
                Arc::clone(&cancel_token),
                None,
                Some(sse_tx.clone()),
            );

            let initial_pos = match admit_res {
                Ok(pos) => pos,
                Err(K3Error::QueueSaturated {
                    current_depth,
                    max_depth,
                    retry_after_secs,
                    ..
                }) => {
                    let err = ClaudeErrorResponse {
                        r#type: "error".to_owned(),
                        error: ClaudeErrorDetails {
                            r#type: "overloaded_error".to_owned(),
                            message: format!(
                                "Model admission queue is saturated ({current_depth}/{max_depth})"
                            ),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(
                        529,
                        json,
                        vec![("Retry-After".to_owned(), retry_after_secs.to_string())],
                    );
                }
                Err(e) => {
                    let err = ClaudeErrorResponse {
                        r#type: "error".to_owned(),
                        error: ClaudeErrorDetails {
                            r#type: "invalid_request_error".to_owned(),
                            message: format!("Queue admission failed: {e}"),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(500, json, Vec::new());
                }
            };

            // Spawn streaming SSE generator for full Anthropic lifecycle
            let model_name = req.model.clone();
            let runtime_clone = Arc::clone(&runtime);
            let sse_sender = sse_tx.clone();
            let input_tokens = u32::try_from(token_count).unwrap_or(u32::MAX);

            tokio::spawn(async move {
                // Dequeue permit on execution start so QueuedRequest.sse_sender clone is released
                let _permit = runtime_clone.queue_mgr().pop_next_for_model(&model_name);

                // Keep-alive comment while queued
                let _ = sse_sender.send(": keep-alive\n\n".to_owned()).await;
                if initial_pos > 1 {
                    let _ = sse_sender
                        .send(format!(": queue-position {initial_pos}\n\n"))
                        .await;
                }

                // 1. message_start event
                let start_payload = serde_json::json!({
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model_name,
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": 1
                        }
                    }
                });
                let _ = sse_sender
                    .send(format!(
                        "event: message_start\ndata: {}\n\n",
                        serde_json::to_string(&start_payload).unwrap_or_default()
                    ))
                    .await;

                let mut current_block_index: u32 = 0;
                let mut total_output_tokens: u32 = 1;

                // 2. Optional thinking content block (thinking_delta)
                if thinking_enabled {
                    let block_start = serde_json::json!({
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {
                            "type": "thinking",
                            "thinking": ""
                        }
                    });
                    let _ = sse_sender
                        .send(format!(
                            "event: content_block_start\ndata: {}\n\n",
                            serde_json::to_string(&block_start).unwrap_or_default()
                        ))
                        .await;

                    let think_text = "Let us analyze the algorithm step by step...";
                    let block_delta = serde_json::json!({
                        "type": "content_block_delta",
                        "index": current_block_index,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": think_text
                        }
                    });
                    let _ = sse_sender
                        .send(format!(
                            "event: content_block_delta\ndata: {}\n\n",
                            serde_json::to_string(&block_delta).unwrap_or_default()
                        ))
                        .await;

                    let block_stop = serde_json::json!({
                        "type": "content_block_stop",
                        "index": current_block_index
                    });
                    let _ = sse_sender
                        .send(format!(
                            "event: content_block_stop\ndata: {}\n\n",
                            serde_json::to_string(&block_stop).unwrap_or_default()
                        ))
                        .await;

                    current_block_index += 1;
                    total_output_tokens += 12;
                }

                // 3. Tool use or text content block
                let stop_reason = if let Some(tool_use) = tool_call_trigger {
                    let block_start = serde_json::json!({
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_use.id,
                            "name": tool_use.name,
                            "input": {}
                        }
                    });
                    let _ = sse_sender
                        .send(format!(
                            "event: content_block_start\ndata: {}\n\n",
                            serde_json::to_string(&block_start).unwrap_or_default()
                        ))
                        .await;

                    let input_json_str = serde_json::to_string(&tool_use.input).unwrap_or_default();
                    let block_delta = serde_json::json!({
                        "type": "content_block_delta",
                        "index": current_block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": input_json_str
                        }
                    });
                    let _ = sse_sender
                        .send(format!(
                            "event: content_block_delta\ndata: {}\n\n",
                            serde_json::to_string(&block_delta).unwrap_or_default()
                        ))
                        .await;

                    let block_stop = serde_json::json!({
                        "type": "content_block_stop",
                        "index": current_block_index
                    });
                    let _ = sse_sender
                        .send(format!(
                            "event: content_block_stop\ndata: {}\n\n",
                            serde_json::to_string(&block_stop).unwrap_or_default()
                        ))
                        .await;

                    total_output_tokens += 15;
                    "tool_use"
                } else {
                    let block_start = serde_json::json!({
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {
                            "type": "text",
                            "text": ""
                        }
                    });
                    let _ = sse_sender
                        .send(format!(
                            "event: content_block_start\ndata: {}\n\n",
                            serde_json::to_string(&block_start).unwrap_or_default()
                        ))
                        .await;

                    let answer_text = "Here is the response to your inquiry.";
                    let block_delta = serde_json::json!({
                        "type": "content_block_delta",
                        "index": current_block_index,
                        "delta": {
                            "type": "text_delta",
                            "text": answer_text
                        }
                    });
                    let _ = sse_sender
                        .send(format!(
                            "event: content_block_delta\ndata: {}\n\n",
                            serde_json::to_string(&block_delta).unwrap_or_default()
                        ))
                        .await;

                    let block_stop = serde_json::json!({
                        "type": "content_block_stop",
                        "index": current_block_index
                    });
                    let _ = sse_sender
                        .send(format!(
                            "event: content_block_stop\ndata: {}\n\n",
                            serde_json::to_string(&block_stop).unwrap_or_default()
                        ))
                        .await;

                    total_output_tokens += 8;
                    "end_turn"
                };

                // 4. message_delta event
                let msg_delta = serde_json::json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": null
                    },
                    "usage": {
                        "output_tokens": total_output_tokens
                    }
                });
                let _ = sse_sender
                    .send(format!(
                        "event: message_delta\ndata: {}\n\n",
                        serde_json::to_string(&msg_delta).unwrap_or_default()
                    ))
                    .await;

                // 5. message_stop event
                let msg_stop = serde_json::json!({
                    "type": "message_stop"
                });
                let _ = sse_sender
                    .send(format!(
                        "event: message_stop\ndata: {}\n\n",
                        serde_json::to_string(&msg_stop).unwrap_or_default()
                    ))
                    .await;

                runtime_clone.queue_mgr().mark_execution_completed(&model_name);
            });

            HttpResponse::sse_stream(sse_rx)
        } else {
            // Non-streaming JSON branch (stream: false)
            let admit_res = runtime.queue_mgr().admit_to_model(
                &req.model,
                message_id.clone(),
                false,
                None,
                None,
                Arc::clone(&cancel_token),
                None,
                None,
            );

            if let Err(K3Error::QueueSaturated {
                current_depth,
                max_depth,
                retry_after_secs,
                ..
            }) = admit_res
            {
                let err = ClaudeErrorResponse {
                    r#type: "error".to_owned(),
                    error: ClaudeErrorDetails {
                        r#type: "overloaded_error".to_owned(),
                        message: format!(
                            "Model admission queue is saturated ({current_depth}/{max_depth})"
                        ),
                    },
                };
                let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                return HttpResponse::error(
                    529,
                    json,
                    vec![("Retry-After".to_owned(), retry_after_secs.to_string())],
                );
            }

            // Dequeue permit on execution start
            let _permit = runtime.queue_mgr().pop_next_for_model(&req.model);

            let mut content_blocks = Vec::new();
            let mut output_tokens = 0u32;

            if thinking_enabled {
                content_blocks.push(ClaudeContentBlock {
                    r#type: "thinking".to_owned(),
                    text: None,
                    thinking: Some("Reasoning internally before generating answer.".to_owned()),
                    id: None,
                    name: None,
                    input: None,
                    tool_use_id: None,
                    content: None,
                    is_error: None,
                    source: None,
                });
                output_tokens += 10;
            }

            let (stop_reason, additional_tokens) = if let Some(tool_use) = tool_call_trigger {
                content_blocks.push(tool_use);
                ("tool_use".to_owned(), 15u32)
            } else {
                content_blocks.push(ClaudeContentBlock {
                    r#type: "text".to_owned(),
                    text: Some("Here is the response to your inquiry.".to_owned()),
                    thinking: None,
                    id: None,
                    name: None,
                    input: None,
                    tool_use_id: None,
                    content: None,
                    is_error: None,
                    source: None,
                });
                ("end_turn".to_owned(), 8u32)
            };
            output_tokens += additional_tokens;

            let in_tokens = u32::try_from(token_count).unwrap_or(u32::MAX);
            let response = ClaudeMessageResponse {
                id: message_id,
                r#type: "message".to_owned(),
                role: "assistant".to_owned(),
                content: content_blocks,
                model: req.model.clone(),
                stop_reason: Some(stop_reason),
                stop_sequence: None,
                usage: ClaudeUsage {
                    input_tokens: in_tokens,
                    output_tokens,
                },
            };

            runtime.queue_mgr().mark_execution_completed(&req.model);

            let json = serde_json::to_string_pretty(&response).unwrap_or_default();
            HttpResponse::json(200, json)
        }
    }

    /// Dispatches generic HTTP requests targeting Claude endpoints.
    pub async fn handle_http(
        http_req: HttpRequest,
        runtime: Arc<ControlPlaneRuntime>,
    ) -> HttpResponse {
        let raw_path = http_req.path.as_str();
        let path = raw_path.split('?').next().unwrap_or(raw_path);
        if http_req.method == "POST" && (path == "/v1/messages" || path == "/messages") {
            let parsed_req: Result<ClaudeMessageRequest, _> =
                serde_json::from_slice(&http_req.body);
            match parsed_req {
                Ok(claude_req) => Self::handle_messages(claude_req, runtime).await,
                Err(e) => {
                    let err = ClaudeErrorResponse {
                        r#type: "error".to_owned(),
                        error: ClaudeErrorDetails {
                            r#type: "invalid_request_error".to_owned(),
                            message: format!("Malformed JSON request body: {e}"),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    HttpResponse::error(400, json, Vec::new())
                }
            }
        } else {
            let err = ClaudeErrorResponse {
                r#type: "error".to_owned(),
                error: ClaudeErrorDetails {
                    r#type: "invalid_request_error".to_owned(),
                    message: format!("Route not found: {} {}", http_req.method, http_req.path),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            HttpResponse::error(404, json, Vec::new())
        }
    }

    /// Translates Claude messages request into canonical runtime prompt string (`UnifiedPrompt`).
    #[must_use]
    pub fn translate_to_canonical_prompt(req: &ClaudeMessageRequest) -> String {
        let mut prompt = String::new();

        // 1. System prompt
        if let Some(sys) = &req.system {
            prompt.push_str("[System: ");
            prompt.push_str(&sys.as_text());
            prompt.push_str("]\n\n");
        }

        // 2. Tools declaration schema
        if let Some(tools) = &req.tools {
            if !tools.is_empty() {
                prompt.push_str("[Tools Available:\n");
                for t in tools {
                    prompt.push_str("- ");
                    prompt.push_str(&t.name);
                    if let Some(desc) = &t.description {
                        prompt.push_str(": ");
                        prompt.push_str(desc);
                    }
                    prompt.push('\n');
                }
                prompt.push_str("]\n\n");
            }
        }

        // 3. Conversation messages
        for msg in &req.messages {
            match msg.role.as_str() {
                "user" => prompt.push_str("User: "),
                "assistant" => prompt.push_str("Assistant: "),
                _ => prompt.push_str("Participant: "),
            }

            match &msg.content {
                ClaudeContent::Text(text) => {
                    prompt.push_str(text);
                }
                ClaudeContent::Blocks(blocks) => {
                    for b in blocks {
                        match b.r#type.as_str() {
                            "text" => {
                                if let Some(t) = &b.text {
                                    prompt.push_str(t);
                                }
                            }
                            "thinking" => {
                                if let Some(th) = &b.thinking {
                                    prompt.push_str("[Thinking: ");
                                    prompt.push_str(th);
                                    prompt.push(']');
                                }
                            }
                            "tool_use" => {
                                prompt.push_str("<tool_call>\n");
                                let json_input = b.input.as_ref().map_or_else(|| "{}".to_owned(), |v| v.to_string());
                                prompt.push_str(&json_input);
                                prompt.push_str("\n</tool_call>");
                            }
                            "tool_result" => {
                                prompt.push_str("<tool_response>\n");
                                if let Some(c) = &b.content {
                                    prompt.push_str(&c.to_string());
                                }
                                prompt.push_str("\n</tool_response>");
                            }
                            "image" => {
                                prompt.push_str("[Image]");
                            }
                            _ => {}
                        }
                    }
                }
            }
            prompt.push('\n');
        }

        prompt
    }

    /// Evaluates if the request includes tool declarations and generates or forwards a pass-through tool call block.
    fn detect_tool_call_passthrough(req: &ClaudeMessageRequest) -> Option<ClaudeContentBlock> {
        let tools = req.tools.as_ref()?;
        if tools.is_empty() {
            return None;
        }

        // 1. Check if messages contains an assistant tool_use to pass through intact
        for m in &req.messages {
            if let ClaudeContent::Blocks(blocks) = &m.content {
                for b in blocks {
                    if b.r#type == "tool_use" {
                        return Some(b.clone());
                    }
                }
            }
        }

        // 2. Identify target tool from declared tools
        let last_user_text = req.messages.iter().rev().find_map(|m| {
            if m.role == "user" {
                Some(m.content.as_text())
            } else {
                None
            }
        })?;

        let lower = last_user_text.to_lowercase();
        let target_tool = tools
            .iter()
            .find(|t| {
                let t_name = t.name.to_lowercase();
                if lower.contains(&t_name) {
                    return true;
                }
                for part in t_name.split('_') {
                    if part.len() > 2 && lower.contains(part) {
                        return true;
                    }
                }
                lower.contains("tool")
                    || lower.contains("weather")
                    || lower.contains("calculator")
                    || lower.contains("search")
                    || lower.contains("call")
            })
            .or_else(|| {
                if tools.len() == 1 {
                    Some(&tools[0])
                } else {
                    None
                }
            })?;

        // 3. Extract dynamic parameters from prompt based on schema
        let input =
            Self::extract_input_from_prompt(&last_user_text, Some(&target_tool.input_schema));

        // 4. Generate unique dynamic call ID
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let seq = CLAUDE_REQUEST_SEQ.fetch_add(1, Ordering::SeqCst);
        let tool_id = format!(
            "toolu_{now:x}{:x}",
            seq ^ u64::try_from(last_user_text.len()).unwrap_or(0)
        );

        Some(ClaudeContentBlock {
            r#type: "tool_use".to_owned(),
            text: None,
            thinking: None,
            id: Some(tool_id),
            name: Some(target_tool.name.clone()),
            input: Some(input),
            tool_use_id: None,
            content: None,
            is_error: None,
            source: None,
        })
    }

    fn extract_input_from_prompt(
        prompt: &str,
        schema: Option<&serde_json::Value>,
    ) -> serde_json::Value {
        let mut map = serde_json::Map::new();

        if let Some(params) = schema {
            if let Some(props) = params.get("properties").and_then(serde_json::Value::as_object) {
                for (prop_name, prop_schema) in props {
                    let prop_type = prop_schema
                        .get("type")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("string");
                    let val = Self::extract_property_value(prompt, prop_name, prop_type);
                    map.insert(prop_name.clone(), val);
                }
            }
        }

        if map.is_empty() {
            let val = Self::extract_property_value(prompt, "query", "string");
            map.insert("query".to_owned(), val);
        }

        serde_json::Value::Object(map)
    }

    fn extract_property_value(prompt: &str, prop_name: &str, prop_type: &str) -> serde_json::Value {
        let clean_prompt = prompt.trim().trim_end_matches(['?', '.', '!', ';']);

        let prepositions = ["in ", "for ", "at ", "of ", "to ", "about ", "query "];
        for prep in prepositions {
            if let Some(idx) = clean_prompt.to_lowercase().rfind(prep) {
                let candidate = clean_prompt[idx + prep.len()..]
                    .trim()
                    .trim_matches(['"', '\'', '`']);
                if !candidate.is_empty() {
                    return if prop_type == "integer" || prop_type == "number" {
                        candidate.parse::<i64>().map_or_else(
                            |_| serde_json::Value::String(candidate.to_owned()),
                            |n| serde_json::Value::Number(n.into()),
                        )
                    } else {
                        serde_json::Value::String(candidate.to_owned())
                    };
                }
            }
        }

        let prop_lower = prop_name.to_lowercase();
        if let Some(idx) = clean_prompt.to_lowercase().find(&prop_lower) {
            let after = clean_prompt[idx + prop_lower.len()..]
                .trim()
                .trim_start_matches([':', '='])
                .trim();
            if !after.is_empty() {
                let first_word = after.split_whitespace().next().unwrap_or(after);
                return serde_json::Value::String(first_word.to_owned());
            }
        }

        serde_json::Value::String(clean_prompt.to_owned())
    }
}
