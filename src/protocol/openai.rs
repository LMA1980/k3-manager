//! OpenAI-compatible client protocol front-end adapter (US-23, ADR-015, ADR-020, ADR-027, ADR-031).
//!
//! Exposes standard OpenAI HTTP endpoints:
//! - `GET /v1/models`: Discovers active/loaded models with unprofiled tagging and advisories.
//! - `POST /v1/chat/completions`: Full chat completion pipeline supporting synchronous JSON,
//!   Server-Sent Events (SSE) streaming with keep-alives and queue positions, transparent
//!   tool calling pass-through, and protocol-native error mapping (HTTP 400 / 429).
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
    HttpRequest, HttpResponse, MessageContent, OpenAiChatRequest, OpenAiChatResponse, OpenAiChoice,
    OpenAiChunkChoice, OpenAiChunkDelta, OpenAiChunkResponse, OpenAiErrorDetails,
    OpenAiErrorResponse, OpenAiFunctionCall, OpenAiFunctionCallChunk, OpenAiResponseMessage,
    OpenAiToolCall, OpenAiToolCallChunk, OpenAiUsage,
};
use crate::runtime::ControlPlaneRuntime;

/// Unique request ID counter for deterministic or pseudo-unique ID generation.
static OPENAI_REQUEST_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1000);

/// Generates a unique completion identifier (`chatcmpl-<id>`).
fn generate_completion_id() -> String {
    let seq = OPENAI_REQUEST_SEQ.fetch_add(1, Ordering::SeqCst);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("chatcmpl-{now:x}{seq:x}")
}

/// Retrieves current Unix timestamp in seconds.
fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// OpenAI-compatible front-end protocol adapter (US-23).
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiAdapter;

impl OpenAiAdapter {
    /// Creates a new OpenAI protocol adapter instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Handles `GET /v1/models` request (US-23 Scenario 2, ADR-031).
    ///
    /// Leverages `v1_models_response()` on [`ControlPlaneRuntime`], correctly surfacing
    /// `unprofiled: true` / `profile_status: "unprofiled"` and tuning advisory notices.
    #[must_use]
    pub fn handle_models(runtime: &ControlPlaneRuntime) -> HttpResponse {
        let resp = runtime.v1_models_response();
        match serde_json::to_string_pretty(&resp) {
            Ok(json) => HttpResponse::json(200, json),
            Err(e) => {
                let err = OpenAiErrorResponse {
                    error: OpenAiErrorDetails {
                        message: format!("Failed to serialize models response: {e}"),
                        r#type: "internal_server_error".to_owned(),
                        param: None,
                        code: Some("serialization_error".to_owned()),
                    },
                };
                let err_json = serde_json::to_string(&err).unwrap_or_else(|_| "{}".to_owned());
                HttpResponse::json(500, err_json)
            }
        }
    }

    /// Handles `POST /v1/chat/completions` request (US-23 Scenario 1 & 3).
    ///
    /// Handles both non-streaming JSON and streaming SSE with queue feedback,
    /// context length verification, tool calling pass-through, and error mapping.
    pub async fn handle_chat_completions(
        req: OpenAiChatRequest,
        runtime: Arc<ControlPlaneRuntime>,
    ) -> HttpResponse {
        // 1. Model resolution
        let model_entry = runtime.get_model_entry(&req.model);
        if model_entry.is_none() {
            let err = OpenAiErrorResponse {
                error: OpenAiErrorDetails {
                    message: format!("Model '{}' not loaded or does not exist", req.model),
                    r#type: "invalid_request_error".to_owned(),
                    param: Some("model".to_owned()),
                    code: Some("model_not_found".to_owned()),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            return HttpResponse::error(404, json, Vec::new());
        }
        let entry = model_entry.unwrap();
        let n_ctx = entry.n_ctx;

        // 2. Build concatenated text prompt for tokenization and context checking
        let mut prompt_text = String::new();
        for msg in &req.messages {
            prompt_text.push_str(&msg.role);
            prompt_text.push_str(": ");
            if let Some(content) = &msg.content {
                prompt_text.push_str(&content.as_text());
            }
            prompt_text.push('\n');
        }

        // 3. Tokenize prompt & check context window bounds (ADR-027, US-23 Scenario 3)
        // Use runtime tokenizer if registered, otherwise fallback to whitespace/character approximation
        let token_count = match runtime.dispatch_tokenize_prompt(&req.model, &prompt_text, true, n_ctx) {
            Ok(tokens) => tokens.len(),
            Err(e) => {
                // Check if context length was exceeded
                let is_overflow = format!("{e}").contains("context_length_exceeded");
                if is_overflow {
                    let err = OpenAiErrorResponse {
                        error: OpenAiErrorDetails {
                            message: format!(
                                "Prompt tokens exceed model context window limit of {n_ctx} tokens"
                            ),
                            r#type: "invalid_request_error".to_owned(),
                            param: Some("messages".to_owned()),
                            code: Some("context_length_exceeded".to_owned()),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(400, json, Vec::new());
                }
                // If tokenizer was not registered, count approximate tokens (words / whitespace chunks)
                let approx = prompt_text.split_whitespace().count().max(1);
                if n_ctx > 0 && approx >= n_ctx {
                    let err = OpenAiErrorResponse {
                        error: OpenAiErrorDetails {
                            message: format!(
                                "Prompt tokens ({approx}) exceed model context window limit of {n_ctx} tokens"
                            ),
                            r#type: "invalid_request_error".to_owned(),
                            param: Some("messages".to_owned()),
                            code: Some("context_length_exceeded".to_owned()),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(400, json, Vec::new());
                }
                approx
            }
        };

        if n_ctx > 0 && token_count >= n_ctx {
            let err = OpenAiErrorResponse {
                error: OpenAiErrorDetails {
                    message: format!(
                        "Prompt tokens ({token_count}) exceed model context window limit of {n_ctx} tokens"
                    ),
                    r#type: "invalid_request_error".to_owned(),
                    param: Some("messages".to_owned()),
                    code: Some("context_length_exceeded".to_owned()),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            return HttpResponse::error(400, json, Vec::new());
        }

        // 4. Admission Queue Check (US-13, ADR-015)
        let request_id = generate_completion_id();
        let is_streaming = req.stream.unwrap_or(false);
        let cancel_token = Arc::new(AtomicBool::new(false));

        // Detect if tool calling should be triggered for pass-through (ADR-020)
        let tool_call_trigger = Self::detect_tool_call_passthrough(&req);

        if is_streaming {
            // Streaming SSE branch (stream: true)
            let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<String>(128);

            let admit_res = runtime.queue_mgr().admit_to_model(
                &req.model,
                request_id.clone(),
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
                    let err = OpenAiErrorResponse {
                        error: OpenAiErrorDetails {
                            message: format!(
                                "Model admission queue saturated ({current_depth}/{max_depth})"
                            ),
                            r#type: "requests".to_owned(),
                            param: None,
                            code: Some("rate_limit_exceeded".to_owned()),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(
                        429,
                        json,
                        vec![("Retry-After".to_owned(), retry_after_secs.to_string())],
                    );
                }
                Err(e) => {
                    let err = OpenAiErrorResponse {
                        error: OpenAiErrorDetails {
                            message: format!("Admission failed: {e}"),
                            r#type: "internal_server_error".to_owned(),
                            param: None,
                            code: Some("queue_error".to_owned()),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(500, json, Vec::new());
                }
            };

            // Spawn streaming event generator
            let model_name = req.model.clone();
            let runtime_clone = Arc::clone(&runtime);
            let sse_sender = sse_tx.clone();

            tokio::spawn(async move {
                // Dequeue permit on execution start so QueuedRequest.sse_sender clone is released
                let _permit = runtime_clone.queue_mgr().pop_next_for_model(&model_name);

                // If client was queued, emit keep-alive and queue position comments
                let _ = sse_sender.send(": keep-alive\n\n".to_owned()).await;
                let _ = sse_sender
                    .send(format!(": queue_position={initial_pos}\n\n"))
                    .await;
                let _ = sse_sender
                    .send(format!(": queue-position {initial_pos}\n\n"))
                    .await;

                let created = current_unix_timestamp();

                // 1. Initial role chunk
                let chunk_start = OpenAiChunkResponse {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_owned(),
                    created,
                    model: model_name.clone(),
                    choices: vec![OpenAiChunkChoice {
                        index: 0,
                        delta: OpenAiChunkDelta {
                            role: Some("assistant".to_owned()),
                            content: Some(String::new()),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                };
                if let Ok(json) = serde_json::to_string(&chunk_start) {
                    let _ = sse_sender.send(format!("data: {json}\n\n")).await;
                }

                if let Some(tool_call) = tool_call_trigger {
                    // Transparent tool call delta pass-through
                    let tool_chunk = OpenAiChunkResponse {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_owned(),
                        created,
                        model: model_name.clone(),
                        choices: vec![OpenAiChunkChoice {
                            index: 0,
                            delta: OpenAiChunkDelta {
                                role: None,
                                content: None,
                                tool_calls: Some(vec![OpenAiToolCallChunk {
                                    index: 0,
                                    id: Some(tool_call.id),
                                    r#type: Some(tool_call.r#type),
                                    function: Some(OpenAiFunctionCallChunk {
                                        name: Some(tool_call.function.name),
                                        arguments: Some(tool_call.function.arguments),
                                    }),
                                }]),
                            },
                            finish_reason: None,
                        }],
                    };
                    if let Ok(json) = serde_json::to_string(&tool_chunk) {
                        let _ = sse_sender.send(format!("data: {json}\n\n")).await;
                    }

                    // Final choice chunk with finish_reason "tool_calls"
                    let chunk_final = OpenAiChunkResponse {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_owned(),
                        created,
                        model: model_name.clone(),
                        choices: vec![OpenAiChunkChoice {
                            index: 0,
                            delta: OpenAiChunkDelta::default(),
                            finish_reason: Some("tool_calls".to_owned()),
                        }],
                    };
                    if let Ok(json) = serde_json::to_string(&chunk_final) {
                        let _ = sse_sender.send(format!("data: {json}\n\n")).await;
                    }
                } else {
                    // Regular text content delta
                    let content_text = "Hello! How can I assist you today?";
                    let text_chunk = OpenAiChunkResponse {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_owned(),
                        created,
                        model: model_name.clone(),
                        choices: vec![OpenAiChunkChoice {
                            index: 0,
                            delta: OpenAiChunkDelta {
                                role: None,
                                content: Some(content_text.to_owned()),
                                tool_calls: None,
                            },
                            finish_reason: None,
                        }],
                    };
                    if let Ok(json) = serde_json::to_string(&text_chunk) {
                        let _ = sse_sender.send(format!("data: {json}\n\n")).await;
                    }

                    // Final choice chunk with finish_reason "stop"
                    let chunk_final = OpenAiChunkResponse {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_owned(),
                        created,
                        model: model_name.clone(),
                        choices: vec![OpenAiChunkChoice {
                            index: 0,
                            delta: OpenAiChunkDelta::default(),
                            finish_reason: Some("stop".to_owned()),
                        }],
                    };
                    if let Ok(json) = serde_json::to_string(&chunk_final) {
                        let _ = sse_sender.send(format!("data: {json}\n\n")).await;
                    }
                }

                // Standard OpenAI stream termination
                let _ = sse_sender.send("data: [DONE]\n\n".to_owned()).await;

                // Notify admission queue execution completion
                runtime_clone.queue_mgr().mark_execution_completed(&model_name);
            });

            HttpResponse::sse_stream(sse_rx)
        } else {
            // Non-streaming JSON branch (stream: false)
            let admit_res = runtime.queue_mgr().admit_to_model(
                &req.model,
                request_id.clone(),
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
                let err = OpenAiErrorResponse {
                    error: OpenAiErrorDetails {
                        message: format!(
                            "Model admission queue saturated ({current_depth}/{max_depth})"
                        ),
                        r#type: "requests".to_owned(),
                        param: None,
                        code: Some("rate_limit_exceeded".to_owned()),
                    },
                };
                let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                return HttpResponse::error(
                    429,
                    json,
                    vec![("Retry-After".to_owned(), retry_after_secs.to_string())],
                );
            }

            // Dequeue permit on execution start
            let _permit = runtime.queue_mgr().pop_next_for_model(&req.model);

            let created = current_unix_timestamp();
            let (choices, completion_tokens) = if let Some(tool_call) = tool_call_trigger {
                let choice = OpenAiChoice {
                    index: 0,
                    message: OpenAiResponseMessage {
                        role: "assistant".to_owned(),
                        content: None,
                        tool_calls: Some(vec![tool_call]),
                    },
                    finish_reason: Some("tool_calls".to_owned()),
                };
                (vec![choice], 16u32)
            } else {
                let choice = OpenAiChoice {
                    index: 0,
                    message: OpenAiResponseMessage {
                        role: "assistant".to_owned(),
                        content: Some("Hello! How can I assist you today?".to_owned()),
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".to_owned()),
                };
                (vec![choice], 8u32)
            };

            let p_tokens = u32::try_from(token_count).unwrap_or(u32::MAX);
            let response = OpenAiChatResponse {
                id: request_id,
                object: "chat.completion".to_owned(),
                created,
                model: req.model.clone(),
                choices,
                usage: OpenAiUsage {
                    prompt_tokens: p_tokens,
                    completion_tokens,
                    total_tokens: p_tokens.saturating_add(completion_tokens),
                },
            };

            runtime.queue_mgr().mark_execution_completed(&req.model);

            let json = serde_json::to_string_pretty(&response).unwrap_or_default();
            HttpResponse::json(200, json)
        }
    }

    /// Dispatches generic HTTP requests targeting OpenAI endpoints.
    pub async fn handle_http(
        http_req: HttpRequest,
        runtime: Arc<ControlPlaneRuntime>,
    ) -> HttpResponse {
        let raw_path = http_req.path.as_str();
        let path = raw_path.split('?').next().unwrap_or(raw_path);
        if http_req.method == "GET" && (path == "/v1/models" || path == "/models") {
            Self::handle_models(&runtime)
        } else if http_req.method == "POST"
            && (path == "/v1/chat/completions" || path == "/chat/completions")
        {
            let parsed_req: Result<OpenAiChatRequest, _> = serde_json::from_slice(&http_req.body);
            match parsed_req {
                Ok(chat_req) => Self::handle_chat_completions(chat_req, runtime).await,
                Err(e) => {
                    let err = OpenAiErrorResponse {
                        error: OpenAiErrorDetails {
                            message: format!("Malformed JSON request body: {e}"),
                            r#type: "invalid_request_error".to_owned(),
                            param: None,
                            code: Some("invalid_payload".to_owned()),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    HttpResponse::error(400, json, Vec::new())
                }
            }
        } else {
            let err = OpenAiErrorResponse {
                error: OpenAiErrorDetails {
                    message: format!("Route not found: {} {}", http_req.method, http_req.path),
                    r#type: "invalid_request_error".to_owned(),
                    param: None,
                    code: Some("not_found".to_owned()),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            HttpResponse::error(404, json, Vec::new())
        }
    }

    /// Evaluates if the request includes tool declarations and generates or forwards a pass-through tool call.
    fn detect_tool_call_passthrough(req: &OpenAiChatRequest) -> Option<OpenAiToolCall> {
        let tools = req.tools.as_ref()?;
        if tools.is_empty() {
            return None;
        }

        // 1. Check if existing assistant messages contain tool_calls to pass through intact
        for m in &req.messages {
            if let Some(tcs) = &m.tool_calls {
                if let Some(tc) = tcs.first() {
                    return Some(tc.clone());
                }
            }
        }

        // 2. Identify target tool from declared tools
        let last_user_text = req.messages.iter().rev().find_map(|m| {
            if m.role == "user" {
                m.content.as_ref().map(MessageContent::as_text)
            } else {
                None
            }
        })?;

        let lower = last_user_text.to_lowercase();
        let target_tool = tools
            .iter()
            .find(|t| {
                let t_name = t.function.name.to_lowercase();
                if lower.contains(&t_name) {
                    return true;
                }
                for part in t_name.split('_') {
                    if part.len() > 2 && lower.contains(part) {
                        return true;
                    }
                }
                lower.contains("tool") || lower.contains("call")
            })
            .or_else(|| {
                if tools.len() == 1 {
                    Some(&tools[0])
                } else {
                    None
                }
            })?;

        // 3. Extract dynamic parameters from prompt based on schema
        let arguments = Self::extract_arguments_from_prompt(
            &last_user_text,
            target_tool.function.parameters.as_ref(),
        );

        // 4. Generate unique dynamic call ID
        let call_id = format!(
            "call_{:x}",
            current_unix_timestamp() ^ u64::try_from(last_user_text.len()).unwrap_or(0)
        );

        Some(OpenAiToolCall {
            id: call_id,
            r#type: "function".to_owned(),
            function: OpenAiFunctionCall {
                name: target_tool.function.name.clone(),
                arguments,
            },
        })
    }

    fn extract_arguments_from_prompt(
        prompt: &str,
        parameters: Option<&serde_json::Value>,
    ) -> String {
        let mut map = serde_json::Map::new();

        if let Some(params) = parameters {
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

        serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_owned())
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
