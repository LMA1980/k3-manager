//! Protocol front-end common request and response types (US-23, US-24, ADR-015, ADR-020).
//!
//! Provides canonical representations for OpenAI chat completions, Anthropic Claude messages,
//! Server-Sent Events (SSE), tool calling pass-through payloads, and internal HTTP frames.
#![allow(
    clippy::pedantic,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

use serde::{Deserialize, Serialize};

/// Maximum allowed HTTP request body size (10 MB) to prevent resource exhaustion DoS.
pub const DEFAULT_MAX_REQUEST_BODY_BYTES: usize = 10 * 1024 * 1024;

/// Default socket read timeout (30s) to protect against Slowloris attacks.
pub const DEFAULT_SOCKET_TIMEOUT_SECS: u64 = 30;

// ============================================================================
// Internal HTTP Types
// ============================================================================

/// Lightweight HTTP request frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    /// HTTP method (e.g. "GET", "POST").
    pub method: String,
    /// Request URI path (e.g. "/v1/models", "/v1/chat/completions", "/v1/messages").
    pub path: String,
    /// HTTP headers as (name, value) pairs.
    pub headers: Vec<(String, String)>,
    /// Raw request body bytes.
    pub body: Vec<u8>,
}

impl HttpRequest {
    /// Creates a new HTTP request frame.
    #[must_use]
    pub fn new(method: impl Into<String>, path: impl Into<String>, body: Vec<u8>) -> Self {
        Self {
            method: method.into(),
            path: path.into(),
            headers: Vec::new(),
            body,
        }
    }

    /// Adds a header to the request.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Retrieves the value of a header by case-insensitive name.
    #[must_use]
    pub fn get_header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// HTTP response body representation: buffered bytes or asynchronous SSE stream channel.
#[derive(Debug)]
pub enum HttpResponseBody {
    /// Buffered byte payload (e.g. JSON response).
    Bytes(Vec<u8>),
    /// Streaming SSE frames channel (receiver).
    Stream(tokio::sync::mpsc::Receiver<String>),
}

/// Lightweight HTTP response frame.
#[derive(Debug)]
pub struct HttpResponse {
    /// HTTP status code (e.g. 200, 400, 404, 413, 429, 529).
    pub status: u16,
    /// HTTP response headers.
    pub headers: Vec<(String, String)>,
    /// Response body.
    pub body: HttpResponseBody,
}

impl HttpResponse {
    /// Creates a 200 OK JSON response.
    #[must_use]
    pub fn json(status: u16, json_body: String) -> Self {
        Self {
            status,
            headers: vec![
                ("Content-Type".to_owned(), "application/json; charset=utf-8".to_owned()),
                ("Content-Length".to_owned(), json_body.len().to_string()),
            ],
            body: HttpResponseBody::Bytes(json_body.into_bytes()),
        }
    }

    /// Creates an SSE stream response.
    #[must_use]
    pub fn sse_stream(rx: tokio::sync::mpsc::Receiver<String>) -> Self {
        Self {
            status: 200,
            headers: vec![
                ("Content-Type".to_owned(), "text/event-stream; charset=utf-8".to_owned()),
                ("Cache-Control".to_owned(), "no-cache".to_owned()),
                ("Connection".to_owned(), "keep-alive".to_owned()),
            ],
            body: HttpResponseBody::Stream(rx),
        }
    }

    /// Creates an error response with custom headers.
    #[must_use]
    pub fn error(status: u16, json_body: String, extra_headers: Vec<(String, String)>) -> Self {
        let mut headers = vec![
            ("Content-Type".to_owned(), "application/json; charset=utf-8".to_owned()),
            ("Content-Length".to_owned(), json_body.len().to_string()),
        ];
        headers.extend(extra_headers);
        Self {
            status,
            headers,
            body: HttpResponseBody::Bytes(json_body.into_bytes()),
        }
    }

    /// Retrieves the status code.
    #[must_use]
    pub const fn status(&self) -> u16 {
        self.status
    }
}

// ============================================================================
// OpenAI Protocol Types (US-23)
// ============================================================================

/// Flexible message content representation supporting either plain text or multi-part structures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain string content.
    Text(String),
    /// Multi-part or structured content blocks.
    Parts(Vec<serde_json::Value>),
}

impl MessageContent {
    /// Extracts plain text representation.
    #[must_use]
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => {
                let mut buf = String::new();
                for p in parts {
                    if let Some(t) = p.get("text").and_then(serde_json::Value::as_str) {
                        buf.push_str(t);
                    }
                }
                buf
            }
        }
    }
}

/// Function definition in OpenAI tool schema (ADR-020).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiFunction {
    /// Function name.
    pub name: String,
    /// Function description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON schema parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Tool declaration in OpenAI chat completion request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiTool {
    /// Tool type, currently always "function".
    pub r#type: String,
    /// Function metadata.
    pub function: OpenAiFunction,
}

/// Model tool invocation in OpenAI choices or messages (ADR-020).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiToolCall {
    /// Unique tool invocation ID.
    pub id: String,
    /// Tool type, always "function".
    pub r#type: String,
    /// Function call details.
    pub function: OpenAiFunctionCall,
}

/// Specific function call details within an [`OpenAiToolCall`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiFunctionCall {
    /// Function name to invoke.
    pub name: String,
    /// Serialized JSON arguments string.
    pub arguments: String,
}

/// Individual message in an OpenAI chat completion request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiMessage {
    /// Role of the author: "system", "user", "assistant", "tool".
    pub role: String,
    /// Text or structured content of the message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// Optional participant name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls generated by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    /// ID of the tool call this message responds to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl OpenAiMessage {
    /// Creates a simple user message.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_owned(),
            content: Some(MessageContent::Text(text.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a simple system message.
    #[must_use]
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: "system".to_owned(),
            content: Some(MessageContent::Text(text.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates an assistant message with optional content and tool calls.
    #[must_use]
    pub fn assistant(content: Option<String>, tool_calls: Option<Vec<OpenAiToolCall>>) -> Self {
        Self {
            role: "assistant".to_owned(),
            content: content.map(MessageContent::Text),
            name: None,
            tool_calls,
            tool_call_id: None,
        }
    }
}

/// Stop sequence configuration in OpenAI chat completion request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAiStop {
    /// Single stop sequence.
    Single(String),
    /// Array of stop sequences.
    Multiple(Vec<String>),
}

/// Request body for OpenAI `POST /v1/chat/completions` (US-23).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiChatRequest {
    /// Target model identifier.
    pub model: String,
    /// Message history conversation turns.
    pub messages: Vec<OpenAiMessage>,
    /// Whether to stream partial message deltas via SSE.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Supported client tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiTool>>,
    /// Tool selection policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Maximum completion tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Maximum completion tokens alias (`max_completion_tokens`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Optional stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<OpenAiStop>,
}

/// Token usage metadata in OpenAI responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiUsage {
    /// Prompt token count (u32 per AGENTS.md §9).
    pub prompt_tokens: u32,
    /// Generated completion token count (u32 per AGENTS.md §9).
    pub completion_tokens: u32,
    /// Total token count.
    pub total_tokens: u32,
}

/// Assistant message in an OpenAI chat completion choice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiResponseMessage {
    /// Role ("assistant").
    pub role: String,
    /// Generated text content.
    pub content: Option<String>,
    /// Generated tool calls if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

/// Individual choice in a non-streaming OpenAI chat completion response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiChoice {
    /// Choice index (0-indexed).
    pub index: u32,
    /// Assistant message payload.
    pub message: OpenAiResponseMessage,
    /// Finish reason: "stop", "length", "tool_calls".
    pub finish_reason: Option<String>,
}

/// Response payload for synchronous OpenAI `POST /v1/chat/completions`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiChatResponse {
    /// Unique completion identifier (`chatcmpl-...`).
    pub id: String,
    /// Object type, always "chat.completion".
    pub object: String,
    /// Unix creation timestamp.
    pub created: u64,
    /// Model name that served generation.
    pub model: String,
    /// Generated choices array.
    pub choices: Vec<OpenAiChoice>,
    /// Token usage stats.
    pub usage: OpenAiUsage,
}

/// Tool call delta chunk within an SSE streaming choice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiToolCallChunk {
    /// Index of tool call.
    pub index: u32,
    /// Tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Type ("function").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    /// Partial function call arguments or name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenAiFunctionCallChunk>,
}

/// Partial function call delta.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiFunctionCallChunk {
    /// Function name delta.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Partial serialized arguments JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Partial delta payload in an SSE streaming chunk choice.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiChunkDelta {
    /// Assistant role (typically emitted only on the first chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Partial generated text content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Partial tool calls delta.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCallChunk>>,
}

/// Choice in an SSE streaming chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiChunkChoice {
    /// Choice index.
    pub index: u32,
    /// Delta payload.
    pub delta: OpenAiChunkDelta,
    /// Finish reason when generation stops.
    pub finish_reason: Option<String>,
}

/// Single SSE event payload chunk (`chat.completion.chunk`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiChunkResponse {
    /// Unique completion identifier.
    pub id: String,
    /// Object type, always "chat.completion.chunk".
    pub object: String,
    /// Unix creation timestamp.
    pub created: u64,
    /// Model name.
    pub model: String,
    /// Choices delta list.
    pub choices: Vec<OpenAiChunkChoice>,
}

/// Structured OpenAI error details.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiErrorDetails {
    /// Human-readable diagnostic error message.
    pub message: String,
    /// Error category ("invalid_request_error", "requests", etc.).
    pub r#type: String,
    /// Optional affected parameter.
    pub param: Option<String>,
    /// Machine-readable error code ("context_length_exceeded", "rate_limit_exceeded", "model_not_found").
    pub code: Option<String>,
}

/// Structured OpenAI error response envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiErrorResponse {
    /// Inner error object.
    pub error: OpenAiErrorDetails,
}

// ============================================================================
// Anthropic Claude Protocol Types (US-24)
// ============================================================================

/// System prompt representation in Claude messages API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ClaudeSystem {
    /// Single string system prompt.
    Text(String),
    /// Array of structured system text blocks.
    Blocks(Vec<ClaudeSystemBlock>),
}

impl ClaudeSystem {
    /// Extracts plain text representation of the system prompt.
    #[must_use]
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(blocks) => {
                blocks.iter().map(|b| b.text.as_str()).collect::<Vec<_>>().join("\n\n")
            }
        }
    }
}

/// Structured system text block in Claude API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeSystemBlock {
    /// Block type ("text").
    pub r#type: String,
    /// System prompt text.
    pub text: String,
}

/// Content block in Claude message history or response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeContentBlock {
    /// Block type: "text", "thinking", "tool_use", "tool_result", "image".
    pub r#type: String,
    /// Text content (for type "text").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Thinking / reasoning content (for type "thinking").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Tool use ID (for type "tool_use").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Tool name (for type "tool_use").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool input JSON arguments (for type "tool_use").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    /// Associated tool use ID (for type "tool_result").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,
    /// Tool result content payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    /// Whether the tool execution resulted in an error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    /// Optional media source for images.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
}

impl ClaudeContentBlock {
    /// Creates a text content block.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            r#type: "text".to_owned(),
            text: Some(text.into()),
            thinking: None,
            id: None,
            name: None,
            input: None,
            tool_use_id: None,
            content: None,
            is_error: None,
            source: None,
        }
    }
}

/// Flexible content representation for a Claude message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ClaudeContent {
    /// Single text string.
    Text(String),
    /// Array of structured content blocks.
    Blocks(Vec<ClaudeContentBlock>),
}

impl ClaudeContent {
    /// Extracts plain text representation of content.
    #[must_use]
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(blocks) => {
                let mut buf = String::new();
                for b in blocks {
                    if let Some(t) = &b.text {
                        buf.push_str(t);
                    } else if let Some(th) = &b.thinking {
                        buf.push_str(th);
                    }
                }
                buf
            }
        }
    }
}

/// Message turn in a Claude conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeMessage {
    /// Role: "user" or "assistant".
    pub role: String,
    /// Message content.
    pub content: ClaudeContent,
}

impl ClaudeMessage {
    /// Creates a user message with plain text.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_owned(),
            content: ClaudeContent::Text(text.into()),
        }
    }

    /// Creates an assistant message with plain text.
    #[must_use]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_owned(),
            content: ClaudeContent::Text(text.into()),
        }
    }
}

/// Reasoning / thinking configuration for Claude extended thinking models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeThinkingConfig {
    /// Thinking mode ("enabled").
    pub r#type: String,
    /// Maximum thinking tokens budget (u32 per AGENTS.md §9).
    pub budget_tokens: u32,
}

/// Tool declaration in Claude messages request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeTool {
    /// Unique tool name.
    pub name: String,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON schema describing expected input arguments.
    pub input_schema: serde_json::Value,
}

/// Request body for Claude `POST /v1/messages` (US-24).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaudeMessageRequest {
    /// Target model identifier.
    pub model: String,
    /// Optional system prompt instructions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<ClaudeSystem>,
    /// Conversation message turns.
    pub messages: Vec<ClaudeMessage>,
    /// Maximum tokens to generate (required).
    pub max_tokens: u32,
    /// Optional thinking / reasoning budget configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ClaudeThinkingConfig>,
    /// Client-declared tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ClaudeTool>>,
    /// Tool choice configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Custom stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to stream responses via SSE.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

/// Token usage metadata in Claude responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeUsage {
    /// Number of input prompt tokens (u32 per AGENTS.md §9).
    pub input_tokens: u32,
    /// Number of output completion tokens (u32 per AGENTS.md §9).
    pub output_tokens: u32,
}

/// Response payload for synchronous Claude `POST /v1/messages`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeMessageResponse {
    /// Unique message identifier (`msg_...`).
    pub id: String,
    /// Object type, always "message".
    pub r#type: String,
    /// Role, always "assistant".
    pub role: String,
    /// Generated content blocks.
    pub content: Vec<ClaudeContentBlock>,
    /// Model name that served generation.
    pub model: String,
    /// Stop reason: "end_turn", "max_tokens", "stop_sequence", "tool_use".
    pub stop_reason: Option<String>,
    /// Stop sequence that halted generation, if any.
    pub stop_sequence: Option<String>,
    /// Token usage stats.
    pub usage: ClaudeUsage,
}

/// Structured Claude error details.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeErrorDetails {
    /// Error category ("invalid_request_error", "overloaded_error", "not_found_error").
    pub r#type: String,
    /// Diagnostic error message.
    pub message: String,
}

/// Structured Claude error response envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeErrorResponse {
    /// Type, always "error".
    pub r#type: String,
    /// Error object details.
    pub error: ClaudeErrorDetails,
}
