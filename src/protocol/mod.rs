//! External client protocol front-ends and adapters (US-23, US-24, Milestone M3).
//!
//! Provides network-isolated edge extensions exposing standard `OpenAI` (`GET /v1/models`,
//! `POST /v1/chat/completions`) and Anthropic Claude (`POST /v1/messages`) APIs.

#![allow(clippy::doc_markdown)]

pub mod claude;
pub mod openai;
pub mod server;
pub mod types;

pub use claude::ClaudeAdapter;
pub use openai::OpenAiAdapter;
pub use server::{MAX_REQUEST_HEADERS_BYTES, ProtocolServer, ProtocolServerConfig, ServerError};
pub use types::{
    ClaudeContent, ClaudeContentBlock, ClaudeErrorDetails, ClaudeErrorResponse, ClaudeMessage,
    ClaudeMessageRequest, ClaudeMessageResponse, ClaudeSystem, ClaudeSystemBlock,
    ClaudeThinkingConfig, ClaudeTool, ClaudeUsage, DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_SOCKET_TIMEOUT_SECS, HttpRequest, HttpResponse, HttpResponseBody, MessageContent,
    OpenAiChatRequest, OpenAiChatResponse, OpenAiChoice, OpenAiChunkChoice, OpenAiChunkDelta,
    OpenAiChunkResponse, OpenAiErrorDetails, OpenAiErrorResponse, OpenAiFunction,
    OpenAiFunctionCall, OpenAiFunctionCallChunk, OpenAiMessage, OpenAiResponseMessage, OpenAiStop,
    OpenAiTool, OpenAiToolCall, OpenAiToolCallChunk, OpenAiUsage,
};
