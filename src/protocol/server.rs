//! Lightweight, robust HTTP server and router for external protocols (US-23, US-24, Feature 28).
//!
//! Provides bounded request size limits (10MB default) to prevent memory exhaustion,
//! socket read/write timeouts to mitigate Slowloris attacks, and automatic dispatching
//! to OpenAI and Claude adapters.
#![allow(
    clippy::pedantic,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines
)]

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::protocol::claude::ClaudeAdapter;
use crate::protocol::openai::OpenAiAdapter;
use crate::protocol::types::{
    DEFAULT_MAX_REQUEST_BODY_BYTES, DEFAULT_SOCKET_TIMEOUT_SECS, HttpRequest, HttpResponse,
    HttpResponseBody,
};
use crate::runtime::ControlPlaneRuntime;

/// Maximum allowed header section length (64 KB) to prevent header-bomb DoS attacks.
pub const MAX_REQUEST_HEADERS_BYTES: usize = 64 * 1024;

/// Configuration parameters for the HTTP protocol server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolServerConfig {
    /// Maximum allowed request body size in bytes (default 10 MB).
    pub max_request_body_bytes: usize,
    /// Maximum allowed header section size in bytes (default 64 KB).
    pub max_headers_bytes: usize,
    /// Socket read and write timeout.
    pub socket_timeout: Duration,
    /// Optional TCP socket listen address.
    pub listen_addr: Option<SocketAddr>,
}

impl Default for ProtocolServerConfig {
    fn default() -> Self {
        Self {
            max_request_body_bytes: DEFAULT_MAX_REQUEST_BODY_BYTES,
            max_headers_bytes: MAX_REQUEST_HEADERS_BYTES,
            socket_timeout: Duration::from_secs(DEFAULT_SOCKET_TIMEOUT_SECS),
            listen_addr: None,
        }
    }
}

/// Errors occurring during HTTP parsing and socket serving.
#[derive(Debug)]
pub enum ServerError {
    /// Socket I/O error.
    Io(std::io::Error),
    /// Socket read or write operation timed out.
    Timeout,
    /// Request headers exceeded maximum size bound.
    HeadersTooLarge,
    /// Request body payload exceeded maximum size bound.
    PayloadTooLarge,
    /// Malformed HTTP request line or headers.
    BadRequest(String),
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Timeout => write!(f, "Socket operation timed out"),
            Self::HeadersTooLarge => write!(f, "HTTP headers exceed maximum limit"),
            Self::PayloadTooLarge => write!(f, "HTTP request payload exceeds maximum limit"),
            Self::BadRequest(msg) => write!(f, "Bad HTTP request: {msg}"),
        }
    }
}

impl std::error::Error for ServerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ServerError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// Protocol router and server dispatching requests to OpenAI and Claude endpoints.
#[derive(Debug, Clone)]
pub struct ProtocolServer {
    config: ProtocolServerConfig,
    runtime: Arc<ControlPlaneRuntime>,
}

impl ProtocolServer {
    /// Creates a new protocol server with the specified configuration and runtime.
    #[must_use]
    pub fn new(config: ProtocolServerConfig, runtime: Arc<ControlPlaneRuntime>) -> Self {
        Self { config, runtime }
    }

    /// Accesses the server configuration.
    #[must_use]
    pub const fn config(&self) -> &ProtocolServerConfig {
        &self.config
    }

    /// Accesses the control plane runtime.
    #[must_use]
    pub fn runtime(&self) -> &Arc<ControlPlaneRuntime> {
        &self.runtime
    }

    /// Routes an HTTP request frame to the appropriate protocol adapter.
    pub async fn route_request(&self, req: HttpRequest) -> HttpResponse {
        // Enforce maximum body size check
        if req.body.len() > self.config.max_request_body_bytes {
            let json = serde_json::json!({
                "error": {
                    "message": format!(
                        "Request payload size ({} bytes) exceeds maximum allowed limit of {} bytes",
                        req.body.len(),
                        self.config.max_request_body_bytes
                    ),
                    "type": "invalid_request_error",
                    "code": "payload_too_large"
                }
            })
            .to_string();
            return HttpResponse::error(413, json, Vec::new());
        }

        let path = req.path.split('?').next().unwrap_or(&req.path).to_owned();
        let mut req = req;
        req.path = path.clone();

        if req.method == "GET" && (path == "/v1/models" || path == "/models") {
            OpenAiAdapter::handle_models(&self.runtime)
        } else if path == "/v1/chat/completions" || path == "/chat/completions" {
            OpenAiAdapter::handle_http(req, Arc::clone(&self.runtime)).await
        } else if path == "/v1/messages" || path == "/messages" {
            ClaudeAdapter::handle_http(req, Arc::clone(&self.runtime)).await
        } else {
            let json = serde_json::json!({
                "error": {
                    "message": format!("Route not found: {} {}", req.method, req.path),
                    "type": "invalid_request_error",
                    "code": "not_found"
                }
            })
            .to_string();
            HttpResponse::error(404, json, Vec::new())
        }
    }

    /// Handles a single incoming TCP stream or async I/O connection.
    ///
    /// Reads HTTP request line, headers (up to 64KB bound), and body (up to 10MB bound),
    /// routes to the appropriate handler, and writes back the serialized response.
    ///
    /// # Errors
    /// Returns [`ServerError`] on socket I/O, timeout, or boundary violations.
    pub async fn handle_connection<S>(&self, mut stream: S) -> Result<(), ServerError>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        // 1. Read HTTP request line and headers with timeout
        let read_future = async {
            let mut header_buf = Vec::new();
            let mut temp_buf = [0u8; 1024];
            let mut header_end_idx = None;

            loop {
                let n = stream.read(&mut temp_buf).await?;
                if n == 0 {
                    break;
                }
                header_buf.extend_from_slice(&temp_buf[..n]);
                if header_buf.len() > self.config.max_headers_bytes {
                    return Err(ServerError::HeadersTooLarge);
                }

                if let Some(pos) = header_buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    header_end_idx = Some(pos);
                    break;
                }
            }

            let Some(end_idx) = header_end_idx else {
                return Err(ServerError::BadRequest("Incomplete HTTP headers".to_owned()));
            };

            let header_bytes = &header_buf[..end_idx];
            let remaining_body = header_buf[end_idx + 4..].to_vec();

            // Parse request line and headers
            let header_str = std::str::from_utf8(header_bytes)
                .map_err(|e| ServerError::BadRequest(format!("Non-UTF8 headers: {e}")))?;

            let mut lines = header_str.split("\r\n");
            let req_line = lines
                .next()
                .ok_or_else(|| ServerError::BadRequest("Missing request line".to_owned()))?;

            let mut req_line_parts = req_line.split_whitespace();
            let method = req_line_parts
                .next()
                .ok_or_else(|| ServerError::BadRequest("Missing HTTP method".to_owned()))?
                .to_owned();
            let path = req_line_parts
                .next()
                .ok_or_else(|| ServerError::BadRequest("Missing HTTP path".to_owned()))?
                .to_owned();

            let mut headers = Vec::new();
            let mut content_length: usize = 0;

            for line in lines {
                if let Some((name, val)) = line.split_once(':') {
                    let h_name = name.trim().to_owned();
                    let h_val = val.trim().to_owned();
                    if h_name.eq_ignore_ascii_case("content-length") {
                        if let Ok(cl) = h_val.parse::<usize>() {
                            content_length = cl;
                        }
                    }
                    headers.push((h_name, h_val));
                }
            }

            // Check Content-Length against maximum body size
            if content_length > self.config.max_request_body_bytes {
                return Err(ServerError::PayloadTooLarge);
            }

            // Read remaining body bytes if needed
            let mut body = remaining_body;
            while body.len() < content_length {
                let to_read = (content_length - body.len()).min(temp_buf.len());
                let n = stream.read(&mut temp_buf[..to_read]).await?;
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&temp_buf[..n]);
                if body.len() > self.config.max_request_body_bytes {
                    return Err(ServerError::PayloadTooLarge);
                }
            }

            let mut req = HttpRequest::new(method, path, body);
            req.headers = headers;
            Ok(req)
        };

        let req_result = Self::run_with_timeout(self.config.socket_timeout, read_future).await;

        let req = match req_result {
            Ok(r) => r,
            Err(ServerError::PayloadTooLarge) => {
                let resp_body = serde_json::json!({
                    "error": {
                        "message": format!(
                            "Request payload exceeds maximum allowed limit of {} bytes",
                            self.config.max_request_body_bytes
                        ),
                        "type": "invalid_request_error",
                        "code": "payload_too_large"
                    }
                })
                .to_string();
                let resp = HttpResponse::error(413, resp_body, Vec::new());
                self.write_http_response(&mut stream, resp).await?;
                return Err(ServerError::PayloadTooLarge);
            }
            Err(e) => return Err(e),
        };

        // 2. Route request and write response
        let resp = self.route_request(req).await;
        self.write_http_response(&mut stream, resp).await?;
        Ok(())
    }

    /// Serializes and writes an HTTP response frame to an async stream.
    async fn write_http_response<S>(&self, stream: &mut S, resp: HttpResponse) -> Result<(), ServerError>
    where
        S: AsyncWrite + Unpin,
    {
        let status_reason = match resp.status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            413 => "Payload Too Large",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            529 => "Overloaded",
            _ => "Response",
        };

        let mut head = format!("HTTP/1.1 {} {}\r\n", resp.status, status_reason);
        for (k, v) in &resp.headers {
            head.push_str(k);
            head.push_str(": ");
            head.push_str(v);
            head.push_str("\r\n");
        }

        match resp.body {
            HttpResponseBody::Bytes(bytes) => {
                head.push_str("\r\n");
                stream.write_all(head.as_bytes()).await?;
                if !bytes.is_empty() {
                    stream.write_all(&bytes).await?;
                }
                stream.flush().await?;
            }
            HttpResponseBody::Stream(mut rx) => {
                // Chunked transfer encoding for streaming SSE
                head.push_str("Transfer-Encoding: chunked\r\n\r\n");
                stream.write_all(head.as_bytes()).await?;
                stream.flush().await?;

                while let Some(frame) = rx.recv().await {
                    let chunk_len = frame.len();
                    if chunk_len == 0 {
                        continue;
                    }
                    let chunk_header = format!("{chunk_len:X}\r\n");
                    stream.write_all(chunk_header.as_bytes()).await?;
                    stream.write_all(frame.as_bytes()).await?;
                    stream.write_all(b"\r\n").await?;
                    stream.flush().await?;
                }

                // Final terminating chunk (0\r\n\r\n)
                stream.write_all(b"0\r\n\r\n").await?;
                stream.flush().await?;
            }
        }

        Ok(())
    }

    /// Binds to configured `listen_addr` and serves connections continuously.
    ///
    /// # Errors
    /// Returns [`std::io::Error`] if binding fails or `listen_addr` was not specified.
    pub async fn bind_and_serve(&self) -> Result<(), std::io::Error> {
        let addr = self.config.listen_addr.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::AddrNotAvailable,
                "listen_addr was not configured",
            )
        })?;

        let listener = TcpListener::bind(addr).await?;
        loop {
            let (socket, _) = listener.accept().await?;
            let server_clone = self.clone();
            tokio::spawn(async move {
                let _ = server_clone.handle_connection(socket).await;
            });
        }
    }

    /// Executes an async future with a bounded timeout using `tokio::time::timeout`.
    async fn run_with_timeout<F, T>(timeout: Duration, future: F) -> Result<T, ServerError>
    where
        F: Future<Output = Result<T, ServerError>>,
    {
        if timeout.is_zero() {
            return future.await;
        }

        tokio::time::timeout(timeout, future)
            .await
            .map_err(|_| ServerError::Timeout)?
    }
}
