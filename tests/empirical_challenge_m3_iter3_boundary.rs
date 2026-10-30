//! Empirical Challenge Suite for Milestone M3 Iteration 3 Remediation.
//!
//! Adversarial verification of:
//! 1. Query parameter route tolerance across ProtocolServer, OpenAiAdapter, and ClaudeAdapter.
//! 2. Multiple '?' delimiters in query strings.
//! 3. Empty '?' query strings.
//! 4. Non-GET requests on /v1/models returning HTTP 404.
//! 5. Non-POST requests on chat and messages returning HTTP 404.
//! 6. Trailing slashes strictly returning HTTP 404 (with and without query params).
//! 7. Saturated queues (HTTP 429 / 529 Retry-After) and early client disconnect permit teardown.

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
    ClaudeAdapter, HttpRequest, OpenAiAdapter, ProtocolServer, ProtocolServerConfig,
};
use k3_manager::queue::AdmissionQueueConfig;
use k3_manager::runtime::{AdminCommand, ControlPlaneRuntime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn build_test_runtime() -> Arc<ControlPlaneRuntime> {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid control plane capacity");
    let mock_affinity: Arc<dyn OsAffinityBackend> = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");
    Arc::new(runtime)
}

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
// 1. Multiple '?' delimiters in query string
// ============================================================================

#[tokio::test]
async fn test_challenge_multiple_question_mark_delimiters() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "boundary-test-model", 2048);
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    let openai_body = serde_json::json!({
        "model": "boundary-test-model",
        "messages": [{"role": "user", "content": "Hello double question mark"}],
        "stream": false
    })
    .to_string();

    let claude_body = serde_json::json!({
        "model": "boundary-test-model",
        "messages": [{"role": "user", "content": "Hello double question mark"}],
        "max_tokens": 128
    })
    .to_string();

    // 1. GET /v1/models??extra=1?foo=2 on ProtocolServer
    let req = HttpRequest::new("GET", "/v1/models??extra=1?foo=2", Vec::new());
    let resp = server.route_request(req).await;
    assert_eq!(
        resp.status, 200,
        "ProtocolServer should accept /v1/models??extra=1?foo=2"
    );

    // 2. GET /v1/models??extra=1?foo=2 on OpenAiAdapter::handle_http
    let req = HttpRequest::new("GET", "/v1/models??extra=1?foo=2", Vec::new());
    let resp = OpenAiAdapter::handle_http(req, Arc::clone(&runtime)).await;
    assert_eq!(
        resp.status, 200,
        "OpenAiAdapter should accept /v1/models??extra=1?foo=2"
    );

    // 3. POST /v1/chat/completions??extra=1?foo=2 on ProtocolServer
    let req = HttpRequest::new(
        "POST",
        "/v1/chat/completions??extra=1?foo=2",
        openai_body.clone().into_bytes(),
    );
    let resp = server.route_request(req).await;
    assert_eq!(
        resp.status, 200,
        "ProtocolServer should accept /v1/chat/completions??extra=1?foo=2"
    );

    // 4. POST /v1/chat/completions??extra=1?foo=2 on OpenAiAdapter::handle_http
    let req = HttpRequest::new(
        "POST",
        "/v1/chat/completions??extra=1?foo=2",
        openai_body.into_bytes(),
    );
    let resp = OpenAiAdapter::handle_http(req, Arc::clone(&runtime)).await;
    assert_eq!(
        resp.status, 200,
        "OpenAiAdapter should accept /v1/chat/completions??extra=1?foo=2"
    );

    // 5. POST /v1/messages??extra=1?foo=2 on ProtocolServer
    let req = HttpRequest::new(
        "POST",
        "/v1/messages??extra=1?foo=2",
        claude_body.clone().into_bytes(),
    );
    let resp = server.route_request(req).await;
    assert_eq!(
        resp.status, 200,
        "ProtocolServer should accept /v1/messages??extra=1?foo=2"
    );

    // 6. POST /v1/messages??extra=1?foo=2 on ClaudeAdapter::handle_http
    let req = HttpRequest::new(
        "POST",
        "/v1/messages??extra=1?foo=2",
        claude_body.into_bytes(),
    );
    let resp = ClaudeAdapter::handle_http(req, Arc::clone(&runtime)).await;
    assert_eq!(
        resp.status, 200,
        "ClaudeAdapter should accept /v1/messages??extra=1?foo=2"
    );
}

// ============================================================================
// 2. Empty '?' query string
// ============================================================================

#[tokio::test]
async fn test_challenge_empty_query_strings() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "empty-query-model", 2048);
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    let openai_body = serde_json::json!({
        "model": "empty-query-model",
        "messages": [{"role": "user", "content": "Hello empty query"}],
        "stream": false
    })
    .to_string();

    let claude_body = serde_json::json!({
        "model": "empty-query-model",
        "messages": [{"role": "user", "content": "Hello empty query"}],
        "max_tokens": 128
    })
    .to_string();

    // 1. Models routes with empty '?'
    let req = HttpRequest::new("GET", "/v1/models?", Vec::new());
    let resp = server.route_request(req).await;
    assert_eq!(resp.status, 200, "ProtocolServer should accept /v1/models?");

    let req = HttpRequest::new("GET", "/models?", Vec::new());
    let resp = server.route_request(req).await;
    assert_eq!(resp.status, 200, "ProtocolServer should accept /models?");

    let req = HttpRequest::new("GET", "/v1/models?", Vec::new());
    let resp = OpenAiAdapter::handle_http(req, Arc::clone(&runtime)).await;
    assert_eq!(resp.status, 200, "OpenAiAdapter should accept /v1/models?");

    // 2. Chat completions routes with empty '?'
    let req = HttpRequest::new(
        "POST",
        "/v1/chat/completions?",
        openai_body.clone().into_bytes(),
    );
    let resp = server.route_request(req).await;
    assert_eq!(
        resp.status, 200,
        "ProtocolServer should accept /v1/chat/completions?"
    );

    let req = HttpRequest::new(
        "POST",
        "/chat/completions?",
        openai_body.clone().into_bytes(),
    );
    let resp = server.route_request(req).await;
    assert_eq!(
        resp.status, 200,
        "ProtocolServer should accept /chat/completions?"
    );

    let req = HttpRequest::new("POST", "/v1/chat/completions?", openai_body.into_bytes());
    let resp = OpenAiAdapter::handle_http(req, Arc::clone(&runtime)).await;
    assert_eq!(
        resp.status, 200,
        "OpenAiAdapter should accept /v1/chat/completions?"
    );

    // 3. Messages routes with empty '?'
    let req = HttpRequest::new("POST", "/v1/messages?", claude_body.clone().into_bytes());
    let resp = server.route_request(req).await;
    assert_eq!(
        resp.status, 200,
        "ProtocolServer should accept /v1/messages?"
    );

    let req = HttpRequest::new("POST", "/messages?", claude_body.clone().into_bytes());
    let resp = server.route_request(req).await;
    assert_eq!(resp.status, 200, "ProtocolServer should accept /messages?");

    let req = HttpRequest::new("POST", "/v1/messages?", claude_body.into_bytes());
    let resp = ClaudeAdapter::handle_http(req, Arc::clone(&runtime)).await;
    assert_eq!(
        resp.status, 200,
        "ClaudeAdapter should accept /v1/messages?"
    );
}

// ============================================================================
// 3. Non-GET requests on /v1/models returning 404
// ============================================================================

#[tokio::test]
async fn test_challenge_non_get_requests_on_models_return_404() {
    let runtime = build_test_runtime();
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    let methods = ["POST", "DELETE", "PUT", "PATCH"];

    for method in methods {
        // ProtocolServer /v1/models
        let req = HttpRequest::new(method, "/v1/models", Vec::new());
        let resp = server.route_request(req).await;
        assert_eq!(
            resp.status, 404,
            "ProtocolServer {method} /v1/models must return 404"
        );

        // ProtocolServer /models
        let req = HttpRequest::new(method, "/models", Vec::new());
        let resp = server.route_request(req).await;
        assert_eq!(
            resp.status, 404,
            "ProtocolServer {method} /models must return 404"
        );

        // ProtocolServer with query params
        let req = HttpRequest::new(method, "/v1/models?limit=10", Vec::new());
        let resp = server.route_request(req).await;
        assert_eq!(
            resp.status, 404,
            "ProtocolServer {method} /v1/models?limit=10 must return 404"
        );

        // OpenAiAdapter direct /v1/models
        let req = HttpRequest::new(method, "/v1/models", Vec::new());
        let resp = OpenAiAdapter::handle_http(req, Arc::clone(&runtime)).await;
        assert_eq!(
            resp.status, 404,
            "OpenAiAdapter {method} /v1/models must return 404"
        );

        // OpenAiAdapter direct /models
        let req = HttpRequest::new(method, "/models", Vec::new());
        let resp = OpenAiAdapter::handle_http(req, Arc::clone(&runtime)).await;
        assert_eq!(
            resp.status, 404,
            "OpenAiAdapter {method} /models must return 404"
        );
    }
}

// ============================================================================
// 4. Non-POST requests on chat and messages returning 404
// ============================================================================

#[tokio::test]
async fn test_challenge_non_post_requests_on_chat_and_messages_return_404() {
    let runtime = build_test_runtime();
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    let non_post_methods = ["GET", "DELETE", "PUT", "PATCH", "HEAD"];

    for method in non_post_methods {
        // Chat endpoints on ProtocolServer
        let req = HttpRequest::new(method, "/v1/chat/completions", Vec::new());
        let resp = server.route_request(req).await;
        assert_eq!(
            resp.status, 404,
            "ProtocolServer {method} /v1/chat/completions must return 404"
        );

        let req = HttpRequest::new(method, "/chat/completions", Vec::new());
        let resp = server.route_request(req).await;
        assert_eq!(
            resp.status, 404,
            "ProtocolServer {method} /chat/completions must return 404"
        );

        // Chat endpoints on OpenAiAdapter
        let req = HttpRequest::new(method, "/v1/chat/completions", Vec::new());
        let resp = OpenAiAdapter::handle_http(req, Arc::clone(&runtime)).await;
        assert_eq!(
            resp.status, 404,
            "OpenAiAdapter {method} /v1/chat/completions must return 404"
        );

        // Messages endpoints on ProtocolServer
        let req = HttpRequest::new(method, "/v1/messages", Vec::new());
        let resp = server.route_request(req).await;
        assert_eq!(
            resp.status, 404,
            "ProtocolServer {method} /v1/messages must return 404"
        );

        let req = HttpRequest::new(method, "/messages", Vec::new());
        let resp = server.route_request(req).await;
        assert_eq!(
            resp.status, 404,
            "ProtocolServer {method} /messages must return 404"
        );

        // Messages endpoints on ClaudeAdapter
        let req = HttpRequest::new(method, "/v1/messages", Vec::new());
        let resp = ClaudeAdapter::handle_http(req, Arc::clone(&runtime)).await;
        assert_eq!(
            resp.status, 404,
            "ClaudeAdapter {method} /v1/messages must return 404"
        );
    }
}

// ============================================================================
// 5. Trailing slash handling strictly returning 404
// ============================================================================

#[tokio::test]
async fn test_challenge_trailing_slashes_return_404() {
    let runtime = build_test_runtime();
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    let test_cases = [
        ("GET", "/v1/models/"),
        ("GET", "/models/"),
        ("GET", "/v1/models/?limit=10"),
        ("POST", "/v1/chat/completions/"),
        ("POST", "/chat/completions/"),
        ("POST", "/v1/chat/completions/?stream=false"),
        ("POST", "/v1/messages/"),
        ("POST", "/messages/"),
        ("POST", "/v1/messages/?beta=true"),
    ];

    for (method, path) in test_cases {
        let req = HttpRequest::new(method, path, Vec::new());
        let resp = server.route_request(req).await;
        assert_eq!(
            resp.status, 404,
            "ProtocolServer {method} {path} with trailing slash must strictly return 404"
        );

        if path.contains("models") || path.contains("chat") {
            let req_adapter = HttpRequest::new(method, path, Vec::new());
            let resp_adapter = OpenAiAdapter::handle_http(req_adapter, Arc::clone(&runtime)).await;
            assert_eq!(
                resp_adapter.status, 404,
                "OpenAiAdapter {method} {path} with trailing slash must strictly return 404"
            );
        }

        if path.contains("messages") {
            let req_adapter = HttpRequest::new(method, path, Vec::new());
            let resp_adapter = ClaudeAdapter::handle_http(req_adapter, Arc::clone(&runtime)).await;
            assert_eq!(
                resp_adapter.status, 404,
                "ClaudeAdapter {method} {path} with trailing slash must strictly return 404"
            );
        }
    }
}

// ============================================================================
// 6. Saturated queues (HTTP 429 / 529 Retry-After) and permit teardown
// ============================================================================

#[tokio::test]
async fn test_challenge_saturated_queue_and_client_disconnect_permit_teardown() {
    let runtime = build_test_runtime();
    setup_model(&runtime, "saturation-test-model", 2048);

    // Configure queue limit of 1
    runtime.queue_mgr().register_model(
        "saturation-test-model",
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
        "saturation-test-model",
        "req-holding-slot".to_owned(),
        false,
        None,
        None,
        cancel,
        None,
        None,
    );
    assert!(admit1.is_ok());

    // Second request should saturate the queue
    let openai_body = serde_json::json!({
        "model": "saturation-test-model",
        "messages": [{"role": "user", "content": "Saturated"}],
        "stream": false
    })
    .to_string();

    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));
    let req = HttpRequest::new("POST", "/v1/chat/completions", openai_body.into_bytes());
    let resp = server.route_request(req).await;
    assert_eq!(resp.status, 429, "Saturated queue must return HTTP 429");

    let retry_after = resp
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("Retry-After"));
    assert!(
        retry_after.is_some(),
        "Retry-After header must be present on 429"
    );
    let secs: u64 = retry_after
        .unwrap()
        .1
        .parse()
        .expect("valid integer Retry-After");
    assert!(secs > 0, "Retry-After seconds must be > 0");

    // Clear the pre-occupied slot so the streaming request is admitted
    runtime
        .queue_mgr()
        .mark_execution_completed("saturation-test-model");

    // Client disconnect permit teardown test
    let (mut client, server_stream) = tokio::io::duplex(4096);
    let srv = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));
    let server_handle = tokio::spawn(async move { srv.handle_connection(server_stream).await });

    let chat_req = serde_json::json!({
        "model": "saturation-test-model",
        "messages": [{"role": "user", "content": "Drop disconnect"}],
        "stream": true
    })
    .to_string();

    let wire_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
        chat_req.len(),
        chat_req
    );

    client.write_all(wire_req.as_bytes()).await.unwrap();

    let mut buf = [0u8; 128];
    let n = client.read(&mut buf).await.unwrap();
    assert!(n > 0);
    // Abrupt drop
    drop(client);

    let res = server_handle.await.unwrap();
    println!("server_handle result: {res:?}");

    // Give time for any background cleanup
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        runtime.queue_mgr().queue_depth("saturation-test-model"),
        0,
        "Queue depth must return to 0 after disconnect"
    );
}
