//! US-13 Dedicated Scenario Test Suite: Patient Request Queuing & SSE Keep-Alive (Features 29, 30, 31, ADR-015).
//!
//! Verifies:
//! - Strict FIFO admission order preservation and single-prompt forward-pass exclusivity.
//! - Disabled wait timeouts by default and zero artificial token/usage quotas.
//! - Streaming feedback: periodic SSE `: keep-alive\n\n` and advancing `: queue-position <pos>\n\n`.
//! - Graceful saturation rejection (depth 64 cap and client timeout limit) returning `K3Error::QueueSaturated`.
//! - Client disconnect queue pruning with immediate position advancement.
//! - Multi-model queue independence via `AdmissionQueueManager`.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use k3_core::error::K3Error;
use k3_manager::queue::{
    AdmissionQueueConfig, AdmissionQueueManager, DEFAULT_MAX_QUEUE_DEPTH, ModelAdmissionQueue,
    QueueSseFrame,
};

#[test]
fn test_us13_strict_fifo_admission_order() {
    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", AdmissionQueueConfig::default());

    // Admit 5 sequential requests
    for i in 1_u32..=5_u32 {
        let pos = queue
            .admit_simple(format!("req_{i}"), false)
            .expect("admission must succeed");
        assert_eq!(pos, i);
    }
    assert_eq!(queue.len(), 5);

    // Verify FIFO dequeue order
    for i in 1..=5 {
        let permit = queue.pop_next().expect("pop_next must yield permit");
        assert_eq!(permit.request_id, format!("req_{i}"));
        assert_eq!(permit.model_id, "qwen2.5-7b");
        queue.mark_execution_completed();
    }
    assert!(queue.is_empty());
}

#[test]
fn test_us13_disabled_wait_timeout_by_default() {
    let cfg = AdmissionQueueConfig::default();
    assert!(cfg.wait_timeout.is_none()); // Wait timeouts disabled by default per ADR-015

    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", cfg);
    queue
        .admit_simple("patient_req".to_owned(), false)
        .expect("admission succeeds");

    // Request remains queued indefinitely until popped or cancelled
    assert_eq!(queue.len(), 1);
    let permit = queue.pop_next().expect("permit granted");
    assert_eq!(permit.request_id, "patient_req");
}

#[test]
fn test_us13_zero_token_and_usage_quota_throttling() {
    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", AdmissionQueueConfig::default());

    // 100 sequential requests processed without quota throttling or counter limits
    for i in 1..=100 {
        let pos = queue
            .admit_simple(format!("quota_free_{i}"), false)
            .unwrap();
        assert_eq!(pos, 1);
        let permit = queue.pop_next().unwrap();
        assert_eq!(permit.request_id, format!("quota_free_{i}"));
        queue.mark_execution_completed();
    }

    assert_eq!(queue.metrics().total_admitted, 100);
    assert_eq!(queue.metrics().total_dispatched, 100);
    assert_eq!(queue.metrics().total_rejected, 0);
}

#[test]
fn test_us13_exact_capacity_saturation_64th_vs_65th() {
    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", AdmissionQueueConfig::default());
    assert_eq!(queue.max_depth(), DEFAULT_MAX_QUEUE_DEPTH);
    assert_eq!(queue.max_depth(), 64);

    // Admit exactly 64 requests
    for i in 1_u32..=64_u32 {
        let pos = queue
            .admit_simple(format!("req_{i}"), false)
            .expect("must admit");
        assert_eq!(pos, i);
    }
    assert_eq!(queue.len(), 64);

    // 65th request is rejected with K3Error::QueueSaturated
    let err = queue
        .admit_simple("req_65".to_owned(), false)
        .expect_err("65th request must be rejected");

    match err {
        K3Error::QueueSaturated {
            model_id,
            current_depth,
            max_depth,
            retry_after_secs,
        } => {
            assert_eq!(model_id, "qwen2.5-7b");
            assert_eq!(current_depth, 64);
            assert_eq!(max_depth, 64);
            assert!(retry_after_secs >= 5);
        }
        other => panic!("expected QueueSaturated, got {other:?}"),
    }

    // Existing 64 requests remain intact and unaffected
    assert_eq!(queue.len(), 64);
    assert_eq!(queue.metrics().total_admitted, 64);
    assert_eq!(queue.metrics().total_rejected, 1);
}

#[test]
fn test_us13_zero_depth_queue_immediate_saturation() {
    let config = AdmissionQueueConfig {
        max_depth: 0, // Configured zero capacity ceiling
        ..AdmissionQueueConfig::default()
    };

    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", config);
    let err = queue
        .admit_simple("instant_reject".to_owned(), false)
        .expect_err("must reject immediately");

    match err {
        K3Error::QueueSaturated {
            current_depth,
            max_depth,
            ..
        } => {
            assert_eq!(current_depth, 0);
            assert_eq!(max_depth, 0);
        }
        other => panic!("expected QueueSaturated, got {other:?}"),
    }
}

#[test]
fn test_us13_dynamic_wait_time_exceeds_client_timeout() {
    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", AdmissionQueueConfig::default());

    // Enqueue 10 requests
    for i in 1..=10 {
        queue.admit_simple(format!("bg_{i}"), false).unwrap();
    }

    // Client requests 100ms timeout, but estimated wait is 10 * 25ms * 16 passes = 4000ms
    let cancel_token = Arc::new(AtomicBool::new(false));
    let tight_timeout = Duration::from_millis(100);

    let err = queue
        .admit(
            "timeout_sensitive".to_owned(),
            false,
            Some(tight_timeout),
            Some(25.0),
            cancel_token,
            None,
            None,
        )
        .expect_err("should reject upfront due to estimated wait > timeout");

    match err {
        K3Error::QueueSaturated {
            current_depth,
            retry_after_secs,
            ..
        } => {
            assert_eq!(current_depth, 10);
            assert!(retry_after_secs >= 4);
        }
        other => panic!("expected QueueSaturated, got {other:?}"),
    }
}

#[test]
fn test_us13_client_disconnect_pruning_and_position_advancement() {
    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", AdmissionQueueConfig::default());

    let (tx4, mut rx4) = tokio::sync::mpsc::channel::<String>(16);
    let (tx5, mut rx5) = tokio::sync::mpsc::channel::<String>(16);

    // Admit req1, req2, req3 (non-streaming)
    queue.admit_simple("req1".to_owned(), false).unwrap();
    queue.admit_simple("req2".to_owned(), false).unwrap();
    queue.admit_simple("req3".to_owned(), false).unwrap();

    // Admit req4 and req5 with streaming SSE channels
    queue
        .admit(
            "req4".to_owned(),
            true,
            None,
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            Some(tx4),
        )
        .unwrap();

    queue
        .admit(
            "req5".to_owned(),
            true,
            None,
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            Some(tx5),
        )
        .unwrap();

    assert_eq!(queue.len(), 5);

    // Client at position 3 ("req3") disconnects
    let pruned = queue.cancel("req3");
    assert!(pruned);
    assert_eq!(queue.len(), 4);
    assert_eq!(queue.metrics().total_cancelled, 1);

    // req4 received advanced position notification (from 4 to 3)
    let msg4 = rx4.try_recv().expect("req4 should receive position update");
    assert!(msg4.contains("queue-position 3") || msg4.contains("position=3"));

    // req5 received advanced position notification (from 5 to 4)
    let msg5 = rx5.try_recv().expect("req5 should receive position update");
    assert!(msg5.contains("queue-position 4") || msg5.contains("position=4"));
}

#[test]
fn test_us13_sse_keepalive_and_position_frames_formatting() {
    let keep_alive_frame = QueueSseFrame::KeepAlive.format_frame();
    assert_eq!(keep_alive_frame, ": keep-alive\n\n");

    let pos_frame = QueueSseFrame::QueuePosition(7).format_frame();
    assert_eq!(pos_frame, ": queue-position 7\n\n");

    let legacy_pos_frame = QueueSseFrame::format_position_comment(7);
    assert_eq!(legacy_pos_frame, ": position=7\n\n");

    let combined = QueueSseFrame::format_combined_keep_alive(7);
    assert_eq!(combined, ": keep-alive position=7\n\n");
}

#[test]
fn test_us13_multi_model_admission_manager_independence() {
    let manager = AdmissionQueueManager::default();

    // Register 2 distinct models with default depth 64
    manager.register_model("qwen2.5-coder-7b", None);
    manager.register_model("gemma-2-9b-it", None);

    // Saturate qwen2.5-coder-7b to 64
    for i in 1_u32..=64_u32 {
        let pos = manager
            .admit_to_model(
                "qwen2.5-coder-7b",
                format!("coder_req_{i}"),
                false,
                None,
                None,
                Arc::new(AtomicBool::new(false)),
                None,
                None,
            )
            .unwrap();
        assert_eq!(pos, i);
    }
    assert_eq!(manager.queue_depth("qwen2.5-coder-7b"), 64);

    // 65th request on qwen2.5-coder-7b is rejected
    let coder_err = manager.admit_to_model(
        "qwen2.5-coder-7b",
        "coder_req_65".to_owned(),
        false,
        None,
        None,
        Arc::new(AtomicBool::new(false)),
        None,
        None,
    );
    assert!(coder_err.is_err());

    // Meanwhile, gemma-2-9b-it queue is completely independent and admits cleanly
    let gemma_pos = manager
        .admit_to_model(
            "gemma-2-9b-it",
            "gemma_req_1".to_owned(),
            false,
            None,
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            None,
        )
        .expect("independent model queue must admit");
    assert_eq!(gemma_pos, 1);
    assert_eq!(manager.queue_depth("gemma-2-9b-it"), 1);
    assert_eq!(manager.total_queue_depth(), 65);
}

#[test]
fn test_us13_infinite_and_nan_latency_estimation_guard() {
    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", AdmissionQueueConfig::default());
    queue.admit_simple("req_1".to_owned(), false).unwrap();

    // Default pass latency is 25.0ms; passes per request is 16; pending slots = 1 -> 400.0ms
    let default_wait = queue.estimate_wait_ms(None);
    assert_eq!(default_wait, 400.0);

    // Passing f64::INFINITY is safely filtered out and falls back to default 25.0ms
    let inf_wait = queue.estimate_wait_ms(Some(f64::INFINITY));
    assert_eq!(inf_wait, 400.0);

    // Passing f64::NAN is safely filtered out and falls back to default 25.0ms
    let nan_wait = queue.estimate_wait_ms(Some(f64::NAN));
    assert_eq!(nan_wait, 400.0);

    // Passing negative latency is safely filtered out
    let neg_wait = queue.estimate_wait_ms(Some(-5.0));
    assert_eq!(neg_wait, 400.0);

    // Valid positive finite latency is applied: 10.0ms * 16 * 1 = 160.0ms
    let valid_wait = queue.estimate_wait_ms(Some(10.0));
    assert_eq!(valid_wait, 160.0);
}
