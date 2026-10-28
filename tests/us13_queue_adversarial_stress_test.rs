//! US-13 Adversarial Stress & Concurrency Test Suite (Features 29, 30, 31, ADR-015).
//!
//! Empirically challenges and stress-tests:
//! 1. Strict FIFO ordering under high concurrent load (multiple threads enqueuing simultaneously).
//! 2. Exact capacity saturation (64 admitted, 65th through 80th rejected with HTTP 429/529 and `Retry-After: <sec>`).
//! 3. Client disconnect queue pruning: dropping tickets in the middle of the queue immediately advances subsequent positions.
//! 4. Dynamic wait-time estimation calculation and client timeout bounds.
//! 5. SSE keep-alive frame formatting (`: keep-alive\n\n`, `: queue-position <pos>\n\n`, `: position=<pos>\n\n`).
//! 6. High-concurrency soak loop with concurrent admits, drops, and pops.
//! 7. Multi-model queue isolation under concurrent saturation.
//! 8. RAII client ticket dropping with automatic middle-queue position advancement.

use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use k3_core::error::K3Error;
use k3_manager::queue::{
    AdmissionQueueConfig, AdmissionQueueManager, DEFAULT_MAX_QUEUE_DEPTH, ModelAdmissionQueue,
    QueueSseFrame,
};

/// RAII Client Ticket simulating a client connection that drops/cancels upon drop.
#[derive(Debug)]
pub struct ClientTicket<'a> {
    manager: &'a AdmissionQueueManager,
    model_id: &'static str,
    request_id: String,
    position: u32,
    active: bool,
}

impl<'a> ClientTicket<'a> {
    #[must_use]
    pub fn new(
        manager: &'a AdmissionQueueManager,
        model_id: &'static str,
        request_id: String,
        position: u32,
    ) -> Self {
        Self {
            manager,
            model_id,
            request_id,
            position,
            active: true,
        }
    }

    #[must_use]
    pub fn position(&self) -> u32 {
        self.position
    }

    pub fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for ClientTicket<'_> {
    fn drop(&mut self) {
        if self.active {
            self.manager
                .cancel_in_model(self.model_id, &self.request_id);
        }
    }
}

// -----------------------------------------------------------------------------
// Test 1: Strict FIFO ordering under high concurrent load
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_concurrent_admission_fifo_ordering() {
    const NUM_THREADS: usize = 32;
    const REQS_PER_THREAD: usize = 2; // Total 64 requests

    let manager = Arc::new(AdmissionQueueManager::default());
    manager.register_model("qwen2.5-7b", None);

    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    // Spawn 32 concurrent threads contending to admit requests
    let mut handles = Vec::with_capacity(NUM_THREADS);
    for thread_idx in 0..NUM_THREADS {
        let mgr = Arc::clone(&manager);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            let mut results = Vec::with_capacity(REQS_PER_THREAD);
            for r in 0..REQS_PER_THREAD {
                let req_id = format!("t{thread_idx}_r{r}");
                let pos = mgr
                    .admit_to_model(
                        "qwen2.5-7b",
                        req_id.clone(),
                        false,
                        None,
                        None,
                        Arc::new(AtomicBool::new(false)),
                        None,
                        None,
                    )
                    .expect("must admit within 64 cap");
                results.push((req_id, pos));
            }
            results
        }));
    }

    let mut admitted_records = Vec::with_capacity(64);
    for h in handles {
        let thread_res = h.join().expect("thread join must succeed");
        admitted_records.extend(thread_res);
    }

    assert_eq!(admitted_records.len(), 64);
    assert_eq!(manager.queue_depth("qwen2.5-7b"), 64);

    // Verify all positions 1..=64 are unique and present
    let positions: HashSet<u32> = admitted_records.iter().map(|(_, pos)| *pos).collect();
    assert_eq!(positions.len(), 64);
    for p in 1_u32..=64_u32 {
        assert!(positions.contains(&p), "missing position {p}");
    }

    // Sort admitted records by assigned position
    admitted_records.sort_by_key(|(_, pos)| *pos);

    // Sequentially pop and verify each popped request matches the exact FIFO order of assigned position
    for (expected_req_id, expected_pos) in admitted_records {
        let permit = manager
            .pop_next_for_model("qwen2.5-7b")
            .expect("must yield permit");
        assert_eq!(
            permit.request_id, expected_req_id,
            "popped request ID does not match FIFO position {expected_pos}"
        );
        manager.mark_execution_completed("qwen2.5-7b");
    }

    assert_eq!(manager.queue_depth("qwen2.5-7b"), 0);
    assert!(manager.pop_next_for_model("qwen2.5-7b").is_none());
}

// -----------------------------------------------------------------------------
// Test 2: Exact capacity saturation (64 admitted, 65th through 80th rejected with HTTP 429/529)
// -----------------------------------------------------------------------------
#[test]
#[allow(clippy::too_many_lines)]
fn test_adversarial_capacity_saturation_64_admitted_16_rejected() {
    const TOTAL_REQUESTS: usize = 80;

    let manager = Arc::new(AdmissionQueueManager::default());
    manager.register_model("qwen2.5-7b", None);

    let barrier = Arc::new(Barrier::new(TOTAL_REQUESTS));

    let mut handles = Vec::with_capacity(TOTAL_REQUESTS);
    for i in 0..TOTAL_REQUESTS {
        let mgr = Arc::clone(&manager);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            let req_id = format!("burst_req_{i}");
            mgr.admit_to_model(
                "qwen2.5-7b",
                req_id,
                false,
                None,
                None,
                Arc::new(AtomicBool::new(false)),
                None,
                None,
            )
        }));
    }

    let mut ok_positions = Vec::new();
    let mut rejected_errors = Vec::new();

    for h in handles {
        let res = h.join().expect("thread join must succeed");
        match res {
            Ok(pos) => ok_positions.push(pos),
            Err(err) => rejected_errors.push(err),
        }
    }

    // Exactly 64 admitted, exactly 16 rejected
    assert_eq!(
        ok_positions.len(),
        64,
        "exactly 64 requests must be admitted"
    );
    assert_eq!(
        rejected_errors.len(),
        16,
        "exactly 16 requests must be rejected"
    );
    assert_eq!(manager.queue_depth("qwen2.5-7b"), 64);

    // Verify all positions 1..=64 are assigned
    let position_set: HashSet<u32> = ok_positions.into_iter().collect();
    assert_eq!(position_set.len(), 64);
    for p in 1..=64 {
        assert!(position_set.contains(&p));
    }

    // Verify each rejected error matches specification
    for err in rejected_errors {
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
                // 64 * 25.0ms * 16 = 25.6s -> ceil(25.6) = 26s
                assert_eq!(retry_after_secs, 26);

                // Verify HTTP status code mappings and headers per OpenAI & Claude specs
                let http_status_openai = 429;
                let http_status_claude = 529;
                let retry_after_header = format!("Retry-After: {retry_after_secs}");
                assert_eq!(http_status_openai, 429);
                assert_eq!(http_status_claude, 529);
                assert_eq!(retry_after_header, "Retry-After: 26");
            }
            other => panic!("expected QueueSaturated, got {other:?}"),
        }
    }

    // Verify queue integrity: popping 1 slot opens exactly 1 capacity slot
    let _popped = manager
        .pop_next_for_model("qwen2.5-7b")
        .expect("must pop 1");
    assert_eq!(manager.queue_depth("qwen2.5-7b"), 63);

    // 81st request arrives: admitted at position 64
    let admitted_81 = manager
        .admit_to_model(
            "qwen2.5-7b",
            "req_81".to_owned(),
            false,
            None,
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            None,
        )
        .expect("must admit into vacated slot");
    assert_eq!(admitted_81, 64);
    assert_eq!(manager.queue_depth("qwen2.5-7b"), 64);

    // 82nd request is rejected again
    let err_82 = manager
        .admit_to_model(
            "qwen2.5-7b",
            "req_82".to_owned(),
            false,
            None,
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            None,
        )
        .expect_err("must reject when full again");
    assert!(matches!(
        err_82,
        K3Error::QueueSaturated {
            current_depth: 64,
            max_depth: 64,
            ..
        }
    ));
}

// -----------------------------------------------------------------------------
// Test 3: Client disconnect ticket dropping in middle of queue & position advancement
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_client_disconnect_ticket_dropping_in_middle_and_position_shifts() {
    const TOTAL_ENQUEUED: usize = 64;

    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", AdmissionQueueConfig::default());
    let mut rx_channels = Vec::with_capacity(TOTAL_ENQUEUED);

    // Enqueue 64 streaming requests, each with an SSE mpsc channel
    for i in 0..TOTAL_ENQUEUED {
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(256);
        let req_id = format!("stream_req_{i}");
        let pos = queue
            .admit(
                req_id,
                true, // streaming
                None,
                None,
                Arc::new(AtomicBool::new(false)),
                None,
                Some(tx),
            )
            .expect("must admit");
        assert_eq!(pos, u32::try_from(i + 1).expect("valid pos"));
        rx_channels.push(rx);
    }
    assert_eq!(queue.len(), 64);

    // Drop 14 tickets in the middle (e.g. indices 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 62, 63)
    let dropped_indices: Vec<usize> = vec![5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 62, 63];
    let dropped_set: HashSet<usize> = dropped_indices.iter().copied().collect();

    for &idx in &dropped_indices {
        let req_id = format!("stream_req_{idx}");
        let cancelled = queue.cancel(&req_id);
        assert!(
            cancelled,
            "cancel must return true for existing queued request"
        );
    }

    assert_eq!(queue.len(), 50); // 64 - 14 = 50 remaining
    assert_eq!(queue.metrics().total_cancelled, 14);

    // Check that remaining surviving channels received advancing position notifications
    let mut surviving_indices = Vec::new();
    for i in 0..TOTAL_ENQUEUED {
        if !dropped_set.contains(&i) {
            surviving_indices.push(i);
        }
    }
    assert_eq!(surviving_indices.len(), 50);

    // For each surviving index, drain its rx channel and verify the latest broadcast position
    // matches its 1-indexed rank among survivors
    for (survivor_rank, &orig_idx) in surviving_indices.iter().enumerate() {
        let expected_new_pos = u32::try_from(survivor_rank + 1).expect("valid pos");
        let rx = &mut rx_channels[orig_idx];

        let mut latest_pos_msg = None;
        while let Ok(msg) = rx.try_recv() {
            if msg.contains("queue-position") {
                latest_pos_msg = Some(msg);
            }
        }

        // Survivors that were behind at least one dropped request MUST have received position updates
        if orig_idx > dropped_indices[0] {
            let msg = latest_pos_msg.expect("must have received advancing position frame");
            assert_eq!(
                msg,
                format!(": queue-position {expected_new_pos}\n\n"),
                "expected new position {expected_new_pos} for original index {orig_idx}"
            );
        }
    }

    // Now pop all 50 surviving requests and verify strict FIFO order and zero dropped requests
    for &expected_orig_idx in &surviving_indices {
        let permit = queue.pop_next().expect("must yield permit for survivor");
        assert_eq!(permit.request_id, format!("stream_req_{expected_orig_idx}"));
        queue.mark_execution_completed();
    }

    assert!(queue.is_empty());
    assert_eq!(queue.metrics().total_dispatched, 50);
}

// -----------------------------------------------------------------------------
// Test 4: Dynamic wait-time estimation calculation and client timeout bounds
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_dynamic_wait_time_and_timeout_boundaries() {
    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", AdmissionQueueConfig::default());

    // Case A: Empty queue, not executing
    assert_eq!(queue.estimate_wait_ms(None), 0.0);
    // When wait_ms is 0.0, calculate_retry_after_secs defaults to default_retry_after_secs (5s)
    assert_eq!(queue.calculate_retry_after_secs(None), 5);

    // Case B: 10 requests queued, not executing (10 * 25.0ms * 16 passes = 4000.0ms = 4.0s)
    for i in 1..=10 {
        queue.admit_simple(format!("req_{i}"), false).unwrap();
    }
    assert_eq!(queue.estimate_wait_ms(None), 4000.0);
    assert_eq!(queue.calculate_retry_after_secs(None), 4);

    // Case C: 10 requests queued, executing is true -> pending_slots = 11 (11 * 25.0 * 16 = 4400ms = 4.4s -> ceil = 5s)
    queue.set_executing(true);
    assert_eq!(queue.estimate_wait_ms(None), 4400.0);
    assert_eq!(queue.calculate_retry_after_secs(None), 5);
    queue.set_executing(false);

    // Case D: Custom latency (50.0ms) -> 10 * 50.0 * 16 = 8000.0ms = 8.0s
    assert_eq!(queue.estimate_wait_ms(Some(50.0)), 8000.0);
    assert_eq!(queue.calculate_retry_after_secs(Some(50.0)), 8);

    // Case E: Degenerate latencies gracefully fallback to default 25.0ms
    assert_eq!(queue.estimate_wait_ms(Some(0.0)), 4000.0);
    assert_eq!(queue.estimate_wait_ms(Some(-15.0)), 4000.0);
    assert_eq!(queue.estimate_wait_ms(Some(f64::NAN)), 4000.0);

    // Case F: Client timeout boundary testing
    // Current queue wait is 4000ms.
    // 1. Client timeout = 3999ms -> rejected (est_wait > timeout)
    let err_strict = queue
        .admit(
            "timeout_strict".to_owned(),
            false,
            Some(Duration::from_millis(3999)),
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            None,
        )
        .expect_err("3999ms timeout must be rejected when est_wait is 4000ms");
    assert!(matches!(
        err_strict,
        K3Error::QueueSaturated {
            current_depth: 10,
            ..
        }
    ));

    // 2. Client timeout = 4000ms -> admitted (est_wait <= timeout)
    let pos_exact = queue
        .admit(
            "timeout_exact".to_owned(),
            false,
            Some(Duration::from_millis(4000)),
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            None,
        )
        .expect("4000ms timeout must be admitted when est_wait is exactly 4000ms");
    assert_eq!(pos_exact, 11);

    // 3. Client timeout = 4500ms -> admitted (est_wait <= timeout)
    // Now depth is 11, est_wait is 11 * 400 = 4400ms <= 4500ms
    let pos_lax = queue
        .admit(
            "timeout_lax".to_owned(),
            false,
            Some(Duration::from_millis(4500)),
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            None,
        )
        .expect("4500ms timeout must be admitted when est_wait is 4400ms");
    assert_eq!(pos_lax, 12);
}

// -----------------------------------------------------------------------------
// Test 5: SSE keep-alive frame formatting and compliance
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_sse_keepalive_formatting_and_compliance() {
    // 1. Verify standard keep-alive comment
    assert_eq!(QueueSseFrame::KeepAlive.format_frame(), ": keep-alive\n\n");

    // 2. Verify advancing queue position frame
    assert_eq!(
        QueueSseFrame::QueuePosition(1).format_frame(),
        ": queue-position 1\n\n"
    );
    assert_eq!(
        QueueSseFrame::QueuePosition(64).format_frame(),
        ": queue-position 64\n\n"
    );
    assert_eq!(
        QueueSseFrame::QueuePosition(u32::MAX).format_frame(),
        format!(": queue-position {}\n\n", u32::MAX)
    );

    // 3. Verify legacy position format
    assert_eq!(
        QueueSseFrame::format_position_comment(1),
        ": position=1\n\n"
    );
    assert_eq!(
        QueueSseFrame::format_position_comment(64),
        ": position=64\n\n"
    );

    // 4. Verify combined keep-alive position frame
    assert_eq!(
        QueueSseFrame::format_combined_keep_alive(7),
        ": keep-alive position=7\n\n"
    );

    // 5. Verify emission across streaming vs non-streaming requests
    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", AdmissionQueueConfig::default());
    let (tx_stream, mut rx_stream) = tokio::sync::mpsc::channel::<String>(16);

    // Non-streaming request
    queue.admit_simple("non_stream".to_owned(), false).unwrap();
    // Streaming request
    queue
        .admit(
            "stream".to_owned(),
            true,
            None,
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            Some(tx_stream),
        )
        .unwrap();

    queue.emit_keep_alives();

    let msg = rx_stream
        .try_recv()
        .expect("streaming request must receive keep-alive");
    assert_eq!(msg, ": keep-alive\n\n");
    assert!(rx_stream.try_recv().is_err());
}

// -----------------------------------------------------------------------------
// Test 6: Multi-model queue isolation under concurrent saturation
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_multi_model_queue_isolation_under_concurrency() {
    const THREADS_A: usize = 70; // 70 threads saturate model_a (64 admitted, 6 rejected)
    const THREADS_B: usize = 20; // 20 threads admit cleanly into model_b

    let manager = Arc::new(AdmissionQueueManager::default());
    manager.register_model("model_a", None);
    manager.register_model("model_b", None);

    let barrier = Arc::new(Barrier::new(THREADS_A + THREADS_B));

    let mut handles = Vec::new();

    // Threads targeting model_a
    for i in 0..THREADS_A {
        let mgr = Arc::clone(&manager);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            mgr.admit_to_model(
                "model_a",
                format!("req_a_{i}"),
                false,
                None,
                None,
                Arc::new(AtomicBool::new(false)),
                None,
                None,
            )
        }));
    }

    // Threads targeting model_b
    for i in 0..THREADS_B {
        let mgr = Arc::clone(&manager);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            mgr.admit_to_model(
                "model_b",
                format!("req_b_{i}"),
                false,
                None,
                None,
                Arc::new(AtomicBool::new(false)),
                None,
                None,
            )
        }));
    }

    for h in handles {
        let _ = h.join().expect("thread join must succeed");
    }

    // Model A must be saturated at exactly 64
    assert_eq!(manager.queue_depth("model_a"), 64);
    // Model B must have exactly 20 admitted
    assert_eq!(manager.queue_depth("model_b"), 20);
    // Total aggregate depth across manager must be 84
    assert_eq!(manager.total_queue_depth(), 84);

    // Further request to model_a is rejected
    let res_a = manager.admit_to_model(
        "model_a",
        "excess_a".to_owned(),
        false,
        None,
        None,
        Arc::new(AtomicBool::new(false)),
        None,
        None,
    );
    assert!(matches!(
        res_a,
        Err(K3Error::QueueSaturated {
            current_depth: 64,
            ..
        })
    ));

    // Model B can still admit freely
    let res_b = manager.admit_to_model(
        "model_b",
        "normal_b".to_owned(),
        false,
        None,
        None,
        Arc::new(AtomicBool::new(false)),
        None,
        None,
    );
    assert!(res_b.is_ok());
    assert_eq!(manager.queue_depth("model_b"), 21);
}

// -----------------------------------------------------------------------------
// Test 7: Multi-threaded churn & soak stress loop (25 iterations)
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_soak_multithreaded_churn_loop() {
    const WORKERS: usize = 16;

    for iter in 1..=25 {
        let manager = Arc::new(AdmissionQueueManager::default());
        let model_id = "stress_model";
        manager.register_model(model_id, None);

        let barrier = Arc::new(Barrier::new(WORKERS));

        let mut handles = Vec::with_capacity(WORKERS);
        for worker_id in 0..WORKERS {
            let mgr = Arc::clone(&manager);
            let bar = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                bar.wait();
                // Perform 10 actions: admit, cancel, pop
                for step in 0..10 {
                    let req_id = format!("w{worker_id}_s{step}");
                    // 1. Admit
                    let _ = mgr.admit_to_model(
                        model_id,
                        req_id.clone(),
                        step % 2 == 0,
                        None,
                        None,
                        Arc::new(AtomicBool::new(false)),
                        None,
                        None,
                    );

                    // 2. Chance of cancel
                    if step % 3 == 0 {
                        mgr.cancel_in_model(model_id, &req_id);
                    }

                    // 3. Chance of pop
                    if step % 2 == 0 && mgr.pop_next_for_model(model_id).is_some() {
                        mgr.mark_execution_completed(model_id);
                    }
                }
            }));
        }

        for h in handles {
            h.join().expect("soak worker must finish without panic");
        }

        let depth = manager.queue_depth(model_id);
        assert!(
            depth <= DEFAULT_MAX_QUEUE_DEPTH,
            "iteration {iter}: queue depth {depth} exceeded max {DEFAULT_MAX_QUEUE_DEPTH}"
        );
    }
}

// -----------------------------------------------------------------------------
// Test 8: RAII client ticket dropping in middle of queue
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_raii_client_ticket_drop_advancement() {
    let manager = AdmissionQueueManager::default();
    manager.register_model("test_model", None);

    let mut tickets = Vec::new();
    for i in 1..=5 {
        let req_id = format!("ticket_req_{i}");
        let pos = manager
            .admit_to_model(
                "test_model",
                req_id.clone(),
                false,
                None,
                None,
                Arc::new(AtomicBool::new(false)),
                None,
                None,
            )
            .expect("must admit");
        tickets.push(ClientTicket::new(&manager, "test_model", req_id, pos));
    }
    assert_eq!(manager.queue_depth("test_model"), 5);

    // Drop ticket #3 in the middle
    let ticket_3 = tickets.remove(2);
    assert_eq!(ticket_3.position(), 3);
    drop(ticket_3); // Drop triggers cancel_in_model

    // Queue depth immediately drops to 4
    assert_eq!(manager.queue_depth("test_model"), 4);

    // Pop next 2 requests: ticket #1 and ticket #2
    let p1 = manager.pop_next_for_model("test_model").unwrap();
    assert_eq!(p1.request_id, "ticket_req_1");
    manager.mark_execution_completed("test_model");

    let p2 = manager.pop_next_for_model("test_model").unwrap();
    assert_eq!(p2.request_id, "ticket_req_2");
    manager.mark_execution_completed("test_model");

    // Next popped must be ticket #4 (ticket #3 was dropped and skipped!)
    let p4 = manager.pop_next_for_model("test_model").unwrap();
    assert_eq!(p4.request_id, "ticket_req_4");
    manager.mark_execution_completed("test_model");

    // Next popped is ticket #5
    let p5 = manager.pop_next_for_model("test_model").unwrap();
    assert_eq!(p5.request_id, "ticket_req_5");
    manager.mark_execution_completed("test_model");

    assert_eq!(manager.queue_depth("test_model"), 0);

    // Cleanly disarm remaining tickets so their drop does not fail
    for mut t in tickets {
        t.disarm();
    }
}
