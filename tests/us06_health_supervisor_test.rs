//! Unit tests for US-06.3: Deadline-bounded UDS I/O & Health Supervisor (#75, #76).

#![allow(
    clippy::assert_is_empty,
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use std::mem::size_of;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::ipc::layout::ArenaHeader;
use k3_core::ipc::protocol::{
    IpcMessage, WorkerHeartbeatResponse, recv_msg_frame, send_msg_frame,
};
use k3_core::ipc::shm::SharedMemoryArena;
use k3_manager::domain::MockOsAffinityBackend;
use k3_manager::{
    ControlPlaneRuntime, ForwardPassRequest, HealthSupervisor, ManagedWorker, ManagerError,
    RecoveryAction, RecoveryTarget, SimpleTokenizer, WorkerPlacement,
};

fn init_dummy_arena(name: &str) -> SharedMemoryArena {
    let mut arena = SharedMemoryArena::create(name, 65536).expect("create arena");
    let hdr = ArenaHeader {
        total_bytes: 65536,
        n_ctx: 64,
        token_slots_offset: 1024,
        output_slabs_offset: 2048,
        vocab_size: 10,
        ..Default::default()
    };
    let slice = arena.slice_mut(0, 128).expect("slice hdr");
    // SAFETY: Transmuting valid aligned ArenaHeader struct to its byte representation
    let hdr_bytes = unsafe {
        std::slice::from_raw_parts(
            (&raw const hdr).cast::<u8>(),
            size_of::<ArenaHeader>(),
        )
    };
    slice[..hdr_bytes.len()].copy_from_slice(hdr_bytes);
    arena
}

// ============================================================================
// US-06.3.1 (#75): Deadline-Bounded Socket I/O Unit Tests
// ============================================================================

#[test]
fn test_sub_issue_75_heartbeat_timeout_returns_worker_unresponsive() {
    let (mgr_sock, dummy_worker) = UnixStream::pair().expect("socketpair");
    let arena = init_dummy_arena("shm_hb_timeout");

    let placement = WorkerPlacement::new("test-hb", "/dummy", "/dummy.sock", "x100")
        .with_heartbeat_timeout(Duration::from_millis(50));

    let mut worker = ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 65536, 64, 4);

    // Dummy worker reads the request but intentionally NEVER replies
    let worker_read_handle = std::thread::spawn(move || {
        let (msg, _) = recv_msg_frame(&dummy_worker).expect("dummy receive");
        assert!(matches!(msg, IpcMessage::WorkerHeartbeatRequest(_)));
        // Do not reply, hold socket open to simulate wedged worker
        std::thread::sleep(Duration::from_millis(200));
        let _ = dummy_worker;
    });

    let res = worker.send_heartbeat();
    assert!(res.is_err(), "heartbeat must fail on unresponsive socket");

    match res.unwrap_err() {
        ManagerError::WorkerUnresponsive {
            model_id,
            pid,
            elapsed_ms,
        } => {
            assert_eq!(model_id, "test-hb");
            assert!(pid.is_none());
            assert!(
                elapsed_ms >= 30,
                "elapsed_ms must reflect deadline bound (observed {elapsed_ms}ms)"
            );
        }
        other => panic!("expected WorkerUnresponsive, got {other:?}"),
    }

    assert!(worker.has_timed_out());
    let _ = worker_read_handle.join();
}

#[test]
fn test_sub_issue_75_forward_pass_timeout_returns_worker_unresponsive() {
    let (mgr_sock, dummy_worker) = UnixStream::pair().expect("socketpair");
    let arena = init_dummy_arena("shm_pass_timeout");

    let placement = WorkerPlacement::new("test-pass", "/dummy", "/dummy.sock", "x100")
        .with_pass_timeout(Duration::from_millis(50));

    let mut worker = ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 65536, 64, 4);

    // Dummy worker receives the ForwardPassCommand but stalls without replying
    let worker_read_handle = std::thread::spawn(move || {
        let (msg, _) = recv_msg_frame(&dummy_worker).expect("dummy receive");
        assert!(matches!(msg, IpcMessage::ForwardPassCommand(_)));
        std::thread::sleep(Duration::from_millis(200));
        let _ = dummy_worker;
    });

    let req = ForwardPassRequest::new_prefill(1001, 1, vec![1, 2, 3], vec![0]);
    let res = worker.execute_forward_pass(&req);
    assert!(res.is_err(), "forward pass must fail on unresponsive socket");

    match res.unwrap_err() {
        ManagerError::WorkerUnresponsive {
            model_id,
            pid,
            elapsed_ms,
        } => {
            assert_eq!(model_id, "test-pass");
            assert!(pid.is_none());
            assert!(
                elapsed_ms >= 30,
                "elapsed_ms must reflect deadline bound (observed {elapsed_ms}ms)"
            );
        }
        other => panic!("expected WorkerUnresponsive, got {other:?}"),
    }

    assert!(worker.has_timed_out());
    assert!(!worker.is_pass_in_flight());
    let _ = worker_read_handle.join();
}

#[test]
fn test_sub_issue_75_successful_heartbeat_records_latency_and_timestamp() {
    let (mgr_sock, dummy_worker) = UnixStream::pair().expect("socketpair");
    let arena = init_dummy_arena("shm_hb_success");

    let placement = WorkerPlacement::new("test-ok", "/dummy", "/dummy.sock", "x100")
        .with_heartbeat_timeout(Duration::from_millis(500));

    let mut worker = ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 65536, 64, 4);

    let dummy_thread = std::thread::spawn(move || {
        let (msg, _) = recv_msg_frame(&dummy_worker).expect("receive");
        if let IpcMessage::WorkerHeartbeatRequest(req) = msg {
            let resp = IpcMessage::WorkerHeartbeatResponse(WorkerHeartbeatResponse {
                timestamp_ns: req.timestamp_ns,
                is_busy: false,
                completed_passes: 42,
            });
            send_msg_frame(&dummy_worker, &resp, None).expect("send");
        }
    });

    let resp = worker.send_heartbeat().expect("heartbeat should succeed");
    assert_eq!(resp.completed_passes, 42);
    assert!(!worker.has_timed_out());
    assert!(worker.last_heartbeat_latency().is_some());
    assert!(worker.last_heartbeat_time().is_some());

    let _ = dummy_thread.join();
}

// ============================================================================
// US-06.3.2 (#76): Dedicated Spare Recovery Watchdog Thread Unit Tests
// ============================================================================

#[test]
fn test_sub_issue_76_watchdog_thread_affinity_pinning_spare_mask() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");

    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let mut supervisor = HealthSupervisor::with_affinity_backend(
        capacity,
        Arc::clone(&mock_affinity) as Arc<dyn k3_manager::OsAffinityBackend>,
    )
    .expect("create health supervisor");

    assert_eq!(supervisor.spare_recovery_mask(), 0x0020);
    assert_eq!(supervisor.spare_recovery_core(), Some(5));
    assert!(!supervisor.is_watchdog_running());

    supervisor.start_watchdog().expect("start watchdog");
    assert!(supervisor.is_watchdog_running());

    // Verify watchdog thread executed affinity pinning to spare_recovery_mask (0x0020)
    let affinities = mock_affinity.calls();
    assert!(
        affinities.iter().any(|&(_pid, mask)| mask == 0x0020),
        "watchdog thread must pin to spare_recovery_mask 0x0020, recorded: {affinities:?}"
    );

    supervisor.stop_watchdog();
    assert!(!supervisor.is_watchdog_running());
}

#[test]
fn test_sub_issue_76_runaway_control_task_lease_preemption_and_thread_replacement() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");

    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = Arc::new(
        ControlPlaneRuntime::with_affinity_backend(
            capacity.clone(),
            Arc::clone(&mock_affinity) as Arc<dyn k3_manager::OsAffinityBackend>,
        )
        .expect("runtime creation"),
    );

    let mut health_supervisor = HealthSupervisor::with_affinity_backend(
        capacity,
        Arc::clone(&mock_affinity) as Arc<dyn k3_manager::OsAffinityBackend>,
    )
    .expect("health supervisor")
    .with_control_runtime(Arc::clone(&runtime))
    .with_task_lease_duration(Duration::from_millis(50));

    health_supervisor.start_watchdog().expect("start watchdog");

    // Register simple tokenizer
    let mut tok = SimpleTokenizer::new();
    tok.add_token("hello", 1);
    runtime.register_tokenizer("m1", Arc::new(tok));

    // 1. Dispatch a runaway control-plane task with a 50ms lease that simulates an infinite loop
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let cancel_clone = Arc::clone(&cancel_flag);

    let runaway_res: Result<(), ManagerError> = runtime.dispatch_cancellable_task_with_lease(
        Duration::from_millis(50),
        move |token, _core| {
            while !token.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(5));
            }
            cancel_clone.store(true, Ordering::SeqCst);
        },
    );

    // Assert that the lease expired and preemption error was returned
    assert!(runaway_res.is_err(), "runaway task must be preempted");
    match runaway_res.unwrap_err() {
        ManagerError::WorkerUnresponsive {
            model_id,
            elapsed_ms,
            ..
        } => {
            assert_eq!(model_id, "control-plane");
            assert!(
                elapsed_ms >= 40,
                "elapsed time should reflect lease deadline (observed {elapsed_ms}ms)"
            );
        }
        other => panic!("expected WorkerUnresponsive, got {other:?}"),
    }

    assert!(
        cancel_flag.load(Ordering::SeqCst),
        "cooperative cancellation token must be set"
    );

    // 2. Verify that control plane responsiveness is immediately restored on that core
    let post_recovery_tokens = runtime
        .tokenize_prompt("m1", "hello", false, 64)
        .expect("tokenization must immediately succeed after worker thread replacement");
    assert_eq!(post_recovery_tokens, vec![1]);

    // 3. Verify audit events recorded
    let events = health_supervisor.audit_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e.target, RecoveryTarget::ControlTask { .. })),
        "audit event must record control task preemption"
    );

    health_supervisor.stop_watchdog();
}

#[test]
fn test_sub_issue_76_audit_events_recording_and_clearing() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");

    let supervisor = HealthSupervisor::new(capacity).expect("health supervisor");

    assert!(supervisor.audit_events().is_empty());

    let event = k3_manager::RecoveryAuditEvent {
        timestamp: std::time::Instant::now(),
        target: RecoveryTarget::ControlTask {
            task_id: 101,
            core_id: 6,
        },
        reason: "task lease deadline breached (500ms limit)".to_string(),
        action: RecoveryAction::ControlTaskPreempted {
            task_id: 101,
            thread_replaced: true,
        },
    };

    let disp = format!("{event}");
    assert!(disp.contains("RECOVERY: control task 101 on core 6"));
    assert!(disp.contains("thread_replaced: true"));

    supervisor.clear_audit_events();
    assert!(supervisor.audit_events().is_empty());
}
