//! US-06.4.2 (#78): US-06 Scenario 2 E2E Integration Test: Runaway Workload Preemption & Recovery via Spare Capacity.
//!
//! Validates both fault modes under `HealthSupervisor` on the reserved spare recovery core (`0x0020` / core 5):
//! 1. Runaway Compute Worker:
//!    - Trigger a simulated runaway / unresponsive stall (`SIGSTOP`) inside an active `k3-worker` process
//!      while `k3-kv` holds committed context blocks.
//!    - Verify that the spare recovery thread detects the missed deadline, preempts/terminates the runaway
//!      worker (`SIGTERM` -> `SIGKILL`), reaps the dead process, spawns a clean replacement worker re-attached
//!      to the surviving `k3-kv` arena via `SCM_RIGHTS`, and resumes inference without board reboot.
//! 2. Runaway Control Task:
//!    - Trigger a simulated runaway loop inside an active control-plane task.
//!    - Verify that the spare recovery watchdog detects the expired task lease, preempts/aborts the stalled task,
//!      replaces the wedged worker thread, and restores immediate control-plane responsiveness.

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]

use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::ipc::layout::KvBlockTag;
use k3_kv::server::{KvServer, ServerConfig};
use k3_manager::client::KvClient;
use k3_manager::dispatch::ForwardPassRequest;
use k3_manager::domain::{CgroupV2DomainController, MockOsAffinityBackend};
use k3_manager::supervisor::{WorkerPlacement, WorkerSupervisor};
use k3_manager::{
    ControlPlaneRuntime, HealthSupervisor, ManagerError, RecoveryAction, RecoveryTarget,
    SimpleTokenizer,
};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(400);

fn unique_socket_path(prefix: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let seq = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{pid}_{seq}.sock"))
}

fn create_dummy_weights(name: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let seq = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("{name}_{pid}_{seq}.gguf"));
    let mut f = File::create(&path).expect("create dummy weights file");
    f.write_all(b"GGUF_TEST_SYNTHETIC_MODEL_WEIGHTS_S2")
        .expect("write dummy weights");
    path
}

#[test]
fn test_scenario2_runaway_workload_recovery_e2e() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // ------------------------------------------------------------------------
    // Step 1: Start standalone k3-kv server and allocate shared context arena
    // ------------------------------------------------------------------------
    let kv_sock_path = unique_socket_path("k3_kv_rec_s2");
    let kv_config = ServerConfig {
        socket_path: kv_sock_path.clone(),
        max_memory_bytes: 1024 * 1024 * 1024,
    };
    let kv_server = KvServer::new(kv_config);

    let _kv_task = rt.spawn(async move {
        let _ = kv_server.run().await;
    });

    // Wait for k3-kv listener to become active
    let mut kv_client = None;
    for _ in 0..50 {
        if let Ok(client) = KvClient::connect(&kv_sock_path) {
            kv_client = Some(client);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut kv_client = kv_client.expect("connect to k3-kv");
    assert!(kv_client.ping().expect("kv ping") > 0);

    let arena_bytes = 65536_u64;
    let n_ctx = 64_u32;
    let max_blocks = 4_u32;
    let model_id = "model-runaway-s2";

    let (alloc_resp, arena) = kv_client
        .allocate_and_attach(model_id, arena_bytes, n_ctx, max_blocks)
        .expect("allocate and attach arena from k3-kv");

    // ------------------------------------------------------------------------
    // Step 2: Configure capacity, mock affinity, WorkerSupervisor, and initial worker
    // ------------------------------------------------------------------------
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid control capacity");
    assert_eq!(capacity.spare_recovery_mask(), 0x0020); // core 5
    assert_eq!(capacity.active_control_mask(), 0x00C0); // cores 6-7

    let mock_affinity = Arc::new(MockOsAffinityBackend::default());

    let domain_ctrl = Arc::new(CgroupV2DomainController::default().with_affinity_backend(
        Arc::clone(&mock_affinity) as Arc<dyn k3_manager::OsAffinityBackend>,
    ));
    let mut supervisor = WorkerSupervisor::new()
        .with_capacity(capacity.clone())
        .with_domain_controller(domain_ctrl)
        .with_default_heartbeat_timeout(Duration::from_millis(50))
        .with_default_pass_timeout(Duration::from_millis(150));

    let weights_path = create_dummy_weights("weights_rec_s2");
    let worker_sock = unique_socket_path("worker_rec_s2");
    let placement = WorkerPlacement::new(model_id, &weights_path, &worker_sock, "x100")
        .with_heartbeat_timeout(Duration::from_millis(50))
        .with_pass_timeout(Duration::from_millis(150));

    let initial_worker = supervisor
        .spawn_worker(
            placement,
            arena,
            alloc_resp.arena_id,
            alloc_resp.arena_bytes as usize,
            n_ctx,
            max_blocks,
        )
        .expect("spawn initial worker");

    assert!(initial_worker.is_alive());
    let initial_pid = initial_worker.pid().expect("initial worker PID");
    assert_eq!(initial_worker.generation(), 1);

    // Commit context blocks in initial worker before fault injection
    let prompt_tokens = vec![101_u32, 102, 103, 104, 105];
    let prefill_req = ForwardPassRequest::new_prefill(9001, 1, prompt_tokens, vec![0]);
    let prefill_res = initial_worker
        .execute_forward_pass(&prefill_req)
        .expect("initial prefill pass");
    assert!(prefill_res.completed.status.is_success());
    assert_eq!(prefill_res.completed.committed_tokens, 5);

    // Verify block 0 committed tokens in KvBlockMeta
    let block0_meta = initial_worker.get_block_meta(0).expect("read block 0");
    assert_eq!(block0_meta.committed_tokens(), 5);
    assert_eq!(block0_meta.tag_enum(), Some(KvBlockTag::Persist));

    // ------------------------------------------------------------------------
    // Step 3: Create ControlPlaneRuntime & HealthSupervisor pinned to spare core 5
    // ------------------------------------------------------------------------
    let control_runtime = Arc::new(
        ControlPlaneRuntime::with_affinity_backend(
            capacity.clone(),
            Arc::clone(&mock_affinity) as Arc<dyn k3_manager::OsAffinityBackend>,
        )
        .expect("create control runtime"),
    );

    let mut tok = SimpleTokenizer::new();
    tok.add_token("hello", 1);
    tok.add_token("recovery", 2);
    control_runtime.register_tokenizer("model-control", Arc::new(tok));
    control_runtime.record_model_placement(
        model_id.to_owned(),
        "x100".to_owned(),
        Some(initial_pid),
    );

    let ws_arc = Arc::new(Mutex::new(supervisor));

    let mut health_supervisor = HealthSupervisor::with_affinity_backend(
        capacity.clone(),
        Arc::clone(&mock_affinity) as Arc<dyn k3_manager::OsAffinityBackend>,
    )
    .expect("create health supervisor")
    .with_check_interval(Duration::from_millis(20))
    .with_task_lease_duration(Duration::from_millis(50))
    .with_heartbeat_timeout(Duration::from_millis(50))
    .with_pass_timeout(Duration::from_millis(150))
    .with_worker_supervisor(Arc::clone(&ws_arc))
    .with_control_runtime(Arc::clone(&control_runtime));

    // Start dedicated watchdog thread
    health_supervisor.start_watchdog().expect("start watchdog");
    assert!(health_supervisor.is_watchdog_running());

    // Verify watchdog pinned strictly to spare_recovery_mask 0x0020 (core 5)
    let affinities = mock_affinity.calls();
    assert!(
        affinities.iter().any(|&(_pid, mask)| mask == 0x0020),
        "watchdog thread must be pinned to spare recovery core 0x0020, recorded: {affinities:?}"
    );

    // ------------------------------------------------------------------------
    // Step 4: Fault Mode 1: Runaway Compute Worker Recovery
    // ------------------------------------------------------------------------
    // Simulate runaway unresponsive compute worker by freezing its process with SIGSTOP
    {
        let mut ws = ws_arc.lock().expect("lock supervisor");
        let worker = ws.get_worker_mut(model_id).expect("get worker");
        worker.kill(libc::SIGSTOP).expect("send SIGSTOP");
    }

    // Wait for watchdog health check ticks to detect unresponsive worker, terminate, reap, and respawn
    let recovery_deadline = Instant::now() + Duration::from_millis(1500);
    let mut worker_recovered = false;
    let mut new_worker_pid = None;

    while Instant::now() < recovery_deadline {
        std::thread::sleep(Duration::from_millis(30));
        let events = health_supervisor.audit_events();
        for event in events {
            if let RecoveryAction::WorkerTerminatedAndRespawned {
                old_pid,
                new_pid,
                generation,
            } = event.action
            {
                if old_pid == Some(initial_pid) && generation >= 2 {
                    worker_recovered = true;
                    new_worker_pid = new_pid;
                    assert!(
                        event.reason.contains("timed out")
                            || event.reason.contains("deadline")
                            || event.reason.contains("heartbeat"),
                        "audit event reason must indicate heartbeat deadline timeout, observed: '{}'",
                        event.reason
                    );
                    break;
                }
            }
        }
        if worker_recovered {
            break;
        }
    }

    assert!(
        worker_recovered,
        "watchdog must detect runaway worker, terminate, and respawn replacement worker"
    );

    // Verify supervisor state and replacement worker health
    {
        let mut ws = ws_arc.lock().expect("lock supervisor");
        assert!(ws.is_worker_alive(model_id));
        let rep_worker = ws.get_worker_mut(model_id).expect("replacement worker");
        assert_eq!(rep_worker.generation(), 2);
        assert_ne!(rep_worker.pid(), Some(initial_pid));
        assert_eq!(rep_worker.pid(), new_worker_pid);

        // Verify control-plane status plane reflects replacement worker PID and generation 2
        let health_snaps = control_runtime.worker_health_snapshots();
        let model_snap = health_snaps
            .iter()
            .find(|s| s.model_id == model_id)
            .expect("model health snapshot present");
        assert!(model_snap.is_alive, "model snapshot must report alive");
        assert_eq!(
            model_snap.pid, new_worker_pid,
            "model snapshot PID must be updated to new replacement worker PID"
        );
        assert_eq!(
            model_snap.generation, 2,
            "model snapshot generation must be incremented to 2"
        );

        // Verify surviving k3-kv arena context is preserved intact
        let block0_after_recovery = rep_worker.get_block_meta(0).expect("read block 0");
        assert_eq!(
            block0_after_recovery.committed_tokens(),
            5,
            "committed context tokens must survive worker crash and recovery intact"
        );
        assert_eq!(
            block0_after_recovery.tag_enum(),
            Some(KvBlockTag::Persist),
            "retention tag must survive worker crash"
        );

        // Resume inference without board reboot: execute subsequent decode forward pass
        let decode_req = ForwardPassRequest::new_decode(9002, 1, 106, 5, vec![0]);
        let decode_res = rep_worker
            .execute_forward_pass(&decode_req)
            .expect("resume decode pass on recovered worker");
        assert!(decode_res.completed.status.is_success());
        assert_eq!(decode_res.completed.request_id, 9002);
    }

    // ------------------------------------------------------------------------
    // Step 5: Fault Mode 2: Runaway Control Task Preemption & Thread Replacement
    // ------------------------------------------------------------------------
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let cancel_clone = Arc::clone(&cancel_flag);

    // Dispatch a runaway control-plane task with a 50ms lease that simulates an infinite loop
    let runaway_res: Result<(), ManagerError> = control_runtime
        .dispatch_cancellable_task_with_lease(Duration::from_millis(50), move |token, _core| {
            while !token.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(5));
            }
            cancel_clone.store(true, Ordering::SeqCst);
        });

    // Assert that the lease expired and preemption error was returned
    assert!(
        runaway_res.is_err(),
        "runaway control task must be preempted"
    );
    match runaway_res.unwrap_err() {
        ManagerError::WorkerUnresponsive {
            model_id: err_model,
            elapsed_ms,
            ..
        } => {
            assert_eq!(err_model, "control-plane");
            assert!(
                elapsed_ms >= 40,
                "elapsed time should reflect lease deadline (observed {elapsed_ms}ms)"
            );
        }
        other => panic!("expected WorkerUnresponsive, got {other:?}"),
    }

    assert!(
        cancel_flag.load(Ordering::SeqCst),
        "cooperative cancellation token must have been set by preemption"
    );

    // Verify immediate restored control plane responsiveness on that core
    let post_recovery_tokens = control_runtime
        .tokenize_prompt("model-control", "hello recovery", false, 64)
        .expect("tokenization must immediately succeed after worker thread replacement");
    assert_eq!(post_recovery_tokens, vec![1, 2]);

    // ------------------------------------------------------------------------
    // Step 6: Verify Structured Audit Events
    // ------------------------------------------------------------------------
    let all_events = health_supervisor.audit_events();
    let has_worker_event = all_events
        .iter()
        .any(|e| matches!(e.target, RecoveryTarget::ComputeWorker { .. }));
    let has_task_event = all_events
        .iter()
        .any(|e| matches!(e.target, RecoveryTarget::ControlTask { .. }));

    assert!(
        has_worker_event,
        "audit trail must contain compute worker recovery event"
    );
    assert!(
        has_task_event,
        "audit trail must contain control task preemption event"
    );

    // ------------------------------------------------------------------------
    // Step 7: Clean Shutdown
    // ------------------------------------------------------------------------
    health_supervisor.stop_watchdog();
    assert!(!health_supervisor.is_watchdog_running());

    let mut ws = ws_arc.lock().expect("lock supervisor");
    ws.shutdown_all();
}
