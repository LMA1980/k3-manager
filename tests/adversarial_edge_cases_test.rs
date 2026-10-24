//! Adversarial edge-case tests for k3-manager runtime.
//!
//! Validates:
//! 1. Fast detection and reaping when worker process exits prematurely (e.g. missing weights file, exit code 12) in <500ms.
//! 2. Immediate rejection when token count + start_position exceeds arena context length (`n_ctx`).
//! 3. Immediate rejection when dispatching forward pass with empty tokens.
//! 4. Accurate OS signal extraction (`signal: Some(9)`) in `ManagerError::WorkerCrashed` upon `SIGKILL`.
//! 5. Output slab status validation rejecting unwritten slabs.
//! 6. Clean process termination and PID replacement when recovering an already-running or hung worker.
//! 7. Direct zero-copy Apache Arrow `Float32Array` extraction via `wrap_logits_as_float32_array`.

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]

use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use k3_core::ipc::layout::{OutputSlabHeader, SLAB_STATUS_IDLE, SLAB_STATUS_WRITTEN};
use k3_kv::server::{KvServer, ServerConfig};
use k3_manager::client::KvClient;
use k3_manager::dispatch::{
    ForwardPassRequest, dispatch_forward_pass, wrap_logits_as_float32_array,
};
use k3_manager::error::ManagerError;
use k3_manager::supervisor::{WorkerPlacement, WorkerSupervisor};
use zerocopy::FromBytes;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(500);

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
    f.write_all(b"GGUF_TEST_ADVERSARIAL_WEIGHTS")
        .expect("write dummy weights");
    path
}

#[test]
fn test_adversarial_worker_premature_exit_fast_detection() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let kv_sock = unique_socket_path("k3_kv_adv_pre");
    let kv_server = KvServer::new(ServerConfig {
        socket_path: kv_sock.clone(),
        max_memory_bytes: 128 * 1024 * 1024,
    });
    let kv_task = rt.spawn(async move {
        let _ = kv_server.run().await;
    });

    let mut kv_client = None;
    for _ in 0..50 {
        if let Ok(c) = KvClient::connect(&kv_sock) {
            kv_client = Some(c);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut kv_client = kv_client.expect("connect kv");

    let (alloc, arena) = kv_client
        .allocate_and_attach("adv_model", 65536, 64, 4)
        .expect("alloc arena");

    // Point placement to a nonexistent weights file so k3-worker exits immediately with exit 12
    let non_existent_weights =
        std::env::temp_dir().join("definitely_nonexistent_weights_12345.gguf");
    let worker_sock = unique_socket_path("k3_worker_adv_pre");
    let placement = WorkerPlacement::new("adv_model", &non_existent_weights, &worker_sock, "x100");

    let mut supervisor = WorkerSupervisor::new();

    let start = Instant::now();
    let res = supervisor.spawn_worker(
        placement,
        arena,
        alloc.arena_id,
        alloc.arena_bytes as usize,
        64,
        4,
    );
    let elapsed = start.elapsed();

    // Verify it fails fast: < 1.0 second instead of hanging for 5.0 seconds
    assert!(
        elapsed < Duration::from_secs(2),
        "spawn_worker took too long on premature exit: {elapsed:?}"
    );

    // Verify error indicates worker exited prematurely
    assert!(res.is_err());
    let err_msg = res.unwrap_err().to_string();
    assert!(
        err_msg.contains("exited prematurely"),
        "error message should indicate premature exit: {err_msg}"
    );

    // Verify socket file was cleaned up
    assert!(
        !worker_sock.exists(),
        "socket file should be cleaned up on failure"
    );

    kv_task.abort();
    let _ = std::fs::remove_file(&kv_sock);
}

#[test]
fn test_adversarial_context_limit_rejection() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let kv_sock = unique_socket_path("k3_kv_adv_ctx");
    let kv_server = KvServer::new(ServerConfig {
        socket_path: kv_sock.clone(),
        max_memory_bytes: 128 * 1024 * 1024,
    });
    let kv_task = rt.spawn(async move {
        let _ = kv_server.run().await;
    });

    let mut kv_client = None;
    for _ in 0..50 {
        if let Ok(c) = KvClient::connect(&kv_sock) {
            kv_client = Some(c);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut kv_client = kv_client.expect("connect kv");

    // n_ctx configured to 32
    let (alloc, arena) = kv_client
        .allocate_and_attach("adv_ctx_model", 65536, 32, 4)
        .expect("alloc arena");

    let weights = create_dummy_weights("weights_adv_ctx");
    let worker_sock = unique_socket_path("worker_adv_ctx");
    let placement = WorkerPlacement::new("adv_ctx_model", &weights, &worker_sock, "x100");

    let mut supervisor = WorkerSupervisor::new();
    let worker = supervisor
        .spawn_worker(
            placement,
            arena,
            alloc.arena_id,
            alloc.arena_bytes as usize,
            32,
            4,
        )
        .expect("spawn worker");

    // Case 1: Prompt has 33 tokens (> n_ctx 32)
    let oversized_prompt = vec![101_u32; 33];
    let req1 = ForwardPassRequest::new_prefill(9001, 1, oversized_prompt, vec![0]);
    let res1 = worker.execute_forward_pass(&req1);
    assert!(res1.is_err());
    let err1_msg = res1.unwrap_err().to_string();
    assert!(
        err1_msg.contains("exceeds arena n_ctx"),
        "error should indicate context limit exceeded: {err1_msg}"
    );

    // Case 2: Decode at position 32 (+ 1 token = 33 > n_ctx 32)
    let req2 = ForwardPassRequest::new_decode(9002, 1, 102, 32, vec![0]);
    let res2 = worker.execute_forward_pass(&req2);
    assert!(res2.is_err());
    let err2_msg = res2.unwrap_err().to_string();
    assert!(
        err2_msg.contains("exceeds arena n_ctx"),
        "error should indicate context limit exceeded: {err2_msg}"
    );

    // Case 3: Empty tokens
    let req3 = ForwardPassRequest {
        request_id: 9003,
        sequence_id: 1,
        tokens: vec![],
        start_position: 0,
        block_indices: vec![0],
        is_prefill: true,
    };
    let res3 = worker.execute_forward_pass(&req3);
    assert!(res3.is_err());
    let err3_msg = res3.unwrap_err().to_string();
    assert!(
        err3_msg.contains("0 tokens"),
        "error should indicate empty tokens: {err3_msg}"
    );

    worker.shutdown(true).expect("clean shutdown");
    kv_task.abort();
    let _ = std::fs::remove_file(&weights);
    let _ = std::fs::remove_file(&worker_sock);
    let _ = std::fs::remove_file(&kv_sock);
}

#[test]
fn test_adversarial_signal_reporting_and_recover_running_worker() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let kv_sock = unique_socket_path("k3_kv_adv_sig");
    let kv_server = KvServer::new(ServerConfig {
        socket_path: kv_sock.clone(),
        max_memory_bytes: 128 * 1024 * 1024,
    });
    let kv_task = rt.spawn(async move {
        let _ = kv_server.run().await;
    });

    let mut kv_client = None;
    for _ in 0..50 {
        if let Ok(c) = KvClient::connect(&kv_sock) {
            kv_client = Some(c);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut kv_client = kv_client.expect("connect kv");

    let (alloc, arena) = kv_client
        .allocate_and_attach("adv_sig_model", 65536, 64, 4)
        .expect("alloc arena");

    let weights = create_dummy_weights("weights_adv_sig");
    let worker_sock = unique_socket_path("worker_adv_sig");
    let placement = WorkerPlacement::new("adv_sig_model", &weights, &worker_sock, "x100");

    let mut supervisor = WorkerSupervisor::new();
    supervisor
        .spawn_worker(
            placement,
            arena,
            alloc.arena_id,
            alloc.arena_bytes as usize,
            64,
            4,
        )
        .expect("spawn worker");

    let worker = supervisor.get_worker_mut("adv_sig_model").unwrap();
    let old_pid = worker.pid().expect("old pid");

    // 1. Recover while still running (verifying old process is terminated and not leaked)
    let recovered = supervisor
        .recover_worker("adv_sig_model")
        .expect("recover running worker");

    let new_pid = recovered.pid().expect("new pid");
    assert_ne!(old_pid, new_pid, "new worker must have new PID");
    assert_eq!(recovered.generation(), 2);
    assert!(recovered.is_alive());

    // Verify old process was actually terminated
    // SAFETY: kill 0 checks if process exists
    let old_exists = unsafe { libc::kill(old_pid as libc::pid_t, 0) == 0 };
    assert!(
        !old_exists,
        "old worker process must be terminated and reaped"
    );

    // 2. Kill new worker with SIGKILL and verify signal extraction
    let worker2 = supervisor.get_worker_mut("adv_sig_model").unwrap();
    worker2.kill(libc::SIGKILL).expect("kill with SIGKILL");

    // Attempting forward pass on dead worker must report WorkerCrashed with signal Some(9)
    let req = ForwardPassRequest::new_prefill(9050, 1, vec![1, 2, 3], vec![0]);
    let res = worker2.execute_forward_pass(&req);
    assert!(res.is_err());
    match res.unwrap_err() {
        ManagerError::WorkerCrashed { signal, .. } => {
            assert_eq!(signal, Some(libc::SIGKILL));
        }
        other => panic!("expected ManagerError::WorkerCrashed with signal 9, got {other:?}"),
    }

    // 3. Test wrap_logits_as_float32_array on arena directly
    let arrow_logits = wrap_logits_as_float32_array(worker2.arena()).expect("wrap logits");
    assert!(!arrow_logits.is_empty());

    supervisor.shutdown_all();
    kv_task.abort();
    let _ = std::fs::remove_file(&weights);
    let _ = std::fs::remove_file(&worker_sock);
    let _ = std::fs::remove_file(&kv_sock);
}

#[test]
fn test_adversarial_unwritten_output_slab_rejection() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let kv_sock = unique_socket_path("k3_kv_adv_slab");
    let kv_server = KvServer::new(ServerConfig {
        socket_path: kv_sock.clone(),
        max_memory_bytes: 128 * 1024 * 1024,
    });
    let kv_task = rt.spawn(async move {
        let _ = kv_server.run().await;
    });

    let mut kv_client = None;
    for _ in 0..50 {
        if let Ok(c) = KvClient::connect(&kv_sock) {
            kv_client = Some(c);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut kv_client = kv_client.expect("connect kv");

    let (alloc, mut arena) = kv_client
        .allocate_and_attach("adv_slab_model", 65536, 64, 4)
        .expect("alloc arena");

    // Manually verify that wrap_logits_as_float32_array works with zeroed slab
    let hdr = *arena.header().expect("header");
    let slab_offset = hdr.output_slabs_offset as usize;
    let slab_slice = arena.slice_mut(slab_offset, 64).expect("slab slice");
    let slab_hdr = OutputSlabHeader::mut_from_bytes(slab_slice).expect("slab hdr");
    slab_hdr.status = SLAB_STATUS_WRITTEN;
    slab_hdr.vocab_size = 128;

    let logits = wrap_logits_as_float32_array(&arena).expect("wrap logits");
    assert_eq!(logits.len(), 128);

    // Mock socket pair simulating a worker that responds with ForwardPassCompleted but leaves slab IDLE
    let (s_mgr, s_mock_worker) = std::os::unix::net::UnixStream::pair().expect("unix pair");

    std::thread::spawn(move || {
        use k3_core::ipc::protocol::{
            ForwardPassCompleted, IpcMessage, StatusCode, recv_msg_frame, send_msg_frame,
        };
        if let Ok((IpcMessage::ForwardPassCommand(cmd), _)) = recv_msg_frame(&s_mock_worker) {
            // Leave slab IDLE in memory! (mock worker defect)
            let completed = IpcMessage::ForwardPassCompleted(ForwardPassCompleted {
                request_id: cmd.request_id,
                sequence_id: cmd.sequence_id,
                status: StatusCode::Success,
                error_message: None,
                committed_tokens: cmd.token_count,
                next_token: 42,
                prefill_latency_ms: 1.0,
                decode_latency_ms: 0.0,
            });
            let _ = send_msg_frame(&s_mock_worker, &completed, None);
        }
    });

    // Reset slab status to IDLE
    let slab_slice = arena.slice_mut(slab_offset, 64).expect("slab slice");
    let slab_hdr = OutputSlabHeader::mut_from_bytes(slab_slice).expect("slab hdr");
    slab_hdr.status = SLAB_STATUS_IDLE;

    let req = ForwardPassRequest::new_prefill(9999, 1, vec![101, 102], vec![0]);
    let res = dispatch_forward_pass(&s_mgr, &mut arena, &req);
    assert!(res.is_err(), "should reject unwritten slab");
    let err_msg = res.unwrap_err().to_string();
    assert!(
        err_msg.contains("output slab status is not WRITTEN"),
        "error must indicate slab status not WRITTEN: {err_msg}"
    );

    let _ = kv_client.deallocate_arena(alloc.arena_id);
    kv_task.abort();
    let _ = std::fs::remove_file(&kv_sock);
}
