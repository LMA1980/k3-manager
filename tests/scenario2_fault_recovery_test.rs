//! US-05 Scenario 2: Worker Fault Containment & Clean Recovery Integration Test Suite
//!
//! Validates:
//! 1. Multi-worker isolation: Model A and Model B workers running concurrently.
//! 2. Abnormal termination (`SIGKILL`) injected into Worker A mid-generation.
//! 3. Worker B continues executing forward passes completely unaffected.
//! 4. `k3-kv` and Model A shared context memory survive intact with committed token checkpoints.
//! 5. `k3-manager` cleanly reaps failed Worker A, spawns replacement Worker A', and re-attaches
//!    surviving arena via `SCM_RIGHTS`.
//! 6. Generation resumes from committed checkpoint position with preserved context.

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
use std::time::Duration;

use k3_core::ipc::layout::KvBlockTag;
use k3_kv::server::{KvServer, ServerConfig};
use k3_manager::client::KvClient;
use k3_manager::dispatch::ForwardPassRequest;
use k3_manager::supervisor::{WorkerPlacement, WorkerSupervisor};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(200);

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
    f.write_all(b"GGUF_TEST_SYNTHETIC_MODEL_WEIGHTS_V2")
        .expect("write dummy weights");
    path
}

#[test]
fn test_scenario2_fault_containment_and_clean_worker_restart() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // ------------------------------------------------------------------------
    // Step 1: Start standalone k3-kv server
    // ------------------------------------------------------------------------
    let kv_sock_path = unique_socket_path("k3_kv_s2");
    let kv_config = ServerConfig {
        socket_path: kv_sock_path.clone(),
        max_memory_bytes: 1024 * 1024 * 1024,
    };
    let kv_server = KvServer::new(kv_config);

    let kv_task = rt.spawn(async move {
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

    // ------------------------------------------------------------------------
    // Step 2: Allocate independent arenas for Model A and Model B in k3-kv
    // ------------------------------------------------------------------------
    let arena_bytes = 65536_u64;
    let n_ctx = 64_u32;
    let max_blocks = 4_u32;

    let (alloc_a, arena_a) = kv_client
        .allocate_and_attach("model_a", arena_bytes, n_ctx, max_blocks)
        .expect("allocate arena a");
    let (alloc_b, arena_b) = kv_client
        .allocate_and_attach("model_b", arena_bytes, n_ctx, max_blocks)
        .expect("allocate arena b");

    // ------------------------------------------------------------------------
    // Step 3: Spawn concurrent isolated workers for Model A and Model B
    // ------------------------------------------------------------------------
    let weights_a = create_dummy_weights("weights_a");
    let weights_b = create_dummy_weights("weights_b");
    let sock_a = unique_socket_path("worker_a");
    let sock_b = unique_socket_path("worker_b");

    let placement_a = WorkerPlacement::new("model_a", &weights_a, &sock_a, "x100");
    let placement_b = WorkerPlacement::new("model_b", &weights_b, &sock_b, "x100");

    let mut supervisor = WorkerSupervisor::new();

    supervisor
        .spawn_worker(
            placement_a,
            arena_a,
            alloc_a.arena_id,
            alloc_a.arena_bytes as usize,
            n_ctx,
            max_blocks,
        )
        .expect("spawn worker a");

    supervisor
        .spawn_worker(
            placement_b,
            arena_b,
            alloc_b.arena_id,
            alloc_b.arena_bytes as usize,
            n_ctx,
            max_blocks,
        )
        .expect("spawn worker b");

    assert_eq!(supervisor.worker_count(), 2);
    assert!(supervisor.is_worker_alive("model_a"));
    assert!(supervisor.is_worker_alive("model_b"));
    assert!(
        supervisor
            .get_worker("model_a")
            .unwrap()
            .last_placement_outcome()
            .is_some()
    );
    assert!(
        supervisor
            .get_worker("model_b")
            .unwrap()
            .last_placement_outcome()
            .is_some()
    );

    // ------------------------------------------------------------------------
    // Step 4: Execute forward passes on both models concurrently
    // ------------------------------------------------------------------------
    // Model A prompt prefill
    let prompt_a = vec![101_u32, 102, 103, 104, 105];
    let req_a1 = ForwardPassRequest::new_prefill(2001, 1, prompt_a.clone(), vec![0]);
    let res_a1 = supervisor
        .forward_pass("model_a", &req_a1)
        .expect("forward pass on model a");
    assert!(res_a1.completed.status.is_success());
    assert_eq!(res_a1.completed.committed_tokens, 5);

    // Verify Model A checkpointed state in KvBlockMeta
    let block_a0 = supervisor
        .get_worker("model_a")
        .unwrap()
        .get_block_meta(0)
        .expect("read block a 0");
    assert_eq!(block_a0.committed_tokens(), 5);
    assert_eq!(block_a0.tag_enum(), Some(KvBlockTag::Persist));

    // Model B prompt prefill
    let prompt_b = vec![501_u32, 502, 503];
    let req_b1 = ForwardPassRequest::new_prefill(3001, 2, prompt_b.clone(), vec![0]);
    let res_b1 = supervisor
        .forward_pass("model_b", &req_b1)
        .expect("forward pass on model b");
    assert!(res_b1.completed.status.is_success());
    assert_eq!(res_b1.completed.committed_tokens, 3);

    // ------------------------------------------------------------------------
    // Step 5: Inject SIGKILL crash on Worker A
    // ------------------------------------------------------------------------
    let worker_a = supervisor.get_worker_mut("model_a").expect("get worker a");
    let pid_a = worker_a.pid().expect("pid a");

    // Terminate Worker A with SIGKILL
    worker_a
        .kill(libc::SIGKILL)
        .expect("send SIGKILL to worker a");

    // Verify Worker A is dead
    assert!(!supervisor.is_worker_alive("model_a"));

    // ------------------------------------------------------------------------
    // Step 6: Verify Fault Containment — Worker B continues running unaffected
    // ------------------------------------------------------------------------
    assert!(supervisor.is_worker_alive("model_b"));

    // Worker B executes a decode pass without any disruption
    let req_b2 = ForwardPassRequest::new_decode(3002, 2, res_b1.next_token(), 3, vec![0]);
    let res_b2 = supervisor
        .forward_pass("model_b", &req_b2)
        .expect("forward pass on model b after worker a crash");
    assert!(res_b2.completed.status.is_success());
    assert_eq!(res_b2.completed.request_id, 3002);

    // Verify k3-kv daemon remains healthy and responsive
    let kv_ping = kv_client.ping().expect("kv ping after crash");
    assert!(kv_ping > 0);

    // ------------------------------------------------------------------------
    // Step 7: Clean Worker A Reap & Restart Recovery with Preserved Arena
    // ------------------------------------------------------------------------
    // Reap the dead Worker A process
    let exit_status = supervisor.reap_worker("model_a").expect("reap worker a");
    assert!(exit_status.is_some());

    // Recover Worker A: re-spawns worker process and re-attaches surviving arena via SCM_RIGHTS
    let recovered_a = supervisor
        .recover_worker("model_a")
        .expect("recover worker a");

    assert!(recovered_a.is_alive());
    assert_ne!(recovered_a.pid().unwrap(), pid_a); // New PID
    assert_eq!(recovered_a.generation(), 2); // Generation incremented
    assert_eq!(
        recovered_a.last_placement_outcome().unwrap().target_id,
        recovered_a.pid().unwrap()
    );

    // Verify preserved context in shared memory (ADR-026, ADR-042)
    let preserved_block_a0 = recovered_a.get_block_meta(0).expect("read recovered block");
    assert_eq!(preserved_block_a0.committed_tokens(), 5);
    assert_eq!(preserved_block_a0.tag_enum(), Some(KvBlockTag::Persist));
    assert_eq!(preserved_block_a0.sampler_rng_state[0], 987654321);

    // ------------------------------------------------------------------------
    // Step 8: Resume forward pass on recovered Worker A from committed position
    // ------------------------------------------------------------------------
    let req_a2 = ForwardPassRequest::new_decode(
        2002,
        1,
        res_a1.next_token(),
        5, // position 5 preserved from checkpoint
        vec![0],
    );

    let res_a2 = supervisor
        .forward_pass("model_a", &req_a2)
        .expect("forward pass on recovered worker a");

    assert!(res_a2.completed.status.is_success());
    assert_eq!(res_a2.completed.request_id, 2002);
    assert_eq!(res_a2.completed.committed_tokens, 1);
    assert!(res_a2.decode_latency_ms() > 0.0);

    // ------------------------------------------------------------------------
    // Step 9: Clean teardown
    // ------------------------------------------------------------------------
    supervisor.shutdown_all();
    assert_eq!(supervisor.worker_count(), 0);

    let _ = kv_client.deallocate_arena(alloc_a.arena_id);
    let _ = kv_client.deallocate_arena(alloc_b.arena_id);

    kv_task.abort();
    let _ = std::fs::remove_file(&weights_a);
    let _ = std::fs::remove_file(&weights_b);
    let _ = std::fs::remove_file(&sock_a);
    let _ = std::fs::remove_file(&sock_b);
    let _ = std::fs::remove_file(&kv_sock_path);
}
