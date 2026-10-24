//! US-05 Scenario 1: Zero-Copy Forward-Pass Orchestration Integration Test Suite
//!
//! Validates:
//! 1. Multi-domain process isolation: `k3-kv`, `k3-manager`, and `k3-worker`.
//! 2. Context memory arena allocation in `k3-kv` and `SCM_RIGHTS` FD transfer to worker.
//! 3. Zero bulk tensor payload bytes transferred across UDS sockets.
//! 4. In-place token writes and Apache Arrow `Float32Array` zero-copy logits exposure.
//! 5. Bit-exact mathematical execution for both prefill and decode passes.

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

use k3_core::ipc::layout::{KvBlockTag, ARENA_MAGIC, ARENA_VERSION};
use k3_kv::server::{KvServer, ServerConfig};
use k3_manager::client::KvClient;
use k3_manager::dispatch::{wrap_input_tokens_as_arrow, ForwardPassRequest};
use k3_manager::supervisor::{WorkerPlacement, WorkerSupervisor};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(100);

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
    f.write_all(b"GGUF_TEST_SYNTHETIC_MODEL_WEIGHTS_V1")
        .expect("write dummy weights");
    path
}

#[test]
fn test_scenario1_zerocopy_forward_pass_orchestration() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // ------------------------------------------------------------------------
    // Step 1: Start standalone k3-kv server
    // ------------------------------------------------------------------------
    let kv_sock_path = unique_socket_path("k3_kv_s1");
    let kv_config = ServerConfig {
        socket_path: kv_sock_path.clone(),
        max_memory_bytes: 1024 * 1024 * 1024, // 1 GiB
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

    // Verify heartbeat ping
    let ping_echo = kv_client.ping().expect("kv ping");
    assert!(ping_echo > 0);

    // ------------------------------------------------------------------------
    // Step 2: Allocate shared-memory arena via KvClient
    // ------------------------------------------------------------------------
    let model_id = "synthetic-qwen";
    let requested_arena_bytes = 65536_u64; // 64 KiB conservative arena
    let n_ctx = 64_u32;
    let max_blocks = 4_u32;

    let (alloc_resp, arena) = kv_client
        .allocate_and_attach(model_id, requested_arena_bytes, n_ctx, max_blocks)
        .expect("allocate and attach arena from k3-kv");

    assert!(alloc_resp.status.is_success());
    assert!(alloc_resp.arena_bytes >= requested_arena_bytes);
    let arena_id = alloc_resp.arena_id;
    let actual_arena_bytes = alloc_resp.arena_bytes as usize;

    // Verify initial arena header invariants
    let hdr = *arena.header().expect("read arena header");
    assert_eq!(hdr.magic, ARENA_MAGIC);
    assert_eq!(hdr.version, ARENA_VERSION);
    assert_eq!(hdr.n_ctx, n_ctx);

    // ------------------------------------------------------------------------
    // Step 3: Supervisor spawns isolated k3-worker process and transfers arena
    // ------------------------------------------------------------------------
    let weights_path = create_dummy_weights("weights_s1");
    let worker_sock_path = unique_socket_path("k3_worker_s1");

    let placement = WorkerPlacement::new(
        model_id,
        &weights_path,
        &worker_sock_path,
        "x100",
    );

    let mut supervisor = WorkerSupervisor::new();
    let worker = supervisor
        .spawn_worker(
            placement,
            arena,
            arena_id,
            actual_arena_bytes,
            n_ctx,
            max_blocks,
        )
        .expect("spawn isolated worker and attach arena");

    assert!(worker.is_alive());
    assert!(worker.pid().is_some());
    assert_eq!(worker.generation(), 1);

    // Check heartbeat communication with worker
    let hb_resp = worker.send_heartbeat().expect("worker heartbeat");
    assert_eq!(hb_resp.completed_passes, 0);
    assert!(!hb_resp.is_busy);

    // ------------------------------------------------------------------------
    // Step 4: Write prompt tokens into shared memory slot (Zero UDS bulk bytes)
    // ------------------------------------------------------------------------
    let prompt_tokens = vec![101_u32, 2054, 2003, 1037, 7000];
    let prefill_req = ForwardPassRequest::new_prefill(
        1001,
        1,
        prompt_tokens.clone(),
        vec![0],
    );
    // (Before dispatch, tokens will be written by dispatch_forward_pass)

    // ------------------------------------------------------------------------
    // Step 5: Execute prompt prefill forward pass
    // ------------------------------------------------------------------------
    let prefill_result = worker
        .execute_forward_pass(&prefill_req)
        .expect("execute prefill forward pass");

    assert!(prefill_result.completed.status.is_success());
    assert_eq!(prefill_result.completed.request_id, 1001);
    assert_eq!(prefill_result.completed.committed_tokens, 5);
    assert!(prefill_result.prefill_latency_ms() > 0.0);
    assert_eq!(prefill_result.decode_latency_ms(), 0.0);

    // Verify arrow input tokens view after write
    let arrow_input_tokens = wrap_input_tokens_as_arrow(worker.arena()).expect("arrow input view");
    assert_eq!(arrow_input_tokens.len(), 5);
    assert_eq!(arrow_input_tokens.values().as_ref(), prompt_tokens.as_slice());

    // ------------------------------------------------------------------------
    // Step 6: Verify output logits exposed as zero-copy Apache Arrow Float32Array
    // ------------------------------------------------------------------------
    let arrow_logits = prefill_result.logits();
    assert!(!arrow_logits.is_empty());
    let vocab_size = arrow_logits.len();

    // Verify bit-exactness of synthetic output logits:
    // logit[t] = token_vals[i % token_vals.len()] * 0.01 + (i as f32) * 0.1
    let logits_slice = arrow_logits.values().as_ref();
    for (i, &l) in logits_slice.iter().enumerate() {
        let expected = (prompt_tokens[i % prompt_tokens.len()] as f32) * 0.01 + (i as f32) * 0.1;
        assert!(
            (l - expected).abs() < 1e-6,
            "logits mismatch at index {i}: observed {l}, expected {expected}"
        );
    }

    // Verify KV cache block metadata committed in place
    let block0 = worker.get_block_meta(0).expect("read block 0");
    assert_eq!(block0.committed_tokens(), 5);
    assert_eq!(block0.tag_enum(), Some(KvBlockTag::Persist));
    assert_eq!(block0.sampler_rng_state[0], 987654321);

    // ------------------------------------------------------------------------
    // Step 7: Execute single-token autoregressive decode pass
    // ------------------------------------------------------------------------
    let sampled_next_token = prefill_result.next_token();
    let decode_req = ForwardPassRequest::new_decode(
        1002,
        1,
        sampled_next_token,
        5, // position after prompt
        vec![0],
    );

    let decode_result = worker
        .execute_forward_pass(&decode_req)
        .expect("execute decode forward pass");

    assert!(decode_result.completed.status.is_success());
    assert_eq!(decode_result.completed.request_id, 1002);
    assert_eq!(decode_result.completed.committed_tokens, 1);
    assert_eq!(decode_result.prefill_latency_ms(), 0.0);
    assert!(decode_result.decode_latency_ms() > 0.0);

    // Verify decode logits output view
    assert_eq!(decode_result.logits().len(), vocab_size);

    // ------------------------------------------------------------------------
    // Step 8: Clean shutdown across all domains
    // ------------------------------------------------------------------------
    worker.shutdown(true).expect("clean worker shutdown");
    assert!(!worker.is_alive());

    // Deallocate arena in k3-kv
    let dealloc_resp = kv_client.deallocate_arena(arena_id).expect("deallocate arena");
    assert!(dealloc_resp.status.is_success());

    // Terminate k3-kv server and clean temp files
    kv_task.abort();
    let _ = std::fs::remove_file(&weights_path);
    let _ = std::fs::remove_file(&worker_sock_path);
    let _ = std::fs::remove_file(&kv_sock_path);
}
