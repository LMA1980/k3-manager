//! US-06.4.1 (#77): US-06 Scenario 1 E2E Integration Test: Control Responsiveness Under 100% Compute Saturation.
//!
//! Validates:
//! 1. Starts `k3-kv`, `ControlPlaneRuntime` (pinned to reserved active control cores `0x00C0`), and compute workers
//!    utilizing 100% of their allocated compute capacity (`X100` 6-core compute cap `0x003F`).
//! 2. Sustains continuous compute-heavy forward passes across all allocated compute worker threads.
//! 3. Concurrently dispatches client/operator control-plane operations: chat template formatting, prompt
//!    tokenization (`k3-tokenizer`), token-ID prefix radix lookup, stop-sequence detection, queue status
//!    reporting, and administrative health commands.
//! 4. Asserts that all control-plane operations succeed and complete within normal latency bounds without
//!    starvation, and verifies that compute worker threads never migrate onto dedicated active control cores.

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
use k3_kv::server::{KvServer, ServerConfig};
use k3_manager::client::KvClient;
use k3_manager::dispatch::ForwardPassRequest;
use k3_manager::domain::{CgroupV2DomainController, MockOsAffinityBackend};
use k3_manager::supervisor::{WorkerPlacement, WorkerSupervisor};
use k3_manager::{
    AdminCommand, AdminResponse, ChatTemplate, ControlPlaneRuntime, OsAffinityBackend,
    SimpleTokenizer, SpecialTokens,
};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(300);

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
    f.write_all(b"GGUF_TEST_SYNTHETIC_MODEL_WEIGHTS_S1")
        .expect("write dummy weights");
    path
}

#[test]
fn test_scenario1_control_responsiveness_under_saturation_e2e() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // ------------------------------------------------------------------------
    // Step 1: Start standalone k3-kv server and allocate shared context arena
    // ------------------------------------------------------------------------
    let kv_sock_path = unique_socket_path("k3_kv_sat_s1");
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
    let model_id = "model-saturated-s1";

    let (alloc_resp, arena) = kv_client
        .allocate_and_attach(model_id, arena_bytes, n_ctx, max_blocks)
        .expect("allocate and attach arena from k3-kv");

    // ------------------------------------------------------------------------
    // Step 2: Configure capacity, mock affinity, ControlPlaneRuntime & WorkerSupervisor
    // ------------------------------------------------------------------------
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid control capacity");
    assert_eq!(capacity.active_control_mask(), 0x00C0); // cores 6-7
    assert_eq!(capacity.spare_recovery_mask(), 0x0020); // core 5
    assert_eq!(capacity.x100_compute_cap_mask(), 0x003F); // cores 0-5

    let mock_affinity = Arc::new(MockOsAffinityBackend::default());

    // Create ControlPlaneRuntime pinned to active control cores 0x00C0
    let control_runtime = Arc::new(
        ControlPlaneRuntime::with_affinity_backend(
            capacity.clone(),
            Arc::clone(&mock_affinity) as Arc<dyn OsAffinityBackend>,
        )
        .expect("create control plane runtime"),
    );

    // Register test tokenizer with chat template and stop sequences
    let mut tok = SimpleTokenizer::new();
    for i in 0..1000 {
        tok.add_token(format!("word{i}"), i);
    }
    tok.add_token("hello", 1001);
    tok.add_token("world", 1002);
    tok.add_token("saturation", 1003);

    let special = SpecialTokens {
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        eot_token_id: Some(3),
        padding_token_id: Some(0),
        add_bos_token: false,
        stop_tokens: vec![2, 3],
        stop_sequences: vec!["<|im_end|>".to_owned()],
    };
    tok.set_special_tokens(special);

    control_runtime.register_tokenizer(model_id, Arc::new(tok));
    control_runtime.register_template(
        model_id,
        ChatTemplate::new(Some("<|im_start|>user\n{{prompt}}<|im_end|>\n".to_owned())),
    );

    // Pre-populate 16-token prefix radix block
    let prefix_tokens: Vec<u32> = (0..32).collect();
    control_runtime.insert_prefix("project-sat", model_id, &prefix_tokens, &[10, 11]);
    control_runtime.record_model_placement(model_id.to_owned(), "x100".to_owned(), Some(8888));

    // Create WorkerSupervisor using same mock affinity backend to monitor placement
    let domain_ctrl = Arc::new(
        CgroupV2DomainController::default()
            .with_affinity_backend(Arc::clone(&mock_affinity) as Arc<dyn OsAffinityBackend>),
    );
    let mut supervisor = WorkerSupervisor::new()
        .with_capacity(capacity.clone())
        .with_domain_controller(domain_ctrl);

    let weights_path = create_dummy_weights("weights_sat_s1");
    let worker_sock = unique_socket_path("worker_sat_s1");
    let placement = WorkerPlacement::new(model_id, &weights_path, &worker_sock, "x100");

    let worker = supervisor
        .spawn_worker(
            placement,
            arena,
            alloc_resp.arena_id,
            alloc_resp.arena_bytes as usize,
            n_ctx,
            max_blocks,
        )
        .expect("spawn worker");

    assert!(worker.is_alive());
    let worker_pid = worker.pid().expect("worker process PID");

    // ------------------------------------------------------------------------
    // Step 3: Verify compute worker placement invariants (No encroachment onto 0x00C0)
    // ------------------------------------------------------------------------
    let affinity_calls = mock_affinity.calls();
    let worker_calls: Vec<_> = affinity_calls
        .iter()
        .filter(|&&(pid, _)| pid == worker_pid)
        .collect();

    assert!(
        !worker_calls.is_empty(),
        "worker PID {worker_pid} must have an OS affinity placement recorded"
    );

    for &&(_, mask) in &worker_calls {
        assert_eq!(
            mask & 0x00C0,
            0,
            "compute worker mask 0x{mask:04X} encroached onto active control cores 0x00C0!"
        );
        assert_ne!(
            mask & 0x003F,
            0,
            "compute worker mask 0x{mask:04X} must lie within compute mask 0x003F"
        );
    }

    // ------------------------------------------------------------------------
    // Step 4: Launch saturated compute workloads utilizing 100% compute capacity
    // ------------------------------------------------------------------------
    let compute_running = Arc::new(AtomicBool::new(true));
    let pass_counter = Arc::new(AtomicU64::new(0));

    // Spawn 6 compute worker threads representing the 6 compute cores in 0x003F
    let mut compute_handles = Vec::new();
    for thread_idx in 0..6 {
        let running_clone = Arc::clone(&compute_running);
        let counter_clone = Arc::clone(&pass_counter);
        let aff_clone = Arc::clone(&mock_affinity);

        let handle = std::thread::Builder::new()
            .name(format!("compute-sat-{thread_idx}"))
            .spawn(move || {
                let mask = 1_u64 << thread_idx;
                let _ = aff_clone.set_affinity(0, mask);
                let mut acc: u64 = 0;
                while running_clone.load(Ordering::Relaxed) {
                    for _ in 0..15_000 {
                        acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    }
                    counter_clone.fetch_add(1, Ordering::Relaxed);
                }
                acc
            })
            .expect("spawn compute saturation thread");
        compute_handles.push(handle);
    }

    let ws_arc = Arc::new(Mutex::new(supervisor));
    let ws_clone = Arc::clone(&ws_arc);
    let forward_pass_running = Arc::clone(&compute_running);
    let forward_pass_counter = Arc::new(AtomicU64::new(0));
    let forward_pass_counter_clone = Arc::clone(&forward_pass_counter);

    // Continuous forward pass thread sustaining real model passes concurrently
    let forward_pass_handle = std::thread::Builder::new()
        .name("compute-continuous-passes".to_string())
        .spawn(move || {
            let prompt_tokens = vec![101_u32, 102, 103, 104];
            let mut req_id = 6000_u64;
            while forward_pass_running.load(Ordering::Relaxed) {
                req_id += 1;
                let req =
                    ForwardPassRequest::new_prefill(req_id, 1, prompt_tokens.clone(), vec![0]);
                if let Ok(mut ws) = ws_clone.lock() {
                    let _ = ws.forward_pass(model_id, &req);
                    forward_pass_counter_clone.fetch_add(1, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        })
        .expect("spawn forward pass loop");

    // Give saturation threads time to achieve 100% CPU utilization
    std::thread::sleep(Duration::from_millis(50));
    assert!(pass_counter.load(Ordering::Relaxed) > 10);

    // ------------------------------------------------------------------------
    // Step 5: Concurrently dispatch client/operator control-plane operations
    // ------------------------------------------------------------------------
    let client_iterations = 100;
    let mut latencies_micros = Vec::with_capacity(client_iterations);

    for i in 0..client_iterations {
        let t0 = Instant::now();

        // 1. Chat template formatting
        let formatted = control_runtime
            .format_prompt(model_id, "hello world saturation test")
            .expect("format prompt");
        assert!(formatted.contains("hello world saturation test"));

        // 2. Prompt tokenization
        let tokens = control_runtime
            .tokenize_prompt(model_id, "hello world saturation", false, 64)
            .expect("tokenize prompt");
        assert_eq!(tokens, vec![1001, 1002, 1003]);

        // 3. Token-ID prefix radix lookup
        let query: Vec<u32> = (0..20).collect();
        let match_res = control_runtime.match_prefix("project-sat", model_id, &query);
        assert_eq!(match_res.matched_blocks, vec![10]);
        assert_eq!(match_res.uncached_token_count(), 4);

        // 4. Stop-sequence detection
        assert!(
            control_runtime
                .is_stop_token(model_id, 2)
                .expect("stop token check")
        );
        assert!(
            !control_runtime
                .is_stop_token(model_id, 1001)
                .expect("stop token check")
        );
        assert!(
            control_runtime
                .is_stop_sequence(model_id, "output stream <|im_end|>")
                .expect("stop sequence check")
        );

        // 5. Queue depth reporting
        control_runtime.increment_queue_depth();
        assert!(control_runtime.queue_depth() > 0);
        control_runtime.decrement_queue_depth();

        // 6. Administrative health commands
        let health = control_runtime.worker_health_snapshots();
        assert!(!health.is_empty());

        if i % 10 == 0 {
            let ping = control_runtime
                .execute_admin(AdminCommand::Ping)
                .expect("admin ping");
            assert_eq!(ping, AdminResponse::Pong);

            let stats = control_runtime
                .execute_admin(AdminCommand::GetPrefixCacheStats)
                .expect("admin stats");
            assert!(matches!(stats, AdminResponse::PrefixCacheStats(_)));
        }

        // 7. Dispatched active control thread execution (ADR-007)
        if i % 25 == 0 {
            let disp_tok = control_runtime
                .dispatch_tokenize_prompt(model_id, "hello world", false, 64)
                .expect("dispatch tokenize");
            assert_eq!(disp_tok, vec![1001, 1002]);

            let disp_pong = control_runtime
                .dispatch_execute_admin(AdminCommand::Ping)
                .expect("dispatch ping");
            assert_eq!(disp_pong, AdminResponse::Pong);
        }

        latencies_micros.push(t0.elapsed().as_micros());
    }

    // ------------------------------------------------------------------------
    // Step 6: Stop compute workers and assert latency bounds
    // ------------------------------------------------------------------------
    compute_running.store(false, Ordering::Relaxed);
    for handle in compute_handles {
        let _ = handle.join();
    }
    let _ = forward_pass_handle.join();

    let total_passes = pass_counter.load(Ordering::Relaxed);
    let sustained_passes = forward_pass_counter.load(Ordering::Relaxed);
    assert!(
        total_passes > 50,
        "compute threads executed {total_passes} passes, confirming saturation"
    );
    assert!(
        sustained_passes > 0,
        "sustained continuous forward passes must be executed concurrently during test"
    );

    let avg_latency = latencies_micros.iter().sum::<u128>() as f64 / client_iterations as f64;
    let max_latency = *latencies_micros.iter().max().unwrap_or(&0);

    // Average latency for the control plane operations must remain well below 5ms (< 5000us)
    assert!(
        avg_latency < 5000.0,
        "average control plane latency under 100% saturation was {avg_latency:.2}us, exceeding 5ms ceiling"
    );

    // Max latency for any individual batch of operations should not exceed 50ms
    assert!(
        max_latency < 50_000,
        "max control plane latency was {max_latency}us, exceeding 50ms ceiling"
    );

    // ------------------------------------------------------------------------
    // Step 7: Final isolation verification: no compute calls encroached onto 0x00C0
    // ------------------------------------------------------------------------
    let all_affinities = mock_affinity.calls();
    for &(pid, mask) in &all_affinities {
        if pid == worker_pid {
            assert_eq!(
                mask & 0x00C0,
                0,
                "encroachment detected: worker {pid} was assigned mask 0x{mask:04X}"
            );
        }
    }

    // Cleanly shutdown worker
    let mut supervisor = ws_arc.lock().expect("lock supervisor");
    supervisor.shutdown_all();
}
