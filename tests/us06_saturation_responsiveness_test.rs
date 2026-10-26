//! US-06 Scenario 1 E2E Test: Control Responsiveness Under 100% Compute Saturation (#66, #74).
//!
//! Asserts that while compute workers are 100% saturated executing continuous forward passes / compute loops,
//! concurrent requests submitted to `ControlPlaneRuntime` (prompt tokenization, chat template formatting,
//! prefix radix lookup, stop-sequence detection, queue depth, worker health, and administrative commands)
//! execute immediately within normal baseline latency bounds without being starved by compute workers.

#![allow(
    clippy::assert_is_empty,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_manager::domain::MockOsAffinityBackend;
use k3_manager::{
    AdminCommand, AdminResponse, ChatTemplate, ControlPlaneRuntime, SimpleTokenizer, SpecialTokens,
};

fn setup_runtime() -> (Arc<ControlPlaneRuntime>, Arc<MockOsAffinityBackend>) {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(
        capacity,
        Arc::clone(&mock_affinity) as Arc<dyn k3_manager::OsAffinityBackend>,
    )
    .expect("runtime creation");

    let runtime = Arc::new(runtime);

    // Register a test tokenizer and chat template
    let mut tok = SimpleTokenizer::new();
    for i in 0..1000 {
        tok.add_token(format!("word{i}"), i);
    }
    tok.add_token("hello", 1001);
    tok.add_token("world", 1002);
    tok.add_token("status", 1003);

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

    runtime.register_tokenizer("model-saturated", Arc::new(tok));
    runtime.register_template(
        "model-saturated",
        ChatTemplate::new(Some("<|im_start|>user\n{{prompt}}<|im_end|>\n".to_owned())),
    );

    // Pre-populate 16-token prefix blocks
    let prefix_tokens: Vec<u32> = (0..32).collect();
    runtime.insert_prefix("project-test", "model-saturated", &prefix_tokens, &[10, 11]);

    // Record model placement in status plane
    runtime.record_model_placement("model-saturated".to_owned(), "x100".to_owned(), Some(9999));

    (runtime, mock_affinity)
}

#[test]
fn test_scenario1_control_responsiveness_under_100_percent_compute_saturation() {
    let (runtime, _) = setup_runtime();

    // 1. Launch saturated compute workers utilizing 100% CPU capacity across 4 worker threads
    let compute_running = Arc::new(AtomicBool::new(true));
    let forward_pass_counter = Arc::new(AtomicU64::new(0));

    let mut compute_handles = Vec::new();
    for worker_id in 0..4 {
        let running_clone = Arc::clone(&compute_running);
        let counter_clone = Arc::clone(&forward_pass_counter);

        let handle = std::thread::Builder::new()
            .name(format!("compute-worker-{worker_id}"))
            .spawn(move || {
                let mut x: u64 = 0;
                while running_clone.load(Ordering::Relaxed) {
                    // Heavy compute forward pass simulation (100% saturation)
                    for _ in 0..10_000 {
                        x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    }
                    counter_clone.fetch_add(1, Ordering::Relaxed);
                }
                x
            })
            .expect("spawn compute thread");
        compute_handles.push(handle);
    }

    // Give compute workers a moment to ramp up to 100% CPU saturation
    std::thread::sleep(Duration::from_millis(50));
    assert!(
        forward_pass_counter.load(Ordering::Relaxed) > 0,
        "compute workers must be actively executing forward passes"
    );

    // 2. Concurrently submit control plane requests while compute workers are 100% saturated
    let client_iterations = 200;
    let mut latencies_micros = Vec::with_capacity(client_iterations);

    for i in 0..client_iterations {
        let t0 = Instant::now();

        // Operation A: Chat template formatting
        let formatted = runtime
            .format_prompt("model-saturated", "hello world")
            .expect("format prompt");
        assert!(formatted.contains("hello world"));

        // Operation B: Prompt tokenization and context bounds admission
        let tokens = runtime
            .tokenize_prompt("model-saturated", "hello world status", false, 128)
            .expect("tokenize prompt");
        assert_eq!(tokens, vec![1001, 1002, 1003]);

        // Operation C: Prefix radix index lookup
        let query_tokens: Vec<u32> = (0..20).collect();
        let match_res = runtime.match_prefix("project-test", "model-saturated", &query_tokens);
        assert_eq!(match_res.matched_blocks, vec![10]);
        assert_eq!(match_res.uncached_token_count(), 4);

        // Operation D: Streaming stop-sequence detection
        assert!(
            runtime
                .is_stop_token("model-saturated", 2)
                .expect("stop token check")
        );
        assert!(
            !runtime
                .is_stop_token("model-saturated", 1001)
                .expect("stop token check")
        );

        // Operation E: Queue status reporting
        runtime.increment_queue_depth();
        assert!(runtime.queue_depth() > 0);
        runtime.decrement_queue_depth();

        // Operation F: Worker health snapshots
        let health = runtime.worker_health_snapshots();
        assert_eq!(health.len(), 1);
        assert!(health[0].is_alive);

        // Operation G: Administrative commands dispatch
        if i % 10 == 0 {
            let ping = runtime
                .execute_admin(AdminCommand::Ping)
                .expect("admin ping");
            assert_eq!(ping, AdminResponse::Pong);

            let stats = runtime
                .execute_admin(AdminCommand::GetPrefixCacheStats)
                .expect("admin cache stats");
            assert!(matches!(stats, AdminResponse::PrefixCacheStats(_)));
        }

        let elapsed = t0.elapsed();
        latencies_micros.push(elapsed.as_micros());
    }

    // 3. Stop saturated compute workers and join threads
    compute_running.store(false, Ordering::Relaxed);
    for handle in compute_handles {
        let _ = handle.join();
    }

    let total_forward_passes = forward_pass_counter.load(Ordering::Relaxed);
    assert!(
        total_forward_passes > 50,
        "compute workers executed {total_forward_passes} passes, confirming saturation during test"
    );

    // 4. Assert latency bounds: control operations must not be starved
    let avg_latency = latencies_micros.iter().sum::<u128>() as f64 / client_iterations as f64;
    let max_latency = *latencies_micros.iter().max().unwrap_or(&0);

    // Average latency for the entire suite of 7 control operations should be well below 5ms (5000us)
    assert!(
        avg_latency < 5000.0,
        "average control plane latency under 100% compute saturation was {avg_latency:.2}us, exceeding 5ms ceiling"
    );

    // Max latency for any individual batch of operations should not exceed 50ms
    assert!(
        max_latency < 50_000,
        "max control plane latency under 100% compute saturation was {max_latency}us, exceeding 50ms ceiling"
    );
}
