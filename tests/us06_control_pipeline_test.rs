//! Integration tests for US-06.2: Control-Plane Orchestration Pipeline & Tokenization Responsiveness (#66, #72, #73, #74).

#![allow(
    clippy::assert_is_empty,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss
)]

use std::sync::Arc;
use std::time::Instant;

use k3_core::K3Error;
use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_manager::domain::MockOsAffinityBackend;
use k3_manager::{
    AdminCommand, AdminResponse, ChatTemplate, ControlPlaneRuntime, IncrementalStopDetector,
    PrefixRadixIndex, SimpleTokenizer, SpecialTokens, Tokenizer, format_chat_prompt,
    validate_and_tokenize,
};

fn create_test_simple_tokenizer() -> SimpleTokenizer {
    let mut tok = SimpleTokenizer::new();
    for i in 0..500 {
        tok.add_token(format!("tok{i}"), i);
    }
    tok.add_token("hello", 1000);
    tok.add_token("world", 1001);
    tok.add_token("k3", 1002);
    tok.add_token("platform", 1003);

    let special = SpecialTokens {
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        eot_token_id: Some(3),
        padding_token_id: Some(0),
        add_bos_token: false,
        stop_tokens: vec![2, 3, 999],
        stop_sequences: vec!["<|im_end|>".to_owned()],
    };
    tok.set_special_tokens(special);
    tok
}

// ============================================================================
// Sub-issue #72 (US-06.2.1): Chat Template, Tokenizer & Stop Detection Tests
// ============================================================================

#[test]
fn test_sub_issue_72_chat_template_and_tokenization() {
    let tokenizer = create_test_simple_tokenizer();
    let template = ChatTemplate::new(Some(
        "<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n".to_owned(),
    ));

    let formatted = format_chat_prompt(&template, "hello world");
    assert_eq!(
        formatted,
        "<|im_start|>user\nhello world<|im_end|>\n<|im_start|>assistant\n"
    );

    // Tokenize without chat template
    let tokens = validate_and_tokenize(&tokenizer, "hello world", None, false, 10).unwrap();
    assert_eq!(tokens, vec![1000, 1001]);
}

#[test]
fn test_sub_issue_72_context_length_exceeded_admission_rejection() {
    let tokenizer = create_test_simple_tokenizer();

    // Valid admission
    let tokens =
        validate_and_tokenize(&tokenizer, "hello world k3 platform", None, false, 8).unwrap();
    assert_eq!(tokens.len(), 4);

    // Exact limit rejection: requested 4 tokens with n_ctx == 4 (ADR-027 requires tokens.len() < n_ctx)
    let res_exact = validate_and_tokenize(&tokenizer, "hello world k3 platform", None, false, 4);
    match res_exact {
        Err(K3Error::ContextLengthExceeded {
            requested_tokens,
            max_context_tokens,
        }) => {
            assert_eq!(requested_tokens, 4);
            assert_eq!(max_context_tokens, 4);
        }
        other => panic!("expected ContextLengthExceeded, got {other:?}"),
    }

    // Over limit rejection: requested 4 tokens with n_ctx == 2
    let res_over = validate_and_tokenize(&tokenizer, "hello world k3 platform", None, false, 2);
    match res_over {
        Err(K3Error::ContextLengthExceeded {
            requested_tokens,
            max_context_tokens,
        }) => {
            assert_eq!(requested_tokens, 4);
            assert_eq!(max_context_tokens, 2);
        }
        other => panic!("expected ContextLengthExceeded, got {other:?}"),
    }

    // Zero context capacity rejection
    let res_zero = validate_and_tokenize(&tokenizer, "hello", None, false, 0);
    assert!(matches!(
        res_zero,
        Err(K3Error::ContextLengthExceeded { .. })
    ));
}

#[test]
fn test_sub_issue_72_streaming_stop_detection() {
    let tokenizer = create_test_simple_tokenizer();
    let mut detector = IncrementalStopDetector::new(tokenizer.special_tokens().clone());

    // Generate normal tokens
    assert!(!detector.push_token(1000));
    assert!(!detector.push_token(1001));
    assert_eq!(detector.observed_tokens(), &[1000, 1001]);
    assert!(!detector.is_stopped());

    // Encounter EOS token (2)
    assert!(detector.push_token(2));
    assert!(detector.is_stopped());

    // Generator halts
    assert!(detector.push_token(1002));
    assert!(detector.is_stopped());

    // Reset and test EOT token (3)
    detector.reset();
    assert!(!detector.is_stopped());
    assert!(!detector.push_token(1002));
    assert!(detector.push_token(3));
    assert!(detector.is_stopped());
}

// ============================================================================
// Sub-issue #73 (US-06.2.2): Prefix Radix Index & 16-Token Block Tests
// ============================================================================

#[test]
fn test_sub_issue_73_prefix_radix_exact_and_partial_matching() {
    let index = PrefixRadixIndex::new();

    let block0: Vec<u32> = (0..16).collect();
    let block1: Vec<u32> = (16..32).collect();
    let block2: Vec<u32> = (32..48).collect();

    let mut full_tokens = Vec::new();
    full_tokens.extend_from_slice(&block0);
    full_tokens.extend_from_slice(&block1);
    full_tokens.extend_from_slice(&block2);
    assert_eq!(full_tokens.len(), 48);

    // Insert 3 blocks
    let inserted = index.insert("tenant-a", "qwen-7b", &full_tokens, &[100, 101, 102]);
    assert_eq!(inserted, 3);

    // 1. Exact 3-block match (48 tokens)
    let res_exact = index.match_longest_prefix("tenant-a", "qwen-7b", &full_tokens);
    assert_eq!(res_exact.matched_blocks, vec![100, 101, 102]);
    assert_eq!(res_exact.matched_token_count, 48);
    assert!(res_exact.uncached_tokens.is_empty());

    // 2. Partial block match (e.g., 38 tokens = 2 full blocks + 6 trailing tokens)
    let mut partial_tokens = Vec::new();
    partial_tokens.extend_from_slice(&block0);
    partial_tokens.extend_from_slice(&block1);
    partial_tokens.extend_from_slice(&[900, 901, 902, 903, 904, 905]);
    assert_eq!(partial_tokens.len(), 38);

    let res_partial = index.match_longest_prefix("tenant-a", "qwen-7b", &partial_tokens);
    assert_eq!(res_partial.matched_blocks, vec![100, 101]);
    assert_eq!(res_partial.matched_token_count, 32);
    assert_eq!(
        res_partial.uncached_tokens,
        vec![900, 901, 902, 903, 904, 905]
    );
    assert_eq!(res_partial.uncached_token_count(), 6);

    // 3. Short query under 16 tokens
    let short_tokens = vec![1, 2, 3, 4, 5];
    let res_short = index.match_longest_prefix("tenant-a", "qwen-7b", &short_tokens);
    assert!(res_short.is_empty());
    assert_eq!(res_short.matched_token_count, 0);
    assert_eq!(res_short.uncached_tokens, short_tokens);
}

#[test]
fn test_sub_issue_73_key_isolation_multi_tenant() {
    let index = PrefixRadixIndex::new();
    let block0: Vec<u32> = (100..116).collect();

    index.insert("project-alpha", "llama-3", &block0, &[50]);

    // Same project and model hits
    let hit = index.match_longest_prefix("project-alpha", "llama-3", &block0);
    assert_eq!(hit.matched_blocks, vec![50]);

    // Different project must not leak or match (ADR-004)
    let cross_proj = index.match_longest_prefix("project-beta", "llama-3", &block0);
    assert!(cross_proj.is_empty());
    assert_eq!(cross_proj.uncached_tokens, block0);

    // Different model within same project must not match
    let cross_model = index.match_longest_prefix("project-alpha", "mistral-7b", &block0);
    assert!(cross_model.is_empty());
}

#[test]
fn test_sub_issue_73_eviction_and_sub_millisecond_latency() {
    let index = PrefixRadixIndex::new();
    let block0: Vec<u32> = (0..16).collect();
    let block1: Vec<u32> = (16..32).collect();

    let mut tokens = Vec::new();
    tokens.extend_from_slice(&block0);
    tokens.extend_from_slice(&block1);

    index.insert("p1", "m1", &tokens, &[10, 20]);
    assert_eq!(index.stats().total_cached_blocks, 2);

    // Invalidate specific block
    assert!(index.invalidate_block("p1", "m1", 20));
    assert_eq!(
        index
            .match_longest_prefix("p1", "m1", &tokens)
            .matched_blocks,
        vec![10]
    );

    // Benchmark sub-millisecond lookup latency
    let start = Instant::now();
    let iterations = 1000;
    for _ in 0..iterations {
        let _ = index.match_longest_prefix("p1", "m1", &tokens);
    }
    let elapsed = start.elapsed();
    let per_lookup_micros = elapsed.as_micros() as f64 / f64::from(iterations);

    // Must be well below 1,000 microseconds (1 millisecond) per lookup
    assert!(
        per_lookup_micros < 1000.0,
        "average lookup latency {per_lookup_micros:.2}us exceeds 1ms target"
    );
}

// ============================================================================
// Sub-issue #74 (US-06.2.3): Active Control Thread Pool & Non-Blocking Plane Tests
// ============================================================================

#[test]
fn test_sub_issue_74_control_plane_runtime_execution() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");

    // 1. Active control threads bound to cores 6 and 7
    assert_eq!(runtime.active_thread_count(), 2);
    assert_eq!(runtime.active_control_mask(), 0x00C0);

    // 2. Round-robin dispatch
    let c1 = runtime.dispatch_control_task(|c| c).unwrap();
    let c2 = runtime.dispatch_control_task(|c| c).unwrap();
    assert_eq!(c1, 6);
    assert_eq!(c2, 7);

    // 3. Status plane: queue depth
    assert_eq!(runtime.queue_depth(), 0);
    runtime.increment_queue_depth();
    assert_eq!(runtime.queue_depth(), 1);
    runtime.decrement_queue_depth();
    assert_eq!(runtime.queue_depth(), 0);

    // 4. Model placements and worker health snapshots
    runtime.record_model_placement("qwen2.5-7b".to_owned(), "x100".to_owned(), Some(42));
    let placements = runtime.active_model_placements();
    assert_eq!(placements, vec!["qwen2.5-7b"]);

    let health = runtime.worker_health_snapshots();
    assert_eq!(health.len(), 1);
    assert_eq!(health[0].model_id, "qwen2.5-7b");
    assert!(health[0].is_alive);

    // 5. Admin command plane
    let ping = runtime.execute_admin(AdminCommand::Ping).unwrap();
    assert_eq!(ping, AdminResponse::Pong);

    let queue_resp = runtime.execute_admin(AdminCommand::GetQueueDepth).unwrap();
    assert_eq!(queue_resp, AdminResponse::QueueDepth(0));

    let placements_resp = runtime
        .execute_admin(AdminCommand::GetActivePlacements)
        .unwrap();
    assert_eq!(
        placements_resp,
        AdminResponse::ActivePlacements(vec!["qwen2.5-7b".to_owned()])
    );
}
