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
    PrefixRadixIndex, PromptPreparationRequest, SimpleTokenizer, SpecialTokens, Tokenizer,
    format_chat_prompt, validate_and_tokenize,
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

#[test]
fn test_sub_issue_73_true_lru_recency_and_multi_pass_eviction() {
    let index = PrefixRadixIndex::new();

    let b0: Vec<u32> = (0..16).collect();
    let b1: Vec<u32> = (16..32).collect();
    let b2: Vec<u32> = (32..48).collect();

    // 1. Test True LRU Recency: queries update access tick
    // Insert block A at tick 1, block B at tick 2
    index.insert("p1", "m1", &b0, &[10]);
    index.insert("p1", "m1", &b1, &[20]);

    // Access block A repeatedly via match_longest_prefix
    for _ in 0..5 {
        let res = index.match_longest_prefix("p1", "m1", &b0);
        assert_eq!(res.matched_blocks, vec![10]);
    }

    // Evict 1 block: block B (unaccessed) must be evicted, not block A (recently accessed)
    let evicted = index.evict_lru("p1", "m1", 1);
    assert_eq!(evicted, vec![20]);
    assert_eq!(
        index.match_longest_prefix("p1", "m1", &b0).matched_blocks,
        vec![10]
    );

    // 2. Test Multi-Pass Leaf Eviction across tree depth
    let mut seq = Vec::new();
    seq.extend_from_slice(&b0);
    seq.extend_from_slice(&b1);
    seq.extend_from_slice(&b2);

    let deep_index = PrefixRadixIndex::new();
    deep_index.insert("p2", "m2", &seq, &[100, 101, 102]);
    assert_eq!(deep_index.stats().total_cached_blocks, 3);

    // Evict 2 blocks from 3-block deep linear chain (initial leaves = 1)
    let evicted_deep = deep_index.evict_lru("p2", "m2", 2);
    assert_eq!(evicted_deep, vec![102, 101]);
    assert_eq!(deep_index.stats().total_cached_blocks, 1);
    assert_eq!(
        deep_index
            .match_longest_prefix("p2", "m2", &seq)
            .matched_blocks,
        vec![100]
    );
}

#[test]
fn test_sub_issue_73_invalidate_model_across_all_projects_and_clear_cache_admin() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");

    let b0: Vec<u32> = (0..32).collect();

    // Populate prefix blocks for model-shared across two different projects
    runtime.insert_prefix("project-one", "model-shared", &b0, &[1, 2]);
    runtime.insert_prefix("project-two", "model-shared", &b0, &[3, 4]);
    runtime.insert_prefix("project-one", "model-other", &b0, &[5, 6]);

    let stats_before = runtime.prefix_cache_stats();
    assert_eq!(stats_before.total_projects, 2);
    assert_eq!(stats_before.total_models, 2);
    assert_eq!(stats_before.total_cached_blocks, 6);

    // Clear model-shared across all projects (project_id = None, model_id = Some)
    let clear_resp = runtime
        .execute_admin(AdminCommand::ClearPrefixCache {
            project_id: None,
            model_id: Some("model-shared".to_owned()),
        })
        .unwrap();
    assert_eq!(clear_resp, AdminResponse::CacheCleared(4));

    // Entries for model-shared in both project-one and project-two are gone
    assert!(
        runtime
            .match_prefix("project-one", "model-shared", &b0)
            .is_empty()
    );
    assert!(
        runtime
            .match_prefix("project-two", "model-shared", &b0)
            .is_empty()
    );

    // model-other in project-one remains intact
    assert_eq!(
        runtime
            .match_prefix("project-one", "model-other", &b0)
            .matched_blocks,
        vec![5, 6]
    );
}

#[test]
fn test_sub_issue_74_worker_thread_panic_resilience() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");

    // 1. Dispatch task that panics
    let panic_res = runtime.dispatch_control_task(|_| -> () {
        panic!("simulated transient task panic");
    });
    assert!(panic_res.is_err());
    let err_msg = format!("{:?}", panic_res.unwrap_err());
    assert!(err_msg.contains("panicked"));

    // 2. Dispatch subsequent task to the worker pool: thread is still alive and responds cleanly
    let ok_res = runtime.dispatch_control_task(|core| core);
    assert!(ok_res.is_ok());
    let core = ok_res.unwrap();
    assert!(core == 6 || core == 7);
}

#[test]
fn test_sub_issue_72_custom_template_priority_and_prepared_prompt() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");

    let mut tok = create_test_simple_tokenizer();
    tok.add_token("<custom>", 2000);
    tok.add_token("</custom>", 2001);
    runtime.register_tokenizer("model-custom", Arc::new(tok));

    // Register custom chat template
    let custom_template = ChatTemplate::new(Some("<custom> {{prompt}} </custom>".to_owned()));
    runtime.register_template("model-custom", custom_template);

    // Both format_prompt and tokenize_prompt use the custom template
    let formatted = runtime
        .format_prompt("model-custom", "hello world")
        .unwrap();
    assert_eq!(formatted, "<custom> hello world </custom>");

    // prepare_prompt returns both formatted_text and token_ids
    let prep_req = PromptPreparationRequest::new("hello world", true, 64);
    let prepared = runtime.prepare_prompt("model-custom", &prep_req).unwrap();
    assert_eq!(prepared.formatted_text, "<custom> hello world </custom>");
    assert_eq!(prepared.token_count(), 4);
    assert_eq!(prepared.token_ids, vec![2000, 1000, 1001, 2001]);

    // Get tokenizer, decode, and decode_piece
    let decoded = runtime.decode("model-custom", &[1000, 1001]).unwrap();
    assert_eq!(decoded, "hello world");

    let piece = runtime.decode_piece("model-custom", 1000).unwrap();
    assert_eq!(piece, Some("hello".to_owned()));

    let stop_piece = runtime.decode_piece("model-custom", 2).unwrap();
    assert_eq!(stop_piece, None); // stop token returns None
}

#[test]
fn test_sub_issue_74_dispatched_control_plane_methods() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");

    let mut tok = create_test_simple_tokenizer();
    tok.add_token("[INST]", 2002);
    tok.add_token("[/INST]", 2003);
    runtime.register_tokenizer("m-disp", Arc::new(tok));
    runtime.register_template(
        "m-disp",
        ChatTemplate::new(Some("[INST] {{prompt}} [/INST]".to_owned())),
    );

    // 1. dispatch_format_prompt
    let formatted = runtime
        .dispatch_format_prompt("m-disp".to_owned(), "test message".to_owned())
        .unwrap();
    assert_eq!(formatted, "[INST] test message [/INST]");

    // 2. dispatch_tokenize_prompt
    let tokens = runtime
        .dispatch_tokenize_prompt("m-disp".to_owned(), "hello world".to_owned(), false, 16)
        .unwrap();
    assert_eq!(tokens, vec![1000, 1001]);

    // 3. dispatch_prepare_prompt
    let prep = runtime
        .dispatch_prepare_prompt("m-disp".to_owned(), "hello world".to_owned(), true, 16)
        .unwrap();
    assert_eq!(prep.formatted_text, "[INST] hello world [/INST]");
    assert_eq!(prep.token_ids, vec![2002, 1000, 1001, 2003]);

    // 4. dispatch_match_prefix
    let b0: Vec<u32> = (0..16).collect();
    runtime.insert_prefix("proj-x", "m-disp", &b0, &[99]);
    let match_res = runtime
        .dispatch_match_prefix("proj-x".to_owned(), "m-disp".to_owned(), b0)
        .unwrap();
    assert_eq!(match_res.matched_blocks, vec![99]);
}

#[test]
fn test_sub_issue_73_strict_lru_branching_eviction() {
    let index = PrefixRadixIndex::new();

    // Branch 1: Node P (block 1) -> Leaf X (block 2)
    let b1: Vec<u32> = (0..16).collect();
    let b2: Vec<u32> = (16..32).collect();
    let mut seq1 = Vec::new();
    seq1.extend_from_slice(&b1);
    seq1.extend_from_slice(&b2);
    index.insert("p-lru", "m-lru", &seq1, &[1, 2]);

    // Branch 2: Root Node Y (block 3)
    let b3: Vec<u32> = (32..48).collect();
    index.insert("p-lru", "m-lru", &b3, &[3]);

    assert_eq!(index.stats().total_cached_blocks, 3);

    // Initial state:
    // P (block 1, tick 1) -> X (block 2, tick 2)
    // Y (block 3, tick 3)
    // Leaves at start: X (tick 2), Y (tick 3)
    // Request eviction of 2 blocks:
    // 1st eviction: X (block 2) is evicted because tick 2 < tick 3.
    // Once X is evicted, P (block 1, tick 1) is now a leaf!
    // 2nd eviction: leaves are P (tick 1) and Y (tick 3).
    // P (tick 1) is OLDER than Y (tick 3), so P MUST be evicted next!
    // Strict LRU eviction order: [2, 1]. Y (block 3) remains!
    let evicted = index.evict_lru("p-lru", "m-lru", 2);
    assert_eq!(evicted, vec![2, 1]);
    assert_eq!(index.stats().total_cached_blocks, 1);

    // Verify Y (block 3) is intact and can be matched
    let match_y = index.match_longest_prefix("p-lru", "m-lru", &b3);
    assert_eq!(match_y.matched_blocks, vec![3]);

    // Verify seq1 is completely gone
    let match_seq1 = index.match_longest_prefix("p-lru", "m-lru", &seq1);
    assert!(match_seq1.is_empty());
}

#[test]
fn test_sub_issue_72_general_tokenizer_default_template() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");

    let tok = create_test_simple_tokenizer();
    runtime.register_tokenizer("plain-model", Arc::new(tok));

    // format_prompt returns prompt unchanged (identity template) instead of ModelNotFound
    let formatted = runtime
        .format_prompt("plain-model", "test message")
        .unwrap();
    assert_eq!(formatted, "test message");

    // get_template returns default ChatTemplate::new(None)
    let tmpl = runtime.get_template("plain-model").unwrap();
    assert_eq!(tmpl.format_prompt("sample"), "sample");
    assert!(tmpl.raw_template().is_none());
}

#[test]
fn test_sub_issue_74_admin_register_model_validation() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");

    // Traversal rejection
    let res_traversal = runtime.execute_admin(AdminCommand::RegisterModel {
        model_id: "../escaped-model".to_owned(),
        weights_path: std::path::PathBuf::from("/tmp/weights.gguf"),
        cluster: "x100".to_owned(),
    });
    assert!(res_traversal.is_err());

    // Whitespace rejection
    let res_whitespace = runtime.execute_admin(AdminCommand::RegisterModel {
        model_id: "  spaced  ".to_owned(),
        weights_path: std::path::PathBuf::from("/tmp/weights.gguf"),
        cluster: "x100".to_owned(),
    });
    assert!(res_whitespace.is_err());

    // Empty model_id rejection
    let res_empty = runtime.execute_admin(AdminCommand::RegisterModel {
        model_id: String::new(),
        weights_path: std::path::PathBuf::from("/tmp/weights.gguf"),
        cluster: "x100".to_owned(),
    });
    assert!(res_empty.is_err());

    // Valid model_id succeeds
    let res_ok = runtime.execute_admin(AdminCommand::RegisterModel {
        model_id: "valid-qwen-7b".to_owned(),
        weights_path: std::path::PathBuf::from("/tmp/weights.gguf"),
        cluster: "x100".to_owned(),
    });
    assert_eq!(
        res_ok.unwrap(),
        AdminResponse::ModelRegistered("valid-qwen-7b".to_owned())
    );
}

#[test]
fn test_sub_issue_72_text_stop_sequence_and_dispatched_detection() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");

    let mut tok = create_test_simple_tokenizer();
    let special = SpecialTokens {
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        eot_token_id: Some(3),
        padding_token_id: Some(0),
        add_bos_token: false,
        stop_tokens: vec![2, 3],
        stop_sequences: vec!["<|im_end|>".to_owned(), "### STOP".to_owned()],
    };
    tok.set_special_tokens(special);
    runtime.register_tokenizer("m-stop", Arc::new(tok));

    // 1. Synchronous text stop sequence checks
    assert!(
        runtime
            .is_stop_sequence("m-stop", "response text <|im_end|>\n")
            .unwrap()
    );
    assert!(
        runtime
            .is_stop_sequence("m-stop", "halt now ### STOP immediately")
            .unwrap()
    );
    assert!(!runtime.is_stop_sequence("m-stop", "normal output").unwrap());

    // 2. Dispatched stop checks on dedicated active control threads
    let disp_token_stop = runtime
        .dispatch_is_stop_token("m-stop".to_owned(), 2)
        .unwrap();
    assert!(disp_token_stop);

    let disp_token_ok = runtime
        .dispatch_is_stop_token("m-stop".to_owned(), 1000)
        .unwrap();
    assert!(!disp_token_ok);

    let disp_seq_stop = runtime
        .dispatch_is_stop_sequence("m-stop".to_owned(), "final tokens <|im_end|>".to_owned())
        .unwrap();
    assert!(disp_seq_stop);

    let disp_seq_ok = runtime
        .dispatch_is_stop_sequence("m-stop".to_owned(), "keep generating".to_owned())
        .unwrap();
    assert!(!disp_seq_ok);

    // 3. IncrementalStopDetector push_token_with_text
    let mut detector = runtime.create_stop_detector("m-stop").unwrap();
    assert!(!detector.is_stopped());
    assert!(!detector.push_token_with_text(1000, Some("hello")));
    assert!(!detector.is_stopped());

    // Stop via text fragment
    assert!(detector.push_token_with_text(1001, Some("<|im_end|>")));
    assert!(detector.is_stopped());
}

#[test]
fn test_sub_issue_74_dispatched_admin_commands_on_active_control_threads() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime creation");

    // 1. Dispatched ping
    let ping_resp = runtime.dispatch_execute_admin(AdminCommand::Ping).unwrap();
    assert_eq!(ping_resp, AdminResponse::Pong);

    // 2. Dispatched InspectCapacity
    match runtime
        .dispatch_execute_admin(AdminCommand::InspectCapacity)
        .unwrap()
    {
        AdminResponse::Capacity(cap) => {
            assert_eq!(cap.active_control_mask, 0x00C0);
        }
        other => panic!("expected Capacity response, got {other:?}"),
    }

    // 3. Dispatched RegisterModel
    let reg_resp = runtime
        .dispatch_execute_admin(AdminCommand::RegisterModel {
            model_id: "disp-model".to_owned(),
            weights_path: std::path::PathBuf::from("/tmp/weights.gguf"),
            cluster: "x100".to_owned(),
        })
        .unwrap();
    assert_eq!(
        reg_resp,
        AdminResponse::ModelRegistered("disp-model".to_owned())
    );

    // 4. Dispatched GetActivePlacements
    let placements_resp = runtime
        .dispatch_execute_admin(AdminCommand::GetActivePlacements)
        .unwrap();
    assert_eq!(
        placements_resp,
        AdminResponse::ActivePlacements(vec!["disp-model".to_owned()])
    );

    // 5. Dispatched prefix insert
    let b0: Vec<u32> = (0..16).collect();
    let inserted = runtime
        .dispatch_insert_prefix("disp-p", "disp-model", b0.clone(), vec![77])
        .unwrap();
    assert_eq!(inserted, 1);

    // 6. Dispatched prefix match
    let match_res = runtime
        .dispatch_match_prefix("disp-p", "disp-model", b0)
        .unwrap();
    assert_eq!(match_res.matched_blocks, vec![77]);

    // 7. Dispatched ClearPrefixCache
    let clear_resp = runtime
        .dispatch_execute_admin(AdminCommand::ClearPrefixCache {
            project_id: Some("disp-p".to_owned()),
            model_id: Some("disp-model".to_owned()),
        })
        .unwrap();
    assert_eq!(clear_resp, AdminResponse::CacheCleared(1));
}
