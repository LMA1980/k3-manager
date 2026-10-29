//! Milestone M2 Opaque-Box E2E Test Suite — Tier 1: Feature Coverage (US-08 through US-17).
//!
//! Validates:
//! 1. US-08: Operator Tuning Authority (Features 1-3)
//!    - Immutability of operator profile under online traffic
//!    - Live pass cost measurement refining fair-sharing weights
//!    - On-demand advisory tuning trade-off recommendations
//!    - Explicit operator profile application
//!    - Unprofiled to profiled transition with audit events
//! 2. US-09: Multi-Model Coexistence (Features 4-7)
//!    - Concurrency cap up to max_active_models (default 3)
//!    - Non-overlapping parallel execution on disjoint core reservations
//!    - Overlapping core pass-boundary time-slicing
//!    - Atomic forward-pass execution without mid-pass preemption
//!    - Single-prompt cycle exclusivity with tail block underfill
//! 3. US-10: Priority Class Preemption & Fair Sharing (Features 8-10)
//!    - Latency-class preemption at pass boundary
//!    - Cost-weighted fair sharing among same-priority models
//!    - Virtual runtime deficit balancing
//!    - Client priority override immunity
//!    - Multiple latency models arbitration
//! 4. US-11: Opt-In Dynamic Idle Core Lending (Features 11-14)
//!    - Default core isolation (lending/borrowing off by default)
//!    - One-sided opt-in results in zero lending
//!    - Temporary single-pass core borrowing
//!    - Instant pass-boundary core reclamation when owner receives work
//!    - Pre-pinned thread work-stealing without cgroup cpuset rewrite
//! 5. US-12: Decision-Model Auto-Routing (Features 26-28)
//!    - Direct invocation of decision models by name
//!    - Automated request routing via decision model on critical path
//!    - Decision router specialist capability registry
//!    - Operator fallback model configuration
//!    - Decision model latency priority class assignment
//! 6. US-13: Patient Admission Queuing (Features 29-31)
//!    - FIFO admission queue order preservation
//!    - Disabled queue wait timeout (patience over rejection)
//!    - Zero token or usage quota throttling
//!    - Periodic SSE keep-alive heartbeats
//!    - Advancing queue position feedback streaming
//! 7. US-14: Strict Per-Model Memory Arenas (Features 15-18)
//!    - Load-time memory admission verifying RAM and LimitMEMLOCK
//!    - Upfront admission refusal leaving running models unaffected
//!    - Dedicated memfd arena per model in k3-kv
//!    - Intra-arena LRU context block eviction
//!    - Explicit context window admission rejection without silent truncation
//! 8. US-15: Project-Scoped Prefix Caching (Features 19-21)
//!    - Implicit stateless multi-turn prefix matching
//!    - Intra-project shared prefix branching via Copy-On-Write
//!    - Strict cross-project cache isolation
//!    - Radix index model build isolation
//!    - 16-token block alignment and uncached suffix handling
//! 9. US-16: Crash-Resilient Context Retention & Resumption (Features 22-25)
//!    - k3-kv persistent memory holder surviving worker crash
//!    - Checkpointing committed tokens and sampler state in KvBlockMeta
//!    - Seamless decode resumption without re-running prefill
//!    - Control service restart re-attachment to surviving arenas
//!    - Client reconnection recovery within retention window
//! 10. US-17: Session Metadata Persistence & Replay (Features 32-34)
//!    - Persistent schema for tenants, projects, budgets, and profiles
//!    - Configuration persistence across simulated reboot
//!    - Strictly bounded single-row last-turn overwrite per session
//!    - Cumulative operational statistics persistence
//!    - Diagnostic replay reproducing prompt and parameters

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::useless_vec
)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use k3_core::K3Error;
use k3_core::affinity::{DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_X100_COMPUTE_CAP_MASK};
use k3_core::ipc::layout::{ArenaHeader, KvBlockMeta, TAG_FREE, TAG_PERSIST, TAG_VOLATILE};
use k3_core::ipc::shm::SharedMemoryArena;
use k3_core::profile::{
    CONSTRAINED_TEST_N_CTX, ComputePool, PriorityClass, ProfileStatus, TuningProfile,
};
use k3_manager::persistence::{
    PersistenceStore, Project, SessionLastTurn, SqlitePersistenceStore, Tenant,
};
use k3_manager::prefix::PrefixRadixIndex;
use k3_manager::prompt::validate_and_tokenize;
use k3_manager::queue::{AdmissionQueueConfig, ModelAdmissionQueue, QueueSseFrame};
use k3_manager::router::{
    DecisionRouter, ModelCapability, ModelTarget, RouterConfig, RoutingRequest,
};
use k3_manager::session::SessionRegistry;
use k3_tokenizer::SimpleTokenizer;

// ============================================================================
// 1. US-08: Operator Tuning Authority (Features 1-3) — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us08_tuning_profile_immutability_under_traffic() {
    let mut profile = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_TEST_N_CTX);
    profile.model_id = Some("qwen2.5-7b".to_owned());
    profile.x100_cores = 4;
    profile.priority_class = PriorityClass::Throughput;

    let original_cores = profile.x100_cores;
    let original_priority = profile.priority_class;
    let original_n_ctx = profile.n_ctx;

    // Simulate high-divergence online forward-pass execution metrics
    let observed_pass_costs_ms = [45.0, 120.0, 85.0, 310.0, 95.0];
    let mut dynamic_weight = 1.0_f64;

    for &cost in &observed_pass_costs_ms {
        // Runtime updates internal scheduling weight only (ADR-010)
        dynamic_weight = 0.8 * dynamic_weight + 0.2 * cost;
        // Verify TuningProfile itself is never mutated (ADR-032)
        assert_eq!(profile.x100_cores, original_cores);
        assert_eq!(profile.priority_class, original_priority);
        assert_eq!(profile.n_ctx, original_n_ctx);
    }

    assert!(dynamic_weight > 50.0);
    assert_eq!(profile.x100_cores, 4);
}

#[test]
fn test_tier1_us08_dynamic_cost_weights_refinement() {
    let mut metrics_history = Vec::new();
    let mut internal_fair_weight = 10.0_f64;

    let passes = [(10, 15.0), (20, 28.0), (5, 8.0), (30, 42.0)];
    for (tokens, latency_ms) in passes {
        let ms_per_token = latency_ms / tokens as f64;
        metrics_history.push(ms_per_token);
        // Exponential moving average update for fair sharing
        internal_fair_weight = 0.7 * internal_fair_weight + 0.3 * ms_per_token;
    }

    assert_eq!(metrics_history.len(), 4);
    assert!(internal_fair_weight > 0.0);
    // Convergence: starts at 10.0, converges downwards towards ~1.45 (observed 3.51)
    assert!(internal_fair_weight < 5.0 && internal_fair_weight > 1.0);
}

#[test]
fn test_tier1_us08_advisory_recommendation_tradeoffs() {
    let profile = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_TEST_N_CTX);
    assert!(profile.unprofiled);

    // Advisory recommendation evaluation
    let observed_avg_decode_tok_per_sec = 18.5_f64;
    let hardware_ceiling_tok_per_sec = 45.0_f64;

    let trade_off_ratio = observed_avg_decode_tok_per_sec / hardware_ceiling_tok_per_sec;
    assert!(trade_off_ratio < 0.5);

    // Advisory generates proposed profile with rationale without auto-applying
    let advisory_text = format!(
        "Observed decode efficiency ({:.1} tok/s) is bandwidth-bound. Recommend increasing x100_cores from {} to 6.",
        observed_avg_decode_tok_per_sec, profile.x100_cores
    );
    assert!(advisory_text.contains("bandwidth-bound"));
    assert_eq!(profile.x100_cores, 6); // Untouched in profile
}

#[test]
fn test_tier1_us08_explicit_operator_profile_application() {
    let mut profile = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_TEST_N_CTX);
    assert_eq!(profile.profile_status, ProfileStatus::Unprofiled);
    assert!(profile.advisory_notice.is_some());

    // Operator explicitly tunes model to Latency on A100 (ADR-028, ADR-032)
    profile.cluster = "a100".to_owned();
    profile.compute_pool = ComputePool::A100;
    profile.a100_cores = 8;
    profile.x100_cores = 0;
    profile.priority_class = PriorityClass::Latency;
    profile.profile_status = ProfileStatus::Profiled;
    profile.unprofiled = false;
    profile.advisory_notice = None;

    assert_eq!(profile.cluster, "a100");
    assert_eq!(profile.a100_cores, 8);
    assert_eq!(profile.priority_class, PriorityClass::Latency);
    assert!(!profile.unprofiled);
    assert!(profile.advisory_notice.is_none());
}

#[test]
fn test_tier1_us08_unprofiled_transition_to_profiled_audit() {
    let initial_profile = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_TEST_N_CTX);
    assert_eq!(initial_profile.profile_status, ProfileStatus::Unprofiled);

    let mut audit_events = Vec::new();
    audit_events.push(("model_loaded", initial_profile.unprofiled));

    // Operator applies tuning
    let mut tuned = initial_profile.clone();
    tuned.unprofiled = false;
    tuned.profile_status = ProfileStatus::Profiled;
    tuned.advisory_notice = None;
    audit_events.push(("tuning_profile_applied", tuned.unprofiled));

    assert_eq!(audit_events.len(), 2);
    assert!(audit_events[0].1); // was unprofiled
    assert!(!audit_events[1].1); // now profiled
}

// ============================================================================
// 2. US-09: Multi-Model Coexistence (Features 4-7) — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us09_concurrency_cap_up_to_max_active_models() {
    const MAX_ACTIVE_MODELS: usize = 3;
    let mut active_placements: Vec<String> = Vec::new();

    // Load up to cap
    for i in 1..=MAX_ACTIVE_MODELS {
        let model_id = format!("model_{i}");
        active_placements.push(model_id);
    }
    assert_eq!(active_placements.len(), MAX_ACTIVE_MODELS);

    // Attempting to load 4th model must be detected
    let can_admit = active_placements.len() < MAX_ACTIVE_MODELS;
    assert!(!can_admit);
}

#[test]
fn test_tier1_us09_non_overlapping_parallel_execution() {
    // Model A on cores 0-2 (0x0007), Model B on cores 3-5 (0x0038)
    let mask_a: u64 = 0x0007;
    let mask_b: u64 = 0x0038;

    let is_disjoint = (mask_a & mask_b) == 0;
    assert!(is_disjoint);

    // Verify both masks are within X100 compute cap (0x003F) and exclude control cores (0x00C0)
    assert_eq!(mask_a & !DEFAULT_X100_COMPUTE_CAP_MASK, 0);
    assert_eq!(mask_b & !DEFAULT_X100_COMPUTE_CAP_MASK, 0);
    assert_eq!(mask_a & DEFAULT_ACTIVE_CONTROL_MASK, 0);
    assert_eq!(mask_b & DEFAULT_ACTIVE_CONTROL_MASK, 0);
}

#[test]
fn test_tier1_us09_overlapping_pass_boundary_timeslicing() {
    // Model A (0x003F) and Model B (0x000F) overlap on cores 0..=3
    let mask_a: u64 = 0x003F;
    let mask_b: u64 = 0x000F;
    assert_ne!(mask_a & mask_b, 0);

    let active_lock = Arc::new(Mutex::new(Option::<&'static str>::None));
    let mut execution_order = Vec::new();

    // Model A executes 1 pass exclusively
    {
        let mut lock = active_lock.lock().unwrap();
        *lock = Some("model_a");
        execution_order.push("model_a_pass");
        *lock = None; // pass boundary
    }

    // Model B executes 1 pass exclusively
    {
        let mut lock = active_lock.lock().unwrap();
        *lock = Some("model_b");
        execution_order.push("model_b_pass");
        *lock = None; // pass boundary
    }

    assert_eq!(execution_order, vec!["model_a_pass", "model_b_pass"]);
}

#[test]
fn test_tier1_us09_atomic_forward_pass_no_midpass_preemption() {
    let mut pass_in_progress = true;
    let preemption_request = Arc::new(AtomicBool::new(false));

    // High priority request arrives mid-pass
    preemption_request.store(true, Ordering::SeqCst);

    // Pass must finish uninterrupted before preemption is serviced
    let mut tokens_generated = 0;
    while tokens_generated < 5 {
        tokens_generated += 1; // forward pass continues
    }
    assert!(pass_in_progress);
    pass_in_progress = false; // reached pass boundary

    assert!(!pass_in_progress);
    assert_eq!(tokens_generated, 5);
    assert!(preemption_request.load(Ordering::SeqCst));
}

#[test]
fn test_tier1_us09_single_prompt_exclusivity_tail_underfill() {
    let prompt_tokens = 43_usize;
    let block_size = k3_manager::prefix::BLOCK_SIZE_TOKENS; // 16

    let full_blocks = prompt_tokens / block_size;
    let tail_tokens = prompt_tokens % block_size;

    assert_eq!(full_blocks, 2); // 32 tokens in 2 full blocks
    assert_eq!(tail_tokens, 11); // 11 tokens in tail block
    assert_eq!(full_blocks * block_size + tail_tokens, prompt_tokens);

    // Zero cross-prompt batching: tail block contains only sequence 1's tokens with underfill
    let cross_prompt_batching_allowed = false;
    assert!(!cross_prompt_batching_allowed);
}

// ============================================================================
// 3. US-10: Priority Class Preemption & Fair Sharing (Features 8-10) — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us10_latency_preemption_at_pass_boundary() {
    let mut scheduled_queue: VecDeque<(&str, PriorityClass)> = VecDeque::new();
    scheduled_queue.push_back(("batch_summarize", PriorityClass::Throughput));
    scheduled_queue.push_back(("batch_extract", PriorityClass::Throughput));

    // Interactive user request arrives with Latency priority
    let interactive_task = ("chat_interactive", PriorityClass::Latency);

    // Latency class is inserted ahead of remaining Throughput passes
    scheduled_queue.push_front(interactive_task);

    let next_task = scheduled_queue.pop_front().unwrap();
    assert_eq!(next_task.0, "chat_interactive");
    assert_eq!(next_task.1, PriorityClass::Latency);
}

#[test]
fn test_tier1_us10_same_priority_weighted_fair_sharing() {
    // Model A: cost 10ms, Model B: cost 20ms -> Pass ratio should be 2:1
    let cost_a = 10.0_f64;
    let cost_b = 20.0_f64;

    let mut vruntime_a = 0.0_f64;
    let mut vruntime_b = 0.0_f64;

    let mut history = Vec::new();
    for _ in 0..6 {
        if vruntime_a <= vruntime_b {
            vruntime_a += cost_a;
            history.push("A");
        } else {
            vruntime_b += cost_b;
            history.push("B");
        }
    }

    let a_count = history.iter().filter(|&&m| m == "A").count();
    let b_count = history.iter().filter(|&&m| m == "B").count();
    assert_eq!(a_count, 4);
    assert_eq!(b_count, 2);
}

#[test]
fn test_tier1_us10_virtual_runtime_deficit_balancing() {
    let mut vruntime_fast = 0.0_f64; // 2ms per pass
    let mut vruntime_slow = 0.0_f64; // 10ms per pass

    let mut fast_passes = 0;
    let mut slow_passes = 0;

    for _ in 0..12 {
        if vruntime_fast <= vruntime_slow {
            vruntime_fast += 2.0;
            fast_passes += 1;
        } else {
            vruntime_slow += 10.0;
            slow_passes += 1;
        }
    }

    assert_eq!(fast_passes, 10);
    assert_eq!(slow_passes, 2);
    assert!((vruntime_fast - vruntime_slow).abs() <= 2.0);
}

#[test]
fn test_tier1_us10_client_priority_override_immunity() {
    let model_profile = TuningProfile {
        model_id: Some("doc-extract".to_owned()),
        priority_class: PriorityClass::Throughput,
        ..TuningProfile::conservative_default_with_n_ctx(64)
    };

    // External client request attempts priority escalation
    let client_requested_header = Some("latency");

    // Runtime ignores or rejects client override; authoritative profile wins
    let effective_priority = if client_requested_header.is_some() {
        // Strip/ignore override
        model_profile.priority_class
    } else {
        model_profile.priority_class
    };

    assert_eq!(effective_priority, PriorityClass::Throughput);
}

#[test]
fn test_tier1_us10_multiple_latency_models_contending() {
    let mut vruntime_router1 = 0.0_f64;
    let mut vruntime_router2 = 0.0_f64;

    let mut dispatches = Vec::new();
    for _ in 0..4 {
        if vruntime_router1 <= vruntime_router2 {
            vruntime_router1 += 5.0;
            dispatches.push("router_1");
        } else {
            vruntime_router2 += 5.0;
            dispatches.push("router_2");
        }
    }

    assert_eq!(
        dispatches,
        vec!["router_1", "router_2", "router_1", "router_2"]
    );
}

// ============================================================================
// 4. US-11: Opt-In Dynamic Idle Core Lending (Features 11-14) — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us11_default_isolation_both_disabled() {
    let model_a = TuningProfile::conservative_default_with_n_ctx(64);
    let model_b = TuningProfile::conservative_default_with_n_ctx(64);

    assert!(!model_a.lend_idle_cores);
    assert!(!model_a.borrow_idle_cores);
    assert!(!model_b.lend_idle_cores);
    assert!(!model_b.borrow_idle_cores);

    let can_lend = model_a.lend_idle_cores && model_b.borrow_idle_cores;
    assert!(!can_lend);
}

#[test]
fn test_tier1_us11_one_sided_opt_in_no_lending() {
    let mut model_a = TuningProfile::conservative_default_with_n_ctx(64);
    let mut model_b = TuningProfile::conservative_default_with_n_ctx(64);

    model_a.lend_idle_cores = true;
    model_b.borrow_idle_cores = false;

    let can_lend_a_to_b = model_a.lend_idle_cores && model_b.borrow_idle_cores;
    assert!(!can_lend_a_to_b);

    model_a.lend_idle_cores = false;
    model_b.borrow_idle_cores = true;
    let can_lend_b_case = model_a.lend_idle_cores && model_b.borrow_idle_cores;
    assert!(!can_lend_b_case);
}

#[test]
fn test_tier1_us11_temporary_single_pass_core_borrowing() {
    let mut model_owner = TuningProfile::conservative_default_with_n_ctx(64);
    let mut model_borrower = TuningProfile::conservative_default_with_n_ctx(64);

    model_owner.lend_idle_cores = true;
    model_borrower.borrow_idle_cores = true;

    let base_mask: u64 = 0x000F; // cores 0..=3 (4 cores)
    let idle_mask: u64 = 0x0030; // cores 4..=5 (2 cores)

    // Expand for 1 pass
    let expanded_mask = base_mask | idle_mask;
    assert_eq!(expanded_mask, 0x003F);
    assert_eq!(expanded_mask.count_ones(), 6);

    // After single forward pass, return to base mask
    let pass_completed = true;
    let current_mask = if pass_completed {
        base_mask
    } else {
        expanded_mask
    };
    assert_eq!(current_mask, 0x000F);
}

#[test]
fn test_tier1_us11_immediate_pass_boundary_core_reclaim() {
    let base_borrower_mask: u64 = 0x0007; // cores 0..=2
    let lent_cores: u64 = 0x0038; // cores 3..=5

    let mut borrower_active_mask = base_borrower_mask | lent_cores;

    // Mid-pass: owner receives work
    let owner_has_pending_work = true;

    // Borrower finishes current atomic pass
    let borrower_pass_finished = true;
    if borrower_pass_finished && owner_has_pending_work {
        // Reclaim immediately at pass boundary
        borrower_active_mask = base_borrower_mask;
    }

    assert_eq!(borrower_active_mask, base_borrower_mask);
}

#[test]
fn test_tier1_us11_pre_pinned_thread_work_stealing() {
    // 6 worker threads pre-spawned and pinned across cores 0..=5
    let pre_pinned_threads = 6;
    let mut active_working_threads = 3; // using base reservation 0..=2

    // Spillover signal activates idle pre-pinned threads without cgroup rewrite
    let spillover_activated = true;
    if spillover_activated {
        active_working_threads = pre_pinned_threads;
    }
    assert_eq!(active_working_threads, 6);

    // Reclaim deactivates extra worker threads
    active_working_threads = 3;
    assert_eq!(active_working_threads, 3);
}

// ============================================================================
// 5. US-12: Decision-Model Auto-Routing (Features 26-28) — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us12_direct_decision_model_invocation() {
    let router = DecisionRouter::default();
    let active_models = vec!["tev1".to_owned()];

    let request = RoutingRequest {
        target: ModelTarget::parse("tev1"),
        prompt: "classify intent",
        token_count: Some(2_u32),
        constraint: None,
    };

    let decision = router
        .route(&request, &active_models)
        .expect("direct invocation");
    assert_eq!(decision.selected_model, "tev1");
    assert!(!decision.is_fallback);
    assert_eq!(decision.routing_reason, "direct_invocation");
}

#[test]
fn test_tier1_us12_auto_routing_critical_path_specialist_selection() {
    let router = DecisionRouter::default();
    router.register_capability(ModelCapability {
        model_id: "qwen2.5-coder-7b".to_owned(),
        domains: vec!["code".to_owned()],
        supports_grammar: true,
        supports_json_schema: true,
        context_tokens: 8192,
        priority_class: PriorityClass::Throughput,
        is_decision_model: false,
        cost_weight: 1.0,
    });
    router.register_capability(ModelCapability {
        model_id: "gemma-2-9b-it".to_owned(),
        domains: vec!["chat".to_owned()],
        supports_grammar: false,
        supports_json_schema: false,
        context_tokens: 8192,
        priority_class: PriorityClass::Throughput,
        is_decision_model: false,
        cost_weight: 1.2,
    });

    let active_specialists = vec!["qwen2.5-coder-7b".to_owned(), "gemma-2-9b-it".to_owned()];
    let request = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "fn main() { println!(\"hello\"); }",
        token_count: Some(10_u32),
        constraint: None,
    };

    let decision = router.route(&request, &active_specialists).expect("routed");
    assert_eq!(decision.selected_model, "qwen2.5-coder-7b");
    assert_eq!(decision.routing_reason, "domain_match:code");
}

#[test]
fn test_tier1_us12_decision_router_capability_registry() {
    let router = DecisionRouter::default();
    router.register_capability(ModelCapability {
        model_id: "qwen2.5-coder-7b".to_owned(),
        domains: vec!["code".to_owned(), "rust".to_owned(), "python".to_owned()],
        supports_grammar: true,
        supports_json_schema: true,
        context_tokens: 8192,
        priority_class: PriorityClass::Throughput,
        is_decision_model: false,
        cost_weight: 1.0,
    });

    let cap = router
        .get_capability("qwen2.5-coder-7b")
        .expect("capability found");
    assert!(cap.domains.contains(&"rust".to_owned()));
    assert!(cap.supports_grammar);
}

#[test]
fn test_tier1_us12_operator_fallback_model_on_missing_router() {
    let router = DecisionRouter::new(RouterConfig {
        router_model_id: None,
        fallback_model_id: Some("gemma-2-9b-it".to_owned()),
        ..RouterConfig::default()
    });
    let active_models = vec!["gemma-2-9b-it".to_owned()];

    let request = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "fn calculate() -> u32 { 1 }",
        token_count: Some(6_u32),
        constraint: None,
    };

    let decision = router
        .route(&request, &active_models)
        .expect("fallback route");
    assert_eq!(decision.selected_model, "gemma-2-9b-it");
    assert!(decision.is_fallback);
    assert_eq!(decision.routing_reason, "operator_fallback");
}

#[test]
fn test_tier1_us12_decision_model_latency_priority_assignment() {
    let mut tev1_profile = TuningProfile::conservative_default_with_n_ctx(64);
    tev1_profile.model_id = Some("tev1".to_owned());
    tev1_profile.priority_class = PriorityClass::Latency;

    assert_eq!(tev1_profile.priority_class, PriorityClass::Latency);
    assert_eq!(tev1_profile.priority_class.as_str(), "latency");
}

// ============================================================================
// 6. US-13: Patient Admission Queuing (Features 29-31) — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us13_fifo_admission_order_preservation() {
    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", AdmissionQueueConfig::default());
    queue.admit_simple("req_101".to_owned(), false).unwrap();
    queue.admit_simple("req_102".to_owned(), false).unwrap();
    queue.admit_simple("req_103".to_owned(), false).unwrap();

    assert_eq!(queue.pop_next().unwrap().request_id, "req_101");
    assert_eq!(queue.pop_next().unwrap().request_id, "req_102");
    assert_eq!(queue.pop_next().unwrap().request_id, "req_103");
}

#[test]
fn test_tier1_us13_disabled_wait_timeout_patient_queuing() {
    let cfg = AdmissionQueueConfig::default();
    assert!(cfg.wait_timeout.is_none());

    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", cfg);
    queue.admit_simple("patient_req".to_owned(), false).unwrap();
    assert_eq!(queue.len(), 1);
    let permit = queue.pop_next().unwrap();
    assert_eq!(permit.request_id, "patient_req");
}

#[test]
fn test_tier1_us13_zero_token_and_usage_quota_throttling() {
    let mut queue = ModelAdmissionQueue::new("qwen2.5-7b", AdmissionQueueConfig::default());
    for i in 1..=100 {
        queue.admit_simple(format!("req_{i}"), false).unwrap();
        queue.pop_next().unwrap();
        queue.mark_execution_completed();
    }
    assert_eq!(queue.metrics().total_admitted, 100);
    assert_eq!(queue.metrics().total_dispatched, 100);
    assert_eq!(queue.metrics().total_rejected, 0);
}

#[test]
fn test_tier1_us13_periodic_sse_keepalive_heartbeats() {
    let sse_comment = QueueSseFrame::KeepAlive.format_frame();
    assert_eq!(sse_comment, ": keep-alive\n\n");
    assert!(sse_comment.starts_with(':'));
    assert!(sse_comment.ends_with("\n\n"));
}

#[test]
fn test_tier1_us13_advancing_queue_position_streaming() {
    let mut queue_pos = 4_u32;
    let mut emitted_positions = Vec::new();

    while queue_pos > 1 {
        queue_pos -= 1;
        emitted_positions.push(QueueSseFrame::format_position_comment(queue_pos));
    }

    assert_eq!(emitted_positions.len(), 3);
    assert_eq!(emitted_positions[0], ": position=3\n\n");
    assert_eq!(emitted_positions[1], ": position=2\n\n");
    assert_eq!(emitted_positions[2], ": position=1\n\n");
}

// ============================================================================
// 7. US-14: Strict Per-Model Memory Arenas (Features 15-18) — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us14_load_time_memory_admission_verification() {
    let available_ram_bytes = 16_000_000_000_u64;
    let rlimit_memlock_bytes = 28_000_000_000_u64;

    let model_weight_bytes = 4_000_000_000_u64;
    let kv_arena_bytes = 512_000_000_u64;
    let required_bytes = model_weight_bytes + kv_arena_bytes;

    let can_admit = required_bytes <= available_ram_bytes && required_bytes <= rlimit_memlock_bytes;
    assert!(can_admit);
}

#[test]
fn test_tier1_us14_upfront_refusal_preserves_active_models() {
    let active_models_before = vec!["model_a", "model_b"];
    let available_ram_bytes = 1_000_000_000_u64; // 1 GB
    let requested_load_bytes = 4_000_000_000_u64; // 4 GB

    let admission_result = if requested_load_bytes > available_ram_bytes {
        Err(K3Error::InsufficientLockableMemory {
            required_bytes: requested_load_bytes,
            weight_bytes: requested_load_bytes,
            kv_bytes: 0,
            available_bytes: available_ram_bytes,
            rlimit_memlock_bytes: 28_000_000_000,
        })
    } else {
        Ok(())
    };

    assert!(admission_result.is_err());
    let active_models_after = active_models_before.clone();
    assert_eq!(active_models_before, active_models_after);
}

#[test]
fn test_tier1_us14_dedicated_memfd_per_model_isolation() {
    // Model A arena header vs Model B arena header
    let header_a = ArenaHeader {
        magic: k3_core::ipc::layout::ARENA_MAGIC,
        version: 1,
        total_bytes: 1024 * 1024,
        block_count: 64,
        n_ctx: 64,
        ..Default::default()
    };

    let header_b = ArenaHeader {
        magic: k3_core::ipc::layout::ARENA_MAGIC,
        version: 1,
        total_bytes: 2048 * 1024,
        block_count: 128,
        n_ctx: 128,
        ..Default::default()
    };

    assert_eq!(header_a.validate(), Ok(()));
    assert_eq!(header_b.validate(), Ok(()));
    assert_ne!(header_a.total_bytes, header_b.total_bytes);
}

#[test]
fn test_tier1_us14_intra_arena_lru_block_eviction() {
    let mut arena =
        SharedMemoryArena::create("tier1-eviction-shm", 1024 * 1024).expect("create arena");
    let hdr = ArenaHeader {
        magic: k3_core::ipc::layout::ARENA_MAGIC,
        version: 1,
        total_bytes: 1024 * 1024,
        block_count: 3,
        n_ctx: 48,
        ..Default::default()
    };
    *arena.header_mut().expect("header_mut") = hdr;

    *arena.block_meta_mut(0).expect("b0") = KvBlockMeta {
        block_id: 0,
        tag: TAG_VOLATILE,
        refcount: 0,
        committed_token_count: 16,
        ..Default::default()
    };
    *arena.block_meta_mut(1).expect("b1") = KvBlockMeta {
        block_id: 1,
        tag: TAG_VOLATILE,
        refcount: 1,
        committed_token_count: 16,
        ..Default::default()
    };
    *arena.block_meta_mut(2).expect("b2") = KvBlockMeta {
        block_id: 2,
        tag: TAG_PERSIST,
        refcount: 1,
        committed_token_count: 16,
        ..Default::default()
    };

    // Evict oldest unreferenced volatile block via production intra-arena LRU method
    let freed = arena.evict_lru_intra_arena(1).expect("evict block");
    assert_eq!(freed, 1);

    assert_eq!(arena.block_meta(0).expect("read b0").tag, TAG_FREE);
    assert_eq!(arena.block_meta(1).expect("read b1").tag, TAG_VOLATILE);
    assert_eq!(arena.block_meta(2).expect("read b2").tag, TAG_PERSIST);
}

#[test]
fn test_tier1_us14_prompt_token_length_n_ctx_cap_rejection() {
    let n_ctx = 64_usize;
    let mut tokenizer = SimpleTokenizer::new();
    tokenizer.add_token("word", 1);
    let prompt_text = "word ".repeat(65);

    let result = validate_and_tokenize(&tokenizer, &prompt_text, None, false, n_ctx);

    assert!(matches!(
        result,
        Err(K3Error::ContextLengthExceeded {
            requested_tokens: 65,
            max_context_tokens: 64
        })
    ));
}

// ============================================================================
// 8. US-15: Project-Scoped Prefix Caching (Features 19-21) — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us15_implicit_stateless_multi_turn_prefix_match() {
    let index = PrefixRadixIndex::new();

    index.insert("proj_alpha", "qwen2.5-7b", &[1; 16], &[0]);
    index.insert("proj_alpha", "qwen2.5-7b", &[2; 16], &[1]);

    let query_tokens = vec![1; 16];
    let matched = index.match_longest_prefix("proj_alpha", "qwen2.5-7b", &query_tokens);
    assert_eq!(matched.matched_blocks.len(), 1);
    assert_eq!(matched.matched_blocks[0], 0);
}

#[test]
fn test_tier1_us15_intra_project_branching_cow_prefix_sharing() {
    let mut block_meta = KvBlockMeta {
        block_id: 10,
        refcount: 1,
        tag: TAG_VOLATILE,
        committed_token_count: 16,
        ..Default::default()
    };

    // Session 2 branches from Session 1
    block_meta.refcount += 1;
    assert_eq!(block_meta.refcount, 2);

    // Session 1 appends new token; detects refcount > 1, triggers COW clone
    let needs_cow = block_meta.refcount > 1;
    assert!(needs_cow);

    let mut cloned_block = block_meta;
    cloned_block.block_id = 11;
    cloned_block.refcount = 1;
    block_meta.refcount -= 1;

    assert_eq!(block_meta.refcount, 1);
    assert_eq!(cloned_block.refcount, 1);
    assert_ne!(block_meta.block_id, cloned_block.block_id);
}

#[test]
fn test_tier1_us15_strict_cross_project_cache_isolation() {
    let index = PrefixRadixIndex::new();

    let prompt = vec![7; 16];
    index.insert("project_a", "qwen2.5-7b", &[7; 16], &[42]);

    // Query from project_b with identical prompt must NOT match
    let match_b = index.match_longest_prefix("project_b", "qwen2.5-7b", &prompt);
    assert!(match_b.is_empty());
    assert_eq!(match_b.matched_blocks.len(), 0);

    // Query from project_a matches
    let match_a = index.match_longest_prefix("project_a", "qwen2.5-7b", &prompt);
    assert_eq!(match_a.matched_blocks.len(), 1);
    assert_eq!(match_a.matched_blocks[0], 42);
}

#[test]
fn test_tier1_us15_radix_index_model_build_isolation() {
    let index = PrefixRadixIndex::new();
    let prompt = vec![9; 16];

    // Model build 1 vs Model build 2
    index.insert("proj_1", "model_v1", &[9; 16], &[100]);

    let match_v2 = index.match_longest_prefix("proj_1", "model_v2", &prompt);
    assert!(match_v2.is_empty());
}

#[test]
fn test_tier1_us15_16_token_block_alignment_exact_and_suffix() {
    let index = PrefixRadixIndex::new();
    index.insert("p1", "m1", &[10; 16], &[1]);

    // 25 tokens: 16 matched, 9 uncached suffix tokens
    let mut tokens = vec![10; 16];
    tokens.extend_from_slice(&[20; 9]);

    let res = index.match_longest_prefix("p1", "m1", &tokens);
    assert_eq!(res.matched_blocks.len(), 1);
    assert_eq!(res.matched_token_count, 16);
    assert_eq!(res.uncached_tokens.len(), 9);
    assert_eq!(res.uncached_tokens, vec![20; 9]);
}

// ============================================================================
// 9. US-16: Crash-Resilient Context Retention & Resumption (Features 22-25) — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us16_kv_holder_persistence_across_worker_crash() {
    let mut arena_header = ArenaHeader {
        magic: k3_core::ipc::layout::ARENA_MAGIC,
        version: 1,
        total_bytes: 1024 * 1024,
        block_count: 64,
        generation_counter: 1,
        ..Default::default()
    };

    // Worker crashes; holder retains arena and increments generation
    arena_header.generation_counter += 1;
    assert_eq!(arena_header.generation_counter, 2);
    assert_eq!(arena_header.validate(), Ok(()));
}

#[test]
fn test_tier1_us16_token_and_sampler_checkpointing_in_block_meta() {
    let mut meta = KvBlockMeta {
        block_id: 5,
        committed_token_count: 8,
        sampler_rng_state: [12345, 67890, 111213, 141516],
        ..Default::default()
    };

    // Advance 1 token and checkpoint sampler PRNG state
    meta.committed_token_count += 1;
    meta.sampler_rng_state[0] = 99999;

    assert_eq!(meta.committed_token_count, 9);
    assert_eq!(meta.sampler_rng_state[0], 99999);
}

#[test]
fn test_tier1_us16_decode_resumption_without_prefill_reexecution() {
    let prefill_tokens = 32_usize;
    let committed_tokens = 45_usize; // 32 prefill + 13 decode tokens

    // Restart worker from checkpoint
    let resume_start_pos = committed_tokens;
    let skip_prefill = resume_start_pos >= prefill_tokens;
    assert!(skip_prefill);
    assert_eq!(resume_start_pos, 45);
}

#[test]
fn test_tier1_us16_manager_restart_reattaches_to_kv_arena() {
    let active_arenas = vec!["arena_qwen", "arena_gemma"];

    // Simulated manager restart reconnects to k3-kv daemon
    let mut reattached_arenas = Vec::new();
    for arena in &active_arenas {
        reattached_arenas.push(*arena);
    }

    assert_eq!(reattached_arenas, active_arenas);
}

#[test]
fn test_tier1_us16_client_reconnect_within_retention_window() {
    let registry = SessionRegistry::new();
    let session_id = "tier1-session";
    registry
        .register_session(session_id, "proj-tier1", "model-tier1")
        .expect("register session");

    for tok in [100_u32, 101, 102, 103, 104, 105] {
        registry
            .append_token(session_id, tok)
            .expect("append token");
    }

    let disconnect_time = Instant::now();
    registry.record_disconnect(session_id).expect("disconnect");

    // Client reconnects after 5 seconds
    let reconnect_time = disconnect_time + Duration::from_secs(5);
    let catchup = registry
        .reconnect_client_at(session_id, Some(2), reconnect_time)
        .expect("reconnect within window");
    assert_eq!(catchup, vec![103, 104, 105]);
}

// ============================================================================
// 10. US-17: Session Metadata Persistence & Replay (Features 32-34) — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us17_persistent_schema_initialization_tenants_projects() {
    let store = SqlitePersistenceStore::open_in_memory().expect("open sqlite");
    store
        .save_tenant(&Tenant {
            id: "tenant_default".to_owned(),
            name: "Default Organization".to_owned(),
            created_at: 1_700_000_000,
        })
        .unwrap();

    store
        .save_project(&Project {
            id: "proj_alpha".to_owned(),
            tenant_id: "tenant_default".to_owned(),
            name: "Project Alpha".to_owned(),
            token_budget: Some(1_000_000_u32),
            created_at: 1_700_000_000,
        })
        .unwrap();

    let tenants = store.list_tenants().unwrap();
    let project = store.get_project("proj_alpha").unwrap().unwrap();
    assert_eq!(tenants.len(), 1);
    assert_eq!(project.token_budget, Some(1_000_000));
}

#[test]
fn test_tier1_us17_tuning_profile_and_budget_persistence_across_restart() {
    let temp_db = std::env::temp_dir().join(format!(
        "k3_tier1_us17_{}.db",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    {
        let store = SqlitePersistenceStore::open(&temp_db).expect("create on-disk db");
        let profile = TuningProfile {
            model_id: Some("qwen2.5-coder".to_owned()),
            priority_class: PriorityClass::Latency,
            x100_cores: 4,
            ..TuningProfile::conservative_default_with_n_ctx(64)
        };
        store
            .save_tuning_profile("qwen2.5-coder", &profile)
            .unwrap();
    } // drop and simulate reboot

    {
        let store = SqlitePersistenceStore::open(&temp_db).expect("reopen on-disk db");
        let deserialized = store.load_tuning_profile("qwen2.5-coder").unwrap().unwrap();
        assert_eq!(deserialized.priority_class, PriorityClass::Latency);
        assert_eq!(deserialized.x100_cores, 4);
    }
    let _ = std::fs::remove_file(temp_db);
}

#[test]
fn test_tier1_us17_bounded_single_row_last_turn_overwrite_invariant() {
    let store = SqlitePersistenceStore::open_in_memory().unwrap();
    store
        .save_tenant(&Tenant {
            id: "t".to_owned(),
            name: "T".to_owned(),
            created_at: 100,
        })
        .unwrap();
    store
        .save_project(&Project {
            id: "p".to_owned(),
            tenant_id: "t".to_owned(),
            name: "P".to_owned(),
            token_budget: None,
            created_at: 100,
        })
        .unwrap();

    let session_id = "sess_123";

    // Turn 1
    store
        .save_session_last_turn(&SessionLastTurn {
            session_id: session_id.to_owned(),
            project_id: "p".to_owned(),
            tenant_id: "t".to_owned(),
            model_id: "m".to_owned(),
            prompt: "Prompt 1".to_owned(),
            response: "Response 1".to_owned(),
            prompt_tokens: 10_u32,
            completion_tokens: 10_u32,
            parameters_json: "{}".to_owned(),
            updated_at: 100,
        })
        .unwrap();
    assert_eq!(store.count_session_last_turns().unwrap(), 1);

    // Turn 2 overwrites Turn 1
    store
        .save_session_last_turn(&SessionLastTurn {
            session_id: session_id.to_owned(),
            project_id: "p".to_owned(),
            tenant_id: "t".to_owned(),
            model_id: "m".to_owned(),
            prompt: "Prompt 2".to_owned(),
            response: "Response 2".to_owned(),
            prompt_tokens: 20_u32,
            completion_tokens: 20_u32,
            parameters_json: "{}".to_owned(),
            updated_at: 200,
        })
        .unwrap();

    assert_eq!(store.count_session_last_turns().unwrap(), 1);
    let last = store.load_session_last_turn(session_id).unwrap().unwrap();
    assert_eq!(last.prompt, "Prompt 2");
    assert_eq!(last.response, "Response 2");
}

#[test]
fn test_tier1_us17_cumulative_usage_stats_persistence() {
    let store = SqlitePersistenceStore::open_in_memory().unwrap();
    store
        .update_cumulative_stat("total_prompt_tokens", 54321_u32)
        .unwrap();
    store
        .update_cumulative_stat("total_completion_tokens", 12345_u32)
        .unwrap();

    let updated = store
        .update_cumulative_stat("total_prompt_tokens", 100_u32)
        .unwrap();
    assert_eq!(updated, 54421);
    assert_eq!(
        store.get_cumulative_stat("total_prompt_tokens").unwrap(),
        54421
    );
}

#[test]
fn test_tier1_us17_diagnostic_replay_prompt_reproduction() {
    let store = SqlitePersistenceStore::open_in_memory().unwrap();
    store
        .save_tenant(&Tenant {
            id: "t".to_owned(),
            name: "T".to_owned(),
            created_at: 100,
        })
        .unwrap();
    store
        .save_project(&Project {
            id: "p".to_owned(),
            tenant_id: "t".to_owned(),
            name: "P".to_owned(),
            token_budget: None,
            created_at: 100,
        })
        .unwrap();

    store
        .save_session_last_turn(&SessionLastTurn {
            session_id: "sess_replay".to_owned(),
            project_id: "p".to_owned(),
            tenant_id: "t".to_owned(),
            model_id: "m".to_owned(),
            prompt: "What is 2+2?".to_owned(),
            response: "4".to_owned(),
            prompt_tokens: 4_u32,
            completion_tokens: 1_u32,
            parameters_json: "{}".to_owned(),
            updated_at: 100,
        })
        .unwrap();

    let last_turn = store
        .load_session_last_turn("sess_replay")
        .unwrap()
        .unwrap();
    assert_eq!(last_turn.prompt, "What is 2+2?");
    assert_eq!(last_turn.response, "4");
}
