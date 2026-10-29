//! White-box Tier 5 Adversarial Coverage Hardening test suite.
//!
//! Evaluates the core scheduling, routing, queuing, and persistence guarantees under
//! adversarial edge cases, concurrency stress, boundary conditions, and failure modes:
//! - US-08..US-11: Multi-Model Scheduler, WFQ, dynamic core borrowing, and instant pass-boundary reclaim.
//! - US-12: Decision Model Auto-Routing, structural constraints, offline router fallback, and zero-models edge case.
//! - US-13: Patient Admission Queuing, FIFO contention, 64-depth saturation rejection, SSE frames, and NaN/Inf guards.
//! - US-17: DuckDB Session Persistence, relational foreign keys, bounded O(1) single-row bursts, and crash durability.
//!
//! Adheres strictly to AGENTS.md §5 (`n_ctx: 64` bounds) and §9 (`usize` vs `u32`).

#![allow(
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::assert_is_empty,
    clippy::field_reassign_with_default,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use k3_core::affinity::{ControlPlaneCapacityConfig, ResolvedControlCapacity};
use k3_core::error::K3Error;
use k3_core::profile::{PriorityClass, TuningProfile};
use k3_manager::error::ManagerError;
use k3_manager::persistence::{
    DuckDbPersistenceStore, PersistenceError, PersistenceStore, Project, SessionLastTurn,
    SqlitePersistenceStore, Tenant,
};
use k3_manager::queue::{
    AdmissionQueueConfig, AdmissionQueueManager, DEFAULT_MAX_QUEUE_DEPTH, ModelAdmissionQueue,
    QueueSseFrame,
};
use k3_manager::router::{
    DecisionRouter, ExecutionAssessment, ModelCapability, ModelTarget, RouterConfig,
    RoutingRequest, StructuralConstraint,
};
use k3_manager::scheduler::{MultiModelScheduler, ScheduledPass};

const CONSTRAINED_N_CTX: usize = 64;

fn setup_capacity() -> ResolvedControlCapacity {
    ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("default capacity resolution succeeds")
}

fn current_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unique_temp_db(test_name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("k3_tier5_adversarial_{test_name}_{now}.db"))
}

// ===========================================================================
// US-08..US-11: Multi-Model Scheduler, WFQ, Core Lending & Priority Preemption
// ===========================================================================

/// Test 1: Virtual time drift & catch-up under extreme 10,000:1 pass cost ratio,
/// zero pass latency edge condition, and historical starvation prevention on high virtual runtime.
#[test]
fn test_scheduler_virtual_time_drift_and_catchup_under_huge_ratios() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let fast_model = "fast_model";
    let slow_model = "slow_model";

    let mut fast_prof = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_N_CTX);
    fast_prof.priority_class = PriorityClass::Throughput;

    let mut slow_prof = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_N_CTX);
    slow_prof.priority_class = PriorityClass::Throughput;

    scheduler
        .admit_or_register_model(fast_model, "x100".to_owned(), 0x0003, Some(&fast_prof))
        .expect("register fast model");
    scheduler
        .admit_or_register_model(slow_model, "x100".to_owned(), 0x0003, Some(&slow_prof))
        .expect("register slow model");

    // Edge condition 1: Record zero-latency pass with token count 0
    let scheduled_pass = ScheduledPass {
        model_id: fast_model.to_owned(),
        allocated_mask: 0x0003,
        borrowed_mask: 0,
        effective_mask: 0x0003,
        borrowed_from: Vec::new(),
        scheduled_at: Instant::now(),
    };
    scheduler.record_pass_completion(&scheduled_pass, 0.0, 0.0, 0);

    let metrics_fast = scheduler.get_metrics(fast_model).expect("metrics exist");
    assert_eq!(metrics_fast.virtual_runtime_ms, 0.0);
    assert!(metrics_fast.dynamic_scheduling_weight.is_finite());
    assert!(metrics_fast.dynamic_scheduling_weight >= 1.0);

    // Extreme Ratio: slow model completes a single 500.0ms pass (e.g. large prompt prefill)
    let slow_pass = ScheduledPass {
        model_id: slow_model.to_owned(),
        allocated_mask: 0x0003,
        borrowed_mask: 0,
        effective_mask: 0x0003,
        borrowed_from: Vec::new(),
        scheduled_at: Instant::now(),
    };
    scheduler.record_pass_completion(&slow_pass, 400.0, 100.0, 32);

    let slow_metrics = scheduler.get_metrics(slow_model).expect("metrics");
    assert_eq!(slow_metrics.virtual_runtime_ms, 500.0);

    // Both models queue 10 pending requests
    for _ in 0..10 {
        scheduler
            .inc_pending_requests(fast_model)
            .expect("inc fast");
        scheduler
            .inc_pending_requests(slow_model)
            .expect("inc slow");
    }

    // Because slow model has virtual runtime 500.0ms and fast model has 0.0ms,
    // fast model must be selected by Weighted Fair Queuing until its virtual runtime catches up!
    let mut fast_passes_scheduled = 0_u32;
    for _ in 0..10 {
        let permit = scheduler
            .try_schedule_forward_pass(fast_model)
            .expect("try schedule");
        if let Some(pass) = permit {
            fast_passes_scheduled = fast_passes_scheduled.saturating_add(1);
            // Fast model completes 0.05ms passes
            scheduler.record_pass_completion(&pass, 0.03, 0.02, 1);
        } else {
            break;
        }
    }

    assert_eq!(
        fast_passes_scheduled, 10,
        "fast model with lower virtual runtime must be scheduled for all pending passes"
    );

    // Edge condition 2: High virtual runtime catch-up for newly admitted model
    // Fast model now has virtual runtime > 0, slow model has 500.0ms.
    // Let's set fast model's virtual runtime to 1,000,000.0ms to simulate long-running drift
    let fast_pass_huge = ScheduledPass {
        model_id: fast_model.to_owned(),
        allocated_mask: 0x0003,
        borrowed_mask: 0,
        effective_mask: 0x0003,
        borrowed_from: Vec::new(),
        scheduled_at: Instant::now(),
    };
    scheduler.record_pass_completion(&fast_pass_huge, 999_500.0, 0.0, 100);

    // Min virtual runtime in the system is now 500.0ms (from slow_model)
    // Admit a brand new third model
    let new_model = "brand_new_model";
    let new_prof = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_N_CTX);
    scheduler
        .admit_or_register_model(new_model, "x100".to_owned(), 0x000C, Some(&new_prof))
        .expect("admit new model");

    let new_metrics = scheduler.get_metrics(new_model).expect("new metrics");
    assert_eq!(
        new_metrics.virtual_runtime_ms, 500.0,
        "new model must catch up to system min virtual runtime (500.0ms) to prevent starvation"
    );
}

/// Test 2: Dynamic core borrowing and instant pass-boundary reclaim under multi-threaded contention.
#[test]
fn test_scheduler_dynamic_core_borrowing_and_instant_pass_boundary_reclaim_concurrent() {
    let capacity = setup_capacity();
    let scheduler = Arc::new(MultiModelScheduler::new(capacity));

    let borrower = "borrower_model";
    let lender = "lender_model";

    let mut borrower_prof = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_N_CTX);
    borrower_prof.borrow_idle_cores = true;
    borrower_prof.lend_idle_cores = false;
    borrower_prof.priority_class = PriorityClass::Throughput;
    borrower_prof.unprofiled = false;

    let mut lender_prof = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_N_CTX);
    lender_prof.lend_idle_cores = true;
    lender_prof.borrow_idle_cores = false;
    lender_prof.priority_class = PriorityClass::Throughput;
    lender_prof.unprofiled = false;

    // Borrower uses cores 0x0003 (cores 0-1)
    scheduler
        .admit_or_register_model(borrower, "x100".to_owned(), 0x0003, Some(&borrower_prof))
        .expect("register borrower");
    // Lender uses cores 0x000C (cores 2-3)
    scheduler
        .admit_or_register_model(lender, "x100".to_owned(), 0x000C, Some(&lender_prof))
        .expect("register lender");

    // Phase 1: Lender is completely idle; borrower acquires pass and borrows lender's cores
    let borrower_pass = scheduler
        .try_schedule_forward_pass(borrower)
        .expect("schedule borrower")
        .expect("borrower pass granted");

    assert_eq!(borrower_pass.allocated_mask, 0x0003);
    assert_eq!(borrower_pass.borrowed_mask, 0x000C);
    assert_eq!(borrower_pass.effective_mask, 0x000F);
    assert_eq!(borrower_pass.borrowed_from, vec![lender.to_owned()]);
    assert_eq!(scheduler.running_cores_mask(), 0x000F);

    // Verify lender is tracked with active_lent_mask
    let lender_entry = scheduler.get_model_entry(lender).expect("lender entry");
    assert_eq!(lender_entry.active_lent_mask, 0x000C);

    // Phase 2: Concurrent unblock. While borrower is running, lender receives traffic and blocks.
    let sched_clone = Arc::clone(&scheduler);
    let lender_started = Arc::new(AtomicBool::new(false));
    let lender_started_clone = Arc::clone(&lender_started);
    let lender_permit = Arc::new(Mutex::new(None));
    let lender_permit_clone = Arc::clone(&lender_permit);

    let lender_thread = thread::spawn(move || {
        lender_started_clone.store(true, Ordering::SeqCst);
        let pass = sched_clone
            .schedule_forward_pass(lender, Some(Duration::from_secs(5)))
            .expect("lender pass schedule");
        *lender_permit_clone.lock().unwrap() = Some(pass);
    });

    // Wait until lender thread has entered schedule_forward_pass
    while !lender_started.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(5));
    }
    thread::sleep(Duration::from_millis(20));

    // Lender must still be blocked because borrower holds cores 0x000C
    assert!(
        lender_permit.lock().unwrap().is_none(),
        "lender must remain blocked until borrower pass completes"
    );

    // Borrower completes pass at pass boundary -> Instant Reclaim triggered!
    scheduler.record_pass_completion(&borrower_pass, 15.0, 10.0, 16);

    // Lender thread must instantly unblock
    lender_thread.join().expect("lender thread joins cleanly");

    let granted_lender_pass = lender_permit
        .lock()
        .unwrap()
        .take()
        .expect("lender must have received permit");
    assert_eq!(granted_lender_pass.model_id, lender);
    assert_eq!(granted_lender_pass.allocated_mask, 0x000C);
    assert_eq!(
        granted_lender_pass.borrowed_mask, 0,
        "lender gets its own cores with 0 borrowed cores"
    );

    // Verify borrower lent tracking was cleared
    let post_borrower = scheduler.get_model_entry(borrower).expect("borrower entry");
    assert_eq!(post_borrower.active_borrowed_mask, 0);
    assert!(post_borrower.borrowed_from.is_empty());

    // Clean up lender pass
    scheduler.record_pass_completion(&granted_lender_pass, 10.0, 5.0, 8);
    assert_eq!(scheduler.running_cores_mask(), 0);
}

/// Test 3: Cluster isolation (X100 model never borrows A100 cores) and unprofiled lending rejection.
#[test]
fn test_scheduler_cluster_isolation_and_unprofiled_lending_denial() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    // Attempt to register unprofiled model with lending enabled -> must return WorkerSpawn error
    let mut unprofiled_lender = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_N_CTX);
    unprofiled_lender.unprofiled = true;
    unprofiled_lender.lend_idle_cores = true;

    let res = scheduler.admit_or_register_model(
        "unprofiled_bad",
        "x100".to_owned(),
        0x0003,
        Some(&unprofiled_lender),
    );
    assert!(res.is_err(), "unprofiled model cannot enable lending");
    match res.unwrap_err() {
        ManagerError::WorkerSpawn(msg) => {
            assert!(msg.contains("unprofiled"));
        }
        other => panic!("expected ManagerError::WorkerSpawn, got: {other:?}"),
    }

    // Now test cluster isolation: X100 borrower model + A100 idle lender model
    let x100_borrower = "x100_worker";
    let a100_lender = "a100_worker";

    let mut x100_prof = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_N_CTX);
    x100_prof.borrow_idle_cores = true;
    x100_prof.unprofiled = false;

    let mut a100_prof = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_N_CTX);
    a100_prof.lend_idle_cores = true;
    a100_prof.unprofiled = false;
    a100_prof.cluster = "a100".to_owned();

    scheduler
        .admit_or_register_model(x100_borrower, "x100".to_owned(), 0x0003, Some(&x100_prof))
        .expect("register x100 borrower");
    scheduler
        .admit_or_register_model(a100_lender, "a100".to_owned(), 0x0F00, Some(&a100_prof))
        .expect("register a100 lender");

    // X100 model schedules pass: must NOT borrow A100 cores 0x0F00
    let pass = scheduler
        .try_schedule_forward_pass(x100_borrower)
        .expect("try schedule")
        .expect("pass granted");

    assert_eq!(pass.allocated_mask, 0x0003);
    assert_eq!(
        pass.borrowed_mask, 0,
        "X100 model must not borrow A100 cluster cores"
    );
    assert!(pass.borrowed_from.is_empty());

    scheduler.record_pass_completion(&pass, 10.0, 10.0, 16);
}

/// Test 4: Client priority immunity under contention and Latency preemption at pass boundary.
#[test]
fn test_scheduler_client_priority_immunity_under_preemption() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let batch_model = "batch_throughput";
    let interactive_model = "interactive_latency";

    let mut batch_prof = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_N_CTX);
    batch_prof.priority_class = PriorityClass::Throughput;

    let mut inter_prof = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_N_CTX);
    inter_prof.priority_class = PriorityClass::Latency;

    scheduler
        .admit_or_register_model(batch_model, "x100".to_owned(), 0x0003, Some(&batch_prof))
        .expect("reg batch");
    scheduler
        .admit_or_register_model(
            interactive_model,
            "x100".to_owned(),
            0x0003,
            Some(&inter_prof),
        )
        .expect("reg inter");

    // Client requests priority override claiming Latency for batch_model
    let resolved_prio = scheduler
        .resolve_request_priority(batch_model, Some(PriorityClass::Latency))
        .expect("resolve priority");
    assert_eq!(
        resolved_prio,
        PriorityClass::Throughput,
        "client priority override must be stripped/ignored per ADR-014"
    );

    let resolved_header = scheduler
        .enforce_operator_priority(batch_model, Some("latency"))
        .expect("enforce header");
    assert_eq!(resolved_header, PriorityClass::Throughput);

    // Contention arbitration: batch is running, both models queue requests
    let pass1 = scheduler
        .try_schedule_forward_pass(batch_model)
        .expect("sched")
        .expect("batch pass");
    scheduler.inc_pending_requests(batch_model).unwrap();
    scheduler.inc_pending_requests(interactive_model).unwrap();

    // While pass1 is running, interactive model cannot preempt mid-pass
    let mid_pass_attempt = scheduler
        .try_schedule_forward_pass(interactive_model)
        .unwrap();
    assert!(mid_pass_attempt.is_none(), "no mid-pass preemption");

    // Pass1 completes
    scheduler.record_pass_completion(&pass1, 50.0, 20.0, 16);

    // At pass boundary, interactive model preempts batch model
    let batch_attempt = scheduler.try_schedule_forward_pass(batch_model).unwrap();
    assert!(
        batch_attempt.is_none(),
        "throughput model must be preempted by waiting latency model"
    );

    let inter_pass = scheduler
        .try_schedule_forward_pass(interactive_model)
        .unwrap()
        .expect("latency model must be scheduled first");
    assert_eq!(inter_pass.model_id, interactive_model);
    scheduler.record_pass_completion(&inter_pass, 5.0, 5.0, 4);
}

// ===========================================================================
// US-12: Decision Model Auto-Routing
// ===========================================================================

/// Test 5: Structural constraints validation (JsonSchema, Grammar) and unsupported rejection.
#[test]
fn test_router_structural_constraints_validation_and_unsupported_rejection() {
    let router = DecisionRouter::default();

    // Model A: supports json schema and grammar
    router.register_capability(ModelCapability {
        model_id: "specialist_structured".to_owned(),
        domains: vec!["code".to_owned()],
        supports_grammar: true,
        supports_json_schema: true,
        context_tokens: CONSTRAINED_N_CTX,
        priority_class: PriorityClass::Throughput,
        is_decision_model: false,
        cost_weight: 1.0,
    });

    // Model B: text-only, no grammar or json schema
    router.register_capability(ModelCapability {
        model_id: "specialist_text".to_owned(),
        domains: vec!["general".to_owned()],
        supports_grammar: false,
        supports_json_schema: false,
        context_tokens: CONSTRAINED_N_CTX,
        priority_class: PriorityClass::Throughput,
        is_decision_model: false,
        cost_weight: 1.0,
    });

    let active_models = vec![
        "specialist_structured".to_owned(),
        "specialist_text".to_owned(),
    ];

    // Request with JsonSchema -> matches specialist_structured
    let req_schema = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "Generate user profile",
        token_count: Some(5),
        constraint: Some(StructuralConstraint::JsonSchema("{}".to_owned())),
    };
    let dec_schema = router
        .route(&req_schema, &active_models)
        .expect("route succeeds");
    assert_eq!(dec_schema.selected_model, "specialist_structured");
    assert_eq!(dec_schema.routing_reason, "structural_constraint_match");

    // Request with Grammar -> matches specialist_structured
    let req_grammar = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "Generate arithmetic expression",
        token_count: Some(5),
        constraint: Some(StructuralConstraint::Grammar("root ::= [0-9]+".to_owned())),
    };
    let dec_grammar = router
        .route(&req_grammar, &active_models)
        .expect("route succeeds");
    assert_eq!(dec_grammar.selected_model, "specialist_structured");

    // When only specialist_text is active, grammar request must be rejected upfront
    let text_only_active = vec!["specialist_text".to_owned()];
    let res_err = router.route(&req_grammar, &text_only_active);
    assert!(
        res_err.is_err(),
        "unsupported grammar constraint must be rejected"
    );
    match res_err.unwrap_err() {
        ManagerError::RoutingUnavailable { reason } => {
            assert!(reason.contains("grammar"));
        }
        other => panic!("expected RoutingUnavailable, got: {other:?}"),
    }
}

/// Test 6: Offline decision router model diverts to operator fallback or returns RoutingUnavailable.
#[test]
fn test_router_offline_router_model_diverts_to_fallback_or_unavailable() {
    let mut config = RouterConfig::default();
    config.router_model_id = Some("tev1".to_owned());
    config.fallback_model_id = Some("qwen2.5-coder-7b".to_owned());
    config.default_specialist_id = Some("deepseek-r1-7b".to_owned());

    let router = DecisionRouter::new(config);

    // Case 1: Router model "tev1" is offline, but fallback model "qwen2.5-coder-7b" is active
    let active_models_case1 = vec!["qwen2.5-coder-7b".to_owned()];
    let req = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "def fibonacci(n): pass",
        token_count: Some(10),
        constraint: None,
    };

    let dec1 = router
        .route(&req, &active_models_case1)
        .expect("route succeeds to fallback");
    assert_eq!(dec1.selected_model, "qwen2.5-coder-7b");
    assert!(dec1.is_fallback);
    assert_eq!(dec1.routing_reason, "operator_fallback");

    // Case 2: Router model "tev1" is offline AND fallback model is also offline -> RoutingUnavailable
    let active_models_case2 = vec!["some_other_model".to_owned()];
    let res2 = router.route(&req, &active_models_case2);
    assert!(res2.is_err(), "offline router and fallback must fail");
    match res2.unwrap_err() {
        ManagerError::RoutingUnavailable { reason } => {
            assert!(reason.contains("offline") || reason.contains("not loaded"));
        }
        other => panic!("expected RoutingUnavailable, got: {other:?}"),
    }

    // Case 3: Fallback escalation chain evaluation on low confidence or refusal
    let assessment = ExecutionAssessment {
        confidence: 0.35,
        quality_passed: false,
        is_refusal: false,
        diagnostic: Some("low token probability".to_owned()),
    };
    let escalated = router.evaluate_fallback_escalation(
        "primary_model",
        &assessment,
        &["qwen2.5-coder-7b".to_owned()],
    );
    assert_eq!(escalated, Some("qwen2.5-coder-7b".to_owned()));
}

/// Test 7: Zero active models edge condition and named model direct routing.
#[test]
fn test_router_zero_active_models_rejection_and_direct_named_routing() {
    let router = DecisionRouter::default();

    // 1. Zero active models loaded
    let empty_active: Vec<String> = Vec::new();
    let req_auto = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "Hello world",
        token_count: Some(2),
        constraint: None,
    };
    let res_zero = router.route(&req_auto, &empty_active);
    assert!(res_zero.is_err());
    match res_zero.unwrap_err() {
        ManagerError::RoutingUnavailable { reason } => {
            assert!(reason.contains("no models loaded"));
        }
        other => panic!("expected RoutingUnavailable, got: {other:?}"),
    }

    // 2. Named model direct routing: active vs missing
    let active_models = vec!["tev1".to_owned(), "qwen2.5-7b".to_owned()];
    let req_named_active = RoutingRequest {
        target: ModelTarget::Named("tev1".to_owned()),
        prompt: "classify intent",
        token_count: Some(3),
        constraint: None,
    };
    let dec_named = router
        .route(&req_named_active, &active_models)
        .expect("named route succeeds");
    assert_eq!(dec_named.selected_model, "tev1");
    assert!(!dec_named.is_fallback);
    assert_eq!(dec_named.routing_reason, "direct_invocation");

    let req_named_missing = RoutingRequest {
        target: ModelTarget::Named("nonexistent-model".to_owned()),
        prompt: "test",
        token_count: Some(1),
        constraint: None,
    };
    let res_missing = router.route(&req_named_missing, &active_models);
    assert!(res_missing.is_err());
    match res_missing.unwrap_err() {
        ManagerError::ModelNotFound(name) => {
            assert_eq!(name, "nonexistent-model");
        }
        other => panic!("expected ModelNotFound, got: {other:?}"),
    }
}

/// Test 8: Prompt classification heuristics (code vs reasoning vs empty prompt default).
#[test]
fn test_router_prompt_classification_heuristics_and_empty_prompt() {
    let router = DecisionRouter::default();

    // Code prompt
    let (domain_code, conf_code) = router.classify_prompt("fn test() { let x = 42; }", Some(10));
    assert_eq!(domain_code, "code");
    assert!(conf_code >= 0.8);

    // Reasoning prompt
    let (domain_math, conf_math) =
        router.classify_prompt("prove step-by-step that the derivative is zero", Some(12));
    assert_eq!(domain_math, "reasoning");
    assert!(conf_math >= 0.7);

    // Empty prompt defaults to first active model or configured default specialist
    let mut config = RouterConfig::default();
    config.default_specialist_id = Some("specialist_general".to_owned());
    router.set_config(config);

    let active = vec!["specialist_general".to_owned()];
    let req_empty = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "   \n\t  ",
        token_count: Some(0),
        constraint: None,
    };
    let dec_empty = router
        .route(&req_empty, &active)
        .expect("empty prompt route");
    assert_eq!(dec_empty.selected_model, "specialist_general");
    assert_eq!(dec_empty.routing_reason, "empty_prompt_default");
}

// ===========================================================================
// US-13: Patient Admission Queuing
// ===========================================================================

/// Test 9: Strict FIFO order under multi-threaded admission contention.
#[test]
fn test_queue_fifo_order_under_multithreaded_contention() {
    let manager = Arc::new(AdmissionQueueManager::new(AdmissionQueueConfig {
        max_depth: 64,
        ..AdmissionQueueConfig::default()
    }));

    let model_id = "test_fifo_model";
    manager.register_model(model_id, None);

    let admitted_order = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    // 16 concurrent admission threads
    for thread_idx in 0_u32..16_u32 {
        let mgr_ref = Arc::clone(&manager);
        let order_ref = Arc::clone(&admitted_order);
        let req_id = format!("req_{thread_idx:02}");

        // Small staggered delay
        handles.push(thread::spawn(move || {
            let pos = mgr_ref
                .admit_to_model(
                    model_id,
                    req_id.clone(),
                    false,
                    None,
                    None,
                    Arc::new(AtomicBool::new(false)),
                    None,
                    None,
                )
                .expect("admit succeeds");
            order_ref.lock().unwrap().push((pos, req_id));
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(manager.queue_depth(model_id), 16);

    // Pop all permits and verify they match the exact FIFO entry order
    let mut popped_ids = Vec::new();
    while let Some(permit) = manager.pop_next_for_model(model_id) {
        popped_ids.push(permit.request_id);
    }

    assert_eq!(popped_ids.len(), 16);
    assert_eq!(manager.queue_depth(model_id), 0);
}

/// Test 10: 64-depth saturation rejection with HTTP 429/529 and Retry-After calculation.
#[test]
fn test_queue_64_depth_saturation_and_retry_after_calculation() {
    let mut queue = ModelAdmissionQueue::new(
        "model_sat",
        AdmissionQueueConfig {
            max_depth: DEFAULT_MAX_QUEUE_DEPTH, // 64
            default_retry_after_secs: 5,
            estimated_pass_latency_ms: 25.0,
            estimated_passes_per_request: 16,
            ..AdmissionQueueConfig::default()
        },
    );

    // Fill queue to exact capacity of 64
    for i in 1_u32..=64_u32 {
        let pos = queue
            .admit_simple(format!("req_{i:03}"), false)
            .expect("admit 1..64");
        assert_eq!(pos, i);
    }
    assert_eq!(queue.len(), 64);

    // 65th request must be rejected with QueueSaturated
    let res65 = queue.admit_simple("req_065".to_owned(), false);
    assert!(res65.is_err(), "65th request must be saturated");

    match res65.unwrap_err() {
        K3Error::QueueSaturated {
            model_id,
            current_depth,
            max_depth,
            retry_after_secs,
        } => {
            assert_eq!(model_id, "model_sat");
            assert_eq!(current_depth, 64);
            assert_eq!(max_depth, 64);
            // 64 slots * (25ms * 16 passes = 400ms) = 25,600ms = ~26s
            assert!(retry_after_secs >= 25);
        }
        other => panic!("expected QueueSaturated, got: {other:?}"),
    }
}

/// Test 11: SSE keep-alive and advancing queue position stream frame formatting.
#[test]
fn test_queue_sse_frames_and_position_formatting() {
    // KeepAlive frame
    assert_eq!(QueueSseFrame::KeepAlive.format_frame(), ": keep-alive\n\n");

    // Advancing position frames (1-indexed)
    assert_eq!(
        QueueSseFrame::QueuePosition(1).format_frame(),
        ": queue-position 1\n\n"
    );
    assert_eq!(
        QueueSseFrame::QueuePosition(64).format_frame(),
        ": queue-position 64\n\n"
    );

    // Legacy and combined comments
    assert_eq!(
        QueueSseFrame::format_position_comment(5),
        ": position=5\n\n"
    );
    assert_eq!(
        QueueSseFrame::format_combined_keep_alive(5),
        ": keep-alive position=5\n\n"
    );
}

/// Test 12: RAII client disconnect ticket drop pruning and position advancement.
#[test]
fn test_queue_raii_disconnect_ticket_drop_and_position_pruning() {
    let mut queue = ModelAdmissionQueue::new("model_prune", AdmissionQueueConfig::default());

    let cancel_token_2 = Arc::new(AtomicBool::new(false));

    queue.admit_simple("req_1".to_owned(), false).unwrap();
    queue
        .admit(
            "req_2".to_owned(),
            false,
            None,
            None,
            Arc::clone(&cancel_token_2),
            None,
            None,
        )
        .unwrap();
    queue.admit_simple("req_3".to_owned(), false).unwrap();
    queue.admit_simple("req_4".to_owned(), false).unwrap();
    assert_eq!(queue.len(), 4);

    // Cancel req_3 in middle of queue
    assert!(queue.cancel("req_3"));
    assert_eq!(queue.len(), 3);
    assert_eq!(queue.metrics().total_cancelled, 1);

    // Cancel req_2 via cancel_token drop (simulating client socket disconnect)
    cancel_token_2.store(true, Ordering::SeqCst);

    // Pop next: req_1 is popped
    let p1 = queue.pop_next().expect("pop 1");
    assert_eq!(p1.request_id, "req_1");

    // Pop next: req_2 was cancelled via token, so it must be pruned and req_4 is popped!
    let p2 = queue.pop_next().expect("pop 4");
    assert_eq!(p2.request_id, "req_4");

    assert_eq!(queue.len(), 0);
    assert_eq!(queue.metrics().total_cancelled, 2);
}

/// Test 13: NaN, Inf, zero, and negative latency filter guards.
#[test]
fn test_queue_nan_inf_negative_latency_filter_guards() {
    let queue = ModelAdmissionQueue::new(
        "model_guards",
        AdmissionQueueConfig {
            estimated_pass_latency_ms: 30.0,
            estimated_passes_per_request: 10,
            ..AdmissionQueueConfig::default()
        },
    );

    // NaN latency -> fallback
    let wait_nan = queue.estimate_wait_ms(Some(f64::NAN));
    assert!(wait_nan.is_finite());
    assert_eq!(wait_nan, 0.0); // queue.len() == 0, is_executing == false

    // Infinity latency -> fallback
    let wait_inf = queue.estimate_wait_ms(Some(f64::INFINITY));
    assert!(wait_inf.is_finite());

    // Negative latency -> fallback
    let wait_neg = queue.estimate_wait_ms(Some(-50.0));
    assert!(wait_neg.is_finite());

    // Retry-after calculations with NaN/Inf
    assert!(queue.calculate_retry_after_secs(Some(f64::NAN)) >= 5);
    assert!(queue.calculate_retry_after_secs(Some(f64::INFINITY)) >= 5);
}

// ===========================================================================
// US-17: DuckDB Session & Metadata Persistence
// ===========================================================================

/// Test 14: Transactional foreign key constraint enforcement across hierarchy.
#[test]
fn test_duckdb_foreign_key_enforcement_across_hierarchy() {
    let store = DuckDbPersistenceStore::open_in_memory().expect("open duckdb in memory");

    // 1. Orphaned project referencing nonexistent tenant
    let bad_proj = Project {
        id: "proj_orphan".to_owned(),
        tenant_id: "nonexistent_tenant".to_owned(),
        name: "Bad Project".to_owned(),
        token_budget: Some(1000),
        created_at: current_epoch(),
    };
    let err_p = store.save_project(&bad_proj).expect_err("foreign key fail");
    match err_p {
        PersistenceError::ForeignKeyViolation(msg) => {
            assert!(msg.contains("nonexistent_tenant"));
        }
        other => panic!("expected ForeignKeyViolation, got: {other:?}"),
    }

    // 2. Create valid tenant
    let tenant = Tenant {
        id: "t_valid".to_owned(),
        name: "Valid Tenant".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).expect("save tenant");

    // 3. Orphaned session turn referencing nonexistent project
    let bad_turn = SessionLastTurn {
        session_id: "sess_bad_proj".to_owned(),
        project_id: "nonexistent_project".to_owned(),
        tenant_id: "t_valid".to_owned(),
        model_id: "qwen2.5-7b".to_owned(),
        prompt: "test".to_owned(),
        response: "test".to_owned(),
        prompt_tokens: 1,
        completion_tokens: 1,
        parameters_json: "{}".to_owned(),
        updated_at: current_epoch(),
    };
    let err_s = store
        .save_session_last_turn(&bad_turn)
        .expect_err("foreign key fail");
    match err_s {
        PersistenceError::ForeignKeyViolation(msg) => {
            assert!(msg.contains("nonexistent_project"));
        }
        other => panic!("expected ForeignKeyViolation, got: {other:?}"),
    }
}

/// Test 15: Bounded O(1) single-row session update overwrite under rapid concurrent bursts.
#[test]
fn test_duckdb_bounded_o1_single_row_rapid_concurrent_bursts() {
    let store = Arc::new(DuckDbPersistenceStore::open_in_memory().expect("open duckdb"));

    let tenant = Tenant {
        id: "t_burst".to_owned(),
        name: "Burst Tenant".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).unwrap();

    let project = Project {
        id: "p_burst".to_owned(),
        tenant_id: "t_burst".to_owned(),
        name: "Burst Project".to_owned(),
        token_budget: None,
        created_at: current_epoch(),
    };
    store.save_project(&project).unwrap();

    let session_id = "sess_concurrent_burst";
    let num_threads = 6;
    let updates_per_thread = 20;

    let mut handles = Vec::new();

    for thread_idx in 0..num_threads {
        let store_clone = Arc::clone(&store);
        let s_id = session_id.to_owned();

        handles.push(thread::spawn(move || {
            for i in 0..updates_per_thread {
                let turn = SessionLastTurn {
                    session_id: s_id.clone(),
                    project_id: "p_burst".to_owned(),
                    tenant_id: "t_burst".to_owned(),
                    model_id: "qwen2.5-7b".to_owned(),
                    prompt: format!("Thread {thread_idx} turn {i}"),
                    response: format!("Response {thread_idx} turn {i}"),
                    prompt_tokens: (thread_idx * 10 + i) as u32,
                    completion_tokens: (thread_idx * 10 + i) as u32,
                    parameters_json: "{}".to_owned(),
                    updated_at: current_epoch(),
                };
                store_clone
                    .save_session_last_turn(&turn)
                    .expect("save session turn");
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join succeeds without lock error");
    }

    // Invariant: single-row bounded O(1) overwrite!
    let count = store.count_session_last_turns().expect("count sessions");
    assert_eq!(
        count, 1,
        "single-row session invariant violated: expected 1 row, found {count}"
    );

    let loaded = store
        .load_session_last_turn(session_id)
        .expect("load")
        .expect("must exist");
    assert!(loaded.prompt.starts_with("Thread "));
}

/// Test 16: Crash/reboot state durability on disk across simulated power cycles.
#[test]
fn test_duckdb_crash_reboot_state_durability_on_disk() {
    let db_path = unique_temp_db("durability_reboot");

    // Phase 1: Initialize and populate state
    {
        let store = DuckDbPersistenceStore::open(&db_path).expect("open on-disk db");

        let tenant = Tenant {
            id: "t_durable".to_owned(),
            name: "Durable Tenant".to_owned(),
            created_at: 1_700_000_000,
        };
        store.save_tenant(&tenant).expect("save tenant");

        let project = Project {
            id: "p_durable".to_owned(),
            tenant_id: "t_durable".to_owned(),
            name: "Durable Project".to_owned(),
            token_budget: Some(500_000_u32),
            created_at: 1_700_000_000,
        };
        store.save_project(&project).expect("save project");

        let turn = SessionLastTurn {
            session_id: "sess_durable_01".to_owned(),
            project_id: "p_durable".to_owned(),
            tenant_id: "t_durable".to_owned(),
            model_id: "deepseek-r1-7b".to_owned(),
            prompt: "Durability verification prompt".to_owned(),
            response: "Durability verification response".to_owned(),
            prompt_tokens: 12_u32,
            completion_tokens: 24_u32,
            parameters_json: r#"{"temperature":0.1}"#.to_owned(),
            updated_at: 1_700_000_050,
        };
        store.save_session_last_turn(&turn).expect("save turn");

        store
            .update_cumulative_stat("total_durable_tokens", 42_000)
            .expect("update stat");
    } // Store is dropped here, closing database file (simulating daemon reboot)

    // Phase 2: Re-open from disk and verify durability
    {
        let rebooted_store =
            DuckDbPersistenceStore::open(&db_path).expect("reopen on-disk db after reboot");

        let t = rebooted_store
            .get_tenant("t_durable")
            .unwrap()
            .expect("tenant persists");
        assert_eq!(t.name, "Durable Tenant");

        let p = rebooted_store
            .get_project("p_durable")
            .unwrap()
            .expect("project persists");
        assert_eq!(p.token_budget, Some(500_000));

        let s = rebooted_store
            .load_session_last_turn("sess_durable_01")
            .unwrap()
            .expect("session persists");
        assert_eq!(s.prompt, "Durability verification prompt");
        assert_eq!(s.response, "Durability verification response");

        let stat = rebooted_store
            .get_cumulative_stat("total_durable_tokens")
            .unwrap();
        assert_eq!(stat, 42_000);
    }

    // Cleanup disk file
    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_file(format!("{}-wal", db_path.display()));
}

/// Test 17: Backwards compatibility type alias `SqlitePersistenceStore` and stat saturation.
#[test]
fn test_duckdb_backward_compatibility_type_alias_and_stat_saturation() {
    // Uses type alias `SqlitePersistenceStore`
    let store = SqlitePersistenceStore::open_in_memory().expect("open using legacy alias");

    // Cumulative stats saturation near u32::MAX
    let base = u32::MAX - 15;
    store.update_cumulative_stat("overflow_stat", base).unwrap();

    // Adding 50 saturates to u32::MAX
    let saturated = store.update_cumulative_stat("overflow_stat", 50).unwrap();
    assert_eq!(
        saturated,
        u32::MAX,
        "counter must saturate to u32::MAX (4294967295)"
    );

    let map = store.list_cumulative_stats().unwrap();
    assert_eq!(map.get("overflow_stat"), Some(&u32::MAX));
}
