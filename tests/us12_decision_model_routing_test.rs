//! US-12 Dedicated Scenario Test Suite: Decision-Model Routing (Features 26, 27, 28, ADR-013, ADR-014).
//!
//! Verifies:
//! - Scenario 1: Direct named invocation of decision models (`tev1`, `nimble`).
//! - Scenario 2: Content-aware critical path routing on `model: auto` (code, reasoning, structural constraints).
//! - Scenario 3: Fallback models and structured `RoutingUnavailable` diagnostics when routers or models are unavailable.
//! - Confidence-based fallback escalation chains across specialist tiers.

use k3_core::profile::PriorityClass;
use k3_manager::error::ManagerError;
use k3_manager::router::{
    DecisionRouter, ExecutionAssessment, ModelCapability, ModelTarget, RouterConfig,
    RoutingRequest, StructuralConstraint,
};

fn setup_test_router() -> (DecisionRouter, Vec<String>) {
    let config = RouterConfig {
        router_model_id: Some("tev1".to_owned()),
        fallback_model_id: Some("gemma-2-9b-it".to_owned()),
        fallback_escalation_chain: vec!["qwen2.5-coder-32b".to_owned(), "gemma-2-9b-it".to_owned()],
        default_specialist_id: Some("gemma-2-9b-it".to_owned()),
        confidence_threshold: 0.60,
    };
    let router = DecisionRouter::new(config);

    // Register tev1 decision model
    router.register_capability(ModelCapability {
        model_id: "tev1".to_owned(),
        domains: vec!["classification".to_owned(), "utility".to_owned()],
        supports_grammar: false,
        supports_json_schema: false,
        context_tokens: 2048,
        priority_class: PriorityClass::Latency,
        is_decision_model: true,
        cost_weight: 0.1,
    });

    // Register qwen2.5-coder-7b code specialist
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

    // Register gemma-2-9b-it general/reasoning specialist
    router.register_capability(ModelCapability {
        model_id: "gemma-2-9b-it".to_owned(),
        domains: vec![
            "reasoning".to_owned(),
            "general".to_owned(),
            "chat".to_owned(),
        ],
        supports_grammar: false,
        supports_json_schema: false,
        context_tokens: 8192,
        priority_class: PriorityClass::Throughput,
        is_decision_model: false,
        cost_weight: 1.2,
    });

    // Register qwen2.5-coder-32b heavy escalation specialist
    router.register_capability(ModelCapability {
        model_id: "qwen2.5-coder-32b".to_owned(),
        domains: vec!["code".to_owned()],
        supports_grammar: true,
        supports_json_schema: true,
        context_tokens: 16384,
        priority_class: PriorityClass::Throughput,
        is_decision_model: false,
        cost_weight: 3.5,
    });

    let active_models = vec![
        "tev1".to_owned(),
        "qwen2.5-coder-7b".to_owned(),
        "gemma-2-9b-it".to_owned(),
        "qwen2.5-coder-32b".to_owned(),
    ];

    (router, active_models)
}

#[test]
fn test_scenario1_direct_named_decision_model_invocation() {
    let (router, active_models) = setup_test_router();

    let request = RoutingRequest {
        target: ModelTarget::parse("tev1"),
        prompt: "Classify this query: sentiment analysis",
        token_count: Some(6_u32),
        constraint: None,
    };

    let decision = router
        .route(&request, &active_models)
        .expect("direct invocation must succeed");

    assert_eq!(decision.selected_model, "tev1");
    assert!(!decision.is_fallback);
    assert_eq!(decision.routing_reason, "direct_invocation");
}

#[test]
fn test_scenario1_direct_named_model_not_found_error() {
    let (router, active_models) = setup_test_router();

    let request = RoutingRequest {
        target: ModelTarget::parse("nonexistent-model-xyz"),
        prompt: "Hello world",
        token_count: Some(2_u32),
        constraint: None,
    };

    let result = router.route(&request, &active_models);
    assert!(result.is_err());
    match result.unwrap_err() {
        ManagerError::ModelNotFound(id) => assert_eq!(id, "nonexistent-model-xyz"),
        other => panic!("expected ModelNotFound, got {other:?}"),
    }
}

#[test]
fn test_scenario2_auto_routing_code_domain_dispatch() {
    let (router, active_models) = setup_test_router();

    let request = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "fn calculate_hash<T: Hash>(t: &T) -> u64 { let mut s = DefaultHasher::new(); t.hash(&mut s); s.finish() }",
        token_count: Some(30_u32),
        constraint: None,
    };

    let decision = router
        .route(&request, &active_models)
        .expect("auto routing for code must succeed");

    assert_eq!(decision.selected_model, "qwen2.5-coder-7b");
    assert!(!decision.is_fallback);
    assert_eq!(decision.routing_reason, "domain_match:code");
}

#[test]
fn test_scenario2_auto_routing_reasoning_domain_dispatch() {
    let (router, active_models) = setup_test_router();

    let request = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "Please prove step-by-step therefore that the square root of 2 is irrational.",
        token_count: Some(15_u32),
        constraint: None,
    };

    let decision = router
        .route(&request, &active_models)
        .expect("auto routing for reasoning must succeed");

    assert_eq!(decision.selected_model, "gemma-2-9b-it");
    assert!(!decision.is_fallback);
    assert_eq!(decision.routing_reason, "domain_match:reasoning");
}

#[test]
fn test_scenario2_auto_routing_structural_constraint_json_schema() {
    let (router, active_models) = setup_test_router();

    let request = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "Generate user account details in JSON format.",
        token_count: Some(8_u32),
        constraint: Some(StructuralConstraint::JsonSchema(
            r#"{"type":"object","properties":{"name":{"type":"string"}}}"#.to_owned(),
        )),
    };

    let decision = router
        .route(&request, &active_models)
        .expect("structural constraint routing must succeed");

    assert_eq!(decision.selected_model, "qwen2.5-coder-7b");
    assert!(!decision.is_fallback);
    assert_eq!(decision.routing_reason, "structural_constraint_match");
}

#[test]
fn test_scenario2_auto_routing_structural_constraint_unsupported_rejection() {
    let (router, _) = setup_test_router();
    // Only gemma-2-9b-it is active, which does not support grammar or json schema
    let active_models = vec!["gemma-2-9b-it".to_owned()];

    let request = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "Generate an AST matching this grammar.",
        token_count: Some(10_u32),
        constraint: Some(StructuralConstraint::Grammar(
            "expr ::= term '+' factor".to_owned(),
        )),
    };

    let result = router.route(&request, &active_models);
    assert!(result.is_err());
    match result.unwrap_err() {
        ManagerError::RoutingUnavailable { reason } => {
            assert!(reason.contains("no loaded model with verified grammar engine support"));
        }
        other => panic!("expected RoutingUnavailable, got {other:?}"),
    }
}

#[test]
fn test_scenario3_fallback_model_when_router_offline() {
    let (router, _) = setup_test_router();
    // Neither tev1 (router) nor qwen2.5-coder (code specialist) is active;
    // only gemma-2-9b-it (fallback model) is active
    let active_models = vec!["gemma-2-9b-it".to_owned()];

    let request = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "fn calculate() -> u32 { 42 }",
        token_count: Some(8_u32),
        constraint: None,
    };

    let decision = router
        .route(&request, &active_models)
        .expect("fallback routing must succeed");

    assert_eq!(decision.selected_model, "gemma-2-9b-it");
    assert!(decision.is_fallback);
    assert_eq!(decision.routing_reason, "operator_fallback");
}

#[test]
fn test_scenario3_routing_unavailable_when_router_and_fallback_offline() {
    let (router, _) = setup_test_router();
    // Only an unrecognized model is active; neither router nor fallback model is available
    let active_models = vec!["unregistered-peer".to_owned()];

    let request = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "A query that cannot be routed",
        token_count: Some(5_u32),
        constraint: None,
    };

    let result = router.route(&request, &active_models);
    assert!(result.is_err());
    match result.unwrap_err() {
        ManagerError::RoutingUnavailable { reason } => {
            assert!(reason.contains("decision router not loaded and no fallback configured"));
        }
        other => panic!("expected RoutingUnavailable, got {other:?}"),
    }
}

#[test]
fn test_scenario3_routing_unavailable_when_zero_models_active() {
    let (router, _) = setup_test_router();
    let active_models: Vec<String> = Vec::new();

    let request = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "Any query",
        token_count: Some(2_u32),
        constraint: None,
    };

    let result = router.route(&request, &active_models);
    assert!(result.is_err());
    match result.unwrap_err() {
        ManagerError::RoutingUnavailable { reason } => {
            assert!(reason.contains("no models loaded on runtime"));
        }
        other => panic!("expected RoutingUnavailable, got {other:?}"),
    }
}

#[test]
fn test_fallback_escalation_chain_on_low_confidence() {
    let (router, active_models) = setup_test_router();

    let low_confidence_assessment = ExecutionAssessment {
        confidence: 0.35, // Below 0.60 threshold
        quality_passed: true,
        is_refusal: false,
        diagnostic: Some("uncertain output".to_owned()),
    };

    let next_model = router.evaluate_fallback_escalation(
        "qwen2.5-coder-7b",
        &low_confidence_assessment,
        &active_models,
    );

    assert_eq!(next_model, Some("qwen2.5-coder-32b".to_owned()));
}

#[test]
fn test_fallback_escalation_chain_on_model_refusal() {
    let (router, active_models) = setup_test_router();

    let refusal_assessment = ExecutionAssessment {
        confidence: 0.99,
        quality_passed: true,
        is_refusal: true,
        diagnostic: Some("model refused task".to_owned()),
    };

    let next_model = router.evaluate_fallback_escalation(
        "qwen2.5-coder-7b",
        &refusal_assessment,
        &active_models,
    );

    assert_eq!(next_model, Some("qwen2.5-coder-32b".to_owned()));
}

#[test]
fn test_no_escalation_on_high_confidence_success() {
    let (router, active_models) = setup_test_router();

    let successful_assessment = ExecutionAssessment {
        confidence: 0.92,
        quality_passed: true,
        is_refusal: false,
        diagnostic: None,
    };

    let next_model = router.evaluate_fallback_escalation(
        "qwen2.5-coder-7b",
        &successful_assessment,
        &active_models,
    );

    assert_eq!(next_model, None);
}

#[test]
fn test_empty_prompt_defaults_gracefully() {
    let (router, active_models) = setup_test_router();

    let request = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "   \n\t  ",
        token_count: Some(0_u32),
        constraint: None,
    };

    let decision = router
        .route(&request, &active_models)
        .expect("empty prompt routing must succeed");

    assert_eq!(decision.selected_model, "gemma-2-9b-it");
    assert!(!decision.is_fallback);
    assert_eq!(decision.routing_reason, "empty_prompt_default");
}

#[test]
fn test_scenario3_fallback_when_router_offline_even_if_specialist_active() {
    let (router, _) = setup_test_router();
    // tev1 (router) is OFFLINE;
    // BUT qwen2.5-coder-7b (specialist) IS ACTIVE;
    // and gemma-2-9b-it (fallback) IS ACTIVE.
    let active_models = vec!["qwen2.5-coder-7b".to_owned(), "gemma-2-9b-it".to_owned()];

    let request = RoutingRequest {
        target: ModelTarget::Auto,
        prompt: "fn calculate() -> u32 { 42 }",
        token_count: Some(8_u32),
        constraint: None,
    };

    // Because decision router model (tev1) is offline, it must NOT route to specialist;
    // it MUST route to fallback model (gemma-2-9b-it) per US-12 Scenario 3.
    let decision = router
        .route(&request, &active_models)
        .expect("fallback routing must succeed");

    assert_eq!(decision.selected_model, "gemma-2-9b-it");
    assert!(decision.is_fallback);
    assert_eq!(decision.routing_reason, "operator_fallback");
}
