//! Integration tests for US-08 (#8): Operator-Authoritative Tuning Profiles & Advisory Recommendations.
//!
//! # Core System Assertions
//! - **Scenario 1**: Strict preservation of operator profile under online observation.
//!   Runtime observes live prefill/decode pass costs only to refine fair-sharing weights and virtual runtimes.
//!   Configured cluster placement, core reservation, context window, priority class, and lending flags remain unchanged.
//!   The runtime NEVER auto-mutates a configured `TuningProfile`.
//! - **Scenario 2**: On-demand advisory recommendations comparing live metrics against hardware ceilings
//!   (memory bandwidth for decode, MACs for prefill) without auto-applying changes.
//! - **Scenario 3**: Explicit operator profile application via admin command updating configuration and emitting audit event.
//! - **Scenario 4**: Model not found error when requesting suggestions for unknown models.
//! - **Scenario 5**: Rejection of invalid tuning profiles overlapping dedicated control cores.

use std::sync::Arc;

use k3_core::affinity::{ControlPlaneCapacityConfig, ResolvedControlCapacity};
use k3_core::profile::{PriorityClass, ProfileStatus, TuningProfile};
use k3_manager::domain::MockOsAffinityBackend;
use k3_manager::error::ManagerError;
use k3_manager::runtime::{AdminCommand, AdminResponse, ControlPlaneRuntime, ModelAuditEventType};
use k3_manager::scheduler::{MultiModelScheduler, ScheduledPass};

fn setup_capacity() -> ResolvedControlCapacity {
    ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("default capacity resolution succeeds")
}

#[test]
fn test_scenario1_strict_profile_preservation_under_live_observation() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let model_id = "test-qwen-7b";
    let mut initial_tuning = TuningProfile::conservative_default();
    initial_tuning.model_id = Some(model_id.to_owned());
    initial_tuning.cluster = "x100".to_owned();
    initial_tuning.priority_class = PriorityClass::Latency;
    initial_tuning.n_ctx = 4096;
    initial_tuning.lend_idle_cores = false;
    initial_tuning.borrow_idle_cores = false;
    initial_tuning.profile_status = ProfileStatus::Profiled;
    initial_tuning.unprofiled = false;

    // Register model with explicit operator tuning profile
    scheduler
        .admit_or_register_model(
            model_id,
            "x100".to_owned(),
            0x000F, // Cores 0-3
            Some(&initial_tuning),
        )
        .expect("register succeeds");

    // Acquire scheduled pass
    let pass = scheduler
        .try_schedule_forward_pass(model_id)
        .expect("schedule pass succeeds")
        .expect("model is scheduled");

    assert_eq!(pass.allocated_mask, 0x000F);
    assert_eq!(pass.borrowed_mask, 0);

    // Simulate 50 forward passes where live costs diverge by >500% from initial baselines
    for i in 1..=50 {
        let pass = ScheduledPass {
            model_id: model_id.to_owned(),
            allocated_mask: 0x000F,
            borrowed_mask: 0,
            effective_mask: 0x000F,
            borrowed_from: Vec::new(),
            scheduled_at: std::time::Instant::now(),
        };

        // Extreme cost fluctuation: prefill 250ms, decode 120ms
        let prefill_ms = 250.0 + (f64::from(i) * 5.0);
        let decode_ms = 120.0 + (f64::from(i) * 2.0);
        let tokens = 128;

        scheduler.record_pass_completion(&pass, prefill_ms, decode_ms, tokens);
    }

    // Inspect entry after 50 passes of heavy traffic observation
    let entry = scheduler
        .get_model_entry(model_id)
        .expect("entry must exist");

    // Assert internal metrics refined
    assert_eq!(entry.metrics.total_passes, 50);
    assert!(entry.metrics.avg_pass_cost_ms > 200.0);
    assert!(entry.metrics.dynamic_scheduling_weight > 100.0);
    assert!(entry.metrics.virtual_runtime_ms > 0.0);

    // CRITICAL ASSERTION: The operator tuning profile MUST REMAIN STRICTLY UNCHANGED
    assert_eq!(entry.cluster, "x100", "cluster must remain unchanged");
    assert_eq!(
        entry.reservation_mask, 0x000F,
        "reservation mask must remain unchanged"
    );
    assert_eq!(
        entry.priority_class,
        PriorityClass::Latency,
        "priority class must remain unchanged"
    );
    assert_eq!(entry.n_ctx, 4096, "n_ctx must remain unchanged");
    assert!(
        !entry.lend_idle_cores,
        "lend_idle_cores must remain unchanged"
    );
    assert!(
        !entry.borrow_idle_cores,
        "borrow_idle_cores must remain unchanged"
    );
    assert!(!entry.unprofiled, "unprofiled must remain unchanged");

    if let Some(ref profile) = entry.tuning_profile {
        assert_eq!(profile.cluster, "x100");
        assert_eq!(profile.priority_class, PriorityClass::Latency);
        assert_eq!(profile.n_ctx, 4096);
        assert!(!profile.lend_idle_cores);
        assert!(!profile.borrow_idle_cores);
    } else {
        panic!("tuning profile should be cached");
    }
}

#[test]
fn test_scenario2_ondemand_advisory_recommendations() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let model_id = "qwen-prefill-heavy";

    // 1. Register model with X100 placement and throughput priority
    scheduler
        .admit_or_register_model(model_id, "x100".to_owned(), 0x000F, None)
        .expect("register succeeds");

    // Simulate 20 prefill-heavy forward passes
    for _ in 0..20 {
        let pass = ScheduledPass {
            model_id: model_id.to_owned(),
            allocated_mask: 0x000F,
            borrowed_mask: 0,
            effective_mask: 0x000F,
            borrowed_from: Vec::new(),
            scheduled_at: std::time::Instant::now(),
        };
        // Prefill heavy: 500ms prefill for 100 tokens (5ms/token) vs 2ms/token decode
        scheduler.record_pass_completion(&pass, 500.0, 0.0, 100);
    }

    // Request on-demand tuning suggestion
    let advisory = scheduler
        .suggest_tuning(model_id, None)
        .expect("suggest_tuning succeeds");

    assert_eq!(advisory.bottleneck_classification, "compute-bound");
    assert_eq!(advisory.recommended_cluster, "both");
    assert!(advisory.recommended_ime);
    assert!(advisory.rationale.contains("Observed 20 passes"));
    assert!(
        advisory
            .recommendations
            .iter()
            .any(|r| r.contains("Cluster 1 (A100)"))
    );
    assert!(
        advisory
            .operator_authority_notice
            .contains("preserves operator tuning authority")
    );

    // Verify advisory generation NEVER auto-applied changes to active profile
    let entry = scheduler
        .get_model_entry(model_id)
        .expect("entry must exist");
    assert_eq!(
        entry.cluster, "x100",
        "cluster must NOT auto-switch to both"
    );
    assert_eq!(entry.priority_class, PriorityClass::Throughput);
}

#[test]
fn test_scenario2_bandwidth_bound_decode_advisory() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let model_id = "llama-decode-heavy";
    scheduler
        .admit_or_register_model(model_id, "a100".to_owned(), 0x0F00, None)
        .expect("register succeeds");

    // Simulate 20 decode-heavy forward passes (decode latency dominates)
    for _ in 0..20 {
        let pass = ScheduledPass {
            model_id: model_id.to_owned(),
            allocated_mask: 0x0F00,
            borrowed_mask: 0,
            effective_mask: 0x0F00,
            borrowed_from: Vec::new(),
            scheduled_at: std::time::Instant::now(),
        };
        scheduler.record_pass_completion(&pass, 10.0, 80.0, 1);
    }

    let advisory = scheduler
        .suggest_tuning(model_id, None)
        .expect("suggest_tuning succeeds");

    assert_eq!(advisory.bottleneck_classification, "bandwidth-bound");
    assert!(
        advisory
            .recommendations
            .iter()
            .any(|r| r.contains("memory-bandwidth bound"))
    );
}

#[test]
fn test_scenario3_explicit_operator_profile_application_via_admin_command() {
    let capacity = setup_capacity();
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
        .expect("runtime init succeeds");

    let model_id = "unprofiled-to-tuned";
    let reg_cmd = AdminCommand::RegisterModel {
        model_id: model_id.to_owned(),
        weights_path: std::path::PathBuf::from("/tmp/weights.gguf"),
        cluster: "x100".to_owned(),
    };

    let reg_res = runtime.execute_admin(reg_cmd).expect("register succeeds");
    assert_eq!(reg_res, AdminResponse::ModelRegistered(model_id.to_owned()));

    // Verify initial unprofiled state
    let status_cmd = AdminCommand::GetModelStatus {
        model_id: model_id.to_owned(),
    };
    match runtime.execute_admin(status_cmd).expect("status succeeds") {
        AdminResponse::ModelStatus(entry) => {
            assert!(entry.unprofiled);
            assert_eq!(entry.profile_status, "unprofiled");
            assert_eq!(entry.priority_class, "throughput");
        }
        other => panic!("expected ModelStatus, got {other:?}"),
    }

    // Now operator explicitly applies a tuned profile
    let mut tuned_profile = TuningProfile::conservative_default();
    tuned_profile.model_id = Some(model_id.to_owned());
    tuned_profile.cluster = "x100".to_owned();
    tuned_profile.priority_class = PriorityClass::Latency;
    tuned_profile.n_ctx = 1024;
    tuned_profile.lend_idle_cores = true;
    tuned_profile.borrow_idle_cores = true;

    let apply_cmd = AdminCommand::ApplyTuningProfile {
        model_id: model_id.to_owned(),
        profile: tuned_profile,
    };

    let apply_res = runtime
        .execute_admin(apply_cmd)
        .expect("apply profile succeeds");
    assert_eq!(
        apply_res,
        AdminResponse::TuningProfileApplied(model_id.to_owned())
    );

    // Verify updated status in runtime and compute scheduler
    let status_after = match runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: model_id.to_owned(),
        })
        .expect("status succeeds")
    {
        AdminResponse::ModelStatus(entry) => entry,
        other => panic!("expected ModelStatus, got {other:?}"),
    };

    assert!(!status_after.unprofiled);
    assert_eq!(status_after.profile_status, "profiled");
    assert_eq!(status_after.priority_class, "latency");
    assert_eq!(status_after.n_ctx, 1024);
    assert!(status_after.lend_idle_cores);
    assert!(status_after.borrow_idle_cores);

    // Verify audit event trail contains TuningProfileApplied
    let events_res = runtime
        .execute_admin(AdminCommand::GetAuditEvents)
        .expect("get audit events");
    match events_res {
        AdminResponse::AuditEvents(events) => {
            let applied_event = events
                .iter()
                .find(|e| e.event_type == ModelAuditEventType::TuningProfileApplied);
            assert!(applied_event.is_some(), "must record TuningProfileApplied");
            let event = applied_event.unwrap();
            assert_eq!(event.model_id, model_id);
            assert_eq!(event.priority_class, "latency");
        }
        other => panic!("expected AuditEvents, got {other:?}"),
    }

    // Verify on-demand suggest tuning via AdminCommand::SuggestTuning
    let suggest_res = runtime
        .execute_admin(AdminCommand::SuggestTuning {
            model_id: model_id.to_owned(),
        })
        .expect("suggest tuning succeeds");
    match suggest_res {
        AdminResponse::TuningAdvisory(advisory) => {
            assert!(
                advisory
                    .operator_authority_notice
                    .contains("preserves operator tuning authority")
            );
        }
        other => panic!("expected TuningAdvisory, got {other:?}"),
    }
}

#[test]
fn test_scenario4_nonexistent_model_returns_model_not_found() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let err = scheduler.suggest_tuning("nonexistent-model", None);
    assert!(matches!(err, Err(ManagerError::ModelNotFound(id)) if id == "nonexistent-model"));
}

#[test]
fn test_scenario5_invalid_tuning_profile_active_control_overlap_rejection() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    // Attempt to register a model reserving core 6 (0x0040) which is an active control core (0x00C0)
    let bad_mask = 0x0040_u64;
    let err = scheduler.admit_or_register_model("bad-overlap", "x100".to_owned(), bad_mask, None);

    assert!(
        matches!(err, Err(ManagerError::Affinity(_))),
        "reservation overlapping active control mask must be rejected with Affinity error"
    );
}
