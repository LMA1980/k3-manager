//! US-07 Scenario 2 Integration Test Suite: Explicit Unprofiled Status Surfacing & Operator Tuning Transition.
//!
//! Validates:
//! 1. Suite 1: Unprofiled model registration & initial state (`AdminCommand::RegisterModel` and
//!    `AdminCommand::GetModelStatus` surfacing `unprofiled: true`, `profile_status: "unprofiled"`,
//!    and `CONSERVATIVE_TUNING_ADVISORY`).
//! 2. Suite 2: OpenAI `/v1/models` JSON serialization & deserialization (`v1_models_response()`
//!    alphabetical sorting, `v1_models_json()` formatting, lossless round-trip deserialization with serde_json).
//! 3. Suite 3: Control plane admin query decoupled execution on active control threads (cores 6-7, mask 0x00C0).
//! 4. Suite 4: Operational audit event logging lifecycle (`ModelRegistered`, `ModelLoaded`, `ModelStatusQueried`,
//!    `TuningProfileApplied` with monotonic timestamps, unprofiled flags, and advisory notices).
//! 5. Suite 5: Operator tuning profile transition & advisory clearing (`AdminCommand::ApplyTuningProfile`
//!    transitions status to "profiled", unprofiled=false, advisory_notice=None, and verifies that the
//!    `/v1/models` JSON serialization omits advisory_notice).
//! 6. Suite 6: Adversarial profile sanitization, mixed-estate registry lifecycle, and concurrent active
//!    control thread dispatch.

#![allow(
    clippy::pedantic,
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::doc_markdown
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::profile::{
    CONSERVATIVE_DEFAULT_N_CTX, CONSERVATIVE_TUNING_ADVISORY, CONSTRAINED_TEST_N_CTX, ComputePool,
    ModelListResponse, PriorityClass, ProfileStatus, TuningProfile,
};
use k3_manager::OsAffinityBackend;
use k3_manager::capacity::MockOsAffinityBackend;
use k3_manager::error::ManagerError;
use k3_manager::runtime::{AdminCommand, AdminResponse, ControlPlaneRuntime, ModelAuditEventType};

fn build_test_runtime() -> (ControlPlaneRuntime, Arc<MockOsAffinityBackend>) {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(
        capacity,
        Arc::clone(&mock_affinity) as Arc<dyn OsAffinityBackend>,
    )
    .expect("runtime creation");
    (runtime, mock_affinity)
}

// ============================================================================
// Suite 1: Unprofiled Model Registration & Initial State
// ============================================================================

#[test]
fn test_scenario2_unprofiled_model_registration_and_initial_state() {
    let (runtime, _) = build_test_runtime();
    let model_id = "qwen2.5-coder-7b";

    // 1. Register unprofiled model
    let reg_res = runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-coder-7b.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");
    assert_eq!(reg_res, AdminResponse::ModelRegistered(model_id.to_owned()));

    // 2. Query model status via AdminCommand::GetModelStatus
    let status_resp = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: model_id.to_owned(),
        })
        .expect("get model status");

    let entry = match status_resp {
        AdminResponse::ModelStatus(e) => e,
        other => panic!("expected ModelStatus, got {other:?}"),
    };

    assert_eq!(entry.id, model_id);
    assert_eq!(entry.object, "model");
    assert_eq!(entry.owned_by, "k3");
    assert!(entry.unprofiled, "unprofiled must be true");
    assert_eq!(entry.profile_status, "unprofiled");
    assert_eq!(entry.priority_class, "throughput");
    assert_eq!(entry.cluster, "x100");
    assert_eq!(entry.a100_cores, 0);
    assert_eq!(entry.n_ctx, CONSERVATIVE_DEFAULT_N_CTX);
    assert!(!entry.lend_idle_cores);
    assert!(!entry.borrow_idle_cores);
    assert_eq!(
        entry.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );
    assert!(entry.created > 0);
}

// ============================================================================
// Suite 2: OpenAI /v1/models JSON Serialization & Round-Trip Deserialization
// ============================================================================

#[test]
fn test_scenario2_v1_models_json_schema_and_advisory_rendering() {
    let (runtime, _) = build_test_runtime();
    let model_id = "qwen2.5-coder-7b";

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from("/models/qwen2.5-coder-7b.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    let json = runtime.v1_models_json().expect("serialize /v1/models json");

    // Verify presence of all required JSON tokens and conservative default values
    assert!(json.contains("\"object\": \"list\""));
    assert!(json.contains("\"object\": \"model\""));
    assert!(json.contains(&format!("\"id\": \"{model_id}\"")));
    assert!(json.contains("\"owned_by\": \"k3\""));
    assert!(json.contains("\"unprofiled\": true"));
    assert!(json.contains("\"profile_status\": \"unprofiled\""));
    assert!(json.contains("\"priority_class\": \"throughput\""));
    assert!(json.contains("\"cluster\": \"x100\""));
    assert!(json.contains("\"a100_cores\": 0"));
    assert!(json.contains("\"n_ctx\": 2048"));
    assert!(json.contains("\"lend_idle_cores\": false"));
    assert!(json.contains("\"borrow_idle_cores\": false"));
    assert!(json.contains("\"advisory_notice\":"));
    assert!(json.contains(CONSERVATIVE_TUNING_ADVISORY));
}

#[test]
fn test_scenario2_v1_models_deserialization_and_deterministic_sorting() {
    let (runtime, _) = build_test_runtime();

    // 1. Empty registry verification
    let empty_json = runtime.v1_models_json().expect("empty json");
    let empty_deserialized: ModelListResponse =
        serde_json::from_str(&empty_json).expect("parse empty");
    assert_eq!(empty_deserialized.object, "list");
    assert!(empty_deserialized.data.is_empty());

    // 2. Register models in reverse alphabetical order
    let model_names = [
        "zeta-model-7b",
        "alpha-model-7b",
        "mu-model-7b",
        "beta-model-7b",
    ];
    for name in model_names {
        runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: (*name).to_owned(),
                weights_path: PathBuf::from(format!("/models/{name}.gguf")),
                cluster: "x100".to_owned(),
            })
            .expect("register");
    }

    // 3. Serialize and deserialize
    let json = runtime.v1_models_json().expect("serialize json");
    let deserialized: ModelListResponse =
        serde_json::from_str(&json).expect("deserialize /v1/models json");
    assert_eq!(deserialized, runtime.v1_models_response());
    assert_eq!(deserialized.data.len(), 4);

    // 4. Verify deterministic alphabetical ordering
    let ids: Vec<&str> = deserialized.data.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(
        ids,
        vec![
            "alpha-model-7b",
            "beta-model-7b",
            "mu-model-7b",
            "zeta-model-7b"
        ]
    );

    // 5. Verify every entry has valid unprofiled conservative fields
    for entry in &deserialized.data {
        assert!(entry.unprofiled);
        assert_eq!(entry.profile_status, "unprofiled");
        assert_eq!(entry.priority_class, "throughput");
        assert_eq!(entry.cluster, "x100");
        assert_eq!(entry.a100_cores, 0);
        assert_eq!(entry.n_ctx, 2048);
        assert!(!entry.lend_idle_cores);
        assert!(!entry.borrow_idle_cores);
        assert_eq!(
            entry.advisory_notice.as_deref(),
            Some(CONSERVATIVE_TUNING_ADVISORY)
        );
    }
}

// ============================================================================
// Suite 3: Control Plane Admin Query Decoupled Execution (Cores 6-7, 0x00C0)
// ============================================================================

#[test]
fn test_scenario2_admin_command_get_model_status_and_list_models() {
    let (runtime, mock_affinity) = build_test_runtime();
    let model_id = "deepseek-r1-distill";

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from("/models/deepseek-r1.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Verify runtime worker pool was initialized and pinned to active control cores 0x00C0
    let aff_calls = mock_affinity.calls();
    assert!(
        !aff_calls.is_empty(),
        "runtime worker threads must be pinned"
    );
    for &(_, mask) in &aff_calls {
        assert_ne!(
            mask & 0x00C0,
            0,
            "control plane worker threads must pin to a core within active control mask 0x00C0"
        );
        assert_eq!(
            mask & !0x00C0,
            0,
            "control plane worker threads must never pin outside active control mask 0x00C0"
        );
    }

    // 1. Direct execution of GetModelStatus
    let status_resp = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: model_id.to_owned(),
        })
        .expect("direct get model status");

    let entry = match status_resp {
        AdminResponse::ModelStatus(e) => e,
        other => panic!("expected ModelStatus, got {other:?}"),
    };
    assert_eq!(entry.id, model_id);
    assert!(entry.unprofiled);
    assert_eq!(entry.profile_status, "unprofiled");
    assert_eq!(
        entry.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    // 2. Decoupled control thread dispatch execution via dispatch_execute_admin (ADR-007)
    let dispatched_resp = runtime
        .dispatch_execute_admin(AdminCommand::GetModelStatus {
            model_id: model_id.to_owned(),
        })
        .expect("dispatched get model status");
    assert_eq!(dispatched_resp, AdminResponse::ModelStatus(entry.clone()));

    // 3. AdminCommand::ListModels via dispatch
    let list_resp = runtime
        .dispatch_execute_admin(AdminCommand::ListModels)
        .expect("dispatched list models");
    match list_resp {
        AdminResponse::ModelList(list) => {
            assert_eq!(list.len(), 1);
            assert_eq!(list[0], entry);
        }
        other => panic!("expected ModelList, got {other:?}"),
    }

    // 4. Querying nonexistent model returns ModelNotFound
    let err_not_found = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: "ghost-model".to_owned(),
        })
        .unwrap_err();
    assert!(matches!(err_not_found, ManagerError::ModelNotFound(_)));

    // 5. Querying with dangerous / path traversal model IDs fails security validation
    let err_traversal = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: "../escape-path".to_owned(),
        })
        .unwrap_err();
    assert!(matches!(err_traversal, ManagerError::WorkerSpawn(_)));

    let err_cgroup = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: "cgroup.procs".to_owned(),
        })
        .unwrap_err();
    assert!(matches!(err_cgroup, ManagerError::WorkerSpawn(_)));
}

// ============================================================================
// Suite 4: Operational Audit Event Logging Lifecycle
// ============================================================================

#[test]
fn test_scenario2_operational_audit_events_lifecycle_and_advisory() {
    let (runtime, _) = build_test_runtime();
    runtime.clear_audit_events();
    assert!(runtime.audit_events().is_empty());

    let model_id = "audit-demo-model";

    // 1. ModelRegistered event
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from("/models/audit-demo.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register");

    // 2. ModelLoaded event
    runtime.record_model_placement(model_id.to_owned(), "x100".to_owned(), Some(4242));

    // 3. ModelStatusQueried event via AdminCommand
    let _ = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: model_id.to_owned(),
        })
        .expect("get status");

    // 4. ModelStatusQueried event via get_model_entry
    let _ = runtime.get_model_entry(model_id).expect("model entry");

    // 5. TuningProfileApplied event
    let mut tuned = TuningProfile::conservative_default();
    tuned.apply_operator_tuning(
        ComputePool::X100,
        4,
        0,
        PriorityClass::Throughput,
        1024,
        false,
        false,
    );
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: model_id.to_owned(),
            profile: tuned,
        })
        .expect("apply tuning");

    // 6. ModelStatusQueried post-tuning
    let _ = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: model_id.to_owned(),
        })
        .expect("get status post-tuning");

    // Query audit events via AdminCommand::GetAuditEvents
    let audit_resp = runtime
        .execute_admin(AdminCommand::GetAuditEvents)
        .expect("get audit events");
    let events = match audit_resp {
        AdminResponse::AuditEvents(evts) => evts,
        other => panic!("expected AuditEvents, got {other:?}"),
    };

    assert_eq!(events.len(), 6);

    // Event 0: ModelRegistered (unprofiled = true)
    assert_eq!(events[0].event_type, ModelAuditEventType::ModelRegistered);
    assert_eq!(events[0].model_id, model_id);
    assert!(events[0].unprofiled);
    assert_eq!(events[0].profile_status, "unprofiled");
    assert_eq!(events[0].advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));
    assert!(events[0].details.contains("registered model"));

    // Event 1: ModelLoaded (unprofiled = true)
    assert_eq!(events[1].event_type, ModelAuditEventType::ModelLoaded);
    assert_eq!(events[1].model_id, model_id);
    assert!(events[1].unprofiled);
    assert_eq!(events[1].advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));

    // Event 2 & 3: ModelStatusQueried (unprofiled = true)
    for evt in &events[2..4] {
        assert_eq!(evt.event_type, ModelAuditEventType::ModelStatusQueried);
        assert_eq!(evt.model_id, model_id);
        assert!(evt.unprofiled);
        assert_eq!(evt.advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));
    }

    // Event 4: TuningProfileApplied (unprofiled = false)
    assert_eq!(
        events[4].event_type,
        ModelAuditEventType::TuningProfileApplied
    );
    assert_eq!(events[4].model_id, model_id);
    assert!(!events[4].unprofiled);
    assert_eq!(events[4].profile_status, "profiled");
    assert_eq!(events[4].advisory(), None);
    assert!(
        events[4]
            .details
            .contains("applied operator tuning profile")
    );

    // Event 5: ModelStatusQueried post-tuning (unprofiled = false)
    assert_eq!(
        events[5].event_type,
        ModelAuditEventType::ModelStatusQueried
    );
    assert_eq!(events[5].model_id, model_id);
    assert!(!events[5].unprofiled);
    assert_eq!(events[5].profile_status, "profiled");
    assert_eq!(events[5].advisory(), None);

    // Monotonic timestamps verification
    for w in events.windows(2) {
        assert!(
            w[0].timestamp_ms <= w[1].timestamp_ms,
            "timestamps must be non-decreasing: {} vs {}",
            w[0].timestamp_ms,
            w[1].timestamp_ms
        );
    }

    // Test clear_audit_events
    runtime.clear_audit_events();
    assert!(runtime.audit_events().is_empty());
}

// ============================================================================
// Suite 5: Operator Tuning Profile Transition & Advisory Clearing
// ============================================================================

#[test]
fn test_scenario2_operator_tuning_profile_transition_clears_unprofiled() {
    let (runtime, _) = build_test_runtime();
    let model_id = "llama3-70b-instruct";

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from("/models/llama3-70b.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register");

    // Initially unprofiled
    let init_status = runtime.get_model_entry(model_id).expect("found");
    assert!(init_status.unprofiled);
    assert_eq!(init_status.profile_status, "unprofiled");
    assert_eq!(
        init_status.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    let init_json = runtime.v1_models_json().expect("init json");
    assert!(init_json.contains("\"unprofiled\": true"));
    assert!(init_json.contains("\"advisory_notice\":"));

    // Operator applies tuning profile (ADR-031, ADR-032)
    let mut tuned_profile = TuningProfile::conservative_default();
    tuned_profile.apply_operator_tuning(
        ComputePool::A100,
        0,
        8,
        PriorityClass::Latency,
        4096,
        true,
        false,
    );

    let apply_res = runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: model_id.to_owned(),
            profile: tuned_profile,
        })
        .expect("apply tuning profile");
    assert_eq!(
        apply_res,
        AdminResponse::TuningProfileApplied(model_id.to_owned())
    );

    // Post-transition assertions
    let tuned_entry = runtime
        .get_model_entry(model_id)
        .expect("found post-tuning");
    assert!(!tuned_entry.unprofiled, "unprofiled flag must be false");
    assert_eq!(
        tuned_entry.profile_status, "profiled",
        "status must become profiled"
    );
    assert_eq!(
        tuned_entry.advisory_notice, None,
        "advisory notice must be cleared"
    );
    assert_eq!(tuned_entry.cluster, "a100");
    assert_eq!(tuned_entry.a100_cores, 8);
    assert_eq!(tuned_entry.priority_class, "latency");
    assert_eq!(tuned_entry.n_ctx, 4096);
    assert!(tuned_entry.lend_idle_cores);
    assert!(!tuned_entry.borrow_idle_cores);

    // Assert /v1/models JSON strictly omits advisory_notice
    let tuned_json = runtime.v1_models_json().expect("tuned json");
    assert!(tuned_json.contains("\"unprofiled\": false"));
    assert!(tuned_json.contains("\"profile_status\": \"profiled\""));
    assert!(tuned_json.contains("\"cluster\": \"a100\""));
    assert!(tuned_json.contains("\"priority_class\": \"latency\""));
    assert!(tuned_json.contains("\"n_ctx\": 4096"));
    assert!(
        !tuned_json.contains("advisory_notice"),
        "advisory_notice key must be omitted from JSON when None"
    );
}

// ============================================================================
// Suite 6: Adversarial Profile Sanitization & Mixed-Estate Concurrency
// ============================================================================

#[test]
fn test_scenario2_adversarial_tuning_profile_sanitization() {
    let (runtime, _) = build_test_runtime();
    let model_id = "adversarial-target-model";

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from("/models/target.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register");

    // Construct an adversarial profile maliciously attempting to keep unprofiled true with fake advisory
    let mut bad_profile = TuningProfile::conservative_default();
    bad_profile.unprofiled = true;
    bad_profile.profile_status = ProfileStatus::Unprofiled;
    bad_profile.advisory_notice = Some("FAKE_MALICIOUS_ADVISORY".to_owned());
    bad_profile.cluster = "a100".to_owned();
    bad_profile.n_ctx = 8192;

    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: model_id.to_owned(),
            profile: bad_profile,
        })
        .expect("apply");

    let entry = runtime.get_model_entry(model_id).expect("entry found");
    assert!(
        !entry.unprofiled,
        "manager must enforce unprofiled=false on tuning"
    );
    assert_eq!(
        entry.profile_status, "profiled",
        "manager must enforce profile_status=profiled"
    );
    assert_eq!(
        entry.advisory_notice, None,
        "manager must clear advisory notice regardless of input"
    );

    let json = runtime.v1_models_json().expect("json");
    assert!(!json.contains("FAKE_MALICIOUS_ADVISORY"));
}

#[test]
fn test_scenario2_mixed_estate_end_to_end_lifecycle() {
    let (runtime, _) = build_test_runtime();

    // Model 1: Default unprofiled (n_ctx = 2048)
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "m1-default-unprofiled".to_owned(),
            weights_path: PathBuf::from("/models/m1.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register m1");

    // Model 2: Constrained unprofiled (n_ctx = 64)
    let profile_64 = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_TEST_N_CTX);
    runtime.record_model_placement_with_tuning(
        "m2-constrained-unprofiled".to_owned(),
        "x100".to_owned(),
        Some(1002),
        Some(profile_64),
    );

    // Model 3: Operator tuned (A100, latency, n_ctx = 4096)
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "m3-operator-tuned".to_owned(),
            weights_path: PathBuf::from("/models/m3.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register m3");
    let mut tuned = TuningProfile::conservative_default();
    tuned.apply_operator_tuning(
        ComputePool::A100,
        0,
        8,
        PriorityClass::Latency,
        4096,
        true,
        false,
    );
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "m3-operator-tuned".to_owned(),
            profile: tuned,
        })
        .expect("tune m3");

    // Inspect /v1/models collection
    let resp = runtime.v1_models_response();
    assert_eq!(resp.data.len(), 3);

    // Verify m1
    let m1 = resp
        .data
        .iter()
        .find(|e| e.id == "m1-default-unprofiled")
        .unwrap();
    assert!(m1.unprofiled);
    assert_eq!(m1.n_ctx, 2048);
    assert_eq!(
        m1.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    // Verify m2
    let m2 = resp
        .data
        .iter()
        .find(|e| e.id == "m2-constrained-unprofiled")
        .unwrap();
    assert!(m2.unprofiled);
    assert_eq!(m2.n_ctx, 64);
    assert_eq!(
        m2.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    // Verify m3
    let m3 = resp
        .data
        .iter()
        .find(|e| e.id == "m3-operator-tuned")
        .unwrap();
    assert!(!m3.unprofiled);
    assert_eq!(m3.profile_status, "profiled");
    assert_eq!(m3.n_ctx, 4096);
    assert_eq!(m3.cluster, "a100");
    assert_eq!(m3.advisory_notice, None);

    // Unregister m1
    runtime
        .execute_admin(AdminCommand::UnregisterModel {
            model_id: "m1-default-unprofiled".to_owned(),
        })
        .expect("unregister m1");
    let resp_after = runtime.v1_models_response();
    assert_eq!(resp_after.data.len(), 2);
    assert!(
        resp_after
            .data
            .iter()
            .all(|e| e.id != "m1-default-unprofiled")
    );

    // Re-register m1 restores conservative unprofiled defaults
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "m1-default-unprofiled".to_owned(),
            weights_path: PathBuf::from("/models/m1.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("re-register m1");
    let m1_restored = runtime
        .get_model_entry("m1-default-unprofiled")
        .expect("m1 restored");
    assert!(m1_restored.unprofiled);
    assert_eq!(m1_restored.profile_status, "unprofiled");
    assert_eq!(
        m1_restored.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );
}

#[test]
fn test_scenario2_concurrent_active_control_thread_dispatch() {
    let (runtime_raw, _) = build_test_runtime();
    let runtime = Arc::new(runtime_raw);

    // Pre-register models
    for i in 0..8 {
        runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: format!("conc-model-{i}"),
                weights_path: PathBuf::from(format!("/models/m{i}.gguf")),
                cluster: "x100".to_owned(),
            })
            .expect("register");
    }

    const CLIENT_THREADS: usize = 4;
    const QUERIES_PER_THREAD: usize = 50;

    let mut handles = Vec::with_capacity(CLIENT_THREADS);
    for t_idx in 0..CLIENT_THREADS {
        let rt = Arc::clone(&runtime);
        handles.push(thread::spawn(move || {
            for q_idx in 0..QUERIES_PER_THREAD {
                let m_id = format!("conc-model-{}", (t_idx + q_idx) % 8);
                // Dispatch GetModelStatus to active control thread pool
                let status_res =
                    rt.dispatch_execute_admin(AdminCommand::GetModelStatus { model_id: m_id });
                assert!(status_res.is_ok());

                // Dispatch ListModels
                let list_res = rt.dispatch_execute_admin(AdminCommand::ListModels);
                assert!(list_res.is_ok());
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join cleanly");
    }

    // Verify all models remained consistent
    let final_resp = runtime.v1_models_response();
    assert_eq!(final_resp.data.len(), 8);
    for entry in &final_resp.data {
        assert!(entry.unprofiled);
        assert_eq!(
            entry.advisory_notice.as_deref(),
            Some(CONSERVATIVE_TUNING_ADVISORY)
        );
    }
}
