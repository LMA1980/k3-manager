//! US-07 Milestone 3 Scenario 2 Empirical Challenge Test Suite.
//!
//! Adversarial stress harness validating:
//! 1. OpenAI `/v1/models` JSON serialization, schema invariants, large estate (100+ models),
//!    round-trip deserialization, field omission (`advisory_notice` strictly absent when profiled),
//!    and deterministic lexical sorting.
//! 2. High-volume concurrency on control plane admin commands: mixed registration, status querying,
//!    tuning transitions, audit queries, and model listings under parallel reader/writer contention
//!    on active control threads without lock inversion, deadlocks, or state corruption.
//! 3. Operational audit log lifecycle: causal event sequence verification, monotonic timestamp
//!    ordering, advisory propagation, and audit history preservation across unregister/re-register.
//! 4. Operator tuning profile transitions: multi-stage successive transitions, cluster/priority/n_ctx
//!    mutations, and adversarial profile sanitization (preventing spoofed unprofiled/advisory flags).

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
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::profile::{
    CONSERVATIVE_DEFAULT_N_CTX, CONSERVATIVE_TUNING_ADVISORY, ComputePool, ModelListResponse,
    PriorityClass, ProfileStatus, TuningProfile,
};
use k3_manager::OsAffinityBackend;
use k3_manager::capacity::MockOsAffinityBackend;
use k3_manager::error::ManagerError;
use k3_manager::runtime::{AdminCommand, ControlPlaneRuntime, ModelAuditEventType};

fn build_challenge_runtime() -> (ControlPlaneRuntime, Arc<MockOsAffinityBackend>) {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid control capacity configuration");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(
        capacity,
        Arc::clone(&mock_affinity) as Arc<dyn OsAffinityBackend>,
    )
    .expect("control plane runtime creation");
    (runtime, mock_affinity)
}

// ============================================================================
// 1. OpenAI /v1/models JSON Serialization & Round-Trip Deserialization Stress
// ============================================================================

#[test]
fn test_challenge_v1_models_large_estate_roundtrip_and_deterministic_sorting() {
    let (runtime, _) = build_challenge_runtime();

    // 1. Verify empty estate JSON
    let empty_json = runtime.v1_models_json().expect("empty v1 models json");
    let empty_val: serde_json::Value =
        serde_json::from_str(&empty_json).expect("valid json for empty estate");
    assert_eq!(empty_val["object"], "list");
    assert_eq!(empty_val["data"], serde_json::json!([]));

    let empty_roundtrip: ModelListResponse =
        serde_json::from_str(&empty_json).expect("deserialize empty list");
    assert_eq!(empty_roundtrip, runtime.v1_models_response());

    // 2. Register 100 models in deliberately shuffled / pseudorandom order
    const MODEL_COUNT: usize = 100;
    let mut generated_ids = Vec::with_capacity(MODEL_COUNT);
    for i in 0..MODEL_COUNT {
        // Pseudo-hash to generate non-sequential model IDs
        let pseudo_hash = (i * 73 + 17) % MODEL_COUNT;
        let model_id = format!("model-{pseudo_hash:03}-tier-{}", i % 5);
        generated_ids.push(model_id.clone());

        runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: model_id.clone(),
                weights_path: PathBuf::from(format!("/models/{model_id}.gguf")),
                cluster: "x100".to_owned(),
            })
            .expect("register model");
    }

    // 3. Apply operator tuning to every 3rd model (creating a mixed estate)
    for (idx, model_id) in generated_ids.iter().enumerate() {
        if idx % 3 == 0 {
            let mut tuned = TuningProfile::conservative_default();
            tuned.apply_operator_tuning(
                if idx % 2 == 0 {
                    ComputePool::A100
                } else {
                    ComputePool::Both
                },
                4,
                4,
                PriorityClass::Latency,
                4096 + idx * 128,
                true,
                false,
            );
            runtime
                .execute_admin(AdminCommand::ApplyTuningProfile {
                    model_id: model_id.clone(),
                    profile: tuned,
                })
                .expect("apply tuning profile");
        }
    }

    // 4. Generate /v1/models JSON and verify lossless round-trip deserialization
    let json_str = runtime.v1_models_json().expect("serialize mixed estate");
    let deserialized: ModelListResponse =
        serde_json::from_str(&json_str).expect("lossless round-trip deserialization");

    let live_response = runtime.v1_models_response();
    assert_eq!(
        deserialized, live_response,
        "round-trip deserialized response must strictly match in-memory response"
    );
    assert_eq!(deserialized.data.len(), MODEL_COUNT);

    // 5. Verify deterministic alphabetical ordering of all 100 models
    let mut sorted_ids = generated_ids.clone();
    sorted_ids.sort_unstable();
    sorted_ids.dedup();

    let actual_ids: Vec<String> = deserialized.data.iter().map(|e| e.id.clone()).collect();
    assert_eq!(
        actual_ids, sorted_ids,
        "deserialized model entries must be strictly sorted alphabetically by id"
    );

    // 6. Inspect individual model entries and JSON token representations
    let parsed_json: serde_json::Value = serde_json::from_str(&json_str).expect("valid json value");
    let json_items = parsed_json["data"].as_array().expect("data array");

    for (entry, json_val) in deserialized.data.iter().zip(json_items) {
        assert_eq!(json_val["id"], entry.id);
        assert_eq!(json_val["object"], "model");
        assert_eq!(json_val["owned_by"], "k3");
        assert_eq!(json_val["unprofiled"], entry.unprofiled);
        assert_eq!(json_val["profile_status"], entry.profile_status);

        if entry.unprofiled {
            assert_eq!(entry.profile_status, "unprofiled");
            assert_eq!(
                entry.advisory_notice.as_deref(),
                Some(CONSERVATIVE_TUNING_ADVISORY)
            );
            assert_eq!(
                json_val["advisory_notice"].as_str(),
                Some(CONSERVATIVE_TUNING_ADVISORY),
                "unprofiled models must render advisory notice in JSON"
            );
        } else {
            assert_eq!(entry.profile_status, "profiled");
            assert_eq!(entry.advisory_notice, None);
            assert!(
                json_val.get("advisory_notice").is_none(),
                "profiled models must completely omit 'advisory_notice' key from JSON"
            );
        }
    }
}

#[test]
fn test_challenge_v1_models_schema_robustness_and_optional_fields() {
    // Test robustness when parsing JSON with optional fields, null fields, or extra unknown fields
    let sample_raw_json = r#"{
        "object": "list",
        "data": [
            {
                "id": "model-unprofiled-standard",
                "object": "model",
                "created": 1730000000,
                "owned_by": "k3",
                "unprofiled": true,
                "profile_status": "unprofiled",
                "priority_class": "throughput",
                "cluster": "x100",
                "a100_cores": 0,
                "n_ctx": 2048,
                "lend_idle_cores": false,
                "borrow_idle_cores": false,
                "advisory_notice": "Model is running with safe conservative defaults (unprofiled). Run 'k3-diag' to profile hardware performance and apply an operator tuning profile.",
                "unknown_future_field": "ignore_me"
            },
            {
                "id": "model-profiled-omitted-advisory",
                "object": "model",
                "created": 1730000010,
                "owned_by": "k3",
                "unprofiled": false,
                "profile_status": "profiled",
                "priority_class": "latency",
                "cluster": "a100",
                "a100_cores": 8,
                "n_ctx": 8192,
                "lend_idle_cores": true,
                "borrow_idle_cores": false
            },
            {
                "id": "model-profiled-null-advisory",
                "object": "model",
                "created": 1730000020,
                "owned_by": "k3",
                "unprofiled": false,
                "profile_status": "profiled",
                "priority_class": "throughput",
                "cluster": "both",
                "a100_cores": 8,
                "n_ctx": 4096,
                "lend_idle_cores": false,
                "borrow_idle_cores": false,
                "advisory_notice": null
            }
        ]
    }"#;

    let parsed: ModelListResponse = serde_json::from_str(sample_raw_json)
        .expect("deserialization with unknown and null fields");
    assert_eq!(parsed.data.len(), 3);

    // Entry 0: unprofiled
    assert!(parsed.data[0].unprofiled);
    assert_eq!(parsed.data[0].profile_status, "unprofiled");
    assert_eq!(
        parsed.data[0].advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    // Entry 1: omitted advisory
    assert!(!parsed.data[1].unprofiled);
    assert_eq!(parsed.data[1].profile_status, "profiled");
    assert_eq!(parsed.data[1].advisory_notice, None);

    // Entry 2: null advisory
    assert!(!parsed.data[2].unprofiled);
    assert_eq!(parsed.data[2].profile_status, "profiled");
    assert_eq!(parsed.data[2].advisory_notice, None);
}

// ============================================================================
// 2. High-Volume Concurrency Stress on Control Plane Admin Commands
// ============================================================================

#[test]
fn test_challenge_concurrency_stress_admin_plane_parallel_readers_and_writers() {
    let (runtime_raw, _) = build_challenge_runtime();
    let runtime = Arc::new(runtime_raw);

    // Seed initial models
    const BASE_MODELS: usize = 12;
    for i in 0..BASE_MODELS {
        runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: format!("seed-model-{i}"),
                weights_path: PathBuf::from(format!("/models/seed-{i}.gguf")),
                cluster: "x100".to_owned(),
            })
            .expect("register seed model");
    }

    const WORKER_THREADS: usize = 16;
    const OPS_PER_THREAD: usize = 150; // 16 * 150 = 2,400 concurrent operations

    let start_gate = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::with_capacity(WORKER_THREADS);

    for t_idx in 0..WORKER_THREADS {
        let rt = Arc::clone(&runtime);
        let gate = Arc::clone(&start_gate);

        handles.push(thread::spawn(move || {
            // Spin until synchronized start
            while !gate.load(Ordering::Acquire) {
                thread::yield_now();
            }

            for op_idx in 0..OPS_PER_THREAD {
                let target_seed = format!("seed-model-{}", (t_idx + op_idx) % BASE_MODELS);

                match (t_idx + op_idx) % 6 {
                    0 => {
                        // GetModelStatus via direct execution
                        let res = rt.execute_admin(AdminCommand::GetModelStatus {
                            model_id: target_seed,
                        });
                        assert!(res.is_ok());
                    }
                    1 => {
                        // GetModelStatus via dispatched execution on control thread pool
                        let res = rt.dispatch_execute_admin(AdminCommand::GetModelStatus {
                            model_id: target_seed,
                        });
                        assert!(res.is_ok());
                    }
                    2 => {
                        // ListModels via dispatch
                        let res = rt.dispatch_execute_admin(AdminCommand::ListModels);
                        assert!(res.is_ok());
                    }
                    3 => {
                        // v1_models_response & v1_models_json serialization
                        let resp = rt.v1_models_response();
                        assert!(resp.data.len() >= BASE_MODELS);
                        let json = rt.v1_models_json();
                        assert!(json.is_ok());
                    }
                    4 => {
                        // Transient model churn: Register and Unregister
                        let churn_id = format!("churn-t{t_idx}-op{op_idx}");
                        let reg_res = rt.execute_admin(AdminCommand::RegisterModel {
                            model_id: churn_id.clone(),
                            weights_path: PathBuf::from(format!("/models/{churn_id}.gguf")),
                            cluster: "x100".to_owned(),
                        });
                        assert!(reg_res.is_ok());

                        let unreg_res =
                            rt.execute_admin(AdminCommand::UnregisterModel { model_id: churn_id });
                        assert!(unreg_res.is_ok());
                    }
                    5 => {
                        // Apply operator tuning profile update
                        let mut tuned = TuningProfile::conservative_default();
                        tuned.apply_operator_tuning(
                            ComputePool::X100,
                            5,
                            0,
                            PriorityClass::Latency,
                            2048 + (op_idx % 4) * 512,
                            false,
                            false,
                        );
                        let tune_res = rt.execute_admin(AdminCommand::ApplyTuningProfile {
                            model_id: target_seed,
                            profile: tuned,
                        });
                        assert!(tune_res.is_ok());
                    }
                    _ => unreachable!(),
                }
            }
        }));
    }

    // Release all threads simultaneously
    start_gate.store(true, Ordering::Release);

    for h in handles {
        h.join().expect("thread join must not panic or deadlock");
    }

    // Post-stress validation: verify all base models are still present and healthy
    let final_resp = runtime.v1_models_response();
    assert_eq!(
        final_resp.data.len(),
        BASE_MODELS,
        "transient churn models must be cleanly unregistered, leaving exactly BASE_MODELS"
    );

    // Audit events must be populated without corruption
    let audit_evts = runtime.audit_events();
    assert!(
        audit_evts.len() >= BASE_MODELS,
        "audit trail must contain all operations recorded during concurrency"
    );

    // Verify timestamp monotonicity across recorded events
    for w in audit_evts.windows(2) {
        assert!(
            w[0].timestamp_ms <= w[1].timestamp_ms,
            "timestamps must be monotonically non-decreasing: {} vs {}",
            w[0].timestamp_ms,
            w[1].timestamp_ms
        );
    }
}

// ============================================================================
// 3. Operational Audit Log Ordering, Monotonicity & Lifecycle Preservation
// ============================================================================

#[test]
fn test_challenge_audit_log_causal_ordering_and_preservation_across_lifecycle() {
    let (runtime, _) = build_challenge_runtime();
    runtime.clear_audit_events();
    assert!(runtime.audit_events().is_empty());

    let model_id = "lifecycle-audit-target";

    // 1. Register unprofiled model -> ModelRegistered event
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from("/models/target.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // 2. Query model status -> ModelStatusQueried event
    let _ = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: model_id.to_owned(),
        })
        .expect("get model status");

    // 3. Read model entry -> ModelStatusQueried event
    let _ = runtime.get_model_entry(model_id).expect("get model entry");

    // 4. Simulate worker placement load -> ModelLoaded event
    runtime.record_model_placement(model_id.to_owned(), "x100".to_owned(), Some(9999));

    // 5. Apply operator tuning profile -> TuningProfileApplied event
    let mut tuned = TuningProfile::conservative_default();
    tuned.apply_operator_tuning(
        ComputePool::A100,
        0,
        8,
        PriorityClass::Latency,
        8192,
        true,
        false,
    );
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: model_id.to_owned(),
            profile: tuned,
        })
        .expect("apply tuning");

    // 6. Query model status post-tuning -> ModelStatusQueried event
    let _ = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: model_id.to_owned(),
        })
        .expect("get model status post-tuning");

    // 7. Unregister model -> verifies model is removed from registry
    runtime
        .execute_admin(AdminCommand::UnregisterModel {
            model_id: model_id.to_owned(),
        })
        .expect("unregister model");
    assert!(runtime.get_model_entry(model_id).is_none());

    // 8. Re-register model under conservative unprofiled defaults -> ModelRegistered event
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from("/models/target.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("re-register model");

    // Verify audit events snapshot
    let events = runtime.audit_events();
    assert_eq!(
        events.len(),
        7,
        "audit trail must record all 7 events across full lifecycle"
    );

    // Event 0: ModelRegistered (unprofiled=true)
    assert_eq!(events[0].event_type, ModelAuditEventType::ModelRegistered);
    assert_eq!(events[0].model_id, model_id);
    assert!(events[0].unprofiled);
    assert_eq!(events[0].profile_status, "unprofiled");
    assert_eq!(events[0].advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));

    // Event 1 & 2: ModelStatusQueried (unprofiled=true)
    for evt in &events[1..=2] {
        assert_eq!(evt.event_type, ModelAuditEventType::ModelStatusQueried);
        assert_eq!(evt.model_id, model_id);
        assert!(evt.unprofiled);
        assert_eq!(evt.advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));
    }

    // Event 3: ModelLoaded (unprofiled=true)
    assert_eq!(events[3].event_type, ModelAuditEventType::ModelLoaded);
    assert_eq!(events[3].model_id, model_id);
    assert!(events[3].unprofiled);
    assert_eq!(events[3].advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));

    // Event 4: TuningProfileApplied (unprofiled=false)
    assert_eq!(
        events[4].event_type,
        ModelAuditEventType::TuningProfileApplied
    );
    assert_eq!(events[4].model_id, model_id);
    assert!(!events[4].unprofiled);
    assert_eq!(events[4].profile_status, "profiled");
    assert_eq!(events[4].advisory(), None);

    // Event 5: ModelStatusQueried post-tuning (unprofiled=false)
    assert_eq!(
        events[5].event_type,
        ModelAuditEventType::ModelStatusQueried
    );
    assert_eq!(events[5].model_id, model_id);
    assert!(!events[5].unprofiled);
    assert_eq!(events[5].profile_status, "profiled");
    assert_eq!(events[5].advisory(), None);

    // Event 6: Re-registered model (unprofiled=true restored)
    assert_eq!(events[6].event_type, ModelAuditEventType::ModelRegistered);
    assert_eq!(events[6].model_id, model_id);
    assert!(
        events[6].unprofiled,
        "re-registered model must restore unprofiled=true"
    );
    assert_eq!(events[6].profile_status, "unprofiled");
    assert_eq!(events[6].advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));

    // Check strict monotonicity of timestamps
    for w in events.windows(2) {
        assert!(
            w[0].timestamp_ms <= w[1].timestamp_ms,
            "audit event timestamps must be non-decreasing"
        );
    }
}

// ============================================================================
// 4. Operator Tuning Transitions & Adversarial Profile Ingestion
// ============================================================================

#[test]
fn test_challenge_operator_tuning_multi_transition_and_adversarial_sanitization() {
    let (runtime, _) = build_challenge_runtime();
    let model_id = "multi-tuning-target";

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from("/models/multi.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Phase 0: Baseline unprofiled
    let base_entry = runtime.get_model_entry(model_id).expect("base entry");
    assert!(base_entry.unprofiled);
    assert_eq!(base_entry.cluster, "x100");
    assert_eq!(base_entry.n_ctx, CONSERVATIVE_DEFAULT_N_CTX);
    assert_eq!(
        base_entry.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    // Phase 1: First operator tuning -> X100, Latency, n_ctx = 4096
    let mut tuning1 = TuningProfile::conservative_default();
    tuning1.apply_operator_tuning(
        ComputePool::X100,
        6,
        0,
        PriorityClass::Latency,
        4096,
        false,
        false,
    );
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: model_id.to_owned(),
            profile: tuning1,
        })
        .expect("tuning 1");

    let entry1 = runtime.get_model_entry(model_id).expect("entry 1");
    assert!(!entry1.unprofiled);
    assert_eq!(entry1.profile_status, "profiled");
    assert_eq!(entry1.priority_class, "latency");
    assert_eq!(entry1.n_ctx, 4096);
    assert_eq!(entry1.advisory_notice, None);

    // Phase 2: Second operator tuning -> elevate to A100, n_ctx = 8192, core lending enabled
    let mut tuning2 = TuningProfile::conservative_default();
    tuning2.apply_operator_tuning(
        ComputePool::A100,
        0,
        8,
        PriorityClass::Throughput,
        8192,
        true,
        false,
    );
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: model_id.to_owned(),
            profile: tuning2,
        })
        .expect("tuning 2");

    let entry2 = runtime.get_model_entry(model_id).expect("entry 2");
    assert!(!entry2.unprofiled);
    assert_eq!(entry2.profile_status, "profiled");
    assert_eq!(entry2.cluster, "a100");
    assert_eq!(entry2.n_ctx, 8192);
    assert!(entry2.lend_idle_cores);
    assert_eq!(entry2.advisory_notice, None);

    // Phase 3: Adversarial tuning injection -> Maliciously attempt to set unprofiled=true with custom advisory
    let mut adversarial_profile = TuningProfile::conservative_default();
    adversarial_profile.unprofiled = true;
    adversarial_profile.profile_status = ProfileStatus::Unprofiled;
    adversarial_profile.advisory_notice = Some("SPOOFED_UNPROFILED_ADVISORY".to_owned());
    adversarial_profile.cluster = "both".to_owned();
    adversarial_profile.n_ctx = 16384;

    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: model_id.to_owned(),
            profile: adversarial_profile,
        })
        .expect("adversarial tuning apply");

    let entry3 = runtime.get_model_entry(model_id).expect("entry 3");
    assert!(
        !entry3.unprofiled,
        "manager must force unprofiled=false when applying operator tuning"
    );
    assert_eq!(
        entry3.profile_status, "profiled",
        "manager must force profile_status=profiled"
    );
    assert_eq!(
        entry3.advisory_notice, None,
        "manager must strictly clear advisory_notice regardless of payload content"
    );
    assert_eq!(entry3.cluster, "both");
    assert_eq!(entry3.n_ctx, 16384);

    // Verify /v1/models JSON strictly omits "SPOOFED_UNPROFILED_ADVISORY" and "advisory_notice"
    let json = runtime.v1_models_json().expect("json");
    assert!(!json.contains("SPOOFED_UNPROFILED_ADVISORY"));
    assert!(!json.contains("advisory_notice"));

    // Phase 4: Non-existent model rejection
    let bad_model_err = runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "non-existent-phantom-model".to_owned(),
            profile: TuningProfile::conservative_default(),
        })
        .unwrap_err();
    assert!(matches!(bad_model_err, ManagerError::ModelNotFound(_)));

    // Phase 5: Path traversal model ID rejection
    let traversal_err = runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "../../../etc/passwd".to_owned(),
            profile: TuningProfile::conservative_default(),
        })
        .unwrap_err();
    assert!(matches!(traversal_err, ManagerError::WorkerSpawn(_)));
}
