//! Empirical Adversarial Challenge Test Suite for US-07 Milestone 2 (`k3-manager`).
//!
//! Stress-tests:
//! 1. `/v1/models` JSON serialization: empty, single, 50-model scale, field completeness, round-trip fidelity, deterministic sorting.
//! 2. Operator tuning profile application: profile status transitions, clearing advisory notices, JSON field omission.
//! 3. Model unregistration edge cases: nonexistent model, double unregister, queries after unregister, re-registration.
//! 4. Model ID security validation: path traversal, whitespace, control chars, cgroup names across all admin commands.
//! 5. Audit event lifecycle: ordering, event type fidelity, advisory text propagation, timestamp monotonicity, clearing.
//! 6. Audit event high-volume concurrency: 16 threads pushing 1,600+ events concurrently without data races or corruption.
//! 7. Concurrent admin plane under saturation: concurrent admin queries while background control tasks run on active control threads.
//! 8. `WorkerSupervisor` unprofiled placement invariants: A100 rejection, core lending/borrowing rejection, custom n_ctx propagation.

#![allow(
    clippy::pedantic,
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::ipc::shm::SharedMemoryArena;
use k3_core::profile::{
    CONSERVATIVE_DEFAULT_N_CTX, CONSERVATIVE_TUNING_ADVISORY, CONSTRAINED_TEST_N_CTX, ComputePool,
    ModelListResponse, PriorityClass, TuningProfile,
};
use k3_manager::domain::MockOsAffinityBackend;
use k3_manager::error::ManagerError;
use k3_manager::runtime::{AdminCommand, AdminResponse, ControlPlaneRuntime, ModelAuditEventType};
use k3_manager::supervisor::{WorkerPlacement, WorkerSupervisor};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

fn build_challenge_test_runtime() -> ControlPlaneRuntime {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity).expect("runtime creation")
}

// ============================================================================
// 1. /v1/models JSON Serialization & Schema Conformance Stress
// ============================================================================

#[test]
fn test_empirical_v1_models_empty_and_single_and_large_json_serialization() {
    let runtime = build_challenge_test_runtime();

    // 1. Empty registry serialization
    let empty_resp = runtime.v1_models_response();
    assert_eq!(empty_resp.object, "list");
    assert!(empty_resp.data.is_empty());

    let empty_json = runtime.v1_models_json().expect("serialize empty json");
    let deserialized_empty: ModelListResponse =
        serde_json::from_str(&empty_json).expect("deserialize empty");
    assert_eq!(deserialized_empty, empty_resp);

    // 2. Single unprofiled model verification
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "test-unprofiled-7b".to_owned(),
            weights_path: PathBuf::from("/models/test-7b.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register single model");

    let single_resp = runtime.v1_models_response();
    assert_eq!(single_resp.data.len(), 1);
    let entry = &single_resp.data[0];
    assert_eq!(entry.id, "test-unprofiled-7b");
    assert_eq!(entry.object, "model");
    assert_eq!(entry.owned_by, "k3");
    assert!(entry.unprofiled);
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

    let single_json = runtime.v1_models_json().expect("serialize single json");
    assert!(single_json.contains("\"unprofiled\": true"));
    assert!(single_json.contains("\"profile_status\": \"unprofiled\""));
    assert!(single_json.contains("\"advisory_notice\""));
    assert!(single_json.contains(CONSERVATIVE_TUNING_ADVISORY));

    let deserialized_single: ModelListResponse =
        serde_json::from_str(&single_json).expect("deserialize single");
    assert_eq!(deserialized_single, single_resp);

    // 3. Multi-model scale & alphabetical sorting verification
    // Register 50 models in non-alphabetical reverse order
    for i in (0..50).rev() {
        let model_id = format!("model-tier-{i:03}");
        runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id,
                weights_path: PathBuf::from(format!("/models/m{i}.gguf")),
                cluster: "x100".to_owned(),
            })
            .expect("register scale model");
    }

    let scale_resp = runtime.v1_models_response();
    assert_eq!(scale_resp.data.len(), 51); // 1 single + 50 scale

    // Assert strictly sorted alphabetically
    for window in scale_resp.data.windows(2) {
        assert!(
            window[0].id < window[1].id,
            "models not sorted: {} >= {}",
            window[0].id,
            window[1].id
        );
    }

    let scale_json = runtime.v1_models_json().expect("serialize scale json");
    let deserialized_scale: ModelListResponse =
        serde_json::from_str(&scale_json).expect("deserialize scale");
    assert_eq!(deserialized_scale, scale_resp);
}

// ============================================================================
// 2. Operator Tuning Profile Transitions & JSON Field Omission
// ============================================================================

#[test]
fn test_empirical_v1_models_operator_tuned_serialization() {
    let runtime = build_challenge_test_runtime();

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen-tuned-32b".to_owned(),
            weights_path: PathBuf::from("/models/qwen.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Initially unprofiled: advisory_notice must be present in JSON
    let init_json = runtime.v1_models_json().expect("serialize initial");
    assert!(init_json.contains("\"advisory_notice\":"));
    assert!(init_json.contains("\"unprofiled\": true"));

    // Apply operator tuning profile (ADR-031, ADR-032)
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

    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "qwen-tuned-32b".to_owned(),
            profile: tuned_profile,
        })
        .expect("apply operator tuning");

    // After tuning: unprofiled is false, advisory_notice is None and omitted from JSON
    let entry = runtime
        .get_model_entry("qwen-tuned-32b")
        .expect("model entry found");
    assert!(!entry.unprofiled);
    assert_eq!(entry.profile_status, "profiled");
    assert_eq!(entry.priority_class, "latency");
    assert_eq!(entry.cluster, "a100");
    assert_eq!(entry.a100_cores, 8);
    assert_eq!(entry.n_ctx, 4096);
    assert!(entry.lend_idle_cores);
    assert!(!entry.borrow_idle_cores);
    assert_eq!(entry.advisory_notice, None);

    let tuned_json = runtime.v1_models_json().expect("serialize tuned json");
    assert!(tuned_json.contains("\"unprofiled\": false"));
    assert!(tuned_json.contains("\"profile_status\": \"profiled\""));
    assert!(tuned_json.contains("\"priority_class\": \"latency\""));
    assert!(tuned_json.contains("\"cluster\": \"a100\""));
    assert!(tuned_json.contains("\"a100_cores\": 8"));
    assert!(tuned_json.contains("\"n_ctx\": 4096"));
    assert!(tuned_json.contains("\"lend_idle_cores\": true"));
    assert!(tuned_json.contains("\"borrow_idle_cores\": false"));

    // Advisory notice must be omitted (skip_serializing_if = "Option::is_none")
    assert!(
        !tuned_json.contains("advisory_notice"),
        "advisory_notice should be omitted when None in tuned profile: {tuned_json}"
    );

    let deserialized: ModelListResponse =
        serde_json::from_str(&tuned_json).expect("roundtrip deserialization");
    assert_eq!(deserialized.data.len(), 1);
    assert_eq!(deserialized.data[0].advisory_notice, None);
    assert_eq!(deserialized.data[0], entry);
}

// ============================================================================
// 3. Model Unregistration Lifecycle, Double Unregister & Queries
// ============================================================================

#[test]
fn test_empirical_model_unregistration_lifecycle_and_edge_cases() {
    let runtime = build_challenge_test_runtime();

    // 1. Unregistering nonexistent model returns Ok (graceful idempotent removal)
    let res = runtime
        .execute_admin(AdminCommand::UnregisterModel {
            model_id: "ghost-model".to_owned(),
        })
        .expect("unregister nonexistent succeeds idempotently");
    assert_eq!(
        res,
        AdminResponse::ModelUnregistered("ghost-model".to_owned())
    );

    // 2. Register, verify, then unregister
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "ephemeral-model".to_owned(),
            weights_path: PathBuf::from("/models/ephemeral.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    assert!(runtime.get_model_entry("ephemeral-model").is_some());
    assert_eq!(runtime.active_model_placements(), vec!["ephemeral-model"]);

    let unreg_res = runtime
        .execute_admin(AdminCommand::UnregisterModel {
            model_id: "ephemeral-model".to_owned(),
        })
        .expect("unregister registered model");
    assert_eq!(
        unreg_res,
        AdminResponse::ModelUnregistered("ephemeral-model".to_owned())
    );

    // 3. Subsequent status queries must return ModelNotFound / None
    let status_err = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: "ephemeral-model".to_owned(),
        })
        .unwrap_err();
    assert!(matches!(status_err, ManagerError::ModelNotFound(_)));

    assert!(runtime.get_model_entry("ephemeral-model").is_none());
    assert!(runtime.active_model_placements().is_empty());
    assert!(runtime.v1_models_response().data.is_empty());
    assert!(runtime.worker_health_snapshots().is_empty());

    // 4. Applying tuning profile to unregistered model fails
    let apply_err = runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "ephemeral-model".to_owned(),
            profile: TuningProfile::conservative_default(),
        })
        .unwrap_err();
    assert!(matches!(apply_err, ManagerError::ModelNotFound(_)));

    // 5. Double unregister on same model
    let double_unreg = runtime
        .execute_admin(AdminCommand::UnregisterModel {
            model_id: "ephemeral-model".to_owned(),
        })
        .expect("second unregister is idempotent");
    assert_eq!(
        double_unreg,
        AdminResponse::ModelUnregistered("ephemeral-model".to_owned())
    );

    // 6. Re-registration succeeds and reinstates conservative unprofiled state
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "ephemeral-model".to_owned(),
            weights_path: PathBuf::from("/models/ephemeral2.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("re-register succeeds");

    let reinstated = runtime
        .get_model_entry("ephemeral-model")
        .expect("reinstated entry found");
    assert!(reinstated.unprofiled);
    assert_eq!(reinstated.profile_status, "unprofiled");
    assert!(reinstated.advisory_notice.is_some());
}

// ============================================================================
// 4. Model ID Validation & Security Boundary Across Admin Commands
// ============================================================================

#[test]
fn test_empirical_admin_model_id_security_validation() {
    let runtime = build_challenge_test_runtime();

    let invalid_ids = [
        "",
        "   ",
        " leading-space",
        "trailing-space ",
        "path/traversal",
        "back\\slash",
        "../parent",
        ".",
        "..",
        "cgroup.procs",
        "cpuset.cpus",
        "model\0nullbyte",
        "model\nnewline",
    ];

    for bad_id in invalid_ids {
        // RegisterModel
        let reg_err = runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: bad_id.to_owned(),
                weights_path: PathBuf::from("/models/test.gguf"),
                cluster: "x100".to_owned(),
            })
            .unwrap_err();
        assert!(
            matches!(reg_err, ManagerError::WorkerSpawn(_)),
            "RegisterModel should reject '{bad_id}': got {reg_err:?}"
        );

        // UnregisterModel
        let unreg_err = runtime
            .execute_admin(AdminCommand::UnregisterModel {
                model_id: bad_id.to_owned(),
            })
            .unwrap_err();
        assert!(
            matches!(unreg_err, ManagerError::WorkerSpawn(_)),
            "UnregisterModel should reject '{bad_id}': got {unreg_err:?}"
        );

        // GetModelStatus
        let status_err = runtime
            .execute_admin(AdminCommand::GetModelStatus {
                model_id: bad_id.to_owned(),
            })
            .unwrap_err();
        assert!(
            matches!(status_err, ManagerError::WorkerSpawn(_)),
            "GetModelStatus should reject '{bad_id}': got {status_err:?}"
        );

        // ApplyTuningProfile
        let apply_err = runtime
            .execute_admin(AdminCommand::ApplyTuningProfile {
                model_id: bad_id.to_owned(),
                profile: TuningProfile::conservative_default(),
            })
            .unwrap_err();
        assert!(
            matches!(apply_err, ManagerError::WorkerSpawn(_)),
            "ApplyTuningProfile should reject '{bad_id}': got {apply_err:?}"
        );
    }
}

// ============================================================================
// 5. Audit Event Lifecycle, Field Propagation & Monotonicity
// ============================================================================

#[test]
fn test_empirical_audit_events_ordering_and_timestamp_monotonicity() {
    let runtime = build_challenge_test_runtime();
    runtime.clear_audit_events();
    assert!(runtime.audit_events().is_empty());

    // Step 1: Register unprofiled model
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "audit-target".to_owned(),
            weights_path: PathBuf::from("/models/audit.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Step 2: Query model status via AdminCommand
    let _ = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: "audit-target".to_owned(),
        })
        .expect("get status");

    // Step 3: Query model entry via get_model_entry helper
    let _ = runtime.get_model_entry("audit-target").expect("found");

    // Step 4: Apply operator tuning profile
    let mut custom = TuningProfile::conservative_default();
    custom.apply_operator_tuning(
        ComputePool::A100,
        0,
        8,
        PriorityClass::Latency,
        8192,
        false,
        false,
    );
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "audit-target".to_owned(),
            profile: custom,
        })
        .expect("apply tuning");

    // Step 5: Query model status after tuning
    let _ = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: "audit-target".to_owned(),
        })
        .expect("get status post-tuning");

    let events = runtime.audit_events();
    assert_eq!(events.len(), 5);

    // Verify event types and sequence
    assert_eq!(events[0].event_type, ModelAuditEventType::ModelRegistered);
    assert!(events[0].unprofiled);
    assert_eq!(events[0].advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));

    assert_eq!(
        events[1].event_type,
        ModelAuditEventType::ModelStatusQueried
    );
    assert!(events[1].unprofiled);

    assert_eq!(
        events[2].event_type,
        ModelAuditEventType::ModelStatusQueried
    );
    assert!(events[2].unprofiled);

    assert_eq!(
        events[3].event_type,
        ModelAuditEventType::TuningProfileApplied
    );
    assert!(!events[3].unprofiled);
    assert_eq!(events[3].profile_status, "profiled");
    assert_eq!(events[3].cluster, "a100");
    assert_eq!(events[3].priority_class, "latency");
    assert_eq!(events[3].n_ctx, 8192);
    assert_eq!(events[3].advisory(), None);

    assert_eq!(
        events[4].event_type,
        ModelAuditEventType::ModelStatusQueried
    );
    assert!(!events[4].unprofiled);
    assert_eq!(events[4].profile_status, "profiled");
    assert_eq!(events[4].advisory(), None);

    // Verify timestamp monotonicity
    for window in events.windows(2) {
        assert!(
            window[0].timestamp_ms <= window[1].timestamp_ms,
            "timestamps inverted: {} > {}",
            window[0].timestamp_ms,
            window[1].timestamp_ms
        );
    }

    // Verify clear_audit_events
    runtime.clear_audit_events();
    assert!(runtime.audit_events().is_empty());
}

// ============================================================================
// 6. Audit Event High-Volume Concurrency Stress
// ============================================================================

#[test]
fn test_empirical_audit_event_vector_high_volume_concurrency() {
    let runtime = Arc::new(build_challenge_test_runtime());
    runtime.clear_audit_events();

    // Pre-register 10 models
    for i in 0..10 {
        runtime
            .execute_admin(AdminCommand::RegisterModel {
                model_id: format!("model-{i}"),
                weights_path: PathBuf::from(format!("/models/m{i}.gguf")),
                cluster: "x100".to_owned(),
            })
            .expect("pre-register");
    }

    // Clear events from pre-registration so count is exact
    runtime.clear_audit_events();

    const NUM_THREADS: usize = 16;
    const QUERIES_PER_THREAD: usize = 100; // 1,600 queries total

    let mut handles = Vec::with_capacity(NUM_THREADS);
    for t_id in 0..NUM_THREADS {
        let rt = Arc::clone(&runtime);
        handles.push(thread::spawn(move || {
            for q_id in 0..QUERIES_PER_THREAD {
                let m_idx = (t_id + q_id) % 10;
                let model_id = format!("model-{m_idx}");
                // Alternately call GetModelStatus and get_model_entry
                if q_id % 2 == 0 {
                    let _ = rt.execute_admin(AdminCommand::GetModelStatus {
                        model_id: model_id.clone(),
                    });
                } else {
                    let _ = rt.get_model_entry(&model_id);
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }

    let all_events = runtime.audit_events();
    assert_eq!(
        all_events.len(),
        NUM_THREADS * QUERIES_PER_THREAD,
        "exact event count under heavy concurrent writing"
    );

    // Verify all recorded events are valid ModelStatusQueried events
    for event in &all_events {
        assert_eq!(event.event_type, ModelAuditEventType::ModelStatusQueried);
        assert!(event.unprofiled);
        assert_eq!(event.profile_status, "unprofiled");
        assert_eq!(event.advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));
        assert!(event.timestamp_ms > 0);
    }
}

// ============================================================================
// 7. Concurrent Admin Plane Under Heavy Background Workload
// ============================================================================

#[test]
fn test_empirical_concurrent_admin_plane_under_heavy_workload() {
    let runtime = Arc::new(build_challenge_test_runtime());
    let stop_signal = Arc::new(AtomicBool::new(false));

    // Spawn 2 background threads generating continuous control-plane tasks
    let mut bg_handles = Vec::new();
    for i in 0..2 {
        let rt = Arc::clone(&runtime);
        let stop = Arc::clone(&stop_signal);
        bg_handles.push(thread::spawn(move || {
            let mut count = 0_usize;
            while !stop.load(Ordering::Relaxed) {
                // Dispatch prefix insertion and match
                let block: Vec<u32> = (0..16).map(|x| x + (count as u32 % 100)).collect();
                let p_id = format!("bg-proj-{i}");
                let m_id = format!("bg-model-{i}");
                let _ = rt.dispatch_insert_prefix(
                    p_id.clone(),
                    m_id.clone(),
                    block.clone(),
                    vec![count as u32],
                );
                let _ = rt.match_prefix(&p_id, &m_id, &block);
                let _ = rt.increment_queue_depth();
                let _ = rt.decrement_queue_depth();
                count += 1;
                thread::yield_now();
            }
        }));
    }

    // Spawn 8 foreground threads concurrently performing admin commands
    const ADMIN_THREADS: usize = 8;
    const OPERATIONS_PER_THREAD: usize = 50;

    let success_count = Arc::new(AtomicUsize::new(0));
    let mut admin_handles = Vec::with_capacity(ADMIN_THREADS);

    for t_id in 0..ADMIN_THREADS {
        let rt = Arc::clone(&runtime);
        let successes = Arc::clone(&success_count);
        admin_handles.push(thread::spawn(move || {
            for op in 0..OPERATIONS_PER_THREAD {
                let model_id = format!("thread-{t_id}-model-{op}");
                // 1. Register model
                let reg_res = rt.dispatch_execute_admin(AdminCommand::RegisterModel {
                    model_id: model_id.clone(),
                    weights_path: PathBuf::from("/models/m.gguf"),
                    cluster: "x100".to_owned(),
                });
                if reg_res.is_ok() {
                    successes.fetch_add(1, Ordering::Relaxed);
                }

                // 2. Query status
                let status_res = rt.dispatch_execute_admin(AdminCommand::GetModelStatus {
                    model_id: model_id.clone(),
                });
                if status_res.is_ok() {
                    successes.fetch_add(1, Ordering::Relaxed);
                }

                // 3. Serialize v1 models json
                let json_res = rt.v1_models_json();
                if json_res.is_ok() {
                    successes.fetch_add(1, Ordering::Relaxed);
                }

                // 4. List models
                let list_res = rt.dispatch_execute_admin(AdminCommand::ListModels);
                if list_res.is_ok() {
                    successes.fetch_add(1, Ordering::Relaxed);
                }

                // 5. Apply tuning profile
                let mut tuned = TuningProfile::conservative_default();
                tuned.apply_operator_tuning(
                    ComputePool::X100,
                    4,
                    0,
                    PriorityClass::Latency,
                    1024,
                    false,
                    false,
                );
                let apply_res = rt.dispatch_execute_admin(AdminCommand::ApplyTuningProfile {
                    model_id: model_id.clone(),
                    profile: tuned,
                });
                if apply_res.is_ok() {
                    successes.fetch_add(1, Ordering::Relaxed);
                }

                // 6. Unregister model
                let unreg_res =
                    rt.dispatch_execute_admin(AdminCommand::UnregisterModel { model_id });
                if unreg_res.is_ok() {
                    successes.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for h in admin_handles {
        h.join().expect("admin thread finished cleanly");
    }

    // Stop background workload
    stop_signal.store(true, Ordering::Relaxed);
    for h in bg_handles {
        h.join().expect("background thread finished cleanly");
    }

    // All operations should have completed without errors
    let total_expected = ADMIN_THREADS * OPERATIONS_PER_THREAD * 6;
    assert_eq!(
        success_count.load(Ordering::Relaxed),
        total_expected,
        "concurrent admin operations completed successfully without deadlocks or errors"
    );
}

// ============================================================================
// 8. WorkerSupervisor Unprofiled Invariants & Edge Cases
// ============================================================================

#[test]
fn test_empirical_worker_supervisor_unprofiled_admission_invariants() {
    let dummy_weights = PathBuf::from("/tmp/dummy_weights.gguf");
    let dummy_socket = PathBuf::from("/tmp/dummy_socket.sock");
    let arena = SharedMemoryArena::create("emp_unprofiled_test_arena", 4096).expect("create arena");
    let mut supervisor = WorkerSupervisor::new();

    // 1. Unprofiled placement requesting A100 is rejected with ManagerError::Affinity
    let p_a100 =
        WorkerPlacement::new("unprof_a100", &dummy_weights, &dummy_socket, "a100").with_threads(8);

    let dup1 = arena
        .duplicate_fd()
        .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup1", false))
        .expect("dup");
    let err_a100 = supervisor
        .spawn_worker(p_a100, dup1, 1, 4096, CONSTRAINED_TEST_N_CTX as u32, 4)
        .unwrap_err();
    assert!(matches!(err_a100, ManagerError::Affinity(_)));
    assert!(err_a100.to_string().contains("X100 compute pool only"));

    // 2. Unprofiled placement enabling lend_idle_cores is rejected with ManagerError::WorkerSpawn
    let mut tuning_lend = TuningProfile::conservative_default();
    tuning_lend.lend_idle_cores = true;
    let p_lend = WorkerPlacement::unprofiled("unprof_lend", &dummy_weights, &dummy_socket)
        .with_tuning(tuning_lend);

    let dup2 = arena
        .duplicate_fd()
        .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup2", false))
        .expect("dup");
    let err_lend = supervisor
        .spawn_worker(p_lend, dup2, 1, 4096, CONSTRAINED_TEST_N_CTX as u32, 4)
        .unwrap_err();
    assert!(matches!(err_lend, ManagerError::WorkerSpawn(_)));
    assert!(
        err_lend
            .to_string()
            .contains("cannot enable dynamic idle core lending or borrowing")
    );

    // 3. Unprofiled placement enabling borrow_idle_cores is rejected with ManagerError::WorkerSpawn
    let mut tuning_borrow = TuningProfile::conservative_default();
    tuning_borrow.borrow_idle_cores = true;
    let p_borrow = WorkerPlacement::unprofiled("unprof_borrow", &dummy_weights, &dummy_socket)
        .with_tuning(tuning_borrow);

    let dup3 = arena
        .duplicate_fd()
        .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup3", false))
        .expect("dup");
    let err_borrow = supervisor
        .spawn_worker(p_borrow, dup3, 1, 4096, CONSTRAINED_TEST_N_CTX as u32, 4)
        .unwrap_err();
    assert!(matches!(err_borrow, ManagerError::WorkerSpawn(_)));
    assert!(
        err_borrow
            .to_string()
            .contains("cannot enable dynamic idle core lending or borrowing")
    );

    // 4. Unprofiled placement requesting a100_cores > 0 even on x100 is rejected with ManagerError::Affinity
    let mut tuning_a100_cores = TuningProfile::conservative_default();
    tuning_a100_cores.a100_cores = 2;
    let p_a100_cores =
        WorkerPlacement::unprofiled("unprof_a100_cores", &dummy_weights, &dummy_socket)
            .with_tuning(tuning_a100_cores);

    let dup4 = arena
        .duplicate_fd()
        .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup4", false))
        .expect("dup");
    let err_cores = supervisor
        .spawn_worker(
            p_a100_cores,
            dup4,
            1,
            4096,
            CONSTRAINED_TEST_N_CTX as u32,
            4,
        )
        .unwrap_err();
    assert!(matches!(err_cores, ManagerError::Affinity(_)));
    assert!(err_cores.to_string().contains("a100_cores == 0"));

    // 5. Worker requesting >6 threads on X100 is rejected by capacity cap
    let p_excess_threads =
        WorkerPlacement::unprofiled("unprof_excess", &dummy_weights, &dummy_socket).with_threads(7);

    let dup5 = arena
        .duplicate_fd()
        .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup5", false))
        .expect("dup");
    let err_excess = supervisor
        .spawn_worker(
            p_excess_threads,
            dup5,
            1,
            4096,
            CONSTRAINED_TEST_N_CTX as u32,
            4,
        )
        .unwrap_err();
    assert!(matches!(err_excess, ManagerError::Affinity(_)));
}

// ============================================================================
// 9. Randomized Concurrent Registry Stress & Property Oracle
// ============================================================================

#[test]
fn test_empirical_fuzz_randomized_concurrent_registry_oracle() {
    let runtime = Arc::new(build_challenge_test_runtime());

    const THREADS: usize = 8;
    const OPS_PER_THREAD: usize = 40;

    let handles: Vec<_> = (0..THREADS)
        .map(|t_id| {
            let rt = Arc::clone(&runtime);
            thread::spawn(move || {
                for op in 0..OPS_PER_THREAD {
                    let m_idx = (t_id * 17 + op * 31) % 10;
                    let model_id = format!("fuzz-m-{m_idx}");

                    match op % 5 {
                        0 => {
                            // Register
                            let _ = rt.execute_admin(AdminCommand::RegisterModel {
                                model_id: model_id.clone(),
                                weights_path: PathBuf::from(format!("/models/{model_id}.gguf")),
                                cluster: "x100".to_owned(),
                            });
                        }
                        1 => {
                            // Apply tuning
                            let mut tuning = TuningProfile::conservative_default();
                            tuning.apply_operator_tuning(
                                ComputePool::X100,
                                4,
                                0,
                                PriorityClass::Latency,
                                1024,
                                false,
                                false,
                            );
                            let _ = rt.execute_admin(AdminCommand::ApplyTuningProfile {
                                model_id: model_id.clone(),
                                profile: tuning,
                            });
                        }
                        2 => {
                            // Status query
                            let _ = rt.execute_admin(AdminCommand::GetModelStatus {
                                model_id: model_id.clone(),
                            });
                        }
                        3 => {
                            // Unregister
                            let _ = rt.execute_admin(AdminCommand::UnregisterModel {
                                model_id: model_id.clone(),
                            });
                        }
                        _ => {
                            // Invariant 1: Response snapshot JSON round-trip
                            let resp = rt.v1_models_response();
                            let resp_json = serde_json::to_string(&resp).expect("valid json");
                            let roundtrip: ModelListResponse =
                                serde_json::from_str(&resp_json).expect("deserializable");
                            assert_eq!(roundtrip, resp);

                            // Invariant 2: Live JSON snapshot is also valid and sorted
                            let live_json = rt.v1_models_json().expect("valid live json");
                            let live_resp: ModelListResponse =
                                serde_json::from_str(&live_json).expect("live deserializable");
                            for w in live_resp.data.windows(2) {
                                assert!(
                                    w[0].id < w[1].id,
                                    "alphabetical ordering invariant in live json"
                                );
                            }

                            // Invariant 3: Strictly sorted by model id in response snapshot
                            for w in resp.data.windows(2) {
                                assert!(w[0].id < w[1].id, "alphabetical ordering invariant");
                            }

                            // Invariant 4: Profile state consistency
                            for entry in &resp.data {
                                if entry.unprofiled {
                                    assert_eq!(entry.profile_status, "unprofiled");
                                    assert_eq!(entry.priority_class, "throughput");
                                    assert_eq!(entry.cluster, "x100");
                                    assert_eq!(entry.a100_cores, 0);
                                    assert_eq!(
                                        entry.advisory_notice.as_deref(),
                                        Some(CONSERVATIVE_TUNING_ADVISORY)
                                    );
                                } else {
                                    assert_eq!(entry.profile_status, "profiled");
                                    assert_eq!(entry.advisory_notice, None);
                                }
                            }
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread completed cleanly without panic");
    }

    // Final oracle pass
    let final_resp = runtime.v1_models_response();
    let final_json = runtime.v1_models_json().expect("valid json");
    let final_roundtrip: ModelListResponse =
        serde_json::from_str(&final_json).expect("final deserializable");
    assert_eq!(final_roundtrip, final_resp);
}

// ============================================================================
// 10. Cluster Validation & Case-Insensitivity
// ============================================================================

#[test]
fn test_empirical_cluster_validation_and_case_insensitivity() {
    let runtime = build_challenge_test_runtime();

    // Uppercase X100 normalized to x100
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "m-upper-x100".to_owned(),
            weights_path: PathBuf::from("/models/m.gguf"),
            cluster: "X100".to_owned(),
        })
        .expect("X100 accepted");
    let entry_x = runtime.get_model_entry("m-upper-x100").unwrap();
    assert_eq!(entry_x.cluster, "x100");

    // Uppercase A100 normalized to a100
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "m-upper-a100".to_owned(),
            weights_path: PathBuf::from("/models/m.gguf"),
            cluster: "A100".to_owned(),
        })
        .expect("A100 accepted");
    let entry_a = runtime.get_model_entry("m-upper-a100").unwrap();
    assert_eq!(entry_a.cluster, "a100");

    // Unsupported cluster "both" at registration
    let err_both = runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "m-both".to_owned(),
            weights_path: PathBuf::from("/models/m.gguf"),
            cluster: "both".to_owned(),
        })
        .unwrap_err();
    assert!(matches!(err_both, ManagerError::WorkerSpawn(_)));
    assert!(err_both.to_string().contains("unsupported cluster 'both'"));

    // Unsupported arbitrary cluster
    let err_invalid = runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "m-gpu".to_owned(),
            weights_path: PathBuf::from("/models/m.gguf"),
            cluster: "nvidia-gpu".to_owned(),
        })
        .unwrap_err();
    assert!(matches!(err_invalid, ManagerError::WorkerSpawn(_)));
}

// ============================================================================
// 11. Profiled Model Admission on A100
// ============================================================================

#[test]
fn test_empirical_worker_supervisor_profiled_a100_admission() {
    let dummy_weights = PathBuf::from("/tmp/dummy_weights.gguf");
    let dummy_socket = PathBuf::from("/tmp/dummy_socket.sock");
    let arena = SharedMemoryArena::create("profiled_a100_test_arena", 4096).expect("create arena");
    let mut supervisor = WorkerSupervisor::new();

    // Create an operator-tuned profile targeting A100
    let mut tuned = TuningProfile::conservative_default();
    tuned.apply_operator_tuning(
        ComputePool::A100,
        0,
        8,
        PriorityClass::Latency,
        2048,
        true,
        false,
    );
    assert!(!tuned.unprofiled);

    let placement =
        WorkerPlacement::new("profiled_a100_model", &dummy_weights, &dummy_socket, "a100")
            .with_threads(8)
            .with_tuning(tuned);

    let dup = arena
        .duplicate_fd()
        .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup_tuned", false))
        .expect("dup");

    // When spawned, it must pass unprofiled admission (since unprofiled = false)
    // and proceed to process spawn (failing on dummy weights/binary, NOT on AffinityError)
    let err = supervisor
        .spawn_worker(placement, dup, 1, 4096, 64, 4)
        .unwrap_err();

    // Must NOT be rejected by unprofiled affinity guard
    let err_str = err.to_string();
    assert!(
        !err_str.contains("unprofiled defaults require X100 compute pool only"),
        "operator-tuned model should not be rejected by unprofiled guard: {err_str}"
    );
    assert!(
        !err_str.contains("cannot allocate A100 cores"),
        "operator-tuned model should not be rejected by a100_cores guard: {err_str}"
    );
}

// ============================================================================
// 12. Poison Resilience on Audit Events and Model Placements
// ============================================================================

#[test]
fn test_empirical_audit_events_poison_resilience() {
    let runtime = Arc::new(build_challenge_test_runtime());

    // Register initial model
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "poison-test".to_owned(),
            weights_path: PathBuf::from("/models/p.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Intentionally trigger a panic inside a thread that accesses runtime
    let rt_clone = Arc::clone(&runtime);
    let handle = thread::spawn(move || {
        let _entry = rt_clone.get_model_entry("poison-test");
        panic!("simulated panic for poison testing");
    });
    let _ = handle.join(); // thread panics

    // Runtime must still be completely functional (poison resilience via unwrap_or_else)
    let entry = runtime
        .get_model_entry("poison-test")
        .expect("model accessible after thread panic");
    assert_eq!(entry.id, "poison-test");

    let events = runtime.audit_events();
    assert!(!events.is_empty());

    let json = runtime
        .v1_models_json()
        .expect("json serialize after panic");
    assert!(json.contains("poison-test"));
}
