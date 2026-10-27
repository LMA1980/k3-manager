//! Empirical Adversarial Challenge Test Suite for Milestone 2 (US-07: Safe Conservative Defaults).
//!
//! Objectives:
//! 1. Verify unprofiled models default to X100 compute pool only, throughput priority, n_ctx=2048 (or 64 in tests),
//!    a100_cores=0, lend_idle_cores=false, borrow_idle_cores=false, within 0x003F, disjoint from active control cores (0x00C0).
//! 2. Stress-test upfront rejection of unprofiled models attempting A100 allocation.
//! 3. Stress-test upfront rejection of unprofiled models attempting dynamic idle core lending or borrowing.
//! 4. Verify OpenAI `/v1/models` serialization conforms to schema with accurate US-07 fields and round-trip deserialization.
//! 5. Verify operator tuning profile transition clears `unprofiled: true` and advisory notice, updating configuration.
//! 6. Stress-test audit event logging completeness, causality, and thread-safety under 32 concurrent threads.

#![allow(
    clippy::assert_is_empty,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::doc_markdown
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use k3_core::K3Error;
use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::ipc::shm::SharedMemoryArena;
use k3_core::profile::{
    CONSERVATIVE_DEFAULT_N_CTX, CONSERVATIVE_TUNING_ADVISORY, CONSTRAINED_TEST_N_CTX, ComputePool,
    ModelListResponse, PriorityClass, ProfileStatus, TuningProfile,
};
use k3_manager::capacity::MockOsAffinityBackend;
use k3_manager::error::ManagerError;
use k3_manager::runtime::{
    AdminCommand, AdminResponse, ControlPlaneRuntime, ModelAuditEventType, ModelPlacementRecord,
};
use k3_manager::supervisor::{WorkerPlacement, WorkerSupervisor};

fn build_test_runtime() -> ControlPlaneRuntime {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity).expect("runtime creation")
}

// ============================================================================
// 1. Conservative Defaults Registration and Invariant Verification
// ============================================================================

#[test]
fn test_empirical_unprofiled_defaults_registration_and_invariants() {
    let runtime = build_test_runtime();
    runtime.clear_audit_events();

    // Register an unprofiled model
    let reg_res = runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "test-conservative-defaults".to_owned(),
            weights_path: PathBuf::from("/models/test.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    assert_eq!(
        reg_res,
        AdminResponse::ModelRegistered("test-conservative-defaults".to_owned())
    );

    // Query status via AdminCommand::GetModelStatus
    let status_resp = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: "test-conservative-defaults".to_owned(),
        })
        .expect("get model status");

    let entry = match status_resp {
        AdminResponse::ModelStatus(entry) => entry,
        other => panic!("expected ModelStatus response, got {other:?}"),
    };

    // Assert all conservative default invariants (ADR-028, ADR-031, ADR-014, ADR-030)
    assert_eq!(entry.id, "test-conservative-defaults");
    assert_eq!(entry.object, "model");
    assert_eq!(entry.owned_by, "k3");
    assert!(entry.unprofiled, "unprofiled must be true");
    assert_eq!(entry.profile_status, "unprofiled");
    assert_eq!(entry.priority_class, "throughput");
    assert_eq!(entry.cluster, "x100");
    assert_eq!(entry.a100_cores, 0, "a100_cores must be 0 for unprofiled");
    assert_eq!(
        entry.n_ctx, CONSERVATIVE_DEFAULT_N_CTX,
        "n_ctx must default to 2048"
    );
    assert!(!entry.lend_idle_cores, "lend_idle_cores must be disabled");
    assert!(
        !entry.borrow_idle_cores,
        "borrow_idle_cores must be disabled"
    );
    assert_eq!(
        entry.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    // Verify audit event trail
    let events = runtime.audit_events();
    assert_eq!(events.len(), 2); // 1 Register + 1 GetModelStatus
    assert_eq!(events[0].event_type, ModelAuditEventType::ModelRegistered);
    assert_eq!(events[0].model_id, "test-conservative-defaults");
    assert!(events[0].unprofiled);
    assert_eq!(events[0].advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));

    assert_eq!(
        events[1].event_type,
        ModelAuditEventType::ModelStatusQueried
    );
    assert_eq!(events[1].model_id, "test-conservative-defaults");
    assert!(events[1].unprofiled);
}

// ============================================================================
// 2. Upfront Rejection of Unprofiled Models on A100 (ADR-028, ADR-031)
// ============================================================================

#[test]
fn test_empirical_unprofiled_worker_placement_a100_rejection_fuzzer() {
    let dummy_weights = PathBuf::from("/tmp/weights.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket.sock");
    let arena = SharedMemoryArena::create("a100_rejection_arena", 4096).expect("create arena");
    let mut supervisor = WorkerSupervisor::new();

    let dup_arena = |a: &SharedMemoryArena| {
        a.duplicate_fd()
            .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
            .expect("dup arena")
    };

    // Vector 1: placement.cluster = "a100" without explicit tuning (is_unprofiled = true)
    let p1 =
        WorkerPlacement::new("m_a100_1", &dummy_weights, &dummy_socket, "a100").with_threads(8);
    let err1 = supervisor
        .spawn_worker(p1, dup_arena(&arena), 1, 4096, 64, 4)
        .unwrap_err();
    assert!(
        matches!(err1, ManagerError::Affinity(K3Error::AffinityError { .. })),
        "expected AffinityError, got {err1:?}"
    );
    assert!(format!("{err1}").contains("unprofiled defaults require X100 compute pool only"));

    // Vector 2: Case insensitive cluster: "A100"
    let p2 =
        WorkerPlacement::new("m_a100_2", &dummy_weights, &dummy_socket, "A100").with_threads(4);
    let err2 = supervisor
        .spawn_worker(p2, dup_arena(&arena), 2, 4096, 64, 4)
        .unwrap_err();
    assert!(
        matches!(err2, ManagerError::Affinity(K3Error::AffinityError { .. })),
        "expected AffinityError, got {err2:?}"
    );

    // Vector 3: WorkerPlacement::unprofiled constructor modified to target A100
    let mut p3 = WorkerPlacement::unprofiled("m_a100_3", &dummy_weights, &dummy_socket);
    p3.cluster = "a100".to_owned();
    let err3 = supervisor
        .spawn_worker(p3, dup_arena(&arena), 3, 4096, 64, 4)
        .unwrap_err();
    assert!(
        matches!(err3, ManagerError::Affinity(K3Error::AffinityError { .. })),
        "expected AffinityError, got {err3:?}"
    );

    // Vector 4: placement.cluster = "x100" but tuning profile has a100_cores > 0 while unprofiled
    let mut unprofiled_with_a100 = TuningProfile::conservative_default();
    unprofiled_with_a100.a100_cores = 2; // Malicious or inconsistent
    let p4 = WorkerPlacement::new("m_a100_4", &dummy_weights, &dummy_socket, "x100")
        .with_threads(4)
        .with_tuning(unprofiled_with_a100);
    let err4 = supervisor
        .spawn_worker(p4, dup_arena(&arena), 4, 4096, 64, 4)
        .unwrap_err();
    assert!(
        matches!(err4, ManagerError::Affinity(K3Error::AffinityError { .. })),
        "expected AffinityError, got {err4:?}"
    );
    assert!(format!("{err4}").contains("a100_cores == 0"));

    // Vector 5: placement.cluster = "x100" but affinity requests A100 cores (0xFF00)
    let p5 = WorkerPlacement::unprofiled("m_a100_5", &dummy_weights, &dummy_socket)
        .with_affinity("0xFF00");
    let err5 = supervisor
        .spawn_worker(p5, dup_arena(&arena), 5, 4096, 64, 4)
        .unwrap_err();
    assert!(
        matches!(err5, ManagerError::Affinity(K3Error::AffinityError { .. })),
        "expected AffinityError on affinity mismatch, got {err5:?}"
    );
}

// ============================================================================
// 3. Upfront Rejection of Core Lending / Borrowing on Unprofiled Models (ADR-030)
// ============================================================================

#[test]
fn test_empirical_unprofiled_core_lending_borrowing_rejection() {
    let dummy_weights = PathBuf::from("/tmp/weights.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket.sock");
    let arena = SharedMemoryArena::create("lending_rejection_arena", 4096).expect("create arena");
    let mut supervisor = WorkerSupervisor::new();

    let dup_arena = |a: &SharedMemoryArena| {
        a.duplicate_fd()
            .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
            .expect("dup arena")
    };

    // Case 1: lend_idle_cores = true
    let mut tuning_lend = TuningProfile::conservative_default();
    tuning_lend.lend_idle_cores = true;
    let p1 = WorkerPlacement::new("m_lend_1", &dummy_weights, &dummy_socket, "x100")
        .with_threads(4)
        .with_tuning(tuning_lend);
    let err1 = supervisor
        .spawn_worker(p1, dup_arena(&arena), 1, 4096, 64, 4)
        .unwrap_err();
    assert!(
        matches!(err1, ManagerError::WorkerSpawn(_)),
        "expected WorkerSpawn error, got {err1:?}"
    );
    assert!(format!("{err1}").contains("cannot enable dynamic idle core lending or borrowing"));

    // Case 2: borrow_idle_cores = true
    let mut tuning_borrow = TuningProfile::conservative_default();
    tuning_borrow.borrow_idle_cores = true;
    let p2 = WorkerPlacement::new("m_lend_2", &dummy_weights, &dummy_socket, "x100")
        .with_threads(4)
        .with_tuning(tuning_borrow);
    let err2 = supervisor
        .spawn_worker(p2, dup_arena(&arena), 2, 4096, 64, 4)
        .unwrap_err();
    assert!(
        matches!(err2, ManagerError::WorkerSpawn(_)),
        "expected WorkerSpawn error, got {err2:?}"
    );
    assert!(format!("{err2}").contains("cannot enable dynamic idle core lending or borrowing"));

    // Case 3: Both lend and borrow = true
    let mut tuning_both = TuningProfile::conservative_default();
    tuning_both.lend_idle_cores = true;
    tuning_both.borrow_idle_cores = true;
    let p3 = WorkerPlacement::new("m_lend_3", &dummy_weights, &dummy_socket, "x100")
        .with_threads(4)
        .with_tuning(tuning_both);
    let err3 = supervisor
        .spawn_worker(p3, dup_arena(&arena), 3, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err3, ManagerError::WorkerSpawn(_)));
}

// ============================================================================
// 4. X100 Compute Bounds and Active Control Isolation (ADR-007)
// ============================================================================

#[test]
fn test_empirical_unprofiled_x100_compute_cap_and_control_isolation() {
    let dummy_weights = PathBuf::from("/tmp/weights.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket.sock");
    let arena = SharedMemoryArena::create("isolation_arena", 4096).expect("create arena");
    let mut supervisor = WorkerSupervisor::new();

    let dup_arena = |a: &SharedMemoryArena| {
        a.duplicate_fd()
            .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
            .expect("dup arena")
    };

    // Default unprofiled constructor has threads = Some(6)
    let p_default = WorkerPlacement::unprofiled("m_iso_default", &dummy_weights, &dummy_socket);
    assert_eq!(p_default.threads, Some(6));

    // Requesting > 6 threads on X100 (e.g. 7 threads) is rejected
    let p_7 = WorkerPlacement::unprofiled("m_iso_7", &dummy_weights, &dummy_socket).with_threads(7);
    let err_7 = supervisor
        .spawn_worker(p_7, dup_arena(&arena), 1, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_7, ManagerError::Affinity(_)));
    assert!(format!("{err_7}").contains("exceeding X100 max compute limit 6"));

    // Requesting threads = 0 is rejected
    let p_0 = WorkerPlacement::unprofiled("m_iso_0", &dummy_weights, &dummy_socket).with_threads(0);
    let err_0 = supervisor
        .spawn_worker(p_0, dup_arena(&arena), 2, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_0, ManagerError::Affinity(_)));
    assert!(format!("{err_0}").contains("requested 0 threads"));

    // Requesting affinity overlapping active control cores 6 or 7 (0x00C0) is rejected
    let p_c6 = WorkerPlacement::unprofiled("m_iso_c6", &dummy_weights, &dummy_socket)
        .with_affinity("0x0040");
    let err_c6 = supervisor
        .spawn_worker(p_c6, dup_arena(&arena), 3, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_c6, ManagerError::Affinity(_)));
    assert!(format!("{err_c6}").contains("encroaches on dedicated active control cores"));

    let p_c7 = WorkerPlacement::unprofiled("m_iso_c7", &dummy_weights, &dummy_socket)
        .with_affinity("0x0080");
    let err_c7 = supervisor
        .spawn_worker(p_c7, dup_arena(&arena), 4, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_c7, ManagerError::Affinity(_)));

    let p_both = WorkerPlacement::unprofiled("m_iso_both", &dummy_weights, &dummy_socket)
        .with_affinity("0x00C0");
    let err_both = supervisor
        .spawn_worker(p_both, dup_arena(&arena), 5, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_both, ManagerError::Affinity(_)));

    // Requesting spare recovery core 5 (0x0020) is within x100_compute_cap_mask (0x003F) and disjoint from 0x00C0
    assert_eq!(
        0x003F & 0x00C0,
        0,
        "compute cap must be disjoint from control mask"
    );
    assert_ne!(0x003F & 0x0020, 0, "core 5 is in compute cap");
}

// ============================================================================
// 5. OpenAI `/v1/models` Schema Conformance & Round-Trip Deserialization
// ============================================================================

#[test]
fn test_empirical_openai_v1_models_serialization_and_roundtrip() {
    let runtime = build_test_runtime();

    // Register 3 models: 2 unprofiled, 1 to be tuned
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "qwen2.5-coder-7b".to_owned(),
            weights_path: PathBuf::from("/models/qwen.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register qwen");

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "deepseek-r1-distill".to_owned(),
            weights_path: PathBuf::from("/models/deepseek.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register deepseek");

    let v1_resp = runtime.v1_models_response();
    assert_eq!(v1_resp.object, "list");
    assert_eq!(v1_resp.data.len(), 2);
    // Models must be sorted alphabetically by id
    assert_eq!(v1_resp.data[0].id, "deepseek-r1-distill");
    assert_eq!(v1_resp.data[1].id, "qwen2.5-coder-7b");

    // Check JSON serialization
    let json = runtime.v1_models_json().expect("serialize v1 models json");
    assert!(json.contains("\"object\": \"list\""));
    assert!(json.contains("\"object\": \"model\""));
    assert!(json.contains("\"owned_by\": \"k3\""));
    assert!(json.contains("\"unprofiled\": true"));
    assert!(json.contains("\"profile_status\": \"unprofiled\""));
    assert!(json.contains("\"priority_class\": \"throughput\""));
    assert!(json.contains("\"a100_cores\": 0"));
    assert!(json.contains("\"n_ctx\": 2048"));
    assert!(json.contains("\"lend_idle_cores\": false"));
    assert!(json.contains("\"borrow_idle_cores\": false"));
    assert!(json.contains(CONSERVATIVE_TUNING_ADVISORY));

    // Full round-trip deserialization
    let deserialized: ModelListResponse =
        serde_json::from_str(&json).expect("deserialization back to ModelListResponse");
    assert_eq!(deserialized, v1_resp);
    assert_eq!(deserialized.data.len(), 2);
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
// 6. Operator Tuning Profile Transition Clears Unprofiled State (ADR-031, ADR-032)
// ============================================================================

#[test]
fn test_empirical_operator_tuning_transition_adversarial_profile() {
    let runtime = build_test_runtime();

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "llama3-8b".to_owned(),
            weights_path: PathBuf::from("/models/llama3.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register llama3-8b");

    // Initially unprofiled
    let initial_entry = runtime.get_model_entry("llama3-8b").expect("model found");
    assert!(initial_entry.unprofiled);
    assert_eq!(initial_entry.profile_status, "unprofiled");
    assert!(initial_entry.advisory_notice.is_some());

    // Adversarial profile: intentionally passes unprofiled: true and fake advisory to test if manager sanitizes
    let mut adversarial_profile = TuningProfile::conservative_default();
    adversarial_profile.unprofiled = true; // Attempt to stay unprofiled
    adversarial_profile.profile_status = ProfileStatus::Unprofiled;
    adversarial_profile.advisory_notice = Some("malicious fake advisory".to_owned());
    adversarial_profile.apply_operator_tuning(
        ComputePool::A100,
        0,
        8,
        PriorityClass::Latency,
        4096,
        true,
        false,
    );

    // Apply operator tuning profile
    let res = runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "llama3-8b".to_owned(),
            profile: adversarial_profile,
        })
        .expect("apply tuning profile");
    assert_eq!(
        res,
        AdminResponse::TuningProfileApplied("llama3-8b".to_owned())
    );

    // Verify manager sanitization and transition invariants
    let tuned_entry = runtime.get_model_entry("llama3-8b").expect("model found");
    assert!(
        !tuned_entry.unprofiled,
        "manager MUST force unprofiled to false on operator tuning"
    );
    assert_eq!(
        tuned_entry.profile_status, "profiled",
        "profile_status MUST become profiled"
    );
    assert_eq!(
        tuned_entry.advisory_notice, None,
        "advisory_notice MUST be cleared"
    );
    assert_eq!(tuned_entry.cluster, "a100");
    assert_eq!(tuned_entry.a100_cores, 8);
    assert_eq!(tuned_entry.n_ctx, 4096);
    assert_eq!(tuned_entry.priority_class, "latency");
    assert!(tuned_entry.lend_idle_cores);
    assert!(!tuned_entry.borrow_idle_cores);

    // Check /v1/models serialization reflects the tuned state
    let json = runtime.v1_models_json().expect("serialize json");
    assert!(json.contains("\"unprofiled\": false"));
    assert!(json.contains("\"profile_status\": \"profiled\""));
    assert!(!json.contains("malicious fake advisory"));

    // Attempting to apply tuning to a non-existent model returns ModelNotFound
    let not_found_err = runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "non-existent-model".to_owned(),
            profile: TuningProfile::conservative_default(),
        })
        .unwrap_err();
    assert!(matches!(not_found_err, ManagerError::ModelNotFound(_)));
}

// ============================================================================
// 7. Audit Event Trail Ordering, Completeness, and Causality
// ============================================================================

#[test]
fn test_empirical_audit_event_trail_ordering_and_completeness() {
    let runtime = build_test_runtime();
    runtime.clear_audit_events();

    // Step 1: Register
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "audit-test-m".to_owned(),
            weights_path: PathBuf::from("/models/audit.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register");

    // Step 2: Get status
    let _ = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: "audit-test-m".to_owned(),
        })
        .expect("get status");

    // Step 3: Apply tuning
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
            model_id: "audit-test-m".to_owned(),
            profile: tuned,
        })
        .expect("apply tuning");

    // Step 4: Get status post-tuning
    let _ = runtime
        .execute_admin(AdminCommand::GetModelStatus {
            model_id: "audit-test-m".to_owned(),
        })
        .expect("get status 2");

    // Step 5: Record model placement
    runtime.record_model_placement("audit-test-m".to_owned(), "x100".to_owned(), Some(9999));

    // Audit query via AdminCommand
    let audit_resp = runtime
        .execute_admin(AdminCommand::GetAuditEvents)
        .expect("get audit events");

    let events = match audit_resp {
        AdminResponse::AuditEvents(evts) => evts,
        other => panic!("expected AuditEvents, got {other:?}"),
    };

    assert_eq!(events.len(), 5);

    // Sequence & unprofiled progression:
    // Event 0: Register -> unprofiled = true
    assert_eq!(events[0].event_type, ModelAuditEventType::ModelRegistered);
    assert!(events[0].unprofiled);
    assert_eq!(events[0].profile_status, "unprofiled");
    assert_eq!(events[0].advisory(), Some(CONSERVATIVE_TUNING_ADVISORY));

    // Event 1: Status query -> unprofiled = true
    assert_eq!(
        events[1].event_type,
        ModelAuditEventType::ModelStatusQueried
    );
    assert!(events[1].unprofiled);

    // Event 2: Tuning applied -> unprofiled = false
    assert_eq!(
        events[2].event_type,
        ModelAuditEventType::TuningProfileApplied
    );
    assert!(!events[2].unprofiled);
    assert_eq!(events[2].profile_status, "profiled");
    assert_eq!(events[2].advisory(), None);

    // Event 3: Status query post-tuning -> unprofiled = false
    assert_eq!(
        events[3].event_type,
        ModelAuditEventType::ModelStatusQueried
    );
    assert!(!events[3].unprofiled);
    assert_eq!(events[3].profile_status, "profiled");
    assert_eq!(events[3].advisory(), None);

    // Event 4: ModelLoaded
    assert_eq!(events[4].event_type, ModelAuditEventType::ModelLoaded);

    // Timestamps monotonically non-decreasing
    for w in events.windows(2) {
        assert!(w[0].timestamp_ms <= w[1].timestamp_ms);
    }
}

// ============================================================================
// 8. High-Concurrency Stress Test: 32 Threads, 3200 Concurrent Operations
// ============================================================================

#[test]
fn test_empirical_concurrency_stress_audit_and_registry_32_threads() {
    let runtime = Arc::new(build_test_runtime());
    runtime.clear_audit_events();

    let num_threads = 32;
    let ops_per_thread = 100;
    let failed = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::with_capacity(num_threads);

    for t_idx in 0..num_threads {
        let r = Arc::clone(&runtime);
        let f = Arc::clone(&failed);

        handles.push(thread::spawn(move || {
            for op_idx in 0..ops_per_thread {
                let model_id = format!("model-t{t_idx}-o{op_idx}");
                // 1. Register model
                if let Err(e) = r.execute_admin(AdminCommand::RegisterModel {
                    model_id: model_id.clone(),
                    weights_path: PathBuf::from(format!("/models/{model_id}.gguf")),
                    cluster: "x100".to_owned(),
                }) {
                    eprintln!("failed to register {model_id}: {e}");
                    f.store(true, Ordering::SeqCst);
                    return;
                }

                // 2. Query model status
                if r.get_model_entry(&model_id).is_none() {
                    eprintln!("failed to get entry for {model_id}");
                    f.store(true, Ordering::SeqCst);
                    return;
                }

                // 3. Serialize v1 models json
                if let Err(e) = r.v1_models_json() {
                    eprintln!("v1_models_json failed: {e}");
                    f.store(true, Ordering::SeqCst);
                    return;
                }

                // 4. Apply tuning profile for every 5th model
                if op_idx % 5 == 0 {
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
                    if let Err(e) = r.execute_admin(AdminCommand::ApplyTuningProfile {
                        model_id: model_id.clone(),
                        profile: tuning,
                    }) {
                        eprintln!("failed to apply tuning for {model_id}: {e}");
                        f.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }

    assert!(
        !failed.load(Ordering::SeqCst),
        "concurrency stress test reported failures"
    );

    // Verify all models registered
    let placements = runtime.active_model_placements();
    assert_eq!(placements.len(), num_threads * ops_per_thread);

    // Verify audit events recorded without corruption
    let audit_events = runtime.audit_events();
    assert!(
        audit_events.len() >= num_threads * ops_per_thread * 2,
        "expected at least 6400 audit events, got {}",
        audit_events.len()
    );

    // Verify /v1/models response valid JSON
    let json = runtime.v1_models_json().expect("valid json");
    let resp: ModelListResponse = serde_json::from_str(&json).expect("valid parse");
    assert_eq!(resp.data.len(), num_threads * ops_per_thread);
}

// ============================================================================
// 9. Constrained Test Harness Context Window Bound (n_ctx = 64)
// ============================================================================

#[test]
fn test_empirical_constrained_test_harness_n_ctx_64() {
    let profile_64 = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_TEST_N_CTX);
    assert!(profile_64.unprofiled);
    assert_eq!(profile_64.n_ctx, 64);
    assert_eq!(profile_64.cluster, "x100");
    assert_eq!(profile_64.a100_cores, 0);
    assert_eq!(profile_64.priority_class, PriorityClass::Throughput);
    assert!(!profile_64.lend_idle_cores);
    assert!(!profile_64.borrow_idle_cores);

    // Placement record with 64 n_ctx
    let record = ModelPlacementRecord::with_tuning(
        "model-64",
        Some(PathBuf::from("/models/m64.gguf")),
        Some(100),
        profile_64,
    );
    let entry = record.to_model_entry();
    assert_eq!(entry.n_ctx, 64);
    assert!(entry.unprofiled);
    assert_eq!(
        entry.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );
}

// ============================================================================
// 10. Profiled Model with Operator Tuning Permitted on A100
// ============================================================================

#[test]
fn test_empirical_supervisor_accepts_profiled_model_on_a100() {
    let dummy_weights = PathBuf::from("/tmp/weights.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket.sock");

    // Construct an explicitly profiled tuning profile targeting A100
    let mut profiled_tuning = TuningProfile::conservative_default();
    profiled_tuning.apply_operator_tuning(
        ComputePool::A100,
        0,
        8,
        PriorityClass::Latency,
        4096,
        true,
        false,
    );

    assert!(!profiled_tuning.unprofiled);
    assert_eq!(profiled_tuning.cluster, "a100");
    assert_eq!(profiled_tuning.a100_cores, 8);
    assert!(profiled_tuning.lend_idle_cores);

    let placement = WorkerPlacement::new("profiled-a100", &dummy_weights, &dummy_socket, "a100")
        .with_threads(8)
        .with_tuning(profiled_tuning);

    let arena = SharedMemoryArena::create("profiled_a100_arena", 4096).expect("create arena");
    let dup_arena = arena
        .duplicate_fd()
        .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
        .expect("dup arena");

    let mut supervisor = WorkerSupervisor::new();

    // Spawning fails at process spawn / file access rather than upfront unprofiled affinity validation
    let err = supervisor
        .spawn_worker(placement, dup_arena, 1, 4096, 64, 4)
        .unwrap_err();

    // Crucial check: it did NOT fail with "unprofiled defaults require X100 compute pool only"
    let err_str = format!("{err}");
    assert!(
        !err_str.contains("unprofiled defaults require X100 compute pool only"),
        "profiled model should not be rejected by unprofiled guard"
    );
    assert!(
        !err_str.contains("cannot enable dynamic idle core lending or borrowing"),
        "profiled model should not be rejected by unprofiled lending guard"
    );
}
