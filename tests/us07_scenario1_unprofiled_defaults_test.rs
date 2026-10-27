//! US-07 Scenario 1 Integration Test Suite: Safe Conservative Defaults for Unprofiled Models.
//!
//! Validates:
//! 1. Suite 1: Placement defaults and unprofiled constructor invariants (cluster="x100", 6 threads,
//!    default conservative profile: x100_cores=6, a100_cores=0, priority_class=Throughput,
//!    n_ctx=2048, lend_idle_cores=false, borrow_idle_cores=false, unprofiled=true,
//!    profile_status=Unprofiled, advisory_notice=CONSERVATIVE_TUNING_ADVISORY).
//! 2. Suite 2: Supervisor admission validation & affinity boundary invariants (cores 0-5 mask 0x003F,
//!    max 6 threads, strictly disjoint from active control cores 0x00C0).
//! 3. Suite 3: Rejection paths for unprofiled invariants (rejects A100 cluster with ManagerError::Affinity,
//!    rejects a100_cores > 0 with ManagerError::Affinity, rejects lend_idle_cores with ManagerError::WorkerSpawn,
//!    rejects borrow_idle_cores with ManagerError::WorkerSpawn, rejects out-of-pool affinity).
//! 4. Suite 4: Context window bounds (2048 default vs 64 test override per AGENTS.md Rule 5) and
//!    prompt admission enforcement.
//! 5. Suite 5: State persistence and operator authority retention (defaults persist across queries;
//!    applying operator tuning clears unprofiled/advisory and unlocks operator-configured cluster/resources).
//! 6. Suite 6: Mock OS affinity and worker spawn under conservative defaults (using MockOsAffinityBackend
//!    to assert affinity mask is bounded by 0x003F and disjoint from 0x00C0 and 0xFF00).

#![allow(
    clippy::pedantic,
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::doc_markdown
)]

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use k3_core::K3Error;
use k3_core::affinity::{
    ControlPlaneCapacityConfig, DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_SPARE_RECOVERY_MASK,
    DEFAULT_X100_COMPUTE_CAP_MASK,
};
use k3_core::ipc::shm::SharedMemoryArena;
use k3_core::profile::{
    CONSERVATIVE_DEFAULT_A100_CORES, CONSERVATIVE_DEFAULT_N_CTX, CONSERVATIVE_DEFAULT_X100_CORES,
    CONSERVATIVE_TUNING_ADVISORY, CONSTRAINED_TEST_N_CTX, ComputePool, ModelProfile, PriorityClass,
    ProfileStatus, TuningProfile,
};
use k3_kv::server::{KvServer, ServerConfig};
use k3_manager::OsAffinityBackend;
use k3_manager::capacity::MockOsAffinityBackend;
use k3_manager::client::KvClient;
use k3_manager::domain::CgroupV2DomainController;
use k3_manager::error::ManagerError;
use k3_manager::prompt::{SimpleTokenizer, validate_and_tokenize};
use k3_manager::runtime::{
    AdminCommand, AdminResponse, ControlPlaneRuntime, ModelAuditEventType, ModelPlacementRecord,
};
use k3_manager::supervisor::{WorkerPlacement, WorkerSupervisor};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(10_000);

fn unique_socket_path(prefix: &str) -> PathBuf {
    let pid = std::process::id();
    let seq = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{pid}_{seq}.sock"))
}

fn create_dummy_weights(name: &str) -> PathBuf {
    let pid = std::process::id();
    let seq = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("{name}_{pid}_{seq}.gguf"));
    let mut f = File::create(&path).expect("create dummy weights file");
    f.write_all(b"GGUF_TEST_SYNTHETIC_MODEL_WEIGHTS_US07_SCENARIO1")
        .expect("write dummy weights");
    path
}

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

fn create_test_arena(name: &str, size_bytes: usize) -> SharedMemoryArena {
    SharedMemoryArena::create(name, size_bytes).expect("create shared memory arena")
}

fn duplicate_arena(arena: &SharedMemoryArena, size_bytes: usize) -> SharedMemoryArena {
    let fd = arena.duplicate_fd().expect("duplicate arena fd");
    SharedMemoryArena::from_existing_fd(fd, size_bytes, "dup_arena", false)
        .expect("attach to dup fd")
}

// ============================================================================
// Suite 1: Placement Defaults and Unprofiled Constructor Invariants
// ============================================================================

#[test]
fn test_unprofiled_worker_placement_constructor_invariants() {
    let dummy_weights = PathBuf::from("/tmp/weights_constructor.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_constructor.sock");
    let model_id = "test-unprofiled-model";

    let placement = WorkerPlacement::unprofiled(model_id, &dummy_weights, &dummy_socket);

    // Verify placement top-level fields
    assert_eq!(placement.model_id, model_id);
    assert_eq!(placement.weights_path, dummy_weights);
    assert_eq!(placement.socket_path, dummy_socket);
    assert_eq!(placement.cluster, "x100");
    assert_eq!(placement.threads, Some(6));
    assert!(placement.affinity.is_none());
    assert!(placement.memlock_limit.is_none());
    assert!(placement.worker_binary.is_none());
    assert!(placement.heartbeat_timeout.is_none());
    assert!(placement.pass_timeout.is_none());

    // Verify attached tuning profile
    let tuning = placement.tuning.expect("tuning profile must be attached");
    assert_eq!(tuning.model_id.as_deref(), Some(model_id));
    assert_eq!(tuning.compute_pool, ComputePool::X100);
    assert_eq!(tuning.cluster, "x100");
    assert_eq!(tuning.x100_cores, CONSERVATIVE_DEFAULT_X100_CORES);
    assert_eq!(tuning.a100_cores, CONSERVATIVE_DEFAULT_A100_CORES);
    assert_eq!(tuning.priority_class, PriorityClass::Throughput);
    assert_eq!(tuning.n_ctx, CONSERVATIVE_DEFAULT_N_CTX);
    assert!(!tuning.lend_idle_cores);
    assert!(!tuning.borrow_idle_cores);
    assert!(tuning.unprofiled);
    assert_eq!(tuning.profile_status, ProfileStatus::Unprofiled);
    assert_eq!(
        tuning.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );
    assert!(tuning.is_unprofiled());
}

#[test]
fn test_bare_worker_placement_automatic_defaults_attachment() {
    let dummy_weights = PathBuf::from("/tmp/weights_bare.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_bare.sock");
    let model_id = "bare-unprofiled-model";

    // Caller constructs bare placement without calling with_tuning or unprofiled
    let bare_placement =
        WorkerPlacement::new(model_id, &dummy_weights, &dummy_socket, "x100").with_threads(4);
    assert!(bare_placement.tuning.is_none());

    // When processed by WorkerSupervisor, defaults are attached with call-site n_ctx
    let arena = create_test_arena("bare_defaults_arena", 4096);
    let mut supervisor = WorkerSupervisor::new();

    // Spawning with bare placement attaches conservative defaults with n_ctx=64
    let err = supervisor
        .spawn_worker(bare_placement, arena, 1, 4096, 64, 4)
        .unwrap_err();

    // Validation fails because worker binary is not found in dummy test path,
    // BUT admission checks passed because cluster was "x100" and threads was 4 (<= 6).
    // Verify that the error is binary lookup or IPC, NOT an affinity or lending error.
    assert!(
        matches!(err, ManagerError::WorkerSpawn(_)),
        "expected worker spawn error, got {err:?}"
    );
    let err_str = format!("{err}");
    assert!(
        !err_str.contains("unprofiled defaults require X100"),
        "bare placement on X100 must be admitted as valid unprofiled default"
    );
}

// ============================================================================
// Suite 2: Supervisor Admission Validation & Affinity Boundary Invariants
// ============================================================================

#[test]
fn test_supervisor_admission_x100_compute_cap_and_control_isolation() {
    let dummy_weights = PathBuf::from("/tmp/weights_cap.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_cap.sock");
    let arena = create_test_arena("cap_arena", 4096);
    let mut supervisor = WorkerSupervisor::new();

    // Hardware mask invariants:
    // DEFAULT_X100_COMPUTE_CAP_MASK = 0x003F (cores 0-5)
    // DEFAULT_ACTIVE_CONTROL_MASK = 0x00C0 (cores 6-7)
    // DEFAULT_SPARE_RECOVERY_MASK = 0x0020 (core 5)
    assert_eq!(DEFAULT_X100_COMPUTE_CAP_MASK, 0x003F);
    assert_eq!(DEFAULT_ACTIVE_CONTROL_MASK, 0x00C0);
    assert_eq!(DEFAULT_SPARE_RECOVERY_MASK, 0x0020);

    // Compute cap and active control cores are strictly disjoint
    assert_eq!(
        DEFAULT_X100_COMPUTE_CAP_MASK & DEFAULT_ACTIVE_CONTROL_MASK,
        0,
        "compute cap mask must never overlap active control mask"
    );

    // Spare recovery core (core 5, 0x0020) is within the compute cap mask
    assert_ne!(
        DEFAULT_X100_COMPUTE_CAP_MASK & DEFAULT_SPARE_RECOVERY_MASK,
        0,
        "spare recovery core 5 is lendable compute capacity within 0x003F"
    );

    // Requesting core 6 (0x0040) is rejected upfront
    let p_c6 =
        WorkerPlacement::unprofiled("m_c6", &dummy_weights, &dummy_socket).with_affinity("0x0040");
    let err_c6 = supervisor
        .spawn_worker(p_c6, duplicate_arena(&arena, 4096), 1, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_c6, ManagerError::Affinity(_)));
    assert!(format!("{err_c6}").contains("encroaches on dedicated active control cores"));

    // Requesting core 7 (0x0080) is rejected upfront
    let p_c7 =
        WorkerPlacement::unprofiled("m_c7", &dummy_weights, &dummy_socket).with_affinity("0x0080");
    let err_c7 = supervisor
        .spawn_worker(p_c7, duplicate_arena(&arena, 4096), 2, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_c7, ManagerError::Affinity(_)));
    assert!(format!("{err_c7}").contains("encroaches on dedicated active control cores"));

    // Requesting cores 6-7 (0x00C0) is rejected upfront
    let p_c67 =
        WorkerPlacement::unprofiled("m_c67", &dummy_weights, &dummy_socket).with_affinity("6-7");
    let err_c67 = supervisor
        .spawn_worker(p_c67, duplicate_arena(&arena, 4096), 3, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_c67, ManagerError::Affinity(_)));
    assert!(format!("{err_c67}").contains("encroaches on dedicated active control cores"));
}

#[test]
fn test_supervisor_admission_thread_bounds_and_zero_threads() {
    let dummy_weights = PathBuf::from("/tmp/weights_threads.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_threads.sock");
    let arena = create_test_arena("threads_arena", 4096);
    let mut supervisor = WorkerSupervisor::new();

    // Requesting 0 threads is rejected upfront
    let p_0 =
        WorkerPlacement::unprofiled("m_threads_0", &dummy_weights, &dummy_socket).with_threads(0);
    let err_0 = supervisor
        .spawn_worker(p_0, duplicate_arena(&arena, 4096), 1, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_0, ManagerError::Affinity(_)));
    assert!(format!("{err_0}").contains("requested 0 threads; must be greater than 0"));

    // Requesting 7 threads on X100 is rejected upfront (> 6 max compute cap)
    let p_7 =
        WorkerPlacement::unprofiled("m_threads_7", &dummy_weights, &dummy_socket).with_threads(7);
    let err_7 = supervisor
        .spawn_worker(p_7, duplicate_arena(&arena, 4096), 2, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_7, ManagerError::Affinity(_)));
    assert!(format!("{err_7}").contains("exceeding X100 max compute limit 6"));

    // Requesting 8 threads on X100 is rejected upfront
    let p_8 =
        WorkerPlacement::unprofiled("m_threads_8", &dummy_weights, &dummy_socket).with_threads(8);
    let err_8 = supervisor
        .spawn_worker(p_8, duplicate_arena(&arena, 4096), 3, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_8, ManagerError::Affinity(_)));
    assert!(format!("{err_8}").contains("exceeding X100 max compute limit 6"));

    // Requesting 6 threads on X100 passes admission
    let p_6 =
        WorkerPlacement::unprofiled("m_threads_6", &dummy_weights, &dummy_socket).with_threads(6);
    let res_6 = supervisor.spawn_worker(p_6, duplicate_arena(&arena, 4096), 4, 4096, 64, 4);
    assert!(!matches!(res_6.unwrap_err(), ManagerError::Affinity(_)));

    // Requesting 1 thread on X100 passes admission
    let p_1 =
        WorkerPlacement::unprofiled("m_threads_1", &dummy_weights, &dummy_socket).with_threads(1);
    let res_1 = supervisor.spawn_worker(p_1, duplicate_arena(&arena, 4096), 5, 4096, 64, 4);
    assert!(!matches!(res_1.unwrap_err(), ManagerError::Affinity(_)));
}

// ============================================================================
// Suite 3: Rejection Paths for Unprofiled Invariants
// ============================================================================

#[test]
fn test_rejection_unprofiled_targeting_a100_cluster() {
    let dummy_weights = PathBuf::from("/tmp/weights_a100.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_a100.sock");
    let arena = create_test_arena("a100_rejection_arena", 4096);
    let mut supervisor = WorkerSupervisor::new();

    // Vector 1: placement.cluster = "a100" without explicit tuning (is_unprofiled = true)
    let p1 = WorkerPlacement::new("m_a100_v1", &dummy_weights, &dummy_socket, "a100");
    let err1 = supervisor
        .spawn_worker(p1, duplicate_arena(&arena, 4096), 1, 4096, 64, 4)
        .unwrap_err();
    assert!(
        matches!(err1, ManagerError::Affinity(K3Error::AffinityError { .. })),
        "expected AffinityError, got {err1:?}"
    );
    assert!(format!("{err1}").contains("unprofiled defaults require X100 compute pool only"));

    // Vector 2: Case-insensitive cluster = "A100"
    let p2 = WorkerPlacement::new("m_a100_v2", &dummy_weights, &dummy_socket, "A100");
    let err2 = supervisor
        .spawn_worker(p2, duplicate_arena(&arena, 4096), 2, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(
        err2,
        ManagerError::Affinity(K3Error::AffinityError { .. })
    ));
    assert!(format!("{err2}").contains("unprofiled defaults require X100 compute pool only"));

    // Vector 3: WorkerPlacement::unprofiled with modified cluster = "a100"
    let mut p3 = WorkerPlacement::unprofiled("m_a100_v3", &dummy_weights, &dummy_socket);
    p3.cluster = "a100".to_owned();
    let err3 = supervisor
        .spawn_worker(p3, duplicate_arena(&arena, 4096), 3, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(
        err3,
        ManagerError::Affinity(K3Error::AffinityError { .. })
    ));
    assert!(format!("{err3}").contains("unprofiled defaults require X100 compute pool only"));

    // Vector 4: WorkerPlacement::new with cluster "a100" and explicit conservative_default()
    let p4 = WorkerPlacement::new("m_a100_v4", &dummy_weights, &dummy_socket, "a100")
        .with_tuning(TuningProfile::conservative_default());
    let err4 = supervisor
        .spawn_worker(p4, duplicate_arena(&arena, 4096), 4, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(
        err4,
        ManagerError::Affinity(K3Error::AffinityError { .. })
    ));
    assert!(format!("{err4}").contains("unprofiled defaults require X100 compute pool only"));
}

#[test]
fn test_rejection_unprofiled_allocating_a100_cores() {
    let dummy_weights = PathBuf::from("/tmp/weights_a100_cores.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_a100_cores.sock");
    let arena = create_test_arena("a100_cores_arena", 4096);
    let mut supervisor = WorkerSupervisor::new();

    // Unprofiled model placed on X100 but tuning profile has a100_cores = 2
    let mut bad_tuning = TuningProfile::conservative_default();
    bad_tuning.a100_cores = 2;

    let p = WorkerPlacement::new("m_a100_cores", &dummy_weights, &dummy_socket, "x100")
        .with_threads(4)
        .with_tuning(bad_tuning);

    let err = supervisor
        .spawn_worker(p, duplicate_arena(&arena, 4096), 1, 4096, 64, 4)
        .unwrap_err();

    assert!(
        matches!(err, ManagerError::Affinity(K3Error::AffinityError { .. })),
        "expected AffinityError, got {err:?}"
    );
    assert!(format!("{err}").contains("unprofiled defaults require a100_cores == 0"));
}

#[test]
fn test_rejection_unprofiled_dynamic_idle_core_lending_and_borrowing() {
    let dummy_weights = PathBuf::from("/tmp/weights_lending.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_lending.sock");
    let arena = create_test_arena("lending_arena", 4096);
    let mut supervisor = WorkerSupervisor::new();

    // Case 1: lend_idle_cores = true
    let mut tuning_lend = TuningProfile::conservative_default();
    tuning_lend.lend_idle_cores = true;
    let p1 = WorkerPlacement::new("m_lend", &dummy_weights, &dummy_socket, "x100")
        .with_threads(4)
        .with_tuning(tuning_lend);
    let err1 = supervisor
        .spawn_worker(p1, duplicate_arena(&arena, 4096), 1, 4096, 64, 4)
        .unwrap_err();
    assert!(
        matches!(err1, ManagerError::WorkerSpawn(_)),
        "expected WorkerSpawn error, got {err1:?}"
    );
    assert!(format!("{err1}").contains("cannot enable dynamic idle core lending or borrowing"));

    // Case 2: borrow_idle_cores = true
    let mut tuning_borrow = TuningProfile::conservative_default();
    tuning_borrow.borrow_idle_cores = true;
    let p2 = WorkerPlacement::new("m_borrow", &dummy_weights, &dummy_socket, "x100")
        .with_threads(4)
        .with_tuning(tuning_borrow);
    let err2 = supervisor
        .spawn_worker(p2, duplicate_arena(&arena, 4096), 2, 4096, 64, 4)
        .unwrap_err();
    assert!(
        matches!(err2, ManagerError::WorkerSpawn(_)),
        "expected WorkerSpawn error, got {err2:?}"
    );
    assert!(format!("{err2}").contains("cannot enable dynamic idle core lending or borrowing"));

    // Case 3: Both lend_idle_cores and borrow_idle_cores = true
    let mut tuning_both = TuningProfile::conservative_default();
    tuning_both.lend_idle_cores = true;
    tuning_both.borrow_idle_cores = true;
    let p3 = WorkerPlacement::new("m_both", &dummy_weights, &dummy_socket, "x100")
        .with_threads(4)
        .with_tuning(tuning_both);
    let err3 = supervisor
        .spawn_worker(p3, duplicate_arena(&arena, 4096), 3, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err3, ManagerError::WorkerSpawn(_)));
    assert!(format!("{err3}").contains("cannot enable dynamic idle core lending or borrowing"));
}

#[test]
fn test_rejection_unprofiled_targeting_a100_affinity_mask() {
    let dummy_weights = PathBuf::from("/tmp/weights_mask.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_mask.sock");
    let arena = create_test_arena("mask_arena", 4096);
    let mut supervisor = WorkerSupervisor::new();

    // Placement targeting X100 cluster but specifying affinity mask in A100 range (0xFF00)
    let p = WorkerPlacement::unprofiled("m_out_of_pool", &dummy_weights, &dummy_socket)
        .with_affinity("0xFF00");

    let err = supervisor
        .spawn_worker(p, duplicate_arena(&arena, 4096), 1, 4096, 64, 4)
        .unwrap_err();

    assert!(
        matches!(err, ManagerError::Affinity(K3Error::AffinityError { .. })),
        "expected AffinityError on mask outside X100, got {err:?}"
    );
    assert!(format!("{err}").contains("specifies cores outside X100 mask 0x00FF"));
}

// ============================================================================
// Suite 4: Context Window Bounds (2048 Default vs 64 Test Override)
// ============================================================================

#[test]
fn test_context_window_bounds_conservative_default_2048() {
    assert_eq!(CONSERVATIVE_DEFAULT_N_CTX, 2048);

    let default_tuning = TuningProfile::conservative_default();
    assert_eq!(default_tuning.n_ctx, 2048);
    assert!(default_tuning.unprofiled);
    assert_eq!(default_tuning.profile_status, ProfileStatus::Unprofiled);

    let model_prof = ModelProfile::unprofiled_default("test-model-2048");
    assert_eq!(model_prof.tuning.n_ctx, 2048);
    assert!(model_prof.is_unprofiled());

    let entry = model_prof.to_model_entry(1_729_000_000);
    assert_eq!(entry.n_ctx, 2048);
    assert!(entry.unprofiled);

    let rec = ModelPlacementRecord::new_unprofiled(
        "rec-model-2048",
        Some(PathBuf::from("/tmp/rec.gguf")),
        "x100",
        None,
    );
    assert_eq!(rec.n_ctx, 2048);
    assert!(rec.unprofiled);
}

#[test]
fn test_context_window_bounds_constrained_test_override_64() {
    assert_eq!(CONSTRAINED_TEST_N_CTX, 64);

    let test_tuning = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_TEST_N_CTX);
    assert_eq!(test_tuning.n_ctx, 64);
    assert!(test_tuning.unprofiled);
    assert_eq!(test_tuning.profile_status, ProfileStatus::Unprofiled);
    assert_eq!(
        test_tuning.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    let model_prof_64 = ModelProfile::unprofiled_with_n_ctx("test-model-64", 64);
    assert_eq!(model_prof_64.tuning.n_ctx, 64);
    assert!(model_prof_64.is_unprofiled());

    let entry_64 = model_prof_64.to_model_entry(1_729_000_000);
    assert_eq!(entry_64.n_ctx, 64);
    assert!(entry_64.unprofiled);
}

#[test]
fn test_context_length_admission_enforcement() {
    let mut tokenizer = SimpleTokenizer::new();
    for i in 0..100 {
        tokenizer.add_token(format!("tok{i}"), i);
    }

    // 1. Constrained test harness bound (n_ctx = 64)
    // 63 tokens admitted (tokens.len() < n_ctx)
    let prompt_63: String = (0..63).map(|i| format!("tok{i} ")).collect();
    let res_admitted = validate_and_tokenize(&tokenizer, &prompt_63, None, false, 64);
    assert!(res_admitted.is_ok());
    assert_eq!(res_admitted.unwrap().len(), 63);

    // 64 tokens rejected (tokens.len() >= n_ctx)
    let prompt_64: String = (0..64).map(|i| format!("tok{i} ")).collect();
    let res_rejected = validate_and_tokenize(&tokenizer, &prompt_64, None, false, 64);
    match res_rejected {
        Err(K3Error::ContextLengthExceeded {
            requested_tokens,
            max_context_tokens,
        }) => {
            assert_eq!(requested_tokens, 64);
            assert_eq!(max_context_tokens, 64);
        }
        other => panic!("expected ContextLengthExceeded, got {other:?}"),
    }

    // 2. Production runtime default bound (n_ctx = 2048)
    // 100 tokens well within 2048 bound
    let prompt_100: String = (0..100).map(|i| format!("tok{i} ")).collect();
    let res_2048 = validate_and_tokenize(&tokenizer, &prompt_100, None, false, 2048);
    assert!(res_2048.is_ok());
    assert_eq!(res_2048.unwrap().len(), 100);

    // Zero context limit rejected
    let res_zero = validate_and_tokenize(&tokenizer, "tok0", None, false, 0);
    assert!(matches!(
        res_zero,
        Err(K3Error::ContextLengthExceeded { .. })
    ));
}

// ============================================================================
// Suite 5: State Persistence and Operator Authority Retention
// ============================================================================

#[test]
fn test_unprofiled_state_persistence_across_multiple_queries() {
    let (runtime, _) = build_test_runtime();
    let model_id = "model-persistent-s1";

    let reg_res = runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from("/models/persistent.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");
    assert_eq!(reg_res, AdminResponse::ModelRegistered(model_id.to_owned()));

    // Query 20 consecutive times via AdminCommand::GetModelStatus
    for _ in 0..20 {
        let status_resp = runtime
            .execute_admin(AdminCommand::GetModelStatus {
                model_id: model_id.to_owned(),
            })
            .expect("get status");
        let entry = match status_resp {
            AdminResponse::ModelStatus(e) => e,
            other => panic!("expected ModelStatus, got {other:?}"),
        };
        assert_eq!(entry.id, model_id);
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

    // Query 10 consecutive times via v1_models_response()
    for _ in 0..10 {
        let resp = runtime.v1_models_response();
        assert_eq!(resp.data.len(), 1);
        let entry = &resp.data[0];
        assert_eq!(entry.id, model_id);
        assert!(entry.unprofiled);
        assert_eq!(entry.profile_status, "unprofiled");
        assert_eq!(
            entry.advisory_notice.as_deref(),
            Some(CONSERVATIVE_TUNING_ADVISORY)
        );
    }
}

#[test]
fn test_operator_tuning_transition_clears_unprofiled_and_advisory() {
    let (runtime, _) = build_test_runtime();
    let model_id = "model-authority-s1";

    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from("/models/auth.gguf"),
            cluster: "x100".to_owned(),
        })
        .expect("register model");

    // Initially unprofiled
    let init_status = runtime.get_model_entry(model_id).expect("found");
    assert!(init_status.unprofiled);
    assert_eq!(init_status.profile_status, "unprofiled");
    assert_eq!(
        init_status.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );

    // Operator applies authoritative tuning profile (ADR-031, ADR-032)
    let mut tuned = TuningProfile::conservative_default();
    tuned.apply_operator_tuning(
        ComputePool::Both,
        4,
        8,
        PriorityClass::Latency,
        4096,
        true,
        false,
    );

    let tune_res = runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: model_id.to_owned(),
            profile: tuned,
        })
        .expect("apply tuning profile");
    assert_eq!(
        tune_res,
        AdminResponse::TuningProfileApplied(model_id.to_owned())
    );

    // Verify post-transition state
    let post_status = runtime
        .get_model_entry(model_id)
        .expect("found post-tuning");
    assert!(!post_status.unprofiled);
    assert_eq!(post_status.profile_status, "profiled");
    assert_eq!(post_status.advisory_notice, None);
    assert_eq!(post_status.cluster, "both");
    assert_eq!(post_status.a100_cores, 8);
    assert_eq!(post_status.priority_class, "latency");
    assert_eq!(post_status.n_ctx, 4096);
    assert!(post_status.lend_idle_cores);
    assert!(!post_status.borrow_idle_cores);

    // Verify audit log captured transition
    let events = runtime.audit_events();
    let tune_evt = events
        .iter()
        .find(|e| e.event_type == ModelAuditEventType::TuningProfileApplied)
        .expect("TuningProfileApplied event recorded");
    assert_eq!(tune_evt.model_id, model_id);
    assert!(!tune_evt.unprofiled);
    assert_eq!(tune_evt.profile_status, "profiled");
    assert_eq!(tune_evt.advisory_notice, None);
}

// ============================================================================
// Suite 6: Mock OS Affinity and Worker Spawn Under Conservative Defaults
// ============================================================================

#[test]
fn test_e2e_unprofiled_worker_spawn_and_affinity_pinning() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // Step 1: Start background k3-kv server
    let kv_sock_path = unique_socket_path("k3_kv_s1");
    let kv_config = ServerConfig {
        socket_path: kv_sock_path.clone(),
        max_memory_bytes: 1024 * 1024 * 1024,
    };
    let kv_server = KvServer::new(kv_config);
    let _kv_task = rt.spawn(async move {
        let _ = kv_server.run().await;
    });

    // Wait for k3-kv listener
    let mut kv_client = None;
    for _ in 0..50 {
        if let Ok(client) = KvClient::connect(&kv_sock_path) {
            kv_client = Some(client);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut kv_client = kv_client.expect("connect to k3-kv server");

    // Step 2: Allocate arena in k3-kv with small bounds (n_ctx = 64)
    let arena_bytes = 65_536_u64;
    let n_ctx = 64_u32;
    let max_blocks = 4_u32;
    let model_id = "unprofiled-e2e-worker";

    let (alloc_resp, arena) = kv_client
        .allocate_and_attach(model_id, arena_bytes, n_ctx, max_blocks)
        .expect("allocate arena from k3-kv");

    // Step 3: Configure WorkerSupervisor with capacity and MockOsAffinityBackend
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    assert_eq!(capacity.active_control_mask(), 0x00C0);
    assert_eq!(capacity.spare_recovery_mask(), 0x0020);
    assert_eq!(capacity.x100_compute_cap_mask(), 0x003F);

    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let domain_ctrl = Arc::new(
        CgroupV2DomainController::default()
            .with_affinity_backend(Arc::clone(&mock_affinity) as Arc<dyn OsAffinityBackend>),
    );
    let mut supervisor = WorkerSupervisor::new()
        .with_capacity(capacity)
        .with_domain_controller(domain_ctrl);

    // Step 4: Construct WorkerPlacement::unprofiled
    let weights_path = create_dummy_weights("weights_e2e_s1");
    let worker_sock = unique_socket_path("worker_e2e_s1");
    let placement = WorkerPlacement::unprofiled(model_id, &weights_path, &worker_sock);

    // Step 5: Spawn isolated worker
    let worker = supervisor
        .spawn_worker(
            placement,
            arena,
            alloc_resp.arena_id,
            alloc_resp.arena_bytes as usize,
            n_ctx,
            max_blocks,
        )
        .expect("spawn unprofiled worker process");

    assert!(worker.is_alive(), "worker process must be running");
    let worker_pid = worker.pid().expect("worker process PID");

    // Step 6: Verify OS affinity records in mock backend
    let affinity_calls = mock_affinity.calls();
    let worker_calls: Vec<_> = affinity_calls
        .iter()
        .filter(|&&(pid, _)| pid == worker_pid)
        .collect();

    assert!(
        !worker_calls.is_empty(),
        "worker PID {worker_pid} must have an OS affinity placement recorded"
    );

    for &&(_, mask) in &worker_calls {
        // Must lie strictly within X100 compute cap (0x003F)
        assert_ne!(
            mask & 0x003F,
            0,
            "compute worker mask 0x{mask:04X} must lie within compute cap 0x003F"
        );
        // Must NEVER encroach onto dedicated active control cores (0x00C0)
        assert_eq!(
            mask & 0x00C0,
            0,
            "compute worker mask 0x{mask:04X} encroached onto active control cores 0x00C0!"
        );
        // Must NEVER encroach onto A100 cores (0xFF00)
        assert_eq!(
            mask & 0xFF00,
            0,
            "unprofiled compute worker mask 0x{mask:04X} encroached onto A100 cores 0xFF00!"
        );
    }

    // Step 7: Send heartbeat IPC to verify worker readiness
    let heartbeat_res = worker.send_heartbeat();
    assert!(
        heartbeat_res.is_ok(),
        "unprofiled worker must respond to heartbeat IPC, got {heartbeat_res:?}"
    );

    // Step 8: Clean shutdown
    supervisor.shutdown_all();
}
