//! Empirical Adversarial Challenge Test Suite for Milestone 3 (US-07 Scenario 1).
//!
//! Objectives:
//! 1. Stress-test exhaustive bitmask and core affinity boundaries (cores 0-15 single/multi/disjoint combinations).
//! 2. Stress-test thread count boundary conditions (0, 1-6, 7, 8+, oversubscription vs mask count).
//! 3. Stress-test cluster placement rejection (A100, Both, SoC, malformed strings).
//! 4. Stress-test dynamic core lending/borrowing rejection under all permutations.
//! 5. Stress-test context window boundaries and prompt token admission (n_ctx=64 and n_ctx=2048).
//! 6. Verify MockOsAffinityBackend invariants across multi-worker concurrent execution.
//! 7. Verify concurrent registry isolation and unprofiled state persistence under load.

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
use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::ipc::shm::SharedMemoryArena;
use k3_core::profile::{CONSERVATIVE_TUNING_ADVISORY, ComputePool, PriorityClass, TuningProfile};
use k3_kv::server::{KvServer, ServerConfig};
use k3_manager::OsAffinityBackend;
use k3_manager::capacity::MockOsAffinityBackend;
use k3_manager::client::KvClient;
use k3_manager::domain::CgroupV2DomainController;
use k3_manager::error::ManagerError;
use k3_manager::prompt::{SimpleTokenizer, validate_and_tokenize};
use k3_manager::runtime::{AdminCommand, ControlPlaneRuntime};
use k3_manager::supervisor::{WorkerPlacement, WorkerSupervisor};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(50_000);

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
    f.write_all(b"GGUF_TEST_SYNTHETIC_CHALLENGE_US07_SCENARIO1")
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
// Challenge 1: Exhaustive Single-Core and Multi-Core Affinity Bitmask Stress
// ============================================================================

#[test]
fn test_challenge_exhaustive_core_affinity_boundaries_0_to_15() {
    let dummy_weights = PathBuf::from("/tmp/weights_ch1.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_ch1.sock");
    let arena = create_test_arena("ch1_arena", 4096);
    let mut supervisor = WorkerSupervisor::new();

    // Iterate through every physical core on the 16-core SoC (cores 0..15)
    for core in 0..16 {
        let mask: u64 = 1 << core;
        let hex_str = format!("0x{mask:04X}");
        let p =
            WorkerPlacement::unprofiled(format!("m_core_{core}"), &dummy_weights, &dummy_socket)
                .with_threads(1)
                .with_affinity(&hex_str);

        let res =
            supervisor.spawn_worker(p, duplicate_arena(&arena, 4096), core as u64, 4096, 64, 4);

        if core <= 5 {
            // Cores 0-5 are within X100 compute cap (0x003F) -> Must pass admission!
            // (Spawn fails on binary/file lookup, NOT on affinity)
            let err = res.unwrap_err();
            assert!(
                !matches!(err, ManagerError::Affinity(_)),
                "Core {core} (mask {hex_str}) is within X100 compute cap (0x003F) and must pass affinity admission, got {err:?}"
            );
        } else if core == 6 || core == 7 {
            // Cores 6-7 are dedicated active control cores (0x00C0) -> Must be rejected upfront!
            let err = res.unwrap_err();
            assert!(
                matches!(err, ManagerError::Affinity(K3Error::AffinityError { .. })),
                "Core {core} (mask {hex_str}) encroaches on active control and must be rejected with AffinityError, got {err:?}"
            );
            assert!(
                format!("{err}").contains("encroaches on dedicated active control cores"),
                "Core {core} error message must cite active control encroachment: {err}"
            );
        } else {
            // Cores 8-15 belong to Cluster 1 (A100, 0xFF00) -> Must be rejected upfront!
            let err = res.unwrap_err();
            assert!(
                matches!(err, ManagerError::Affinity(K3Error::AffinityError { .. })),
                "Core {core} (mask {hex_str}) is outside X100 and must be rejected with AffinityError, got {err:?}"
            );
            assert!(
                format!("{err}").contains("outside X100 mask 0x00FF"),
                "Core {core} error message must cite outside X100 mask: {err}"
            );
        }
    }
}

#[test]
fn test_challenge_disjoint_and_adversarial_affinity_masks() {
    let dummy_weights = PathBuf::from("/tmp/weights_ch_masks.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_ch_masks.sock");
    let arena = create_test_arena("ch_masks_arena", 4096);
    let mut supervisor = WorkerSupervisor::new();

    // 1. Valid disjoint combinations within 0x003F
    let valid_masks = [
        ("0,2,4", 3),
        ("1,3,5", 3),
        ("0-1, 4-5", 4),
        ("0x002A", 3), // cores 1, 3, 5
        ("0x0015", 3), // cores 0, 2, 4
        ("0x003F", 6), // full compute cap
    ];
    for (idx, (mask_str, thread_cnt)) in valid_masks.iter().enumerate() {
        let p =
            WorkerPlacement::unprofiled(format!("m_valid_{idx}"), &dummy_weights, &dummy_socket)
                .with_threads(*thread_cnt)
                .with_affinity(*mask_str);

        let err = supervisor
            .spawn_worker(
                p,
                duplicate_arena(&arena, 4096),
                100 + idx as u64,
                4096,
                64,
                4,
            )
            .unwrap_err();
        assert!(
            !matches!(err, ManagerError::Affinity(_)),
            "Valid mask {mask_str} must be admitted, got {err:?}"
        );
    }

    // 2. Invalid masks overlapping control or A100 or exceeding limits
    let invalid_masks = [
        ("0x0040", "encroaches on dedicated active control cores"), // core 6
        ("0x0080", "encroaches on dedicated active control cores"), // core 7
        ("0x00C0", "encroaches on dedicated active control cores"), // cores 6-7
        ("0-6", "encroaches on dedicated active control cores"),    // includes 6
        ("0x007F", "encroaches on dedicated active control cores"), // cores 0-6
        ("0x00FF", "encroaches on dedicated active control cores"), // cores 0-7
        ("0x0100", "outside X100 mask 0x00FF"),                     // core 8 (A100)
        ("0xFF00", "outside X100 mask 0x00FF"),                     // cores 8-15
        ("8-15", "outside X100 mask 0x00FF"),                       // A100 cpuset
        ("0xFFFF", "encroaches on dedicated active control cores"), // whole SoC
    ];
    for (idx, (mask_str, expected_err)) in invalid_masks.iter().enumerate() {
        let p =
            WorkerPlacement::unprofiled(format!("m_invalid_{idx}"), &dummy_weights, &dummy_socket)
                .with_threads(1)
                .with_affinity(*mask_str);

        let err = supervisor
            .spawn_worker(
                p,
                duplicate_arena(&arena, 4096),
                200 + idx as u64,
                4096,
                64,
                4,
            )
            .unwrap_err();
        assert!(
            matches!(err, ManagerError::Affinity(_)),
            "Invalid mask {mask_str} must be rejected with AffinityError, got {err:?}"
        );
        assert!(
            format!("{err}").contains(expected_err),
            "Expected '{expected_err}' for mask {mask_str}, got '{err}'"
        );
    }

    // 3. Zero and malformed mask rejection
    let malformed_masks = [
        "0", "0x0000", "0X0", "-1", "5-1",  // inverted
        "0-16", // exceeds 16 cores
        "16", "0xGG", "", "   ", ",,",
    ];
    for (idx, malformed) in malformed_masks.iter().enumerate() {
        let p = WorkerPlacement::unprofiled(
            format!("m_malformed_{idx}"),
            &dummy_weights,
            &dummy_socket,
        )
        .with_threads(1)
        .with_affinity(*malformed);

        let res = supervisor.spawn_worker(
            p,
            duplicate_arena(&arena, 4096),
            300 + idx as u64,
            4096,
            64,
            4,
        );
        assert!(
            res.is_err(),
            "Malformed mask '{malformed}' must be rejected upfront"
        );
    }
}

// ============================================================================
// Challenge 2: Thread Bounds & Oversubscription Stress
// ============================================================================

#[test]
fn test_challenge_thread_bounds_and_oversubscription() {
    let dummy_weights = PathBuf::from("/tmp/weights_ch_threads.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_ch_threads.sock");
    let arena = create_test_arena("ch_threads_arena", 4096);
    let mut supervisor = WorkerSupervisor::new();

    // 1. Thread count over compute cap (6)
    for over_threads in [7, 8, 9, 16, 64, 1024, u32::MAX] {
        let p = WorkerPlacement::unprofiled(
            format!("m_over_{over_threads}"),
            &dummy_weights,
            &dummy_socket,
        )
        .with_threads(over_threads);

        let err = supervisor
            .spawn_worker(
                p,
                duplicate_arena(&arena, 4096),
                over_threads as u64,
                4096,
                64,
                4,
            )
            .unwrap_err();
        assert!(
            matches!(err, ManagerError::Affinity(_)),
            "Thread count {over_threads} must be rejected with AffinityError, got {err:?}"
        );
        assert!(
            format!("{err}").contains("exceeding X100 max compute limit 6"),
            "Error for {over_threads} must cite limit 6: {err}"
        );
    }

    // 2. Thread count exceeding mask available cores
    // e.g., mask 0x0003 has 2 cores, requesting 3 threads
    let p_over_mask = WorkerPlacement::unprofiled("m_over_mask", &dummy_weights, &dummy_socket)
        .with_affinity("0x0003")
        .with_threads(3);

    let err_over_mask = supervisor
        .spawn_worker(p_over_mask, duplicate_arena(&arena, 4096), 1, 4096, 64, 4)
        .unwrap_err();
    assert!(matches!(err_over_mask, ManagerError::Affinity(_)));
    assert!(
        format!("{err_over_mask}").contains("exceeding 2 available cores in affinity mask 0x0003")
    );

    // 3. Thread count matching mask available cores
    let p_match_mask = WorkerPlacement::unprofiled("m_match_mask", &dummy_weights, &dummy_socket)
        .with_affinity("0x0003")
        .with_threads(2);

    let err_match = supervisor
        .spawn_worker(p_match_mask, duplicate_arena(&arena, 4096), 2, 4096, 64, 4)
        .unwrap_err();
    assert!(!matches!(err_match, ManagerError::Affinity(_)));
}

// ============================================================================
// Challenge 3: Cluster Rejection & Dynamic Core Lending Lockdown Stress
// ============================================================================

#[test]
fn test_challenge_cluster_rejection_and_lending_lockdown() {
    let dummy_weights = PathBuf::from("/tmp/weights_ch_cluster.gguf");
    let dummy_socket = PathBuf::from("/tmp/socket_ch_cluster.sock");
    let arena = create_test_arena("ch_cluster_arena", 4096);
    let mut supervisor = WorkerSupervisor::new();

    // 1. Rejected cluster targets for unprofiled models
    for bad_cluster in ["a100", "A100", "both", "Both", "soc", "foo", ""] {
        let p = WorkerPlacement::new(
            format!("m_cluster_{bad_cluster}"),
            &dummy_weights,
            &dummy_socket,
            bad_cluster,
        );
        let res = supervisor.spawn_worker(p, duplicate_arena(&arena, 4096), 1, 4096, 64, 4);
        assert!(
            res.is_err(),
            "Cluster target '{bad_cluster}' must be rejected for unprofiled model"
        );
    }

    // 2. Rejection of dynamic idle core lending / borrowing permutations
    let lending_permutations = [(true, false), (false, true), (true, true)];
    for (idx, (lend, borrow)) in lending_permutations.iter().enumerate() {
        let mut tuning = TuningProfile::conservative_default();
        tuning.lend_idle_cores = *lend;
        tuning.borrow_idle_cores = *borrow;

        let p = WorkerPlacement::new(
            format!("m_lend_{idx}"),
            &dummy_weights,
            &dummy_socket,
            "x100",
        )
        .with_threads(4)
        .with_tuning(tuning);

        let err = supervisor
            .spawn_worker(
                p,
                duplicate_arena(&arena, 4096),
                10 + idx as u64,
                4096,
                64,
                4,
            )
            .unwrap_err();
        assert!(
            matches!(err, ManagerError::WorkerSpawn(_)),
            "Lending permutation ({lend}, {borrow}) must be rejected with WorkerSpawn, got {err:?}"
        );
        assert!(
            format!("{err}").contains("cannot enable dynamic idle core lending or borrowing"),
            "Error message must cite dynamic idle lending/borrowing: {err}"
        );
    }

    // 3. Rejection of a100_cores > 0 for unprofiled model
    for bad_a100_cores in [1, 2, 4, 8, 16] {
        let mut tuning = TuningProfile::conservative_default();
        tuning.a100_cores = bad_a100_cores;

        let p = WorkerPlacement::new(
            format!("m_a100_cores_{bad_a100_cores}"),
            &dummy_weights,
            &dummy_socket,
            "x100",
        )
        .with_threads(4)
        .with_tuning(tuning);

        let err = supervisor
            .spawn_worker(
                p,
                duplicate_arena(&arena, 4096),
                20 + bad_a100_cores as u64,
                4096,
                64,
                4,
            )
            .unwrap_err();
        assert!(
            matches!(err, ManagerError::Affinity(_)),
            "a100_cores={bad_a100_cores} must be rejected with AffinityError, got {err:?}"
        );
        assert!(
            format!("{err}").contains("unprofiled defaults require a100_cores == 0"),
            "Error must cite a100_cores == 0: {err}"
        );
    }
}

// ============================================================================
// Challenge 4: Context Window Bounds & Boundary Token Admission Stress
// ============================================================================

#[test]
fn test_challenge_context_window_boundary_admission_stress() {
    let mut tokenizer = SimpleTokenizer::new();
    for i in 0..3000 {
        tokenizer.add_token(format!("w{i}"), i);
    }

    // Test exact boundary at n_ctx = 64
    let p_63: String = (0..63).map(|i| format!("w{i} ")).collect();
    let res_63 = validate_and_tokenize(&tokenizer, &p_63, None, false, 64);
    assert_eq!(res_63.expect("63 tokens admitted").len(), 63);

    let p_64: String = (0..64).map(|i| format!("w{i} ")).collect();
    let err_64 = validate_and_tokenize(&tokenizer, &p_64, None, false, 64).unwrap_err();
    match err_64 {
        K3Error::ContextLengthExceeded {
            requested_tokens,
            max_context_tokens,
        } => {
            assert_eq!(requested_tokens, 64);
            assert_eq!(max_context_tokens, 64);
        }
        other => panic!("expected ContextLengthExceeded, got {other:?}"),
    }

    let p_65: String = (0..65).map(|i| format!("w{i} ")).collect();
    let err_65 = validate_and_tokenize(&tokenizer, &p_65, None, false, 64).unwrap_err();
    match err_65 {
        K3Error::ContextLengthExceeded {
            requested_tokens,
            max_context_tokens,
        } => {
            assert_eq!(requested_tokens, 65);
            assert_eq!(max_context_tokens, 64);
        }
        other => panic!("expected ContextLengthExceeded, got {other:?}"),
    }

    // Test exact boundary at n_ctx = 2048
    let p_2047: String = (0..2047).map(|i| format!("w{i} ")).collect();
    let res_2047 = validate_and_tokenize(&tokenizer, &p_2047, None, false, 2048);
    assert_eq!(res_2047.expect("2047 tokens admitted").len(), 2047);

    let p_2048: String = (0..2048).map(|i| format!("w{i} ")).collect();
    let err_2048 = validate_and_tokenize(&tokenizer, &p_2048, None, false, 2048).unwrap_err();
    match err_2048 {
        K3Error::ContextLengthExceeded {
            requested_tokens,
            max_context_tokens,
        } => {
            assert_eq!(requested_tokens, 2048);
            assert_eq!(max_context_tokens, 2048);
        }
        other => panic!("expected ContextLengthExceeded, got {other:?}"),
    }

    // Test extreme oversizing (2500 tokens against 2048)
    let p_2500: String = (0..2500).map(|i| format!("w{i} ")).collect();
    let err_2500 = validate_and_tokenize(&tokenizer, &p_2500, None, false, 2048).unwrap_err();
    match err_2500 {
        K3Error::ContextLengthExceeded {
            requested_tokens,
            max_context_tokens,
        } => {
            assert_eq!(requested_tokens, 2500);
            assert_eq!(max_context_tokens, 2048);
        }
        other => panic!("expected ContextLengthExceeded, got {other:?}"),
    }
}

// ============================================================================
// Challenge 5: Multi-Worker Concurrent Spawning and Mock OS Affinity Verification
// ============================================================================

#[test]
fn test_challenge_multi_worker_mock_os_affinity_invariants() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // 1. Start background k3-kv server
    let kv_sock_path = unique_socket_path("k3_kv_ch5");
    let kv_config = ServerConfig {
        socket_path: kv_sock_path.clone(),
        max_memory_bytes: 1024 * 1024 * 1024,
    };
    let kv_server = KvServer::new(kv_config);
    let _kv_task = rt.spawn(async move {
        let _ = kv_server.run().await;
    });

    // 2. Connect client
    let mut kv_client = None;
    for _ in 0..50 {
        if let Ok(client) = KvClient::connect(&kv_sock_path) {
            kv_client = Some(client);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut kv_client = kv_client.expect("connect to k3-kv server");

    // 3. Supervisor with mock affinity backend
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let domain_ctrl = Arc::new(
        CgroupV2DomainController::default()
            .with_affinity_backend(Arc::clone(&mock_affinity) as Arc<dyn OsAffinityBackend>),
    );
    let mut supervisor = WorkerSupervisor::new()
        .with_capacity(capacity)
        .with_domain_controller(domain_ctrl);

    // 4. Spawn 3 distinct unprofiled workers
    let n_workers = 3;
    let mut spawned_pids = Vec::new();

    for i in 0..n_workers {
        let model_id = format!("ch5-unprofiled-worker-{i}");
        let (alloc_resp, arena) = kv_client
            .allocate_and_attach(&model_id, 65_536, 64, 4)
            .expect("allocate arena");

        let weights_path = create_dummy_weights(&format!("weights_ch5_{i}"));
        let socket_path = unique_socket_path(&format!("worker_ch5_{i}"));
        let placement = WorkerPlacement::unprofiled(&model_id, &weights_path, &socket_path);

        let worker = supervisor
            .spawn_worker(
                placement,
                arena,
                alloc_resp.arena_id,
                alloc_resp.arena_bytes as usize,
                64,
                4,
            )
            .expect("spawn worker");

        assert!(worker.is_alive(), "worker {model_id} must be alive");
        let pid = worker.pid().expect("worker PID");
        spawned_pids.push(pid);

        // Verify heartbeat IPC
        let hb = worker.send_heartbeat();
        assert!(hb.is_ok(), "worker {model_id} heartbeat ok");
    }

    // 5. Invariant validation on recorded OS affinity calls
    let affinity_calls = mock_affinity.calls();
    assert!(
        affinity_calls.len() >= n_workers,
        "Every spawned worker must have an affinity call recorded"
    );

    for &(pid, mask) in &affinity_calls {
        if spawned_pids.contains(&pid) {
            // Invariant 1: Mask must be non-zero and within X100 compute cap (0x003F)
            assert_ne!(
                mask & 0x003F,
                0,
                "Worker PID {pid} mask 0x{mask:04X} must lie within 0x003F"
            );
            assert_eq!(
                mask & !0x003F,
                0,
                "Worker PID {pid} mask 0x{mask:04X} had bits outside compute cap 0x003F!"
            );

            // Invariant 2: Mask must NEVER overlap active control cores (0x00C0)
            assert_eq!(
                mask & 0x00C0,
                0,
                "Worker PID {pid} encroached onto active control cores 0x00C0!"
            );

            // Invariant 3: Mask must NEVER overlap A100 cores (0xFF00)
            assert_eq!(
                mask & 0xFF00,
                0,
                "Worker PID {pid} encroached onto A100 cores 0xFF00!"
            );
        }
    }

    // 6. Clean teardown
    supervisor.shutdown_all();
}

// ============================================================================
// Challenge 6: Concurrent Registry Operations and State Isolation
// ============================================================================

#[test]
fn test_challenge_concurrent_registry_operations_and_state_isolation() {
    let (runtime, _) = build_test_runtime();
    let runtime = Arc::new(runtime);

    let n_threads = 8;
    let iterations_per_thread = 20;
    let mut handles = Vec::new();

    for t_id in 0..n_threads {
        let rt = Arc::clone(&runtime);
        handles.push(std::thread::spawn(move || {
            for i in 0..iterations_per_thread {
                let model_id = format!("model-conc-t{t_id}-i{i}");
                let reg_res = rt.execute_admin(AdminCommand::RegisterModel {
                    model_id: model_id.clone(),
                    weights_path: PathBuf::from(format!("/models/{model_id}.gguf")),
                    cluster: "x100".to_owned(),
                });
                assert!(reg_res.is_ok());

                // Read status
                let st = rt.get_model_entry(&model_id).expect("entry must exist");
                assert!(st.unprofiled);
                assert_eq!(st.profile_status, "unprofiled");
                assert_eq!(st.cluster, "x100");
                assert_eq!(st.a100_cores, 0);
                assert_eq!(st.n_ctx, 2048);
                assert!(!st.lend_idle_cores);
                assert!(!st.borrow_idle_cores);
                assert_eq!(
                    st.advisory_notice.as_deref(),
                    Some(CONSERVATIVE_TUNING_ADVISORY)
                );

                // For even iterations, apply operator tuning
                if i % 2 == 0 {
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
                    let tune_res = rt.execute_admin(AdminCommand::ApplyTuningProfile {
                        model_id: model_id.clone(),
                        profile: tuned,
                    });
                    assert!(tune_res.is_ok());

                    let post = rt.get_model_entry(&model_id).expect("entry post-tuning");
                    assert!(!post.unprofiled);
                    assert_eq!(post.profile_status, "profiled");
                    assert_eq!(post.advisory_notice, None);
                    assert_eq!(post.cluster, "both");
                    assert_eq!(post.a100_cores, 8);
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }

    // Verify aggregate `/v1/models` representation consistency
    let resp = runtime.v1_models_response();
    let total_registered = n_threads * iterations_per_thread;
    assert_eq!(resp.data.len(), total_registered);

    // Verify deterministic sorting
    for w in resp.data.windows(2) {
        assert!(w[0].id <= w[1].id, "models must be sorted alphabetically");
    }

    // Verify tuned vs unprofiled partitions
    let unprofiled_count = resp.data.iter().filter(|e| e.unprofiled).count();
    let profiled_count = resp.data.iter().filter(|e| !e.unprofiled).count();
    assert_eq!(unprofiled_count, total_registered / 2);
    assert_eq!(profiled_count, total_registered / 2);
}
