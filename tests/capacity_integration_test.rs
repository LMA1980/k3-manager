//! Integration Test Suite for `k3-manager` capacity resolution and control-plane scheduling.
//!
//! Validates:
//! - Feature 16: Capacity configuration integration and contract re-exports
//! - Feature 17: `ManagerError` structured affinity error mapping and diagnostic propagation
//! - Feature 18: Manager task scheduler round-robin distribution and supervisor spare core watchdog binding
//! - Feature 19: End-to-end integration with `WorkerPlacement` affinity validation and runaway worker containment
//!
//! Specifications: ADR-007, ADR-008, ADR-041, Issue #69, PROJECT.md § Interface Contracts.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::undocumented_unsafe_blocks,
    clippy::too_many_lines,
    clippy::doc_markdown
)]

use std::error::Error;
use std::sync::Arc;
use std::thread;

use k3_core::K3Error;
use k3_core::ipc::shm::SharedMemoryArena;
use k3_manager::error::ManagerError;
use k3_manager::supervisor::{WorkerPlacement, WorkerSupervisor};
use k3_manager::{
    A100_CLUSTER_MASK, ControlPlaneCapacityConfig, ControlPlaneScheduler,
    DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_SPARE_RECOVERY_MASK, DEFAULT_X100_COMPUTE_CAP_MASK,
    ResolvedControlCapacityExt, X100_CLUSTER_MASK,
};

// ============================================================================
// Integration Tests: Manager Capacity & Scheduler Workloads
// ============================================================================

#[test]
fn test_manager_capacity_default_resolution_and_reexports() {
    assert_eq!(X100_CLUSTER_MASK, 0x00FF);
    assert_eq!(A100_CLUSTER_MASK, 0xFF00);

    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");

    // 1. Verify active control mask is cores 6, 7 (0x00C0)
    assert_eq!(res.active_control_mask, DEFAULT_ACTIVE_CONTROL_MASK);
    assert_eq!(res.active_control_cores, vec![6, 7]);

    // 2. Verify spare recovery core is core 5 (0x0020)
    assert_eq!(res.spare_recovery_mask, DEFAULT_SPARE_RECOVERY_MASK);
    assert_eq!(res.spare_recovery_core(), Some(5));

    // 3. Verify compute cap mask is cores 0..5 (0x003F)
    assert_eq!(res.x100_compute_cap_mask, DEFAULT_X100_COMPUTE_CAP_MASK);
    assert_eq!(res.x100_max_compute_cores, 6);
    assert_eq!(res.a100_max_compute_cores, 8);
}

#[test]
fn test_manager_scheduler_round_robin_distribution() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res);

    // Verify sequential tasks cycle across cores 6 and 7
    assert_eq!(scheduler.schedule_control_task(), 6);
    assert_eq!(scheduler.schedule_control_task(), 7);
    assert_eq!(scheduler.schedule_control_task(), 6);
    assert_eq!(scheduler.schedule_control_task(), 7);

    // Verify watchdog recovery core is core 5
    assert_eq!(scheduler.watchdog_recovery_core(), Some(5));
}

#[test]
fn test_manager_error_affinity_mapping_and_diagnostics() {
    let err = K3Error::AffinityError {
        requested_mask: 0x00FF,
        requested_threads: 8,
        reason: "control cores cannot exceed available online cores".to_string(),
    };

    // Verify diagnostic code
    assert_eq!(err.diagnostic_code(), "affinity_error");

    // Map to ManagerError via From
    let mgr_err: ManagerError = err.clone().into();

    // Verify variant discrimination to ManagerError::Affinity
    assert!(matches!(mgr_err, ManagerError::Affinity(_)));

    let display = format!("{mgr_err}");

    // Verify display string contains affinity diagnostic details
    assert!(display.contains("[k3-manager affinity error]"));
    assert!(display.contains("[affinity_error]"));
    assert!(display.contains("mask: 0x00FF"));
    assert!(display.contains("threads: 8"));

    // Verify std::error::Error::source returns inner K3Error
    let source = mgr_err.source().expect("source must be Some");
    assert_eq!(format!("{source}"), format!("{err}"));
}

#[test]
fn test_manager_worker_placement_validation_accepts_valid_compute_mask() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res.clone());

    let weights = std::path::PathBuf::from("/tmp/dummy_weights.gguf");
    let socket = std::path::PathBuf::from("/tmp/dummy_socket.sock");

    // Worker placement within X100 compute cap (cores 0..3: mask 0x000F)
    let mut placement = WorkerPlacement::new("model_a", &weights, &socket, "x100");
    placement.affinity = Some("0x000F".to_string());

    assert!(res.validate_worker_placement(&placement).is_ok());
    assert!(scheduler.validate_worker_placement(&placement).is_ok());

    // Worker placement utilizing lendable core 5 (cores 0..5: mask 0x003F)
    placement.affinity = Some("0x003F".to_string());
    assert!(res.validate_worker_placement(&placement).is_ok());
    assert!(scheduler.validate_worker_placement(&placement).is_ok());

    // Worker placement on A100 cluster (mask 0xFF00)
    let mut placement_a100 = WorkerPlacement::new("model_b", &weights, &socket, "a100");
    placement_a100.affinity = Some("0xFF00".to_string());
    assert!(res.validate_worker_placement(&placement_a100).is_ok());
    assert!(scheduler.validate_worker_placement(&placement_a100).is_ok());
}

#[test]
fn test_manager_worker_placement_validation_rejects_encroaching_on_active_control() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res.clone());

    let weights = std::path::PathBuf::from("/tmp/dummy_weights.gguf");
    let socket = std::path::PathBuf::from("/tmp/dummy_socket.sock");

    // Rogue worker attempting to pin to core 6 (active control core)
    let mut placement = WorkerPlacement::new("rogue_model", &weights, &socket, "x100");
    placement.affinity = Some("0x0040".to_string());

    let res_err = res.validate_worker_placement(&placement);
    assert!(res_err.is_err());
    assert!(scheduler.validate_worker_placement(&placement).is_err());

    let err = res_err.unwrap_err();
    assert!(matches!(err, ManagerError::Affinity(_)));
    let err_msg = format!("{err:?}");
    assert!(err_msg.contains("encroaches on dedicated active control cores"));
    assert!(err_msg.contains("0x00C0"));

    // Rogue worker attempting to consume all 8 X100 cores (mask 0x00FF)
    placement.affinity = Some("0x00FF".to_string());
    assert!(res.validate_worker_placement(&placement).is_err());
    assert!(scheduler.validate_worker_placement(&placement).is_err());
}

#[test]
fn test_manager_worker_placement_validation_rejects_invalid_affinity_string() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res);

    let weights = std::path::PathBuf::from("/tmp/dummy_weights.gguf");
    let socket = std::path::PathBuf::from("/tmp/dummy_socket.sock");

    let mut placement = WorkerPlacement::new("invalid_aff_model", &weights, &socket, "x100");
    placement.affinity = Some("not_a_valid_number".to_string());

    let err = scheduler.validate_worker_placement(&placement).unwrap_err();
    assert!(matches!(err, ManagerError::WorkerSpawn(_)));
}

#[test]
fn test_manager_scheduler_concurrent_dispatch_load() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = Arc::new(ControlPlaneScheduler::new(res));

    // 16 concurrent manager client workers dispatching 1,000 tasks each
    let mut handles = Vec::new();
    for _ in 0..16 {
        let sched = Arc::clone(&scheduler);
        handles.push(thread::spawn(move || {
            let mut c6 = 0_usize;
            let mut c7 = 0_usize;
            for _ in 0..1000 {
                let core = sched.schedule_control_task();
                if core == 6 {
                    c6 += 1;
                } else if core == 7 {
                    c7 += 1;
                } else {
                    panic!("unexpected core: {core}");
                }
            }
            (c6, c7)
        }));
    }

    let mut total_6 = 0;
    let mut total_7 = 0;
    for h in handles {
        let (c6, c7) = h.join().expect("thread join");
        total_6 += c6;
        total_7 += c7;
    }

    assert_eq!(total_6 + total_7, 16_000);
    assert_eq!(total_6, 8_000);
    assert_eq!(total_7, 8_000);
}

#[test]
fn test_manager_constrained_degraded_board_topology() {
    // Degraded board topology with only cores 0..3 online (0x000F)
    let min_cfg = ControlPlaneCapacityConfig::minimal();
    let res = min_cfg
        .resolve_with_online_mask(0x000F)
        .expect("resolve 4-core minimal");
    let scheduler = ControlPlaneScheduler::new(res);

    // Single active control core is core 3
    assert_eq!(scheduler.schedule_control_task(), 3);
    assert_eq!(scheduler.schedule_control_task(), 3);
    // Zero spare cores in minimal mode
    assert_eq!(scheduler.watchdog_recovery_core(), None);
}

#[test]
fn test_manager_scheduler_capacity_accessor() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res);

    assert_eq!(scheduler.capacity().active_control_mask, 0x00C0);
    assert_eq!(scheduler.capacity().spare_recovery_mask, 0x0020);
    assert_eq!(scheduler.capacity().x100_compute_cap_mask, 0x003F);
}

#[test]
fn test_supervisor_spawn_worker_rejects_exceeding_x100_thread_cap() {
    let dummy_weights = std::path::PathBuf::from("/tmp/dummy_weights.gguf");
    let dummy_socket = std::path::PathBuf::from("/tmp/dummy_socket.sock");
    let arena = SharedMemoryArena::create("thread_cap_test_arena", 4096).expect("create arena");
    let mut supervisor = WorkerSupervisor::new();

    // Requesting 7 threads on X100 exceeds the 6-thread cap (cores 0..5, 0x003F)
    let placement = WorkerPlacement::new("model_7threads", &dummy_weights, &dummy_socket, "x100")
        .with_threads(7);

    let dup_arena = arena
        .duplicate_fd()
        .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
        .expect("dup arena");

    let err = supervisor
        .spawn_worker(placement, dup_arena, 1, 4096, 64, 4)
        .unwrap_err();

    assert!(matches!(err, ManagerError::Affinity(_)));
    let err_str = format!("{err}");
    assert!(err_str.contains("exceeding X100 max compute limit 6"));
}

#[test]
fn test_supervisor_spawn_worker_rejects_active_control_encroachment() {
    let dummy_weights = std::path::PathBuf::from("/tmp/dummy_weights.gguf");
    let dummy_socket = std::path::PathBuf::from("/tmp/dummy_socket.sock");
    let arena = SharedMemoryArena::create("encroach_test_arena", 4096).expect("create arena");
    let mut supervisor = WorkerSupervisor::new();

    // Attempting to spawn worker requesting cores 6 or 7 (0x00C0, 0x0040, 0x0080, 0x00FF)
    for bad_aff in &["0x0040", "0x0080", "0x00C0", "0x00FF"] {
        let placement = WorkerPlacement::new("rogue_model", &dummy_weights, &dummy_socket, "x100")
            .with_affinity(*bad_aff);

        let dup_arena = arena
            .duplicate_fd()
            .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
            .expect("dup arena");

        let err = supervisor
            .spawn_worker(placement, dup_arena, 1, 4096, 64, 4)
            .unwrap_err();

        assert!(matches!(err, ManagerError::Affinity(_)));
        let err_str = format!("{err}");
        assert!(err_str.contains("encroaches on dedicated active control cores 0x00C0"));
    }
}

#[test]
fn test_supervisor_spawn_worker_rejects_exceeding_a100_thread_cap() {
    let dummy_weights = std::path::PathBuf::from("/tmp/dummy_weights.gguf");
    let dummy_socket = std::path::PathBuf::from("/tmp/dummy_socket.sock");
    let arena = SharedMemoryArena::create("a100_cap_test_arena", 4096).expect("create arena");
    let mut supervisor = WorkerSupervisor::new();

    // Requesting 9 threads on A100 exceeds the 8-thread compute limit
    let placement = WorkerPlacement::new("model_9threads", &dummy_weights, &dummy_socket, "a100")
        .with_threads(9);

    let dup_arena = arena
        .duplicate_fd()
        .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
        .expect("dup arena");

    let err = supervisor
        .spawn_worker(placement, dup_arena, 1, 4096, 64, 4)
        .unwrap_err();

    assert!(matches!(err, ManagerError::Affinity(_)));
    let err_str = format!("{err}");
    assert!(err_str.contains("exceeding A100 max compute limit 8"));
}

#[test]
fn test_supervisor_spawn_worker_rejects_zero_threads() {
    let dummy_weights = std::path::PathBuf::from("/tmp/dummy_weights.gguf");
    let dummy_socket = std::path::PathBuf::from("/tmp/dummy_socket.sock");
    let arena = SharedMemoryArena::create("zero_threads_arena", 4096).expect("create arena");
    let mut supervisor = WorkerSupervisor::new();

    let placement = WorkerPlacement::new("model_0threads", &dummy_weights, &dummy_socket, "x100")
        .with_threads(0);

    let dup_arena = arena
        .duplicate_fd()
        .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
        .expect("dup arena");

    let err = supervisor
        .spawn_worker(placement, dup_arena, 1, 4096, 64, 4)
        .unwrap_err();

    assert!(matches!(err, ManagerError::Affinity(_)));
    let err_str = format!("{err}");
    assert!(err_str.contains("must be greater than 0"));
}

#[test]
fn test_worker_placement_thread_validation() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let weights = std::path::PathBuf::from("/tmp/dummy_weights.gguf");
    let socket = std::path::PathBuf::from("/tmp/dummy_socket.sock");

    // Valid 6 threads on X100
    let valid_x100 = WorkerPlacement::new("m1", &weights, &socket, "x100").with_threads(6);
    assert!(res.validate_worker_placement(&valid_x100).is_ok());

    // Valid 8 threads on A100
    let valid_a100 = WorkerPlacement::new("m2", &weights, &socket, "a100").with_threads(8);
    assert!(res.validate_worker_placement(&valid_a100).is_ok());

    // Threads exceeding mask count
    let mut excess_threads = WorkerPlacement::new("m3", &weights, &socket, "x100").with_threads(4);
    excess_threads.affinity = Some("0x0003".to_string()); // only 2 cores
    let err = res.validate_worker_placement(&excess_threads).unwrap_err();
    assert!(matches!(err, ManagerError::Affinity(_)));
    assert!(format!("{err}").contains("exceeding 2 available cores in affinity mask 0x0003"));
}

#[test]
fn test_worker_placement_cross_cluster_mask_rejection() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let weights = std::path::PathBuf::from("/tmp/dummy_weights.gguf");
    let socket = std::path::PathBuf::from("/tmp/dummy_socket.sock");

    // X100 worker placement specifying A100 cores (0xFF00)
    let p_x100_a100_cores =
        WorkerPlacement::new("m_x100_bad", &weights, &socket, "x100").with_affinity("0xFF00");
    let err = res
        .validate_worker_placement(&p_x100_a100_cores)
        .unwrap_err();
    assert!(matches!(err, ManagerError::Affinity(_)));
    assert!(format!("{err}").contains("specifies cores outside X100 mask 0x00FF"));

    // A100 worker placement specifying X100 cores (0x003F)
    let p_a100_x100_cores =
        WorkerPlacement::new("m_a100_bad", &weights, &socket, "a100").with_affinity("0x003F");
    let err_a100 = res
        .validate_worker_placement(&p_a100_x100_cores)
        .unwrap_err();
    assert!(matches!(err_a100, ManagerError::Affinity(_)));
    assert!(format!("{err_a100}").contains("specifies cores outside A100 mask 0xFF00"));

    // Unsupported cluster string rejection
    let p_bad_cluster =
        WorkerPlacement::new("m_bad_cluster", &weights, &socket, "gpu").with_threads(2);
    let err_cluster = res.validate_worker_placement(&p_bad_cluster).unwrap_err();
    assert!(matches!(err_cluster, ManagerError::WorkerSpawn(_)));
    assert!(format!("{err_cluster}").contains("unsupported cluster 'gpu'"));
}

#[test]
fn test_worker_placement_non_contiguous_affinity() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let weights = std::path::PathBuf::from("/tmp/dummy_weights.gguf");
    let socket = std::path::PathBuf::from("/tmp/dummy_socket.sock");

    // Non-contiguous affinity mask 0x0005 (cores 0 and 2) with 2 threads -> OK
    let p_valid_non_contig = WorkerPlacement::new("m_non_contig", &weights, &socket, "x100")
        .with_affinity("0x0005")
        .with_threads(2);
    assert!(res.validate_worker_placement(&p_valid_non_contig).is_ok());

    // Non-contiguous affinity mask 0x0005 with 3 threads -> rejected (3 > 2)
    let p_excess_threads = WorkerPlacement::new("m_non_contig_excess", &weights, &socket, "x100")
        .with_affinity("0x0005")
        .with_threads(3);
    let err = res
        .validate_worker_placement(&p_excess_threads)
        .unwrap_err();
    assert!(matches!(err, ManagerError::Affinity(_)));
    assert!(format!("{err}").contains("exceeding 2 available cores in affinity mask 0x0005"));
}
