//! Integration tests for cgroup v2 domain placement and OS affinity fallback.
//!
//! Verifies US-06.1.2 requirements:
//! - Canonical `cpuset.cpus` bitmask-to-list formatting and parsing (ranges, single cores, disjoint runs).
//! - Deterministic cgroup v2 leaf directory creation (`.../control`, `.../engines/<model>`).
//! - `cpuset.cpus` formatting and `cgroup.procs` / `cgroup.threads` migration.
//! - Seamless fallback to OS thread/process affinity on `EACCES` / `ENOENT` / unwritable cgroup.
//! - `SpacemiT` HMP `/proc/set_ai_thread` registration for A100 cores.
//! - Supervisor `spawn_worker` and `recover_worker` integration.
//! - Control plane scheduler helper integration.

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::ipc::shm::SharedMemoryArena;
use k3_manager::capacity::{
    CgroupV2DomainController, ControlPlaneScheduler, DomainPlacementMode, MockCgroupFs,
    MockOsAffinityBackend, format_cpuset_list, parse_cpuset_list, resolve_worker_compute_mask,
};
use k3_manager::error::ManagerError;
use k3_manager::supervisor::{WorkerPlacement, WorkerSupervisor};

#[test]
fn test_cpuset_list_formatting_and_parsing_roundtrip() {
    let test_cases: &[(u64, &str)] = &[
        (0x0001, "0"),
        (0x0002, "1"),
        (0x0020, "5"),
        (0x0080, "7"),
        (0x8000, "15"),
        (0x0003, "0-1"),
        (0x003F, "0-5"),
        (0x00C0, "6-7"),
        (0x00E0, "5-7"),
        (0x00FF, "0-7"),
        (0xFF00, "8-15"),
        (0xFFFF, "0-15"),
        (0x00A5, "0,2,5,7"),
        (0x00F3, "0-1,4-7"),
        (0x0447, "0-2,6,10"),
        (0x73C7, "0-2,6-9,12-14"),
    ];

    for &(mask, expected_str) in test_cases {
        let formatted = format_cpuset_list(mask)
            .unwrap_or_else(|e| panic!("failed to format mask 0x{mask:04X}: {e}"));
        assert_eq!(
            formatted, expected_str,
            "formatting mismatch for 0x{mask:04X}"
        );

        let parsed = parse_cpuset_list(expected_str)
            .unwrap_or_else(|e| panic!("failed to parse '{expected_str}': {e}"));
        assert_eq!(
            parsed, mask,
            "parsed mask mismatch for '{expected_str}' (expected 0x{mask:04X}, got 0x{parsed:04X})"
        );
    }
}

#[test]
fn test_cpuset_list_invalid_masks_and_strings_rejection() {
    // 1. Invalid masks
    assert!(format_cpuset_list(0).is_err());
    assert!(format_cpuset_list(0x1_0000).is_err());
    assert!(format_cpuset_list(0x8000_0000_0000_0000).is_err());

    // 2. Invalid string syntax
    let bad_strings = &[
        "", "   ", ",", "0,,1", ",0", "0,", "16", "0-16", "15-16", "7-5",
        "5-5", // single core as range start=end should still parse or be handled cleanly
        "0-1-2", "abc", "-1",
        "0 - 5", // whitespace in tokens should be trimmed or rejected gracefully
    ];

    for &bad in bad_strings {
        if bad == "5-5" || bad == "0 - 5" {
            // "5-5" is mathematically valid (core 5), and "0 - 5" with whitespace trims to 0-5
            let _ = parse_cpuset_list(bad);
        } else {
            let res = parse_cpuset_list(bad);
            assert!(
                res.is_err(),
                "expected rejection for '{bad}', but got: {res:?}"
            );
        }
    }
}

#[test]
fn test_cgroup_v2_control_plane_placement_happy_path() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/sys/fs/cgroup/k3.slice")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    let cap = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("default capacity resolution");

    // Place control plane with explicit PID 12345
    let outcome = controller
        .place_control_plane(12345, &cap)
        .expect("control placement");

    assert_eq!(outcome.target_id, 12345);
    match outcome.mode {
        DomainPlacementMode::CgroupV2 {
            cgroup_path,
            cpuset,
        } => {
            assert_eq!(
                cgroup_path,
                PathBuf::from("/sys/fs/cgroup/k3.slice/control")
            );
            assert_eq!(cpuset, "5-7");
        }
        DomainPlacementMode::OsAffinityFallback { .. } => {
            panic!("expected CgroupV2 placement, got OsAffinityFallback");
        }
    }

    // Verify mock filesystem state
    assert!(fs.is_dir_created("/sys/fs/cgroup/k3.slice/control"));
    assert_eq!(
        fs.written_file("/sys/fs/cgroup/k3.slice/control/cpuset.cpus")
            .as_deref(),
        Some("5-7\n")
    );
    assert_eq!(
        fs.written_file("/sys/fs/cgroup/k3.slice/control/cgroup.procs")
            .as_deref(),
        Some("12345\n")
    );
    assert_eq!(aff.calls().len(), 0);
}

#[test]
fn test_cgroup_v2_control_plane_placement_target_pid_zero_resolves_current_pid() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/sys/fs/cgroup/k3.slice")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    let cap = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("default capacity resolution");

    // PID 0 must resolve to current process ID
    let current_pid = std::process::id();
    let outcome = controller
        .place_control_plane(0, &cap)
        .expect("control placement");

    assert_eq!(outcome.target_id, current_pid);
    assert_eq!(
        fs.written_file("/sys/fs/cgroup/k3.slice/control/cgroup.procs")
            .as_deref(),
        Some(format!("{current_pid}\n")).as_deref()
    );
}

#[test]
fn test_cgroup_v2_engine_worker_placement_happy_path() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/sys/fs/cgroup/k3.slice")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    let outcome = controller
        .place_worker(4200, "qwen2.5-7b", 0x003F)
        .expect("worker placement");

    assert_eq!(outcome.target_id, 4200);
    match outcome.mode {
        DomainPlacementMode::CgroupV2 {
            cgroup_path,
            cpuset,
        } => {
            assert_eq!(
                cgroup_path,
                PathBuf::from("/sys/fs/cgroup/k3.slice/engines/qwen2.5-7b")
            );
            assert_eq!(cpuset, "0-5");
        }
        DomainPlacementMode::OsAffinityFallback { .. } => {
            panic!("expected CgroupV2 placement, got OsAffinityFallback");
        }
    }

    assert!(fs.is_dir_created("/sys/fs/cgroup/k3.slice/engines/qwen2.5-7b"));
    assert_eq!(
        fs.written_file("/sys/fs/cgroup/k3.slice/engines/qwen2.5-7b/cpuset.cpus")
            .as_deref(),
        Some("0-5\n")
    );
    assert_eq!(
        fs.written_file("/sys/fs/cgroup/k3.slice/engines/qwen2.5-7b/cgroup.procs")
            .as_deref(),
        Some("4200\n")
    );
    assert_eq!(aff.calls().len(), 0);
}

#[test]
fn test_cgroup_v2_thread_migration_fallback_when_procs_fails() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/sys/fs/cgroup/k3.slice")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    // Inject error on cgroup.procs only (simulating cgroup v2 threaded domain)
    fs.inject_error(
        "/sys/fs/cgroup/k3.slice/engines/deepseek/cgroup.procs",
        io::ErrorKind::InvalidInput,
    );

    let outcome = controller
        .place_worker(4300, "deepseek", 0x003F)
        .expect("worker placement with thread fallback");

    assert_eq!(outcome.target_id, 4300);
    assert!(outcome.mode.is_cgroup_v2());

    // Verify cgroup.threads was successfully written instead
    assert_eq!(
        fs.written_file("/sys/fs/cgroup/k3.slice/engines/deepseek/cgroup.threads")
            .as_deref(),
        Some("4300\n")
    );
    assert_eq!(aff.calls().len(), 0);
}

#[test]
fn test_cgroup_v2_eacces_directory_creation_fallback_to_os_affinity() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/sys/fs/cgroup/k3.slice")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    // Inject EACCES (PermissionDenied) on the engine leaf directory
    fs.inject_error(
        "/sys/fs/cgroup/k3.slice/engines/restricted-model",
        io::ErrorKind::PermissionDenied,
    );

    let outcome = controller
        .place_worker(7777, "restricted-model", 0x003F)
        .expect("worker placement fallback");

    assert_eq!(outcome.target_id, 7777);
    match outcome.mode {
        DomainPlacementMode::OsAffinityFallback { mask, reason } => {
            assert_eq!(mask, 0x003F);
            assert!(
                reason.contains("cgroup v2 engine placement failed"),
                "reason: {reason}"
            );
        }
        DomainPlacementMode::CgroupV2 { .. } => {
            panic!("expected OsAffinityFallback, got CgroupV2");
        }
    }

    assert!(aff.has_called(7777, 0x003F));
}

#[test]
fn test_cgroup_v2_enoent_base_fallback_to_os_affinity() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/sys/fs/cgroup/k3.slice")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    // Inject ENOENT (NotFound) on root cgroup path
    fs.inject_error("/sys/fs/cgroup/k3.slice", io::ErrorKind::NotFound);

    let cap = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("capacity resolution");

    let outcome = controller
        .place_control_plane(8888, &cap)
        .expect("control placement fallback");

    assert_eq!(outcome.target_id, 8888);
    match outcome.mode {
        DomainPlacementMode::OsAffinityFallback { mask, reason } => {
            assert_eq!(mask, 0x00E0);
            assert!(
                reason.contains("cgroup v2 control placement failed"),
                "reason: {reason}"
            );
        }
        DomainPlacementMode::CgroupV2 { .. } => {
            panic!("expected OsAffinityFallback, got CgroupV2");
        }
    }

    assert!(aff.has_called(8888, 0x00E0));
}

#[test]
fn test_cgroup_v2_cpuset_write_failure_fallback_to_os_affinity() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/sys/fs/cgroup/k3.slice")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    // Inject ReadOnlyFilesystem on cpuset.cpus
    fs.inject_error(
        "/sys/fs/cgroup/k3.slice/engines/ro-model/cpuset.cpus",
        io::ErrorKind::ReadOnlyFilesystem,
    );

    let outcome = controller
        .place_worker(9999, "ro-model", 0x001F)
        .expect("fallback on cpuset write failure");

    assert!(outcome.mode.is_os_affinity_fallback());
    assert!(aff.has_called(9999, 0x001F));
}

#[test]
fn test_a100_cluster_proc_set_ai_thread_invariant() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/sys/fs/cgroup/k3.slice")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    // Inject permission error to trigger OS affinity fallback
    fs.inject_error(
        "/sys/fs/cgroup/k3.slice/engines/a100-worker",
        io::ErrorKind::PermissionDenied,
    );

    // Pinning to Cluster 1 (A100 cores 8–15, mask 0xFF00)
    let a100_placement = controller
        .place_worker(1010, "a100-worker", 0xFF00)
        .expect("a100 worker placement");

    assert!(a100_placement.mode.is_os_affinity_fallback());
    assert!(aff.has_called(1010, 0xFF00));
    assert_eq!(aff.ai_thread_writes(), vec![1010]);

    // Pinning to Cluster 0 (X100 cores 0–5, mask 0x003F)
    fs.inject_error(
        "/sys/fs/cgroup/k3.slice/engines/x100-worker",
        io::ErrorKind::PermissionDenied,
    );
    let x100_placement = controller
        .place_worker(2020, "x100-worker", 0x003F)
        .expect("x100 worker placement");

    assert!(x100_placement.mode.is_os_affinity_fallback());
    assert!(aff.has_called(2020, 0x003F));
    // Must NOT have added 2020 to ai_thread_writes
    assert_eq!(aff.ai_thread_writes(), vec![1010]);
}

#[test]
fn test_affinity_fallback_failure_propagates_error() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/sys/fs/cgroup/k3.slice")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    // Both cgroup v2 and OS affinity fail
    fs.inject_error(
        "/sys/fs/cgroup/k3.slice/engines/fail-model",
        io::ErrorKind::PermissionDenied,
    );
    aff.fail_pid(3030);

    let err = controller
        .place_worker(3030, "fail-model", 0x003F)
        .unwrap_err();

    assert!(matches!(err, ManagerError::Affinity(_)));
}

#[test]
fn test_invalid_core_masks_rejected_without_cgroup_or_affinity_calls() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/sys/fs/cgroup/k3.slice")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    // Zero mask
    let err_zero = controller.place_worker(4040, "zero-mask", 0).unwrap_err();
    assert!(matches!(err_zero, ManagerError::Affinity(_)));

    // Out-of-bounds mask (bit 16 set)
    let err_oob = controller
        .place_worker(4040, "oob-mask", 0x1_0000)
        .unwrap_err();
    assert!(matches!(err_oob, ManagerError::Affinity(_)));

    assert_eq!(fs.created_dirs().len(), 0);
    assert_eq!(aff.calls().len(), 0);
}

#[test]
fn test_control_plane_scheduler_place_control_plane() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/mock/cgroup")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    let cap = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("capacity");
    let scheduler = ControlPlaneScheduler::new(cap);

    let outcome = scheduler
        .place_control_plane(5050, &controller)
        .expect("scheduler control placement");

    assert_eq!(outcome.target_id, 5050);
    assert!(outcome.mode.is_cgroup_v2());
    assert_eq!(
        fs.written_file("/mock/cgroup/control/cgroup.procs")
            .as_deref(),
        Some("5050\n")
    );
}

#[test]
fn test_worker_supervisor_place_control_plane_helper() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = Arc::new(
        CgroupV2DomainController::new("/mock/cgroup")
            .with_fs(fs.clone())
            .with_affinity_backend(aff.clone()),
    );

    let supervisor = WorkerSupervisor::new().with_domain_controller(controller);

    let cap = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("capacity");

    let outcome = supervisor
        .place_control_plane(6060, &cap)
        .expect("supervisor control placement");

    assert_eq!(outcome.target_id, 6060);
    assert!(outcome.mode.is_cgroup_v2());
    assert_eq!(
        fs.written_file("/mock/cgroup/control/cgroup.procs")
            .as_deref(),
        Some("6060\n")
    );
}

#[test]
fn test_model_id_validation_rejects_empty_and_traversal() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/mock/cgroup")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    let invalid_models = &[
        "",
        "   ",
        ".",
        "..",
        "../control",
        "../../etc/passwd",
        "org/model",
        "nested\\dir",
        "model\0null",
        "model\ncontrol",
        "model\rcontrol",
        "model\tcontrol",
        "model name",
        " qwen2.5",
        "qwen2.5 ",
        "\x1b[31mmodel",
        "model\x7f",
        ".hidden",
        "cgroup.procs",
        "cgroup.threads",
        "cpuset.cpus",
    ];

    for &bad_model in invalid_models {
        let err = controller
            .place_worker(1001, bad_model, 0x003F)
            .expect_err(&format!("expected error for bad model_id '{bad_model}'"));
        assert!(
            matches!(err, ManagerError::WorkerSpawn(_)),
            "expected WorkerSpawn error for '{bad_model}', got {err:?}"
        );
    }

    // Verify /mock/cgroup/control was not created or modified by traversal attempts
    assert!(!fs.is_dir_created("/mock/cgroup/control"));
    assert_eq!(fs.written_file("/mock/cgroup/control/cgroup.procs"), None);
    assert_eq!(aff.calls().len(), 0);
}

#[test]
fn test_place_worker_rejects_worker_pid_zero_and_overflow() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/mock/cgroup")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    // Worker PID 0 must be rejected
    let err_zero = controller
        .place_worker(0, "qwen2.5-7b", 0x003F)
        .expect_err("worker pid 0 must be rejected");
    assert!(matches!(err_zero, ManagerError::WorkerSpawn(_)));

    // Worker PID exceeding positive i32 bound must be rejected
    let err_overflow = controller
        .place_worker(u32::MAX, "qwen2.5-7b", 0x003F)
        .expect_err("worker pid u32::MAX must be rejected");
    assert!(matches!(err_overflow, ManagerError::WorkerSpawn(_)));

    let err_i32_plus_one = controller
        .place_worker((i32::MAX as u32) + 1, "qwen2.5-7b", 0x003F)
        .expect_err("worker pid i32::MAX + 1 must be rejected");
    assert!(matches!(err_i32_plus_one, ManagerError::WorkerSpawn(_)));

    // Target PID overflow in place_control_plane must also be rejected
    let cap = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("capacity");
    let err_ctrl_overflow = controller
        .place_control_plane(u32::MAX, &cap)
        .expect_err("target pid u32::MAX must be rejected");
    assert!(matches!(err_ctrl_overflow, ManagerError::WorkerSpawn(_)));

    assert_eq!(fs.created_dirs().len(), 0);
    assert_eq!(aff.calls().len(), 0);
}

#[test]
fn test_cgroup_v2_cpuset_write_einval_fallback_to_os_affinity() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/mock/cgroup")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    // Inject EINVAL (InvalidInput) simulating parent cpuset boundary refusal
    fs.inject_error(
        "/mock/cgroup/engines/einval-model/cpuset.cpus",
        io::ErrorKind::InvalidInput,
    );

    let outcome = controller
        .place_worker(7070, "einval-model", 0x003F)
        .expect("fallback on EINVAL");

    assert!(outcome.mode.is_os_affinity_fallback());
    assert!(aff.has_called(7070, 0x003F));
}

#[test]
fn test_cgroup_v2_include_spare_in_control_option() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());

    let cap = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("capacity");

    // 1. Controller with include_spare = false -> active only (cores 6-7, "6-7")
    let ctrl_no_spare = CgroupV2DomainController::new("/mock/no_spare")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone())
        .with_include_spare(false);
    assert!(!ctrl_no_spare.include_spare_in_control());

    let outcome_no_spare = ctrl_no_spare
        .place_control_plane(101, &cap)
        .expect("no spare placement");
    assert_eq!(
        fs.written_file("/mock/no_spare/control/cpuset.cpus")
            .as_deref(),
        Some("6-7\n")
    );
    match outcome_no_spare.mode {
        DomainPlacementMode::CgroupV2 { cpuset, .. } => assert_eq!(cpuset, "6-7"),
        DomainPlacementMode::OsAffinityFallback { .. } => panic!("expected CgroupV2"),
    }

    // 2. Controller with include_spare = true -> active + spare (cores 5-7, "5-7")
    let ctrl_with_spare = CgroupV2DomainController::new("/mock/with_spare")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone())
        .with_include_spare(true);
    assert!(ctrl_with_spare.include_spare_in_control());

    let outcome_with_spare = ctrl_with_spare
        .place_control_plane(102, &cap)
        .expect("with spare placement");
    assert_eq!(
        fs.written_file("/mock/with_spare/control/cpuset.cpus")
            .as_deref(),
        Some("5-7\n")
    );
    match outcome_with_spare.mode {
        DomainPlacementMode::CgroupV2 { cpuset, .. } => assert_eq!(cpuset, "5-7"),
        DomainPlacementMode::OsAffinityFallback { .. } => panic!("expected CgroupV2"),
    }
}

#[test]
fn test_cgroup_v2_subtree_control_intermediate_ancestors() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = CgroupV2DomainController::new("/mock/cgroup")
        .with_fs(fs.clone())
        .with_affinity_backend(aff.clone());

    let outcome = controller
        .place_worker(8080, "qwen2.5", 0x003F)
        .expect("worker placement");
    assert!(outcome.mode.is_cgroup_v2());

    // Verify intermediate cgroup subtree_control has +cpuset enabled
    assert_eq!(
        fs.written_file("/mock/cgroup/engines/cgroup.subtree_control")
            .as_deref(),
        Some("+cpuset\n")
    );
    assert_eq!(
        fs.written_file("/mock/cgroup/cgroup.subtree_control")
            .as_deref(),
        Some("+cpuset\n")
    );

    // Verify strictly TOP-DOWN write order: root ancestor -> engines ancestor -> leaf cpuset -> leaf procs
    let paths = fs.written_paths();
    let root_idx = paths
        .iter()
        .position(|p| p == &PathBuf::from("/mock/cgroup/cgroup.subtree_control"))
        .expect("root subtree control written");
    let engines_idx = paths
        .iter()
        .position(|p| p == &PathBuf::from("/mock/cgroup/engines/cgroup.subtree_control"))
        .expect("engines subtree control written");
    let cpuset_idx = paths
        .iter()
        .position(|p| p == &PathBuf::from("/mock/cgroup/engines/qwen2.5/cpuset.cpus"))
        .expect("cpuset.cpus written");
    let procs_idx = paths
        .iter()
        .position(|p| p == &PathBuf::from("/mock/cgroup/engines/qwen2.5/cgroup.procs"))
        .expect("cgroup.procs written");

    assert!(
        root_idx < engines_idx,
        "root ancestor must be enabled BEFORE engines intermediate cgroup (top-down rule)"
    );
    assert!(
        engines_idx < cpuset_idx,
        "engines subtree control must be enabled BEFORE leaf cpuset.cpus write"
    );
    assert!(
        cpuset_idx < procs_idx,
        "cpuset.cpus must be written BEFORE cgroup.procs migration"
    );
}

#[test]
fn test_parse_cpuset_list_rejects_non_digit_plus_signs_and_malformed_ranges() {
    let rejected = &["+0", "0-+5", "+0-5", "-1", "1-", "0-1-2", "--", "+1,4"];

    for &bad in rejected {
        let res = parse_cpuset_list(bad);
        assert!(
            res.is_err(),
            "expected rejection for '{bad}', but got: {res:?}"
        );
    }
}

#[test]
fn test_parse_cpuset_list_error_diagnostics_and_worker_mask_resolution() {
    let dummy_weights = PathBuf::from("/tmp/weights.gguf");
    let dummy_socket = PathBuf::from("/tmp/sock.sock");

    // 1. parse_cpuset_list directly returns detailed AffinityError diagnostics
    let err_oob = parse_cpuset_list("0-16").unwrap_err();
    match err_oob {
        ManagerError::Affinity(k3_err) => {
            assert!(
                k3_err.to_string().contains("16"),
                "expected core 16 diagnostic in {k3_err}"
            );
        }
        other => panic!("expected ManagerError::Affinity, got {other:?}"),
    }

    let err_inv = parse_cpuset_list("5-3").unwrap_err();
    match err_inv {
        ManagerError::Affinity(k3_err) => {
            assert!(
                k3_err.to_string().contains("inverted"),
                "expected inverted range diagnostic in {k3_err}"
            );
        }
        other => panic!("expected ManagerError::Affinity, got {other:?}"),
    }

    // 2. resolve_worker_compute_mask rejects invalid cpuset strings with WorkerSpawn
    let mut p_oob = WorkerPlacement::new("m", &dummy_weights, &dummy_socket, "x100");
    p_oob.affinity = Some("0-16".to_string());
    assert!(matches!(
        resolve_worker_compute_mask(&p_oob).unwrap_err(),
        ManagerError::WorkerSpawn(_)
    ));

    let mut p_inv = WorkerPlacement::new("m", &dummy_weights, &dummy_socket, "x100");
    p_inv.affinity = Some("5-3".to_string());
    assert!(matches!(
        resolve_worker_compute_mask(&p_inv).unwrap_err(),
        ManagerError::WorkerSpawn(_)
    ));
}

#[test]
fn test_exhaustive_cpuset_roundtrip_all_16bit_masks() {
    for mask in 1_u64..=0xFFFF {
        let formatted = format_cpuset_list(mask)
            .unwrap_or_else(|e| panic!("failed to format 0x{mask:04X}: {e}"));
        let parsed = parse_cpuset_list(&formatted)
            .unwrap_or_else(|e| panic!("failed to parse '{formatted}' for 0x{mask:04X}: {e}"));
        assert_eq!(
            parsed, mask,
            "roundtrip mismatch: mask=0x{mask:04X}, formatted='{formatted}', parsed=0x{parsed:04X}"
        );
    }
}

#[test]
fn test_multi_model_worker_placement_concurrent_supervisors() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = Arc::new(
        CgroupV2DomainController::new("/mock/cgroup")
            .with_fs(fs.clone())
            .with_affinity_backend(aff.clone()),
    );

    let models_and_masks = [
        ("model-alpha", 0x0007_u64, 10001_u32),
        ("model-beta", 0x0038_u64, 10002_u32),
        ("model-gamma", 0x0F00_u64, 10003_u32),
        ("model-delta", 0xF000_u64, 10004_u32),
    ];

    let handles: Vec<_> = models_and_masks
        .iter()
        .map(|&(model, mask, pid)| {
            let ctrl = controller.clone();
            std::thread::spawn(move || {
                ctrl.place_worker(pid, model, mask)
                    .unwrap_or_else(|e| panic!("failed placement for {model}: {e}"))
            })
        })
        .collect();

    for (h, &(model, mask, pid)) in handles.into_iter().zip(models_and_masks.iter()) {
        let outcome = h.join().expect("thread join");
        assert_eq!(outcome.target_id, pid);
        assert!(outcome.mode.is_cgroup_v2());

        let expected_cpuset = format_cpuset_list(mask).unwrap();
        let cpuset_path = format!("/mock/cgroup/engines/{model}/cpuset.cpus");
        let procs_path = format!("/mock/cgroup/engines/{model}/cgroup.procs");

        assert_eq!(
            fs.written_file(&cpuset_path).as_deref(),
            Some(format!("{expected_cpuset}\n").as_str())
        );
        assert_eq!(
            fs.written_file(&procs_path).as_deref(),
            Some(format!("{pid}\n").as_str())
        );
    }
}

fn create_dummy_gguf_weights(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("k3_domain_test_{name}_{}.gguf", std::process::id()));
    let magic: [u8; 4] = *b"GGUF";
    let version: u32 = 3;
    let tensor_count: u64 = 0;
    let kv_count: u64 = 0;

    let mut data = Vec::new();
    data.extend_from_slice(&magic);
    data.extend_from_slice(&version.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
    data.extend_from_slice(&kv_count.to_le_bytes());

    std::fs::write(&p, data).expect("write dummy gguf");
    p
}

static TEST_SOCK_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn unique_socket_path(name: &str) -> PathBuf {
    let pid = std::process::id();
    let seq = TEST_SOCK_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("{name}_{pid}_{seq}.sock"))
}

#[test]
fn test_supervisor_spawn_worker_early_validation_rejection() {
    let dummy_weights = PathBuf::from("/tmp/dummy_early_weights.gguf");
    let dummy_socket = PathBuf::from("/tmp/dummy_early_sock.sock");

    let arena = SharedMemoryArena::create("early_val_arena", 4096).expect("create arena");
    let mut supervisor = WorkerSupervisor::new();

    // 1. Invalid model_id rejection before spawn
    let invalid_models = [
        "",
        "   ",
        "model/traversal",
        "model a",
        "model\n",
        "cgroup.procs",
        ".hidden",
    ];
    for bad_id in invalid_models {
        let placement = WorkerPlacement::new(bad_id, &dummy_weights, &dummy_socket, "x100");
        let dup_arena = arena
            .duplicate_fd()
            .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
            .expect("dup");
        let err = supervisor
            .spawn_worker(placement, dup_arena, 1, 4096, 64, 4)
            .unwrap_err();
        assert!(matches!(err, ManagerError::WorkerSpawn(_)));
    }

    // 2. Invalid affinity rejection before spawn
    let invalid_affinities = ["-1", "not_a_num", "0-16", "16", "0x10000"];
    for bad_aff in invalid_affinities {
        let mut placement =
            WorkerPlacement::new("valid_model", &dummy_weights, &dummy_socket, "x100");
        placement.affinity = Some(bad_aff.to_string());
        let dup_arena = arena
            .duplicate_fd()
            .and_then(|fd| SharedMemoryArena::from_existing_fd(fd, 4096, "dup", false))
            .expect("dup");
        let err = supervisor
            .spawn_worker(placement, dup_arena, 1, 4096, 64, 4)
            .unwrap_err();
        assert!(
            matches!(
                err,
                ManagerError::WorkerSpawn(_) | ManagerError::Affinity(_)
            ),
            "expected rejection for '{bad_aff}', got {err:?}"
        );
    }

    assert_eq!(supervisor.worker_count(), 0);
}

#[test]
fn test_supervisor_spawn_worker_with_mock_domain_controller() {
    let fs = Arc::new(MockCgroupFs::new());
    let aff = Arc::new(MockOsAffinityBackend::new());
    let controller = Arc::new(
        CgroupV2DomainController::new("/mock/cgroup")
            .with_fs(fs.clone())
            .with_affinity_backend(aff.clone()),
    );

    let weights_path = create_dummy_gguf_weights("mock_sup_worker");
    let sock_path = unique_socket_path("m1");
    let arena = SharedMemoryArena::create("sup_arena", 65536).expect("create arena");

    let placement =
        WorkerPlacement::new("mock-model", &weights_path, &sock_path, "x100").with_affinity("0-5");

    let mut supervisor = WorkerSupervisor::new().with_domain_controller(controller);

    let worker = supervisor
        .spawn_worker(placement, arena, 100, 65536, 64, 4)
        .expect("spawn worker");

    assert!(worker.is_alive());
    let worker_pid = worker.pid().expect("worker pid");
    assert_eq!(worker.generation(), 1);

    // Verify mock cgroup fs has received leaf directory, cpuset, and procs writes
    assert!(fs.is_dir_created("/mock/cgroup/engines/mock-model"));
    assert_eq!(
        fs.written_file("/mock/cgroup/engines/mock-model/cpuset.cpus")
            .as_deref(),
        Some("0-5\n")
    );
    assert_eq!(
        fs.written_file("/mock/cgroup/engines/mock-model/cgroup.procs")
            .as_deref(),
        Some(format!("{worker_pid}\n").as_str())
    );

    let outcome = worker.last_placement_outcome().expect("outcome");
    assert_eq!(outcome.target_id, worker_pid);
    assert!(outcome.mode.is_cgroup_v2());

    // 2. Fallback to OS affinity when cgroup v2 write fails
    fs.inject_error(
        "/mock/cgroup/engines/fallback-model",
        io::ErrorKind::PermissionDenied,
    );
    let weights_fallback = create_dummy_gguf_weights("mock_sup_fallback");
    let sock_fallback = unique_socket_path("m2");
    let arena_fallback =
        SharedMemoryArena::create("sup_arena_fallback", 65536).expect("create arena");
    let placement_fallback =
        WorkerPlacement::new("fallback-model", &weights_fallback, &sock_fallback, "x100")
            .with_affinity("0-5");

    let worker_fallback = supervisor
        .spawn_worker(placement_fallback, arena_fallback, 101, 65536, 64, 4)
        .expect("spawn worker fallback");

    assert!(worker_fallback.is_alive());
    let fallback_pid = worker_fallback.pid().expect("fallback pid");
    let fallback_outcome = worker_fallback.last_placement_outcome().expect("outcome");
    assert_eq!(fallback_outcome.target_id, fallback_pid);
    assert!(fallback_outcome.mode.is_os_affinity_fallback());
    assert!(aff.has_called(fallback_pid, 0x003F));

    // Teardown
    supervisor.shutdown_all();
    let _ = std::fs::remove_file(&weights_path);
    let _ = std::fs::remove_file(&sock_path);
    let _ = std::fs::remove_file(&weights_fallback);
    let _ = std::fs::remove_file(&sock_fallback);
}
