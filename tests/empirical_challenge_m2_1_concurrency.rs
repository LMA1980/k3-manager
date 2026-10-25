//! Empirical Challenge & Stress Test Suite for `k3-manager` Milestone M2.
//!
//! Stress-tests:
//! 1. `ControlPlaneScheduler` concurrency, thread scalability, and fairness (32 & 64 threads, 100,000+ tasks).
//! 2. Round-robin fairness with cloned schedulers across concurrent threads.
//! 3. Minimal 1-core topology under 64-thread contention (128,000 tasks).
//! 4. Odd/unaligned task count fairness: max delta across active cores is <= 1.
//! 5. Watchdog recovery core binding invariants across default, minimal, degraded, and custom topologies.
//! 6. Concurrent capacity inspection under active scheduling contention.
//! 7. Direct `ResolvedControlCapacityExt` and `ControlPlaneScheduler` placement validation.
//! 8. Adversarial worker placement boundary validation (lendable spare, encroaching active cores, malformed masks).

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::undocumented_unsafe_blocks,
    clippy::too_many_lines
)]

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use k3_manager::error::ManagerError;
use k3_manager::supervisor::WorkerPlacement;
use k3_manager::{
    A100_CLUSTER_MASK, ControlPlaneCapacityConfig, ControlPlaneScheduler,
    DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_SPARE_RECOVERY_MASK, DEFAULT_X100_COMPUTE_CAP_MASK,
    ResolvedControlCapacityExt, X100_CLUSTER_MASK,
};

// ============================================================================
// 1. Concurrency & Fairness Stress Tests (32 & 64 threads, 100,000+ tasks)
// ============================================================================

#[test]
fn test_scheduler_stress_32_threads_100k_tasks_exact_fairness() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = Arc::new(ControlPlaneScheduler::new(res));

    const NUM_THREADS: usize = 32;
    const TASKS_PER_THREAD: usize = 3125; // 32 * 3,125 = 100,000 tasks total

    let mut handles = Vec::with_capacity(NUM_THREADS);
    for _ in 0..NUM_THREADS {
        let sched = Arc::clone(&scheduler);
        handles.push(thread::spawn(move || {
            let mut c6_count = 0_usize;
            let mut c7_count = 0_usize;
            for _ in 0..TASKS_PER_THREAD {
                let core = sched.schedule_control_task();
                match core {
                    6 => c6_count += 1,
                    7 => c7_count += 1,
                    other => panic!("unexpected active control core: {other}"),
                }
            }
            (c6_count, c7_count)
        }));
    }

    let mut total_c6 = 0;
    let mut total_c7 = 0;
    for h in handles {
        let (c6, c7) = h.join().expect("thread join failed");
        total_c6 += c6;
        total_c7 += c7;
    }

    assert_eq!(total_c6 + total_c7, 100_000);
    // Since 100,000 is an even number and round_robin_counter advances sequentially by 1,
    // the global assignment MUST be exactly 50,000 to core 6 and 50,000 to core 7.
    assert_eq!(total_c6, 50_000, "Core 6 should have exactly 50,000 tasks");
    assert_eq!(total_c7, 50_000, "Core 7 should have exactly 50,000 tasks");
}

#[test]
fn test_scheduler_stress_cloned_instances_across_32_threads() {
    // When ControlPlaneScheduler is cloned, it must share the underlying atomic counter
    // so independent clones maintain unified round-robin sequence.
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res);

    const NUM_THREADS: usize = 32;
    const TASKS_PER_THREAD: usize = 3125; // 100,000 total

    let mut handles = Vec::with_capacity(NUM_THREADS);
    for _ in 0..NUM_THREADS {
        let sched_clone = scheduler.clone();
        handles.push(thread::spawn(move || {
            let mut c6_count = 0_usize;
            let mut c7_count = 0_usize;
            for _ in 0..TASKS_PER_THREAD {
                let core = sched_clone.schedule_control_task();
                match core {
                    6 => c6_count += 1,
                    7 => c7_count += 1,
                    other => panic!("unexpected active control core: {other}"),
                }
            }
            (c6_count, c7_count)
        }));
    }

    let mut total_c6 = 0;
    let mut total_c7 = 0;
    for h in handles {
        let (c6, c7) = h.join().expect("thread join failed");
        total_c6 += c6;
        total_c7 += c7;
    }

    assert_eq!(total_c6 + total_c7, 100_000);
    assert_eq!(total_c6, 50_000, "Cloned scheduler must share atomic counter");
    assert_eq!(total_c7, 50_000, "Cloned scheduler must share atomic counter");
}

#[test]
fn test_scheduler_stress_high_concurrency_64_threads() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = Arc::new(ControlPlaneScheduler::new(res));

    const NUM_THREADS: usize = 64;
    const TASKS_PER_THREAD: usize = 2000; // 128,000 tasks total

    let mut handles = Vec::with_capacity(NUM_THREADS);
    for _ in 0..NUM_THREADS {
        let sched = Arc::clone(&scheduler);
        handles.push(thread::spawn(move || {
            let mut c6 = 0_usize;
            let mut c7 = 0_usize;
            for _ in 0..TASKS_PER_THREAD {
                let core = sched.schedule_control_task();
                if core == 6 {
                    c6 += 1;
                } else if core == 7 {
                    c7 += 1;
                }
            }
            (c6, c7)
        }));
    }

    let mut total_6 = 0;
    let mut total_7 = 0;
    for h in handles {
        let (c6, c7) = h.join().expect("thread join failed");
        total_6 += c6;
        total_7 += c7;
    }

    assert_eq!(total_6 + total_7, 128_000);
    assert_eq!(total_6, 64_000);
    assert_eq!(total_7, 64_000);
}

#[test]
fn test_scheduler_stress_unaligned_task_count_fairness_delta() {
    // 32 threads running 3,127 tasks each = 100,064 tasks total (divisible by 2)
    // plus an extra single thread task to make it odd: 100,065 tasks
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = Arc::new(ControlPlaneScheduler::new(res));

    const NUM_THREADS: usize = 32;
    const TASKS_PER_THREAD: usize = 3127; // 100,064 total

    let mut handles = Vec::with_capacity(NUM_THREADS);
    for _ in 0..NUM_THREADS {
        let sched = Arc::clone(&scheduler);
        handles.push(thread::spawn(move || {
            let mut c6 = 0_usize;
            let mut c7 = 0_usize;
            for _ in 0..TASKS_PER_THREAD {
                if sched.schedule_control_task() == 6 {
                    c6 += 1;
                } else {
                    c7 += 1;
                }
            }
            (c6, c7)
        }));
    }

    let mut total_6 = 0;
    let mut total_7 = 0;
    for h in handles {
        let (c6, c7) = h.join().expect("thread join failed");
        total_6 += c6;
        total_7 += c7;
    }

    assert_eq!(total_6 + total_7, 100_064);
    assert_eq!(total_6, 50_032);
    assert_eq!(total_7, 50_032);

    // Schedule 1 more task: delta between core 6 and 7 must be exactly 1
    let extra = scheduler.schedule_control_task();
    assert_eq!(extra, 6); // First core after even cycle
}

#[test]
fn test_scheduler_stress_minimal_1core_64_threads() {
    let min_cfg = ControlPlaneCapacityConfig::minimal();
    let min_res = min_cfg.resolve_default().expect("resolve minimal");
    let scheduler = Arc::new(ControlPlaneScheduler::new(min_res));

    assert_eq!(scheduler.watchdog_recovery_core(), None);
    assert_eq!(scheduler.capacity().active_control_cores, vec![7]);

    const NUM_THREADS: usize = 64;
    const TASKS_PER_THREAD: usize = 2000; // 128,000 tasks total

    let mut handles = Vec::with_capacity(NUM_THREADS);
    for _ in 0..NUM_THREADS {
        let sched = Arc::clone(&scheduler);
        handles.push(thread::spawn(move || {
            let mut count = 0_usize;
            for _ in 0..TASKS_PER_THREAD {
                let core = sched.schedule_control_task();
                assert_eq!(core, 7);
                count += 1;
            }
            count
        }));
    }

    let mut total = 0;
    for h in handles {
        total += h.join().expect("thread join");
    }

    assert_eq!(total, 128_000);
}

#[test]
fn test_scheduler_stress_3core_active_distribution() {
    // 3 active control cores: 5, 6, 7
    let cfg = ControlPlaneCapacityConfig::new()
        .with_active_cores(3)
        .with_spare_cores(0);
    let res = cfg.resolve_default().expect("resolve 3-core active");
    let scheduler = Arc::new(ControlPlaneScheduler::new(res));

    assert_eq!(scheduler.capacity().active_control_cores, vec![5, 6, 7]);
    assert_eq!(scheduler.watchdog_recovery_core(), None);

    const NUM_THREADS: usize = 30;
    const TASKS_PER_THREAD: usize = 3000; // 90,000 tasks total (divisible by 3)

    let mut handles = Vec::with_capacity(NUM_THREADS);
    for _ in 0..NUM_THREADS {
        let sched = Arc::clone(&scheduler);
        handles.push(thread::spawn(move || {
            let mut c5 = 0_usize;
            let mut c6 = 0_usize;
            let mut c7 = 0_usize;
            for _ in 0..TASKS_PER_THREAD {
                match sched.schedule_control_task() {
                    5 => c5 += 1,
                    6 => c6 += 1,
                    7 => c7 += 1,
                    other => panic!("unexpected core: {other}"),
                }
            }
            (c5, c6, c7)
        }));
    }

    let mut total_5 = 0;
    let mut total_6 = 0;
    let mut total_7 = 0;
    for h in handles {
        let (c5, c6, c7) = h.join().expect("thread join");
        total_5 += c5;
        total_6 += c6;
        total_7 += c7;
    }

    assert_eq!(total_5 + total_6 + total_7, 90_000);
    assert_eq!(total_5, 30_000, "Core 5 must have 30,000 tasks");
    assert_eq!(total_6, 30_000, "Core 6 must have 30,000 tasks");
    assert_eq!(total_7, 30_000, "Core 7 must have 30,000 tasks");
}

// ============================================================================
// 2. Watchdog Recovery Core Invariants
// ============================================================================

#[test]
fn test_watchdog_recovery_core_invariants() {
    // 1. Default 8-core X100: watchdog must be Some(5)
    let default_cfg = ControlPlaneCapacityConfig::default();
    let default_res = default_cfg.resolve_default().expect("default resolution");
    let default_sched = ControlPlaneScheduler::new(default_res);
    assert_eq!(default_sched.watchdog_recovery_core(), Some(5));

    // Watchdog core must not be in active control cores
    let watchdog_core = default_sched.watchdog_recovery_core().unwrap();
    assert!(!default_sched.capacity().active_control_cores.contains(&watchdog_core));
    assert_eq!(default_sched.capacity().active_control_mask & (1 << watchdog_core), 0);
    // Watchdog core must be in X100 cluster mask
    assert_ne!(X100_CLUSTER_MASK & (1 << watchdog_core), 0);
    // Watchdog core must NOT be in A100 cluster mask
    assert_eq!(A100_CLUSTER_MASK & (1 << watchdog_core), 0);

    // 2. Minimal 1-core config: watchdog must be None
    let min_cfg = ControlPlaneCapacityConfig::minimal();
    let min_res = min_cfg.resolve_default().expect("minimal resolution");
    let min_sched = ControlPlaneScheduler::new(min_res);
    assert_eq!(min_sched.watchdog_recovery_core(), None);

    // 3. Degraded topologies with minimal config: watchdog must be None
    for online_mask in [0x000F, 0x0007, 0x0003, 0x0001] {
        let degraded_res = min_cfg
            .resolve_with_online_mask(online_mask)
            .expect("resolve degraded minimal");
        let degraded_sched = ControlPlaneScheduler::new(degraded_res);
        assert_eq!(
            degraded_sched.watchdog_recovery_core(),
            None,
            "Minimal config on mask 0x{online_mask:04X} must yield None watchdog core"
        );
    }

    // 4. Custom config with explicit 0 spare cores: watchdog must be None
    let no_spare_cfg = ControlPlaneCapacityConfig::new()
        .with_active_cores(2)
        .with_spare_cores(0);
    let no_spare_res = no_spare_cfg.resolve_default().expect("resolve no spare");
    let no_spare_sched = ControlPlaneScheduler::new(no_spare_res);
    assert_eq!(no_spare_sched.watchdog_recovery_core(), None);

    // 5. Custom config with 2 spare cores (e.g. cores 4 and 5): watchdog must return primary
    let two_spare_cfg = ControlPlaneCapacityConfig::new()
        .with_active_cores(2)
        .with_spare_cores(2);
    let two_spare_res = two_spare_cfg.resolve_default().expect("resolve 2 spare");
    let two_spare_sched = ControlPlaneScheduler::new(two_spare_res);
    // Cores 4 and 5 are spare, primary is 4
    assert_eq!(two_spare_sched.watchdog_recovery_core(), Some(4));
    assert_eq!(two_spare_sched.capacity().spare_recovery_cores, vec![4, 5]);
}

// ============================================================================
// 3. Concurrent Capacity Inspection Under Load
// ============================================================================

#[test]
fn test_concurrent_capacity_inspection_under_scheduling_load() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = Arc::new(ControlPlaneScheduler::new(res));
    let stop_signal = Arc::new(AtomicUsize::new(0));

    // Reader thread constantly inspecting capacity
    let sched_read = Arc::clone(&scheduler);
    let stop_read = Arc::clone(&stop_signal);
    let reader_handle = thread::spawn(move || {
        let mut read_cycles = 0;
        while stop_read.load(Ordering::Relaxed) == 0 {
            let cap = sched_read.capacity();
            assert_eq!(cap.active_control_mask, DEFAULT_ACTIVE_CONTROL_MASK);
            assert_eq!(cap.spare_recovery_mask, DEFAULT_SPARE_RECOVERY_MASK);
            assert_eq!(cap.x100_compute_cap_mask, DEFAULT_X100_COMPUTE_CAP_MASK);
            assert_eq!(sched_read.watchdog_recovery_core(), Some(5));
            read_cycles += 1;
        }
        read_cycles
    });

    // 16 worker threads dispatching tasks concurrently
    let mut workers = Vec::new();
    for _ in 0..16 {
        let sched_work = Arc::clone(&scheduler);
        workers.push(thread::spawn(move || {
            for _ in 0..5000 {
                let core = sched_work.schedule_control_task();
                assert!(core == 6 || core == 7);
            }
        }));
    }

    for w in workers {
        w.join().expect("worker join failed");
    }

    stop_signal.store(1, Ordering::Relaxed);
    let read_count = reader_handle.join().expect("reader join failed");
    assert!(read_count > 100, "Reader thread should execute many cycles safely");
}

// ============================================================================
// 4. Worker Placement Validation Stress & Edge Cases
// ============================================================================

#[test]
fn test_worker_placement_adversarial_affinity_parsing() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res.clone());

    let dummy_weights = PathBuf::from("/tmp/weights.gguf");
    let dummy_socket = PathBuf::from("/tmp/sock.sock");

    // Invalid hex strings
    for bad_affinity in [
        "not_a_number",
        "0xZZZZ",
        "0XGG",
        "0x",
        "0X",
        "-1",
        "0x1000000000000000000", // overflow u64
        "0x12 34",
    ] {
        let mut p = WorkerPlacement::new("m", &dummy_weights, &dummy_socket, "x100");
        p.affinity = Some(bad_affinity.to_string());
        let err = scheduler.validate_worker_placement(&p).unwrap_err();
        assert!(
            matches!(err, ManagerError::WorkerSpawn(_)),
            "Expected WorkerSpawn error for affinity '{bad_affinity}', got {err:?}"
        );
        // Also verify extension trait direct call
        let ext_err = res.validate_worker_placement(&p).unwrap_err();
        assert!(matches!(ext_err, ManagerError::WorkerSpawn(_)));
    }

    // None affinity is always accepted
    let mut p_none = WorkerPlacement::new("m", &dummy_weights, &dummy_socket, "x100");
    p_none.affinity = None;
    assert!(scheduler.validate_worker_placement(&p_none).is_ok());
    assert!(res.validate_worker_placement(&p_none).is_ok());

    // Valid formats (decimal, lowercase hex, uppercase hex)
    for valid_affinity in ["15", "0x0f", "0X0F", "0x003F", "63"] {
        let mut p = WorkerPlacement::new("m", &dummy_weights, &dummy_socket, "x100");
        p.affinity = Some(valid_affinity.to_string());
        assert!(
            scheduler.validate_worker_placement(&p).is_ok(),
            "Expected valid affinity for '{valid_affinity}'"
        );
        assert!(res.validate_worker_placement(&p).is_ok());
    }
}

#[test]
fn test_worker_placement_lendable_spare_and_encroachment_boundaries() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res.clone());

    let dummy_weights = PathBuf::from("/tmp/weights.gguf");
    let dummy_socket = PathBuf::from("/tmp/sock.sock");

    // Lendable spare core 5 (0x0020) must be ACCEPTED for compute workers
    let mut p_spare = WorkerPlacement::new("spare_worker", &dummy_weights, &dummy_socket, "x100");
    p_spare.affinity = Some("0x0020".to_string());
    assert!(
        scheduler.validate_worker_placement(&p_spare).is_ok(),
        "Lendable spare core 5 must be accepted for worker placement per ADR-007"
    );
    assert!(res.validate_worker_placement(&p_spare).is_ok());

    // Active control core 6 (0x0040) must be REJECTED with ManagerError::Affinity
    let mut p_c6 = WorkerPlacement::new("c6_rogue", &dummy_weights, &dummy_socket, "x100");
    p_c6.affinity = Some("0x0040".to_string());
    let err_c6 = scheduler.validate_worker_placement(&p_c6).unwrap_err();
    assert!(matches!(err_c6, ManagerError::Affinity(_)));
    assert!(format!("{err_c6:?}").contains("encroaches on dedicated active control cores"));

    // Active control core 7 (0x0080) must be REJECTED with ManagerError::Affinity
    let mut p_c7 = WorkerPlacement::new("c7_rogue", &dummy_weights, &dummy_socket, "x100");
    p_c7.affinity = Some("0x0080".to_string());
    let err_c7 = scheduler.validate_worker_placement(&p_c7).unwrap_err();
    assert!(matches!(err_c7, ManagerError::Affinity(_)));
    assert!(format!("{err_c7:?}").contains("encroaches on dedicated active control cores"));

    // Both active control cores 6 and 7 (0x00C0) must be REJECTED
    let mut p_both = WorkerPlacement::new("both_rogue", &dummy_weights, &dummy_socket, "x100");
    p_both.affinity = Some("0x00C0".to_string());
    let err_both = scheduler.validate_worker_placement(&p_both).unwrap_err();
    assert!(matches!(err_both, ManagerError::Affinity(_)));

    // Cluster case insensitivity: "X100", "x100"
    let mut p_upper = WorkerPlacement::new("upper_x100", &dummy_weights, &dummy_socket, "X100");
    p_upper.affinity = Some("0x0040".to_string());
    assert!(scheduler.validate_worker_placement(&p_upper).is_err());

    // Verify error source propagation
    assert!(err_c6.source().is_some());
}
