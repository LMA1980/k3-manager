//! Empirical Challenge Test Suite for Milestone M2: Error Routing & Placement Validation.
//!
//! Subagent: `challenger_m2_2`
//! Focus:
//! 1. `From<K3Error>` correctly produces `ManagerError::Affinity` for `K3Error::AffinityError`
//!    and `ManagerError::TensorBounds` for other variants.
//! 2. `validate_worker_placement` correctly rejects encroachments on active control cores,
//!    correctly rejects exceeding the compute cap, and cleanly accepts valid compute placements.
//!
//! Specifications: ADR-007, ADR-008, ADR-041, Issue #69, PROJECT.md § Interface Contracts.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::items_after_statements
)]

use std::error::Error;
use std::path::PathBuf;

use k3_core::K3Error;
use k3_manager::error::ManagerError;
use k3_manager::supervisor::WorkerPlacement;
use k3_manager::{
    A100_CLUSTER_MASK, ControlPlaneCapacityConfig, ControlPlaneScheduler,
    DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_SPARE_RECOVERY_MASK, DEFAULT_X100_COMPUTE_CAP_MASK,
    ResolvedControlCapacityExt, X100_CLUSTER_MASK,
};

fn dummy_placement(model_id: &str, cluster: &str, affinity: Option<&str>) -> WorkerPlacement {
    let mut p = WorkerPlacement::new(
        model_id,
        PathBuf::from("/tmp/weights.gguf"),
        PathBuf::from("/tmp/worker.sock"),
        cluster,
    );
    p.affinity = affinity.map(ToString::to_string);
    p
}

// ============================================================================
// 1. Error Routing Empirical Challenges (From<K3Error> for ManagerError)
// ============================================================================

#[test]
fn test_empirical_error_routing_affinity_error_to_affinity_variant() {
    let test_cases = [
        (0x00FF_u64, 8_usize, "online core count exhaustion"),
        (0x00C0_u64, 2_usize, "overlapping dedicated active control cores"),
        (0x0020_u64, 1_usize, "spare recovery core conflict"),
        (0x0000_u64, 0_usize, "zero active control cores requested"),
        (0xFFFF_u64, 16_usize, "cross-cluster mask bounds exceeded"),
        (u64::MAX, 64_usize, "all bits set invalid mask"),
    ];

    for (mask, threads, reason_str) in test_cases {
        let k3_err = K3Error::AffinityError {
            requested_mask: mask,
            requested_threads: threads,
            reason: reason_str.to_string(),
        };

        // 1. Conversion via From / Into
        let mgr_err: ManagerError = k3_err.clone().into();

        // 2. Exact variant discrimination
        match &mgr_err {
            ManagerError::Affinity(inner) => {
                assert_eq!(inner, &k3_err);
                if let K3Error::AffinityError {
                    requested_mask,
                    requested_threads,
                    reason,
                } = inner
                {
                    assert_eq!(*requested_mask, mask);
                    assert_eq!(*requested_threads, threads);
                    assert_eq!(reason, reason_str);
                } else {
                    panic!("inner error was not AffinityError");
                }
            }
            other => panic!("expected ManagerError::Affinity, got {other:?}"),
        }

        // 3. Display formatting contains diagnostic prefix and details
        let display = format!("{mgr_err}");
        assert!(
            display.contains("[k3-manager affinity error]"),
            "Display missing prefix: {display}"
        );
        assert!(
            display.contains("[affinity_error]"),
            "Display missing inner diagnostic code: {display}"
        );
        assert!(
            display.contains(&format!("0x{mask:04X}")),
            "Display missing mask: {display}"
        );
        assert!(
            display.contains(&format!("threads: {threads}")),
            "Display missing thread count: {display}"
        );
        assert!(
            display.contains(reason_str),
            "Display missing reason: {display}"
        );

        // 4. Source error inspection & downcasting
        let source = mgr_err.source().expect("source() must be Some");
        let downcasted = source
            .downcast_ref::<K3Error>()
            .expect("downcast_ref to K3Error must succeed");
        assert_eq!(downcasted, &k3_err);
    }
}

#[test]
fn test_empirical_error_routing_non_affinity_variants_to_tensor_bounds() {
    let other_k3_errors: Vec<K3Error> = vec![
        K3Error::TensorBoundsError {
            tensor_name: "attn_q.weight".to_string(),
            offset: 1024,
            byte_len: 4096,
            buffer_len: 2048,
            reason: "tensor slice exceeds backing buffer length".to_string(),
        },
        K3Error::CapabilityMismatch {
            architecture: "unsupported_arch_v9".to_string(),
            reason: "hardware lacks required matrix vector extension".to_string(),
        },
        K3Error::InsufficientLockableMemory {
            required_bytes: 32 * 1024 * 1024 * 1024,
            weight_bytes: 28 * 1024 * 1024 * 1024,
            kv_bytes: 4 * 1024 * 1024 * 1024,
            available_bytes: 16 * 1024 * 1024 * 1024,
            rlimit_memlock_bytes: 24 * 1024 * 1024 * 1024,
        },
        K3Error::ContextLengthExceeded {
            requested_tokens: 8192,
            max_context_tokens: 4096,
        },
    ];

    for k3_err in other_k3_errors {
        let mgr_err: ManagerError = k3_err.clone().into();

        // 1. Variant discrimination: must route to TensorBounds
        match &mgr_err {
            ManagerError::TensorBounds(inner) => {
                assert_eq!(inner, &k3_err);
            }
            ManagerError::Affinity(_) => {
                panic!("non-affinity error incorrectly routed to ManagerError::Affinity: {k3_err:?}");
            }
            other => {
                panic!("expected ManagerError::TensorBounds, got {other:?}");
            }
        }

        // 2. Display formatting
        let display = format!("{mgr_err}");
        assert!(
            display.contains("[k3-manager tensor bounds error]"),
            "Display missing tensor bounds prefix: {display}"
        );
        assert!(
            display.contains(k3_err.diagnostic_code()),
            "Display missing inner diagnostic code: {display}"
        );

        // 3. Source error inspection & downcasting
        let source = mgr_err.source().expect("source() must be Some");
        let downcasted = source
            .downcast_ref::<K3Error>()
            .expect("downcast_ref to K3Error must succeed");
        assert_eq!(downcasted, &k3_err);
    }
}

// ============================================================================
// 2. WorkerPlacement Validation Empirical Challenges
// ============================================================================

#[test]
fn test_empirical_worker_placement_cleanly_accepts_valid_compute_subsets() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res.clone());

    // In default resolution:
    // active_control_mask = 0x00C0 (cores 6, 7)
    // spare_recovery_mask = 0x0020 (core 5)
    // x100_compute_cap_mask = 0x003F (cores 0..5, max 6 cores)
    assert_eq!(res.active_control_mask, DEFAULT_ACTIVE_CONTROL_MASK);
    assert_eq!(res.spare_recovery_mask, DEFAULT_SPARE_RECOVERY_MASK);
    assert_eq!(res.x100_compute_cap_mask, DEFAULT_X100_COMPUTE_CAP_MASK);

    // Test all 64 possible subsets of the compute cap mask (0x0000 to 0x003F)
    for mask in 0_u64..=0x003F {
        // Test hex with leading 0x
        let p_hex_lower = dummy_placement("worker_test", "x100", Some(&format!("0x{mask:04x}")));
        assert!(
            res.validate_worker_placement(&p_hex_lower).is_ok(),
            "failed on lower hex 0x{mask:04x}"
        );
        assert!(
            scheduler.validate_worker_placement(&p_hex_lower).is_ok(),
            "scheduler failed on lower hex 0x{mask:04x}"
        );

        // Test hex with leading 0X
        let p_hex_upper = dummy_placement("worker_test", "X100", Some(&format!("0X{mask:04X}")));
        assert!(
            res.validate_worker_placement(&p_hex_upper).is_ok(),
            "failed on upper hex 0X{mask:04X}"
        );
        assert!(
            scheduler.validate_worker_placement(&p_hex_upper).is_ok(),
            "scheduler failed on upper hex 0X{mask:04X}"
        );

        // Test decimal string
        let p_dec = dummy_placement("worker_test", "x100", Some(&format!("{mask}")));
        assert!(
            res.validate_worker_placement(&p_dec).is_ok(),
            "failed on decimal {mask}"
        );
        assert!(
            scheduler.validate_worker_placement(&p_dec).is_ok(),
            "scheduler failed on decimal {mask}"
        );
    }

    // Verify affinity None is cleanly accepted
    let p_none = dummy_placement("worker_none", "x100", None);
    assert!(res.validate_worker_placement(&p_none).is_ok());
    assert!(scheduler.validate_worker_placement(&p_none).is_ok());
}

#[test]
fn test_empirical_worker_placement_strictly_rejects_active_control_encroachment() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res.clone());

    // Active control cores are 6 and 7 (mask 0x00C0)
    // Individual encroaching masks:
    let rogue_masks = [
        (0x0040_u64, "core 6 alone"),
        (0x0080_u64, "core 7 alone"),
        (0x00C0_u64, "cores 6 and 7 together"),
        (0x0041_u64, "core 0 and core 6"),
        (0x0082_u64, "core 1 and core 7"),
        (0x00FF_u64, "all 8 cores"),
        (0x007F_u64, "cores 0..6 encroaching on 6"),
        (0x00BF_u64, "cores 0..5 and 7 encroaching on 7"),
    ];

    for (mask, label) in rogue_masks {
        let p = dummy_placement("encroacher_model", "x100", Some(&format!("0x{mask:04X}")));

        // Test via extension trait
        let res_err = res.validate_worker_placement(&p);
        assert!(
            res_err.is_err(),
            "Expected error for {label} (0x{mask:04X}), got Ok"
        );
        let err = res_err.unwrap_err();

        // Must be ManagerError::Affinity
        match err {
            ManagerError::Affinity(K3Error::AffinityError {
                requested_mask,
                requested_threads,
                reason,
            }) => {
                assert_eq!(requested_mask, mask);
                assert_eq!(requested_threads, mask.count_ones() as usize);
                assert!(
                    reason.contains("encroaches on dedicated active control cores 0x00C0"),
                    "Reason missing expected encroachment message: {reason}"
                );
                assert!(
                    reason.contains("encroacher_model"),
                    "Reason missing model ID: {reason}"
                );
                let expected_intersection = mask & DEFAULT_ACTIVE_CONTROL_MASK;
                assert!(
                    reason.contains(&format!("intersection: 0x{expected_intersection:04X}")),
                    "Reason missing intersection: {reason}"
                );
            }
            other => panic!("expected ManagerError::Affinity for {label}, got {other:?}"),
        }

        // Test via scheduler
        let sched_err = scheduler.validate_worker_placement(&p);
        assert!(sched_err.is_err());
        assert!(matches!(sched_err.unwrap_err(), ManagerError::Affinity(_)));
    }
}

#[test]
fn test_empirical_worker_placement_exhaustive_encroachment_fuzzer() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");

    // Every combination where (mask & 0x00C0) != 0 must be rejected as encroachment
    // Sample combinations across compute mask (0..=0x003F) combined with encroachment bits (1..=3) << 6
    for compute_subset in 0_u64..=0x003F {
        for control_sub in 1_u64..=3 {
            let mask = compute_subset | (control_sub << 6);
            assert_ne!(mask & DEFAULT_ACTIVE_CONTROL_MASK, 0);

            let p = dummy_placement("fuzz_worker", "x100", Some(&format!("0x{mask:04X}")));
            let err = res.validate_worker_placement(&p).unwrap_err();
            match err {
                ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask,
                    requested_threads,
                    reason,
                }) => {
                    assert_eq!(requested_mask, mask);
                    assert_eq!(requested_threads, mask.count_ones() as usize);
                    assert!(reason.contains("encroaches on dedicated active control cores"));
                }
                other => panic!("expected ManagerError::Affinity, got {other:?}"),
            }
        }
    }
}

#[test]
fn test_empirical_worker_placement_rejects_exceeding_compute_cap_without_encroaching() {
    // Construct a custom configuration where compute cap is narrower than non-control cores:
    // e.g. active control mask = 0x00C0 (cores 6, 7), spare = 0x0020 (core 5),
    // but custom compute cap = 0x000F (cores 0..3 only).
    // Cores 4 and 5 (mask 0x0030) are NOT active control cores, but EXCEED compute cap!
    let custom_cfg = ControlPlaneCapacityConfig::new()
        .with_custom_active_control_mask(0x00C0)
        .with_custom_spare_recovery_mask(0x0020)
        .with_custom_x100_compute_cap_mask(0x000F);

    let res = custom_cfg
        .resolve_with_online_mask(0x00FF)
        .expect("resolve custom config");
    let scheduler = ControlPlaneScheduler::new(res.clone());

    assert_eq!(res.active_control_mask, 0x00C0);
    assert_eq!(res.x100_compute_cap_mask, 0x000F);

    // 1. Valid placement within custom compute cap (0x000F)
    let p_valid = dummy_placement("custom_model", "x100", Some("0x0007"));
    assert!(res.validate_worker_placement(&p_valid).is_ok());
    assert!(scheduler.validate_worker_placement(&p_valid).is_ok());

    // 2. Cores 4 and 5 do not encroach on 0x00C0, but exceed compute cap 0x000F
    let excess_cases = [
        (0x0010_u64, "core 4 alone"),
        (0x0020_u64, "core 5 alone"),
        (0x0030_u64, "cores 4 and 5"),
        (0x001F_u64, "cores 0..4 (exceeding cap by core 4)"),
        (0x003F_u64, "cores 0..5 (exceeding cap by cores 4 and 5)"),
    ];

    for (mask, label) in excess_cases {
        let p = dummy_placement("exceed_model", "x100", Some(&format!("0x{mask:04X}")));

        // Extension trait
        let res_err = res.validate_worker_placement(&p);
        assert!(res_err.is_err(), "Expected error for {label}, got Ok");
        let err = res_err.unwrap_err();

        match err {
            ManagerError::Affinity(K3Error::AffinityError {
                requested_mask,
                requested_threads,
                reason,
            }) => {
                assert_eq!(requested_mask, mask);
                assert_eq!(requested_threads, mask.count_ones() as usize);
                assert!(
                    reason.contains("exceeds X100 compute cap mask 0x000F"),
                    "Reason missing compute cap message: {reason}"
                );
                assert!(
                    reason.contains("exceed_model"),
                    "Reason missing model ID: {reason}"
                );
            }
            other => panic!("expected ManagerError::Affinity, got {other:?}"),
        }

        // Scheduler method
        let sched_err = scheduler.validate_worker_placement(&p);
        assert!(sched_err.is_err());
        assert!(matches!(sched_err.unwrap_err(), ManagerError::Affinity(_)));
    }
}

#[test]
fn test_empirical_worker_placement_constrained_topology() {
    // 4-core minimal topology (online: 0x000F)
    // Minimal config: 1 active control core (core 3, mask 0x0008), 0 spare cores, compute cap 0x0007
    let min_cfg = ControlPlaneCapacityConfig::minimal();
    let res = min_cfg
        .resolve_with_online_mask(0x000F)
        .expect("resolve 4-core minimal");
    let scheduler = ControlPlaneScheduler::new(res.clone());

    assert_eq!(res.active_control_mask, 0x0008);
    assert_eq!(res.x100_compute_cap_mask, 0x0007);

    // 1. Valid compute placements on cores 0..2 (subsets of 0x0007)
    for mask in 0_u64..=0x0007 {
        let p = dummy_placement("min_worker", "x100", Some(&format!("0x{mask:04X}")));
        assert!(res.validate_worker_placement(&p).is_ok());
        assert!(scheduler.validate_worker_placement(&p).is_ok());
    }

    // 2. Encroachment on core 3 (0x0008)
    let p_encroach = dummy_placement("min_encroach", "x100", Some("0x0008"));
    let err = res.validate_worker_placement(&p_encroach).unwrap_err();
    match err {
        ManagerError::Affinity(K3Error::AffinityError {
            requested_mask,
            requested_threads,
            reason,
        }) => {
            assert_eq!(requested_mask, 0x0008);
            assert_eq!(requested_threads, 1);
            assert!(reason.contains("encroaches on dedicated active control cores 0x0008"));
        }
        other => panic!("expected ManagerError::Affinity, got {other:?}"),
    }
}

#[test]
fn test_empirical_worker_placement_a100_cluster_isolation() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res.clone());

    // Placements targeting A100 must not be constrained by X100 control cores or X100 compute cap
    let a100_masks = [
        "0xFF00", // Full A100 cluster
        "0x0100", // Core 8
        "0x8000", // Core 15
        "0x5500", // Even A100 cores
        "0xAA00", // Odd A100 cores
    ];

    for mask_str in a100_masks {
        let p_lower = dummy_placement("a100_worker", "a100", Some(mask_str));
        assert!(
            res.validate_worker_placement(&p_lower).is_ok(),
            "A100 placement rejected: {mask_str}"
        );
        assert!(
            scheduler.validate_worker_placement(&p_lower).is_ok(),
            "A100 placement rejected by scheduler: {mask_str}"
        );

        let p_upper = dummy_placement("a100_worker", "A100", Some(mask_str));
        assert!(res.validate_worker_placement(&p_upper).is_ok());
        assert!(scheduler.validate_worker_placement(&p_upper).is_ok());
    }

    assert_eq!(A100_CLUSTER_MASK, 0xFF00);
    assert_eq!(X100_CLUSTER_MASK, 0x00FF);
}

#[test]
fn test_empirical_worker_placement_invalid_affinity_strings_yield_worker_spawn_error() {
    let cfg = ControlPlaneCapacityConfig::default();
    let res = cfg.resolve_default().expect("default resolution");
    let scheduler = ControlPlaneScheduler::new(res.clone());

    let invalid_strings = [
        ("", "empty string"),
        ("0x", "bare prefix 0x"),
        ("0X", "bare prefix 0X"),
        ("0xZZ", "invalid hex chars"),
        ("0x12G", "invalid hex digit G"),
        ("-1", "negative decimal"),
        ("-100", "negative integer"),
        ("0x-1", "negative hex"),
        ("abc", "unprefixed text"),
        ("cores 0-5", "human range notation"),
        ("0x 1", "space inside hex"),
        ("0x10000000000000000", "u64 overflow hex (65 bits)"),
        ("18446744073709551616", "u64 overflow decimal (2^64)"),
    ];

    for (invalid_str, label) in invalid_strings {
        let p = dummy_placement("bad_affinity_worker", "x100", Some(invalid_str));

        // Extension trait
        let res_err = res.validate_worker_placement(&p);
        assert!(
            res_err.is_err(),
            "Expected failure for {label} ('{invalid_str}')"
        );
        match res_err.unwrap_err() {
            ManagerError::WorkerSpawn(msg) => {
                assert!(
                    msg.contains("invalid hex affinity") || msg.contains("invalid integer affinity"),
                    "Unexpected WorkerSpawn message for {label}: {msg}"
                );
            }
            other => panic!("expected ManagerError::WorkerSpawn for {label}, got {other:?}"),
        }

        // Scheduler method
        let sched_err = scheduler.validate_worker_placement(&p);
        assert!(sched_err.is_err());
        assert!(matches!(
            sched_err.unwrap_err(),
            ManagerError::WorkerSpawn(_)
        ));
    }
}

#[test]
fn test_empirical_manager_error_source_and_display_coherence() {
    let aff_err = ManagerError::Affinity(K3Error::AffinityError {
        requested_mask: 0x00C0,
        requested_threads: 2,
        reason: "active control collision".to_string(),
    });
    assert_eq!(
        format!("{aff_err}"),
        "[k3-manager affinity error] [affinity_error] invalid affinity configuration (mask: 0x00C0, threads: 2): active control collision"
    );
    assert!(aff_err.source().is_some());

    let tb_err = ManagerError::TensorBounds(K3Error::TensorBoundsError {
        tensor_name: "tok_embeddings.weight".to_string(),
        offset: 0,
        byte_len: 256,
        buffer_len: 128,
        reason: "out of range".to_string(),
    });
    assert_eq!(
        format!("{tb_err}"),
        "[k3-manager tensor bounds error] [tensor_bounds_error] tensor 'tok_embeddings.weight' (offset: 0, len: 256, buffer_len: 128): out of range"
    );
    assert!(tb_err.source().is_some());

    let io_err = ManagerError::Io(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        "permission denied",
    ));
    assert!(format!("{io_err}").contains("[k3-manager io error]"));
    assert!(io_err.source().is_some());

    let spawn_err = ManagerError::WorkerSpawn("binary not found".to_string());
    assert_eq!(
        format!("{spawn_err}"),
        "[k3-manager worker spawn error] binary not found"
    );
    assert!(spawn_err.source().is_none());
}
