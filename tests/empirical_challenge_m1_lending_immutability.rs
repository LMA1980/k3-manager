//! Empirical Adversarial Challenge Test Suite for Milestone M1 (US-08 through US-11).
//!
//! Evaluates:
//! 1. Challenge 1: Opt-in idle core lending with asymmetric configurations and zero core leakage.
//! 2. Challenge 2: Instant reclaim under concurrent race conditions and borrower crash/failure.
//! 3. Challenge 3: Operator tuning authority immutability across 1,000 passes with wild latency variances (1ms–10,000ms)
//!    and non-mutating on-demand advisory recommendations.

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::manual_range_contains,
    clippy::assert_is_empty
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use k3_core::affinity::{
    ControlPlaneCapacityConfig, DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_SPARE_RECOVERY_MASK,
    ResolvedControlCapacity,
};
use k3_core::profile::{CostProfile, PriorityClass, ProfileStatus, TuningProfile};
use k3_manager::error::ManagerError;
use k3_manager::scheduler::{MultiModelScheduler, ScheduledPass};

fn setup_capacity() -> ResolvedControlCapacity {
    ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("default capacity resolution succeeds")
}

// ============================================================================
// CHALLENGE 1: Opt-In Idle Core Lending & Zero Core Leakage
// ============================================================================

#[test]
fn test_challenge1a_asymmetric_opt_in_matrix_zero_leakage() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    // Matrix of (owner_lend, borrower_borrow)
    let test_matrix = [
        (false, false, false, "both disabled"),
        (true, false, false, "owner opts in, borrower opts out"),
        (false, true, false, "owner opts out, borrower opts in"),
        (true, true, true, "both opted in"),
    ];

    for (owner_lend, borrower_borrow, expect_borrow, desc) in test_matrix {
        let owner_id = format!("owner-{owner_lend}-{borrower_borrow}");
        let borrower_id = format!("borrower-{owner_lend}-{borrower_borrow}");

        let mut owner_profile = TuningProfile::conservative_default();
        owner_profile.profile_status = ProfileStatus::Profiled;
        owner_profile.unprofiled = false;
        owner_profile.lend_idle_cores = owner_lend;
        owner_profile.borrow_idle_cores = false;

        let mut borrower_profile = TuningProfile::conservative_default();
        borrower_profile.profile_status = ProfileStatus::Profiled;
        borrower_profile.unprofiled = false;
        borrower_profile.lend_idle_cores = false;
        borrower_profile.borrow_idle_cores = borrower_borrow;

        scheduler
            .admit_or_register_model(&owner_id, "x100".to_owned(), 0x0003, Some(&owner_profile))
            .unwrap_or_else(|_| panic!("register owner for {desc}"));
        scheduler
            .admit_or_register_model(
                &borrower_id,
                "x100".to_owned(),
                0x000C,
                Some(&borrower_profile),
            )
            .unwrap_or_else(|_| panic!("register borrower for {desc}"));

        // Owner is idle
        assert!(
            scheduler.get_model_entry(&owner_id).unwrap().is_idle(),
            "owner must be idle for {desc}"
        );

        // Borrower requests forward pass
        let pass = scheduler
            .try_schedule_forward_pass(&borrower_id)
            .unwrap()
            .unwrap_or_else(|| panic!("pass must be granted for {desc}"));

        assert_eq!(pass.allocated_mask, 0x000C);
        if expect_borrow {
            assert_eq!(
                pass.borrowed_mask, 0x0003,
                "must borrow cores 0-1 when both opted in: {desc}"
            );
            assert_eq!(pass.effective_mask, 0x000F, "effective mask for {desc}");
            assert_eq!(pass.borrowed_from, vec![owner_id.clone()]);
            assert_eq!(scheduler.running_cores_mask(), 0x000F);

            let owner_mid = scheduler.get_model_entry(&owner_id).unwrap();
            assert_eq!(
                owner_mid.active_lent_mask, 0x0003,
                "owner active_lent_mask must track lent cores during pass: {desc}"
            );
        } else {
            assert_eq!(
                pass.borrowed_mask, 0,
                "must NOT borrow cores under asymmetric opt-in: {desc}"
            );
            assert_eq!(pass.effective_mask, 0x000C, "effective mask for {desc}");
            assert!(
                pass.borrowed_from.is_empty(),
                "borrowed_from must be empty for {desc}"
            );
            assert_eq!(scheduler.running_cores_mask(), 0x000C);

            let owner_mid = scheduler.get_model_entry(&owner_id).unwrap();
            assert_eq!(
                owner_mid.active_lent_mask, 0,
                "owner active_lent_mask must be 0 for {desc}"
            );
        }

        // Complete pass
        scheduler.record_pass_completion(&pass, 10.0, 5.0, 8);

        // ZERO CORE LEAKAGE ASSERTIONS
        let owner_after = scheduler.get_model_entry(&owner_id).unwrap();
        let borrower_after = scheduler.get_model_entry(&borrower_id).unwrap();

        assert_eq!(
            owner_after.active_lent_mask, 0,
            "owner active_lent_mask must return to 0 after pass: {desc}"
        );
        assert_eq!(
            owner_after.active_running_mask, 0,
            "owner active_running_mask must be 0: {desc}"
        );
        assert_eq!(
            borrower_after.active_borrowed_mask, 0,
            "borrower active_borrowed_mask must return to 0: {desc}"
        );
        assert_eq!(
            borrower_after.active_running_mask, 0,
            "borrower active_running_mask must return to 0: {desc}"
        );
        assert!(
            borrower_after.borrowed_from.is_empty(),
            "borrower borrowed_from must be empty: {desc}"
        );

        // Clean up scheduler models
        scheduler.unregister_model(&owner_id);
        scheduler.unregister_model(&borrower_id);

        assert_eq!(
            scheduler.running_cores_mask(),
            0,
            "system running cores must be 0 (zero core leakage) for {desc}"
        );
    }
}

#[test]
fn test_challenge1b_multi_peer_selective_lending_and_parallelism() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    // 3 active models:
    // Model A: cores 0-1 (0x0003), lend_idle_cores = true, borrow = false
    // Model B: cores 2-3 (0x000C), lend_idle_cores = false, borrow = false
    // Model C: cores 4-5 (0x0030), lend_idle_cores = false, borrow_idle_cores = true

    let mut profile_a = TuningProfile::conservative_default();
    profile_a.profile_status = ProfileStatus::Profiled;
    profile_a.unprofiled = false;
    profile_a.lend_idle_cores = true;

    let mut profile_b = TuningProfile::conservative_default();
    profile_b.profile_status = ProfileStatus::Profiled;
    profile_b.unprofiled = false;
    profile_b.lend_idle_cores = false;

    let mut profile_c = TuningProfile::conservative_default();
    profile_c.profile_status = ProfileStatus::Profiled;
    profile_c.unprofiled = false;
    profile_c.borrow_idle_cores = true;

    scheduler
        .admit_or_register_model("model-a", "x100".to_owned(), 0x0003, Some(&profile_a))
        .expect("register A");
    scheduler
        .admit_or_register_model("model-b", "x100".to_owned(), 0x000C, Some(&profile_b))
        .expect("register B");
    scheduler
        .admit_or_register_model("model-c", "x100".to_owned(), 0x0030, Some(&profile_c))
        .expect("register C");

    // Both A and B are idle. C requests forward pass.
    let pass_c = scheduler
        .try_schedule_forward_pass("model-c")
        .unwrap()
        .expect("pass C granted");

    // Model C must borrow ONLY Model A's cores (0x0003), NOT Model B's cores (0x000C)
    assert_eq!(pass_c.allocated_mask, 0x0030);
    assert_eq!(
        pass_c.borrowed_mask, 0x0003,
        "must borrow only Model A's cores"
    );
    assert_eq!(
        pass_c.borrowed_mask & 0x000C,
        0,
        "must NEVER borrow Model B's non-opted cores"
    );
    assert_eq!(pass_c.effective_mask, 0x0033);
    assert_eq!(pass_c.borrowed_from, vec!["model-a".to_owned()]);

    let entry_b = scheduler.get_model_entry("model-b").unwrap();
    assert_eq!(
        entry_b.active_lent_mask, 0,
        "model B must not have any lent cores"
    );

    // PARALLEL EXECUTION: While C is running on 0x0033 (cores 0, 1, 4, 5),
    // Model B must be able to schedule on cores 2-3 (0x000C) concurrently!
    let pass_b = scheduler
        .try_schedule_forward_pass("model-b")
        .unwrap()
        .expect("model B must run concurrently on disjoint cores");

    assert_eq!(pass_b.allocated_mask, 0x000C);
    assert_eq!(pass_b.borrowed_mask, 0);
    assert_eq!(pass_b.effective_mask, 0x000C);

    // Total running cores must be 0x0033 | 0x000C = 0x003F (cores 0-5)
    assert_eq!(scheduler.running_cores_mask(), 0x003F);

    // Complete B
    scheduler.record_pass_completion(&pass_b, 12.0, 4.0, 5);
    assert_eq!(scheduler.running_cores_mask(), 0x0033);

    // Complete C
    scheduler.record_pass_completion(&pass_c, 24.0, 8.0, 10);
    assert_eq!(scheduler.running_cores_mask(), 0x0000);

    // Check all models for zero leakage
    for m in &["model-a", "model-b", "model-c"] {
        let e = scheduler.get_model_entry(m).unwrap();
        assert_eq!(e.active_lent_mask, 0, "{m} active_lent_mask");
        assert_eq!(e.active_borrowed_mask, 0, "{m} active_borrowed_mask");
        assert_eq!(e.active_running_mask, 0, "{m} active_running_mask");
        assert!(e.borrowed_from.is_empty(), "{m} borrowed_from");
    }
}

#[test]
fn test_challenge1c_unprofiled_and_control_plane_isolation_under_lending() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    // Model Unprofiled (cores 0-1: 0x0003) running default unprofiled
    scheduler
        .admit_or_register_model("unprofiled-model", "x100".to_owned(), 0x0003, None)
        .expect("register unprofiled");

    // Profiled Borrower (cores 2-3: 0x000C) with borrow_idle_cores = true
    let mut borrower_profile = TuningProfile::conservative_default();
    borrower_profile.profile_status = ProfileStatus::Profiled;
    borrower_profile.unprofiled = false;
    borrower_profile.borrow_idle_cores = true;

    scheduler
        .admit_or_register_model(
            "borrower-model",
            "x100".to_owned(),
            0x000C,
            Some(&borrower_profile),
        )
        .expect("register borrower");

    // Even though unprofiled model is idle, borrower MUST NOT borrow its cores
    let pass = scheduler
        .try_schedule_forward_pass("borrower-model")
        .unwrap()
        .expect("borrower pass granted");

    assert_eq!(
        pass.borrowed_mask, 0,
        "borrower must not borrow from unprofiled model"
    );
    assert_eq!(pass.effective_mask, 0x000C);
    assert!(pass.borrowed_from.is_empty());

    // Control Plane Isolation Guard:
    // Effective mask must NEVER touch active control mask (0x00C0) or spare recovery mask (0x0020)
    assert_eq!(
        pass.effective_mask & (DEFAULT_ACTIVE_CONTROL_MASK | DEFAULT_SPARE_RECOVERY_MASK),
        0,
        "must never touch control cores"
    );

    scheduler.record_pass_completion(&pass, 15.0, 5.0, 8);
    assert_eq!(scheduler.running_cores_mask(), 0);
}

// ============================================================================
// CHALLENGE 2: Instant Reclaim Under Race Conditions
// ============================================================================

#[test]
fn test_challenge2a_concurrent_race_condition_instant_reclaim() {
    let capacity = setup_capacity();
    let scheduler = Arc::new(MultiModelScheduler::new(capacity));

    let owner_id = "owner-race";
    let borrower_id = "borrower-race";

    let mut owner_profile = TuningProfile::conservative_default();
    owner_profile.profile_status = ProfileStatus::Profiled;
    owner_profile.unprofiled = false;
    owner_profile.lend_idle_cores = true;

    let mut borrower_profile = TuningProfile::conservative_default();
    borrower_profile.profile_status = ProfileStatus::Profiled;
    borrower_profile.unprofiled = false;
    borrower_profile.borrow_idle_cores = true;

    scheduler
        .admit_or_register_model(owner_id, "x100".to_owned(), 0x0003, Some(&owner_profile))
        .expect("register owner");
    scheduler
        .admit_or_register_model(
            borrower_id,
            "x100".to_owned(),
            0x000C,
            Some(&borrower_profile),
        )
        .expect("register borrower");

    const ROUNDS: u32 = 100;
    let stop_signal = Arc::new(AtomicBool::new(false));
    let borrower_passes_completed = Arc::new(AtomicU32::new(0));
    let owner_passes_completed = Arc::new(AtomicU32::new(0));
    let overlap_conflicts = Arc::new(AtomicU32::new(0));

    // Shared state tracking currently executing masks to detect any physical core conflict
    let active_borrower_mask = Arc::new(AtomicU32::new(0));
    let active_owner_mask = Arc::new(AtomicU32::new(0));

    // Thread 1: Borrower worker
    let s_borrower = Arc::clone(&scheduler);
    let stop_b = Arc::clone(&stop_signal);
    let b_count = Arc::clone(&borrower_passes_completed);
    let b_mask = Arc::clone(&active_borrower_mask);
    let o_mask = Arc::clone(&active_owner_mask);
    let conflicts_b = Arc::clone(&overlap_conflicts);

    let borrower_handle = thread::spawn(move || {
        while !stop_b.load(Ordering::Acquire) {
            match s_borrower.schedule_forward_pass(borrower_id, Some(Duration::from_millis(50))) {
                Ok(pass) => {
                    let eff = pass.effective_mask as u32;
                    b_mask.store(eff, Ordering::Release);

                    // Check for physical core overlap conflict with active owner
                    let current_owner = o_mask.load(Ordering::Acquire);
                    if (eff & current_owner) != 0 {
                        conflicts_b.fetch_add(1, Ordering::SeqCst);
                    }

                    // Simulate forward-pass compute latency (50-200 microseconds)
                    thread::yield_now();

                    b_mask.store(0, Ordering::Release);
                    s_borrower.record_pass_completion(&pass, 2.0, 1.0, 4);
                    b_count.fetch_add(1, Ordering::Relaxed);
                }
                Err(ManagerError::WorkerUnresponsive { .. }) => {
                    // Timeout is acceptable under heavy lock contention
                    thread::yield_now();
                }
                Err(e) => panic!("unexpected scheduler error for borrower: {e:?}"),
            }
        }
    });

    // Thread 2: Owner traffic generator & instant reclaim assertor
    let s_owner = Arc::clone(&scheduler);
    let o_count = Arc::clone(&owner_passes_completed);
    let b_mask_o = Arc::clone(&active_borrower_mask);
    let o_mask_o = Arc::clone(&active_owner_mask);
    let conflicts_o = Arc::clone(&overlap_conflicts);

    let owner_handle = thread::spawn(move || {
        for _ in 0..ROUNDS {
            // Owner schedules pass
            match s_owner.schedule_forward_pass(owner_id, Some(Duration::from_millis(500))) {
                Ok(pass) => {
                    let eff = pass.effective_mask as u32;
                    o_mask_o.store(eff, Ordering::Release);

                    // INSTANT RECLAIM INVARIANT: Owner must have acquired its FULL base reservation
                    assert_eq!(
                        pass.allocated_mask, 0x0003,
                        "owner must have allocated its full reservation"
                    );
                    assert_eq!(
                        pass.borrowed_mask, 0,
                        "owner must not have borrowed any cores"
                    );

                    // Verify zero overlap with active borrower execution
                    let current_borrower = b_mask_o.load(Ordering::Acquire);
                    if (eff & current_borrower) != 0 {
                        conflicts_o.fetch_add(1, Ordering::SeqCst);
                    }

                    thread::yield_now();

                    o_mask_o.store(0, Ordering::Release);
                    s_owner.record_pass_completion(&pass, 3.0, 1.0, 4);
                    o_count.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => panic!("unexpected owner schedule error: {e:?}"),
            }
        }
    });

    owner_handle.join().expect("owner thread joins cleanly");
    stop_signal.store(true, Ordering::Release);
    borrower_handle
        .join()
        .expect("borrower thread joins cleanly");

    // ASSERTIONS
    assert_eq!(
        overlap_conflicts.load(Ordering::SeqCst),
        0,
        "ZERO core overlap conflicts allowed during race conditions"
    );
    assert_eq!(
        owner_passes_completed.load(Ordering::SeqCst),
        ROUNDS,
        "all {ROUNDS} owner passes must complete"
    );
    assert!(
        borrower_passes_completed.load(Ordering::SeqCst) > 0,
        "borrower must have completed passes during race conditions"
    );

    // ZERO CORE LEAKAGE
    assert_eq!(
        scheduler.running_cores_mask(),
        0,
        "zero core leakage after {ROUNDS} race condition rounds"
    );
    let owner_entry = scheduler.get_model_entry(owner_id).unwrap();
    let borrower_entry = scheduler.get_model_entry(borrower_id).unwrap();
    assert_eq!(owner_entry.active_lent_mask, 0);
    assert_eq!(owner_entry.active_running_mask, 0);
    assert_eq!(borrower_entry.active_borrowed_mask, 0);
    assert_eq!(borrower_entry.active_running_mask, 0);
}

#[test]
fn test_challenge2b_single_pass_guarantee_no_retention() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let owner_id = "owner-single-pass";
    let borrower_id = "borrower-single-pass";

    let mut owner_profile = TuningProfile::conservative_default();
    owner_profile.profile_status = ProfileStatus::Profiled;
    owner_profile.unprofiled = false;
    owner_profile.lend_idle_cores = true;

    let mut borrower_profile = TuningProfile::conservative_default();
    borrower_profile.profile_status = ProfileStatus::Profiled;
    borrower_profile.unprofiled = false;
    borrower_profile.borrow_idle_cores = true;

    scheduler
        .admit_or_register_model(owner_id, "x100".to_owned(), 0x0003, Some(&owner_profile))
        .expect("register owner");
    scheduler
        .admit_or_register_model(
            borrower_id,
            "x100".to_owned(),
            0x000C,
            Some(&borrower_profile),
        )
        .expect("register borrower");

    // 1. Borrower schedules pass, borrowing Owner's cores 0x0003
    let pass_b1 = scheduler
        .try_schedule_forward_pass(borrower_id)
        .unwrap()
        .expect("borrower pass 1 granted");
    assert_eq!(pass_b1.borrowed_mask, 0x0003);
    assert_eq!(pass_b1.effective_mask, 0x000F);

    // 2. MID-PASS: Owner receives traffic
    scheduler
        .inc_pending_requests(owner_id)
        .expect("owner traffic arrives");
    assert_eq!(
        scheduler
            .get_model_entry(owner_id)
            .unwrap()
            .pending_requests_count,
        1
    );

    // 3. Borrower pass 1 finishes
    scheduler.record_pass_completion(&pass_b1, 15.0, 5.0, 8);

    // 4. In the immediate next step, Borrower requests pass 2 while Owner has pending traffic!
    // Borrower must NOT retain or borrow Owner's cores!
    let pass_b2 = scheduler
        .try_schedule_forward_pass(borrower_id)
        .unwrap()
        .expect("borrower pass 2 granted");

    assert_eq!(
        pass_b2.borrowed_mask, 0,
        "borrower must NEVER retain borrowed cores when owner has pending traffic"
    );
    assert_eq!(pass_b2.allocated_mask, 0x000C);
    assert_eq!(pass_b2.effective_mask, 0x000C);

    // 5. Owner immediately schedules its pass with full base reservation
    let pass_owner = scheduler
        .try_schedule_forward_pass(owner_id)
        .unwrap()
        .expect("owner must immediately reclaim cores");

    assert_eq!(pass_owner.allocated_mask, 0x0003);
    assert_eq!(pass_owner.borrowed_mask, 0);
    assert_eq!(pass_owner.effective_mask, 0x0003);

    // Both running in parallel without core conflict
    assert_eq!(scheduler.running_cores_mask(), 0x000F);

    scheduler.record_pass_completion(&pass_b2, 10.0, 4.0, 6);
    scheduler.record_pass_completion(&pass_owner, 8.0, 2.0, 4);
    assert_eq!(scheduler.running_cores_mask(), 0x0000);
}

#[test]
fn test_challenge2c_instant_reclaim_on_borrower_failure_or_crash() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let owner_id = "owner-failover";
    let borrower_id = "borrower-failover";

    let mut owner_profile = TuningProfile::conservative_default();
    owner_profile.profile_status = ProfileStatus::Profiled;
    owner_profile.unprofiled = false;
    owner_profile.lend_idle_cores = true;

    let mut borrower_profile = TuningProfile::conservative_default();
    borrower_profile.profile_status = ProfileStatus::Profiled;
    borrower_profile.unprofiled = false;
    borrower_profile.borrow_idle_cores = true;

    scheduler
        .admit_or_register_model(owner_id, "x100".to_owned(), 0x0003, Some(&owner_profile))
        .expect("register owner");
    scheduler
        .admit_or_register_model(
            borrower_id,
            "x100".to_owned(),
            0x000C,
            Some(&borrower_profile),
        )
        .expect("register borrower");

    // Case A: Borrower suffers pass failure (timeout / I/O error) while borrowing
    let pass_1 = scheduler
        .try_schedule_forward_pass(borrower_id)
        .unwrap()
        .expect("pass granted");
    assert_eq!(pass_1.borrowed_mask, 0x0003);

    // Owner has pending request
    scheduler.inc_pending_requests(owner_id).unwrap();

    // Borrower pass fails
    scheduler.record_pass_failure(&pass_1);

    // Owner must immediately reclaim full cores
    let pass_owner_1 = scheduler
        .try_schedule_forward_pass(owner_id)
        .unwrap()
        .expect("owner must reclaim immediately after borrower pass failure");
    assert_eq!(pass_owner_1.allocated_mask, 0x0003);
    scheduler.record_pass_completion(&pass_owner_1, 5.0, 2.0, 4);
    assert_eq!(scheduler.running_cores_mask(), 0);

    // Case B: Borrower crashes unexpectedly (SIGSEGV / process termination) while borrowing
    let pass_2 = scheduler
        .try_schedule_forward_pass(borrower_id)
        .unwrap()
        .expect("pass granted");
    assert_eq!(pass_2.borrowed_mask, 0x0003);

    scheduler.inc_pending_requests(owner_id).unwrap();

    // Worker crash event notification
    scheduler.on_worker_crashed(borrower_id);

    // Owner must immediately reclaim full cores
    let pass_owner_2 = scheduler
        .try_schedule_forward_pass(owner_id)
        .unwrap()
        .expect("owner must reclaim immediately after borrower crash");
    assert_eq!(pass_owner_2.allocated_mask, 0x0003);
    scheduler.record_pass_completion(&pass_owner_2, 6.0, 2.0, 4);
    assert_eq!(scheduler.running_cores_mask(), 0);
}

// ============================================================================
// CHALLENGE 3: Operator Tuning Authority Immutability (1,000 Passes) & Advisories
// ============================================================================

#[test]
fn test_challenge3a_1000_passes_wild_latency_profile_byte_identity() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let model_id = "authoritative-model-1000";

    // Build authoritative operator profile
    let mut initial_profile = TuningProfile::conservative_default();
    initial_profile.model_id = Some(model_id.to_owned());
    initial_profile.cluster = "x100".to_owned();
    initial_profile.x100_cores = 4;
    initial_profile.a100_cores = 0;
    initial_profile.priority_class = PriorityClass::Latency;
    initial_profile.n_ctx = 4096;
    initial_profile.lend_idle_cores = false;
    initial_profile.borrow_idle_cores = false;
    initial_profile.profile_status = ProfileStatus::Profiled;
    initial_profile.unprofiled = false;
    initial_profile.advisory_notice = None;

    // Serialize to canonical JSON bytes before any passes
    let initial_json_bytes = serde_json::to_vec(&initial_profile)
        .expect("initial profile serializes cleanly to JSON bytes");

    scheduler
        .admit_or_register_model(model_id, "x100".to_owned(), 0x000F, Some(&initial_profile))
        .expect("register model succeeds");

    // Execute 1,000 forward passes with wild latency variances from 1ms to 10,000ms
    for i in 1..=1000_u32 {
        let pass = ScheduledPass {
            model_id: model_id.to_owned(),
            allocated_mask: 0x000F,
            borrowed_mask: 0,
            effective_mask: 0x000F,
            borrowed_from: Vec::new(),
            scheduled_at: Instant::now(),
        };

        // Deterministic chaotic variance generation:
        // Spans 1.0ms to 10,000.0ms using periodic nonlinear functions
        let cycle = f64::from(i % 100);
        let prefill_ms = 1.0 + (cycle * 10.0) + if i % 13 == 0 { 8900.0 } else { 0.0 };
        let decode_ms = 1.0 + ((99.0 - cycle) * 10.0) + if i % 17 == 0 { 8900.0 } else { 0.0 };

        assert!(
            prefill_ms >= 1.0 && prefill_ms <= 10_000.0,
            "prefill_ms must be within 1.0 to 10000.0 ms: {prefill_ms}"
        );
        assert!(
            decode_ms >= 1.0 && decode_ms <= 10_000.0,
            "decode_ms must be within 1.0 to 10000.0 ms: {decode_ms}"
        );

        let tokens = (i % 512) + 1;

        match i % 3 {
            0 => scheduler.record_pass_completion(&pass, prefill_ms, 0.0, tokens),
            1 => scheduler.record_pass_completion(&pass, 0.0, decode_ms, tokens),
            _ => scheduler.record_pass_completion(&pass, prefill_ms, decode_ms, tokens),
        }
    }

    // Retrieve scheduler entry and stored profile after 1,000 passes of wild variance
    let entry = scheduler
        .get_model_entry(model_id)
        .expect("entry must exist in scheduler");

    // 1. Assert internal metrics legitimately tracked all 1,000 passes
    assert_eq!(
        entry.metrics.total_passes, 1000,
        "all 1,000 passes must be tracked in metrics"
    );
    assert!(
        entry.metrics.avg_pass_cost_ms > 100.0,
        "avg pass cost must reflect wild variance: {}",
        entry.metrics.avg_pass_cost_ms
    );
    assert!(
        entry.metrics.virtual_runtime_ms > 10_000.0,
        "virtual runtime must have advanced: {}",
        entry.metrics.virtual_runtime_ms
    );
    assert!(
        entry.metrics.dynamic_scheduling_weight > 0.0,
        "dynamic scheduling weight must be positive"
    );

    // 2. Assert stored TuningProfile is 100% BYTE-FOR-BYTE IDENTICAL
    let stored_profile = entry
        .tuning_profile
        .as_ref()
        .expect("stored tuning profile must be present");

    let stored_json_bytes = serde_json::to_vec(stored_profile)
        .expect("stored profile serializes cleanly to JSON bytes");

    assert_eq!(
        stored_json_bytes, initial_json_bytes,
        "stored TuningProfile must be 100% byte-for-byte identical after 1,000 passes"
    );

    // 3. Assert struct field equality
    assert_eq!(*stored_profile, initial_profile);

    // 4. Assert model entry authoritative configuration fields
    assert_eq!(entry.cluster, "x100");
    assert_eq!(entry.reservation_mask, 0x000F);
    assert_eq!(entry.priority_class, PriorityClass::Latency);
    assert_eq!(entry.n_ctx, 4096);
    assert!(!entry.lend_idle_cores);
    assert!(!entry.borrow_idle_cores);
    assert!(!entry.unprofiled);

    // 5. Generate on-demand advisory after 1,000 passes
    let advisory = scheduler
        .suggest_tuning(model_id, None)
        .expect("suggest_tuning succeeds after 1,000 passes");

    assert!(
        advisory.rationale.contains("Observed 1000 passes"),
        "advisory rationale must report observed 1,000 passes: {}",
        advisory.rationale
    );
    assert!(
        advisory
            .operator_authority_notice
            .contains("preserves operator tuning authority"),
        "advisory notice must re-affirm operator tuning authority"
    );

    // 6. Assert stored TuningProfile remains 100% byte-for-byte identical EVEN AFTER ADVISORY GENERATION
    let entry_after_advisory = scheduler.get_model_entry(model_id).unwrap();
    let bytes_after_advisory = serde_json::to_vec(
        entry_after_advisory
            .tuning_profile
            .as_ref()
            .expect("profile still present"),
    )
    .unwrap();

    assert_eq!(
        bytes_after_advisory, initial_json_bytes,
        "stored TuningProfile must be 100% byte-for-byte identical after advisory generation"
    );
}

#[test]
fn test_challenge3b_advisory_recommendations_immutability_and_tradeoffs() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let model_id = "advisory-tradeoff-model";

    let mut profile = TuningProfile::conservative_default();
    profile.model_id = Some(model_id.to_owned());
    profile.cluster = "x100".to_owned();
    profile.priority_class = PriorityClass::Throughput;
    profile.profile_status = ProfileStatus::Profiled;
    profile.unprofiled = false;

    let original_bytes = serde_json::to_vec(&profile).unwrap();

    scheduler
        .admit_or_register_model(model_id, "x100".to_owned(), 0x0007, Some(&profile))
        .expect("register model");

    // Simulate 30 passes where prefill is heavy
    for _ in 0..30 {
        let pass = ScheduledPass {
            model_id: model_id.to_owned(),
            allocated_mask: 0x0007,
            borrowed_mask: 0,
            effective_mask: 0x0007,
            borrowed_from: Vec::new(),
            scheduled_at: Instant::now(),
        };
        scheduler.record_pass_completion(&pass, 400.0, 10.0, 64);
    }

    let mut baseline = CostProfile::unprofiled_default(k3_core::profile::ModelIdentity {
        name: "test-model".to_owned(),
        architecture: "llama".to_owned(),
        file_path: "/models/test.gguf".to_owned(),
        file_digest: "abcdef".to_owned(),
    });
    baseline.static_scheduling_weight = 2.5;

    let advisory = scheduler
        .suggest_tuning(model_id, Some(&baseline))
        .expect("suggest_tuning succeeds");

    assert_eq!(advisory.bottleneck_classification, "compute-bound");
    assert_eq!(advisory.recommended_cluster, "both");
    assert!(advisory.recommended_ime);
    assert!(
        advisory.rationale.contains("Baseline static weight: 2.50"),
        "rationale must contain baseline static weight: {}",
        advisory.rationale
    );

    // Profile must remain 100% byte-for-byte identical
    let stored = scheduler
        .get_model_entry(model_id)
        .unwrap()
        .tuning_profile
        .unwrap();
    assert_eq!(serde_json::to_vec(&stored).unwrap(), original_bytes);
}

#[test]
fn test_challenge3c_advisory_edge_cases() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    // 1. Model not found
    let err = scheduler
        .suggest_tuning("non-existent", None)
        .expect_err("non-existent model must return error");
    match err {
        ManagerError::ModelNotFound(id) => assert_eq!(id, "non-existent"),
        other => panic!("expected ModelNotFound, got {other:?}"),
    }

    // 2. Zero passes completed
    scheduler
        .admit_or_register_model("zero-pass-model", "x100".to_owned(), 0x0003, None)
        .expect("register model");

    let advisory = scheduler
        .suggest_tuning("zero-pass-model", None)
        .expect("suggest_tuning on zero-pass model succeeds without panic");

    assert!(advisory.rationale.contains("Observed 0 passes"));
    assert_eq!(advisory.bottleneck_classification, "compute-bound");
    assert!(
        advisory
            .recommendations
            .iter()
            .any(|r| r.contains("conservative defaults (X100, throughput, n_ctx=2048)"))
    );
}
