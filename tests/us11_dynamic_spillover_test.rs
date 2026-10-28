//! Integration tests for US-11 (#11): Dynamic Core Spillover & Reclaim: Opt-In Lending, Idle Reclaim, Pre-Pinned Worker Threads.
//!
//! # Core System Assertions
//! - **Scenario 1**: Default isolation — no core lending or borrowing when flags are disabled.
//!   Unless both the owning model (`lend_idle_cores == true`) and borrowing model (`borrow_idle_cores == true`)
//!   explicitly opt in, idle cores are never shared.
//! - **Scenario 2**: Dynamic core borrowing between opted-in profiled models.
//!   When the owning model is idle, a borrowing model expands its single-pass core allocation to include
//!   the idle model's cores without modifying base reservations.
//! - **Scenario 3**: Instant pass-boundary reclaim.
//!   As soon as the owning model receives an incoming request while its cores are borrowed, the lender's
//!   cores are instantly reclaimed at the immediate completion of the borrower's forward pass.
//! - **Scenario 4**: Pre-pinned worker thread work-stealing without runtime cpuset rewrites.
//!   Permits communicate dynamically via [`ScheduledPass::effective_mask`] on a per-pass basis without
//!   persistent kernel cpuset mutations.
//! - **Scenario 5**: Dedicated control plane isolation guard.
//!   Active control cores (`0x00C0`) and spare recovery core (`0x0020`) can NEVER be lent or borrowed.
//! - **Scenario 6**: Unprofiled models cannot enable lending or borrowing.
//!   Admitting an unprofiled model with spillover flags enabled is rejected upfront per ADR-030.

use std::sync::Arc;

use k3_core::affinity::{
    ControlPlaneCapacityConfig, DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_SPARE_RECOVERY_MASK,
    ResolvedControlCapacity,
};
use k3_core::profile::{ProfileStatus, TuningProfile};
use k3_manager::error::ManagerError;
use k3_manager::scheduler::MultiModelScheduler;

fn setup_capacity() -> ResolvedControlCapacity {
    ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("default capacity resolution succeeds")
}

#[test]
fn test_scenario1_default_isolation_no_lending_when_disabled() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let model_a = "model-a-idle";
    let model_b = "model-b-busy";

    // Default configuration: lend_idle_cores = false, borrow_idle_cores = false
    scheduler
        .admit_or_register_model(model_a, "x100".to_owned(), 0x0003, None)
        .expect("register model A");
    scheduler
        .admit_or_register_model(model_b, "x100".to_owned(), 0x000C, None)
        .expect("register model B");

    // Model A is completely idle
    assert!(scheduler.get_model_entry(model_a).unwrap().is_idle());

    // Model B acquires a pass: MUST NOT borrow from Model A because flags are false
    let pass = scheduler
        .try_schedule_forward_pass(model_b)
        .unwrap()
        .expect("pass B granted");

    assert_eq!(pass.allocated_mask, 0x000C);
    assert_eq!(pass.borrowed_mask, 0, "borrowed mask must be 0 by default");
    assert_eq!(pass.effective_mask, 0x000C);
    assert!(
        pass.borrowed_from.is_empty(),
        "borrowed_from must be empty by default"
    );

    scheduler.record_pass_completion(&pass, 15.0, 5.0, 10);
}

#[test]
fn test_scenario1_asymmetric_opt_in_no_lending() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    // Case 1: Lender opts in, but Borrower has NOT opted in
    let mut lender_profile = TuningProfile::conservative_default();
    lender_profile.profile_status = ProfileStatus::Profiled;
    lender_profile.unprofiled = false;
    lender_profile.lend_idle_cores = true;
    lender_profile.borrow_idle_cores = false;

    let mut borrower_profile = TuningProfile::conservative_default();
    borrower_profile.profile_status = ProfileStatus::Profiled;
    borrower_profile.unprofiled = false;
    borrower_profile.lend_idle_cores = false;
    borrower_profile.borrow_idle_cores = false;

    scheduler
        .admit_or_register_model("lender", "x100".to_owned(), 0x0003, Some(&lender_profile))
        .expect("register lender");
    scheduler
        .admit_or_register_model(
            "borrower",
            "x100".to_owned(),
            0x000C,
            Some(&borrower_profile),
        )
        .expect("register borrower");

    let pass = scheduler
        .try_schedule_forward_pass("borrower")
        .unwrap()
        .expect("pass granted");
    assert_eq!(pass.borrowed_mask, 0, "borrower did not opt in; no lending");
    scheduler.record_pass_completion(&pass, 10.0, 2.0, 5);

    // Case 2: Borrower opts in, but Lender has NOT opted in
    borrower_profile.borrow_idle_cores = true;
    lender_profile.lend_idle_cores = false;

    scheduler
        .admit_or_register_model("lender", "x100".to_owned(), 0x0003, Some(&lender_profile))
        .expect("update lender");
    scheduler
        .admit_or_register_model(
            "borrower",
            "x100".to_owned(),
            0x000C,
            Some(&borrower_profile),
        )
        .expect("update borrower");

    let pass = scheduler
        .try_schedule_forward_pass("borrower")
        .unwrap()
        .expect("pass granted");
    assert_eq!(pass.borrowed_mask, 0, "lender did not opt in; no lending");
    scheduler.record_pass_completion(&pass, 10.0, 2.0, 5);
}

#[test]
fn test_scenario2_dynamic_core_borrowing_both_opted_in() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let lender_id = "lender-model";
    let borrower_id = "borrower-model";

    let mut lender_profile = TuningProfile::conservative_default();
    lender_profile.profile_status = ProfileStatus::Profiled;
    lender_profile.unprofiled = false;
    lender_profile.lend_idle_cores = true;
    lender_profile.borrow_idle_cores = false;

    let mut borrower_profile = TuningProfile::conservative_default();
    borrower_profile.profile_status = ProfileStatus::Profiled;
    borrower_profile.unprofiled = false;
    borrower_profile.lend_idle_cores = false;
    borrower_profile.borrow_idle_cores = true;

    // Lender: cores 0-1 (0x0003)
    // Borrower: cores 2-3 (0x000C)
    scheduler
        .admit_or_register_model(lender_id, "x100".to_owned(), 0x0003, Some(&lender_profile))
        .expect("register lender");
    scheduler
        .admit_or_register_model(
            borrower_id,
            "x100".to_owned(),
            0x000C,
            Some(&borrower_profile),
        )
        .expect("register borrower");

    // Lender is idle
    assert!(scheduler.get_model_entry(lender_id).unwrap().is_idle());

    // Borrower schedules pass: should dynamically borrow Lender's idle cores 0x0003
    let pass = scheduler
        .try_schedule_forward_pass(borrower_id)
        .unwrap()
        .expect("borrower pass granted");

    assert_eq!(pass.model_id, borrower_id);
    assert_eq!(pass.allocated_mask, 0x000C);
    assert_eq!(
        pass.borrowed_mask, 0x0003,
        "must borrow cores 0-1 from lender"
    );
    assert_eq!(
        pass.effective_mask, 0x000F,
        "effective mask = 0x000C | 0x0003"
    );
    assert_eq!(pass.borrowed_from, vec![lender_id.to_owned()]);

    // Inspect scheduler entries while pass is running
    let lender_entry = scheduler.get_model_entry(lender_id).unwrap();
    assert_eq!(
        lender_entry.active_lent_mask, 0x0003,
        "lender active_lent_mask must track lent cores"
    );

    let borrower_entry = scheduler.get_model_entry(borrower_id).unwrap();
    assert_eq!(borrower_entry.active_running_mask, 0x000F);
    assert_eq!(borrower_entry.active_borrowed_mask, 0x0003);
    assert_eq!(borrower_entry.borrowed_from, vec![lender_id.to_owned()]);

    // System running cores mask tracks the entire effective mask
    assert_eq!(scheduler.running_cores_mask(), 0x000F);

    // Complete forward pass: cores must be released cleanly
    scheduler.record_pass_completion(&pass, 20.0, 10.0, 16);

    assert_eq!(scheduler.running_cores_mask(), 0x0000);
    let lender_after = scheduler.get_model_entry(lender_id).unwrap();
    assert_eq!(
        lender_after.active_lent_mask, 0,
        "lent cores must be cleared upon pass completion"
    );
}

#[test]
fn test_scenario3_instant_pass_boundary_reclaim() {
    let capacity = setup_capacity();
    let scheduler = Arc::new(MultiModelScheduler::new(capacity));

    let lender_id = "owner-lender";
    let borrower_id = "worker-borrower";

    let mut lender_profile = TuningProfile::conservative_default();
    lender_profile.profile_status = ProfileStatus::Profiled;
    lender_profile.unprofiled = false;
    lender_profile.lend_idle_cores = true;

    let mut borrower_profile = TuningProfile::conservative_default();
    borrower_profile.profile_status = ProfileStatus::Profiled;
    borrower_profile.unprofiled = false;
    borrower_profile.borrow_idle_cores = true;

    scheduler
        .admit_or_register_model(lender_id, "x100".to_owned(), 0x0003, Some(&lender_profile))
        .expect("register lender");
    scheduler
        .admit_or_register_model(
            borrower_id,
            "x100".to_owned(),
            0x000C,
            Some(&borrower_profile),
        )
        .expect("register borrower");

    // 1. Borrower schedules pass, borrowing cores 0x0003
    let pass_b = scheduler
        .try_schedule_forward_pass(borrower_id)
        .unwrap()
        .expect("borrower pass granted");
    assert_eq!(pass_b.borrowed_mask, 0x0003);

    // 2. MID-PASS: Owner receives an incoming request!
    scheduler
        .inc_pending_requests(lender_id)
        .expect("owner receives traffic");
    assert_eq!(
        scheduler
            .get_model_entry(lender_id)
            .unwrap()
            .pending_requests_count,
        1
    );

    // 3. Borrower pass finishes at pass boundary
    scheduler.record_pass_completion(&pass_b, 18.0, 6.0, 10);

    // 4. INSTANT RECLAIM: Owner immediately acquires its pass with full base reservation
    let pass_l = scheduler
        .try_schedule_forward_pass(lender_id)
        .unwrap()
        .expect("owner must instantly reclaim cores at pass boundary");
    assert_eq!(pass_l.model_id, lender_id);
    assert_eq!(pass_l.allocated_mask, 0x0003);
    assert_eq!(pass_l.borrowed_mask, 0);

    // 5. While owner is running, borrower cannot borrow owner's cores again
    let pass_b2 = scheduler
        .try_schedule_forward_pass(borrower_id)
        .unwrap()
        .expect("borrower can still run on its own base cores 0x000C");
    assert_eq!(pass_b2.allocated_mask, 0x000C);
    assert_eq!(
        pass_b2.borrowed_mask, 0,
        "borrower cannot borrow busy owner's cores"
    );
    assert_eq!(pass_b2.effective_mask, 0x000C);

    // Complete both passes
    scheduler.record_pass_completion(&pass_l, 12.0, 4.0, 8);
    scheduler.record_pass_completion(&pass_b2, 25.0, 8.0, 10);
    assert_eq!(scheduler.running_cores_mask(), 0x0000);
}

#[test]
fn test_scenario5_control_plane_isolation_guard() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    // Active control cores (0x00C0: cores 6-7) and spare recovery core (0x0020: core 5)
    // Verify scheduler NEVER lends out active control or spare recovery cores
    let mut lender_profile = TuningProfile::conservative_default();
    lender_profile.profile_status = ProfileStatus::Profiled;
    lender_profile.unprofiled = false;
    lender_profile.lend_idle_cores = true;

    let mut borrower_profile = TuningProfile::conservative_default();
    borrower_profile.profile_status = ProfileStatus::Profiled;
    borrower_profile.unprofiled = false;
    borrower_profile.borrow_idle_cores = true;

    // Lender configured on cores 0-1 (0x0003)
    scheduler
        .admit_or_register_model("lender", "x100".to_owned(), 0x0003, Some(&lender_profile))
        .expect("register lender");
    scheduler
        .admit_or_register_model(
            "borrower",
            "x100".to_owned(),
            0x000C,
            Some(&borrower_profile),
        )
        .expect("register borrower");

    let pass = scheduler
        .try_schedule_forward_pass("borrower")
        .unwrap()
        .expect("pass granted");

    // Borrowed mask must NEVER intersect active control cores or spare recovery core
    assert_eq!(
        pass.borrowed_mask & DEFAULT_ACTIVE_CONTROL_MASK,
        0,
        "borrowed cores must never overlap active control mask 0x{DEFAULT_ACTIVE_CONTROL_MASK:04X}"
    );
    assert_eq!(
        pass.borrowed_mask & DEFAULT_SPARE_RECOVERY_MASK,
        0,
        "borrowed cores must never overlap spare recovery core 0x{DEFAULT_SPARE_RECOVERY_MASK:04X}"
    );
    assert_eq!(
        pass.effective_mask & (DEFAULT_ACTIVE_CONTROL_MASK | DEFAULT_SPARE_RECOVERY_MASK),
        0,
        "effective mask must never touch protected control plane cores"
    );

    scheduler.record_pass_completion(&pass, 10.0, 2.0, 4);
}

#[test]
fn test_scenario6_unprofiled_model_spillover_rejection() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    // Attempting to register an unprofiled model with lend_idle_cores = true must fail (ADR-030)
    let mut bad_lend_profile = TuningProfile::conservative_default();
    bad_lend_profile.profile_status = ProfileStatus::Unprofiled;
    bad_lend_profile.unprofiled = true;
    bad_lend_profile.lend_idle_cores = true;

    let err_lend = scheduler
        .admit_or_register_model(
            "unprofiled-lender",
            "x100".to_owned(),
            0x0003,
            Some(&bad_lend_profile),
        )
        .expect_err("unprofiled model cannot enable lend_idle_cores");

    match err_lend {
        ManagerError::WorkerSpawn(msg) => {
            assert!(
                msg.contains("unprofiled model") && msg.contains("cannot enable dynamic idle core"),
                "error message must describe unprofiled spillover prohibition: {msg}"
            );
        }
        other => panic!("expected WorkerSpawn, got {other:?}"),
    }

    // Attempting to register an unprofiled model with borrow_idle_cores = true must fail
    let mut bad_borrow_profile = TuningProfile::conservative_default();
    bad_borrow_profile.profile_status = ProfileStatus::Unprofiled;
    bad_borrow_profile.unprofiled = true;
    bad_borrow_profile.borrow_idle_cores = true;

    let err_borrow = scheduler
        .admit_or_register_model(
            "unprofiled-borrower",
            "x100".to_owned(),
            0x0003,
            Some(&bad_borrow_profile),
        )
        .expect_err("unprofiled model cannot enable borrow_idle_cores");

    match err_borrow {
        ManagerError::WorkerSpawn(msg) => {
            assert!(
                msg.contains("unprofiled model") && msg.contains("cannot enable dynamic idle core"),
                "error message must describe unprofiled spillover prohibition: {msg}"
            );
        }
        other => panic!("expected WorkerSpawn, got {other:?}"),
    }
}
