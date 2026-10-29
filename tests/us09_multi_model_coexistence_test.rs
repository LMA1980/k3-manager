//! Integration tests for US-09 (#9): Multi-Model Coexistence: Up to 3 Models, Bitmask Overlap Arbitration, Pass-Boundary Time-Slicing.
//!
//! # Core System Assertions
//! - **Scenario 1**: Non-overlapping reservations execute in parallel without queuing.
//!   Models with disjoint physical core bitmasks (`mask_a & mask_b == 0`) acquire concurrent forward-pass
//!   permits simultaneously and execute without contention.
//! - **Scenario 2**: Overlapping reservations time-slice strictly at forward-pass boundaries.
//!   When two models share compute cores (`mask_a & mask_b != 0`), mid-pass preemption never occurs.
//!   The contested model yields until the running model completes its pass and releases the shared cores.
//! - **Scenario 3**: Concurrency limit enforcement.
//!   Up to 3 active models in memory (`max_active_models = 3` per ADR-043); admitting a 4th model is
//!   rejected upfront with [`ManagerError::ConcurrencyLimitExceeded`].
//! - **Scenario 4**: Single-prompt exclusivity per forward pass.
//!   A single model cannot execute multiple concurrent forward passes; subsequent attempts to schedule
//!   passes for the same model must queue or wait.
//! - **Scenario 5**: Multi-model cluster coexistence.
//!   Concurrent execution across Cluster 0 (X100) and Cluster 1 (A100) with independent bitmask isolation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use k3_core::affinity::{ControlPlaneCapacityConfig, ResolvedControlCapacity};
use k3_manager::error::ManagerError;
use k3_manager::scheduler::MultiModelScheduler;

fn setup_capacity() -> ResolvedControlCapacity {
    ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("default capacity resolution succeeds")
}

#[test]
fn test_scenario1_non_overlapping_parallel_execution() {
    let capacity = setup_capacity();
    let scheduler = Arc::new(MultiModelScheduler::new(capacity));

    let model_a = "qwen-x100-low";
    let model_b = "deepseek-x100-mid";

    // Non-overlapping compute reservations on Cluster 0 (X100):
    // Model A: cores 0-1 (0x0003)
    // Model B: cores 2-3 (0x000C)
    // 0x0003 & 0x000C == 0
    scheduler
        .admit_or_register_model(model_a, "x100".to_owned(), 0x0003, None)
        .expect("register model A succeeds");
    scheduler
        .admit_or_register_model(model_b, "x100".to_owned(), 0x000C, None)
        .expect("register model B succeeds");

    // Both models should acquire forward-pass execution permits simultaneously
    let pass_a = scheduler
        .try_schedule_forward_pass(model_a)
        .expect("schedule A succeeds")
        .expect("pass A permit must be granted immediately");

    let pass_b = scheduler
        .try_schedule_forward_pass(model_b)
        .expect("schedule B succeeds")
        .expect("pass B permit must be granted immediately without queuing");

    assert_eq!(pass_a.model_id, model_a);
    assert_eq!(pass_a.allocated_mask, 0x0003);
    assert_eq!(pass_a.effective_mask, 0x0003);

    assert_eq!(pass_b.model_id, model_b);
    assert_eq!(pass_b.allocated_mask, 0x000C);
    assert_eq!(pass_b.effective_mask, 0x000C);

    // Both models are executing in parallel: locked cores mask = 0x0003 | 0x000C = 0x000F
    assert_eq!(scheduler.running_cores_mask(), 0x000F);

    // Complete Pass A: releases cores 0-1, cores 2-3 remain locked by Pass B
    scheduler.record_pass_completion(&pass_a, 25.0, 10.0, 32);
    assert_eq!(scheduler.running_cores_mask(), 0x000C);

    // Complete Pass B: all cores released
    scheduler.record_pass_completion(&pass_b, 30.0, 12.0, 32);
    assert_eq!(scheduler.running_cores_mask(), 0x0000);
}

#[test]
fn test_scenario2_overlapping_pass_boundary_timeslicing() {
    let capacity = setup_capacity();
    let scheduler = Arc::new(MultiModelScheduler::new(capacity));

    let model_a = "model-alpha";
    let model_b = "model-beta";

    // Overlapping reservations:
    // Model A: cores 0-3 (0x000F)
    // Model B: cores 2-5 (0x003C)
    // Overlap: cores 2-3 (0x000C) != 0
    scheduler
        .admit_or_register_model(model_a, "x100".to_owned(), 0x000F, None)
        .expect("register A succeeds");
    scheduler
        .admit_or_register_model(model_b, "x100".to_owned(), 0x003C, None)
        .expect("register B succeeds");

    // Model A acquires pass permit first
    let pass_a = scheduler
        .try_schedule_forward_pass(model_a)
        .expect("schedule A succeeds")
        .expect("pass A granted");
    assert_eq!(scheduler.running_cores_mask(), 0x000F);

    // Model B attempts immediate schedule: MUST return Ok(None) due to core overlap!
    let pass_b_immediate = scheduler
        .try_schedule_forward_pass(model_b)
        .expect("try schedule B succeeds");
    assert!(
        pass_b_immediate.is_none(),
        "Model B must not execute concurrently over contested cores 0x000C"
    );

    // Mid-pass preemption NEVER occurs: Model B cannot interrupt Model A mid-pass.
    // Spawn background worker for Model B that blocks on pass boundary
    let sched_clone = Arc::clone(&scheduler);
    let model_b_acquired = Arc::new(AtomicBool::new(false));
    let model_b_flag = Arc::clone(&model_b_acquired);

    let handle = thread::spawn(move || {
        let pass_b = sched_clone
            .schedule_forward_pass(model_b, Some(Duration::from_secs(3)))
            .expect("schedule B should unblock at pass boundary");
        model_b_flag.store(true, Ordering::SeqCst);
        pass_b
    });

    // Sleep briefly to confirm Model B remains blocked while Model A is running
    thread::sleep(Duration::from_millis(50));
    assert!(
        !model_b_acquired.load(Ordering::SeqCst),
        "Model B must remain blocked while Model A holds contested cores"
    );

    // Model A completes forward pass: releases 0x000F at pass boundary
    scheduler.record_pass_completion(&pass_a, 50.0, 20.0, 64);

    // Join thread: Model B should now have unblocked and acquired the permit
    let pass_b = handle.join().expect("thread join succeeds");
    assert!(model_b_acquired.load(Ordering::SeqCst));
    assert_eq!(pass_b.model_id, model_b);
    assert_eq!(pass_b.allocated_mask, 0x003C);
    assert_eq!(scheduler.running_cores_mask(), 0x003C);

    // Complete Model B forward pass
    scheduler.record_pass_completion(&pass_b, 40.0, 15.0, 64);
    assert_eq!(scheduler.running_cores_mask(), 0x0000);
}

#[test]
fn test_scenario3_max_active_models_concurrency_limit_exceeded() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    // Default max_active_models is 3 (ADR-043)
    assert_eq!(scheduler.max_active_models(), 3);

    let m1 = "model-1";
    let m2 = "model-2";
    let m3 = "model-3";
    let m4 = "model-4";

    // Register first 3 models: all should succeed
    scheduler
        .admit_or_register_model(m1, "x100".to_owned(), 0x0003, None)
        .expect("register m1 succeeds");
    scheduler
        .admit_or_register_model(m2, "x100".to_owned(), 0x000C, None)
        .expect("register m2 succeeds");
    scheduler
        .admit_or_register_model(m3, "a100".to_owned(), 0x0F00, None)
        .expect("register m3 succeeds");

    assert_eq!(scheduler.active_model_count(), 3);

    // Attempting to register 4th model must fail with ConcurrencyLimitExceeded
    let err = scheduler
        .admit_or_register_model(m4, "a100".to_owned(), 0xF000, None)
        .expect_err("registering 4th model must exceed concurrency limit");

    match err {
        ManagerError::ConcurrencyLimitExceeded { active, max } => {
            assert_eq!(active, 3);
            assert_eq!(max, 3);
            let display_str = format!("{}", ManagerError::ConcurrencyLimitExceeded { active, max });
            assert!(
                display_str.contains("concurrency limit exceeded"),
                "display string must describe limit: {display_str}"
            );
        }
        other => panic!("expected ConcurrencyLimitExceeded, got {other:?}"),
    }

    // Unregister m1: active count drops to 2
    assert!(scheduler.unregister_model(m1));
    assert_eq!(scheduler.active_model_count(), 2);

    // Now m4 can be admitted successfully
    scheduler
        .admit_or_register_model(m4, "a100".to_owned(), 0xF000, None)
        .expect("register m4 must succeed after unregistering m1");
    assert_eq!(scheduler.active_model_count(), 3);
}

#[test]
fn test_scenario4_single_prompt_exclusivity_per_forward_pass() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let model_id = "exclusive-qwen";
    scheduler
        .admit_or_register_model(model_id, "x100".to_owned(), 0x000F, None)
        .expect("register succeeds");

    // Acquire first forward pass permit
    let pass1 = scheduler
        .try_schedule_forward_pass(model_id)
        .expect("schedule succeeds")
        .expect("pass 1 granted");

    // Second request for the same model MUST be refused while pass1 is in-flight
    let pass2 = scheduler
        .try_schedule_forward_pass(model_id)
        .expect("schedule succeeds");
    assert!(
        pass2.is_none(),
        "single-prompt exclusivity forbids concurrent forward passes for the same model"
    );

    // Complete pass 1
    scheduler.record_pass_completion(&pass1, 20.0, 5.0, 16);

    // Now second forward pass can proceed
    let pass2 = scheduler
        .try_schedule_forward_pass(model_id)
        .expect("schedule succeeds")
        .expect("pass 2 granted after pass 1 completes");
    assert_eq!(pass2.model_id, model_id);

    scheduler.record_pass_completion(&pass2, 22.0, 5.5, 16);
    assert_eq!(scheduler.running_cores_mask(), 0x0000);
}

#[test]
fn test_scenario5_three_models_cluster_coexistence() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let primary_x100 = "m-x100-a";
    let secondary_x100 = "m-x100-b";
    let accelerated_a100 = "m-a100";

    // primary_x100: Cores 0-1 (0x0003) on X100
    // secondary_x100: Cores 2-3 (0x000C) on X100
    // accelerated_a100: Cores 8-15 (0xFF00) on A100
    // All 3 bitmasks are mutually disjoint
    scheduler
        .admit_or_register_model(primary_x100, "x100".to_owned(), 0x0003, None)
        .expect("register primary_x100");
    scheduler
        .admit_or_register_model(secondary_x100, "x100".to_owned(), 0x000C, None)
        .expect("register secondary_x100");
    scheduler
        .admit_or_register_model(accelerated_a100, "a100".to_owned(), 0xFF00, None)
        .expect("register accelerated_a100");

    // All three execute in parallel without contention
    let primary_permit = scheduler
        .try_schedule_forward_pass(primary_x100)
        .unwrap()
        .expect("primary_permit granted");
    let secondary_permit = scheduler
        .try_schedule_forward_pass(secondary_x100)
        .unwrap()
        .expect("secondary_permit granted");
    let accelerated_permit = scheduler
        .try_schedule_forward_pass(accelerated_a100)
        .unwrap()
        .expect("accelerated_permit granted");

    assert_eq!(scheduler.running_cores_mask(), 0x000F | 0xFF00);

    // Releasing in arbitrary order
    scheduler.record_pass_completion(&secondary_permit, 15.0, 5.0, 10);
    assert_eq!(scheduler.running_cores_mask(), 0x0003 | 0xFF00);

    scheduler.record_pass_completion(&primary_permit, 20.0, 8.0, 10);
    assert_eq!(scheduler.running_cores_mask(), 0xFF00);

    scheduler.record_pass_completion(&accelerated_permit, 10.0, 2.0, 10);
    assert_eq!(scheduler.running_cores_mask(), 0x0000);
}
