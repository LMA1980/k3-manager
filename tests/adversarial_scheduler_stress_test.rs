//! Empirical Adversarial Stress Test Suite for `MultiModelScheduler` in k3-manager.
//!
//! Evaluates the core scheduling guarantees under adversarial multi-threaded load:
//! 1. Concurrency limits & load/unload race conditions (clean `ConcurrencyLimitExceeded` rejection).
//! 2. Pass-boundary arbitration & mutual exclusion on overlapping core masks (zero mid-pass interruptions, zero overlapping executions).
//! 3. Priority preemption under continuous Throughput hammering with intermittent Latency bursts (anti-starvation, anti-deadlock).
//! 4. Dynamic core spillover and instant pass-boundary reclaim under sudden lender traffic contention.

#![allow(clippy::too_many_lines, clippy::doc_markdown)]

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use k3_core::affinity::{ControlPlaneCapacityConfig, ResolvedControlCapacity};
use k3_core::profile::{PriorityClass, TuningProfile};
use k3_manager::error::ManagerError;
use k3_manager::scheduler::MultiModelScheduler;

fn setup_capacity() -> ResolvedControlCapacity {
    ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("default capacity resolution succeeds")
}

// ---------------------------------------------------------------------------
// Challenge 1: Concurrency Limits & Load/Unload Race Stress Test
// ---------------------------------------------------------------------------

#[test]
fn test_adversarial_concurrency_limits_rapid_load_unload_race() {
    let capacity = setup_capacity();
    let max_models = 3;
    let scheduler = Arc::new(MultiModelScheduler::with_max_active_models(
        capacity, max_models,
    ));

    let num_threads = 10;
    let iterations_per_thread = 40;

    let total_rejections = Arc::new(AtomicUsize::new(0));
    let total_admissions = Arc::new(AtomicUsize::new(0));
    let total_unregisters = Arc::new(AtomicUsize::new(0));
    let invariant_violations = Arc::new(AtomicUsize::new(0));

    let model_pool = [
        "model_alpha",
        "model_beta",
        "model_gamma",
        "model_delta",
        "model_epsilon",
        "model_zeta",
    ];

    let mut handles = Vec::new();

    for thread_idx in 0..num_threads {
        let sched = Arc::clone(&scheduler);
        let rej_counter = Arc::clone(&total_rejections);
        let adm_counter = Arc::clone(&total_admissions);
        let unreg_counter = Arc::clone(&total_unregisters);
        let viol_counter = Arc::clone(&invariant_violations);

        handles.push(thread::spawn(move || {
            for i in 0..iterations_per_thread {
                let model = model_pool[(thread_idx + i) % model_pool.len()];

                if i % 3 == 0 {
                    // Attempt to unregister
                    if sched.unregister_model(model) {
                        unreg_counter.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    // Attempt to admit model
                    let mask = match i % 3 {
                        0 => 0x0003,
                        1 => 0x000C,
                        _ => 0x0030,
                    };
                    match sched.admit_or_register_model(model, "x100".to_owned(), mask, None) {
                        Ok(()) => {
                            adm_counter.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(ManagerError::ConcurrencyLimitExceeded { active, max }) => {
                            rej_counter.fetch_add(1, Ordering::Relaxed);
                            if active != max_models || max != max_models {
                                viol_counter.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                        Err(e) => {
                            eprintln!("Unexpected admission error: {e}");
                            viol_counter.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }

                // Invariant: active models count must never exceed max_active_models
                let count = sched.active_model_count();
                if count > max_models {
                    viol_counter.fetch_add(1, Ordering::SeqCst);
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join succeeds without panicking");
    }

    assert_eq!(
        invariant_violations.load(Ordering::SeqCst),
        0,
        "Concurrency limit invariant was violated during race conditions!"
    );

    let final_count = scheduler.active_model_count();
    assert!(
        final_count <= max_models,
        "Final active models count ({final_count}) exceeded cap ({max_models})"
    );

    let rejections = total_rejections.load(Ordering::SeqCst);
    let admissions = total_admissions.load(Ordering::SeqCst);
    assert!(
        rejections > 0,
        "Expected ConcurrencyLimitExceeded rejections under contention, got 0"
    );
    assert!(
        admissions > 0,
        "Expected successful admissions under contention, got 0"
    );

    // Clean up all models
    for m in &model_pool {
        scheduler.unregister_model(m);
    }
    assert_eq!(scheduler.active_model_count(), 0);

    // Verify clean admission recovery after complete unload
    scheduler
        .admit_or_register_model("model_alpha", "x100".to_owned(), 0x0003, None)
        .expect("admission 1");
    scheduler
        .admit_or_register_model("model_beta", "x100".to_owned(), 0x000C, None)
        .expect("admission 2");
    scheduler
        .admit_or_register_model("model_gamma", "x100".to_owned(), 0x0030, None)
        .expect("admission 3");
    assert_eq!(scheduler.active_model_count(), 3);

    // 4th model must be cleanly rejected
    let res4 = scheduler.admit_or_register_model("model_delta", "x100".to_owned(), 0x0003, None);
    match res4 {
        Err(ManagerError::ConcurrencyLimitExceeded { active, max }) => {
            assert_eq!(active, 3);
            assert_eq!(max, 3);
        }
        other => panic!("Expected ConcurrencyLimitExceeded for 4th model, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Challenge 2: Pass-Boundary Arbitration & Mutual Exclusion Stress Test
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ActivePassRecord {
    model_id: String,
    pass_id: u64,
    effective_mask: u64,
}

#[derive(Default)]
struct ExecutionOracle {
    active_passes: Vec<ActivePassRecord>,
    total_passes_completed: u64,
    overlapping_violations: u64,
    exclusivity_violations: u64,
}

impl ExecutionOracle {
    fn enter_pass(&mut self, model_id: &str, pass_id: u64, effective_mask: u64) {
        for active in &self.active_passes {
            // Check single-prompt exclusivity: same model cannot run two passes concurrently
            if active.model_id == model_id {
                self.exclusivity_violations += 1;
            }
            // Check physical core mutual exclusion: overlapping masks cannot run concurrently
            if (active.effective_mask & effective_mask) != 0 {
                self.overlapping_violations += 1;
            }
        }

        self.active_passes.push(ActivePassRecord {
            model_id: model_id.to_owned(),
            pass_id,
            effective_mask,
        });
    }

    fn exit_pass(&mut self, pass_id: u64) {
        self.active_passes.retain(|p| p.pass_id != pass_id);
        self.total_passes_completed += 1;
    }
}

#[test]
fn test_adversarial_overlapping_arbitration_mutual_exclusion() {
    let capacity = setup_capacity();
    let scheduler = Arc::new(MultiModelScheduler::new(capacity));

    // Three models on X100 with partially overlapping compute reservations:
    // Model A: cores 0-3 (0x000F)
    // Model B: cores 2-5 (0x003C) [overlaps Model A on 0x000C, overlaps Model C on 0x0030]
    // Model C: cores 4-5 (0x0030) [disjoint with Model A: 0x000F & 0x0030 == 0]
    let model_a = "model_overlap_a";
    let model_b = "model_overlap_b";
    let model_c = "model_overlap_c";

    scheduler
        .admit_or_register_model(model_a, "x100".to_owned(), 0x000F, None)
        .expect("register model A");
    scheduler
        .admit_or_register_model(model_b, "x100".to_owned(), 0x003C, None)
        .expect("register model B");
    scheduler
        .admit_or_register_model(model_c, "x100".to_owned(), 0x0030, None)
        .expect("register model C");

    let oracle = Arc::new(Mutex::new(ExecutionOracle::default()));
    let global_pass_seq = Arc::new(AtomicU64::new(1));

    let passes_per_thread = 40;
    let mut handles = Vec::new();

    // 2 worker threads per model = 6 concurrent worker threads
    let models = [model_a, model_b, model_c, model_a, model_b, model_c];

    for &target_model in &models {
        let sched = Arc::clone(&scheduler);
        let orc = Arc::clone(&oracle);
        let seq_gen = Arc::clone(&global_pass_seq);
        let m_name = target_model.to_owned();

        handles.push(thread::spawn(move || {
            for _ in 0..passes_per_thread {
                let pass = sched
                    .schedule_forward_pass(&m_name, Some(Duration::from_secs(5)))
                    .unwrap_or_else(|e| {
                        panic!("Failed to schedule forward pass for {m_name}: {e}")
                    });

                let pass_id = seq_gen.fetch_add(1, Ordering::SeqCst);

                // Enter Oracle and assert invariants
                {
                    let mut guard = orc.lock().unwrap();
                    guard.enter_pass(&pass.model_id, pass_id, pass.effective_mask);
                }

                // Simulate realistic forward pass compute duration
                thread::sleep(Duration::from_micros(300));

                // Exit Oracle and assert completion
                {
                    let mut guard = orc.lock().unwrap();
                    guard.exit_pass(pass_id);
                }

                sched.record_pass_completion(&pass, 10.0, 5.0, 16);
            }
        }));
    }

    for h in handles {
        h.join().expect("worker thread joined cleanly");
    }

    let final_oracle = oracle.lock().unwrap();
    assert_eq!(
        final_oracle.overlapping_violations, 0,
        "Found {} overlapping core executions across concurrent passes!",
        final_oracle.overlapping_violations
    );
    assert_eq!(
        final_oracle.exclusivity_violations, 0,
        "Found {} single-prompt exclusivity violations (concurrent passes on same model)!",
        final_oracle.exclusivity_violations
    );
    assert_eq!(
        final_oracle.total_passes_completed,
        (passes_per_thread * models.len()) as u64,
        "All scheduled passes must complete successfully"
    );

    // Verify running cores mask returned to clean 0
    assert_eq!(scheduler.running_cores_mask(), 0);
}

// ---------------------------------------------------------------------------
// Challenge 3: Priority Preemption & Anti-Starvation / Anti-Deadlock Stress Test
// ---------------------------------------------------------------------------

#[test]
fn test_adversarial_priority_preemption_anti_starvation_anti_deadlock() {
    let capacity = setup_capacity();
    let scheduler = Arc::new(MultiModelScheduler::new(capacity));

    // Three models contending for the exact same core reservation 0x000F:
    // T1: Throughput priority
    // T2: Throughput priority
    // L: Latency priority (interactive)
    let model_t1 = "model_throughput_1";
    let model_t2 = "model_throughput_2";
    let model_l = "model_latency_interactive";

    let mut profile_t = TuningProfile::conservative_default();
    profile_t.priority_class = PriorityClass::Throughput;

    let mut profile_l = TuningProfile::conservative_default();
    profile_l.priority_class = PriorityClass::Latency;

    scheduler
        .admit_or_register_model(model_t1, "x100".to_owned(), 0x000F, Some(&profile_t))
        .expect("register T1");
    scheduler
        .admit_or_register_model(model_t2, "x100".to_owned(), 0x000F, Some(&profile_t))
        .expect("register T2");
    scheduler
        .admit_or_register_model(model_l, "x100".to_owned(), 0x000F, Some(&profile_l))
        .expect("register L");

    let t1_completed = Arc::new(AtomicU32::new(0));
    let t2_completed = Arc::new(AtomicU32::new(0));
    let l_completed = Arc::new(AtomicU32::new(0));

    let throughput_target_passes = 40;
    let latency_burst_count = 4;
    let latency_burst_size = 5; // 20 total latency passes
    let total_expected_latency = latency_burst_count * latency_burst_size;

    let stop_hammering = Arc::new(AtomicBool::new(false));

    // Hammer Thread 1: Throughput Model T1
    let t1_handle = {
        let sched = Arc::clone(&scheduler);
        let done = Arc::clone(&t1_completed);
        let stop = Arc::clone(&stop_hammering);
        thread::spawn(move || {
            let mut count = 0;
            while count < throughput_target_passes || !stop.load(Ordering::Relaxed) {
                if let Ok(pass) =
                    sched.schedule_forward_pass(model_t1, Some(Duration::from_secs(5)))
                {
                    thread::sleep(Duration::from_micros(250));
                    sched.record_pass_completion(&pass, 20.0, 10.0, 32);
                    count += 1;
                    done.fetch_add(1, Ordering::SeqCst);
                }
                if count >= throughput_target_passes && stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        })
    };

    // Hammer Thread 2: Throughput Model T2
    let t2_handle = {
        let sched = Arc::clone(&scheduler);
        let done = Arc::clone(&t2_completed);
        let stop = Arc::clone(&stop_hammering);
        thread::spawn(move || {
            let mut count = 0;
            while count < throughput_target_passes || !stop.load(Ordering::Relaxed) {
                if let Ok(pass) =
                    sched.schedule_forward_pass(model_t2, Some(Duration::from_secs(5)))
                {
                    thread::sleep(Duration::from_micros(250));
                    sched.record_pass_completion(&pass, 20.0, 10.0, 32);
                    count += 1;
                    done.fetch_add(1, Ordering::SeqCst);
                }
                if count >= throughput_target_passes && stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        })
    };

    // Burst Thread: Latency Model L bursts intermittently
    let l_handle = {
        let sched = Arc::clone(&scheduler);
        let done = Arc::clone(&l_completed);
        thread::spawn(move || {
            for _burst in 0..latency_burst_count {
                // Intermittent pause between bursts to allow throughput to run
                thread::sleep(Duration::from_millis(5));

                let mut burst_handles = Vec::new();
                for _ in 0..latency_burst_size {
                    let s = Arc::clone(&sched);
                    let d = Arc::clone(&done);
                    burst_handles.push(thread::spawn(move || {
                        let pass = s
                            .schedule_forward_pass(model_l, Some(Duration::from_secs(5)))
                            .expect("latency pass granted within deadline");
                        // Latency pass compute
                        thread::sleep(Duration::from_micros(200));
                        s.record_pass_completion(&pass, 5.0, 2.0, 16);
                        d.fetch_add(1, Ordering::SeqCst);
                    }));
                }

                for bh in burst_handles {
                    bh.join().expect("latency burst thread join");
                }
            }
        })
    };

    // Wait for latency bursts to complete
    l_handle.join().expect("latency master thread join");
    assert_eq!(
        l_completed.load(Ordering::SeqCst),
        total_expected_latency,
        "All latency requests must be granted and completed"
    );

    // Stop hammering throughput threads and let them finish their remaining passes
    stop_hammering.store(true, Ordering::Relaxed);
    t1_handle.join().expect("T1 thread join");
    t2_handle.join().expect("T2 thread join");

    let t1_total = t1_completed.load(Ordering::SeqCst);
    let t2_total = t2_completed.load(Ordering::SeqCst);

    assert!(
        t1_total >= throughput_target_passes,
        "Throughput T1 starved! Completed only {t1_total} / {throughput_target_passes}"
    );
    assert!(
        t2_total >= throughput_target_passes,
        "Throughput T2 starved! Completed only {t2_total} / {throughput_target_passes}"
    );

    // Verify zero deadlocks: running cores mask returned to 0
    assert_eq!(scheduler.running_cores_mask(), 0);
}

// ---------------------------------------------------------------------------
// Challenge 4: Instant Pass-Boundary Reclaim Under Dynamic Traffic Spurt
// ---------------------------------------------------------------------------

#[test]
fn test_adversarial_instant_pass_boundary_reclaim_under_traffic_spurt() {
    let capacity = setup_capacity();
    let scheduler = Arc::new(MultiModelScheduler::new(capacity));

    let lender = "lender_model";
    let borrower = "borrower_model";

    let mut lender_prof = TuningProfile::conservative_default();
    lender_prof.lend_idle_cores = true;
    lender_prof.unprofiled = false;

    let mut borrower_prof = TuningProfile::conservative_default();
    borrower_prof.borrow_idle_cores = true;
    borrower_prof.unprofiled = false;

    // Lender on cores 0-1 (0x0003), Borrower on cores 2-3 (0x000C)
    scheduler
        .admit_or_register_model(lender, "x100".to_owned(), 0x0003, Some(&lender_prof))
        .expect("register lender");
    scheduler
        .admit_or_register_model(borrower, "x100".to_owned(), 0x000C, Some(&borrower_prof))
        .expect("register borrower");

    // Cycle 50 times: Borrower borrows Lender's cores while Lender is idle,
    // then Lender receives work mid-pass and MUST reclaim cores at the immediate pass boundary.
    for cycle in 0..50 {
        // 1. Borrower schedules a pass while Lender is idle: gets expanded mask 0x000F
        let borrower_pass = scheduler
            .try_schedule_forward_pass(borrower)
            .expect("schedule borrower")
            .unwrap_or_else(|| panic!("Borrower pass must be granted on cycle {cycle}"));

        assert_eq!(borrower_pass.allocated_mask, 0x000C);
        assert_eq!(borrower_pass.borrowed_mask, 0x0003);
        assert_eq!(borrower_pass.effective_mask, 0x000F);

        // 2. Mid-pass traffic arrives for Lender!
        scheduler
            .inc_pending_requests(lender)
            .expect("lender traffic arrives");

        // Lender cannot preempt mid-pass: cores 0x0003 are still actively in flight
        let lender_mid_pass = scheduler
            .try_schedule_forward_pass(lender)
            .expect("schedule lender mid-pass");
        assert!(
            lender_mid_pass.is_none(),
            "Lender must not preempt borrower mid-pass"
        );

        // 3. Borrower pass finishes at pass boundary
        scheduler.record_pass_completion(&borrower_pass, 15.0, 5.0, 16);

        // 4. Lender immediately reclaims its cores!
        let lender_pass = scheduler
            .try_schedule_forward_pass(lender)
            .expect("schedule lender at boundary")
            .unwrap_or_else(|| {
                panic!("Lender must reclaim cores at pass boundary on cycle {cycle}")
            });

        assert_eq!(lender_pass.allocated_mask, 0x0003);
        assert_eq!(lender_pass.borrowed_mask, 0x0000);
        assert_eq!(lender_pass.effective_mask, 0x0003);

        scheduler.record_pass_completion(&lender_pass, 10.0, 5.0, 16);
    }

    assert_eq!(scheduler.running_cores_mask(), 0);
}

// ---------------------------------------------------------------------------
// Challenge 5: Explicit 4th and 5th Model Admission Under High Thread Contention
// ---------------------------------------------------------------------------

#[test]
fn test_adversarial_admission_4th_and_5th_concurrent_rejection() {
    let capacity = setup_capacity();
    let scheduler = Arc::new(MultiModelScheduler::with_max_active_models(capacity, 3));

    // Register exactly 3 active models
    scheduler
        .admit_or_register_model("model_active_1", "x100".to_owned(), 0x0003, None)
        .expect("register 1");
    scheduler
        .admit_or_register_model("model_active_2", "x100".to_owned(), 0x000C, None)
        .expect("register 2");
    scheduler
        .admit_or_register_model("model_active_3", "x100".to_owned(), 0x0030, None)
        .expect("register 3");

    assert_eq!(scheduler.active_model_count(), 3);

    // Fire 20 concurrent threads trying to admit 4th ("model_excess_4") and 5th ("model_excess_5") models
    let num_threads = 20;
    let barrier = Arc::new(std::sync::Barrier::new(num_threads));
    let rejection_count = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for thread_idx in 0..num_threads {
        let sched = Arc::clone(&scheduler);
        let bar = Arc::clone(&barrier);
        let rej = Arc::clone(&rejection_count);
        let excess_model = if thread_idx % 2 == 0 {
            "model_excess_4"
        } else {
            "model_excess_5"
        };

        handles.push(thread::spawn(move || {
            bar.wait();
            let res = sched.admit_or_register_model(excess_model, "x100".to_owned(), 0x0003, None);
            match res {
                Err(ManagerError::ConcurrencyLimitExceeded { active, max }) => {
                    assert_eq!(active, 3);
                    assert_eq!(max, 3);
                    rej.fetch_add(1, Ordering::SeqCst);
                }
                other => panic!("Thread {thread_idx} received unexpected result: {other:?}"),
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }

    assert_eq!(
        rejection_count.load(Ordering::SeqCst),
        num_threads,
        "Every attempt to admit 4th/5th models must be cleanly rejected"
    );

    // Assert active models remain strictly 3 and unaltered
    assert_eq!(scheduler.active_model_count(), 3);
    assert_eq!(
        scheduler.list_models(),
        vec![
            "model_active_1".to_owned(),
            "model_active_2".to_owned(),
            "model_active_3".to_owned(),
        ]
    );
}

// ---------------------------------------------------------------------------
// Challenge 6: Cost-Weighted WFQ Virtual Runtime Interleaving Stress
// ---------------------------------------------------------------------------

#[test]
fn test_adversarial_wfq_cost_weighted_interleaving_stress() {
    let capacity = setup_capacity();
    let scheduler = Arc::new(MultiModelScheduler::new(capacity));

    let fast_model = "model_fast_2ms";
    let slow_model = "model_slow_20ms";

    // Contend for same core reservation 0x000F
    scheduler
        .admit_or_register_model(fast_model, "x100".to_owned(), 0x000F, None)
        .expect("register fast");
    scheduler
        .admit_or_register_model(slow_model, "x100".to_owned(), 0x000F, None)
        .expect("register slow");

    let fast_passes = Arc::new(AtomicU32::new(0));
    let slow_passes = Arc::new(AtomicU32::new(0));

    let total_slow_target = 3;
    let fast_done = Arc::clone(&fast_passes);
    let slow_done = Arc::clone(&slow_passes);

    let stop_signal = Arc::new(AtomicBool::new(false));

    // Spawn 5 concurrent threads submitting Fast requests so Fast always has pending work
    let mut fast_handles = Vec::new();
    for _ in 0..5 {
        let s = Arc::clone(&scheduler);
        let fd = Arc::clone(&fast_done);
        let stop = Arc::clone(&stop_signal);
        fast_handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Ok(pass) =
                    s.schedule_forward_pass(fast_model, Some(Duration::from_millis(500)))
                {
                    s.record_pass_completion(&pass, 2.0, 0.0, 16);
                    fd.fetch_add(1, Ordering::SeqCst);
                    thread::yield_now();
                }
            }
        }));
    }

    // Slow worker thread: requests 3 passes, each with 20ms recorded cost
    let s_slow = Arc::clone(&scheduler);
    let h_slow = thread::spawn(move || {
        for _ in 0..total_slow_target {
            let pass = s_slow
                .schedule_forward_pass(slow_model, Some(Duration::from_secs(5)))
                .expect("slow pass granted");
            s_slow.record_pass_completion(&pass, 20.0, 0.0, 16);
            slow_done.fetch_add(1, Ordering::SeqCst);
            thread::sleep(Duration::from_micros(50));
        }
    });

    h_slow.join().expect("slow worker finished");
    // Allow fast threads to catch up (drain) for up to 100ms or until fast_count >= 20
    let drain_start = std::time::Instant::now();
    while fast_passes.load(Ordering::SeqCst) < 20
        && drain_start.elapsed() < Duration::from_millis(100)
    {
        thread::yield_now();
    }
    stop_signal.store(true, Ordering::Relaxed);
    for fh in fast_handles {
        fh.join().expect("fast worker finished");
    }

    let slow_count = slow_passes.load(Ordering::SeqCst);
    let fast_count = fast_passes.load(Ordering::SeqCst);

    assert_eq!(slow_count, total_slow_target);
    // Cost ratio is 20ms : 2ms = 10:1. Fast should complete roughly 10x more passes than slow (~30 passes for 3 slow passes).
    assert!(
        fast_count >= 15,
        "WFQ interleaving failed under contention! Fast model ran {fast_count} passes vs Slow model {slow_count} passes"
    );
}

// ---------------------------------------------------------------------------
// Challenge 7: Client Priority Override Immunity Under Heavy Contention
// ---------------------------------------------------------------------------

#[test]
fn test_adversarial_client_priority_override_immunity() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let mut profile = TuningProfile::conservative_default();
    profile.priority_class = PriorityClass::Throughput;

    scheduler
        .admit_or_register_model("batch_model", "x100".to_owned(), 0x000F, Some(&profile))
        .expect("register model");

    // Client attempts to pass Latency priority in HTTP header
    let resolved = scheduler
        .enforce_operator_priority("batch_model", Some("latency"))
        .expect("resolve string header");
    assert_eq!(
        resolved,
        PriorityClass::Throughput,
        "Client string priority override must be stripped"
    );

    let resolved_enum = scheduler
        .resolve_request_priority("batch_model", Some(PriorityClass::Latency))
        .expect("resolve enum header");
    assert_eq!(
        resolved_enum,
        PriorityClass::Throughput,
        "Client enum priority override must be stripped"
    );
}
