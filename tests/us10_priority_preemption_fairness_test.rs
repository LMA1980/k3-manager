//! Integration tests for US-10 (#10): Priority Preemption & Cost-Weighted Fair Sharing (WFQ).
//!
//! # Core System Assertions
//! - **Scenario 1**: `PriorityClass::Latency` preempts `PriorityClass::Throughput` at next forward-pass boundary.
//!   When high-priority interactive requests arrive, currently executing throughput forward passes
//!   complete without mid-pass interruption. At the pass boundary, the scheduler immediately yields
//!   the contested cores to the waiting `Latency` model.
//! - **Scenario 2**: Intra-class Weighted Fair Queuing (WFQ) proportional to measured pass costs.
//!   Models within the same priority class contending for overlapping cores interleave passes
//!   proportionally to their measured pass costs (e.g. 2ms vs 50ms yields ~25:1 passes),
//!   guaranteeing equal share of physical compute time rather than pass count.
//! - **Scenario 3**: Strict client priority override immunity.
//!   Client-supplied priority parameters are stripped or ignored; only the operator-configured
//!   `TuningProfile` priority class governs scheduling preemption.
//! - **Scenario 4**: Minimum virtual runtime catch-up on admission.
//!   Newly admitted or waking models catch up to `min_virtual_runtime_ms`, preventing historical starvation
//!   attacks on active peers.

use std::sync::Arc;

use k3_core::affinity::{ControlPlaneCapacityConfig, ResolvedControlCapacity};
use k3_core::profile::{PriorityClass, TuningProfile};
use k3_manager::scheduler::MultiModelScheduler;

fn setup_capacity() -> ResolvedControlCapacity {
    ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("default capacity resolution succeeds")
}

#[test]
fn test_scenario1_latency_preempts_throughput_at_pass_boundary() {
    let capacity = setup_capacity();
    let scheduler = Arc::new(MultiModelScheduler::new(capacity));

    let throughput_model = "batch-model";
    let latency_model = "interactive-model";

    let mut t_profile = TuningProfile::conservative_default();
    t_profile.priority_class = PriorityClass::Throughput;

    let mut l_profile = TuningProfile::conservative_default();
    l_profile.priority_class = PriorityClass::Latency;

    // Both models contend for identical core reservation 0x000F (cores 0-3 on X100)
    scheduler
        .admit_or_register_model(
            throughput_model,
            "x100".to_owned(),
            0x000F,
            Some(&t_profile),
        )
        .expect("register throughput model");
    scheduler
        .admit_or_register_model(latency_model, "x100".to_owned(), 0x000F, Some(&l_profile))
        .expect("register latency model");

    // 1. Throughput model begins a forward pass
    let pass_t1 = scheduler
        .try_schedule_forward_pass(throughput_model)
        .expect("schedule succeeds")
        .expect("throughput pass 1 granted");
    assert_eq!(pass_t1.model_id, throughput_model);
    assert_eq!(scheduler.running_cores_mask(), 0x000F);

    // 2. While throughput pass is running, both models queue pending work
    scheduler
        .inc_pending_requests(throughput_model)
        .expect("inc throughput");
    scheduler
        .inc_pending_requests(latency_model)
        .expect("inc latency");

    // Latency model cannot preempt mid-pass: try_schedule_forward_pass returns Ok(None)
    let immediate_l = scheduler
        .try_schedule_forward_pass(latency_model)
        .expect("try schedule");
    assert!(immediate_l.is_none(), "mid-pass preemption is forbidden");

    // 3. Throughput pass completes at pass boundary
    scheduler.record_pass_completion(&pass_t1, 100.0, 50.0, 32);

    // 4. At the pass boundary, Contention Arbitration MUST pick the Latency model!
    // If throughput tries to schedule first, it MUST be refused because latency has pending requests
    let pass_t2 = scheduler
        .try_schedule_forward_pass(throughput_model)
        .expect("try schedule throughput");
    assert!(
        pass_t2.is_none(),
        "throughput model must be preempted by waiting latency model"
    );

    // Latency model successfully acquires the permit
    let pass_l = scheduler
        .try_schedule_forward_pass(latency_model)
        .expect("schedule latency succeeds")
        .expect("latency permit must be granted ahead of throughput");
    assert_eq!(pass_l.model_id, latency_model);
    assert_eq!(pass_l.allocated_mask, 0x000F);

    // Latency pass completes
    scheduler.record_pass_completion(&pass_l, 10.0, 2.0, 1);

    // Now throughput model can finally acquire its pass
    let pass_t2 = scheduler
        .try_schedule_forward_pass(throughput_model)
        .expect("schedule throughput succeeds")
        .expect("throughput can proceed after latency completes");
    assert_eq!(pass_t2.model_id, throughput_model);

    scheduler.record_pass_completion(&pass_t2, 80.0, 40.0, 32);
    assert_eq!(scheduler.running_cores_mask(), 0x0000);
}

#[test]
fn test_scenario2_wfq_virtual_runtime_cost_weighted_interleaving() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let fast_model = "fast-1b"; // ~2ms per pass
    let slow_model = "slow-7b"; // ~50ms per pass

    // Both in Throughput class, competing for cores 0x000F
    scheduler
        .admit_or_register_model(fast_model, "x100".to_owned(), 0x000F, None)
        .expect("register fast");
    scheduler
        .admit_or_register_model(slow_model, "x100".to_owned(), 0x000F, None)
        .expect("register slow");

    // Both have ongoing pending workload
    scheduler.inc_pending_requests(fast_model).unwrap();
    scheduler.inc_pending_requests(slow_model).unwrap();

    let mut fast_passes = 0;
    let mut slow_passes = 0;

    // Simulate 26 passes under continuous mutual contention
    for _ in 0..26 {
        // Evaluate candidate selection: pick whichever can schedule
        let maybe_fast = scheduler.try_schedule_forward_pass(fast_model).unwrap();
        let permit = if let Some(p) = maybe_fast {
            fast_passes += 1;
            scheduler.inc_pending_requests(fast_model).unwrap();
            (p, true)
        } else {
            let p = scheduler
                .try_schedule_forward_pass(slow_model)
                .unwrap()
                .expect("slow model must be eligible if fast was deferred");
            slow_passes += 1;
            scheduler.inc_pending_requests(slow_model).unwrap();
            (p, false)
        };

        if permit.1 {
            // Fast model executes in 2.0 ms
            scheduler.record_pass_completion(&permit.0, 2.0, 0.0, 1);
        } else {
            // Slow model executes in 50.0 ms
            scheduler.record_pass_completion(&permit.0, 50.0, 0.0, 10);
        }
    }

    let m_fast = scheduler.get_metrics(fast_model).unwrap();
    let m_slow = scheduler.get_metrics(slow_model).unwrap();

    // Fast model: 25 passes * 2ms = 50ms compute time
    // Slow model: 1 pass * 50ms = 50ms compute time
    // Exact 25:1 pass ratio reflects equal 50ms compute time allocation
    assert_eq!(fast_passes, 25, "fast model must receive 25 passes");
    assert_eq!(slow_passes, 1, "slow model must receive 1 pass");
    assert_eq!(m_fast.total_passes, 25);
    assert_eq!(m_slow.total_passes, 1);

    // Virtual runtimes track total compute consumed (50ms each)
    assert!((m_fast.virtual_runtime_ms - 50.0).abs() < 0.01);
    assert!((m_slow.virtual_runtime_ms - 50.0).abs() < 0.01);
}

#[test]
fn test_scenario3_client_priority_override_immunity() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let model_id = "configured-throughput";
    let mut profile = TuningProfile::conservative_default();
    profile.priority_class = PriorityClass::Throughput;

    scheduler
        .admit_or_register_model(model_id, "x100".to_owned(), 0x000F, Some(&profile))
        .expect("register succeeds");

    // Client maliciously or mistakenly sends `PriorityClass::Latency` header
    let resolved = scheduler
        .resolve_request_priority(model_id, Some(PriorityClass::Latency))
        .expect("resolve priority succeeds");

    // ADR-014 / US-10 Scenario 3: Client priority override MUST be ignored; operator profile is authoritative
    assert_eq!(
        resolved,
        PriorityClass::Throughput,
        "operator TuningProfile authority must override client priority claims"
    );

    // Verify when None is passed
    let resolved_none = scheduler
        .resolve_request_priority(model_id, None)
        .expect("resolve priority succeeds");
    assert_eq!(resolved_none, PriorityClass::Throughput);
}

#[test]
fn test_scenario4_starvation_prevention_on_new_admission() {
    let capacity = setup_capacity();
    let scheduler = MultiModelScheduler::new(capacity);

    let model_a = "established-model";
    scheduler
        .admit_or_register_model(model_a, "x100".to_owned(), 0x000F, None)
        .expect("register model A");

    // Model A runs 10 passes of 10ms (accumulating 100ms virtual runtime)
    for _ in 0..10 {
        let pass = scheduler
            .try_schedule_forward_pass(model_a)
            .unwrap()
            .expect("pass A granted");
        scheduler.record_pass_completion(&pass, 10.0, 0.0, 5);
    }

    let m_a = scheduler.get_metrics(model_a).unwrap();
    assert!((m_a.virtual_runtime_ms - 100.0).abs() < 0.01);

    // Now admit Model B
    let model_b = "new-model";
    scheduler
        .admit_or_register_model(model_b, "x100".to_owned(), 0x000F, None)
        .expect("register model B");

    let m_b = scheduler.get_metrics(model_b).unwrap();
    // Model B must catch up to min_virtual_runtime_ms (100.0) instead of starting at 0.0
    assert!(
        m_b.virtual_runtime_ms >= 100.0 - 0.01,
        "new model virtual runtime must catch up to system minimum to prevent starvation: {}",
        m_b.virtual_runtime_ms
    );
}
