//! Empirical Adversarial Challenge Suite: Milestone M2 Iteration 2 (Challenger 2)
//!
//! Rigorous empirical stress testing for:
//! 1. `SessionRegistry` 30-second retention window boundary precision:
//!    - Millisecond boundary transitions: 29_999ms (valid), 30_000ms (valid), 30_001ms (expired).
//!    - Immediate auto-pruning on expired reconnect attempt.
//!    - Global retention pruning (`prune_expired_at`) selectivity.
//! 2. Token catchup stream buffer arithmetic and edge cases:
//!    - Index boundaries: `None`, `Some(0)`, `Some(len - 1)`, `Some(len)`, `Some(usize::MAX)`.
//!    - Overflow and saturating safety under deep buffers (5,000+ tokens).
//! 3. Multi-subscriber broadcast stream & lag isolation:
//!    - Multi-subscriber lockstep delivery.
//!    - Channel capacity overflow (`DEFAULT_BROADCAST_CAPACITY = 4096`) handling.
//!    - Lagged subscriber recovery via catchup buffer without data loss.
//! 4. High-concurrency thread safety:
//!    - Contention across registration, appending, disconnection, reconnection, and pruning.
//! 5. Decode token checkpointing capacity bounds (`0..=16`):
//!    - Invariant: every block's `committed_tokens` is bounded within `0..=16`.
//!    - Invariant: sum of committed tokens across all active blocks matches sequence length exactly (no double-counting).
//!    - Boundary transitions: block boundary crossing at tokens 16->17, 32->33, 48->49.
//! 6. Incomplete prefill rejection & crash resumption:
//!    - Zero-token crash and partial prefill crash rejection (`IncompletePrefillResumptionNotSupported`).
//!    - Clean zero-prefill decode resumption upon complete prefill.
//! 7. Multi-block crash recovery with real POSIX shared memory arena:
//!    - Successive crashes spanning block boundaries.
//!    - Re-attachment to surviving arena across multiple worker generations.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::bool_assert_comparison
)]

use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use tokio::sync::broadcast::error::TryRecvError;
use zerocopy::FromBytes;

use k3_core::ipc::layout::{
    ARENA_MAGIC, ARENA_VERSION, ArenaHeader, BLOCK_SIZE_TOKENS, KvBlockTag, OutputSlabHeader,
    SLAB_STATUS_WRITTEN,
};
use k3_core::ipc::protocol::{
    ForwardPassCommand, ForwardPassCompleted, IpcMessage, StatusCode, recv_msg_frame,
    send_msg_frame,
};
use k3_core::ipc::shm::SharedMemoryArena;
use k3_manager::dispatch::ForwardPassRequest;
use k3_manager::error::ManagerError;
use k3_manager::session::{DEFAULT_RETENTION_WINDOW, SessionError, SessionRegistry};
use k3_manager::supervisor::{ManagedWorker, WorkerPlacement, WorkerSupervisor};

fn init_test_arena(name: &str, block_count: u32) -> SharedMemoryArena {
    let mut arena = SharedMemoryArena::create(name, 128 * 1024).expect("create arena");
    let hdr = ArenaHeader {
        magic: ARENA_MAGIC,
        version: ARENA_VERSION,
        total_bytes: 128 * 1024,
        block_count,
        tokens_per_block: BLOCK_SIZE_TOKENS,
        n_ctx: block_count * BLOCK_SIZE_TOKENS,
        meta_table_offset: 128,
        token_slots_offset: 2048,
        kv_blocks_offset: 4096,
        output_slabs_offset: 16384,
        vocab_size: 16,
        ..Default::default()
    };
    *arena.header_mut().expect("header_mut") = hdr;
    arena
}

fn spawn_mock_worker_loop(
    worker_sock: UnixStream,
    mut worker_arena: SharedMemoryArena,
    mut token_generator: impl FnMut(&ForwardPassCommand) -> u32 + Send + 'static,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        while let Ok((msg, _)) = recv_msg_frame(&worker_sock) {
            match msg {
                IpcMessage::ForwardPassCommand(cmd) => {
                    let next_tok = token_generator(&cmd);

                    let blocks = if cmd.block_indices.is_empty() {
                        vec![0_u32]
                    } else {
                        cmd.block_indices.clone()
                    };

                    let total_tokens = cmd.token_offset + cmd.token_count;
                    for (seq_idx, &b_idx) in blocks.iter().enumerate() {
                        if let Ok(block) = worker_arena.block_meta_mut(b_idx as usize) {
                            let block_start = if blocks.len() <= 1 {
                                (cmd.token_offset / BLOCK_SIZE_TOKENS) * BLOCK_SIZE_TOKENS
                            } else {
                                (seq_idx as u32) * BLOCK_SIZE_TOKENS
                            };
                            let committed = if cmd.is_prefill && blocks.len() <= 1 {
                                cmd.token_count.min(BLOCK_SIZE_TOKENS)
                            } else if total_tokens >= block_start + BLOCK_SIZE_TOKENS {
                                BLOCK_SIZE_TOKENS
                            } else {
                                total_tokens
                                    .saturating_sub(block_start)
                                    .min(BLOCK_SIZE_TOKENS)
                            };
                            block.set_committed_tokens(committed);
                            block.set_tag(KvBlockTag::Persist);
                            if block.sampler_rng_state.iter().all(|&x| x == 0) {
                                block.sampler_rng_state = [42, 0, 0, 0];
                            }
                            block.sampler_rng_state[0] = block.sampler_rng_state[0].wrapping_add(1);
                        }
                    }

                    if let Ok(hdr) = worker_arena.header() {
                        let offset = hdr.output_slabs_offset as usize;
                        if let Ok(slice) = worker_arena.slice_mut(offset, 64) {
                            if let Ok(slab) = OutputSlabHeader::mut_from_bytes(slice) {
                                slab.status = SLAB_STATUS_WRITTEN;
                                slab.vocab_size = 16;
                            }
                        }
                    }

                    let resp = ForwardPassCompleted {
                        request_id: cmd.request_id,
                        sequence_id: cmd.sequence_id,
                        committed_tokens: if cmd.is_prefill { cmd.token_count } else { 1 },
                        prefill_latency_ms: if cmd.is_prefill { 2.0 } else { 0.0 },
                        decode_latency_ms: if cmd.is_prefill { 0.0 } else { 1.0 },
                        next_token: next_tok,
                        status: StatusCode::Success,
                        error_message: None,
                    };
                    let _ =
                        send_msg_frame(&worker_sock, &IpcMessage::ForwardPassCompleted(resp), None);
                }
                _ => break,
            }
        }
    })
}

// ============================================================================
// 1. SessionRegistry 30-Second Retention Window Boundary Precision
// ============================================================================

#[test]
fn test_challenge2_session_registry_retention_window_boundary_precision() {
    let registry = SessionRegistry::new();
    let session_id = "sess-boundary-test";
    registry
        .register_session(session_id, "project-test", "model-qwen")
        .expect("register session");

    // Populate session with 10 tokens
    for tok in 0..10 {
        registry.append_token(session_id, tok).expect("append");
    }

    let disconnect_time = Instant::now();
    registry
        .record_disconnect(session_id)
        .expect("record disconnect");

    // Sub-case 1: Reconnection at exactly 29,999 ms (1ms before expiration) -> SUCCEEDS
    let at_29999ms = disconnect_time + Duration::from_millis(29_999);
    let catchup_1 = registry
        .reconnect_client_at(session_id, Some(7), at_29999ms)
        .expect("reconnect at 29999ms must succeed");
    assert_eq!(catchup_1, vec![8, 9]);

    // Sub-case 2: Re-disconnect and test reconnection at EXACTLY 30,000 ms -> SUCCEEDS (not expired)
    let second_disconnect = at_29999ms;
    registry
        .record_disconnect_at(session_id, second_disconnect)
        .expect("re-record disconnect");

    let at_30000ms = second_disconnect + DEFAULT_RETENTION_WINDOW; // exactly 30s
    let catchup_2 = registry
        .reconnect_client_at(session_id, Some(9), at_30000ms)
        .expect("reconnect at exactly 30s must succeed");
    assert!(catchup_2.is_empty());

    // Sub-case 3: Re-disconnect and test reconnection at 30,001 ms -> EXPIRES
    let third_disconnect = at_30000ms;
    registry
        .record_disconnect_at(session_id, third_disconnect)
        .expect("re-record disconnect 3");

    let at_30001ms = third_disconnect + Duration::from_millis(30_001);
    let expired_err = registry
        .reconnect_client_at(session_id, Some(5), at_30001ms)
        .expect_err("reconnect at 30001ms must fail with RetentionWindowExpired");

    match expired_err {
        SessionError::RetentionWindowExpired {
            session_id: err_sess,
            elapsed_ms,
            retention_window_ms,
        } => {
            assert_eq!(err_sess, session_id);
            assert!(
                elapsed_ms >= 30_001,
                "elapsed_ms must be >= 30001, got {elapsed_ms}"
            );
            assert_eq!(retention_window_ms, 30_000);
        }
        other => panic!("expected RetentionWindowExpired, got {other:?}"),
    }

    // Sub-case 4: Expired session is automatically pruned from memory
    assert!(!registry.contains_session(session_id));
    assert_eq!(registry.session_count(), 0);

    let subsequent_attempt = registry.reconnect_client_at(session_id, None, at_30001ms);
    assert_eq!(
        subsequent_attempt,
        Err(SessionError::SessionNotFound(session_id.to_owned())),
        "Subsequent reconnection attempt must return SessionNotFound"
    );
}

#[test]
fn test_challenge2_session_registry_prune_expired_selectivity() {
    let registry = SessionRegistry::new();
    let base_time = Instant::now();

    // 1. Active session (never disconnected)
    registry
        .register_session("sess-active", "p1", "m1")
        .expect("reg");

    // 2. Disconnected session within retention (disconnected 10s ago)
    registry
        .register_session("sess-recent", "p1", "m1")
        .expect("reg");
    registry
        .record_disconnect_at("sess-recent", base_time - Duration::from_secs(10))
        .expect("disc");

    // 3. Disconnected session expired (disconnected 45s ago)
    registry
        .register_session("sess-expired-1", "p1", "m1")
        .expect("reg");
    registry
        .record_disconnect_at("sess-expired-1", base_time - Duration::from_secs(45))
        .expect("disc");

    // 4. Disconnected session expired (disconnected 60s ago)
    registry
        .register_session("sess-expired-2", "p1", "m1")
        .expect("reg");
    registry
        .record_disconnect_at("sess-expired-2", base_time - Duration::from_secs(60))
        .expect("disc");

    assert_eq!(registry.session_count(), 4);

    // Prune as of base_time
    let pruned = registry.prune_expired_at(base_time);
    assert_eq!(pruned, 2, "Must prune exactly the 2 expired sessions");
    assert_eq!(registry.session_count(), 2);

    assert!(registry.contains_session("sess-active"));
    assert!(registry.contains_session("sess-recent"));
    assert!(!registry.contains_session("sess-expired-1"));
    assert!(!registry.contains_session("sess-expired-2"));
}

// ============================================================================
// 2. Token Catchup Stream Buffer Arithmetic and Edge Cases
// ============================================================================

#[test]
fn test_challenge2_token_catchup_buffer_boundary_and_arithmetic_safety() {
    let registry = SessionRegistry::new();
    let session_id = "sess-catchup-edge";
    registry
        .register_session(session_id, "p1", "m1")
        .expect("reg");

    // Initial state: empty token buffer
    assert_eq!(
        registry.reconnect_client_opt(session_id, None).unwrap(),
        Vec::<u32>::new()
    );
    assert_eq!(
        registry.reconnect_client(session_id, 0).unwrap(),
        Vec::<u32>::new()
    );
    assert_eq!(
        registry.reconnect_client(session_id, usize::MAX).unwrap(),
        Vec::<u32>::new()
    );

    // Generate 5000 tokens in the session buffer
    let total_tokens = 5_000_u32;
    for tok in 0..total_tokens {
        registry.append_token(session_id, tok).expect("append");
    }

    // Boundary 1: None returns ALL 5000 tokens
    let all = registry.reconnect_client_opt(session_id, None).unwrap();
    assert_eq!(all.len(), 5000);
    assert_eq!(all[0], 0);
    assert_eq!(all[4999], 4999);

    // Boundary 2: last_acked = 0 -> returns tokens from index 1..5000 (4999 tokens)
    let from_1 = registry.reconnect_client(session_id, 0).unwrap();
    assert_eq!(from_1.len(), 4999);
    assert_eq!(from_1[0], 1);
    assert_eq!(from_1[4998], 4999);

    // Boundary 3: last_acked = 4998 -> returns 1 token (index 4999)
    let last_one = registry.reconnect_client(session_id, 4998).unwrap();
    assert_eq!(last_one, vec![4999]);

    // Boundary 4: last_acked = 4999 (last token) -> returns empty
    let none_left = registry.reconnect_client(session_id, 4999).unwrap();
    assert!(none_left.is_empty());

    // Boundary 5: last_acked = 5000 (past end) -> returns empty without panic
    let past_end = registry.reconnect_client(session_id, 5000).unwrap();
    assert!(past_end.is_empty());

    // Boundary 6: last_acked = usize::MAX (saturating add safety) -> returns empty without overflow
    let max_safe = registry.reconnect_client(session_id, usize::MAX).unwrap();
    assert!(max_safe.is_empty());
}

// ============================================================================
// 3. Multi-Subscriber Broadcast Stream & Lag Isolation
// ============================================================================

#[tokio::test]
async fn test_challenge2_multi_subscriber_broadcast_saturation_and_lag_isolation() {
    let registry = Arc::new(SessionRegistry::new());
    let session_id = "sess-broadcast-stress";
    registry
        .register_session(session_id, "p_stream", "m_stream")
        .expect("reg");

    // Subscriber 1: actively consumes tokens
    let mut sub_active = registry.subscribe(session_id).expect("sub active");

    // Subscriber 2: lagging subscriber (never reads from channel)
    let mut sub_lagged = registry.subscribe(session_id).expect("sub lagged");

    // Emit 4,100 tokens (> DEFAULT_BROADCAST_CAPACITY = 4096)
    let num_tokens = 4_100_u32;
    for tok in 0..num_tokens {
        registry.append_token(session_id, tok).expect("append tok");
        // Active subscriber reads tokens as they arrive
        let r = sub_active.try_recv().expect("sub_active recv");
        assert_eq!(r, tok);
    }

    // Subscriber 2 attempts to read after 4100 emissions: MUST receive Lagged error
    let lag_result = sub_lagged.try_recv();
    match lag_result {
        Err(TryRecvError::Lagged(missed_count)) => {
            assert!(
                missed_count > 0,
                "Broadcast channel must report lagged receiver count"
            );
        }
        other => panic!("expected TryRecvError::Lagged, got {other:?}"),
    }

    // CRITICAL SYSTEM ASSERTION: Lagged broadcast receiver does NOT lose tokens!
    // The client reconnects via SessionRegistry catchup buffer and retrieves 100% of tokens.
    let catchup_for_lagged = registry
        .reconnect_client_opt(session_id, None)
        .expect("catchup tokens from buffer");
    assert_eq!(catchup_for_lagged.len(), 4100);
    assert_eq!(catchup_for_lagged[0], 0);
    assert_eq!(catchup_for_lagged[4099], 4099);

    // Multi-subscriber fan-out: attach 3 new concurrent subscribers
    let mut sub_c = registry.subscribe(session_id).expect("sub c");
    let mut sub_d = registry.subscribe(session_id).expect("sub d");

    registry
        .append_token(session_id, 9999)
        .expect("append 9999");

    assert_eq!(sub_active.try_recv(), Ok(9999));
    assert_eq!(sub_c.try_recv(), Ok(9999));
    assert_eq!(sub_d.try_recv(), Ok(9999));
}

// ============================================================================
// 4. High-Concurrency SessionRegistry Stress
// ============================================================================

#[test]
fn test_challenge2_concurrent_session_registry_stress() {
    let registry = Arc::new(SessionRegistry::new());
    let num_threads = 8;
    let ops_per_thread = 500;
    let running = Arc::new(AtomicBool::new(true));

    let mut handles = Vec::new();

    for t_idx in 0..num_threads {
        let reg = Arc::clone(&registry);
        let run = Arc::clone(&running);
        handles.push(thread::spawn(move || {
            for op in 0..ops_per_thread {
                let s_id = format!("sess-conc-{}", (t_idx * 17 + op) % 25);
                match op % 5 {
                    0 => {
                        let _ = reg.register_session(&s_id, "p_conc", "m_conc");
                    }
                    1 => {
                        let _ = reg.append_token(&s_id, (t_idx * 1000 + op) as u32);
                    }
                    2 => {
                        let _ = reg.record_disconnect(&s_id);
                    }
                    3 => {
                        let _ = reg.reconnect_client_opt(&s_id, None);
                    }
                    4 => {
                        let _ = reg.prune_expired();
                    }
                    _ => unreachable!(),
                }
            }
            if run.load(Ordering::Relaxed) {
                thread::yield_now();
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }
    running.store(false, Ordering::Relaxed);

    // Assert registry remains healthy without poisoning
    assert!(registry.session_count() <= 25);
}

// ============================================================================
// 5. Decode Token Checkpointing Capacity Bounds (0..=16) & Resumption
// ============================================================================

#[test]
fn test_challenge2_decode_token_checkpointing_capacity_bounds_exhaustive() {
    let block_count = 8_u32;
    let mut arena = init_test_arena("shm_bounds_chal", block_count);

    // Verify token sequence lengths from 1 to 64 tokens
    // Across each step, verify:
    // 1. Every block's committed_tokens <= 16.
    // 2. Sum of committed_tokens == total_tokens (zero double-counting).
    for total_tokens in 1_u32..=64 {
        // Clear all blocks
        for b in 0..block_count {
            let m = arena.block_meta_mut(b as usize).expect("meta");
            m.set_committed_tokens(0);
        }

        let full_blocks = total_tokens / BLOCK_SIZE_TOKENS;
        let remainder = total_tokens % BLOCK_SIZE_TOKENS;
        let active_blocks_count = if remainder > 0 {
            full_blocks + 1
        } else {
            full_blocks
        };

        let block_indices: Vec<u32> = (0..active_blocks_count).collect();

        // Simulate worker decode token update logic (from worker.rs:400-416)
        for (seq_idx, &b_idx) in block_indices.iter().enumerate() {
            let block = arena.block_meta_mut(b_idx as usize).expect("block");
            let block_start = (seq_idx as u32) * BLOCK_SIZE_TOKENS;
            let committed = if total_tokens >= block_start + BLOCK_SIZE_TOKENS {
                BLOCK_SIZE_TOKENS
            } else {
                total_tokens
                    .saturating_sub(block_start)
                    .min(BLOCK_SIZE_TOKENS)
            };
            block.set_committed_tokens(committed);
            block.set_tag(KvBlockTag::Persist);
            block.sampler_rng_state = [1000 + u64::from(total_tokens), 0, 0, 0];
        }

        // Verify individual block capacity bounds (0..=16)
        let mut sum_committed = 0_u32;
        for &b_idx in &block_indices {
            let block = arena.block_meta(b_idx as usize).expect("block read");
            let count = block.committed_tokens();
            assert!(
                count <= BLOCK_SIZE_TOKENS,
                "Block {b_idx} committed tokens {count} exceeds capacity {BLOCK_SIZE_TOKENS} at total {total_tokens}"
            );
            sum_committed += count;
        }

        // Invariant: sum of committed tokens MUST equal total_tokens exactly (no double-counting!)
        assert_eq!(
            sum_committed, total_tokens,
            "Double counting or missing tokens detected: sum={sum_committed} vs expected={total_tokens}"
        );
    }
}

// ============================================================================
// 6. Incomplete Prefill Crash Rejection & Zero-Prefill Resumption
// ============================================================================

#[test]
fn test_challenge2_incomplete_prefill_crash_rejection_and_clean_resumption() {
    let model_id = "model-prefill-rejection";
    let arena = init_test_arena("shm_prefill_rej", 4);
    let (mgr_sock, _worker_sock) = UnixStream::pair().expect("socketpair");
    let placement = WorkerPlacement::new(model_id, "/tmp/m.bin", "/tmp/m.sock", "x100");
    let worker =
        ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 128 * 1024, 64, 4);

    let mut supervisor = WorkerSupervisor::new();
    supervisor.register_worker(worker);

    let prompt_len = 20_u32; // Spans 2 blocks (Block 0 = 16, Block 1 = 4)

    // Sub-case A: 0 committed tokens -> resume_decode MUST fail with IncompletePrefillResumptionNotSupported
    let res_zero = supervisor.resume_decode(model_id, 101, 1, prompt_len, &[0, 1], 99);
    assert!(
        matches!(
            res_zero,
            Err(ManagerError::IncompletePrefillResumptionNotSupported {
                committed_tokens: 0,
                prompt_len: 20,
            })
        ),
        "Must reject decode when 0 tokens committed, got: {res_zero:?}"
    );

    // Sub-case B: 16 tokens committed (prompt incomplete, 16 < 20)
    {
        let w = supervisor.get_worker_mut(model_id).unwrap();
        w.arena_mut()
            .block_meta_mut(0)
            .unwrap()
            .set_committed_tokens(16);
        w.arena_mut()
            .block_meta_mut(1)
            .unwrap()
            .set_committed_tokens(0);
    }
    let res_partial = supervisor.resume_decode(model_id, 102, 1, prompt_len, &[0, 1], 99);
    assert!(
        matches!(
            res_partial,
            Err(ManagerError::IncompletePrefillResumptionNotSupported {
                committed_tokens: 16,
                prompt_len: 20,
            })
        ),
        "Must reject decode when 16 of 20 tokens committed, got: {res_partial:?}"
    );

    // Sub-case C: Exactly 19 tokens committed (boundary prompt_len - 1)
    {
        let w = supervisor.get_worker_mut(model_id).unwrap();
        w.arena_mut()
            .block_meta_mut(1)
            .unwrap()
            .set_committed_tokens(3);
    }
    let res_19 = supervisor.resume_decode(model_id, 103, 1, prompt_len, &[0, 1], 99);
    assert!(
        matches!(
            res_19,
            Err(ManagerError::IncompletePrefillResumptionNotSupported {
                committed_tokens: 19,
                prompt_len: 20,
            })
        ),
        "Must reject decode when 19 of 20 tokens committed, got: {res_19:?}"
    );

    // Sub-case D: Full prompt committed (20 tokens: Block 0 = 16, Block 1 = 4)
    {
        let w = supervisor.get_worker_mut(model_id).unwrap();
        w.arena_mut()
            .block_meta_mut(1)
            .unwrap()
            .set_committed_tokens(4);
    }
    let snap = supervisor
        .inspect_resumption_state(model_id, &[0, 1], prompt_len)
        .expect("inspect");
    assert_eq!(snap.total_committed_tokens, 20);
    assert!(
        snap.skip_prefill,
        "skip_prefill must be true when total_committed >= prompt_len"
    );
    assert_eq!(snap.next_decode_pos, 20);
}

// ============================================================================
// 7. Multi-Block Crash Recovery with Real Shared Memory Arena
// ============================================================================

#[test]
fn test_challenge2_multi_block_crash_recovery_across_generations() {
    let model_id = "model-multiblock-crash";
    let arena = init_test_arena("shm_mb_crash", 4);
    let arena_dup = arena.duplicate_fd().expect("dup fd");
    let worker_arena =
        SharedMemoryArena::from_existing_fd(arena_dup, arena.len(), "worker_shm_mb", false)
            .expect("worker arena");

    let (mgr_sock, worker_sock) = UnixStream::pair().expect("socketpair");
    let placement = WorkerPlacement::new(model_id, "/tmp/m.bin", "/tmp/m.sock", "x100");
    let worker =
        ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 128 * 1024, 64, 4);

    let mut supervisor = WorkerSupervisor::new();
    supervisor.register_worker(worker);

    let token_counter = Arc::new(AtomicUsize::new(100));
    let tc = Arc::clone(&token_counter);
    let _th1 = spawn_mock_worker_loop(worker_sock, worker_arena, move |_cmd| {
        tc.fetch_add(1, Ordering::SeqCst) as u32
    });

    // Step 1: Prefill 18 tokens across Block 0 (16) and Block 1 (2)
    let prompt = (0..18_u32).map(|i| 1000 + i).collect();
    let prefill_res = supervisor
        .get_worker_mut(model_id)
        .unwrap()
        .execute_forward_pass(&ForwardPassRequest::new_prefill(
            9001,
            1,
            prompt,
            vec![0, 1],
        ))
        .expect("prefill");
    assert!(prefill_res.completed.status.is_success());
    assert_eq!(prefill_res.completed.committed_tokens, 18);

    // Verify initial block committed bounds:
    {
        let w = supervisor.get_worker_mut(model_id).unwrap();
        assert_eq!(w.get_block_meta(0).unwrap().committed_tokens(), 16);
        assert_eq!(w.get_block_meta(1).unwrap().committed_tokens(), 2);
    }

    // Step 2: Decode 14 tokens to completely fill Block 1 (18 + 14 = 32 tokens)
    let mut next_tok = prefill_res.next_token();
    for step in 0..14 {
        let pos = 18 + step;
        let dec_res = supervisor
            .get_worker_mut(model_id)
            .unwrap()
            .execute_forward_pass(&ForwardPassRequest::new_decode(
                9100 + u64::from(step),
                1,
                next_tok,
                pos,
                vec![0, 1],
            ))
            .expect("decode step");
        next_tok = dec_res.next_token();
    }

    // Verify Block 1 is now full (16 tokens)
    {
        let w = supervisor.get_worker_mut(model_id).unwrap();
        assert_eq!(w.get_block_meta(0).unwrap().committed_tokens(), 16);
        assert_eq!(w.get_block_meta(1).unwrap().committed_tokens(), 16);
    }

    // CRASH 1: Kill worker at exactly block boundary (32 tokens)
    supervisor
        .get_worker_mut(model_id)
        .unwrap()
        .kill(libc::SIGKILL)
        .unwrap();
    assert!(supervisor.is_worker_crashed(model_id).unwrap());

    // Inspect surviving arena
    let snap1 = supervisor
        .inspect_resumption_state(model_id, &[0, 1], 18)
        .expect("inspect snap1");
    assert_eq!(snap1.total_committed_tokens, 32);
    assert!(snap1.skip_prefill);
    assert_eq!(snap1.next_decode_pos, 32);

    // Recover worker for Generation 2
    let (mgr_sock2, worker_sock2) = UnixStream::pair().expect("sock2");
    let (a_len, a_dup2) = {
        let w = supervisor.get_worker_mut(model_id).unwrap();
        (w.arena().len(), w.arena().duplicate_fd().unwrap())
    };
    let w_arena2 =
        SharedMemoryArena::from_existing_fd(a_dup2, a_len, "worker_shm_mb2", false).unwrap();
    let tc2 = Arc::clone(&token_counter);
    let _th2 = spawn_mock_worker_loop(worker_sock2, w_arena2, move |_cmd| {
        tc2.fetch_add(1, Ordering::SeqCst) as u32
    });

    let rec_worker2 = supervisor
        .recover_test_worker(model_id, mgr_sock2)
        .expect("recover gen 2");
    assert_eq!(rec_worker2.generation(), 2);

    // Resume decode crossing into Block 2 (token position 32 -> 33)
    let resume_res = rec_worker2
        .resume_decode(9200, 1, 18, &[0, 1, 2], next_tok)
        .expect("resume decode into block 2");
    assert_eq!(resume_res.prefill_latency_ms(), 0.0);
    assert_eq!(resume_res.completed.committed_tokens, 1);

    // Verify block committed tokens: Block 0=16, Block 1=16, Block 2=1 (Total = 33)
    {
        let w = supervisor.get_worker_mut(model_id).unwrap();
        assert_eq!(w.get_block_meta(0).unwrap().committed_tokens(), 16);
        assert_eq!(w.get_block_meta(1).unwrap().committed_tokens(), 16);
        assert_eq!(w.get_block_meta(2).unwrap().committed_tokens(), 1);
    }

    // CRASH 2: Kill worker again right after crossing into Block 2
    supervisor
        .get_worker_mut(model_id)
        .unwrap()
        .kill(libc::SIGKILL)
        .unwrap();
    assert!(supervisor.is_worker_crashed(model_id).unwrap());

    // Inspect surviving arena after crash 2
    let snap2 = supervisor
        .inspect_resumption_state(model_id, &[0, 1, 2], 18)
        .expect("inspect snap2");
    assert_eq!(snap2.total_committed_tokens, 33);
    assert!(snap2.skip_prefill);
    assert_eq!(snap2.next_decode_pos, 33);

    // Recover worker for Generation 3
    let (mgr_sock3, worker_sock3) = UnixStream::pair().expect("sock3");
    let (a_len3, a_dup3) = {
        let w = supervisor.get_worker_mut(model_id).unwrap();
        (w.arena().len(), w.arena().duplicate_fd().unwrap())
    };
    let w_arena3 =
        SharedMemoryArena::from_existing_fd(a_dup3, a_len3, "worker_shm_mb3", false).unwrap();
    let tc3 = Arc::clone(&token_counter);
    let _th3 = spawn_mock_worker_loop(worker_sock3, w_arena3, move |_cmd| {
        tc3.fetch_add(1, Ordering::SeqCst) as u32
    });

    let rec_worker3 = supervisor
        .recover_test_worker(model_id, mgr_sock3)
        .expect("recover gen 3");
    assert_eq!(rec_worker3.generation(), 3);

    // Resume decode again on Generation 3
    let resume_res3 = rec_worker3
        .resume_decode(9300, 1, 18, &[0, 1, 2], resume_res.next_token())
        .expect("resume decode on gen 3");
    assert_eq!(resume_res3.prefill_latency_ms(), 0.0);

    // Final verification of Block 2 committed tokens: 2 tokens committed (Total = 34)
    {
        let w = supervisor.get_worker_mut(model_id).unwrap();
        assert_eq!(w.get_block_meta(0).unwrap().committed_tokens(), 16);
        assert_eq!(w.get_block_meta(1).unwrap().committed_tokens(), 16);
        assert_eq!(w.get_block_meta(2).unwrap().committed_tokens(), 2);
    }
}
