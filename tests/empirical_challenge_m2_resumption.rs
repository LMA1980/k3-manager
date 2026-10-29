//! Empirical Challenge Test Suite for Milestone M2: Resumable Generation & Fault Recovery.
//!
//! Validates:
//! 1. Mid-generation worker crash: Crash worker after committing N tokens. Re-attach supervisor
//!    to surviving POSIX shared memory arena. Verify decode resumes directly at token N without re-executing prefill.
//! 2. Worker crash at token zero (during prefill): Verify supervisor detects uncommitted state, rejects
//!    premature decode resumption with explicit error, and safely re-executes prefill.
//! 3. Client reconnect within 30-second retention window: Verify disconnected client reconnecting receives
//!    buffered catchup tokens without generation stalling.
//! 4. Client reconnect after retention window expiry (>30s): Verify session is cleaned up and expired
//!    request rejected cleanly.
//! 5. Deterministic sampler PRNG continuation: Verify decode tokens and PRNG states generated
//!    after resumption match identical uninterrupted decode sequence bit-for-bit.
//! 6. Cascaded multi-crash resumption stress: Verify surviving arena context through successive crashes
//!    and restarts across multiple worker generations.

#![allow(
    clippy::pedantic,
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::bool_assert_comparison
)]

use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

use zerocopy::FromBytes;

use k3_core::ipc::layout::{
    ARENA_MAGIC, ARENA_VERSION, ArenaHeader, KvBlockTag, OutputSlabHeader, SLAB_STATUS_WRITTEN,
};
use k3_core::ipc::protocol::{
    ForwardPassCommand, ForwardPassCompleted, IpcMessage, StatusCode, recv_msg_frame,
    send_msg_frame,
};
use k3_core::ipc::shm::SharedMemoryArena;
use k3_manager::dispatch::ForwardPassRequest;
use k3_manager::error::ManagerError;
use k3_manager::session::{SessionError, SessionRegistry};
use k3_manager::supervisor::{ManagedWorker, WorkerPlacement, WorkerSupervisor};

fn init_challenge_arena(name: &str) -> SharedMemoryArena {
    let mut arena = SharedMemoryArena::create(name, 65536).expect("create arena");
    let hdr = ArenaHeader {
        magic: ARENA_MAGIC,
        version: ARENA_VERSION,
        total_bytes: 65536,
        block_count: 4,
        tokens_per_block: 16,
        n_ctx: 64,
        meta_table_offset: 128,
        token_slots_offset: 1024,
        kv_blocks_offset: 2048,
        output_slabs_offset: 4096,
        vocab_size: 10,
        ..Default::default()
    };
    *arena.header_mut().expect("header_mut") = hdr;
    arena
}

fn spawn_mock_worker_thread(
    worker_sock: UnixStream,
    mut worker_arena: SharedMemoryArena,
    mut next_token_generator: impl FnMut(&ForwardPassCommand) -> u32 + Send + 'static,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while let Ok((msg, _)) = recv_msg_frame(&worker_sock) {
            match msg {
                IpcMessage::ForwardPassCommand(cmd) => {
                    let next_tok = next_token_generator(&cmd);

                    let blocks = if cmd.block_indices.is_empty() {
                        vec![0_u32]
                    } else {
                        cmd.block_indices.clone()
                    };
                    let total_tokens = cmd.token_offset + cmd.token_count;
                    for (seq_idx, &b_idx) in blocks.iter().enumerate() {
                        if let Ok(block) = worker_arena.block_meta_mut(b_idx as usize) {
                            let block_start = if blocks.len() <= 1 {
                                (cmd.token_offset / 16) * 16
                            } else {
                                (seq_idx as u32) * 16
                            };
                            let committed = if cmd.is_prefill && blocks.len() <= 1 {
                                cmd.token_count.min(16)
                            } else if total_tokens >= block_start + 16 {
                                16
                            } else {
                                total_tokens.saturating_sub(block_start).min(16)
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
                                slab.vocab_size = 10;
                            }
                        }
                    }

                    let resp = ForwardPassCompleted {
                        request_id: cmd.request_id,
                        sequence_id: cmd.sequence_id,
                        committed_tokens: if cmd.is_prefill { cmd.token_count } else { 1 },
                        prefill_latency_ms: if cmd.is_prefill { 1.5 } else { 0.0 },
                        decode_latency_ms: if cmd.is_prefill { 0.0 } else { 0.8 },
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
// 1. Mid-generation worker crash: Crash worker after committing N tokens.
//    Re-attach supervisor to surviving POSIX shm arena. Verify decode resumes directly
//    at token N without re-executing prefill.
// ============================================================================

#[test]
fn test_empirical_challenge_1_mid_generation_worker_crash_zero_prefill_resumption() {
    let model_id = "model-midgen-crash";
    let arena = init_challenge_arena("shm_chal_1");
    let arena_dup = arena.duplicate_fd().expect("dup fd");
    let worker_arena =
        SharedMemoryArena::from_existing_fd(arena_dup, arena.len(), "worker_shm_1", false)
            .expect("worker arena");

    let (mgr_sock, worker_sock) = UnixStream::pair().expect("socketpair");
    let placement = WorkerPlacement::new(model_id, "/tmp/model.bin", "/tmp/model.sock", "x100");
    let worker = ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 65536, 64, 4);

    let mut supervisor = WorkerSupervisor::new();
    supervisor.register_worker(worker);

    let mut token_counter = 200_u32;
    let _th1 = spawn_mock_worker_thread(worker_sock, worker_arena, move |_cmd| {
        let tok = token_counter;
        token_counter += 1;
        tok
    });

    // Step 1: Commit prompt tokens via prefill forward pass (prompt_len = 6)
    let prompt_tokens = vec![101_u32, 102, 103, 104, 105, 106];
    let prefill_req = ForwardPassRequest::new_prefill(1001, 1, prompt_tokens, vec![0]);
    let prefill_res = supervisor
        .get_worker_mut(model_id)
        .expect("worker ref")
        .execute_forward_pass(&prefill_req)
        .expect("prefill pass");
    assert!(prefill_res.completed.status.is_success());
    assert_eq!(prefill_res.completed.committed_tokens, 6);
    assert!(prefill_res.prefill_latency_ms() > 0.0);
    assert_eq!(prefill_res.decode_latency_ms(), 0.0);

    let mut next_token = prefill_res.next_token();

    // Step 2: Commit 3 decode tokens in sequence (total committed = 6 + 3 = 9 tokens)
    for step in 0..3 {
        let decode_req =
            ForwardPassRequest::new_decode(1002 + step, 1, next_token, 6 + step as u32, vec![0]);
        let decode_res = supervisor
            .get_worker_mut(model_id)
            .expect("worker ref")
            .execute_forward_pass(&decode_req)
            .expect("decode step");
        assert!(decode_res.completed.status.is_success());
        assert_eq!(decode_res.completed.committed_tokens, 1);
        next_token = decode_res.next_token();
    }

    // Verify 9 tokens committed in Block 0
    let pre_crash_rng = {
        let worker_ref = supervisor.get_worker_mut(model_id).expect("worker ref");
        let b0 = worker_ref.get_block_meta(0).expect("block 0");
        assert_eq!(b0.committed_tokens(), 9);
        assert_eq!(b0.tag_enum(), Some(KvBlockTag::Persist));
        b0.sampler_rng_state
    };

    // Step 3: Simulate worker crash mid-generation (SIGKILL)
    supervisor
        .get_worker_mut(model_id)
        .expect("worker ref")
        .kill(libc::SIGKILL)
        .expect("kill worker");

    // Step 4: Verify supervisor detects worker crash
    assert!(supervisor.is_worker_crashed(model_id).unwrap());
    assert!(!supervisor.is_worker_alive(model_id));

    // Step 5: Inspect surviving resumption state before recovery
    let snap = supervisor
        .inspect_resumption_state(model_id, &[0], 6)
        .expect("inspect resumption state");
    assert_eq!(snap.total_committed_tokens, 9);
    assert!(snap.skip_prefill);
    assert_eq!(snap.next_decode_pos, 9);
    assert_eq!(snap.sampler_rng_state, pre_crash_rng);

    // Step 6: Re-attach supervisor to surviving arena via recover_test_worker
    let (mgr_sock2, worker_sock2) = UnixStream::pair().expect("socketpair 2");
    let (arena_len, arena_dup2) = {
        let worker_ref = supervisor.get_worker_mut(model_id).unwrap();
        (
            worker_ref.arena().len(),
            worker_ref.arena().duplicate_fd().expect("dup fd 2"),
        )
    };
    let worker_arena2 =
        SharedMemoryArena::from_existing_fd(arena_dup2, arena_len, "worker_shm_2", false)
            .expect("worker arena 2");

    let _th2 = spawn_mock_worker_thread(worker_sock2, worker_arena2, move |_cmd| 305);

    let recovered_worker = supervisor
        .recover_test_worker(model_id, mgr_sock2)
        .expect("recover worker");
    assert!(recovered_worker.is_alive());
    assert_eq!(recovered_worker.generation(), 2);

    // Step 7: Verify decode resumes directly at token N=9 WITHOUT re-executing prefill
    let resume_res = recovered_worker
        .resume_decode(2001, 1, 6, &[0], next_token)
        .expect("resume decode pass");

    assert!(resume_res.completed.status.is_success());
    assert_eq!(resume_res.completed.committed_tokens, 1);
    assert_eq!(resume_res.prefill_latency_ms(), 0.0);
    assert!(resume_res.decode_latency_ms() > 0.0);

    // Verify token count is now 10 (9 + 1) in surviving arena
    let b0_after = recovered_worker.get_block_meta(0).expect("block 0 after");
    assert_eq!(b0_after.committed_tokens(), 10);
    assert_eq!(b0_after.tag_enum(), Some(KvBlockTag::Persist));

    // Step 8: Subsequent decode pass continues smoothly
    let next_decode_req =
        ForwardPassRequest::new_decode(2002, 1, resume_res.next_token(), 10, vec![0]);
    let cont_res = recovered_worker
        .execute_forward_pass(&next_decode_req)
        .expect("subsequent decode");
    assert!(cont_res.completed.status.is_success());
    assert_eq!(cont_res.completed.committed_tokens, 1);
    assert_eq!(
        recovered_worker
            .get_block_meta(0)
            .expect("b0")
            .committed_tokens(),
        11
    );
}

// ============================================================================
// 2. Worker crash at token zero (during prefill): Verify supervisor detects
//    uncommitted state, rejects premature decode resumption, and safely re-executes prefill.
// ============================================================================

#[test]
fn test_empirical_challenge_2_worker_crash_at_token_zero_safe_prefill_reexecution() {
    let model_id = "model-crash-token-zero";
    let arena = init_challenge_arena("shm_chal_2");
    let (mgr_sock, _worker_sock) = UnixStream::pair().expect("socketpair");
    let placement = WorkerPlacement::new(model_id, "/tmp/model.bin", "/tmp/model.sock", "x100");
    let worker = ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 65536, 64, 4);

    let mut supervisor = WorkerSupervisor::new();
    supervisor.register_worker(worker);

    // Initial state: token zero, 0 committed tokens in Block 0
    {
        let worker_ref = supervisor.get_worker_mut(model_id).expect("worker ref");
        let b0 = worker_ref.get_block_meta(0).expect("block 0");
        assert_eq!(b0.committed_tokens(), 0);
        worker_ref
            .kill(libc::SIGKILL)
            .expect("kill worker at token zero");
    }
    assert!(supervisor.is_worker_crashed(model_id).unwrap());

    // Calling resume_decode when prefill is incomplete returns explicit error (Task 4)
    let resume_attempt = supervisor.resume_decode(model_id, 3000, 1, 5, &[0], 100);
    assert!(
        matches!(
            resume_attempt,
            Err(ManagerError::IncompletePrefillResumptionNotSupported {
                committed_tokens: 0,
                prompt_len: 5,
            })
        ),
        "Resume decode MUST fail when prefill is incomplete, got: {resume_attempt:?}"
    );

    // Verify supervisor detects uncommitted state: skip_prefill == false
    let snap = supervisor
        .inspect_resumption_state(model_id, &[0], 5)
        .expect("inspect resumption");
    assert_eq!(snap.total_committed_tokens, 0);
    assert!(!snap.skip_prefill);
    assert_eq!(snap.next_decode_pos, 0);

    // Re-attach supervisor to surviving arena via recover_test_worker
    let (mgr_sock2, worker_sock2) = UnixStream::pair().expect("socketpair 2");
    let (arena_len, arena_dup2) = {
        let worker_ref = supervisor.get_worker_mut(model_id).unwrap();
        (
            worker_ref.arena().len(),
            worker_ref.arena().duplicate_fd().expect("dup fd 2"),
        )
    };
    let worker_arena2 =
        SharedMemoryArena::from_existing_fd(arena_dup2, arena_len, "worker_shm_2_b", false)
            .expect("worker arena 2");

    let _th2 = spawn_mock_worker_thread(worker_sock2, worker_arena2, move |_cmd| 555);

    let recovered_worker = supervisor
        .recover_test_worker(model_id, mgr_sock2)
        .expect("recover worker");
    assert!(recovered_worker.is_alive());
    assert_eq!(recovered_worker.generation(), 2);

    // Supervisor safely re-executes prefill since skip_prefill is false
    let prompt_tokens = vec![501_u32, 502, 503, 504, 505];
    let prefill_req = ForwardPassRequest::new_prefill(3001, 1, prompt_tokens, vec![0]);
    let prefill_res = recovered_worker
        .execute_forward_pass(&prefill_req)
        .expect("safely re-execute prefill");

    assert!(prefill_res.completed.status.is_success());
    assert_eq!(prefill_res.completed.committed_tokens, 5);
    assert!(prefill_res.prefill_latency_ms() > 0.0);
    assert_eq!(prefill_res.decode_latency_ms(), 0.0);

    // Verify block 0 is now committed with 5 tokens
    let b0_recovered = recovered_worker.get_block_meta(0).expect("b0 recovered");
    assert_eq!(b0_recovered.committed_tokens(), 5);
    assert_eq!(b0_recovered.tag_enum(), Some(KvBlockTag::Persist));

    // Resumption state now correctly reflects committed prefill
    let snap_after = supervisor
        .inspect_resumption_state(model_id, &[0], 5)
        .expect("inspect after prefill");
    assert_eq!(snap_after.total_committed_tokens, 5);
    assert!(snap_after.skip_prefill);
    assert_eq!(snap_after.next_decode_pos, 5);
}

// ============================================================================
// 3. Client reconnect within 30-second retention window: Verify disconnected client
//    reconnecting receives buffered catchup tokens without generation stalling.
// ============================================================================

#[tokio::test]
async fn test_empirical_challenge_3_client_reconnect_within_retention_window_no_stall() {
    let registry = Arc::new(SessionRegistry::new());
    let session_id = "sess-reconnect-window";
    registry
        .register_session(session_id, "project-reconnect", "qwen-2.5-7b")
        .expect("register session");

    // Initial subscriber receives first batch
    let mut initial_rx = registry.subscribe(session_id).expect("subscribe");

    // Producer emits 5 tokens: [100, 101, 102, 103, 104]
    for tok in 100..105 {
        registry.append_token(session_id, tok).expect("emit");
    }

    let mut client_received = Vec::new();
    for _ in 0..5 {
        if let Ok(tok) = initial_rx.try_recv() {
            client_received.push(tok);
        }
    }
    assert_eq!(client_received, vec![100, 101, 102, 103, 104]);
    let last_acked_idx = 4; // token 104 at index 4

    // Client disconnects at T0
    let disconnect_time = Instant::now();
    drop(initial_rx);
    registry.record_disconnect(session_id).expect("disconnect");

    // Worker continues generating in background WITHOUT STALLING:
    // Emits tokens [105, 106, 107, 108, 109, 110]
    let bg_reg = Arc::clone(&registry);
    let produce_handle = tokio::spawn(async move {
        for tok in 105..111 {
            bg_reg
                .append_token("sess-reconnect-window", tok)
                .expect("emit bg token");
        }
    });

    produce_handle.await.expect("join producer");

    // Client reconnects within 30-second retention window (e.g. after ~5ms < 30s)
    let reconnect_time = Instant::now();
    assert!(reconnect_time.duration_since(disconnect_time) < Duration::from_secs(30));

    let mut reconnected_rx = registry
        .subscribe(session_id)
        .expect("reconnected subscribe");
    let catchup_tokens = registry
        .reconnect_client(session_id, last_acked_idx)
        .expect("reconnect within retention window");

    // Assert client receives missed catchup tokens without gaps
    assert_eq!(
        catchup_tokens,
        vec![105, 106, 107, 108, 109, 110],
        "Client must receive all generated tokens produced while disconnected"
    );

    // Subsequent live tokens stream directly to reconnected client
    registry.append_token(session_id, 111).expect("emit live");
    registry.append_token(session_id, 112).expect("emit live");

    let live_tok1 = reconnected_rx.try_recv().expect("recv live 1");
    let live_tok2 = reconnected_rx.try_recv().expect("recv live 2");
    assert_eq!(live_tok1, 111);
    assert_eq!(live_tok2, 112);

    client_received.extend(catchup_tokens);
    client_received.push(live_tok1);
    client_received.push(live_tok2);

    assert_eq!(
        client_received,
        vec![
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112
        ],
        "Client must assemble complete uninterrupted token sequence"
    );
}

// ============================================================================
// 4. Client reconnect after retention window expiry (>30s): Verify session is
//    cleaned up and expired request rejected cleanly.
// ============================================================================

#[test]
fn test_empirical_challenge_4_client_reconnect_after_retention_window_expiry_rejected() {
    let registry = SessionRegistry::new();
    let session_id = "sess-expire-test";
    registry
        .register_session(session_id, "project-expire", "qwen-2.5-7b")
        .expect("register session");

    for tok in 200..205 {
        registry.append_token(session_id, tok).expect("emit");
    }

    let disconnect_time = Instant::now();
    registry
        .record_disconnect(session_id)
        .expect("mark disconnected");

    // Simulate reconnect after 35 seconds (> 30s retention window)
    let expired_time = disconnect_time + Duration::from_secs(35);

    let res = registry.reconnect_client_at(session_id, Some(4), expired_time);
    assert!(
        matches!(res, Err(SessionError::RetentionWindowExpired { .. })),
        "Expired reconnect MUST be rejected with RetentionWindowExpired"
    );

    // Verify session has been cleaned up and purged from memory
    let second_attempt = registry.reconnect_client_at(session_id, Some(4), expired_time);
    assert_eq!(
        second_attempt,
        Err(SessionError::SessionNotFound(session_id.to_owned())),
        "Purged session must no longer be found in active sessions"
    );

    // Verify new clean session can be created without conflict
    registry
        .register_session(session_id, "project-expire", "qwen-2.5-7b")
        .expect("register clean session");
    registry
        .append_token(session_id, 300)
        .expect("emit in new session");
    let catchup = registry
        .reconnect_client_opt(session_id, None)
        .expect("reconnect on clean session");
    assert_eq!(catchup, vec![300]);
}

// ============================================================================
// 5. Deterministic sampler PRNG continuation: Verify decode tokens generated
//    after resumption match identical uninterrupted decode sequence.
// ============================================================================

#[test]
fn test_empirical_challenge_5_deterministic_sampler_prng_continuation_across_crash() {
    let prompt_tokens = vec![401_u32, 402, 403, 404, 405, 406];

    // Phase A: Generate uninterrupted baseline sequence (Prompt + 5 Decode steps)
    let model_a = "model-det-baseline";
    let arena_a = init_challenge_arena("shm_chal_5_a");
    let arena_dup_a = arena_a.duplicate_fd().expect("dup fd a");
    let worker_arena_a =
        SharedMemoryArena::from_existing_fd(arena_dup_a, arena_a.len(), "worker_shm_5_a", false)
            .expect("worker arena a");

    let (mgr_sock_a, worker_sock_a) = UnixStream::pair().expect("socketpair a");
    let placement_a = WorkerPlacement::new(model_a, "/tmp/model.bin", "/tmp/model.sock", "x100");
    let worker_a =
        ManagedWorker::create_test_worker(placement_a, mgr_sock_a, arena_a, 1, 65536, 64, 4);

    let mut supervisor_a = WorkerSupervisor::new();
    supervisor_a.register_worker(worker_a);

    let mut token_gen_a = 500_u32;
    let _th_a = spawn_mock_worker_thread(worker_sock_a, worker_arena_a, move |_cmd| {
        let t = token_gen_a;
        token_gen_a += 1;
        t
    });

    let prefill_req_a = ForwardPassRequest::new_prefill(5001, 1, prompt_tokens.clone(), vec![0]);
    let prefill_res_a = supervisor_a
        .get_worker_mut(model_a)
        .expect("worker a")
        .execute_forward_pass(&prefill_req_a)
        .expect("prefill a");
    assert!(prefill_res_a.completed.status.is_success());

    let mut curr_token_a = prefill_res_a.next_token();
    let mut baseline_tokens = Vec::new();
    let mut baseline_rng_states = Vec::new();

    for step in 0..5 {
        let decode_req =
            ForwardPassRequest::new_decode(5002 + step, 1, curr_token_a, 6 + step as u32, vec![0]);
        let decode_res = supervisor_a
            .get_worker_mut(model_a)
            .expect("worker a")
            .execute_forward_pass(&decode_req)
            .expect("decode a");
        curr_token_a = decode_res.next_token();
        baseline_tokens.push(curr_token_a);

        let b0 = supervisor_a
            .get_worker_mut(model_a)
            .expect("worker a")
            .get_block_meta(0)
            .expect("b0");
        baseline_rng_states.push(b0.sampler_rng_state);
    }

    assert_eq!(baseline_tokens.len(), 5);
    assert_eq!(baseline_rng_states.len(), 5);

    // Phase B: Interrupted sequence with mid-generation crash at step 2,
    //          then zero-prefill resumption for steps 3..5
    let model_b = "model-det-resumed";
    let arena_b = init_challenge_arena("shm_chal_5_b");
    let arena_dup_b = arena_b.duplicate_fd().expect("dup fd b");
    let worker_arena_b =
        SharedMemoryArena::from_existing_fd(arena_dup_b, arena_b.len(), "worker_shm_5_b", false)
            .expect("worker arena b");

    let (mgr_sock_b, worker_sock_b) = UnixStream::pair().expect("socketpair b");
    let placement_b = WorkerPlacement::new(model_b, "/tmp/model.bin", "/tmp/model.sock", "x100");
    let worker_b =
        ManagedWorker::create_test_worker(placement_b, mgr_sock_b, arena_b, 1, 65536, 64, 4);

    let mut supervisor_b = WorkerSupervisor::new();
    supervisor_b.register_worker(worker_b);

    let mut token_gen_b = 500_u32;
    let _th_b = spawn_mock_worker_thread(worker_sock_b, worker_arena_b, move |_cmd| {
        let t = token_gen_b;
        token_gen_b += 1;
        t
    });

    let prefill_req_b = ForwardPassRequest::new_prefill(6001, 1, prompt_tokens.clone(), vec![0]);
    let prefill_res_b = supervisor_b
        .get_worker_mut(model_b)
        .expect("worker b")
        .execute_forward_pass(&prefill_req_b)
        .expect("prefill b");
    assert_eq!(prefill_res_b.next_token(), prefill_res_a.next_token());

    let mut curr_token_b = prefill_res_b.next_token();
    let mut resumed_tokens = Vec::new();
    let mut resumed_rng_states = Vec::new();

    // Run first 2 decode steps before crash
    for step in 0..2 {
        let decode_req =
            ForwardPassRequest::new_decode(6002 + step, 1, curr_token_b, 6 + step as u32, vec![0]);
        let decode_res = supervisor_b
            .get_worker_mut(model_b)
            .expect("worker b")
            .execute_forward_pass(&decode_req)
            .expect("decode b pre-crash");
        curr_token_b = decode_res.next_token();
        resumed_tokens.push(curr_token_b);

        let b0 = supervisor_b
            .get_worker_mut(model_b)
            .expect("worker b")
            .get_block_meta(0)
            .expect("b0");
        resumed_rng_states.push(b0.sampler_rng_state);
    }

    assert_eq!(resumed_tokens, baseline_tokens[0..2]);
    assert_eq!(resumed_rng_states, baseline_rng_states[0..2]);

    // FAULT INJECTION: Kill worker B mid-generation via SIGKILL
    supervisor_b
        .get_worker_mut(model_b)
        .expect("worker b")
        .kill(libc::SIGKILL)
        .expect("kill worker b");
    assert!(supervisor_b.is_worker_crashed(model_b).unwrap());

    // Inspect resumption snapshot from surviving shared memory
    let snap = supervisor_b
        .inspect_resumption_state(model_b, &[0], 6)
        .expect("inspect resumption b");
    assert_eq!(snap.total_committed_tokens, 8); // 6 prompt + 2 decode
    assert!(snap.skip_prefill);
    assert_eq!(snap.next_decode_pos, 8);
    assert_eq!(snap.sampler_rng_state, baseline_rng_states[1]);

    // Re-attach supervisor to surviving arena via recover_test_worker
    let (mgr_sock_b2, worker_sock_b2) = UnixStream::pair().expect("socketpair b2");
    let (arena_len_b, arena_dup_b2) = {
        let worker_ref = supervisor_b.get_worker_mut(model_b).unwrap();
        (
            worker_ref.arena().len(),
            worker_ref.arena().duplicate_fd().expect("dup fd b2"),
        )
    };
    let worker_arena_b2 =
        SharedMemoryArena::from_existing_fd(arena_dup_b2, arena_len_b, "worker_shm_5_b2", false)
            .expect("worker arena b2");

    let mut token_gen_b2 = 503_u32;
    let _th_b2 = spawn_mock_worker_thread(worker_sock_b2, worker_arena_b2, move |_cmd| {
        let t = token_gen_b2;
        token_gen_b2 += 1;
        t
    });

    let recovered_worker_b = supervisor_b
        .recover_test_worker(model_b, mgr_sock_b2)
        .expect("recover worker b");
    assert_eq!(recovered_worker_b.generation(), 2);

    // Resume decode step 2 without prefill
    let resume_res = recovered_worker_b
        .resume_decode(7001, 1, 6, &[0], curr_token_b)
        .expect("resume decode b");
    assert_eq!(resume_res.prefill_latency_ms(), 0.0);
    assert!(resume_res.decode_latency_ms() > 0.0);

    curr_token_b = resume_res.next_token();
    resumed_tokens.push(curr_token_b);
    let b0_rec = recovered_worker_b.get_block_meta(0).expect("b0 rec");
    resumed_rng_states.push(b0_rec.sampler_rng_state);

    // Verify step 2 matches baseline exactly
    assert_eq!(curr_token_b, baseline_tokens[2]);
    assert_eq!(b0_rec.sampler_rng_state, baseline_rng_states[2]);

    // Continue decode steps 3 and 4
    for step in 3..5 {
        let decode_req =
            ForwardPassRequest::new_decode(7002 + step, 1, curr_token_b, 6 + step as u32, vec![0]);
        let decode_res = recovered_worker_b
            .execute_forward_pass(&decode_req)
            .expect("continue decode b");
        curr_token_b = decode_res.next_token();
        resumed_tokens.push(curr_token_b);

        let b0 = recovered_worker_b.get_block_meta(0).expect("b0");
        resumed_rng_states.push(b0.sampler_rng_state);
    }

    assert_eq!(
        resumed_tokens, baseline_tokens,
        "Resumed decode tokens must be bit-exact identical to uninterrupted baseline"
    );
    assert_eq!(
        resumed_rng_states, baseline_rng_states,
        "Resumed PRNG states must be bit-exact identical to uninterrupted baseline"
    );
}

// ============================================================================
// 6. Cascaded multi-crash resumption stress: Verify surviving arena context
//    through successive crashes and restarts across multiple worker generations.
// ============================================================================

#[test]
fn test_empirical_challenge_6_cascaded_multi_crash_resumption_stress() {
    let model_id = "model-cascaded-crashes";
    let arena = init_challenge_arena("shm_chal_6");
    let arena_dup = arena.duplicate_fd().expect("dup fd 6");
    let worker_arena =
        SharedMemoryArena::from_existing_fd(arena_dup, arena.len(), "worker_shm_6", false)
            .expect("worker arena");

    let (mgr_sock, worker_sock) = UnixStream::pair().expect("socketpair 6");
    let placement = WorkerPlacement::new(model_id, "/tmp/model.bin", "/tmp/model.sock", "x100");
    let worker = ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 65536, 64, 4);

    let mut supervisor = WorkerSupervisor::new();
    supervisor.register_worker(worker);

    let mut tok_counter = 700_u32;
    let _th = spawn_mock_worker_thread(worker_sock, worker_arena, move |_cmd| {
        let t = tok_counter;
        tok_counter += 1;
        t
    });

    // Prefill 4 tokens
    let prompt = vec![10_u32, 20, 30, 40];
    let prefill_res = supervisor
        .get_worker_mut(model_id)
        .expect("worker ref")
        .execute_forward_pass(&ForwardPassRequest::new_prefill(8001, 1, prompt, vec![0]))
        .expect("prefill");
    let mut current_tok = prefill_res.next_token();

    // Cascaded crash-resume cycle across 3 successive generations (generations 1 -> 2 -> 3 -> 4)
    for cycle in 1_u32..=3 {
        let pos = supervisor
            .get_worker_mut(model_id)
            .expect("worker ref")
            .get_block_meta(0)
            .unwrap()
            .committed_tokens();
        let decode_res = supervisor
            .get_worker_mut(model_id)
            .expect("worker ref")
            .execute_forward_pass(&ForwardPassRequest::new_decode(
                8100 + u64::from(cycle),
                1,
                current_tok,
                pos,
                vec![0],
            ))
            .expect("decode");
        current_tok = decode_res.next_token();

        let expected_tokens = 4_u32 + cycle;
        assert_eq!(
            supervisor
                .get_worker_mut(model_id)
                .expect("worker ref")
                .get_block_meta(0)
                .unwrap()
                .committed_tokens(),
            expected_tokens
        );

        // Crash current worker
        supervisor
            .get_worker_mut(model_id)
            .expect("worker ref")
            .kill(libc::SIGKILL)
            .expect("kill worker");
        assert!(supervisor.is_worker_crashed(model_id).unwrap());

        // Verify resumption state inspects surviving arena correctly
        let snap = supervisor
            .inspect_resumption_state(model_id, &[0], 4)
            .expect("inspect resumption");
        assert_eq!(snap.total_committed_tokens, expected_tokens);
        assert!(snap.skip_prefill);
        assert_eq!(snap.next_decode_pos, expected_tokens);

        // Recover worker with new socketpair and worker thread
        let (mgr_sock_c, worker_sock_c) = UnixStream::pair().expect("socketpair cycle");
        let (arena_len_c, arena_dup_c) = {
            let worker_ref = supervisor.get_worker_mut(model_id).unwrap();
            (
                worker_ref.arena().len(),
                worker_ref.arena().duplicate_fd().expect("dup fd cycle"),
            )
        };
        let worker_arena_c = SharedMemoryArena::from_existing_fd(
            arena_dup_c,
            arena_len_c,
            &format!("worker_shm_6_{cycle}"),
            false,
        )
        .expect("worker arena cycle");

        let mut c_tok = 800 + cycle * 10;
        let _th_c = spawn_mock_worker_thread(worker_sock_c, worker_arena_c, move |_cmd| {
            let t = c_tok;
            c_tok += 1;
            t
        });

        let rec = supervisor
            .recover_test_worker(model_id, mgr_sock_c)
            .expect("recover");
        assert_eq!(rec.generation(), u64::from(cycle) + 1);
        assert!(rec.is_alive());
    }

    // Final resume decode on generation 4
    let final_worker = supervisor.get_worker_mut(model_id).expect("final worker");
    assert_eq!(final_worker.generation(), 4);

    let resume_res = final_worker
        .resume_decode(8999, 1, 4, &[0], current_tok)
        .expect("final resume");
    assert_eq!(resume_res.prefill_latency_ms(), 0.0);
    assert_eq!(
        final_worker.get_block_meta(0).unwrap().committed_tokens(),
        8 // 4 prefill + 3 prior decodes + 1 final decode
    );
}
