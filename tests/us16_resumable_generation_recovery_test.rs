//! US-16 Test Suite: Resumable Generation Recovery After Worker Crashes
//!
//! Verifies:
//! 1. `k3-kv` shared memory arena survives worker crashes (`SIGKILL`), retaining block metadata & PRNG states.
//! 2. Zero-prefill decode loop resumption (`skip_prefill = true`, next decode starts from `committed_token_count`).
//! 3. Resumption state inspection with prompt length comparison and PRNG recovery.
//! 4. Client reconnection and missed token catchup stream within 30-second retention window.
//! 5. Expired retention window (>30s) rejection and new session transition.
//! 6. Multi-subscriber broadcast over reconnected decode streams.

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::bool_assert_comparison
)]

use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use zerocopy::FromBytes;

use k3_core::ipc::layout::{
    ARENA_MAGIC, ARENA_VERSION, ArenaHeader, KvBlockTag, OutputSlabHeader, SLAB_STATUS_WRITTEN,
};
use k3_core::ipc::protocol::{
    ForwardPassCompleted, IpcMessage, StatusCode, recv_msg_frame, send_msg_frame,
};
use k3_core::ipc::shm::SharedMemoryArena;
use k3_manager::session::{SessionError, SessionRegistry};
use k3_manager::supervisor::{ManagedWorker, WorkerPlacement, WorkerSupervisor};

fn init_resumable_arena(name: &str) -> SharedMemoryArena {
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

#[test]
fn test_us16_worker_crash_arena_survival_and_detection() {
    let mut arena = init_resumable_arena("shm_crash_survival");

    // Populate block 0: 16 committed tokens, tag Volatile
    let b0 = arena.block_meta_mut(0).expect("block 0");
    b0.committed_token_count = 16;
    b0.tag = KvBlockTag::Volatile.as_u8();
    b0.sampler_rng_state = [111, 222, 333, 444];

    // Populate block 1: 10 committed tokens, tag Volatile
    let b1 = arena.block_meta_mut(1).expect("block 1");
    b1.committed_token_count = 10;
    b1.tag = KvBlockTag::Volatile.as_u8();
    b1.sampler_rng_state = [555, 666, 777, 888];

    let (mgr_sock, _worker_sock) = UnixStream::pair().expect("socketpair");
    let placement = WorkerPlacement::new(
        "resumable-model",
        "/tmp/model.bin",
        "/tmp/model.sock",
        "x100",
    );
    let worker = ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 65536, 64, 4);

    let mut supervisor = WorkerSupervisor::new();
    supervisor.register_worker(worker);

    // 1. Worker is initially alive
    assert!(!supervisor.is_worker_crashed("resumable-model").unwrap());

    // 2. Simulate worker crash via SIGKILL
    let worker_ref = supervisor
        .get_worker_mut("resumable-model")
        .expect("worker ref");
    worker_ref.kill(libc::SIGKILL).expect("kill worker");

    // 3. Supervisor detects worker crash
    assert!(supervisor.is_worker_crashed("resumable-model").unwrap());

    // 4. Verify shared memory arena survives in supervisor with identical metadata
    let surviving_worker = supervisor
        .get_worker_mut("resumable-model")
        .expect("worker ref");
    let meta0 = surviving_worker.get_block_meta(0).expect("read block 0");
    assert_eq!(meta0.committed_tokens(), 16);
    assert_eq!(meta0.tag_enum(), Some(KvBlockTag::Volatile));
    assert_eq!(meta0.sampler_rng_state, [111, 222, 333, 444]);

    let meta1 = surviving_worker.get_block_meta(1).expect("read block 1");
    assert_eq!(meta1.committed_tokens(), 10);
    assert_eq!(meta1.sampler_rng_state, [555, 666, 777, 888]);
}

#[test]
fn test_us16_zero_prefill_resumption_state_inspection() {
    let mut arena = init_resumable_arena("shm_resumption_state");

    // Blocks hold a total of 26 tokens (16 + 10)
    let b0 = arena.block_meta_mut(0).expect("block 0");
    b0.committed_token_count = 16;
    b0.sampler_rng_state = [100, 200, 300, 400];

    let b1 = arena.block_meta_mut(1).expect("block 1");
    b1.committed_token_count = 10;
    b1.sampler_rng_state = [500, 600, 700, 800];

    let (mgr_sock, _worker_sock) = UnixStream::pair().expect("socketpair");
    let placement =
        WorkerPlacement::new("inspect-model", "/tmp/model.bin", "/tmp/model.sock", "x100");
    let worker = ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 65536, 64, 4);

    let mut supervisor = WorkerSupervisor::new();
    supervisor.register_worker(worker);

    // Case 1: prompt_len = 16 <= total_committed (26) -> prefill was fully done
    let snap_full = supervisor
        .inspect_resumption_state("inspect-model", &[0, 1], 16)
        .expect("inspect resumption");
    assert_eq!(snap_full.total_committed_tokens, 26);
    assert!(
        snap_full.skip_prefill,
        "Must skip prefill when prompt is fully committed"
    );
    assert_eq!(snap_full.next_decode_pos, 26);
    assert_eq!(snap_full.sampler_rng_state, [500, 600, 700, 800]);

    // Case 2: prompt_len = 32 > total_committed (26) -> prefill incomplete
    let snap_partial = supervisor
        .inspect_resumption_state("inspect-model", &[0, 1], 32)
        .expect("inspect resumption");
    assert_eq!(snap_partial.total_committed_tokens, 26);
    assert!(
        !snap_partial.skip_prefill,
        "Cannot skip prefill when prompt is partially committed"
    );
    assert_eq!(snap_partial.next_decode_pos, 26);
}

#[test]
fn test_us16_zero_prefill_decode_loop_resumption_dispatch() {
    let mut arena = init_resumable_arena("shm_resumption_dispatch");

    // Setup blocks 0 and 1 with 26 committed tokens
    let b0 = arena.block_meta_mut(0).expect("block 0");
    b0.committed_token_count = 16;
    b0.sampler_rng_state = [11, 22, 33, 44];

    let b1 = arena.block_meta_mut(1).expect("block 1");
    b1.committed_token_count = 10;
    b1.sampler_rng_state = [55, 66, 77, 88];

    let dup_fd = arena.duplicate_fd().expect("duplicate_fd");
    let arena_len = arena.len();
    let mut worker_arena =
        SharedMemoryArena::from_existing_fd(dup_fd, arena_len, "worker_arena", false)
            .expect("worker_arena");

    let (mgr_sock, worker_sock) = UnixStream::pair().expect("socketpair");
    let placement = WorkerPlacement::new(
        "dispatch-model",
        "/tmp/model.bin",
        "/tmp/model.sock",
        "x100",
    );
    let mut worker = ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 65536, 64, 4);

    // Mock worker thread simulates restarted worker receiving resume decode request
    let worker_thread = std::thread::spawn(move || {
        let (msg, _) = recv_msg_frame(&worker_sock).expect("worker recv frame");
        if let IpcMessage::ForwardPassCommand(cmd) = msg {
            // Verify: resumption dispatched as a decode pass with token_offset == 26 (zero prefill!)
            assert!(!cmd.is_prefill, "Must be single-token decode pass");
            assert_eq!(
                cmd.token_offset, 26,
                "Must start decode at committed position 26"
            );
            assert_eq!(cmd.token_count, 1, "Must evaluate single token");
            assert_eq!(cmd.block_indices, vec![0, 1]);

            // Mock worker writes logits output slab status
            let hdr = *worker_arena.header().expect("header");
            let slab_offset = usize::try_from(hdr.output_slabs_offset).unwrap();
            let slab_slice = worker_arena.slice_mut(slab_offset, 64).expect("slab slice");
            let slab_hdr = OutputSlabHeader::mut_from_bytes(slab_slice).expect("slab hdr");
            slab_hdr.status = SLAB_STATUS_WRITTEN;
            slab_hdr.vocab_size = 10;

            // Respond with success and next token 43
            let resp = ForwardPassCompleted {
                request_id: cmd.request_id,
                sequence_id: cmd.sequence_id,
                committed_tokens: 1,
                prefill_latency_ms: 0.0,
                decode_latency_ms: 2.5,
                next_token: 43,
                status: StatusCode::Success,
                error_message: None,
            };
            send_msg_frame(&worker_sock, &IpcMessage::ForwardPassCompleted(resp), None)
                .expect("worker send response");
        } else {
            panic!("unexpected IPC message: {msg:?}");
        }
    });

    // Call resume_decode with prompt_len = 16
    let res = worker
        .resume_decode(1001, 1, 16, &[0, 1], 42)
        .expect("resume decode");

    assert_eq!(res.next_token(), 43);
    assert_eq!(res.prefill_latency_ms(), 0.0);
    assert!(res.decode_latency_ms() > 0.0);

    worker_thread.join().expect("join worker thread");
}

#[test]
fn test_us16_client_reconnect_within_retention_window() {
    let registry = SessionRegistry::new();
    let session_id = "session_live";

    // Register active session
    registry
        .register_session(session_id, "project-alpha", "qwen-2.5-7b")
        .expect("register session");

    // Server generates initial tokens: 100, 101, 102, 103, 104, 105
    for token in [100_u32, 101, 102, 103, 104, 105] {
        registry
            .append_token(session_id, token)
            .expect("append token");
    }

    let client_last_acked_token_idx = 2; // Client received [100, 101, 102]

    // Client disconnects at T0
    let disconnect_time = Instant::now();
    registry
        .record_disconnect(session_id)
        .expect("record disconnect");

    // Client reconnects after 5 seconds (simulated within 30-second window)
    let reconnect_time = disconnect_time + Duration::from_secs(5);
    let catchup_tokens = registry
        .reconnect_client_at(
            session_id,
            Some(client_last_acked_token_idx),
            reconnect_time,
        )
        .expect("reconnect within retention window");

    // Catchup stream provides unacknowledged tokens
    assert_eq!(catchup_tokens, vec![103, 104, 105]);

    // Subsequent generation continues seamlessly
    registry
        .append_token(session_id, 106)
        .expect("append token 106");
    let all_tokens = registry.get_tokens(session_id).expect("get tokens");
    assert_eq!(all_tokens.last().copied(), Some(106));
    assert_eq!(all_tokens.len(), 7);
}

#[test]
fn test_us16_client_reconnect_after_retention_window_expired() {
    let registry = SessionRegistry::new();
    let session_id = "session_old";

    // Register session with default 30-second retention window
    registry
        .register_session(session_id, "project-alpha", "qwen-2.5-7b")
        .expect("register session");

    for token in [100_u32, 101, 102] {
        registry
            .append_token(session_id, token)
            .expect("append token");
    }

    let disconnect_time = Instant::now();
    registry
        .record_disconnect(session_id)
        .expect("record disconnect");

    // Client reconnects after 35 seconds (> 30s retention window)
    let expired_time = disconnect_time + Duration::from_secs(35);
    let res = registry.reconnect_client_at(session_id, Some(2), expired_time);

    assert!(
        matches!(res, Err(SessionError::RetentionWindowExpired { .. })),
        "Resumption must be rejected when retention window expired, got: {res:?}"
    );

    // Expired session is pruned and removed from memory
    let pruned = registry.prune_expired_at(expired_time);
    assert_eq!(pruned, 0); // Already auto-pruned upon expired reconnect attempt
    assert!(!registry.contains_session(session_id));
    assert_eq!(registry.session_count(), 0);
}

#[test]
fn test_us16_multiple_client_subscribers_on_reconnected_stream() {
    let registry = SessionRegistry::new();
    let session_id = "stream-session";

    registry
        .register_session(session_id, "project-broadcast", "qwen-2.5-7b")
        .expect("register session");

    // Two client subscribers subscribe to the reconnected decode stream
    let mut sub1 = registry.subscribe(session_id).expect("subscribe sub1");
    let mut sub2 = registry.subscribe(session_id).expect("subscribe sub2");

    // Newly generated tokens after resumption
    let newly_generated = [200_u32, 201_u32, 202_u32];

    for &tok in &newly_generated {
        registry
            .append_token(session_id, tok)
            .expect("append token");
    }

    // Verify all subscribers received identical token streams in lockstep
    assert_eq!(sub1.try_recv(), Ok(200));
    assert_eq!(sub1.try_recv(), Ok(201));
    assert_eq!(sub1.try_recv(), Ok(202));

    assert_eq!(sub2.try_recv(), Ok(200));
    assert_eq!(sub2.try_recv(), Ok(201));
    assert_eq!(sub2.try_recv(), Ok(202));
}
