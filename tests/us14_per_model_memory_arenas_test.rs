//! US-14: Strict Per-Model Memory Arenas & Context Window Admission Scenario Test Suite.
//!
//! Validates:
//! 1. Load-time memory admission under 28 GiB `LimitMEMLOCK` ceiling (status code 11 / InsufficientLockableMemory).
//! 2. Upfront refusal preserves already-running models without disruption.
//! 3. Dedicated per-model memory arenas with intra-arena LRU block eviction and zero cross-arena leakage.
//! 4. Prompt context window admission: prompts with `tokens.len() >= n_ctx` rejected with status 12 without silent truncation.
//! 5. Multi-slot footprint scaling (`active_slots: usize`).

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::useless_vec
)]

use k3_core::error::K3Error;
use k3_core::ipc::layout::{ArenaHeader, KvBlockMeta, TAG_FREE, TAG_PERSIST, TAG_VOLATILE};
use k3_core::ipc::protocol::StatusCode;
use k3_core::ipc::shm::{DEFAULT_LIMIT_MEMLOCK_BYTES, SharedMemoryArena};
use k3_kv::holder::{ArenaAllocationConfig, KvError, KvHolder, compute_arena_layout};
use k3_manager::prompt::validate_and_tokenize;
use k3_tokenizer::SimpleTokenizer;

#[test]
fn test_us14_load_time_memory_admission_rejection_status_11() {
    let n_ctx = 64_u32;
    let config = ArenaAllocationConfig::new("model-admission-test", n_ctx, 64, 2);
    let layout = compute_arena_layout(&config).expect("compute arena layout");
    assert!(layout.total_bytes > 0);

    // Holder with ceiling lower than required arena layout
    let ceiling = layout.total_bytes - 1;
    let holder = KvHolder::with_memlock_ceiling(ceiling);

    let err = holder
        .allocate(&config)
        .expect_err("must refuse admission when exceeding LimitMEMLOCK");

    assert!(matches!(err, KvError::InsufficientLockableMemory { .. }));
    assert_eq!(err.status_code(), StatusCode::InsufficientLockableMemory);
    assert_eq!(err.exit_code(), 11);
    assert_eq!(StatusCode::InsufficientLockableMemory as u32, 11);
}

#[test]
fn test_us14_upfront_refusal_preserves_active_running_models() {
    // Reference 28 GiB operator ceiling
    assert_eq!(DEFAULT_LIMIT_MEMLOCK_BYTES, 28 * 1024 * 1024 * 1024);

    let n_ctx_a = 64_u32;
    let config_a = ArenaAllocationConfig::new("model-a", n_ctx_a, 64, 2);
    let layout_a = compute_arena_layout(&config_a).expect("layout a");

    // Sized holder allowing Model A but not both A and a huge Model B
    let holder = KvHolder::with_memlock_ceiling(layout_a.total_bytes + 1024);

    // Model A is allocated successfully
    let alloc_a = holder.allocate(&config_a).expect("allocate model a");
    assert!(alloc_a.arena_id > 0);
    assert_eq!(holder.total_allocated_bytes(), layout_a.total_bytes);

    let arena_a = holder.get_arena(alloc_a.arena_id).expect("get arena a");
    let hdr_a = arena_a.header().expect("read header a");
    assert_eq!(hdr_a.n_ctx, n_ctx_a);
    assert_eq!(hdr_a.validate(), Ok(()));

    // Model B requests large allocation that exceeds remaining lockable limit
    let config_b = ArenaAllocationConfig::new("model-b", 2048, 128, 8);
    let layout_b = compute_arena_layout(&config_b).expect("layout b");
    assert!(layout_a.total_bytes + layout_b.total_bytes > holder.memlock_ceiling_bytes());

    let err_b = holder
        .allocate(&config_b)
        .expect_err("must refuse model b upfront");
    assert!(matches!(err_b, KvError::InsufficientLockableMemory { .. }));
    assert_eq!(err_b.status_code(), StatusCode::InsufficientLockableMemory);

    // Assert Model A remains completely operational and untouched
    assert_eq!(holder.total_allocated_bytes(), layout_a.total_bytes);
    let surviving_hdr_a = arena_a.header().expect("read header a again");
    assert_eq!(surviving_hdr_a.n_ctx, n_ctx_a);
    assert_eq!(surviving_hdr_a.validate(), Ok(()));
}

#[test]
fn test_us14_dedicated_per_model_arenas_and_intra_arena_lru_eviction() {
    let mut arena_a =
        SharedMemoryArena::create("model-a-shm", 1024 * 1024).expect("create arena a");
    let mut arena_b =
        SharedMemoryArena::create("model-b-shm", 2048 * 1024).expect("create arena b");

    let header_a = ArenaHeader {
        magic: k3_core::ipc::layout::ARENA_MAGIC,
        version: 1,
        total_bytes: 1024 * 1024,
        block_count: 4,
        n_ctx: 64,
        ..Default::default()
    };

    let header_b = ArenaHeader {
        magic: k3_core::ipc::layout::ARENA_MAGIC,
        version: 1,
        total_bytes: 2048 * 1024,
        block_count: 2,
        n_ctx: 128,
        ..Default::default()
    };

    assert_eq!(header_a.validate(), Ok(()));
    assert_eq!(header_b.validate(), Ok(()));
    assert_ne!(header_a.total_bytes, header_b.total_bytes);

    *arena_a.header_mut().expect("header_mut a") = header_a;
    *arena_b.header_mut().expect("header_mut b") = header_b;

    // Model A arena has 4 blocks
    *arena_a.block_meta_mut(0).expect("block 0") = KvBlockMeta {
        block_id: 0,
        tag: TAG_VOLATILE,
        refcount: 0, // Unreferenced -> eligible for eviction
        committed_token_count: 16,
        ..Default::default()
    };
    *arena_a.block_meta_mut(1).expect("block 1") = KvBlockMeta {
        block_id: 1,
        tag: TAG_VOLATILE,
        refcount: 2, // Referenced by in-flight sessions -> protected
        committed_token_count: 16,
        ..Default::default()
    };
    *arena_a.block_meta_mut(2).expect("block 2") = KvBlockMeta {
        block_id: 2,
        tag: TAG_PERSIST, // Checkpointed -> protected
        refcount: 0,
        committed_token_count: 16,
        ..Default::default()
    };
    *arena_a.block_meta_mut(3).expect("block 3") = KvBlockMeta {
        block_id: 3,
        tag: TAG_VOLATILE,
        refcount: 0, // Unreferenced -> eligible for eviction
        committed_token_count: 16,
        ..Default::default()
    };

    // Model B arena has completely separate blocks
    *arena_b.block_meta_mut(0).expect("block 0 in b") = KvBlockMeta {
        block_id: 100,
        tag: TAG_VOLATILE,
        refcount: 1,
        committed_token_count: 16,
        ..Default::default()
    };
    *arena_b.block_meta_mut(1).expect("block 1 in b") = KvBlockMeta {
        block_id: 101,
        tag: TAG_VOLATILE,
        refcount: 0,
        committed_token_count: 16,
        ..Default::default()
    };

    // Model A needs a block: scans strictly within arena_a using production find_evictable_blocks()
    let evictable_in_a = arena_a
        .find_evictable_blocks()
        .expect("find evictable in a");
    assert_eq!(evictable_in_a, vec![0, 3]);

    // Evict 1 block in arena_a using genuine production evict_lru_intra_arena()
    let freed = arena_a
        .evict_lru_intra_arena(1)
        .expect("evict 1 block in a");
    assert_eq!(freed, 1);

    let b0_a = arena_a.block_meta(0).expect("read block 0 in a");
    assert_eq!(b0_a.tag, TAG_FREE);
    assert_eq!(b0_a.committed_tokens(), 0);
    assert!(!b0_a.is_evictable());

    // Protected blocks in arena_a are intact
    let b1_a = arena_a.block_meta(1).expect("read block 1 in a");
    assert_eq!(b1_a.tag, TAG_VOLATILE);
    assert_eq!(b1_a.refcount, 2);
    assert!(!b1_a.is_evictable());

    let b2_a = arena_a.block_meta(2).expect("read block 2 in a");
    assert_eq!(b2_a.tag, TAG_PERSIST);
    assert!(!b2_a.is_evictable());

    // Model B arena is completely untouched (zero cross-arena leakage)
    let b0_b = arena_b.block_meta(0).expect("read block 0 in b");
    assert_eq!(b0_b.block_id, 100);
    assert_eq!(b0_b.refcount, 1);

    let b1_b = arena_b.block_meta(1).expect("read block 1 in b");
    assert_eq!(b1_b.block_id, 101);
    assert_eq!(b1_b.tag, TAG_VOLATILE);

    // Evict second eligible block in arena_a
    let freed_second = arena_a
        .evict_lru_intra_arena(1)
        .expect("evict second block in a");
    assert_eq!(freed_second, 1);
    let b3_a = arena_a.block_meta(3).expect("read block 3 in a");
    assert_eq!(b3_a.tag, TAG_FREE);

    // No more evictable blocks in arena_a (only protected blocks remain)
    let freed_none = arena_a.evict_lru_intra_arena(1).expect("no more evictable");
    assert_eq!(freed_none, 0);
}

#[test]
fn test_us14_prompt_admission_context_length_exceeded_status_12() {
    let n_ctx = 64_usize;
    let mut tokenizer = SimpleTokenizer::new();
    tokenizer.add_token("tok", 1);

    // 1. Prompt with 63 tokens: admitted (< n_ctx, allows at least 1 slot for decode)
    let prompt_63 = "tok ".repeat(63);
    let res_admitted = validate_and_tokenize(&tokenizer, &prompt_63, None, false, n_ctx);
    assert!(res_admitted.is_ok());
    assert_eq!(res_admitted.unwrap().len(), 63);

    // 2. Prompt with exactly 64 tokens (== n_ctx): rejected upfront (0 decode slots remain)
    let prompt_64 = "tok ".repeat(64);
    let res_64 = validate_and_tokenize(&tokenizer, &prompt_64, None, false, n_ctx);
    assert_eq!(
        res_64,
        Err(K3Error::ContextLengthExceeded {
            requested_tokens: 64,
            max_context_tokens: 64,
        })
    );

    // 3. Prompt with 128 tokens (> n_ctx): rejected upfront without silent truncation
    let prompt_128 = "tok ".repeat(128);
    let res_128 = validate_and_tokenize(&tokenizer, &prompt_128, None, false, n_ctx);
    assert_eq!(
        res_128,
        Err(K3Error::ContextLengthExceeded {
            requested_tokens: 128,
            max_context_tokens: 64,
        })
    );

    // 4. Test validate_and_tokenize control-plane prompt admission with multi-token vocabulary
    let mut tokenizer = SimpleTokenizer::new();
    tokenizer.add_token("test", 1);
    tokenizer.add_token("token", 2);
    tokenizer.add_token("sequence", 3);
    tokenizer.add_token("repetition", 4);
    let prompt_text = "test token sequence repetition ".repeat(20);
    let tokenized_res = validate_and_tokenize(&tokenizer, &prompt_text, None, false, 16);
    assert!(matches!(
        tokenized_res,
        Err(K3Error::ContextLengthExceeded { .. })
    ));

    assert_eq!(StatusCode::ContextLengthExceeded as u32, 12);
}

#[test]
fn test_us14_active_slots_scaling_memory_footprint() {
    let n_ctx = 64_u32; // 4 blocks of 16 tokens per slot

    // 1 slot: 4 blocks
    let config_1 = ArenaAllocationConfig::new("slot-test-1", n_ctx, 64, 2).with_active_slots(1);
    let layout_1 = compute_arena_layout(&config_1).expect("layout 1 slot");
    assert_eq!(layout_1.block_count, 4);

    // 2 slots: 8 blocks
    let config_2 = ArenaAllocationConfig::new("slot-test-2", n_ctx, 64, 2).with_active_slots(2);
    let layout_2 = compute_arena_layout(&config_2).expect("layout 2 slots");
    assert_eq!(layout_2.block_count, 8);
    assert!(layout_2.total_bytes > layout_1.total_bytes);

    // 4 slots: 16 blocks
    let config_4 = ArenaAllocationConfig::new("slot-test-4", n_ctx, 64, 2).with_active_slots(4);
    let layout_4 = compute_arena_layout(&config_4).expect("layout 4 slots");
    assert_eq!(layout_4.block_count, 16);
    assert!(layout_4.total_bytes > layout_2.total_bytes);

    // Admission check in KvHolder with constrained ceiling
    let ceiling = layout_1.total_bytes + 10;
    let holder = KvHolder::with_memlock_ceiling(ceiling);

    // 1 slot fits within ceiling
    assert!(holder.allocate(&config_1).is_ok());

    // 2 slots exceed ceiling and is rejected with InsufficientLockableMemory
    let err_2 = holder
        .allocate(&config_2)
        .expect_err("2 slots must exceed ceiling");
    assert!(matches!(err_2, KvError::InsufficientLockableMemory { .. }));
    assert_eq!(err_2.status_code(), StatusCode::InsufficientLockableMemory);
    assert_eq!(err_2.exit_code(), 11);
}
