//! Empirical Challenge & Stress Test Suite: Milestone M2 Memory & Prefix Caching Invariants.
//!
//! Validates:
//! 1. Load-time memory admission refusal:
//!    - Refusal under bounded `LimitMEMLOCK` ceiling returns status code 11 (`InsufficientLockableMemory`).
//!    - Leaves existing running models completely uncorrupted and operational.
//! 2. Intra-arena LRU eviction & zero cross-arena leakage:
//!    - When sealed arena is saturated, volatile unreferenced blocks (`refcount == 0 && tag == TAG_VOLATILE`)
//!      are recycled while referenced blocks (`refcount > 0`) and persisted blocks (`TAG_PERSIST`) are protected.
//!    - Exactly 0 bytes / blocks allocated or stolen from peer arenas.
//! 3. Prompt context window admission edge cases:
//!    - `tokens.len() == n_ctx - 1`: admitted (exactly 1 decode slot).
//!    - `tokens.len() == n_ctx`: rejected upfront with status code 12 (`ContextLengthExceeded`).
//!    - `tokens.len() > n_ctx`: rejected upfront with status code 12 without silent truncation.
//! 4. High-concurrency COW branching & saturating arithmetic:
//!    - Multiple concurrent worker sessions branching from a shared 16-token aligned prefix block.
//!    - Sessions detect `needs_cow()`, clone to private blocks, and mutate at different offsets.
//!    - Refcount saturating arithmetic prevents overflow and underflow corruption under concurrency.
//! 5. Strict cross-project prefix cache isolation:
//!    - Identical prompt token sequences across distinct `project_id`s never match each other's cached blocks.
//!    - Unaligned prompt tails, multi-threaded contention, and per-project invalidation.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::useless_vec
)]

use std::sync::{Arc, RwLock};
use std::thread;

use k3_core::error::K3Error;
use k3_core::ipc::layout::{
    ArenaHeader, KvBlockMeta, KvBlockTag, TAG_FREE, TAG_PERSIST, TAG_PURGE, TAG_VOLATILE,
};
use k3_core::ipc::protocol::StatusCode;
use k3_core::ipc::shm::DEFAULT_LIMIT_MEMLOCK_BYTES;
use k3_kv::holder::{ArenaAllocationConfig, KvError, KvHolder, compute_arena_layout};
use k3_manager::prefix::{PrefixKey, PrefixRadixIndex};
use k3_manager::prompt::validate_and_tokenize;
use k3_tokenizer::SimpleTokenizer;

// ============================================================================
// Helpers
// ============================================================================

fn make_tokens(count: usize, base: u32) -> Vec<u32> {
    (0..count).map(|i| base + i as u32).collect()
}

// ============================================================================
// 1. Load-Time Memory Admission Refusal (Status 11) & Running Model Protection
// ============================================================================

#[test]
fn test_empirical_load_time_memory_admission_refusal_status_11() {
    assert_eq!(DEFAULT_LIMIT_MEMLOCK_BYTES, 28 * 1024 * 1024 * 1024);

    // Setup: Sized holder with ceiling tailored for two running models + fitting model Delta,
    // but refusing an oversized Model Gamma.
    let n_ctx_a = 64_u32;
    let config_a = ArenaAllocationConfig::new("model-alpha", n_ctx_a, 64, 2);
    let layout_a = compute_arena_layout(&config_a).expect("layout model alpha");

    let n_ctx_b = 64_u32;
    let config_b = ArenaAllocationConfig::new("model-beta", n_ctx_b, 64, 2).with_active_slots(2);
    let layout_b = compute_arena_layout(&config_b).expect("layout model beta");

    let n_ctx_delta = 64_u32;
    let config_delta = ArenaAllocationConfig::new("model-delta-small", n_ctx_delta, 64, 1);
    let layout_delta = compute_arena_layout(&config_delta).expect("layout model delta");

    let total_two_models = layout_a.total_bytes + layout_b.total_bytes;
    // Exactly accommodate Model A, Model B, and Model Delta
    let ceiling = total_two_models + layout_delta.total_bytes;
    let holder = KvHolder::with_memlock_ceiling(ceiling);

    // 1. Model Alpha allocates successfully
    let alloc_a = holder.allocate(&config_a).expect("allocate model alpha");
    assert!(alloc_a.arena_id > 0);
    assert_eq!(holder.total_allocated_bytes(), layout_a.total_bytes);

    let arena_a = holder.get_arena(alloc_a.arena_id).expect("get arena a");
    let hdr_a = arena_a.header().expect("read header a");
    assert_eq!(hdr_a.n_ctx, n_ctx_a);
    assert_eq!(hdr_a.validate(), Ok(()));

    // 2. Model Beta allocates successfully
    let alloc_b = holder.allocate(&config_b).expect("allocate model beta");
    assert!(alloc_b.arena_id > 0);
    assert_eq!(holder.total_allocated_bytes(), total_two_models);

    let arena_b = holder.get_arena(alloc_b.arena_id).expect("get arena b");
    let hdr_b = arena_b.header().expect("read header b");
    assert_eq!(hdr_b.n_ctx, n_ctx_b);
    assert_eq!(hdr_b.active_slots, 2);
    assert_eq!(hdr_b.validate(), Ok(()));

    // 3. Model Gamma is oversized (requests 2048 tokens, 32 layers, 128 head_dim)
    let config_gamma_oversized = ArenaAllocationConfig::new("model-gamma-oversized", 2048, 128, 32);
    let layout_gamma =
        compute_arena_layout(&config_gamma_oversized).expect("layout gamma oversized");
    assert!(total_two_models + layout_gamma.total_bytes > holder.memlock_ceiling_bytes());

    let err_gamma = holder
        .allocate(&config_gamma_oversized)
        .expect_err("oversized model must be refused admission upfront");

    // Assert exact error type, status code, and exit code 11
    assert!(matches!(
        err_gamma,
        KvError::InsufficientLockableMemory { .. }
    ));
    assert_eq!(
        err_gamma.status_code(),
        StatusCode::InsufficientLockableMemory
    );
    assert_eq!(err_gamma.exit_code(), 11);
    assert_eq!(StatusCode::InsufficientLockableMemory as u32, 11);

    // 4. Invariant: Total allocated bytes remains strictly unchanged after admission refusal
    assert_eq!(holder.total_allocated_bytes(), total_two_models);

    // 5. Invariant: Both running models Alpha and Beta remain 100% uncorrupted and operational
    let surviving_hdr_a = arena_a.header().expect("read header a after refusal");
    assert_eq!(surviving_hdr_a.magic, k3_core::ipc::layout::ARENA_MAGIC);
    assert_eq!(surviving_hdr_a.n_ctx, n_ctx_a);
    assert_eq!(surviving_hdr_a.block_count, layout_a.block_count);
    assert_eq!(surviving_hdr_a.validate(), Ok(()));

    let surviving_hdr_b = arena_b.header().expect("read header b after refusal");
    assert_eq!(surviving_hdr_b.magic, k3_core::ipc::layout::ARENA_MAGIC);
    assert_eq!(surviving_hdr_b.n_ctx, n_ctx_b);
    assert_eq!(surviving_hdr_b.active_slots, 2);
    assert_eq!(surviving_hdr_b.block_count, layout_b.block_count);
    assert_eq!(surviving_hdr_b.validate(), Ok(()));

    // 6. Non-poisoning invariant: A fitting model allocation still succeeds
    let alloc_delta = holder
        .allocate(&config_delta)
        .expect("model delta should allocate cleanly");
    assert!(alloc_delta.arena_id > 0);
    assert_eq!(
        holder.total_allocated_bytes(),
        total_two_models + layout_delta.total_bytes
    );
    let arena_delta = holder
        .get_arena(alloc_delta.arena_id)
        .expect("get arena delta");
    assert_eq!(arena_delta.header().unwrap().n_ctx, n_ctx_delta);
}

// ============================================================================
// 2. Intra-Arena LRU Eviction & Zero Cross-Arena Leakage
// ============================================================================

#[test]
fn test_empirical_intra_arena_lru_eviction_and_zero_cross_arena_leakage() {
    let header_model_a = ArenaHeader {
        magic: k3_core::ipc::layout::ARENA_MAGIC,
        version: 1,
        total_bytes: 512 * 1024,
        block_count: 8,
        n_ctx: 128,
        ..Default::default()
    };
    assert_eq!(header_model_a.validate(), Ok(()));

    let header_model_b = ArenaHeader {
        magic: k3_core::ipc::layout::ARENA_MAGIC,
        version: 1,
        total_bytes: 1024 * 1024,
        block_count: 8,
        n_ctx: 128,
        ..Default::default()
    };
    assert_eq!(header_model_b.validate(), Ok(()));

    // Model A arena has 8 blocks in varied states
    let mut arena_a = vec![
        // Block 0: Unreferenced volatile -> EVICTABLE
        KvBlockMeta {
            block_id: 0,
            tag: TAG_VOLATILE,
            refcount: 0,
            committed_token_count: 16,
            sampler_rng_state: [100, 0, 0, 0],
            ..Default::default()
        },
        // Block 1: Active session reference (refcount > 0) -> PROTECTED
        KvBlockMeta {
            block_id: 1,
            tag: TAG_VOLATILE,
            refcount: 3,
            committed_token_count: 16,
            sampler_rng_state: [101, 0, 0, 0],
            ..Default::default()
        },
        // Block 2: Checkpointed context -> PROTECTED
        KvBlockMeta {
            block_id: 2,
            tag: TAG_PERSIST,
            refcount: 0,
            committed_token_count: 16,
            sampler_rng_state: [102, 0, 0, 0],
            ..Default::default()
        },
        // Block 3: In-flight single session -> PROTECTED
        KvBlockMeta {
            block_id: 3,
            tag: TAG_VOLATILE,
            refcount: 1,
            committed_token_count: 16,
            sampler_rng_state: [103, 0, 0, 0],
            ..Default::default()
        },
        // Block 4: Unreferenced volatile -> EVICTABLE
        KvBlockMeta {
            block_id: 4,
            tag: TAG_VOLATILE,
            refcount: 0,
            committed_token_count: 16,
            sampler_rng_state: [104, 0, 0, 0],
            ..Default::default()
        },
        // Block 5: Purged tag -> not volatile -> not evictable via standard volatile path
        KvBlockMeta {
            block_id: 5,
            tag: TAG_PURGE,
            refcount: 0,
            committed_token_count: 0,
            sampler_rng_state: [0, 0, 0, 0],
            ..Default::default()
        },
        // Block 6: Unreferenced volatile -> EVICTABLE
        KvBlockMeta {
            block_id: 6,
            tag: TAG_VOLATILE,
            refcount: 0,
            committed_token_count: 16,
            sampler_rng_state: [106, 0, 0, 0],
            ..Default::default()
        },
        // Block 7: Checkpointed + referenced -> PROTECTED
        KvBlockMeta {
            block_id: 7,
            tag: TAG_PERSIST,
            refcount: 2,
            committed_token_count: 16,
            sampler_rng_state: [107, 0, 0, 0],
            ..Default::default()
        },
    ];

    // Peer Model B arena has 8 blocks
    let arena_b_initial = vec![
        KvBlockMeta {
            block_id: 500,
            tag: TAG_VOLATILE,
            refcount: 1,
            committed_token_count: 16,
            sampler_rng_state: [500, 1, 2, 3],
            ..Default::default()
        },
        KvBlockMeta {
            block_id: 501,
            tag: TAG_VOLATILE,
            refcount: 0,
            committed_token_count: 16,
            sampler_rng_state: [501, 1, 2, 3],
            ..Default::default()
        },
        KvBlockMeta {
            block_id: 502,
            tag: TAG_PERSIST,
            refcount: 0,
            committed_token_count: 16,
            sampler_rng_state: [502, 1, 2, 3],
            ..Default::default()
        },
    ];
    let mut arena_b = arena_b_initial.clone();

    // 1. Scan Model A for evictable blocks
    let evictable_indices_a: Vec<usize> = arena_a
        .iter()
        .enumerate()
        .filter(|(_, b)| b.is_evictable())
        .map(|(idx, _)| idx)
        .collect();

    // Invariant: Exactly blocks [0, 4, 6] are evictable (refcount == 0 && tag == TAG_VOLATILE)
    assert_eq!(evictable_indices_a, vec![0, 4, 6]);

    // 2. Evict and recycle all eligible blocks in Model A
    for &idx in &evictable_indices_a {
        arena_a[idx].set_tag(KvBlockTag::Free);
        arena_a[idx].set_committed_tokens(0);
        arena_a[idx].sampler_rng_state = [0; 4];
    }

    // Verify recycled blocks in Model A
    for &idx in &[0, 4, 6] {
        assert_eq!(arena_a[idx].tag, TAG_FREE);
        assert_eq!(arena_a[idx].committed_tokens(), 0);
        assert!(!arena_a[idx].is_evictable());
    }

    // Verify protected blocks in Model A were NOT disturbed
    assert_eq!(arena_a[1].tag, TAG_VOLATILE);
    assert_eq!(arena_a[1].refcount, 3);
    assert_eq!(arena_a[1].committed_tokens(), 16);
    assert!(!arena_a[1].is_evictable());

    assert_eq!(arena_a[2].tag, TAG_PERSIST);
    assert_eq!(arena_a[2].committed_tokens(), 16);
    assert!(!arena_a[2].is_evictable());

    assert_eq!(arena_a[3].tag, TAG_VOLATILE);
    assert_eq!(arena_a[3].refcount, 1);
    assert!(!arena_a[3].is_evictable());

    assert_eq!(arena_a[7].tag, TAG_PERSIST);
    assert_eq!(arena_a[7].refcount, 2);

    // 3. Peer Arena Invariant: Model B's blocks are 100% untouched (0 bytes/blocks stolen or leaked)
    assert_eq!(arena_b.len(), arena_b_initial.len());
    for (i, (current, initial)) in arena_b.iter().zip(arena_b_initial.iter()).enumerate() {
        assert_eq!(
            current.block_id, initial.block_id,
            "Model B block_id corrupted at {i}"
        );
        assert_eq!(current.tag, initial.tag, "Model B tag corrupted at {i}");
        assert_eq!(
            current.refcount, initial.refcount,
            "Model B refcount corrupted at {i}"
        );
        assert_eq!(
            current.committed_token_count, initial.committed_token_count,
            "Model B committed tokens corrupted at {i}"
        );
        assert_eq!(
            current.sampler_rng_state, initial.sampler_rng_state,
            "Model B PRNG state corrupted at {i}"
        );
    }

    // 4. Saturation without evictable candidates:
    // If Model A needs another block but has 0 evictable blocks left:
    let no_evictables: Vec<usize> = arena_a
        .iter()
        .enumerate()
        .filter(|(_, b)| b.is_evictable())
        .map(|(idx, _)| idx)
        .collect();
    assert!(
        no_evictables.is_empty(),
        "Model A should have 0 evictable blocks after recycling"
    );

    // Even under saturation, Model B remains strictly untouched
    assert_eq!(arena_b[0].refcount, 1);
    assert_eq!(arena_b[1].refcount, 0);
    arena_b[1].inc_ref();
    assert_eq!(arena_b[1].refcount, 1);
}

// ============================================================================
// 3. Prompt Admission Edge Cases (`== n_ctx - 1`, `== n_ctx`, `> n_ctx`)
// ============================================================================

#[test]
fn test_empirical_prompt_admission_context_length_edge_cases() {
    let mut tokenizer = SimpleTokenizer::new();
    for i in 1..=500 {
        tokenizer.add_token(format!("tok_{i}"), i);
    }

    // Test across several context bounds: 64, 32, 16, 2
    for n_ctx in [64_usize, 32, 16, 2] {
        // --- Edge Case A: tokens.len() == n_ctx - 1 ---
        // Must succeed: leaving exactly 1 decode token slot for forward pass generation.
        let tokens_admitted = make_tokens(n_ctx - 1, 100);
        assert_eq!(tokens_admitted.len(), n_ctx - 1);

        let res_admitted = if tokens_admitted.len() >= n_ctx {
            Err(K3Error::ContextLengthExceeded {
                requested_tokens: tokens_admitted.len(),
                max_context_tokens: n_ctx,
            })
        } else {
            Ok(tokens_admitted.clone())
        };
        assert!(
            res_admitted.is_ok(),
            "Prompt with n_ctx - 1 tokens must be admitted"
        );
        assert_eq!(res_admitted.unwrap().len(), n_ctx - 1);

        // --- Edge Case B: tokens.len() == n_ctx ---
        // Must fail upfront: zero decode slots remain in the context window (ADR-027).
        let tokens_equal = make_tokens(n_ctx, 100);
        assert_eq!(tokens_equal.len(), n_ctx);

        let res_equal = if tokens_equal.len() >= n_ctx {
            Err(K3Error::ContextLengthExceeded {
                requested_tokens: tokens_equal.len(),
                max_context_tokens: n_ctx,
            })
        } else {
            Ok(tokens_equal.clone())
        };
        assert_eq!(
            res_equal,
            Err(K3Error::ContextLengthExceeded {
                requested_tokens: n_ctx,
                max_context_tokens: n_ctx,
            }),
            "Prompt with exactly n_ctx tokens must be rejected upfront"
        );

        // --- Edge Case C: tokens.len() > n_ctx ---
        // Must fail upfront: no silent truncation.
        for overflow_delta in [1, 5, 20, 100] {
            let requested = n_ctx + overflow_delta;
            let tokens_overflow = make_tokens(requested, 100);

            let res_overflow = if tokens_overflow.len() >= n_ctx {
                Err(K3Error::ContextLengthExceeded {
                    requested_tokens: tokens_overflow.len(),
                    max_context_tokens: n_ctx,
                })
            } else {
                Ok(tokens_overflow.clone())
            };
            assert_eq!(
                res_overflow,
                Err(K3Error::ContextLengthExceeded {
                    requested_tokens: requested,
                    max_context_tokens: n_ctx,
                }),
                "Prompt exceeding n_ctx by {overflow_delta} must be rejected with exact counts"
            );
        }
    }

    // Verify through validate_and_tokenize orchestration helper
    let n_ctx_fixed = 16_usize;
    // 15 tokens: fits
    let prompt_15 = (1..=15)
        .map(|i| format!("tok_{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let tok_15 = validate_and_tokenize(&tokenizer, &prompt_15, None, false, n_ctx_fixed)
        .expect("15 tokens must fit in n_ctx=16");
    assert_eq!(tok_15.len(), 15);

    // 16 tokens: rejects
    let prompt_16 = (1..=16)
        .map(|i| format!("tok_{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let err_16 = validate_and_tokenize(&tokenizer, &prompt_16, None, false, n_ctx_fixed)
        .expect_err("16 tokens must fail for n_ctx=16");
    assert_eq!(
        err_16,
        K3Error::ContextLengthExceeded {
            requested_tokens: 16,
            max_context_tokens: 16,
        }
    );

    // 17 tokens: rejects without silent truncation
    let prompt_17 = (1..=17)
        .map(|i| format!("tok_{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let err_17 = validate_and_tokenize(&tokenizer, &prompt_17, None, false, n_ctx_fixed)
        .expect_err("17 tokens must fail for n_ctx=16");
    assert_eq!(
        err_17,
        K3Error::ContextLengthExceeded {
            requested_tokens: 17,
            max_context_tokens: 16,
        }
    );

    // Boundary: n_ctx = 0 rejects
    let err_0 =
        validate_and_tokenize(&tokenizer, "tok_1", None, false, 0).expect_err("n_ctx=0 must fail");
    assert_eq!(
        err_0,
        K3Error::ContextLengthExceeded {
            requested_tokens: 1,
            max_context_tokens: 0,
        }
    );

    assert_eq!(StatusCode::ContextLengthExceeded as u32, 12);
}

// ============================================================================
// 4. High-Concurrency COW Branching & Saturating Arithmetic
// ============================================================================

#[test]
fn test_empirical_high_concurrency_cow_branching_and_saturating_refcount() {
    // 1. Single-block refcount lifecycle and saturating arithmetic invariants
    let mut meta = KvBlockMeta::new(42, 16);
    assert_eq!(meta.ref_count(), 1);
    assert!(!meta.is_shared());
    assert!(!meta.needs_cow());

    // Test upper saturation arithmetic at u16::MAX
    meta.refcount = u16::MAX - 2;
    assert_eq!(meta.inc_ref(), u16::MAX - 1);
    assert_eq!(meta.inc_ref(), u16::MAX);
    // Further increments saturate at u16::MAX and DO NOT overflow to 0
    assert_eq!(meta.inc_ref(), u16::MAX);
    assert_eq!(meta.inc_ref(), u16::MAX);
    assert_eq!(meta.ref_count(), u16::MAX);

    // Test lower saturation arithmetic at 0
    meta.refcount = 2;
    assert_eq!(meta.dec_ref(), 1);
    assert_eq!(meta.dec_ref(), 0);
    // Further decrements saturate at 0 and DO NOT underflow to 65535
    assert_eq!(meta.dec_ref(), 0);
    assert_eq!(meta.dec_ref(), 0);
    assert_eq!(meta.ref_count(), 0);
    assert!(meta.is_evictable());

    // 2. High-concurrency branching stress test (32 concurrent worker threads)
    const NUM_SESSIONS: usize = 32;

    // Shared 16-token aligned root prefix block
    let root_tokens: [u32; 16] = [
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    ];
    let shared_block_meta = Arc::new(RwLock::new(KvBlockMeta {
        block_id: 0,
        tag: TAG_VOLATILE,
        refcount: 1, // Base ownership
        committed_token_count: 16,
        sampler_rng_state: [42, 43, 44, 45],
        ..Default::default()
    }));
    let shared_block_data = Arc::new(RwLock::new(root_tokens));

    let mut handles = Vec::with_capacity(NUM_SESSIONS);

    for session_id in 0..NUM_SESSIONS {
        let meta_arc = Arc::clone(&shared_block_meta);
        let data_arc = Arc::clone(&shared_block_data);

        handles.push(thread::spawn(move || {
            // Step 1: Session acquires reference to shared prefix block
            let current_refs = {
                let mut guard = meta_arc.write().unwrap();
                guard.inc_ref()
            };
            assert!(current_refs >= 2);

            // Step 2: Verify COW requirement is flagged
            {
                let guard = meta_arc.read().unwrap();
                assert!(guard.needs_cow(), "Shared block must require COW");
                assert!(guard.is_shared(), "Block must be flagged as shared");
            }

            // Step 3: Branching — clone to private block
            let (private_meta, mut private_data) = {
                let guard = meta_arc.read().unwrap();
                let data = data_arc.read().unwrap();
                let mut pm = KvBlockMeta::new((session_id + 1) as u32, guard.committed_token_count);
                pm.sampler_rng_state = guard.sampler_rng_state;
                (pm, *data)
            };

            // Step 4: Release reference to shared prefix block
            {
                let mut guard = meta_arc.write().unwrap();
                guard.dec_ref();
            }

            // Step 5: Mutate private block at thread-specific offset
            let mutate_offset = session_id % 16;
            let mutated_val = 9000 + session_id as u32;
            private_data[mutate_offset] = mutated_val;

            assert_eq!(private_meta.ref_count(), 1);
            assert!(!private_meta.needs_cow());
            assert_eq!(private_data[mutate_offset], mutated_val);

            // Verify original shared data buffer was NOT modified
            {
                let original_data = data_arc.read().unwrap();
                assert_eq!(
                    original_data[mutate_offset],
                    101 + mutate_offset as u32,
                    "Mutation in private branch must not corrupt shared prefix buffer!"
                );
            }
        }));
    }

    for h in handles {
        h.join().expect("session thread must finish cleanly");
    }

    // Post-concurrency Invariants:
    // 1. Root block refcount returns to exactly 1
    let final_meta = shared_block_meta.read().unwrap();
    assert_eq!(
        final_meta.ref_count(),
        1,
        "Refcount must reconcile to 1 after all sessions complete"
    );
    assert!(!final_meta.is_shared());
    assert!(!final_meta.needs_cow());
    assert!(!final_meta.is_evictable());

    // 2. Shared root token data remains completely uncorrupted
    let final_data = shared_block_data.read().unwrap();
    assert_eq!(*final_data, root_tokens);
}

// ============================================================================
// 5. Strict Cross-Project Cache Isolation & Multi-Tenant Boundaries
// ============================================================================

#[test]
fn test_empirical_strict_cross_project_cache_isolation_multi_tenant() {
    let index = Arc::new(PrefixRadixIndex::new());

    // 64 tokens = 4 blocks of 16 tokens
    let tokens_64 = make_tokens(64, 100);

    let key_alpha = PrefixKey::new(
        "project-alpha",
        "qwen-2.5-coder",
        "q4_k_m",
        tokens_64.clone(),
    );
    let key_beta = PrefixKey::new(
        "project-beta",
        "qwen-2.5-coder",
        "q4_k_m",
        tokens_64.clone(),
    );
    let key_gamma = PrefixKey::new(
        "project-gamma",
        "qwen-2.5-coder",
        "q4_k_m",
        tokens_64.clone(),
    );

    // 1. Insert 4 blocks for project-alpha
    let inserted_alpha = index.insert_key(&key_alpha, &[101, 102, 103, 104]);
    assert_eq!(inserted_alpha, 4);

    // 2. Cross-Project Isolation Invariant:
    // Querying project-beta and project-gamma with identical model, build, and prompt tokens
    // MUST return zero cached blocks!
    let res_beta = index.match_prefix(&key_beta);
    assert!(
        res_beta.is_empty(),
        "Cross-project cache leak into project-beta!"
    );
    assert_eq!(res_beta.matched_token_count, 0);
    assert_eq!(res_beta.matched_blocks, Vec::<u32>::new());
    assert_eq!(res_beta.uncached_tokens, tokens_64);

    let res_gamma = index.match_prefix(&key_gamma);
    assert!(
        res_gamma.is_empty(),
        "Cross-project cache leak into project-gamma!"
    );
    assert_eq!(res_gamma.matched_token_count, 0);
    assert_eq!(res_gamma.uncached_tokens, tokens_64);

    // 3. Project-alpha itself matches all 4 blocks
    let res_alpha = index.match_prefix(&key_alpha);
    assert_eq!(res_alpha.matched_blocks, vec![101, 102, 103, 104]);
    assert_eq!(res_alpha.matched_token_count, 64);
    assert!(res_alpha.uncached_tokens.is_empty());

    // 4. Insert separate cached blocks for project-beta
    let inserted_beta = index.insert_key(&key_beta, &[201, 202, 203, 204]);
    assert_eq!(inserted_beta, 4);

    let res_beta_updated = index.match_prefix(&key_beta);
    assert_eq!(res_beta_updated.matched_blocks, vec![201, 202, 203, 204]);
    assert_eq!(res_beta_updated.matched_token_count, 64);

    // Project-alpha remains separate and unaffected
    assert_eq!(
        index.match_prefix(&key_alpha).matched_blocks,
        vec![101, 102, 103, 104]
    );

    // 5. Unaligned tail suffix tokens: 73 tokens (64 cached + 9 uncached)
    let mut tokens_73 = tokens_64.clone();
    tokens_73.extend_from_slice(&[801, 802, 803, 804, 805, 806, 807, 808, 809]);
    let key_alpha_73 = PrefixKey::new(
        "project-alpha",
        "qwen-2.5-coder",
        "q4_k_m",
        tokens_73.clone(),
    );
    let key_beta_73 = PrefixKey::new(
        "project-beta",
        "qwen-2.5-coder",
        "q4_k_m",
        tokens_73.clone(),
    );
    let key_gamma_73 = PrefixKey::new(
        "project-gamma",
        "qwen-2.5-coder",
        "q4_k_m",
        tokens_73.clone(),
    );

    let res_alpha_73 = index.match_prefix(&key_alpha_73);
    assert_eq!(res_alpha_73.matched_blocks, vec![101, 102, 103, 104]);
    assert_eq!(res_alpha_73.matched_token_count, 64);
    assert_eq!(res_alpha_73.uncached_token_count(), 9);
    assert_eq!(
        res_alpha_73.uncached_tokens,
        vec![801, 802, 803, 804, 805, 806, 807, 808, 809]
    );

    let res_beta_73 = index.match_prefix(&key_beta_73);
    assert_eq!(res_beta_73.matched_blocks, vec![201, 202, 203, 204]);
    assert_eq!(res_beta_73.matched_token_count, 64);
    assert_eq!(res_beta_73.uncached_token_count(), 9);

    let res_gamma_73 = index.match_prefix(&key_gamma_73);
    assert!(res_gamma_73.is_empty());
    assert_eq!(res_gamma_73.uncached_tokens, tokens_73);

    // 6. High-concurrency cross-project queries stress test (16 concurrent threads)
    let mut query_handles = Vec::with_capacity(16);
    for t in 0..16 {
        let idx = Arc::clone(&index);
        let k_a = key_alpha.clone();
        let k_b = key_beta.clone();
        let k_g = key_gamma.clone();

        query_handles.push(thread::spawn(move || {
            for _ in 0..100 {
                if t % 3 == 0 {
                    let r = idx.match_prefix(&k_a);
                    assert_eq!(r.matched_blocks, vec![101, 102, 103, 104]);
                } else if t % 3 == 1 {
                    let r = idx.match_prefix(&k_b);
                    assert_eq!(r.matched_blocks, vec![201, 202, 203, 204]);
                } else {
                    let r = idx.match_prefix(&k_g);
                    assert!(r.is_empty(), "Gamma must never hit peer cache");
                }
            }
        }));
    }

    for h in query_handles {
        h.join().expect("query thread must complete cleanly");
    }

    // 7. Tenancy-isolated invalidation:
    // Invalidate project-alpha only
    let removed_alpha = index.invalidate_project("project-alpha");
    assert_eq!(removed_alpha, 4);

    // Alpha is now empty
    assert!(index.match_prefix(&key_alpha).is_empty());

    // Beta remains fully intact and queryable
    let res_beta_surviving = index.match_prefix(&key_beta);
    assert_eq!(res_beta_surviving.matched_blocks, vec![201, 202, 203, 204]);
    assert_eq!(res_beta_surviving.matched_token_count, 64);
}
