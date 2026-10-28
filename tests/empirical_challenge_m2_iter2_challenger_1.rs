//! Empirical Adversarial Challenge Suite: Milestone M2 Iteration 2 (Challenger 1)
//!
//! Rigorous stress testing for:
//! 1. `SharedMemoryArena::evict_lru_intra_arena`:
//!    - Verifies intra-arena eviction strictly confined to caller's arena.
//!    - Verifies peer arenas are completely isolated (zero block / byte leakage).
//!    - Verifies non-volatile (`TAG_PERSIST`, `TAG_PURGE`, `TAG_FREE`) and referenced (`refcount > 0`) blocks are never evicted.
//!    - Verifies reclaimed blocks transition to `TAG_FREE` with `committed_tokens = 0`.
//!    - Boundary tests: `target_free_blocks = 0`, `target_free_blocks > evictable_count`, and saturation.
//! 2. `PrefixRadixIndex` model build disambiguation (`model_build`) and cross-project isolation:
//!    - Stress matrix across 10 projects, multiple models, and distinct model builds (`q4_k_m`, `q8_0`, `fp16`, legacy `""`).
//!    - Verifies zero cross-project leakage and zero cross-build collision.
//!    - Verifies scoped invalidations (`invalidate_model_with_build`, `invalidate_project`).
//!    - Token boundary alignment (0 tokens, unaligned 15/17/31/35 tokens, full blocks).
//! 3. Saturating refcount arithmetic (`KvBlockMeta`):
//!    - Upper ceiling saturation at `u16::MAX` without overflow.
//!    - Lower floor saturation at 0 without underflow.
//!    - State transitions (`is_shared`, `needs_cow`, `is_evictable`).
//!    - High-concurrency multi-threaded COW branching and refcount reconciliation.
//! 4. Load-time memory admission refusal:
//!    - Strict `LimitMEMLOCK` boundary enforcement with status code 11 (`InsufficientLockableMemory`).
//!    - Non-disruption and operational integrity of running model arenas.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::identity_op
)]

use std::sync::{Arc, RwLock};
use std::thread;

use k3_core::ipc::layout::{
    ARENA_MAGIC, ArenaHeader, KvBlockMeta, KvBlockTag, TAG_FREE, TAG_PERSIST, TAG_PURGE,
    TAG_VOLATILE,
};
use k3_core::ipc::protocol::StatusCode;
use k3_core::ipc::shm::SharedMemoryArena;
use k3_kv::holder::{ArenaAllocationConfig, KvError, KvHolder, compute_arena_layout};
use k3_manager::prefix::{PrefixKey, PrefixRadixIndex};

fn make_tokens(count: usize, base: u32) -> Vec<u32> {
    (0..count).map(|i| base + i as u32).collect()
}

// ============================================================================
// 1. SharedMemoryArena Intra-Arena LRU Eviction & Cross-Arena Isolation
// ============================================================================

#[test]
fn test_stress_shared_memory_arena_evict_lru_intra_arena_isolation() {
    // Create 3 genuine shared memory arenas
    let mut arena_alpha =
        SharedMemoryArena::create("shm-arena-alpha", 1024 * 1024).expect("create arena alpha");
    let mut arena_beta =
        SharedMemoryArena::create("shm-arena-beta", 2048 * 1024).expect("create arena beta");
    let mut arena_gamma =
        SharedMemoryArena::create("shm-arena-gamma", 1024 * 1024).expect("create arena gamma");

    let header_alpha = ArenaHeader {
        magic: ARENA_MAGIC,
        version: 1,
        total_bytes: 1024 * 1024,
        block_count: 8,
        n_ctx: 128,
        ..Default::default()
    };
    let header_beta = ArenaHeader {
        magic: ARENA_MAGIC,
        version: 1,
        total_bytes: 2048 * 1024,
        block_count: 6,
        n_ctx: 128,
        ..Default::default()
    };
    let header_gamma = ArenaHeader {
        magic: ARENA_MAGIC,
        version: 1,
        total_bytes: 1024 * 1024,
        block_count: 4,
        n_ctx: 64,
        ..Default::default()
    };

    *arena_alpha.header_mut().expect("header alpha") = header_alpha;
    *arena_beta.header_mut().expect("header beta") = header_beta;
    *arena_gamma.header_mut().expect("header gamma") = header_gamma;

    // Populate Arena Alpha (8 blocks with varied tags and refcounts)
    // Block 0: Volatile, unreferenced -> EVICTABLE #1
    *arena_alpha.block_meta_mut(0).unwrap() = KvBlockMeta {
        block_id: 10,
        tag: TAG_VOLATILE,
        refcount: 0,
        committed_token_count: 16,
        sampler_rng_state: [100, 0, 0, 0],
        ..Default::default()
    };
    // Block 1: Volatile, actively referenced (refcount=3) -> PROTECTED
    *arena_alpha.block_meta_mut(1).unwrap() = KvBlockMeta {
        block_id: 11,
        tag: TAG_VOLATILE,
        refcount: 3,
        committed_token_count: 16,
        sampler_rng_state: [101, 0, 0, 0],
        ..Default::default()
    };
    // Block 2: Persisted checkpoint -> PROTECTED
    *arena_alpha.block_meta_mut(2).unwrap() = KvBlockMeta {
        block_id: 12,
        tag: TAG_PERSIST,
        refcount: 0,
        committed_token_count: 16,
        sampler_rng_state: [102, 0, 0, 0],
        ..Default::default()
    };
    // Block 3: Volatile, single reference (refcount=1) -> PROTECTED
    *arena_alpha.block_meta_mut(3).unwrap() = KvBlockMeta {
        block_id: 13,
        tag: TAG_VOLATILE,
        refcount: 1,
        committed_token_count: 16,
        sampler_rng_state: [103, 0, 0, 0],
        ..Default::default()
    };
    // Block 4: Volatile, unreferenced -> EVICTABLE #2
    *arena_alpha.block_meta_mut(4).unwrap() = KvBlockMeta {
        block_id: 14,
        tag: TAG_VOLATILE,
        refcount: 0,
        committed_token_count: 16,
        sampler_rng_state: [104, 0, 0, 0],
        ..Default::default()
    };
    // Block 5: Purged tag -> PROTECTED from LRU volatile eviction
    *arena_alpha.block_meta_mut(5).unwrap() = KvBlockMeta {
        block_id: 15,
        tag: TAG_PURGE,
        refcount: 0,
        committed_token_count: 0,
        sampler_rng_state: [0, 0, 0, 0],
        ..Default::default()
    };
    // Block 6: Volatile, unreferenced -> EVICTABLE #3
    *arena_alpha.block_meta_mut(6).unwrap() = KvBlockMeta {
        block_id: 16,
        tag: TAG_VOLATILE,
        refcount: 0,
        committed_token_count: 16,
        sampler_rng_state: [106, 0, 0, 0],
        ..Default::default()
    };
    // Block 7: Persisted + referenced -> PROTECTED
    *arena_alpha.block_meta_mut(7).unwrap() = KvBlockMeta {
        block_id: 17,
        tag: TAG_PERSIST,
        refcount: 5,
        committed_token_count: 16,
        sampler_rng_state: [107, 0, 0, 0],
        ..Default::default()
    };

    // Populate Arena Beta (6 blocks)
    for i in 0..6 {
        *arena_beta.block_meta_mut(i).unwrap() = KvBlockMeta {
            block_id: 200 + i as u32,
            tag: if i % 2 == 0 {
                TAG_VOLATILE
            } else {
                TAG_PERSIST
            },
            refcount: (i as u16) % 3,
            committed_token_count: 16,
            sampler_rng_state: [200 + i as u64, 1, 2, 3],
            ..Default::default()
        };
    }
    // Snapshot initial Arena Beta state
    let beta_initial: Vec<KvBlockMeta> =
        (0..6).map(|i| *arena_beta.block_meta(i).unwrap()).collect();

    // Populate Arena Gamma (4 blocks)
    for i in 0..4 {
        *arena_gamma.block_meta_mut(i).unwrap() = KvBlockMeta {
            block_id: 300 + i as u32,
            tag: TAG_VOLATILE,
            refcount: i as u16,
            committed_token_count: 16,
            sampler_rng_state: [300 + i as u64, 9, 9, 9],
            ..Default::default()
        };
    }
    let gamma_initial: Vec<KvBlockMeta> = (0..4)
        .map(|i| *arena_gamma.block_meta(i).unwrap())
        .collect();

    // 1. Verify find_evictable_blocks identifies exactly [0, 4, 6] in Arena Alpha
    let evictable = arena_alpha
        .find_evictable_blocks()
        .expect("find evictable in alpha");
    assert_eq!(evictable, vec![0, 4, 6]);

    // 2. Evict with target_free_blocks = 0 -> must return 0 and change nothing
    let freed_zero = arena_alpha
        .evict_lru_intra_arena(0)
        .expect("evict 0 blocks");
    assert_eq!(freed_zero, 0);
    assert_eq!(
        arena_alpha.find_evictable_blocks().unwrap(),
        vec![0, 4, 6],
        "0-target eviction must not mutate any block"
    );

    // 3. Evict target_free_blocks = 2 from Arena Alpha
    let freed_two = arena_alpha
        .evict_lru_intra_arena(2)
        .expect("evict 2 blocks from alpha");
    assert_eq!(freed_two, 2);

    // Invariant: Reclaimed blocks transition to TAG_FREE with committed_tokens = 0
    let b0 = arena_alpha.block_meta(0).unwrap();
    assert_eq!(b0.tag, TAG_FREE);
    assert_eq!(b0.committed_tokens(), 0);
    assert_eq!(b0.block_id, 10);
    assert!(!b0.is_evictable());

    let b4 = arena_alpha.block_meta(4).unwrap();
    assert_eq!(b4.tag, TAG_FREE);
    assert_eq!(b4.committed_tokens(), 0);
    assert_eq!(b4.block_id, 14);
    assert!(!b4.is_evictable());

    // Invariant: Block 6 was not yet evicted and remains volatile
    let b6 = arena_alpha.block_meta(6).unwrap();
    assert_eq!(b6.tag, TAG_VOLATILE);
    assert_eq!(b6.committed_tokens(), 16);
    assert!(b6.is_evictable());

    // Invariant: Protected blocks in Arena Alpha are untouched
    let b1 = arena_alpha.block_meta(1).unwrap();
    assert_eq!(b1.tag, TAG_VOLATILE);
    assert_eq!(b1.refcount, 3);
    assert!(!b1.is_evictable());

    let b2 = arena_alpha.block_meta(2).unwrap();
    assert_eq!(b2.tag, TAG_PERSIST);
    assert_eq!(b2.refcount, 0);
    assert!(!b2.is_evictable());

    let b3 = arena_alpha.block_meta(3).unwrap();
    assert_eq!(b3.tag, TAG_VOLATILE);
    assert_eq!(b3.refcount, 1);
    assert!(!b3.is_evictable());

    let b5 = arena_alpha.block_meta(5).unwrap();
    assert_eq!(b5.tag, TAG_PURGE);
    assert!(!b5.is_evictable());

    let b7 = arena_alpha.block_meta(7).unwrap();
    assert_eq!(b7.tag, TAG_PERSIST);
    assert_eq!(b7.refcount, 5);

    // 4. Invariant: Strict Cross-Arena Isolation (Zero leakage to Arena Beta or Gamma)
    for i in 0..6 {
        let curr_b = arena_beta.block_meta(i).unwrap();
        assert_eq!(
            *curr_b, beta_initial[i],
            "Arena Beta block {i} leaked/modified!"
        );
    }
    for i in 0..4 {
        let curr_g = arena_gamma.block_meta(i).unwrap();
        assert_eq!(
            *curr_g, gamma_initial[i],
            "Arena Gamma block {i} leaked/modified!"
        );
    }

    // 5. Evict excess blocks: request 10 blocks when only 1 evictable block remains (block 6)
    let freed_excess = arena_alpha
        .evict_lru_intra_arena(10)
        .expect("evict excess from alpha");
    assert_eq!(
        freed_excess, 1,
        "Must return exactly remaining evictable count"
    );

    let b6_after = arena_alpha.block_meta(6).unwrap();
    assert_eq!(b6_after.tag, TAG_FREE);
    assert_eq!(b6_after.committed_tokens(), 0);

    // 6. Evict when 0 evictable blocks remain in Arena Alpha
    let freed_exhausted = arena_alpha
        .evict_lru_intra_arena(5)
        .expect("evict exhausted");
    assert_eq!(freed_exhausted, 0);
    assert!(arena_alpha.find_evictable_blocks().unwrap().is_empty());

    // Verify all protected blocks in Alpha still completely intact
    assert_eq!(arena_alpha.block_meta(1).unwrap().refcount, 3);
    assert_eq!(arena_alpha.block_meta(2).unwrap().tag, TAG_PERSIST);
    assert_eq!(arena_alpha.block_meta(3).unwrap().refcount, 1);
    assert_eq!(arena_alpha.block_meta(5).unwrap().tag, TAG_PURGE);
    assert_eq!(arena_alpha.block_meta(7).unwrap().refcount, 5);

    // Re-verify Arena Beta and Gamma are still 100% identical to initial snapshot
    for i in 0..6 {
        assert_eq!(*arena_beta.block_meta(i).unwrap(), beta_initial[i]);
    }
    for i in 0..4 {
        assert_eq!(*arena_gamma.block_meta(i).unwrap(), gamma_initial[i]);
    }
}

// ============================================================================
// 2. Prefix Radix Index Model Build Disambiguation & Tenancy Isolation
// ============================================================================

#[test]
fn test_stress_prefix_caching_build_disambiguation_and_cross_project() {
    let index = PrefixRadixIndex::new();

    const NUM_PROJECTS: usize = 10;
    let models = ["llama-3-8b", "qwen-2.5-7b"];
    let builds = ["q4_k_m", "q8_0", "fp16", ""]; // Note: "" tests legacy compatibility

    // 48 tokens = 3 blocks of 16 tokens
    let tokens_48 = make_tokens(48, 1000);

    // 1. Populate full matrix: 10 projects x 2 models x 4 builds = 80 distinct scopes
    for p in 0..NUM_PROJECTS {
        let proj_id = format!("project-{p}");
        for (m_idx, model_id) in models.iter().enumerate() {
            for (b_idx, build) in builds.iter().enumerate() {
                let key = PrefixKey::new(&proj_id, *model_id, *build, tokens_48.clone());
                // Unique deterministic block IDs for this specific scope
                let b0 = (p * 10000 + m_idx * 1000 + b_idx * 100 + 1) as u32;
                let b1 = (p * 10000 + m_idx * 1000 + b_idx * 100 + 2) as u32;
                let b2 = (p * 10000 + m_idx * 1000 + b_idx * 100 + 3) as u32;

                let inserted = index.insert_key(&key, &[b0, b1, b2]);
                assert_eq!(inserted, 3);
            }
        }
    }

    // 2. Verify complete isolation across all 80 scopes:
    // Every query MUST return ONLY its own blocks, never any cross-project or cross-build blocks.
    for p in 0..NUM_PROJECTS {
        let proj_id = format!("project-{p}");
        for (m_idx, model_id) in models.iter().enumerate() {
            for (b_idx, build) in builds.iter().enumerate() {
                let key = PrefixKey::new(&proj_id, *model_id, *build, tokens_48.clone());
                let result = index.match_prefix(&key);

                let expected_b0 = (p * 10000 + m_idx * 1000 + b_idx * 100 + 1) as u32;
                let expected_b1 = (p * 10000 + m_idx * 1000 + b_idx * 100 + 2) as u32;
                let expected_b2 = (p * 10000 + m_idx * 1000 + b_idx * 100 + 3) as u32;

                assert_eq!(
                    result.matched_blocks,
                    vec![expected_b0, expected_b1, expected_b2],
                    "Cross-scope collision for project={proj_id}, model={model_id}, build={build}"
                );
                assert_eq!(result.matched_token_count, 48);
                assert!(result.uncached_tokens.is_empty());
            }
        }
    }

    // 3. Test Invalidation Scoping:
    // Invalidate ONLY project-3, llama-3-8b, build q4_k_m
    let removed = index.invalidate_model_with_build("project-3", "llama-3-8b", "q4_k_m");
    assert_eq!(removed, 3);

    // Target scope is now empty
    let key_target = PrefixKey::new("project-3", "llama-3-8b", "q4_k_m", tokens_48.clone());
    assert!(index.match_prefix(&key_target).is_empty());

    // Sibling build q8_0 in project-3, llama-3-8b is UNTOUCHED
    let key_sibling_build = PrefixKey::new("project-3", "llama-3-8b", "q8_0", tokens_48.clone());
    let res_sibling = index.match_prefix(&key_sibling_build);
    assert_eq!(
        res_sibling.matched_blocks,
        vec![
            30000 + 0 + 100 + 1,
            30000 + 0 + 100 + 2,
            30000 + 0 + 100 + 3
        ]
    );

    // Sibling model qwen-2.5-7b in project-3 is UNTOUCHED
    let key_sibling_model = PrefixKey::new("project-3", "qwen-2.5-7b", "q4_k_m", tokens_48.clone());
    assert_eq!(
        index.match_prefix(&key_sibling_model).matched_blocks,
        vec![
            30000 + 1000 + 0 + 1,
            30000 + 1000 + 0 + 2,
            30000 + 1000 + 0 + 3
        ]
    );

    // Sibling project project-4 with identical model and build is UNTOUCHED
    let key_sibling_project =
        PrefixKey::new("project-4", "llama-3-8b", "q4_k_m", tokens_48.clone());
    assert_eq!(
        index.match_prefix(&key_sibling_project).matched_blocks,
        vec![40000 + 0 + 0 + 1, 40000 + 0 + 0 + 2, 40000 + 0 + 0 + 3]
    );

    // 4. Invalidate entire project-7
    let removed_proj7 = index.invalidate_project("project-7");
    // 2 models * 4 builds * 3 blocks = 24 blocks
    assert_eq!(removed_proj7, 24);

    for model_id in &models {
        for build in &builds {
            let k = PrefixKey::new("project-7", *model_id, *build, tokens_48.clone());
            assert!(index.match_prefix(&k).is_empty());
        }
    }

    // Other projects (e.g. project-6) remain 100% operational
    let key_proj6 = PrefixKey::new("project-6", "llama-3-8b", "q8_0", tokens_48.clone());
    assert_eq!(
        index.match_prefix(&key_proj6).matched_blocks,
        vec![
            60000 + 0 + 100 + 1,
            60000 + 0 + 100 + 2,
            60000 + 0 + 100 + 3
        ]
    );

    // 5. Token Alignment and Suffix Boundary Stress
    // Test 0 tokens
    let key_0 = PrefixKey::new("project-0", "llama-3-8b", "q4_k_m", vec![]);
    let res_0 = index.match_prefix(&key_0);
    assert!(res_0.is_empty());
    assert_eq!(res_0.matched_token_count, 0);
    assert!(res_0.uncached_tokens.is_empty());

    // Test 15 tokens (< 16 block boundary)
    let tokens_15 = make_tokens(15, 1000);
    let key_15 = PrefixKey::new("project-0", "llama-3-8b", "q4_k_m", tokens_15.clone());
    let res_15 = index.match_prefix(&key_15);
    assert!(res_15.is_empty());
    assert_eq!(res_15.matched_token_count, 0);
    assert_eq!(res_15.uncached_tokens, tokens_15);

    // Test 35 tokens (32 matching tokens = 2 blocks, 3 trailing uncached tokens)
    let mut tokens_35 = tokens_48[0..32].to_vec();
    tokens_35.extend_from_slice(&[9991, 9992, 9993]);
    let key_35 = PrefixKey::new("project-0", "llama-3-8b", "q4_k_m", tokens_35);
    let res_35 = index.match_prefix(&key_35);
    assert_eq!(res_35.matched_blocks, vec![1, 2]);
    assert_eq!(res_35.matched_token_count, 32);
    assert_eq!(res_35.uncached_tokens, vec![9991, 9992, 9993]);
    assert_eq!(res_35.uncached_token_count(), 3);
}

// ============================================================================
// 3. Saturating Refcount Arithmetic & High-Concurrency COW Stress
// ============================================================================

#[test]
fn test_stress_saturating_refcount_arithmetic_and_concurrency() {
    let mut meta = KvBlockMeta::new(99, 16);

    // Initial state invariants
    assert_eq!(meta.ref_count(), 1);
    assert!(!meta.is_shared());
    assert!(!meta.needs_cow());
    assert!(!meta.is_evictable());

    // 1. Upper boundary saturation test (ceiling at u16::MAX = 65535)
    meta.refcount = u16::MAX - 2;
    assert_eq!(meta.inc_ref(), u16::MAX - 1);
    assert_eq!(meta.inc_ref(), u16::MAX);
    // Saturating increments must NEVER wrap around to 0
    for _ in 0..100 {
        assert_eq!(meta.inc_ref(), u16::MAX);
    }
    assert_eq!(meta.ref_count(), u16::MAX);
    assert!(meta.is_shared());
    assert!(meta.needs_cow());
    assert!(!meta.is_evictable());

    // 2. Lower boundary saturation test (floor at 0)
    meta.refcount = 2;
    assert_eq!(meta.dec_ref(), 1);
    assert_eq!(meta.ref_count(), 1);
    assert!(!meta.is_shared());
    assert!(!meta.needs_cow());
    assert!(!meta.is_evictable());

    assert_eq!(meta.dec_ref(), 0);
    assert_eq!(meta.ref_count(), 0);
    assert!(!meta.is_shared());
    assert!(!meta.needs_cow());
    assert!(
        meta.is_evictable(),
        "Volatile block at refcount 0 must be evictable"
    );

    // Saturating decrements must NEVER underflow to 65535
    for _ in 0..100 {
        assert_eq!(meta.dec_ref(), 0);
    }
    assert_eq!(meta.ref_count(), 0);
    assert!(meta.is_evictable());

    // Non-volatile tags at refcount 0 must NOT be evictable
    meta.set_tag(KvBlockTag::Persist);
    assert!(!meta.is_evictable());
    meta.set_tag(KvBlockTag::Free);
    assert!(!meta.is_evictable());
    meta.set_tag(KvBlockTag::Purge);
    assert!(!meta.is_evictable());
    meta.set_tag(KvBlockTag::Volatile);
    assert!(meta.is_evictable());

    // 3. High-Concurrency Multi-Threaded Stress Test (64 concurrent worker sessions)
    const NUM_THREADS: usize = 64;
    let initial_data: [u32; 16] = [
        11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176,
    ];

    let shared_meta = Arc::new(RwLock::new(KvBlockMeta {
        block_id: 1,
        tag: TAG_VOLATILE,
        refcount: 1, // Single initial owner
        committed_token_count: 16,
        sampler_rng_state: [1, 2, 3, 4],
        ..Default::default()
    }));
    let shared_data = Arc::new(RwLock::new(initial_data));

    let mut handles = Vec::with_capacity(NUM_THREADS);

    for thread_id in 0..NUM_THREADS {
        let meta_clone = Arc::clone(&shared_meta);
        let data_clone = Arc::clone(&shared_data);

        handles.push(thread::spawn(move || {
            // Worker joins shared prefix: inc_ref
            let current_refs = {
                let mut guard = meta_clone.write().unwrap();
                guard.inc_ref()
            };
            assert!(current_refs >= 2);

            // Read lock: detect COW requirement
            {
                let guard = meta_clone.read().unwrap();
                assert!(guard.needs_cow());
                assert!(guard.is_shared());
            }

            // Bifurcate / Clone to private block
            let (private_meta, mut private_data) = {
                let guard = meta_clone.read().unwrap();
                let data = data_clone.read().unwrap();
                let mut pm =
                    KvBlockMeta::new((1000 + thread_id) as u32, guard.committed_token_count);
                pm.sampler_rng_state = guard.sampler_rng_state;
                (pm, *data)
            };

            // Release shared block: dec_ref
            {
                let mut guard = meta_clone.write().unwrap();
                guard.dec_ref();
            }

            // Mutate private copy
            let offset = thread_id % 16;
            private_data[offset] = 50000 + thread_id as u32;

            assert_eq!(private_meta.ref_count(), 1);
            assert!(!private_meta.needs_cow());
            assert_eq!(private_data[offset], 50000 + thread_id as u32);

            // Assert original shared buffer is unmutated
            {
                let original = data_clone.read().unwrap();
                assert_eq!(
                    original[offset], initial_data[offset],
                    "Mutation in private block leaked into shared buffer!"
                );
            }
        }));
    }

    for h in handles {
        h.join().expect("thread finished cleanly");
    }

    // Invariant: After all threads detach, shared block reconciles back to refcount 1
    let final_meta = shared_meta.read().unwrap();
    assert_eq!(final_meta.ref_count(), 1);
    assert!(!final_meta.is_shared());
    assert!(!final_meta.needs_cow());
    assert!(!final_meta.is_evictable());

    let final_data = shared_data.read().unwrap();
    assert_eq!(*final_data, initial_data);
}

// ============================================================================
// 4. Memory Admission Under LimitMEMLOCK Ceiling & Preservation
// ============================================================================

#[test]
fn test_stress_load_time_memory_admission_and_isolation() {
    let n_ctx = 64_u32;
    let config_a = ArenaAllocationConfig::new("model-live-a", n_ctx, 64, 2);
    let layout_a = compute_arena_layout(&config_a).expect("layout a");

    let config_b = ArenaAllocationConfig::new("model-live-b", n_ctx, 64, 2);
    let layout_b = compute_arena_layout(&config_b).expect("layout b");

    // Sized holder allowing exactly Model A + Model B, but refusing any Model C
    let ceiling = layout_a.total_bytes + layout_b.total_bytes;
    let holder = KvHolder::with_memlock_ceiling(ceiling);

    let alloc_a = holder.allocate(&config_a).expect("allocate a");
    let alloc_b = holder.allocate(&config_b).expect("allocate b");
    assert_eq!(holder.total_allocated_bytes(), ceiling);

    // Attempt to allocate Model C requesting 1 additional block (should fail upfront)
    let config_c = ArenaAllocationConfig::new("model-overflow", n_ctx, 64, 1);
    let err_c = holder
        .allocate(&config_c)
        .expect_err("must refuse admission when ceiling exceeded");

    assert!(matches!(err_c, KvError::InsufficientLockableMemory { .. }));
    assert_eq!(err_c.status_code(), StatusCode::InsufficientLockableMemory);
    assert_eq!(err_c.exit_code(), 11);

    // Verify existing models A and B are completely undisturbed
    assert_eq!(holder.total_allocated_bytes(), ceiling);
    let arena_a = holder.get_arena(alloc_a.arena_id).unwrap();
    assert_eq!(arena_a.header().unwrap().n_ctx, n_ctx);
    assert_eq!(arena_a.header().unwrap().validate(), Ok(()));

    let arena_b = holder.get_arena(alloc_b.arena_id).unwrap();
    assert_eq!(arena_b.header().unwrap().n_ctx, n_ctx);
    assert_eq!(arena_b.header().unwrap().validate(), Ok(()));
}
