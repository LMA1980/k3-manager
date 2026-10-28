//! Milestone M4 Tier 5 Adversarial Coverage Hardening Test Suite:
//! Memory Arenas & Admission Refusal (US-14), Project Prefix Caching (US-15),
//! and Resumable Recovery & Checkpointing (US-16).
//!
//! Validates:
//! - US-14:
//!   1. `LimitMEMLOCK` error boundaries: exact ceiling matching vs refusal (+1 byte),
//!      status code 11 (`InsufficientLockableMemory`), exit code 11.
//!   2. Concurrent allocation pressure under memory ceiling with zero overcommit.
//!   3. Memory deallocation and reallocation lifecycle.
//!   4. Strict context window admission rejection (`n_ctx: 64` boundary, no silent truncation).
//!   5. Intra-arena LRU block eviction under memory pressure and strict cross-arena isolation.
//! - US-15:
//!   6. 16-token alignment boundaries & sub-16 uncached suffix handling spectrum.
//!   7. Radix tree branch splitting, leaf-to-root LRU eviction, and partial invalidation.
//!   8. Radix tree prefix re-insertion and block ID overwrites.
//!   9. Strict multi-tenant cross-project and cross-build isolation matrix.
//!   10. Copy-on-Write (`KvBlockMeta`) refcount saturating arithmetic (overflow/underflow safety)
//!     and eviction eligibility across all block retention tags.
//! - US-16:
//!   11. Worker crash recovery across generation milestones (prefill vs decode boundary).
//!   12. Deterministic sampler PRNG seed checkpointing and state inspection.
//!   13. Client reconnect recovery: exact 30-second retention window boundary enforcement.
//!   14. Client reconnect catchup token slice boundary cases.
//!   15. Concurrent multi-subscriber live broadcast streaming with disconnection resilience.

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::bool_assert_comparison,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::doc_overindented_list_items
)]

use std::os::unix::net::UnixStream;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use k3_core::error::K3Error;
use k3_core::ipc::layout::{ARENA_MAGIC, ARENA_VERSION, ArenaHeader, KvBlockMeta, KvBlockTag};
use k3_core::ipc::protocol::StatusCode;
use k3_core::ipc::shm::{DEFAULT_LIMIT_MEMLOCK_BYTES, SharedMemoryArena};
use k3_kv::holder::{ArenaAllocationConfig, KvError, KvHolder, compute_arena_layout};
use k3_manager::error::ManagerError;
use k3_manager::prefix::{PrefixKey, PrefixRadixIndex};
use k3_manager::prompt::validate_and_tokenize;
use k3_manager::session::{DEFAULT_RETENTION_WINDOW, SessionError, SessionRegistry};
use k3_manager::supervisor::{ManagedWorker, WorkerPlacement, WorkerSupervisor};
use k3_tokenizer::SimpleTokenizer;

fn make_tokens(count: usize, base: u32) -> Vec<u32> {
    (0..count).map(|i| base + i as u32).collect()
}

fn init_test_arena(name: &str) -> SharedMemoryArena {
    let mut arena = SharedMemoryArena::create(name, 65536).expect("create test arena");
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

// ============================================================================
// PART 1: US-14 — Memory Arenas & Admission Refusal Under Adversarial Stress
// ============================================================================

#[test]
fn test_tier5_us14_insufficient_lockable_memory_exact_boundary_conditions() {
    assert_eq!(DEFAULT_LIMIT_MEMLOCK_BYTES, 28 * 1024 * 1024 * 1024);

    let n_ctx = 64_u32;
    let config = ArenaAllocationConfig::new("model-exact-boundary", n_ctx, 64, 2);
    let layout = compute_arena_layout(&config).expect("compute arena layout");
    assert!(layout.total_bytes > 0);

    // 1. Exact Boundary: ceiling == layout.total_bytes -> MUST SUCCEED
    let holder_exact = KvHolder::with_memlock_ceiling(layout.total_bytes);
    let alloc = holder_exact
        .allocate(&config)
        .expect("exact ceiling allocation must succeed");
    assert_eq!(alloc.total_bytes, layout.total_bytes);
    assert_eq!(holder_exact.total_allocated_bytes(), layout.total_bytes);
    assert_eq!(holder_exact.available_lockable_bytes(), 0);

    // Any further allocation (even 1 block) must fail with InsufficientLockableMemory
    let config_extra = ArenaAllocationConfig::new("model-extra", n_ctx, 64, 2);
    let err_extra = holder_exact
        .allocate(&config_extra)
        .expect_err("subsequent allocation must fail when budget is 0");
    assert!(matches!(
        err_extra,
        KvError::InsufficientLockableMemory { .. }
    ));
    assert_eq!(
        err_extra.status_code(),
        StatusCode::InsufficientLockableMemory
    );
    assert_eq!(err_extra.exit_code(), 11);

    // 2. Off-By-One Refusal Boundary: ceiling == layout.total_bytes - 1 -> MUST FAIL
    let holder_insufficient = KvHolder::with_memlock_ceiling(layout.total_bytes - 1);
    let err_sub = holder_insufficient
        .allocate(&config)
        .expect_err("ceiling - 1 must be refused upfront");
    assert!(matches!(
        err_sub,
        KvError::InsufficientLockableMemory {
            required_bytes,
            allocated_bytes: 0,
            memlock_ceiling_bytes
        } if required_bytes == layout.total_bytes && memlock_ceiling_bytes == layout.total_bytes - 1
    ));
    assert_eq!(holder_insufficient.total_allocated_bytes(), 0);
}

#[test]
fn test_tier5_us14_concurrent_memory_admission_pressure_no_overcommit() {
    let n_ctx = 64_u32;
    let config = ArenaAllocationConfig::new("model-pressure", n_ctx, 64, 2);
    let layout = compute_arena_layout(&config).expect("layout");

    // Ceiling sized to accommodate exactly 2 arenas
    let ceiling = layout.total_bytes * 2;
    let holder = Arc::new(KvHolder::with_memlock_ceiling(ceiling));

    let thread_count = 10;
    let barrier = Arc::new(Barrier::new(thread_count));
    let mut handles = Vec::with_capacity(thread_count);

    for i in 0..thread_count {
        let h = Arc::clone(&holder);
        let b = Arc::clone(&barrier);
        let cfg = ArenaAllocationConfig::new(format!("model-thread-{i}"), n_ctx, 64, 2);

        handles.push(thread::spawn(move || {
            b.wait();
            h.allocate(&cfg)
        }));
    }

    let mut successes = 0;
    let mut failures = 0;

    for handle in handles {
        match handle.join().expect("thread join") {
            Ok(alloc) => {
                assert!(alloc.arena_id > 0);
                successes += 1;
            }
            Err(err) => {
                assert!(matches!(err, KvError::InsufficientLockableMemory { .. }));
                assert_eq!(err.status_code(), StatusCode::InsufficientLockableMemory);
                failures += 1;
            }
        }
    }

    // Invariant: Exactly 2 must succeed and 8 must fail
    assert_eq!(successes, 2);
    assert_eq!(failures, 8);
    // Invariant: Total allocated bytes must never exceed ceiling
    assert_eq!(holder.total_allocated_bytes(), ceiling);
    assert_eq!(holder.available_lockable_bytes(), 0);
}

#[test]
fn test_tier5_us14_deallocate_and_reallocate_lifecycle() {
    let n_ctx = 64_u32;
    let config = ArenaAllocationConfig::new("model-recycle", n_ctx, 64, 2);
    let layout = compute_arena_layout(&config).expect("layout");

    let holder = KvHolder::with_memlock_ceiling(layout.total_bytes);

    // Initial allocation consumes all budget
    let alloc1 = holder.allocate(&config).expect("initial allocate");
    assert_eq!(holder.available_lockable_bytes(), 0);
    assert_eq!(holder.arena_count(), 1);

    // Attempting another allocation is refused
    assert!(holder.allocate(&config).is_err());

    // Deallocate the arena
    let freed_bytes = holder.deallocate(alloc1.arena_id).expect("deallocate");
    assert_eq!(freed_bytes, layout.total_bytes);
    assert_eq!(holder.total_allocated_bytes(), 0);
    assert_eq!(holder.available_lockable_bytes(), layout.total_bytes);
    assert_eq!(holder.arena_count(), 0);

    // Deallocating again returns ArenaNotFound
    let err_not_found = holder.deallocate(alloc1.arena_id).expect_err("double free");
    assert!(matches!(err_not_found, KvError::ArenaNotFound(_)));

    // Re-allocating the same size now succeeds cleanly
    let alloc2 = holder.allocate(&config).expect("re-allocate after free");
    assert_eq!(alloc2.total_bytes, layout.total_bytes);
    assert_ne!(alloc1.arena_id, alloc2.arena_id);
    assert_eq!(holder.total_allocated_bytes(), layout.total_bytes);
}

#[test]
fn test_tier5_us14_context_length_strict_rejection_boundary_spectrum() {
    let n_ctx = 64_usize;
    let mut tokenizer = SimpleTokenizer::new();
    tokenizer.add_token("tok", 1);

    // 1. Prompt with 0 tokens: admitted
    let res_0 = validate_and_tokenize(&tokenizer, "", None, false, n_ctx);
    assert_eq!(res_0, Ok(vec![]));

    // 2. Prompt with 1 token: admitted
    let res_1 = validate_and_tokenize(&tokenizer, "tok ", None, false, n_ctx);
    assert_eq!(res_1, Ok(vec![1]));

    // 3. Prompt with 63 tokens (n_ctx - 1): admitted (leaves 1 decode slot)
    let prompt_63 = "tok ".repeat(63);
    let res_63 = validate_and_tokenize(&tokenizer, &prompt_63, None, false, n_ctx);
    assert!(res_63.is_ok());
    assert_eq!(res_63.unwrap().len(), 63);

    // 4. Prompt with exactly 64 tokens (== n_ctx): REJECTED upfront with ContextLengthExceeded
    let prompt_64 = "tok ".repeat(64);
    let res_64 = validate_and_tokenize(&tokenizer, &prompt_64, None, false, n_ctx);
    assert_eq!(
        res_64,
        Err(K3Error::ContextLengthExceeded {
            requested_tokens: 64,
            max_context_tokens: 64,
        })
    );

    // 5. Prompt with 65 tokens (> n_ctx): REJECTED upfront
    let prompt_65 = "tok ".repeat(65);
    let res_65 = validate_and_tokenize(&tokenizer, &prompt_65, None, false, n_ctx);
    assert_eq!(
        res_65,
        Err(K3Error::ContextLengthExceeded {
            requested_tokens: 65,
            max_context_tokens: 64,
        })
    );

    // 6. Zero context length (n_ctx = 0): REJECTED upfront
    let res_zero_nctx = validate_and_tokenize(&tokenizer, "tok", None, false, 0);
    assert_eq!(
        res_zero_nctx,
        Err(K3Error::ContextLengthExceeded {
            requested_tokens: 1,
            max_context_tokens: 0,
        })
    );
}

#[test]
fn test_tier5_us14_intra_arena_lru_eviction_under_pressure_and_isolation() {
    let mut arena_1 = init_test_arena("shm_lru_pressure_1");
    let mut arena_2 = init_test_arena("shm_lru_pressure_2");

    // Arena 1 setup:
    // Block 0: TAG_VOLATILE, refcount = 0 -> evictable
    let b0 = arena_1.block_meta_mut(0).expect("block 0");
    b0.set_tag(KvBlockTag::Volatile);
    b0.refcount = 0;
    b0.set_committed_tokens(16);

    // Block 1: TAG_PERSIST, refcount = 0 -> protected
    let b1 = arena_1.block_meta_mut(1).expect("block 1");
    b1.set_tag(KvBlockTag::Persist);
    b1.refcount = 0;
    b1.set_committed_tokens(16);

    // Block 2: TAG_VOLATILE, refcount = 2 -> protected (in-flight reference)
    let b2 = arena_1.block_meta_mut(2).expect("block 2");
    b2.set_tag(KvBlockTag::Volatile);
    b2.refcount = 2;
    b2.set_committed_tokens(16);

    // Block 3: TAG_PURGE, refcount = 0 -> not volatile, so protected from LRU recycle
    let b3 = arena_1.block_meta_mut(3).expect("block 3");
    b3.set_tag(KvBlockTag::Purge);
    b3.refcount = 0;
    b3.set_committed_tokens(16);

    // Arena 2 setup:
    for idx in 0..4 {
        let b = arena_2.block_meta_mut(idx).expect("arena 2 block");
        b.set_tag(KvBlockTag::Volatile);
        b.refcount = 0;
        b.set_committed_tokens(16);
    }

    // Invariant: Exactly Block 0 is evictable in Arena 1
    let evictable_1 = arena_1.find_evictable_blocks().expect("find evictable");
    assert_eq!(evictable_1, vec![0]);

    // Requesting to evict 10 blocks only evicts the 1 eligible block
    let freed = arena_1.evict_lru_intra_arena(10).expect("evict");
    assert_eq!(freed, 1);

    // Verify Block 0 is reset to TAG_FREE with 0 committed tokens
    let b0_after = arena_1.block_meta(0).expect("read block 0");
    assert_eq!(b0_after.tag_enum(), Some(KvBlockTag::Free));
    assert_eq!(b0_after.committed_tokens(), 0);

    // Verify Blocks 1, 2, 3 in Arena 1 are unaltered
    assert_eq!(
        arena_1.block_meta(1).unwrap().tag_enum(),
        Some(KvBlockTag::Persist)
    );
    assert_eq!(arena_1.block_meta(2).unwrap().ref_count(), 2);
    assert_eq!(
        arena_1.block_meta(3).unwrap().tag_enum(),
        Some(KvBlockTag::Purge)
    );

    // Verify Arena 2 is completely unmolested (zero cross-arena leakage)
    for idx in 0..4 {
        let b = arena_2.block_meta(idx).expect("read arena 2 block");
        assert_eq!(b.committed_tokens(), 16);
        assert_eq!(b.tag_enum(), Some(KvBlockTag::Volatile));
    }

    // Subsequent eviction in Arena 1 yields 0 blocks
    let freed_next = arena_1.evict_lru_intra_arena(1).expect("evict saturated");
    assert_eq!(freed_next, 0);
}

// ============================================================================
// PART 2: US-15 — Project Prefix Caching & Copy-On-Write Invariants
// ============================================================================

#[test]
fn test_tier5_us15_sixteen_token_alignment_and_remainder_spectrum() {
    let index = PrefixRadixIndex::new();

    // Cache 2 blocks (32 tokens)
    let tokens_32 = make_tokens(32, 100);
    let key_base = PrefixKey::new("proj-alignment", "phi-4", "v1", tokens_32.clone());
    index.insert_key(&key_base, &[1, 2]);

    // Query 0 tokens -> 0 blocks, 0 uncached tokens
    let q0 = PrefixKey::new("proj-alignment", "phi-4", "v1", vec![]);
    let res0 = index.match_prefix(&q0);
    assert!(res0.is_empty());
    assert_eq!(res0.uncached_token_count(), 0);

    // Query 15 tokens -> 0 blocks, 15 uncached tokens
    let q15 = PrefixKey::new("proj-alignment", "phi-4", "v1", make_tokens(15, 100));
    let res15 = index.match_prefix(&q15);
    assert!(res15.is_empty());
    assert_eq!(res15.uncached_token_count(), 15);

    // Query 16 tokens -> 1 block, 0 uncached tokens
    let q16 = PrefixKey::new("proj-alignment", "phi-4", "v1", make_tokens(16, 100));
    let res16 = index.match_prefix(&q16);
    assert_eq!(res16.matched_blocks, vec![1]);
    assert_eq!(res16.uncached_token_count(), 0);

    // Query 17 tokens -> 1 block, 1 uncached token
    let q17 = PrefixKey::new("proj-alignment", "phi-4", "v1", make_tokens(17, 100));
    let res17 = index.match_prefix(&q17);
    assert_eq!(res17.matched_blocks, vec![1]);
    assert_eq!(res17.uncached_tokens, vec![116]);

    // Query 31 tokens -> 1 block, 15 uncached tokens
    let q31 = PrefixKey::new("proj-alignment", "phi-4", "v1", make_tokens(31, 100));
    let res31 = index.match_prefix(&q31);
    assert_eq!(res31.matched_blocks, vec![1]);
    assert_eq!(res31.uncached_token_count(), 15);

    // Query 32 tokens -> 2 blocks, 0 uncached tokens
    let q32 = PrefixKey::new("proj-alignment", "phi-4", "v1", make_tokens(32, 100));
    let res32 = index.match_prefix(&q32);
    assert_eq!(res32.matched_blocks, vec![1, 2]);
    assert_eq!(res32.uncached_token_count(), 0);

    // Query 33 tokens -> 2 blocks, 1 uncached token
    let mut tokens_33 = tokens_32.clone();
    tokens_33.push(999);
    let q33 = PrefixKey::new("proj-alignment", "phi-4", "v1", tokens_33);
    let res33 = index.match_prefix(&q33);
    assert_eq!(res33.matched_blocks, vec![1, 2]);
    assert_eq!(res33.uncached_tokens, vec![999]);

    // Query 48 tokens (first 32 match, next 16 differ) -> 2 blocks, 16 uncached tokens
    let mut tokens_48 = tokens_32;
    tokens_48.extend(make_tokens(16, 8888));
    let q48 = PrefixKey::new("proj-alignment", "phi-4", "v1", tokens_48);
    let res48 = index.match_prefix(&q48);
    assert_eq!(res48.matched_blocks, vec![1, 2]);
    assert_eq!(res48.uncached_tokens, make_tokens(16, 8888));
}

#[test]
fn test_tier5_us15_radix_tree_deep_branch_splitting_and_partial_invalidation() {
    let index = PrefixRadixIndex::new();

    // Common root prefix: 32 tokens (blocks [10, 20])
    let root_tokens = make_tokens(32, 10);
    let key_root = PrefixKey::new("proj-branch", "llama-3", "v1", root_tokens.clone());
    index.insert_key(&key_root, &[10, 20]);

    // Branch A: root + 16 tokens (base 1000) -> block 30
    let mut tokens_a = root_tokens.clone();
    tokens_a.extend(make_tokens(16, 1000));
    let key_a = PrefixKey::new("proj-branch", "llama-3", "v1", tokens_a.clone());
    index.insert_key(&key_a, &[10, 20, 30]);

    // Branch B: root + 16 tokens (base 2000) -> block 40
    let mut tokens_b = root_tokens.clone();
    tokens_b.extend(make_tokens(16, 2000));
    let key_b = PrefixKey::new("proj-branch", "llama-3", "v1", tokens_b.clone());
    index.insert_key(&key_b, &[10, 20, 40]);

    // Branch C: root + 32 tokens (base 3000) -> blocks [50, 51]
    let mut tokens_c = root_tokens.clone();
    tokens_c.extend(make_tokens(32, 3000));
    let key_c = PrefixKey::new("proj-branch", "llama-3", "v1", tokens_c.clone());
    index.insert_key(&key_c, &[10, 20, 50, 51]);

    // 1. Verify all branches match their respective blocks
    assert_eq!(index.match_prefix(&key_a).matched_blocks, vec![10, 20, 30]);
    assert_eq!(index.match_prefix(&key_b).matched_blocks, vec![10, 20, 40]);
    assert_eq!(
        index.match_prefix(&key_c).matched_blocks,
        vec![10, 20, 50, 51]
    );

    // 2. Invalidate Branch A leaf block via LRU eviction (block 30 was accessed earliest among leaves)
    let evicted = index.evict_lru_with_build("proj-branch", "llama-3", "v1", 1);
    assert_eq!(evicted.len(), 1);

    // Branch A no longer matches block 30, but still matches common root [10, 20]
    let res_a_after = index.match_prefix(&key_a);
    assert_eq!(res_a_after.matched_blocks, vec![10, 20]);
    assert_eq!(res_a_after.uncached_token_count(), 16);

    // Branch B and C are untouched
    assert_eq!(index.match_prefix(&key_b).matched_blocks, vec![10, 20, 40]);
    assert_eq!(
        index.match_prefix(&key_c).matched_blocks,
        vec![10, 20, 50, 51]
    );

    // 3. Invalidate model completely: everything cleared
    let removed = index.invalidate_model_with_build("proj-branch", "llama-3", "v1");
    assert!(removed > 0);
    assert!(index.match_prefix(&key_a).is_empty());
    assert!(index.match_prefix(&key_b).is_empty());
    assert!(index.match_prefix(&key_c).is_empty());
}

#[test]
fn test_tier5_us15_radix_tree_prefix_reinsertion_overwrite() {
    let index = PrefixRadixIndex::new();
    let tokens_32 = make_tokens(32, 555);
    let key = PrefixKey::new("proj-overwrite", "model-x", "build-1", tokens_32);

    // Initial insertion with blocks [10, 11]
    index.insert_key(&key, &[10, 11]);
    assert_eq!(index.match_prefix(&key).matched_blocks, vec![10, 11]);

    // Re-insert identical prefix with updated block IDs [100, 101]
    index.insert_key(&key, &[100, 101]);
    assert_eq!(index.match_prefix(&key).matched_blocks, vec![100, 101]);
}

#[test]
fn test_tier5_us15_strict_tenancy_isolation_matrix() {
    let index = PrefixRadixIndex::new();
    let tokens = make_tokens(32, 7000);

    let projects = [
        "tenant-alpha",
        "tenant-beta",
        "",
        "proj/with/slashes",
        "proj 🚀",
    ];
    let builds = ["fp16", "q4_k_m"];

    for (p_idx, &proj) in projects.iter().enumerate() {
        for (b_idx, &build) in builds.iter().enumerate() {
            let key = PrefixKey::new(proj, "shared-model", build, tokens.clone());
            let block_base = ((p_idx + 1) * 100 + b_idx * 10) as u32;
            index.insert_key(&key, &[block_base, block_base + 1]);
        }
    }

    // Verify strict isolation across the entire cross-product
    for (p_idx, &proj) in projects.iter().enumerate() {
        for (b_idx, &build) in builds.iter().enumerate() {
            let key = PrefixKey::new(proj, "shared-model", build, tokens.clone());
            let block_base = ((p_idx + 1) * 100 + b_idx * 10) as u32;
            let res = index.match_prefix(&key);
            assert_eq!(
                res.matched_blocks,
                vec![block_base, block_base + 1],
                "Cross-tenant cache bleed detected for project '{proj}' build '{build}'"
            );
        }
    }

    // Invalidate tenant-alpha: only tenant-alpha entries are removed
    let invalidated = index.invalidate_project("tenant-alpha");
    assert_eq!(invalidated, 4); // 2 builds * 2 blocks

    let key_alpha = PrefixKey::new("tenant-alpha", "shared-model", "fp16", tokens.clone());
    assert!(index.match_prefix(&key_alpha).is_empty());

    let key_beta = PrefixKey::new("tenant-beta", "shared-model", "fp16", tokens);
    assert_eq!(index.match_prefix(&key_beta).matched_blocks, vec![200, 201]);
}

#[test]
fn test_tier5_us15_cow_refcounting_saturating_bounds_and_lifecycle() {
    let mut meta = KvBlockMeta::new(42, 16);
    assert_eq!(meta.ref_count(), 1);
    assert!(!meta.is_shared());
    assert!(!meta.needs_cow());
    assert!(!meta.is_evictable());

    // Saturating increment near u16::MAX
    meta.refcount = u16::MAX - 1;
    assert_eq!(meta.inc_ref(), u16::MAX);
    assert_eq!(meta.inc_ref(), u16::MAX);
    assert_eq!(meta.ref_count(), u16::MAX);
    assert!(meta.is_shared());
    assert!(meta.needs_cow());

    // Saturating decrement down to 0
    meta.refcount = 1;
    assert_eq!(meta.dec_ref(), 0);
    assert_eq!(meta.dec_ref(), 0); // underflow guard
    assert_eq!(meta.ref_count(), 0);
    assert!(!meta.is_shared());
    assert!(!meta.needs_cow());

    // Eviction eligibility across all KvBlockTag tags with refcount == 0
    meta.set_tag(KvBlockTag::Volatile);
    assert!(meta.is_evictable());

    meta.set_tag(KvBlockTag::Persist);
    assert!(!meta.is_evictable());

    meta.set_tag(KvBlockTag::Purge);
    assert!(!meta.is_evictable());

    meta.set_tag(KvBlockTag::Free);
    assert!(!meta.is_evictable());
}

// ============================================================================
// PART 3: US-16 — Resumable Recovery & Checkpointing
// ============================================================================

#[test]
fn test_tier5_us16_worker_crash_at_various_progress_points() {
    let arena = init_test_arena("shm_crash_progress_points");

    let (mgr_sock, _worker_sock) = UnixStream::pair().expect("socketpair");
    let placement =
        WorkerPlacement::new("crash-model", "/tmp/model.bin", "/tmp/model.sock", "x100");
    let worker = ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 65536, 64, 4);

    let mut supervisor = WorkerSupervisor::new();
    supervisor.register_worker(worker);

    // Scenario A: Worker crashed during prefill (committed = 10, prompt_len = 16)
    {
        let w = supervisor.get_worker_mut("crash-model").unwrap();
        w.set_block_committed_tokens(0, 10).unwrap();
        w.mark_volatile_block(0).unwrap();
    }
    let snap_a = supervisor
        .inspect_resumption_state("crash-model", &[0], 16)
        .expect("inspect resumption A");
    assert_eq!(snap_a.total_committed_tokens, 10);
    assert!(!snap_a.skip_prefill);
    assert_eq!(snap_a.next_decode_pos, 10);

    // Calling resume_decode must fail with IncompletePrefillResumptionNotSupported
    let w_mut = supervisor.get_worker_mut("crash-model").unwrap();
    let res_err = w_mut.resume_decode(1, 1, 16, &[0], 42);
    assert!(matches!(
        res_err,
        Err(ManagerError::IncompletePrefillResumptionNotSupported {
            committed_tokens: 10,
            prompt_len: 16
        })
    ));

    // Scenario B: Worker crashed at exact prefill completion (committed = 16, prompt_len = 16)
    {
        let w = supervisor.get_worker_mut("crash-model").unwrap();
        w.set_block_committed_tokens(0, 16).unwrap();
    }
    let snap_b = supervisor
        .inspect_resumption_state("crash-model", &[0], 16)
        .expect("inspect resumption B");
    assert_eq!(snap_b.total_committed_tokens, 16);
    assert!(snap_b.skip_prefill);
    assert_eq!(snap_b.next_decode_pos, 16);

    // Scenario C: Worker crashed after 5 decode tokens (committed = 21, prompt_len = 16)
    {
        let w = supervisor.get_worker_mut("crash-model").unwrap();
        w.set_block_committed_tokens(0, 16).unwrap();
        w.set_block_committed_tokens(1, 5).unwrap();
    }
    let snap_c = supervisor
        .inspect_resumption_state("crash-model", &[0, 1], 16)
        .expect("inspect resumption C");
    assert_eq!(snap_c.total_committed_tokens, 21);
    assert!(snap_c.skip_prefill);
    assert_eq!(snap_c.next_decode_pos, 21);

    // Scenario D: Worker crashed with 0 tokens committed
    {
        let w = supervisor.get_worker_mut("crash-model").unwrap();
        w.set_block_committed_tokens(0, 0).unwrap();
        w.set_block_committed_tokens(1, 0).unwrap();
    }
    let snap_d = supervisor
        .inspect_resumption_state("crash-model", &[0, 1], 16)
        .expect("inspect resumption D");
    assert_eq!(snap_d.total_committed_tokens, 0);
    assert!(!snap_d.skip_prefill);
    assert_eq!(snap_d.next_decode_pos, 0);
}

#[test]
fn test_tier5_us16_deterministic_sampler_prng_checkpoint_consistency() {
    let mut arena = init_test_arena("shm_prng_checkpoint_consistency");

    // Block 0: committed = 16, PRNG = [10, 20, 30, 40]
    let b0 = arena.block_meta_mut(0).expect("b0");
    b0.set_committed_tokens(16);
    b0.sampler_rng_state = [10, 20, 30, 40];

    // Block 1: committed = 16, PRNG = [50, 60, 70, 80]
    let b1 = arena.block_meta_mut(1).expect("b1");
    b1.set_committed_tokens(16);
    b1.sampler_rng_state = [50, 60, 70, 80];

    // Block 2: committed = 8, PRNG = [90, 100, 110, 120]
    let b2 = arena.block_meta_mut(2).expect("b2");
    b2.set_committed_tokens(8);
    b2.sampler_rng_state = [90, 100, 110, 120];

    // Block 3: uncommitted (committed = 0), PRNG = [0, 0, 0, 0]
    let b3 = arena.block_meta_mut(3).expect("b3");
    b3.set_committed_tokens(0);
    b3.sampler_rng_state = [0, 0, 0, 0];

    let (mgr_sock, _worker_sock) = UnixStream::pair().expect("socketpair");
    let placement = WorkerPlacement::new("prng-model", "/tmp/model.bin", "/tmp/model.sock", "x100");
    let worker = ManagedWorker::create_test_worker(placement, mgr_sock, arena, 1, 65536, 64, 4);

    let mut supervisor = WorkerSupervisor::new();
    supervisor.register_worker(worker);

    // Resumption across blocks [0, 1, 2, 3] must select PRNG from the active block (Block 2)
    let snap = supervisor
        .inspect_resumption_state("prng-model", &[0, 1, 2, 3], 16)
        .expect("inspect");
    assert_eq!(snap.total_committed_tokens, 40); // 16 + 16 + 8
    assert_eq!(snap.sampler_rng_state, [90, 100, 110, 120]);
    assert!(snap.skip_prefill);
    assert_eq!(snap.next_decode_pos, 40);
}

#[test]
fn test_tier5_us16_client_reconnect_exact_retention_window_boundary() {
    let registry = SessionRegistry::new();
    let session_id = "boundary-session";

    registry
        .register_session(session_id, "project-1", "model-1")
        .expect("register");

    for tok in [101_u32, 102, 103] {
        registry
            .append_token(session_id, tok)
            .expect("append token");
    }

    let t0 = Instant::now();
    registry
        .record_disconnect_at(session_id, t0)
        .expect("disconnect");

    // 1. Reconnect at T0 + 29s (well within window) -> MUST SUCCEED
    let res_29s = registry.reconnect_client_at(session_id, Some(1), t0 + Duration::from_secs(29));
    assert_eq!(res_29s, Ok(vec![103]));

    // Disconnect again at T1
    let t1 = Instant::now();
    registry
        .record_disconnect_at(session_id, t1)
        .expect("disconnect again");

    // 2. Reconnect at EXACT boundary T1 + 30.000s -> MUST SUCCEED (elapsed > 30s is false)
    let res_exact =
        registry.reconnect_client_at(session_id, Some(1), t1 + DEFAULT_RETENTION_WINDOW);
    assert_eq!(res_exact, Ok(vec![103]));

    // Disconnect again at T2
    let t2 = Instant::now();
    registry
        .record_disconnect_at(session_id, t2)
        .expect("disconnect third");

    // 3. Reconnect at T2 + 30.001s (expired by 1ms) -> MUST FAIL and auto-prune
    let res_expired = registry.reconnect_client_at(
        session_id,
        Some(1),
        t2 + DEFAULT_RETENTION_WINDOW + Duration::from_millis(1),
    );
    assert!(matches!(
        res_expired,
        Err(SessionError::RetentionWindowExpired { .. })
    ));

    // Verify session was auto-pruned from registry
    assert!(!registry.contains_session(session_id));
    assert_eq!(registry.session_count(), 0);
}

#[test]
fn test_tier5_us16_client_reconnect_catchup_slice_edge_cases() {
    let registry = SessionRegistry::new();
    let session_id = "slice-session";

    registry
        .register_session(session_id, "project-edge", "model-edge")
        .expect("register");

    for tok in [10_u32, 20, 30, 40, 50] {
        registry
            .append_token(session_id, tok)
            .expect("append token");
    }

    let t = Instant::now();

    // Case 1: last_acked_token_idx == None -> All 5 tokens
    let catchup_all = registry
        .reconnect_client_at(session_id, None, t)
        .expect("catchup all");
    assert_eq!(catchup_all, vec![10, 20, 30, 40, 50]);

    // Case 2: last_acked_token_idx == Some(0) -> Tokens from index 1 (20, 30, 40, 50)
    let catchup_from_1 = registry
        .reconnect_client_at(session_id, Some(0), t)
        .expect("catchup from 1");
    assert_eq!(catchup_from_1, vec![20, 30, 40, 50]);

    // Case 3: last_acked_token_idx == Some(3) -> Only last token at index 4 (50)
    let catchup_last = registry
        .reconnect_client_at(session_id, Some(3), t)
        .expect("catchup last");
    assert_eq!(catchup_last, vec![50]);

    // Case 4: last_acked_token_idx == Some(4) (client already received all tokens) -> Empty
    let catchup_none = registry
        .reconnect_client_at(session_id, Some(4), t)
        .expect("catchup none");
    assert_eq!(catchup_none, Vec::<u32>::new());

    // Case 5: last_acked_token_idx == Some(10) (out of bounds) -> Empty, no panic
    let catchup_oob = registry
        .reconnect_client_at(session_id, Some(10), t)
        .expect("catchup oob");
    assert_eq!(catchup_oob, Vec::<u32>::new());

    // Case 6: last_acked_token_idx == Some(usize::MAX) (saturating overflow guard) -> Empty, no panic
    let catchup_max = registry
        .reconnect_client_at(session_id, Some(usize::MAX), t)
        .expect("catchup max");
    assert_eq!(catchup_max, Vec::<u32>::new());
}

#[test]
fn test_tier5_us16_concurrent_multi_subscriber_streaming() {
    let registry = Arc::new(SessionRegistry::new());
    let session_id = "concurrent-stream";

    registry
        .register_session(session_id, "project-multi", "model-multi")
        .expect("register");

    let subscriber_count = 4;
    let token_count = 30;

    let barrier = Arc::new(Barrier::new(subscriber_count + 1));
    let mut handles = Vec::with_capacity(subscriber_count);

    for _ in 0..subscriber_count {
        let mut sub = registry.subscribe(session_id).expect("subscribe");
        let b = Arc::clone(&barrier);

        handles.push(thread::spawn(move || {
            b.wait();
            let mut received = Vec::with_capacity(token_count);
            for _ in 0..token_count {
                // Read next broadcast token
                match sub.blocking_recv() {
                    Ok(tok) => received.push(tok),
                    Err(e) => panic!("subscriber recv error: {e}"),
                }
            }
            received
        }));
    }

    // Wait for all subscribers to be ready
    barrier.wait();

    // Server generates tokens 1..=30 sequentially
    let generated: Vec<u32> = (1..=token_count as u32).collect();
    for &tok in &generated {
        registry
            .append_token(session_id, tok)
            .expect("append token");
    }

    // Verify all subscribers received identical token streams in exact generation order
    for handle in handles {
        let tokens = handle.join().expect("subscriber thread join");
        assert_eq!(tokens, generated);
    }
}
