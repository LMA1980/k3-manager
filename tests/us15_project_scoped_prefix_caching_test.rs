//! US-15 Test Suite: Project-Scoped Prefix Caching & COW Refcounting
//!
//! Verifies:
//! 1. `PrefixKey` model build disambiguation (`model_id` + `model_build`).
//! 2. Strict cross-project prefix caching isolation (`project_id`).
//! 3. 16-token paged KV cache block alignment and uncached tail suffix tokens.
//! 4. Copy-on-Write (`KvBlockMeta`) refcounting lifecycle, saturating arithmetic, and eviction eligibility.
//! 5. Simulated COW paged block sharing and copy-on-write bifurcation.

#![allow(clippy::cast_possible_truncation, clippy::assert_is_empty)]

use k3_core::ipc::layout::KvBlockMeta;
use k3_manager::prefix::{PrefixKey, PrefixRadixIndex};

fn make_tokens(count: usize, base: u32) -> Vec<u32> {
    (0..count).map(|i| base + i as u32).collect()
}

#[test]
fn test_us15_prefix_key_model_build_disambiguation() {
    let index = PrefixRadixIndex::new();

    let tokens_32 = make_tokens(32, 1000);
    let key_q4 = PrefixKey::new("project-alpha", "llama-3-8b", "q4_k_m", tokens_32.clone());
    let key_q8 = PrefixKey::new("project-alpha", "llama-3-8b", "q8_0", tokens_32.clone());

    // 1. Insert 2 blocks for q4_k_m
    let inserted = index.insert_key(&key_q4, &[10, 11]);
    assert_eq!(inserted, 2);

    // 2. Query with q8_0 -> must miss completely due to model_build disambiguation
    let miss_res = index.match_prefix(&key_q8);
    assert!(miss_res.is_empty());
    assert_eq!(miss_res.matched_blocks, Vec::<u32>::new());
    assert_eq!(miss_res.uncached_tokens, tokens_32);

    // 3. Query with q4_k_m -> must hit both blocks
    let hit_res = index.match_prefix(&key_q4);
    assert_eq!(hit_res.matched_blocks, vec![10, 11]);
    assert_eq!(hit_res.matched_token_count, 32);
    assert!(hit_res.uncached_tokens.is_empty());

    // 4. Insert separate blocks for q8_0
    let inserted_q8 = index.insert_key(&key_q8, &[20, 21]);
    assert_eq!(inserted_q8, 2);

    // 5. Query both - each gets its own independent cached blocks
    let res_q4 = index.match_prefix(&key_q4);
    assert_eq!(res_q4.matched_blocks, vec![10, 11]);

    let res_q8 = index.match_prefix(&key_q8);
    assert_eq!(res_q8.matched_blocks, vec![20, 21]);

    // 6. Invalidate only q4_k_m
    let invalidated = index.invalidate_model_with_build("project-alpha", "llama-3-8b", "q4_k_m");
    assert_eq!(invalidated, 2);

    // q4 is gone, q8 remains intact
    assert!(index.match_prefix(&key_q4).is_empty());
    assert_eq!(index.match_prefix(&key_q8).matched_blocks, vec![20, 21]);
}

#[test]
fn test_us15_strict_cross_project_isolation() {
    let index = PrefixRadixIndex::new();

    // 48 tokens = 3 blocks of 16 tokens
    let tokens_48 = make_tokens(48, 500);

    let key_corp_a = PrefixKey::new("corp-a", "qwen-2.5-coder", "fp16", tokens_48.clone());
    let key_corp_b = PrefixKey::new("corp-b", "qwen-2.5-coder", "fp16", tokens_48.clone());

    // Insert for corp-a
    let inserted_a = index.insert_key(&key_corp_a, &[101, 102, 103]);
    assert_eq!(inserted_a, 3);

    // Query from corp-b MUST yield 0 matches despite identical model and prompt
    let query_b = index.match_prefix(&key_corp_b);
    assert!(query_b.is_empty(), "Cross-project cache leak detected!");
    assert_eq!(query_b.matched_token_count, 0);
    assert_eq!(query_b.uncached_tokens.len(), 48);

    // Insert separate blocks for corp-b
    let inserted_b = index.insert_key(&key_corp_b, &[201, 202, 203]);
    assert_eq!(inserted_b, 3);

    assert_eq!(
        index.match_prefix(&key_corp_a).matched_blocks,
        vec![101, 102, 103]
    );
    assert_eq!(
        index.match_prefix(&key_corp_b).matched_blocks,
        vec![201, 202, 203]
    );

    // Invalidate corp-a only
    let removed_a = index.invalidate_project("corp-a");
    assert_eq!(removed_a, 3);

    assert!(index.match_prefix(&key_corp_a).is_empty());
    assert_eq!(
        index.match_prefix(&key_corp_b).matched_blocks,
        vec![201, 202, 203]
    );
}

#[test]
fn test_us15_sixteen_token_block_alignment_and_uncached_tail() {
    let index = PrefixRadixIndex::new();

    // Insert 2 full blocks (32 tokens)
    let tokens_32 = make_tokens(32, 1);
    let key_base = PrefixKey::new("proj-x", "phi-4", "v1", tokens_32.clone());
    index.insert_key(&key_base, &[50, 51]);

    // Query with 37 tokens (32 matching tokens + 5 trailing unaligned tokens)
    let mut tokens_37 = tokens_32.clone();
    tokens_37.extend_from_slice(&[991, 992, 993, 994, 995]);
    let key_37 = PrefixKey::new("proj-x", "phi-4", "v1", tokens_37);

    let res_37 = index.match_prefix(&key_37);
    assert_eq!(res_37.matched_blocks, vec![50, 51]);
    assert_eq!(res_37.matched_token_count, 32);
    assert_eq!(res_37.uncached_tokens, vec![991, 992, 993, 994, 995]);
    assert_eq!(res_37.uncached_token_count(), 5);

    // Query with 15 tokens (< 16 tokens block boundary) -> cannot match any paged block
    let tokens_15 = make_tokens(15, 1);
    let key_15 = PrefixKey::new("proj-x", "phi-4", "v1", tokens_15.clone());
    let res_15 = index.match_prefix(&key_15);
    assert!(res_15.is_empty());
    assert_eq!(res_15.matched_token_count, 0);
    assert_eq!(res_15.uncached_tokens, tokens_15);

    // Query with 48 tokens: 32 tokens match, next 16 differ from any cached branch
    let mut tokens_48 = tokens_32;
    tokens_48.extend(make_tokens(16, 777));
    let key_48 = PrefixKey::new("proj-x", "phi-4", "v1", tokens_48);
    let res_48 = index.match_prefix(&key_48);
    assert_eq!(res_48.matched_blocks, vec![50, 51]);
    assert_eq!(res_48.matched_token_count, 32);
    assert_eq!(res_48.uncached_tokens.len(), 16);
    assert_eq!(res_48.uncached_tokens, make_tokens(16, 777));
}

#[test]
fn test_us15_cow_refcounting_lifecycle_and_invariants() {
    let mut meta = KvBlockMeta::new(42, 0);

    // Initial state: single owner (ref_count == 1)
    assert_eq!(meta.ref_count(), 1);
    assert!(!meta.is_shared());
    assert!(!meta.needs_cow());
    assert!(!meta.is_evictable());

    // Increment refcount (shared with another slot/session)
    let ref1 = meta.inc_ref();
    assert_eq!(ref1, 2);
    assert_eq!(meta.ref_count(), 2);
    assert!(meta.is_shared());
    assert!(meta.needs_cow());
    assert!(!meta.is_evictable());

    // Increment to 3
    let ref2 = meta.inc_ref();
    assert_eq!(ref2, 3);
    assert_eq!(meta.ref_count(), 3);
    assert!(meta.is_shared());
    assert!(meta.needs_cow());

    // Decrement once -> 2 (still shared)
    let ref3 = meta.dec_ref();
    assert_eq!(ref3, 2);
    assert!(meta.is_shared());
    assert!(meta.needs_cow());
    assert!(!meta.is_evictable());

    // Decrement to 1 (unique owner again)
    let ref4 = meta.dec_ref();
    assert_eq!(ref4, 1);
    assert!(!meta.is_shared());
    assert!(!meta.needs_cow());
    assert!(!meta.is_evictable());

    // Decrement to 0 (freed / evictable)
    let ref5 = meta.dec_ref();
    assert_eq!(ref5, 0);
    assert_eq!(meta.ref_count(), 0);
    assert!(!meta.is_shared());
    assert!(!meta.needs_cow());
    assert!(meta.is_evictable());

    // Underflow protection: decrementing at 0 saturates at 0
    let ref_underflow = meta.dec_ref();
    assert_eq!(ref_underflow, 0);
    assert_eq!(meta.ref_count(), 0);
    assert!(meta.is_evictable());

    // Overflow protection: saturating arithmetic at u16::MAX
    meta.refcount = u16::MAX - 1;
    assert_eq!(meta.inc_ref(), u16::MAX);
    assert_eq!(meta.inc_ref(), u16::MAX);
    assert_eq!(meta.ref_count(), u16::MAX);
}

#[test]
fn test_us15_cow_paged_block_sharing_simulation() {
    // Simulate fork / branch with Copy-on-Write semantics
    // Request 1 has blocks [b0, b1, b2, b3]
    let mut block_table_req1 = vec![
        KvBlockMeta::new(0, 0),
        KvBlockMeta::new(1, 0),
        KvBlockMeta::new(2, 0),
        KvBlockMeta::new(3, 0),
    ];

    // Request 2 shares prompt prefix (blocks 0 and 1)
    for block in block_table_req1.iter_mut().take(2) {
        block.inc_ref();
    }

    assert!(block_table_req1[0].needs_cow());
    assert!(block_table_req1[1].needs_cow());
    assert!(!block_table_req1[2].needs_cow());
    assert!(!block_table_req1[3].needs_cow());

    // When Request 2 attempts to append new tokens into block 1:
    // It detects needs_cow() == true, so it must duplicate block 1 into a new block 4
    let mut block4 = KvBlockMeta::new(4, 0);
    // Old block 1 is decremented
    block_table_req1[1].dec_ref();
    // Block 1 is now no longer shared
    assert!(!block_table_req1[1].needs_cow());
    // New block 4 is uniquely owned
    assert_eq!(block4.ref_count(), 1);
    assert!(!block4.needs_cow());

    // When Request 1 completes, decrement its blocks
    for block in &mut block_table_req1 {
        block.dec_ref();
    }
    // Block 0 was shared with Request 2, so its refcount dropped from 2 to 1 (not yet evictable)
    assert_eq!(block_table_req1[0].ref_count(), 1);
    assert!(!block_table_req1[0].is_evictable());

    // Blocks 1, 2, 3 were private to Request 1, so their refcount dropped to 0 (evictable)
    assert!(block_table_req1[1].is_evictable());
    assert!(block_table_req1[2].is_evictable());
    assert!(block_table_req1[3].is_evictable());

    // When Request 2 finishes with block 0 and block 4:
    block_table_req1[0].dec_ref();
    block4.dec_ref();
    assert!(block_table_req1[0].is_evictable());
    assert!(block4.is_evictable());
}
