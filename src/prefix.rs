//! Control-plane token-ID prefix radix index & block lookup (US-06.2.2, ADR-004, `features/kv-cache.md`).
//!
//! Provides a concurrent, non-blocking-read prefix radix index keyed by `(project_id, model_id, token_ids)`
//! aligned to 16-token KV cache blocks (`BLOCK_SIZE_TOKENS = 16`). Supports longest-prefix match queries,
//! block insertion, LRU eviction, and per-project/per-model invalidation without blocking on compute workers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock};

/// Number of token IDs per KV cache paged block (ADR-004, `features/kv-cache.md`).
pub const BLOCK_SIZE_TOKENS: usize = 16;

/// 16-token array representation of a single KV cache block.
pub type BlockTokens = [u32; BLOCK_SIZE_TOKENS];

static GLOBAL_ACCESS_TICK: AtomicU64 = AtomicU64::new(1);

/// Longest-prefix match result returning cached physical block IDs and uncached suffix tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatchResult {
    /// Ordered list of matched physical KV cache block indices.
    pub matched_blocks: Vec<u32>,
    /// Remaining trailing prompt tokens that are not yet cached in the prefix index.
    pub uncached_tokens: Vec<u32>,
    /// Total number of tokens matched in the prefix (`matched_blocks.len() * 16`).
    pub matched_token_count: usize,
}

impl PrefixMatchResult {
    /// Returns `true` if no full 16-token blocks were matched.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.matched_blocks.is_empty()
    }

    /// Number of cached 16-token blocks matched.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.matched_blocks.len()
    }

    /// Number of remaining uncached suffix tokens.
    #[must_use]
    pub fn uncached_token_count(&self) -> usize {
        self.uncached_tokens.len()
    }
}

/// Statistics snapshot of the control-plane prefix radix index.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PrefixCacheStats {
    /// Number of distinct projects currently holding entries in the index.
    pub total_projects: usize,
    /// Number of distinct models currently registered.
    pub total_models: usize,
    /// Total number of 16-token KV cache blocks indexed across all models.
    pub total_cached_blocks: usize,
    /// Total number of prefix match queries performed.
    pub lookup_count: u64,
    /// Total number of queries that matched at least one 16-token block.
    pub hit_count: u64,
    /// Total number of queries with zero matched blocks.
    pub miss_count: u64,
}

impl PrefixCacheStats {
    /// Returns the cache hit ratio as a fraction in `[0.0, 1.0]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        if self.lookup_count == 0 {
            0.0
        } else {
            self.hit_count as f64 / self.lookup_count as f64
        }
    }
}

#[derive(Debug)]
struct RadixTreeNode {
    block_id: u32,
    last_accessed_tick: AtomicU64,
    children: HashMap<BlockTokens, Box<RadixTreeNode>>,
}

impl RadixTreeNode {
    fn new(block_id: u32) -> Self {
        Self {
            block_id,
            last_accessed_tick: AtomicU64::new(GLOBAL_ACCESS_TICK.fetch_add(1, Ordering::Relaxed)),
            children: HashMap::new(),
        }
    }

    fn count_subtree_blocks(&self) -> usize {
        1 + self
            .children
            .values()
            .map(|c| c.count_subtree_blocks())
            .sum::<usize>()
    }
}

#[derive(Debug, Default)]
struct ModelPrefixTree {
    root_children: HashMap<BlockTokens, Box<RadixTreeNode>>,
    total_blocks: usize,
}

impl ModelPrefixTree {
    fn match_longest_prefix(&self, token_ids: &[u32]) -> PrefixMatchResult {
        let full_blocks = token_ids.len() / BLOCK_SIZE_TOKENS;
        let mut matched_blocks = Vec::with_capacity(full_blocks);

        let mut curr_children = &self.root_children;
        for i in 0..full_blocks {
            let offset = i * BLOCK_SIZE_TOKENS;
            let mut chunk = [0_u32; BLOCK_SIZE_TOKENS];
            chunk.copy_from_slice(&token_ids[offset..offset + BLOCK_SIZE_TOKENS]);

            if let Some(child) = curr_children.get(&chunk) {
                child.last_accessed_tick.store(
                    GLOBAL_ACCESS_TICK.fetch_add(1, Ordering::Relaxed),
                    Ordering::Relaxed,
                );
                matched_blocks.push(child.block_id);
                curr_children = &child.children;
            } else {
                break;
            }
        }

        let matched_token_count = matched_blocks.len() * BLOCK_SIZE_TOKENS;
        let uncached_tokens = token_ids[matched_token_count..].to_vec();

        PrefixMatchResult {
            matched_blocks,
            uncached_tokens,
            matched_token_count,
        }
    }

    fn insert(&mut self, token_ids: &[u32], block_ids: &[u32]) -> usize {
        let full_blocks = token_ids.len() / BLOCK_SIZE_TOKENS;
        let to_insert = full_blocks.min(block_ids.len());
        if to_insert == 0 {
            return 0;
        }

        let mut curr_children = &mut self.root_children;
        let mut newly_added = 0;

        for (i, &target_block_id) in block_ids.iter().take(to_insert).enumerate() {
            let offset = i * BLOCK_SIZE_TOKENS;
            let mut chunk = [0_u32; BLOCK_SIZE_TOKENS];
            chunk.copy_from_slice(&token_ids[offset..offset + BLOCK_SIZE_TOKENS]);

            let entry = curr_children.entry(chunk).or_insert_with(|| {
                newly_added += 1;
                Box::new(RadixTreeNode::new(target_block_id))
            });

            entry.block_id = target_block_id;
            entry.last_accessed_tick.store(
                GLOBAL_ACCESS_TICK.fetch_add(1, Ordering::Relaxed),
                Ordering::Relaxed,
            );
            curr_children = &mut entry.children;
        }

        self.total_blocks += newly_added;
        to_insert
    }

    fn invalidate_block_recursive(
        children: &mut HashMap<BlockTokens, Box<RadixTreeNode>>,
        target_block_id: u32,
    ) -> usize {
        let mut removed_count = 0;
        let mut keys_to_remove = Vec::new();

        for (k, node) in children.iter_mut() {
            if node.block_id == target_block_id {
                keys_to_remove.push(*k);
            } else {
                removed_count +=
                    Self::invalidate_block_recursive(&mut node.children, target_block_id);
            }
        }

        for k in keys_to_remove {
            if let Some(removed_node) = children.remove(&k) {
                removed_count += removed_node.count_subtree_blocks();
            }
        }

        removed_count
    }

    fn invalidate_block(&mut self, block_id: u32) -> bool {
        let removed = Self::invalidate_block_recursive(&mut self.root_children, block_id);
        self.total_blocks = self.total_blocks.saturating_sub(removed);
        removed > 0
    }

    fn collect_leaves(
        children: &HashMap<BlockTokens, Box<RadixTreeNode>>,
        leaves: &mut Vec<(u64, u32)>,
    ) {
        for node in children.values() {
            if node.children.is_empty() {
                leaves.push((
                    node.last_accessed_tick.load(Ordering::Relaxed),
                    node.block_id,
                ));
            } else {
                Self::collect_leaves(&node.children, leaves);
            }
        }
    }

    fn evict_lru(&mut self, max_blocks: usize) -> Vec<u32> {
        let mut evicted = Vec::new();
        while evicted.len() < max_blocks {
            let mut leaves = Vec::new();
            Self::collect_leaves(&self.root_children, &mut leaves);
            if leaves.is_empty() {
                break;
            }
            // Strictly select the oldest leaf currently in the tree
            if let Some((_, oldest_block_id)) = leaves.iter().min_by_key(|(tick, _)| *tick) {
                let id = *oldest_block_id;
                if self.invalidate_block(id) {
                    evicted.push(id);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        evicted
    }
}

type PrefixTreeMap = HashMap<String, HashMap<String, Arc<RwLock<ModelPrefixTree>>>>;

/// Concurrent control-plane prefix radix index keyed by `(project_id, model_id, token_ids)`.
#[derive(Debug, Default)]
pub struct PrefixRadixIndex {
    trees: RwLock<PrefixTreeMap>,
    lookup_count: AtomicU64,
    hit_count: AtomicU64,
    miss_count: AtomicU64,
}

impl PrefixRadixIndex {
    /// Creates a new, empty [`PrefixRadixIndex`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Performs a non-blocking longest-prefix match query.
    ///
    /// Matches the longest sequence of complete 16-token blocks (`BLOCK_SIZE_TOKENS = 16`)
    /// and returns both the cached block IDs and remaining uncached suffix tokens.
    #[must_use]
    pub fn match_longest_prefix(
        &self,
        project_id: &str,
        model_id: &str,
        token_ids: &[u32],
    ) -> PrefixMatchResult {
        self.lookup_count.fetch_add(1, Ordering::Relaxed);

        let tree_arc = {
            let map = self.trees.read().unwrap_or_else(PoisonError::into_inner);
            map.get(project_id)
                .and_then(|models| models.get(model_id))
                .cloned()
        };

        let result = if let Some(tree) = tree_arc {
            let tree_guard = tree.read().unwrap_or_else(PoisonError::into_inner);
            tree_guard.match_longest_prefix(token_ids)
        } else {
            PrefixMatchResult {
                matched_blocks: Vec::new(),
                uncached_tokens: token_ids.to_vec(),
                matched_token_count: 0,
            }
        };

        if result.is_empty() {
            self.miss_count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.hit_count.fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    /// Inserts 16-token aligned KV cache blocks into the radix index for `(project_id, model_id)`.
    ///
    /// Returns the number of 16-token blocks successfully registered.
    pub fn insert(
        &self,
        project_id: &str,
        model_id: &str,
        token_ids: &[u32],
        block_ids: &[u32],
    ) -> usize {
        let tree_arc = {
            let map = self.trees.read().unwrap_or_else(PoisonError::into_inner);
            map.get(project_id)
                .and_then(|models| models.get(model_id))
                .cloned()
        };
        let tree_arc = if let Some(arc) = tree_arc {
            arc
        } else {
            let mut map = self.trees.write().unwrap_or_else(PoisonError::into_inner);
            map.entry(project_id.to_owned())
                .or_default()
                .entry(model_id.to_owned())
                .or_insert_with(|| Arc::new(RwLock::new(ModelPrefixTree::default())))
                .clone()
        };

        let mut tree = tree_arc.write().unwrap_or_else(PoisonError::into_inner);
        tree.insert(token_ids, block_ids)
    }

    /// Invalidates all cached prefix blocks for a specific model within a project.
    ///
    /// Returns the number of blocks invalidated.
    pub fn invalidate_model(&self, project_id: &str, model_id: &str) -> usize {
        let removed = {
            let mut map = self.trees.write().unwrap_or_else(PoisonError::into_inner);
            let rem = map
                .get_mut(project_id)
                .and_then(|models| models.remove(model_id));
            if let Some(models) = map.get(project_id) {
                if models.is_empty() {
                    map.remove(project_id);
                }
            }
            rem
        };

        if let Some(tree) = removed {
            let guard = tree.read().unwrap_or_else(PoisonError::into_inner);
            guard.total_blocks
        } else {
            0
        }
    }

    /// Invalidates all cached prefix blocks for a model across all registered projects.
    ///
    /// Returns the total number of blocks invalidated.
    pub fn invalidate_model_all_projects(&self, model_id: &str) -> usize {
        let mut map = self.trees.write().unwrap_or_else(PoisonError::into_inner);
        let mut total_invalidated = 0;
        for models in map.values_mut() {
            if let Some(tree) = models.remove(model_id) {
                let guard = tree.read().unwrap_or_else(PoisonError::into_inner);
                total_invalidated += guard.total_blocks;
            }
        }
        map.retain(|_, models| !models.is_empty());
        total_invalidated
    }

    /// Invalidates all cached prefix blocks across all models for a specific project.
    ///
    /// Returns the total number of blocks invalidated.
    pub fn invalidate_project(&self, project_id: &str) -> usize {
        let removed = {
            let mut map = self.trees.write().unwrap_or_else(PoisonError::into_inner);
            map.remove(project_id)
        };

        if let Some(models) = removed {
            let mut total_invalidated = 0;
            for tree in models.values() {
                let guard = tree.read().unwrap_or_else(PoisonError::into_inner);
                total_invalidated += guard.total_blocks;
            }
            total_invalidated
        } else {
            0
        }
    }

    /// Invalidates a specific block ID from a `(project_id, model_id)` tree.
    ///
    /// Returns `true` if the block was found and invalidated.
    pub fn invalidate_block(&self, project_id: &str, model_id: &str, block_id: u32) -> bool {
        let tree_arc = {
            let map = self.trees.read().unwrap_or_else(PoisonError::into_inner);
            map.get(project_id)
                .and_then(|models| models.get(model_id))
                .cloned()
        };

        if let Some(tree) = tree_arc {
            let mut guard = tree.write().unwrap_or_else(PoisonError::into_inner);
            guard.invalidate_block(block_id)
        } else {
            false
        }
    }

    /// Evicts up to `max_blocks` least-recently-used leaf blocks for `(project_id, model_id)`.
    ///
    /// Returns the physical block IDs that were evicted.
    pub fn evict_lru(&self, project_id: &str, model_id: &str, max_blocks: usize) -> Vec<u32> {
        let tree_arc = {
            let map = self.trees.read().unwrap_or_else(PoisonError::into_inner);
            map.get(project_id)
                .and_then(|models| models.get(model_id))
                .cloned()
        };

        if let Some(tree) = tree_arc {
            let mut guard = tree.write().unwrap_or_else(PoisonError::into_inner);
            guard.evict_lru(max_blocks)
        } else {
            Vec::new()
        }
    }

    /// Returns the current statistics snapshot of the prefix radix index.
    #[must_use]
    pub fn stats(&self) -> PrefixCacheStats {
        let map = self.trees.read().unwrap_or_else(PoisonError::into_inner);
        let mut total_projects = 0;
        let mut models_set = std::collections::HashSet::new();
        let mut total_blocks = 0;

        for models in map.values() {
            if !models.is_empty() {
                total_projects += 1;
            }
            for (model, tree) in models {
                models_set.insert(model.clone());
                let guard = tree.read().unwrap_or_else(PoisonError::into_inner);
                total_blocks += guard.total_blocks;
            }
        }

        PrefixCacheStats {
            total_projects,
            total_models: models_set.len(),
            total_cached_blocks: total_blocks,
            lookup_count: self.lookup_count.load(Ordering::Relaxed),
            hit_count: self.hit_count.load(Ordering::Relaxed),
            miss_count: self.miss_count.load(Ordering::Relaxed),
        }
    }

    /// Clears all prefix trees across all projects and models.
    pub fn clear(&self) {
        let mut map = self.trees.write().unwrap_or_else(PoisonError::into_inner);
        map.clear();
    }
}

#[cfg(test)]
#[allow(clippy::assert_is_empty)]
mod tests {
    use super::*;

    fn generate_block_tokens(start: u32) -> Vec<u32> {
        (start..start + 16).collect()
    }

    #[test]
    fn test_exact_16_token_block_matching() {
        let index = PrefixRadixIndex::new();
        let block0 = generate_block_tokens(100);
        let block1 = generate_block_tokens(200);

        let mut full_prompt = Vec::new();
        full_prompt.extend_from_slice(&block0);
        full_prompt.extend_from_slice(&block1);

        // Insert 2 blocks
        let inserted = index.insert("proj-1", "model-x", &full_prompt, &[10, 11]);
        assert_eq!(inserted, 2);

        // Query with exact 2 blocks (32 tokens)
        let res = index.match_longest_prefix("proj-1", "model-x", &full_prompt);
        assert_eq!(res.matched_blocks, vec![10, 11]);
        assert_eq!(res.matched_token_count, 32);
        assert!(res.uncached_tokens.is_empty());
        assert!(!res.is_empty());
        assert_eq!(res.block_count(), 2);
    }

    #[test]
    fn test_partial_16_token_block_matching() {
        let index = PrefixRadixIndex::new();
        let block0 = generate_block_tokens(100);
        let block1 = generate_block_tokens(200);

        let mut full_prompt = Vec::new();
        full_prompt.extend_from_slice(&block0);
        full_prompt.extend_from_slice(&block1);
        index.insert("proj-1", "model-x", &full_prompt, &[10, 11]);

        // 1. Query with partial unaligned tokens (e.g. 25 tokens = 1 block + 9 partial)
        let mut partial_query = block0.clone();
        partial_query.extend_from_slice(&[999, 998, 997, 996, 995, 994, 993, 992, 991]);
        assert_eq!(partial_query.len(), 25);

        let res1 = index.match_longest_prefix("proj-1", "model-x", &partial_query);
        assert_eq!(res1.matched_blocks, vec![10]);
        assert_eq!(res1.matched_token_count, 16);
        assert_eq!(
            res1.uncached_tokens,
            vec![999, 998, 997, 996, 995, 994, 993, 992, 991]
        );
        assert_eq!(res1.uncached_token_count(), 9);

        // 2. Query with fewer than 16 tokens (e.g. 8 tokens)
        let short_query = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let res2 = index.match_longest_prefix("proj-1", "model-x", &short_query);
        assert!(res2.is_empty());
        assert_eq!(res2.matched_token_count, 0);
        assert_eq!(res2.uncached_tokens, short_query);
    }

    #[test]
    fn test_project_and_model_key_isolation() {
        let index = PrefixRadixIndex::new();
        let block0 = generate_block_tokens(100);

        // Insert into (proj-A, model-1)
        index.insert("proj-A", "model-1", &block0, &[42]);

        // Same tokens in (proj-A, model-1) matches
        let res_hit = index.match_longest_prefix("proj-A", "model-1", &block0);
        assert_eq!(res_hit.matched_blocks, vec![42]);

        // Cross-project query (proj-B, model-1) must miss (ADR-004)
        let res_cross_proj = index.match_longest_prefix("proj-B", "model-1", &block0);
        assert!(res_cross_proj.is_empty());
        assert_eq!(res_cross_proj.uncached_tokens, block0);

        // Cross-model query (proj-A, model-2) must miss
        let res_cross_model = index.match_longest_prefix("proj-A", "model-2", &block0);
        assert!(res_cross_model.is_empty());
    }

    #[test]
    fn test_block_invalidation_and_tree_pruning() {
        let index = PrefixRadixIndex::new();
        let b0 = generate_block_tokens(10);
        let b1 = generate_block_tokens(20);
        let b2 = generate_block_tokens(30);

        let mut seq = Vec::new();
        seq.extend_from_slice(&b0);
        seq.extend_from_slice(&b1);
        seq.extend_from_slice(&b2);

        index.insert("p1", "m1", &seq, &[100, 101, 102]);
        assert_eq!(index.stats().total_cached_blocks, 3);

        // Invalidate specific block 101 (child of 100, parent of 102)
        // Subtree containing 101 and 102 is pruned
        assert!(index.invalidate_block("p1", "m1", 101));

        let res = index.match_longest_prefix("p1", "m1", &seq);
        assert_eq!(res.matched_blocks, vec![100]);
        assert_eq!(res.matched_token_count, 16);
        assert_eq!(res.uncached_tokens.len(), 32);

        // Invalidate model
        let cleared = index.invalidate_model("p1", "m1");
        assert_eq!(cleared, 1);
        assert!(index.match_longest_prefix("p1", "m1", &seq).is_empty());

        // Invalidate project
        index.insert("p2", "m1", &b0, &[200]);
        index.insert("p2", "m2", &b0, &[201]);
        assert_eq!(index.invalidate_project("p2"), 2);
        assert!(index.match_longest_prefix("p2", "m1", &b0).is_empty());
        assert!(index.match_longest_prefix("p2", "m2", &b0).is_empty());
    }

    #[test]
    fn test_lru_eviction() {
        let index = PrefixRadixIndex::new();
        let b0 = generate_block_tokens(10);
        let b1 = generate_block_tokens(20);

        let mut seq = Vec::new();
        seq.extend_from_slice(&b0);
        seq.extend_from_slice(&b1);

        index.insert("p1", "m1", &seq, &[10, 20]);
        assert_eq!(index.stats().total_cached_blocks, 2);

        // Evict LRU leaf block (block 20)
        let evicted = index.evict_lru("p1", "m1", 1);
        assert_eq!(evicted, vec![20]);

        // Block 10 remains
        let res = index.match_longest_prefix("p1", "m1", &seq);
        assert_eq!(res.matched_blocks, vec![10]);
    }

    #[test]
    fn test_prefix_cache_statistics_and_hit_rate() {
        let index = PrefixRadixIndex::new();
        let b0 = generate_block_tokens(10);

        index.insert("p1", "m1", &b0, &[1]);

        let stats_init = index.stats();
        assert_eq!(stats_init.total_projects, 1);
        assert_eq!(stats_init.total_models, 1);
        assert_eq!(stats_init.total_cached_blocks, 1);
        assert_eq!(stats_init.lookup_count, 0);

        // 1 Hit
        let _ = index.match_longest_prefix("p1", "m1", &b0);
        // 1 Miss
        let _ = index.match_longest_prefix("p1", "m1", &[1, 2, 3]);

        let stats = index.stats();
        assert_eq!(stats.lookup_count, 2);
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
        assert!((stats.hit_rate() - 0.5).abs() < f64::EPSILON);
    }
}
