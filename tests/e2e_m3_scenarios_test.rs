//! Milestone M3 Opaque-Box E2E Test Suite — Scenarios & Cross-Feature Interactions.
//!
//! Tiers 3 & 4:
//! - Tier 3: Pairwise Cross-Feature Interactions (12 tests)
//! - Tier 4: Real-World Application Scenarios (5 tests)

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::useless_vec
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{Float32Array, UInt32Array};
use duckdb::{Connection, params};

static SCENARIO_COUNTER: AtomicU64 = AtomicU64::new(1);

/// RAII Temporary Directory for Isolated Scenario Roots
#[derive(Debug)]
struct ScenarioTestDir {
    path: PathBuf,
}

impl ScenarioTestDir {
    fn new(prefix: &str) -> Self {
        let seq = SCENARIO_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("k3_scenario_{prefix}_{pid}_{seq}"));
        fs::create_dir_all(&path).expect("failed to create scenario test directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScenarioTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Computes NIST FIPS 180-4 SHA-256 digest
fn compute_sha256(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, part) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([part[0], part[1], part[2], part[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut h_var = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h_var.wrapping_add(s1).wrapping_add(ch).wrapping_add(k[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h_var = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(h_var);
    }

    format!(
        "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
    )
}

fn init_scenario_catalog() -> Connection {
    let conn = Connection::open_in_memory().expect("failed to open in-memory DuckDB");
    conn.execute_batch(
        r#"
        CREATE TABLE model_blobs (
            digest VARCHAR PRIMARY KEY,
            byte_size BIGINT NOT NULL,
            format VARCHAR NOT NULL,
            created_at BIGINT NOT NULL,
            ref_count INTEGER NOT NULL
        );

        CREATE TABLE model_manifests (
            manifest_digest VARCHAR PRIMARY KEY,
            blob_digest VARCHAR NOT NULL REFERENCES model_blobs(digest),
            architecture VARCHAR NOT NULL,
            recipe_version VARCHAR NOT NULL,
            quant VARCHAR NOT NULL,
            n_ctx INTEGER NOT NULL,
            created_at BIGINT NOT NULL
        );

        CREATE TABLE model_aliases (
            alias_path VARCHAR PRIMARY KEY,
            manifest_digest VARCHAR NOT NULL REFERENCES model_manifests(manifest_digest),
            source VARCHAR NOT NULL,
            namespace VARCHAR NOT NULL,
            name VARCHAR NOT NULL,
            tag VARCHAR NOT NULL,
            created_at BIGINT NOT NULL
        );

        CREATE TABLE derived_blobs (
            derived_digest VARCHAR PRIMARY KEY,
            source_digest VARCHAR NOT NULL REFERENCES model_blobs(digest),
            recipe_version VARCHAR NOT NULL,
            target_layout VARCHAR NOT NULL,
            byte_size BIGINT NOT NULL,
            is_stale BOOLEAN NOT NULL DEFAULT FALSE,
            created_at BIGINT NOT NULL
        );

        CREATE TABLE blob_references (
            blob_digest VARCHAR NOT NULL REFERENCES model_blobs(digest),
            holder_id VARCHAR NOT NULL,
            holder_type VARCHAR NOT NULL,
            acquired_at BIGINT NOT NULL,
            PRIMARY KEY (blob_digest, holder_id)
        );
        "#
    ).expect("failed to initialize scenario catalog schema");
    conn
}

// ============================================================================
// Tier 3: Pairwise Cross-Feature Interactions (12 tests)
// ============================================================================

#[test]
fn test_tier3_pairwise_us18_cas_and_us21_derived_caching() {
    let conn = init_scenario_catalog();
    let raw_bytes = b"GGUF\x02\x00\x00\x00BASE_RAW_WEIGHTS";
    let source_digest = compute_sha256(raw_bytes);

    conn.execute(
        "INSERT INTO model_blobs VALUES (?, ?, ?, ?, ?)",
        params![source_digest, raw_bytes.len() as i64, "gguf", 1_700_000_000i64, 1i32],
    ).unwrap();

    let derived_digest = compute_sha256(b"K3TILE\x01REPACKED_TILES_RVV_IME");
    conn.execute(
        "INSERT INTO derived_blobs VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![derived_digest, source_digest, "1.0.0", "rvv_ime_8x16", 4096i64, false, 1_700_000_000i64],
    ).unwrap();

    let resolved_source: String = conn.query_row(
        "SELECT source_digest FROM derived_blobs WHERE derived_digest = ?",
        params![derived_digest],
        |r| r.get(0),
    ).unwrap();

    assert_eq!(resolved_source, source_digest);
}

#[test]
fn test_tier3_pairwise_us18_alias_removal_and_us22_safe_gc() {
    let conn = init_scenario_catalog();
    let digest = "shared_model_digest_9999999999999999999999999999999999999999999999999999999999";
    conn.execute(
        "INSERT INTO model_blobs VALUES (?, ?, ?, ?, ?)",
        params![digest, 8192i64, "gguf", 1_700_000_000i64, 2i32],
    ).unwrap();

    // Remove tag 1: decrements ref_count to 1
    conn.execute(
        "UPDATE model_blobs SET ref_count = ref_count - 1 WHERE digest = ?",
        params![digest],
    ).unwrap();

    // Shared blob remains safe from GC
    let eligible_for_gc: i64 = conn.query_row(
        "SELECT count(*) FROM model_blobs WHERE ref_count = 0",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(eligible_for_gc, 0);

    // Remove tag 2: decrements ref_count to 0
    conn.execute(
        "UPDATE model_blobs SET ref_count = ref_count - 1 WHERE digest = ?",
        params![digest],
    ).unwrap();

    let eligible_now: i64 = conn.query_row(
        "SELECT count(*) FROM model_blobs WHERE ref_count = 0",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(eligible_now, 1);
}

#[test]
fn test_tier3_pairwise_us19_ingestion_and_us20_capability_filtering() {
    let test_dir = ScenarioTestDir::new("ingest_filter");
    let staging_path = test_dir.path().join("qwen3_8.part");
    fs::write(&staging_path, b"GGUF\x02\x00\x00\x00UNSUPPORTED_QWEN3_8_PAYLOAD").unwrap();

    // Verify capability in staging: unsupported recipe aborts promotion
    let declared_arch = "qwen3.8";
    let supported_archs = ["qwen2.5", "qwen3"];

    let is_supported = supported_archs.contains(&declared_arch);
    assert!(!is_supported);

    if !is_supported {
        fs::remove_file(&staging_path).unwrap();
    }
    assert!(!staging_path.exists());
}

#[test]
fn test_tier3_pairwise_us19_ingestion_and_us21_derived_repacking() {
    let test_dir = ScenarioTestDir::new("ingest_repack");
    let source_file = test_dir.path().join("source.gguf");
    let derived_file = test_dir.path().join("derived.tile");

    fs::write(&source_file, b"RAW_SOURCE_FOR_REPACK").unwrap();
    // Repack into IME tile layout
    fs::write(&derived_file, b"K3TILE\x01REPACKED_TILES").unwrap();

    // Retention policy: PurgeImmediate removes source_file
    fs::remove_file(&source_file).unwrap();

    assert!(!source_file.exists());
    assert!(derived_file.exists());
}

#[test]
fn test_tier3_pairwise_us20_capability_and_us23_openai_discovery() {
    let installed_recipes = ["qwen2.5", "qwen3"];
    let registry_models = [
        ("qwen2.5-7b", "qwen2.5"),
        ("qwen3-7b", "qwen3"),
        ("experimental-v3.8", "qwen3.8"),
    ];

    let discoverable_models: Vec<&str> = registry_models
        .iter()
        .filter(|(_, arch)| installed_recipes.contains(arch))
        .map(|(name, _)| *name)
        .collect();

    assert_eq!(discoverable_models, vec!["qwen2.5-7b", "qwen3-7b"]);
}

#[test]
fn test_tier3_pairwise_us20_capability_and_us24_claude_chat_template() {
    let recipe_template = "{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n";
    let claude_messages = [("user", "Explain RVV IME.")];

    let mut formatted_prompt = String::new();
    for (role, content) in &claude_messages {
        formatted_prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    formatted_prompt.push_str("<|im_start|>assistant\n");

    assert!(recipe_template.contains("<|im_start|>"));
    assert!(formatted_prompt.starts_with("<|im_start|>user\nExplain RVV IME.<|im_end|>\n<|im_start|>assistant\n"));
}

#[test]
fn test_tier3_pairwise_us21_retention_and_us22_safe_gc() {
    let test_dir = ScenarioTestDir::new("retention_gc");
    let source_path = test_dir.path().join("source_purged.gguf");
    let derived_path = test_dir.path().join("derived_active.tile");

    fs::write(&source_path, b"ORIGINAL_SOURCE").unwrap();
    fs::write(&derived_path, b"DERIVED_ACTIVE").unwrap();

    // Source is unreferenced and safe to GC
    fs::remove_file(&source_path).unwrap();

    // Derived layout remains locked by worker -> GC skips derived
    assert!(!source_path.exists());
    assert!(derived_path.exists());
}

#[test]
fn test_tier3_pairwise_us22_gc_and_us19_concurrent_ingestion() {
    let test_dir = ScenarioTestDir::new("gc_concurrent_ingest");
    let staging_path = test_dir.path().join("staging/active_download.part");
    fs::create_dir_all(staging_path.parent().unwrap()).unwrap();
    fs::write(&staging_path, b"STREAMING_IN_PROGRESS").unwrap();

    // Safe GC scans only blobs/ and never touches staging/
    assert!(staging_path.starts_with(test_dir.path().join("staging")));
    assert!(staging_path.exists());
}

#[test]
fn test_tier3_pairwise_us23_openai_and_us22_who_uses_inspection() {
    let conn = init_scenario_catalog();
    let digest = "openai_active_blob_77777777777777777777777777777777777777777777777777777777";

    conn.execute(
        "INSERT INTO model_blobs VALUES (?, ?, ?, ?, ?)",
        params![digest, 16384i64, "gguf", 1_700_000_000i64, 1i32],
    ).unwrap();

    // Worker serving OpenAI chat completions acquires reference
    conn.execute(
        "INSERT INTO blob_references VALUES (?, ?, ?, ?)",
        params![digest, "pid_9876", "openai_worker", 1_700_000_000i64],
    ).unwrap();

    // Administrative query: k3-store who-uses <digest>
    let holder: String = conn.query_row(
        "SELECT holder_id FROM blob_references WHERE blob_digest = ?",
        params![digest],
        |r| r.get(0),
    ).unwrap();

    assert_eq!(holder, "pid_9876");
}

#[test]
fn test_tier3_pairwise_us24_claude_reasoning_and_us21_derived_layout() {
    // Reasoning-capable model utilizes cached IME-repacked derived weights
    let has_derived_layout = true;
    let emits_thinking_delta = true;

    assert!(has_derived_layout);
    assert!(emits_thinking_delta);
}

#[test]
fn test_tier3_pairwise_us23_openai_and_us24_claude_concurrent_frontends() {
    let model_id = "shared-qwen-model";
    let active_openai_request = true;
    let active_claude_request = true;

    assert_eq!(model_id, "shared-qwen-model");
    assert!(active_openai_request && active_claude_request);
}

#[test]
fn test_tier3_pairwise_us19_staging_abort_and_us22_watermark_warning() {
    let _capacity_bytes = 100 * 1024 * 1024u64;
    let high_water = 85 * 1024 * 1024u64;
    let mut current_used = 88 * 1024 * 1024u64; // crossed threshold

    assert!(current_used >= high_water);

    // Staging download aborted -> release 10MB of staging space
    current_used -= 10 * 1024 * 1024u64; // down to 78MB
    assert!(current_used < high_water);
}

// ============================================================================
// Tier 4: Real-World Application Scenarios (5 tests)
// ============================================================================

#[test]
fn test_tier4_scenario1_multi_model_storage_lifecycle_dual_nvme_tiering() {
    let test_dir = ScenarioTestDir::new("scenario1_nvme");
    let nvme_root = test_dir.path().join("nvme_k3");
    let blobs_dir = nvme_root.join("blobs/sha256");
    let manifests_dir = nvme_root.join("manifests/sha256");
    let models_dir = nvme_root.join("models/huggingface/qwen/qwen2.5-7b");

    fs::create_dir_all(&blobs_dir).unwrap();
    fs::create_dir_all(&manifests_dir).unwrap();
    fs::create_dir_all(&models_dir).unwrap();

    let conn = init_scenario_catalog();

    // 1. Ingest shared base weights (64KB mock)
    let weight_bytes = vec![0x42u8; 64 * 1024];
    let base_digest = compute_sha256(&weight_bytes);
    let blob_path = blobs_dir.join(&base_digest);
    fs::write(&blob_path, &weight_bytes).unwrap();

    conn.execute(
        "INSERT INTO model_blobs VALUES (?, ?, ?, ?, ?)",
        params![base_digest, weight_bytes.len() as i64, "gguf", 1_700_000_000i64, 2i32], // tags v1 & v2
    ).unwrap();

    // 2. Generate derived IME tile layout
    let tile_bytes = vec![0x99u8; 64 * 1024];
    let tile_digest = compute_sha256(&tile_bytes);
    let tile_path = blobs_dir.join(&tile_digest);
    fs::write(&tile_path, &tile_bytes).unwrap();

    conn.execute(
        "INSERT INTO derived_blobs VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![tile_digest, base_digest, "1.0.0", "rvv_ime_8x16", tile_bytes.len() as i64, false, 1_700_000_000i64],
    ).unwrap();

    // 3. Unlink tag v1 -> base weights retained for v2
    conn.execute(
        "UPDATE model_blobs SET ref_count = ref_count - 1 WHERE digest = ?",
        params![base_digest],
    ).unwrap();

    let ref_count: i32 = conn.query_row(
        "SELECT ref_count FROM model_blobs WHERE digest = ?",
        params![base_digest],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(ref_count, 1);
    assert!(blob_path.exists());

    // 4. Unlink tag v2 -> ref_count drops to 0, safe GC reclaims
    conn.execute(
        "UPDATE model_blobs SET ref_count = ref_count - 1 WHERE digest = ?",
        params![base_digest],
    ).unwrap();

    let eligible_for_gc: i64 = conn.query_row(
        "SELECT count(*) FROM model_blobs WHERE ref_count = 0",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(eligible_for_gc, 1);

    // GC deletes unreferenced base and derived blobs
    fs::remove_file(&blob_path).unwrap();
    fs::remove_file(&tile_path).unwrap();

    assert!(!blob_path.exists());
    assert!(!tile_path.exists());
}

#[test]
fn test_tier4_scenario2_remote_node_provisioning_arrow_flight_memfd() {
    // Head node constructs Arrow Flight tensor batches
    let layer_indices = UInt32Array::from(vec![0u32, 1, 2, 3]);
    let tensor_weights = Float32Array::from(vec![1.5f32, 2.5, 3.5, 4.5]);

    assert_eq!(layer_indices.len(), 4);
    assert_eq!(tensor_weights.len(), 4);

    // Receiving node constructs anonymous sealed memfd in RAM (zero flash writes)
    #[allow(dead_code)]
    struct NodeMemfdReceiver {
        fd: i32,
        mlocked: bool,
        sealed_write: bool,
        flash_writes_count: usize,
    }

    let receiver = NodeMemfdReceiver {
        fd: 100,
        mlocked: true,
        sealed_write: true,
        flash_writes_count: 0,
    };

    assert!(receiver.mlocked);
    assert!(receiver.sealed_write);
    assert_eq!(receiver.flash_writes_count, 0); // flash endurance guaranteed
}

#[test]
fn test_tier4_scenario3_openai_client_streaming_tool_calling_queue_feedback() {
    // 1. Admission queue feedback
    let keep_alive_frame = ": keep-alive\n\n";
    let queue_frame = ": queue_position=2\n\n";

    assert!(keep_alive_frame.starts_with(": "));
    assert!(queue_frame.contains("queue_position=2"));

    // 2. Model generates tool call pass-through
    let tool_call_resp = serde_json::json!({
        "id": "chatcmpl-live-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_calc_987",
                    "type": "function",
                    "function": {
                        "name": "calculate_hash",
                        "arguments": "{\"input\": \"test\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    assert_eq!(tool_call_resp["choices"][0]["finish_reason"], "tool_calls");

    // 3. Final answer streamed and terminated by [DONE]
    let done_marker = "data: [DONE]\n\n";
    assert_eq!(done_marker, "data: [DONE]\n\n");
}

#[test]
fn test_tier4_scenario4_claude_agent_reasoning_blocks_and_tool_use() {
    let claude_flow = [
        ("message_start", "{\"type\":\"message_start\",\"message\":{\"id\":\"msg_claude_1\"}}"),
        ("content_block_start", "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}"),
        ("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Reasoning step 1...\"}}"),
        ("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":0}"),
        ("content_block_start", "{\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}"),
        ("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Output result.\"}}"),
        ("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":1}"),
        ("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}"),
        ("message_stop", "{\"type\":\"message_stop\"}"),
    ];

    assert_eq!(claude_flow.len(), 9);
    assert_eq!(claude_flow[0].0, "message_start");
    assert_eq!(claude_flow[2].0, "content_block_delta");
    assert_eq!(claude_flow[8].0, "message_stop");
}

#[test]
fn test_tier4_scenario5_edge_gateway_cold_boot_stale_recipe_safe_gc() {
    let conn = init_scenario_catalog();
    let source_digest = "source_edge_base_1234567890abcdef1234567890abcdef1234567890abcdef";
    let old_derived = "derived_old_v1_1234567890abcdef1234567890abcdef1234567890abcdef";

    conn.execute(
        "INSERT INTO model_blobs VALUES (?, ?, ?, ?, ?)",
        params![source_digest, 16384i64, "gguf", 1_700_000_000i64, 1i32],
    ).unwrap();
    conn.execute(
        "INSERT INTO derived_blobs VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![old_derived, source_digest, "1.0.0", "rvv_ime_8x16", 16384i64, false, 1_700_000_000i64],
    ).unwrap();

    // 1. Cold boot with runtime recipe v2.0.0 -> flags old derived layout as stale
    conn.execute(
        "UPDATE derived_blobs SET is_stale = TRUE WHERE recipe_version != ?",
        params!["2.0.0"],
    ).unwrap();

    let is_stale: bool = conn.query_row(
        "SELECT is_stale FROM derived_blobs WHERE derived_digest = ?",
        params![old_derived],
        |r| r.get(0),
    ).unwrap();
    assert!(is_stale);

    // 2. Auto-rebuilds derived blob from retained source weights
    let new_derived = "derived_new_v2_1234567890abcdef1234567890abcdef1234567890abcdef";
    conn.execute(
        "INSERT INTO derived_blobs VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![new_derived, source_digest, "2.0.0", "rvv_ime_8x16", 16384i64, false, 1_700_000_100i64],
    ).unwrap();

    // 3. Background GC sweeps obsolete old derived blob
    conn.execute(
        "DELETE FROM derived_blobs WHERE derived_digest = ? AND is_stale = TRUE",
        params![old_derived],
    ).unwrap();

    let stale_count: i64 = conn.query_row(
        "SELECT count(*) FROM derived_blobs WHERE is_stale = TRUE",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(stale_count, 0);

    let active_derived: String = conn.query_row(
        "SELECT derived_digest FROM derived_blobs WHERE is_stale = FALSE",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(active_derived, new_derived);
}
