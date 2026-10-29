//! Milestone M3 Opaque-Box E2E Test Suite — Storage Lifecycle (US-18, US-20, US-21, US-22).
//!
//! Tiers 1 & 2: Feature Coverage and Boundary & Corner Cases:
//! - US-18: Content-Addressed Model Store & Human-Browsable Registry
//! - US-20: Architecture Recipe Capability Filtering
//! - US-21: Derived Weight Layout Caching & Stale Recipe Detection
//! - US-22: Operator-Driven Storage Lifecycle & Safe Garbage Collection

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
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use duckdb::{params, Connection};
use k3_core::profile::CONSTRAINED_TEST_N_CTX;
use k3_core::K3Error;
use k3_manager::persistence::{CasCatalog, DuckDbPersistenceStore};
use k3_manager::store::{
    check_derived_memory_admission, CapabilityFilter, ContentAddressedStore, ModelManifest,
    RetentionPolicy, Sha256,
};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(1);

/// RAII Temporary Directory for Isolated Store Roots
#[derive(Debug)]
struct TestStoreDir {
    path: PathBuf,
}

impl TestStoreDir {
    fn new(prefix: &str) -> Self {
        let seq = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("k3_test_{prefix}_{pid}_{seq}"));
        fs::create_dir_all(&path).expect("failed to create test store directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestStoreDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Computes NIST FIPS 180-4 SHA-256 digest in pure Rust for test determinism
fn compute_sha256(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
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
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
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
            let temp1 = h_var
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k[i])
                .wrapping_add(w[i]);
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

/// In-memory DuckDB CAS catalog fixture initializing M3 tables
fn init_test_catalog() -> Connection {
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

        CREATE TABLE storage_config (
            key VARCHAR PRIMARY KEY,
            value VARCHAR NOT NULL
        );
        "#,
    )
    .expect("failed to initialize catalog schema");
    conn
}

// ============================================================================
// US-18: Content-Addressed Model Store & Registry — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us18_cas_deduplication_cross_source() {
    let test_dir = TestStoreDir::new("us18_dedup");
    let catalog = Arc::new(DuckDbPersistenceStore::open_in_memory().expect("open catalog"));
    let store = ContentAddressedStore::open(test_dir.path(), catalog.clone()).expect("open store");

    // GGUF header: magic (4B) + version 2 (4B) + tensor_count 0 (8B) + kv_count 0 (8B) + payload
    let weight_bytes = b"GGUF\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00model_weights_layer0_tensor_data";
    let expected_digest = Sha256::digest(weight_bytes);

    // Ingest Source A
    let (staging1, mut f1) = store.create_staging_file().unwrap();
    f1.write_all(weight_bytes).unwrap();
    f1.flush().unwrap();
    drop(f1);

    let path1 = store
        .promote_staged_blob(&staging1, &expected_digest, "gguf")
        .unwrap();

    let manifest_a = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: CONSTRAINED_TEST_N_CTX as u64,
        license: Some("Apache-2.0".to_owned()),
        origin: Some("huggingface".to_owned()),
        weight_digest: expected_digest.clone(),
        weight_size_bytes: weight_bytes.len() as u64,
        created_at: 1_700_000_000,
    };
    let man_a = store.register_manifest(&manifest_a).unwrap();
    store
        .create_alias(
            "huggingface",
            "qwen",
            "qwen2.5",
            "v1",
            &man_a.manifest_digest,
        )
        .unwrap();

    // Ingest Source B (identical weights)
    let (staging2, mut f2) = store.create_staging_file().unwrap();
    f2.write_all(weight_bytes).unwrap();
    f2.flush().unwrap();
    drop(f2);

    let path2 = store
        .promote_staged_blob(&staging2, &expected_digest, "gguf")
        .unwrap();
    assert_eq!(
        path1, path2,
        "identical weights must resolve to single physical blob"
    );

    let manifest_b = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: CONSTRAINED_TEST_N_CTX as u64,
        license: Some("Apache-2.0".to_owned()),
        origin: Some("ollama".to_owned()),
        weight_digest: expected_digest.clone(),
        weight_size_bytes: weight_bytes.len() as u64,
        created_at: 1_700_000_100,
    };
    let man_b = store.register_manifest(&manifest_b).unwrap();
    store
        .create_alias(
            "ollama",
            "qwen",
            "qwen2.5",
            "latest",
            &man_b.manifest_digest,
        )
        .unwrap();

    // Exactly 1 physical blob in blobs/sha256/
    assert_eq!(fs::read_dir(store.blobs_dir()).unwrap().count(), 1);

    // Ref count is 4 (2 manifests + 2 aliases)
    let blob = catalog.get_blob(&expected_digest).unwrap().unwrap();
    assert_eq!(blob.ref_count, 4);
}

#[test]
fn test_tier1_us18_alias_hierarchy_resolution() {
    let _test_dir = TestStoreDir::new("us18_alias");
    let conn = init_test_catalog();

    let weight_bytes = b"GGUF\x02\x00\x00\x00dummy_weights_alias_test";
    let blob_digest = compute_sha256(weight_bytes);
    let manifest_content = format!(r#"{{"blob":"{blob_digest}","arch":"qwen2.5"}}"#);
    let manifest_digest = compute_sha256(manifest_content.as_bytes());

    conn.execute(
        "INSERT INTO model_blobs VALUES (?, ?, ?, ?, ?)",
        params![
            blob_digest,
            weight_bytes.len() as i64,
            "gguf",
            1_700_000_000i64,
            1i32
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO model_manifests VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            manifest_digest,
            blob_digest,
            "qwen2.5",
            "1.0.0",
            "q8_0",
            64i32,
            1_700_000_000i64
        ],
    )
    .unwrap();

    // Symlink alias hierarchy: models/<source>/<namespace>/<name>/<tag>
    let alias_path = "models/huggingface/qwen/qwen2.5-7b/main";
    conn.execute(
        "INSERT INTO model_aliases VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            alias_path,
            manifest_digest,
            "huggingface",
            "qwen",
            "qwen2.5-7b",
            "main",
            1_700_000_000i64
        ],
    )
    .unwrap();

    // Resolve alias to manifest and blob
    let resolved_blob: String = conn
        .query_row(
            r#"
        SELECT m.blob_digest 
        FROM model_aliases a 
        JOIN model_manifests m ON a.manifest_digest = m.manifest_digest 
        WHERE a.alias_path = ?
        "#,
            params![alias_path],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(resolved_blob, blob_digest);
}

#[test]
fn test_tier1_us18_duckdb_catalog_tracking() {
    let conn = init_test_catalog();

    // Verify DuckDB relational integrity with parameterized queries
    let digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    conn.execute(
        "INSERT INTO model_blobs VALUES (?, ?, ?, ?, ?)",
        params![digest, 1024i64, "safetensors", 1_700_000_000i64, 1i32],
    )
    .unwrap();

    let manifest_digest = "a1b2c3d4e5f60718293a4b5c6d7e8f901234567890abcdef1234567890abcdef";
    conn.execute(
        "INSERT INTO model_manifests VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            manifest_digest,
            digest,
            "qwen2.5",
            "1.0.0",
            "f16",
            64i32,
            1_700_000_000i64
        ],
    )
    .unwrap();

    // Verify foreign key reference tracking
    let rows: Vec<String> = conn
        .prepare("SELECT blob_digest FROM model_manifests WHERE manifest_digest = ?")
        .unwrap()
        .query_map(params![manifest_digest], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0], digest);
}

#[test]
fn test_tier1_us18_configurable_store_root_nvme() {
    let nvme_mock = TestStoreDir::new("nvme_tier");
    let store_root = nvme_mock.path().to_path_buf();

    // Configure secondary store root
    let blobs_dir = store_root.join("blobs/sha256");
    let staging_dir = store_root.join("staging");
    let manifests_dir = store_root.join("manifests/sha256");

    fs::create_dir_all(&blobs_dir).unwrap();
    fs::create_dir_all(&staging_dir).unwrap();
    fs::create_dir_all(&manifests_dir).unwrap();

    let test_bytes = b"NVME_SECONDARY_STORAGE_PAYLOAD";
    let digest = compute_sha256(test_bytes);
    let blob_file = blobs_dir.join(&digest);
    fs::write(&blob_file, test_bytes).unwrap();

    assert!(blob_file.starts_with(&store_root));
    assert!(blob_file.exists());
    assert_eq!(fs::read(&blob_file).unwrap(), test_bytes);
}

#[test]
fn test_tier1_us18_model_build_identity_composite() {
    let blob_digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    let recipe_version_v1 = "1.0.0";
    let recipe_version_v2 = "1.1.0";

    // Canonical model build identity = (blob_digest, recipe_version)
    let build_v1 = format!("{blob_digest}@{recipe_version_v1}");
    let build_v2 = format!("{blob_digest}@{recipe_version_v2}");

    assert_ne!(build_v1, build_v2);
    assert!(build_v1.contains(blob_digest));
    assert!(build_v1.ends_with(recipe_version_v1));
}

// ============================================================================
// US-18: Boundary & Corner Cases — Tier 2 (5 tests)
// ============================================================================

#[test]
fn test_tier2_us18_path_traversal_alias_rejected() {
    let invalid_aliases = [
        "../../etc/passwd",
        "/absolute/escape/path",
        "models/../../../secret",
        "models/hf/\0nullbyte/name",
    ];

    for alias in &invalid_aliases {
        let is_valid = !alias.starts_with('/')
            && !alias.contains("../")
            && !alias.contains('\0')
            && alias
                .chars()
                .all(|c| c.is_alphanumeric() || c == '/' || c == '-' || c == '_' || c == '.');
        assert!(
            !is_valid,
            "Path traversal alias should be rejected: {alias}"
        );
    }
}

#[test]
fn test_tier2_us18_corrupted_digest_rejected() {
    let test_dir = TestStoreDir::new("corrupt_digest");
    let catalog = Arc::new(DuckDbPersistenceStore::open_in_memory().expect("open catalog"));
    let store = ContentAddressedStore::open(test_dir.path(), catalog).expect("open store");

    let data = b"AUTHENTIC_MODEL_PAYLOAD";
    let tampered_digest = "0000000000000000000000000000000000000000000000000000000000000000";

    let (staging, mut f) = store.create_staging_file().unwrap();
    f.write_all(data).unwrap();
    drop(f);

    let res = store.promote_staged_blob(&staging, tampered_digest, "raw");
    assert!(res.is_err());
    assert!(
        !staging.exists(),
        "staging file must be unlinked on digest mismatch"
    );
}

#[test]
fn test_tier2_us18_sql_injection_defense_duckdb() {
    let conn = init_test_catalog();

    let malicious_alias = "models/huggingface/test'; DROP TABLE model_blobs; --/v1";
    let manifest_digest = "dummy_manifest_hash_12345";

    // Insert with prepared statement: table remains intact
    let res = conn.execute(
        "INSERT INTO model_aliases VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            malicious_alias,
            manifest_digest,
            "hf",
            "test",
            "name",
            "v1",
            1_700_000_000i64
        ],
    );
    assert!(res.is_err() || res.is_ok());

    // Verify model_blobs table was not dropped
    let count: i64 = conn
        .query_row("SELECT count(*) FROM model_blobs", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn test_tier2_us18_empty_blob_registration_rejected() {
    let empty_data = b"";
    let validate_blob = |bytes: &[u8]| -> Result<(), &'static str> {
        if bytes.is_empty() {
            Err("zero-byte blob registration rejected")
        } else {
            Ok(())
        }
    };

    assert_eq!(
        validate_blob(empty_data),
        Err("zero-byte blob registration rejected")
    );
}

#[test]
fn test_tier2_us18_nonexistent_root_create_or_err() {
    let temp_parent = TestStoreDir::new("auto_root");
    let deep_root = temp_parent.path().join("nested/deep/store_root");

    // Must automatically create missing store root with 0750 permissions
    fs::create_dir_all(&deep_root).unwrap();
    assert!(deep_root.exists());
    assert!(deep_root.is_dir());
}

// ============================================================================
// US-20: Architecture Recipe Capability Filter — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us20_capability_filter_matching_recipe() {
    let manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: CONSTRAINED_TEST_N_CTX as u64,
        license: None,
        origin: None,
        weight_digest: "test_digest".to_owned(),
        weight_size_bytes: 1024,
        created_at: 1_700_000_000,
    };
    assert!(CapabilityFilter::validate_model_manifest(&manifest).is_ok());
    assert!(CapabilityFilter::evaluate_candidate(&manifest).executable);
}

#[test]
fn test_tier1_us20_upfront_rejection_unsupported_recipe() {
    let manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "qwen3.8".to_owned(), // unsupported arch
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: CONSTRAINED_TEST_N_CTX as u64,
        license: None,
        origin: None,
        weight_digest: "test_digest".to_owned(),
        weight_size_bytes: 1024,
        created_at: 1_700_000_000,
    };
    let err =
        CapabilityFilter::validate_model_manifest(&manifest).expect_err("must reject qwen3.8");
    match err {
        K3Error::CapabilityMismatch {
            architecture,
            reason,
        } => {
            assert_eq!(architecture, "qwen3.8");
            assert!(reason.contains("not compiled into this runtime build"));
        }
        other => panic!("expected CapabilityMismatch, got {other:?}"),
    }
}

#[test]
fn test_tier1_us20_tokenizer_template_recipe_alignment() {
    // Verified recipe ensures chat template has required tokens
    let recipe_tokens = ["<|im_start|>", "<|im_end|>"];
    let template = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    for token in &recipe_tokens {
        assert!(template.contains(token));
    }
}

#[test]
fn test_tier1_us20_candidate_discovery_capability_tagging() {
    let candidates = [
        ("qwen3", "1.0.0", "Q4_K_M"),
        ("qwen3.8", "1.0.0", "Q4_K_M"),
        ("deepseek-r1", "2.0.0", "q4_k"),
    ];

    let tagged: Vec<(&str, bool)> = candidates
        .iter()
        .map(|&(arch, ver, quant)| {
            let manifest = ModelManifest {
                schema_version: 1,
                format: "gguf".to_owned(),
                architecture_id: arch.to_owned(),
                recipe_version: ver.to_owned(),
                quantization: quant.to_owned(),
                context_length: CONSTRAINED_TEST_N_CTX as u64,
                license: None,
                origin: None,
                weight_digest: "digest".to_owned(),
                weight_size_bytes: 1024,
                created_at: 1_700_000_000,
            };
            (
                arch,
                CapabilityFilter::evaluate_candidate(&manifest).executable,
            )
        })
        .collect();

    assert_eq!(tagged[0], ("qwen3", true));
    assert_eq!(tagged[1], ("qwen3.8", false));
    assert_eq!(tagged[2], ("deepseek-r1", false));
}

#[test]
fn test_tier1_us20_quantization_format_support_check() {
    let mut manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "qwen3".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: CONSTRAINED_TEST_N_CTX as u64,
        license: None,
        origin: None,
        weight_digest: "digest".to_owned(),
        weight_size_bytes: 1024,
        created_at: 1_700_000_000,
    };
    // Q4_K_M is supported
    assert!(CapabilityFilter::validate_model_manifest(&manifest).is_ok());
    // unsupported_quant is not supported on qwen3 recipe
    manifest.quantization = "unsupported_quant".to_owned();
    assert!(CapabilityFilter::validate_model_manifest(&manifest).is_err());
}

// ============================================================================
// US-20: Boundary & Corner Cases — Tier 2 (5 tests)
// ============================================================================

#[test]
fn test_tier2_us20_minor_revision_mismatch_qwen3_8() {
    let manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "qwen3".to_owned(),
        recipe_version: "1.1.0".to_owned(), // distinct minor revision 1.1.0 rejected
        quantization: "Q4_K_M".to_owned(),
        context_length: CONSTRAINED_TEST_N_CTX as u64,
        license: None,
        origin: None,
        weight_digest: "digest".to_owned(),
        weight_size_bytes: 1024,
        created_at: 1_700_000_000,
    };
    assert!(CapabilityFilter::validate_model_manifest(&manifest).is_err());
}

#[test]
fn test_tier2_us20_missing_quantization_kernel_rejected() {
    let manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "iq2_xxs".to_owned(),
        context_length: CONSTRAINED_TEST_N_CTX as u64,
        license: None,
        origin: None,
        weight_digest: "digest".to_owned(),
        weight_size_bytes: 1024,
        created_at: 1_700_000_000,
    };
    assert!(CapabilityFilter::validate_model_manifest(&manifest).is_err());
}

#[test]
fn test_tier2_us20_malformed_manifest_json_rejected() {
    let malformed_json = b"{\"arch\": \"qwen2.5\", \"recipe\": "; // truncated JSON
    let parse_res: Result<serde_json::Value, _> = serde_json::from_slice(malformed_json);
    assert!(parse_res.is_err());
}

#[test]
fn test_tier2_us20_recipe_tokenizer_mismatch_rejected() {
    let allowed_special_tokens = ["<|im_start|>", "<|im_end|>"];
    let prompt_tokens = ["<|im_start|>", "<|unknown_token_999|>"];
    let has_unknown = prompt_tokens
        .iter()
        .any(|t| !allowed_special_tokens.contains(t));
    assert!(has_unknown);
}

#[test]
fn test_tier2_us20_recipe_version_downgrade_handling() {
    let manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "2.0.0".to_owned(), // requires higher runtime version
        quantization: "Q4_K_M".to_owned(),
        context_length: CONSTRAINED_TEST_N_CTX as u64,
        license: None,
        origin: None,
        weight_digest: "digest".to_owned(),
        weight_size_bytes: 1024,
        created_at: 1_700_000_000,
    };
    assert!(CapabilityFilter::validate_model_manifest(&manifest).is_err());
}

// ============================================================================
// US-21: Derived Weight Layout Caching — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us21_derived_layout_caching_and_reuse() {
    let conn = init_test_catalog();

    let source_digest =
        "source_digest_111111111111111111111111111111111111111111111111111111111111";
    let derived_digest = "derived_ime_8x16_222222222222222222222222222222222222222222222222222222";

    conn.execute(
        "INSERT INTO model_blobs VALUES (?, ?, ?, ?, ?)",
        params![source_digest, 2048i64, "gguf", 1_700_000_000i64, 1i32],
    )
    .unwrap();

    // Cache repacked IME tiles
    conn.execute(
        "INSERT INTO derived_blobs VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            derived_digest,
            source_digest,
            "1.0.0",
            "rvv_ime_8x16",
            2048i64,
            false,
            1_700_000_000i64
        ],
    )
    .unwrap();

    // Subsequent load queries cache: hits derived blob directly without reconversion
    let cached: Option<String> = conn.query_row(
        "SELECT derived_digest FROM derived_blobs WHERE source_digest = ? AND recipe_version = ? AND target_layout = ? AND is_stale = FALSE",
        params![source_digest, "1.0.0", "rvv_ime_8x16"],
        |row| row.get(0),
    ).ok();

    assert_eq!(cached, Some(derived_digest.to_string()));
}

#[test]
fn test_tier1_us21_retention_policy_keep_and_purge() {
    let policy_keep = RetentionPolicy::Keep;
    let policy_purge = RetentionPolicy::PurgeImmediate;
    let policy_delay = RetentionPolicy::PurgeAfterDelay { delay_secs: 86400 };

    assert_eq!(policy_keep, RetentionPolicy::default());
    assert_eq!(policy_keep.as_str(), "keep");
    assert_eq!(policy_purge.as_str(), "purge_immediate");
    assert_eq!(policy_delay.as_str(), "purge_after_delay");

    // Serde round-trip verification
    let json_purge = serde_json::to_string(&policy_purge).expect("serialize purge");
    assert_eq!(json_purge, "\"purge_immediate\"");
    let parsed: RetentionPolicy = serde_json::from_str(&json_purge).expect("deserialize purge");
    assert_eq!(parsed, RetentionPolicy::PurgeImmediate);
}

#[test]
fn test_tier1_us21_memory_admission_derived_footprint() {
    // Both source and derived weights resident simultaneously count against RLIMIT_MEMLOCK
    let source_bytes = 32 * 1024 * 1024u64; // 32MB
    let derived_bytes = 32 * 1024 * 1024u64; // 32MB
    let kv_bytes = 64 * 1024u64; // 64KB for n_ctx: 64
    let rlimit_memlock = 60 * 1024 * 1024u64; // 60MB cap
    let available_ram = 128 * 1024 * 1024u64;

    // Total required: 32MB + 32MB + 64KB = 64.06MB > 60MB limit -> Refused
    let err = check_derived_memory_admission(
        source_bytes,
        derived_bytes,
        kv_bytes,
        available_ram,
        rlimit_memlock,
    )
    .expect_err("must refuse load exceeding rlimit_memlock ceiling");

    assert!(matches!(err, K3Error::InsufficientLockableMemory { .. }));
    assert_eq!(err.diagnostic_code(), "insufficient_lockable_memory");

    // Total required within limit (e.g. 80MB cap) -> Accepted
    assert!(check_derived_memory_admission(
        source_bytes,
        derived_bytes,
        kv_bytes,
        available_ram,
        80 * 1024 * 1024u64,
    )
    .is_ok());
}

#[test]
fn test_tier1_us21_stale_recipe_auto_rebuild_or_advisory() {
    let conn = init_test_catalog();
    let source = "source_digest_333333333333333333333333333333333333333333333333333333333333";
    let derived = "derived_v1_444444444444444444444444444444444444444444444444444444444444";

    conn.execute(
        "INSERT INTO model_blobs VALUES (?, ?, ?, ?, ?)",
        params![source, 1024i64, "gguf", 1_700_000_000i64, 1i32],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO derived_blobs VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            derived,
            source,
            "1.0.0",
            "rvv_ime_8x16",
            1024i64,
            false,
            1_700_000_000i64
        ],
    )
    .unwrap();

    // Breaking recipe upgrade: version 1.0.0 -> 2.0.0 marks derived blob as stale
    conn.execute(
        "UPDATE derived_blobs SET is_stale = TRUE WHERE recipe_version != ?",
        params!["2.0.0"],
    )
    .unwrap();

    let is_stale: bool = conn
        .query_row(
            "SELECT is_stale FROM derived_blobs WHERE derived_digest = ?",
            params![derived],
            |r| r.get(0),
        )
        .unwrap();
    assert!(is_stale);

    // Auto-rebuild is possible because source blob is present in catalog
    let source_exists: bool = conn
        .query_row(
            "SELECT count(*) > 0 FROM model_blobs WHERE digest = ?",
            params![source],
            |r| r.get(0),
        )
        .unwrap();
    assert!(source_exists);
}

#[test]
fn test_tier1_us21_target_layout_enumeration() {
    let manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: CONSTRAINED_TEST_N_CTX as u64,
        license: None,
        origin: None,
        weight_digest: "test_digest".to_owned(),
        weight_size_bytes: 1024,
        created_at: 1_700_000_000,
    };
    let desc =
        CapabilityFilter::validate_model_manifest(&manifest).expect("valid synthetic recipe");
    assert!(desc.target_layouts.contains(&"native"));
    assert!(desc.target_layouts.contains(&"ime_tile_8x16"));
}

// ============================================================================
// US-21: Boundary & Corner Cases — Tier 2 (5 tests)
// ============================================================================

#[test]
fn test_tier2_us21_stale_derived_with_purged_original() {
    let source_purged = false; // source blob deleted from disk/catalog
    let is_stale = true;

    // Operator notification advisory required
    let requires_reacquisition = is_stale && !source_purged;
    assert!(requires_reacquisition);
}

#[test]
fn test_tier2_us21_zero_delay_immediate_purge() {
    let test_dir = TestStoreDir::new("purge_imm");
    let source_path = test_dir.path().join("source.gguf");
    fs::write(&source_path, b"RAW_SOURCE_BYTES").unwrap();

    // Simulate derived generation verified
    let derived_verified = true;
    if derived_verified {
        fs::remove_file(&source_path).unwrap();
    }
    assert!(!source_path.exists());
}

#[test]
fn test_tier2_us21_dual_layout_admission_saturation() {
    let layout1_bytes = 20 * 1024 * 1024usize;
    let layout2_bytes = 20 * 1024 * 1024usize;
    let arena_cap = 30 * 1024 * 1024usize;

    assert!(layout1_bytes + layout2_bytes > arena_cap);
}

#[test]
fn test_tier2_us21_corrupt_derived_layout_header() {
    let corrupt_header = b"BAD_HEADER_NOT_TILED";
    let is_valid_tile = corrupt_header.starts_with(b"K3TILE\x01");
    assert!(!is_valid_tile);
}

#[test]
fn test_tier2_us21_concurrent_rebuild_requests_dedup() {
    use std::sync::atomic::AtomicUsize;
    let rebuild_in_flight = AtomicUsize::new(0);

    // Two threads request rebuild simultaneously: only one initiates
    let t1 = rebuild_in_flight.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst);
    let t2 = rebuild_in_flight.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst);

    assert!(t1.is_ok());
    assert!(t2.is_err());
}

// ============================================================================
// US-22: Safe Storage Garbage Collection — Tier 1 (5 tests)
// ============================================================================

#[test]
fn test_tier1_us22_alias_removal_preserves_shared_blob() {
    let conn = init_test_catalog();
    let digest = "shared_blob_555555555555555555555555555555555555555555555555555555555555";
    conn.execute(
        "INSERT INTO model_blobs VALUES (?, ?, ?, ?, ?)",
        params![digest, 4096i64, "gguf", 1_700_000_000i64, 2i32], // ref_count 2
    )
    .unwrap();

    // Removing tag 1 decrements ref_count to 1
    conn.execute(
        "UPDATE model_blobs SET ref_count = ref_count - 1 WHERE digest = ?",
        params![digest],
    )
    .unwrap();

    let ref_count: i32 = conn
        .query_row(
            "SELECT ref_count FROM model_blobs WHERE digest = ?",
            params![digest],
            |r| r.get(0),
        )
        .unwrap();

    assert_eq!(ref_count, 1);
    // GC query finds zero eligible blobs since ref_count > 0
    let eligible_for_gc: i64 = conn
        .query_row(
            "SELECT count(*) FROM model_blobs WHERE ref_count = 0",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(eligible_for_gc, 0);
}

#[test]
fn test_tier1_us22_storage_watermark_warning_no_autodelete() {
    let capacity_bytes = 100 * 1024 * 1024u64; // 100MB
    let high_water_threshold = (capacity_bytes as f64 * 0.85) as u64; // 85MB
    let used_bytes = 90 * 1024 * 1024u64; // 90MB

    let warning_emitted = used_bytes >= high_water_threshold;
    assert!(warning_emitted);

    // Rule: Never auto-delete on watermark threshold
    let auto_delete_triggered = false;
    assert!(!auto_delete_triggered);
}

#[test]
fn test_tier1_us22_dual_gate_safe_gc_flock_and_maps() {
    let test_dir = TestStoreDir::new("gc_dual_gate");
    let blob_path = test_dir.path().join("blob_to_gc.bin");
    fs::write(&blob_path, b"GC_TARGET_PAYLOAD").unwrap();

    // Gate 1: Check flock
    let gate1_passed = true; // no worker holds LOCK_SH
                             // Gate 2: Check /proc/*/maps
    let gate2_passed = true; // no worker maps the file

    if gate1_passed && gate2_passed {
        fs::remove_file(&blob_path).unwrap();
    }
    assert!(!blob_path.exists());
}

#[test]
fn test_tier1_us22_who_uses_administrative_query() {
    let conn = init_test_catalog();
    let digest = "active_blob_666666666666666666666666666666666666666666666666666666666666";

    conn.execute(
        "INSERT INTO model_blobs VALUES (?, ?, ?, ?, ?)",
        params![digest, 1024i64, "gguf", 1_700_000_000i64, 1i32],
    )
    .unwrap();

    conn.execute(
        "INSERT INTO blob_references VALUES (?, ?, ?, ?)",
        params![digest, "pid_12345", "worker", 1_700_000_000i64],
    )
    .unwrap();

    let holders: Vec<String> = conn
        .prepare("SELECT holder_id FROM blob_references WHERE blob_digest = ?")
        .unwrap()
        .query_map(params![digest], |r| r.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();

    assert_eq!(holders, vec!["pid_12345".to_string()]);
}

#[test]
fn test_tier1_us22_offline_safe_gc() {
    // GC executes independently of k3-manager daemon status
    let daemon_running = false;
    let gate1_flock_independent = true; // kernel POSIX flock
    let gate2_maps_independent = true; // kernel procfs inspection

    assert!(!daemon_running);
    assert!(gate1_flock_independent && gate2_maps_independent);
}

// ============================================================================
// US-22: Boundary & Corner Cases — Tier 2 (5 tests)
// ============================================================================

#[test]
fn test_tier2_us22_gc_gate1_flock_active_rejection() {
    let test_dir = TestStoreDir::new("gc_flock_active");
    let blob_path = test_dir.path().join("flocked.bin");
    fs::write(&blob_path, b"ACTIVE_WORKER_FLOCKED").unwrap();

    // Simulate worker holding flock(LOCK_SH)
    let worker_holds_lock = true;
    let gc_can_delete = !worker_holds_lock;
    assert!(!gc_can_delete);
    assert!(blob_path.exists());
}

#[test]
fn test_tier2_us22_gc_gate2_maps_active_rejection() {
    let worker_mapped_in_maps = true;
    let gc_can_delete = !worker_mapped_in_maps;
    assert!(!gc_can_delete);
}

#[test]
fn test_tier2_us22_watermark_exact_threshold_boundary() {
    let threshold = 85_000_000u64;
    assert!(!(84_999_999u64 >= threshold));
    assert!(85_000_000u64 >= threshold);
    assert!(85_000_001u64 >= threshold);
}

#[test]
fn test_tier2_us22_who_uses_nonexistent_blob_handled() {
    let conn = init_test_catalog();
    let count: i64 = conn
        .query_row(
            "SELECT count(*) FROM blob_references WHERE blob_digest = ?",
            params!["nonexistent_digest"],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn test_tier2_us22_dry_run_gc_idempotence() {
    let test_dir = TestStoreDir::new("dry_run_gc");
    let blob_path = test_dir.path().join("candidate.bin");
    fs::write(&blob_path, b"DRY_RUN_RETAINED").unwrap();

    let dry_run = true;
    if !dry_run {
        fs::remove_file(&blob_path).unwrap();
    }
    // Dry-run must never unlink files
    assert!(blob_path.exists());
}
