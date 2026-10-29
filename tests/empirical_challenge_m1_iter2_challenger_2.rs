//! Empirical adversarial verification test harness for Milestone M1 (Challenger 2).
//!
//! Focus areas:
//! 1. DuckDB CAS catalog: SQL injection resistance across all API surfaces, concurrent alias registration/deregistration,
//!    and reference count invariants under multi-threaded race conditions.
//! 2. Capability filtering: upfront rejection of uncompiled recipes (qwen3.8, etc.), mismatched versions, invalid quants,
//!    zero context length, verifying K3Error::CapabilityMismatch without allocating memory.
//! 3. Derived weight layout caching and retention policies (Keep, PurgeImmediate, PurgeAfterDelay),
//!    concurrent derivation requests, and stale recipe detection with auto-rebuild vs operator advisory.

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::assert_is_empty,
    clippy::match_same_arms,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::redundant_closure_for_method_calls,
    clippy::doc_markdown
)]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use k3_core::K3Error;
use k3_manager::persistence::{
    BlobReferenceRecord, CasCatalog, DerivedBlobRecord, DuckDbPersistenceStore, ModelAliasRecord,
    ModelBlobRecord, ModelManifestRecord,
};
use k3_manager::store::{
    CapabilityFilter, ContentAddressedStore, DerivedBlobManager, DerivedResolution, ModelManifest,
    RetentionPolicy, Sha256, StaleRecipeOutcome, check_derived_memory_admission,
};
use k3_tokenizer::SimpleTokenizer;

static GLOBAL_TEST_COUNTER: AtomicU64 = AtomicU64::new(7000);

struct TempStoreDir {
    path: PathBuf,
}

impl TempStoreDir {
    fn new(prefix: &str) -> Self {
        let seq = GLOBAL_TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{seq}", std::process::id()));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempStoreDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn setup_cas_store() -> (ContentAddressedStore, Arc<DuckDbPersistenceStore>, TempStoreDir) {
    let temp_dir = TempStoreDir::new("k3_m1_chal2");
    let catalog = Arc::new(DuckDbPersistenceStore::open_in_memory().expect("open duckdb in memory"));
    let store = ContentAddressedStore::open(temp_dir.path(), catalog.clone()).expect("open CAS store");
    (store, catalog, temp_dir)
}

fn register_dummy_blob(store: &ContentAddressedStore, content: &[u8]) -> (String, PathBuf) {
    let digest = Sha256::digest(content);
    let (staging_path, mut file) = store.create_staging_file().expect("staging file");
    file.write_all(content).expect("write staging");
    drop(file);

    let promoted = store
        .promote_staged_blob(&staging_path, &digest, "raw")
        .expect("promote staging blob");
    (digest, promoted)
}

fn register_dummy_manifest(
    store: &ContentAddressedStore,
    weight_digest: &str,
    arch: &str,
    recipe: &str,
    quant: &str,
) -> ModelManifestRecord {
    let manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: arch.to_owned(),
        recipe_version: recipe.to_owned(),
        quantization: quant.to_owned(),
        context_length: 2048,
        license: Some("MIT".to_owned()),
        origin: Some("https://example.com/model".to_owned()),
        weight_digest: weight_digest.to_owned(),
        weight_size_bytes: 1024,
        created_at: 1000,
    };
    store.register_manifest(&manifest).expect("register manifest")
}

// ============================================================================
// 1. DuckDB CAS Catalog: SQL Injection Stress Testing Across All Methods
// ============================================================================

#[test]
fn test_duckdb_cas_catalog_exhaustive_sql_injection() {
    let (_store, catalog, _tmp) = setup_cas_store();

    let sql_attack_vectors = [
        "'; DROP TABLE model_blobs; --",
        "' OR '1'='1",
        "'; DROP TABLE model_manifests; --",
        "'; DROP TABLE model_aliases; --",
        "'; DROP TABLE derived_blobs; --",
        "'; DROP TABLE blob_references; --",
        "'; DROP TABLE storage_config; --",
        "admin'--",
        "1' OR 1=1 UNION SELECT 1, 2, 3, 4, 5, 6 --",
        "test' AND SLEEP(5) --",
        "\"; DROP TABLE model_blobs; --",
        "'; UPDATE model_blobs SET ref_count = 999999; --",
        "test\0injection",
        "Robert'); DROP TABLE Students;--",
    ];

    for (idx, attack) in sql_attack_vectors.iter().enumerate() {
        // A. Blob lifecycle operations with injection string as digest / fields
        let blob_rec = ModelBlobRecord {
            digest: format!("attack_digest_{idx}_{attack}"),
            size_bytes: 4096,
            blob_type: attack.to_string(),
            created_at: 100,
            last_accessed_at: 200,
            ref_count: 0,
        };
        assert!(
            catalog.save_blob(&blob_rec).is_ok(),
            "save_blob must handle payload safely without executing SQL"
        );

        let retrieved = catalog.get_blob(&blob_rec.digest).expect("get_blob query");
        assert!(retrieved.is_some(), "blob must be retrievable literally");
        assert_eq!(retrieved.as_ref().unwrap().blob_type, *attack);

        assert!(catalog.update_blob_access(&blob_rec.digest, 500).is_ok());
        assert!(catalog.increment_blob_ref(&blob_rec.digest).is_ok());
        assert!(catalog.decrement_blob_ref(&blob_rec.digest).is_ok());

        // B. Manifest lifecycle operations with injection strings in all fields
        let manifest_digest = format!("attack_manifest_{idx}_{attack}");
        let manifest_rec = ModelManifestRecord {
            manifest_digest: manifest_digest.clone(),
            weight_digest: blob_rec.digest.clone(),
            format: attack.to_string(),
            architecture_id: attack.to_string(),
            recipe_version: attack.to_string(),
            quantization: attack.to_string(),
            context_length: 2048,
            license: Some(attack.to_string()),
            origin: Some(attack.to_string()),
            manifest_json: format!("{{\"attack\": \"{attack}\"}}"),
            created_at: 100,
        };
        assert!(
            catalog.save_manifest(&manifest_rec).is_ok(),
            "save_manifest must handle injection payload safely"
        );

        let got_man = catalog.get_manifest(&manifest_digest).expect("get_manifest");
        assert!(got_man.is_some());
        assert_eq!(got_man.as_ref().unwrap().architecture_id, *attack);

        // C. Alias lifecycle operations with injection strings
        let alias_id = format!("attack_alias_{idx}_{attack}");
        let alias_rec = ModelAliasRecord {
            alias_id: alias_id.clone(),
            source: attack.to_string(),
            namespace: attack.to_string(),
            name: attack.to_string(),
            tag: attack.to_string(),
            manifest_digest: manifest_digest.clone(),
            weight_digest: blob_rec.digest.clone(),
            alias_path: format!("models/{attack}"),
            created_at: 100,
        };
        assert!(
            catalog.save_alias(&alias_rec).is_ok(),
            "save_alias must handle injection payload safely"
        );

        let got_alias = catalog.get_alias(&alias_id).expect("get_alias");
        assert!(got_alias.is_some());
        assert_eq!(got_alias.as_ref().unwrap().source, *attack);

        let find_res = catalog
            .find_alias(attack, attack, attack, attack)
            .expect("find_alias");
        assert!(find_res.is_some());

        // D. Derived layout operations with injection strings
        // First create a derived blob record in model_blobs to satisfy foreign key
        let derived_digest = format!("derived_blob_{idx}_{attack}");
        let derived_blob_rec = ModelBlobRecord {
            digest: derived_digest.clone(),
            size_bytes: 2048,
            blob_type: "derived_weights".to_owned(),
            created_at: 100,
            last_accessed_at: 100,
            ref_count: 1,
        };
        assert!(catalog.save_blob(&derived_blob_rec).is_ok());

        let derived_rec = DerivedBlobRecord {
            derived_digest: derived_digest.clone(),
            source_digest: blob_rec.digest.clone(),
            recipe_version: attack.to_string(),
            target_layout: attack.to_string(),
            status: "valid".to_owned(),
            created_at: 100,
        };
        assert!(
            catalog.save_derived_blob(&derived_rec).is_ok(),
            "save_derived_blob must handle injection safely"
        );

        let got_derived = catalog
            .get_derived_blob(&blob_rec.digest, attack, attack)
            .expect("get_derived_blob");
        assert!(got_derived.is_some());
        assert_eq!(got_derived.unwrap().recipe_version, *attack);

        assert!(catalog.update_derived_status(&derived_digest, "stale").is_ok());

        // E. Blob references with injection strings
        let ref_id = format!("ref_{idx}_{attack}");
        let ref_rec = BlobReferenceRecord {
            ref_id: ref_id.clone(),
            blob_digest: blob_rec.digest.clone(),
            ref_type: attack.to_string(),
            ref_source: attack.to_string(),
            created_at: 100,
        };
        assert!(catalog.add_blob_reference(&ref_rec).is_ok());
        let refs = catalog.list_blob_references(&blob_rec.digest).expect("list_blob_references");
        assert!(!refs.is_empty());
        assert!(catalog.remove_blob_reference(&ref_id).expect("remove ref"));

        // F. Storage config with injection strings
        let config_key = format!("cfg_key_{idx}_{attack}");
        assert!(catalog.set_storage_config(&config_key, attack).is_ok());
        let got_cfg = catalog.get_storage_config(&config_key).expect("get config");
        assert_eq!(got_cfg, Some(attack.to_string()));

        // G. Clean up created test entities
        assert!(catalog.delete_alias(&alias_id).expect("delete alias"));
        assert!(catalog.delete_manifest(&manifest_digest).expect("delete manifest"));
        assert!(catalog.delete_derived_blob(&derived_digest).expect("delete derived"));
        assert!(catalog.delete_blob(&derived_digest).expect("delete derived blob"));
        assert!(catalog.delete_blob(&blob_rec.digest).expect("delete base blob"));
    }

    // Crucial check: verify all tables still exist, are intact, and can be queried normally
    assert!(catalog.list_blobs().is_ok());
    assert!(catalog.list_manifests().is_ok());
    assert!(catalog.list_aliases().is_ok());
    assert!(catalog.list_derived_blobs().is_ok());
    assert!(catalog.total_blob_bytes().is_ok());
}

// ============================================================================
// 2. Concurrent Alias Registration, Deregistration & Reference Count Invariants
// ============================================================================

#[test]
fn test_duckdb_concurrent_alias_registration_and_ref_count_invariants() {
    let (store, catalog, _tmp) = setup_cas_store();

    // Register a canonical base blob and manifest
    let base_content = b"shared_weight_blob_for_concurrent_testing";
    let (weight_digest, _path) = register_dummy_blob(&store, base_content);
    let man_rec = register_dummy_manifest(&store, &weight_digest, "synthetic", "1.0.0", "F32");

    // Initially: manifest reference gives ref_count = 1
    let initial_blob = catalog.get_blob(&weight_digest).unwrap().unwrap();
    assert_eq!(
        initial_blob.ref_count, 1,
        "initial ref count must be 1 from manifest reference"
    );

    // Spawn 12 concurrent worker threads registering distinct aliases to the same model
    let num_threads = 12;
    let aliases_per_thread = 5;
    let mut handles = Vec::new();

    for t_idx in 0..num_threads {
        let store_clone = store.clone();
        let man_digest_clone = man_rec.manifest_digest.clone();

        let handle = thread::spawn(move || {
            let mut created_aliases = Vec::new();
            for a_idx in 0..aliases_per_thread {
                let source = format!("src_{t_idx}");
                let namespace = format!("ns_{a_idx}");
                let name = format!("model_{t_idx}_{a_idx}");
                let tag = "latest";

                let alias = store_clone
                    .create_alias(&source, &namespace, &name, tag, &man_digest_clone)
                    .unwrap_or_else(|e| panic!("thread {t_idx} failed to create alias: {e}"));

                created_aliases.push(alias);
            }
            created_aliases
        });
        handles.push(handle);
    }

    let mut all_created_aliases = Vec::new();
    for handle in handles {
        let aliases = handle.join().expect("thread join succeeded");
        all_created_aliases.extend(aliases);
    }

    assert_eq!(all_created_aliases.len(), num_threads * aliases_per_thread);

    // Verify all aliases are resolvable
    for alias in &all_created_aliases {
        let resolved = store.resolve_alias(&alias.alias_id).expect("resolve created alias");
        assert_eq!(resolved.weight_path, store.blobs_dir().join(&weight_digest));
        assert!(resolved.weight_path.exists());
    }

    // Verify reference count invariant:
    // Total references = 1 (manifest) + (num_threads * aliases_per_thread) (aliases)
    let expected_refs = 1 + (num_threads * aliases_per_thread) as u32;
    let blob_after_add = catalog.get_blob(&weight_digest).unwrap().unwrap();
    assert_eq!(
        blob_after_add.ref_count, expected_refs,
        "ref_count on model_blobs must exactly equal total references"
    );

    let db_refs = catalog.list_blob_references(&weight_digest).unwrap();
    assert_eq!(
        db_refs.len(),
        expected_refs as usize,
        "active blob_references count must match ref_count"
    );

    // Now test concurrent deregistration / removal of half of the aliases
    let to_remove = all_created_aliases.split_off(all_created_aliases.len() / 2);
    let remove_count = to_remove.len();

    let mut remove_handles = Vec::new();
    let to_remove_chunks: Vec<_> = to_remove.chunks(aliases_per_thread.max(1)).map(|c| c.to_vec()).collect();

    for chunk in to_remove_chunks {
        let catalog_clone = catalog.clone();
        let handle = thread::spawn(move || {
            for alias in chunk {
                // Delete alias from catalog
                let deleted = catalog_clone.delete_alias(&alias.alias_id).unwrap();
                assert!(deleted);

                // Remove reference
                let ref_id = format!("alias:{}", alias.alias_id);
                let _ = catalog_clone.remove_blob_reference(&ref_id);
            }
        });
        remove_handles.push(handle);
    }

    for handle in remove_handles {
        handle.join().expect("remove thread join succeeded");
    }

    // Verify updated ref_count
    let expected_after_remove = expected_refs - (remove_count as u32);
    let blob_after_remove = catalog.get_blob(&weight_digest).unwrap().unwrap();
    assert_eq!(
        blob_after_remove.ref_count, expected_after_remove,
        "ref_count must decrease deterministically after reference removal"
    );

    // Verify recalculate_ref_counts produces identical invariant
    catalog.recalculate_ref_counts().unwrap();
    let blob_after_recalc = catalog.get_blob(&weight_digest).unwrap().unwrap();
    assert_eq!(
        blob_after_recalc.ref_count, expected_after_remove,
        "recalculate_ref_counts must maintain invariant"
    );
}

// ============================================================================
// 3. Capability Filtering Stress Tests
// ============================================================================

#[test]
fn test_capability_filtering_uncompiled_recipes_and_revisions() {
    // A. Architecture revisions that are not compiled into the runtime
    let uncompiled_test_cases = [
        ("qwen3.8", "1.0.0", "Q4_K_M"),
        ("qwen3.5", "1.0.0", "Q4_K_M"),
        ("llama-3.2", "1.0.0", "Q4_K_M"),
        ("deepseek-v3", "1.0.0", "Q4_K_M"),
        ("mistral", "1.0.0", "Q4_K_M"),
        ("synthetic", "2.0.0", "Q4_K_M"), // synthetic is compiled at 1.0.0 only
        ("qwen3", "0.9.0", "Q4_K_M"),     // older recipe version
        ("qwen3", "1.1.0", "Q4_K_M"),     // newer recipe version
        ("qwen3", "2.0.0", "Q4_K_M"),
        ("llama", "0.1.0", "Q4_K_M"),
        ("llama", "3.0.0", "Q4_K_M"),
        ("", "1.0.0", "Q4_K_M"),          // empty architecture
        ("unknown_architecture_xyz", "1.0.0", "Q4_K_M"),
    ];

    for (arch, recipe, quant) in uncompiled_test_cases {
        let manifest = ModelManifest {
            schema_version: 1,
            format: "gguf".to_owned(),
            architecture_id: arch.to_owned(),
            recipe_version: recipe.to_owned(),
            quantization: quant.to_owned(),
            context_length: 2048,
            license: None,
            origin: None,
            weight_digest: "1111111111111111111111111111111111111111111111111111111111111111".to_owned(),
            weight_size_bytes: 1024,
            created_at: 100,
        };

        // 1. evaluate_candidate must flag non-executable
        let eval = CapabilityFilter::evaluate_candidate(&manifest);
        assert!(
            !eval.executable,
            "architecture '{arch}' with recipe '{recipe}' must be evaluated as non-executable"
        );
        assert!(eval.reason.is_some());
        let reason = eval.reason.unwrap();
        assert!(
            reason.contains("not compiled into this runtime build"),
            "reason should explain lack of compiled recipe: {reason}"
        );

        // 2. validate_model_manifest must return upfront K3Error::CapabilityMismatch
        let err = CapabilityFilter::validate_model_manifest(&manifest)
            .expect_err("validate_model_manifest must reject uncompiled architecture upfront");

        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
        assert_eq!(err.diagnostic_code(), "capability_mismatch");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("not compiled into this runtime build"),
            "error message must mention not compiled: {err_msg}"
        );

        // 3. align_prompt_environment must reject uncompiled recipe
        let align_err = CapabilityFilter::align_prompt_environment(arch, recipe)
            .expect_err("align_prompt_environment must fail for uncompiled recipe");
        assert!(matches!(align_err, K3Error::CapabilityMismatch { .. }));
        assert_eq!(align_err.diagnostic_code(), "capability_mismatch");
    }
}

#[test]
fn test_capability_filtering_invalid_quants_and_formats() {
    // Valid architectures with invalid / unsupported quants
    let invalid_quant_cases = [
        ("qwen3", "1.0.0", "Q2_K"),
        ("qwen3", "1.0.0", "IQ1_S"),
        ("qwen3", "1.0.0", "EXL2"),
        ("qwen3", "1.0.0", "INT4_AWQ"),
        ("qwen3", "1.0.0", "Q8_0"),      // qwen3 compiled only supports Q4_K_M, F32, F16, BF16
        ("llama", "1.0.0", "Q3_K_M"),
        ("synthetic", "1.0.0", "Q5_K_M"),
        ("synthetic", "1.0.0", ""),      // empty quant string
        ("synthetic", "1.0.0", "UNKNOWN_QUANT_XYZ"),
    ];

    for (arch, recipe, quant) in invalid_quant_cases {
        let manifest = ModelManifest {
            schema_version: 1,
            format: "gguf".to_owned(),
            architecture_id: arch.to_owned(),
            recipe_version: recipe.to_owned(),
            quantization: quant.to_owned(),
            context_length: 2048,
            license: None,
            origin: None,
            weight_digest: "2222222222222222222222222222222222222222222222222222222222222222".to_owned(),
            weight_size_bytes: 1024,
            created_at: 100,
        };

        let eval = CapabilityFilter::evaluate_candidate(&manifest);
        assert!(!eval.executable);
        assert!(!eval.quantization_supported);
        assert!(eval.reason.unwrap().contains("quantization format"));

        let err = CapabilityFilter::validate_model_manifest(&manifest)
            .expect_err("unsupported quant must be rejected upfront");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
        assert_eq!(err.diagnostic_code(), "capability_mismatch");
        assert!(err.to_string().contains("quantization format"));
    }

    // Invalid format test cases (e.g. onnx, pt, torchscript)
    let invalid_formats = ["onnx", "pt", "torchscript", "bin", "h5", "", "unknown_format"];
    for fmt in invalid_formats {
        let manifest = ModelManifest {
            schema_version: 1,
            format: fmt.to_owned(),
            architecture_id: "synthetic".to_owned(),
            recipe_version: "1.0.0".to_owned(),
            quantization: "Q4_K_M".to_owned(),
            context_length: 2048,
            license: None,
            origin: None,
            weight_digest: "3333333333333333333333333333333333333333333333333333333333333333".to_owned(),
            weight_size_bytes: 1024,
            created_at: 100,
        };

        let err = CapabilityFilter::validate_model_manifest(&manifest)
            .expect_err("unsupported format must fail upfront");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
        assert!(err.to_string().contains("unsupported model format"));
    }

    // Invalid context length = 0
    let zero_ctx_manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 0, // Invalid!
        license: None,
        origin: None,
        weight_digest: "4444444444444444444444444444444444444444444444444444444444444444".to_owned(),
        weight_size_bytes: 1024,
        created_at: 100,
    };

    let eval_zero = CapabilityFilter::evaluate_candidate(&zero_ctx_manifest);
    assert!(!eval_zero.executable);
    assert!(eval_zero.reason.unwrap().contains("context length must be greater than 0"));

    let err_zero = CapabilityFilter::validate_model_manifest(&zero_ctx_manifest)
        .expect_err("zero context length must fail upfront");
    assert!(matches!(err_zero, K3Error::CapabilityMismatch { .. }));
    assert!(err_zero.to_string().contains("context length"));
}

#[test]
fn test_capability_filtering_tokenizer_alignment_and_stop_sequences() {
    // Test aligned prompt environments across compiled recipes
    let compiled = [("synthetic", "1.0.0"), ("qwen3", "1.0.0"), ("llama", "1.0.0")];

    for (arch, ver) in compiled {
        let env = CapabilityFilter::align_prompt_environment(arch, ver)
            .unwrap_or_else(|e| panic!("failed to align prompt env for {arch}@{ver}: {e}"));

        assert_eq!(env.architecture_id, arch);
        assert_eq!(env.recipe_version, ver);

        // Verify prompt formatting is deterministic
        let prompt = "How are you?";
        let formatted = env.chat_template.format_prompt(prompt);
        assert!(!formatted.is_empty());
        assert!(formatted.contains(prompt));

        // Dynamically register words present in formatted prompt into SimpleTokenizer
        let mut tokenizer = SimpleTokenizer::new();
        for (i, word) in formatted.split_whitespace().enumerate() {
            tokenizer.add_token(word, i as u32 + 100);
        }

        // Format and tokenize within bounds
        let tokens = env.format_and_tokenize(&tokenizer, prompt, 1024).expect("tokenize");
        assert!(!tokens.is_empty());

        // Format and tokenize exceeding bounds returns ContextLengthExceeded
        let err = env
            .format_and_tokenize(&tokenizer, prompt, tokens.len())
            .expect_err("exceeding tokens must fail");
        assert!(matches!(err, K3Error::ContextLengthExceeded { .. }));
    }
}

// ============================================================================
// 4. Derived Weight Layout Caching, Retention & Stale Recipe Stress Tests
// ============================================================================

#[test]
fn test_derived_weights_cache_concurrency_and_deduplication() {
    let (store, _catalog, _tmp) = setup_cas_store();
    let manager = DerivedBlobManager::new(Arc::new(store.clone()));

    let source_content = b"heavy_raw_weights_for_concurrent_derived_test";
    let (source_digest, _path) = register_dummy_blob(&store, source_content);

    // Spawn 8 concurrent threads attempting to derive the SAME layout at the same time
    let num_threads = 8;
    let mut handles = Vec::new();

    for t_idx in 0..num_threads {
        let manager_clone = manager.clone();
        let s_dig = source_digest.clone();

        let handle = thread::spawn(move || {
            let converter = |_src: &Path, dst: &Path| -> Result<u64, K3Error> {
                // Simulate some work
                thread::sleep(std::time::Duration::from_millis(5));
                let mut data = Vec::new();
                data.extend_from_slice(b"IME_TILE_8x16_DERIVED:");
                data.extend_from_slice(b"heavy_raw_weights_for_concurrent_derived_test");
                fs::write(dst, &data).map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "derived_test".to_owned(),
                    reason: e.to_string(),
                })?;
                Ok(data.len() as u64)
            };

            manager_clone.get_or_create_derived_layout(
                &s_dig,
                "1.0.0",
                "ime_tile_8x16",
                RetentionPolicy::Keep,
                converter,
            )
        });
        handles.push((t_idx, handle));
    }

    let mut first_digest: Option<String> = None;
    for (t_idx, handle) in handles {
        let res = handle
            .join()
            .expect("thread join")
            .unwrap_or_else(|e| panic!("thread {t_idx} failed: {e}"));

        let digest = match res {
            DerivedResolution::Created { derived_digest, derived_path } => {
                assert!(derived_path.exists());
                derived_digest
            }
            DerivedResolution::Cached { derived_digest, derived_path } => {
                assert!(derived_path.exists());
                derived_digest
            }
        };

        if let Some(ref fd) = first_digest {
            assert_eq!(fd, &digest, "all threads must receive identical derived digest");
        } else {
            first_digest = Some(digest);
        }
    }

    // Exactly 2 blobs on disk: 1 source + 1 derived
    let files: Vec<_> = fs::read_dir(store.blobs_dir()).unwrap().flatten().collect();
    assert_eq!(files.len(), 2, "no duplicate derived blobs should be created on disk");
}

#[test]
fn test_derived_weights_retention_policies_edge_cases() {
    let (store, catalog, _tmp) = setup_cas_store();
    let manager = DerivedBlobManager::new(Arc::new(store.clone()));

    // 1. PurgeImmediate with source referenced by manifest
    let source_1 = b"source_payload_purge_immediate";
    let (digest_1, _) = register_dummy_blob(&store, source_1);
    let _man_1 = register_dummy_manifest(&store, &digest_1, "synthetic", "1.0.0", "F32");

    let initial_ref = catalog.get_blob(&digest_1).unwrap().unwrap().ref_count;
    assert_eq!(initial_ref, 1);

    let res_imm = manager
        .get_or_create_derived_layout(
            &digest_1,
            "1.0.0",
            "ime_tile_8x16",
            RetentionPolicy::PurgeImmediate,
            |_src, dst| {
                fs::write(dst, b"tiled_1").unwrap();
                Ok(7)
            },
        )
        .expect("derive with purge immediate");
    assert!(matches!(res_imm, DerivedResolution::Created { .. }));

    // Storage config must have purge_immediate recorded
    let cfg = catalog.get_storage_config(&format!("purge_immediate:{digest_1}")).unwrap();
    assert_eq!(cfg, Some("true".to_owned()));

    // ref_count decremented from 1 to 0
    let after_ref = catalog.get_blob(&digest_1).unwrap().unwrap().ref_count;
    assert_eq!(after_ref, 0);

    // 2. PurgeAfterDelay with zero delay (boundary condition: immediate expiration)
    let source_2 = b"source_payload_purge_after_zero_delay";
    let (digest_2, _) = register_dummy_blob(&store, source_2);

    let res_delay_zero = manager
        .get_or_create_derived_layout(
            &digest_2,
            "1.0.0",
            "ime_tile_8x16",
            RetentionPolicy::PurgeAfterDelay { delay_secs: 0 },
            |_src, dst| {
                fs::write(dst, b"tiled_2").unwrap();
                Ok(7)
            },
        )
        .expect("derive with delay 0");
    assert!(matches!(res_delay_zero, DerivedResolution::Created { .. }));

    let cfg_delay_zero = catalog.get_storage_config(&format!("purge_after:{digest_2}")).unwrap();
    assert!(cfg_delay_zero.is_some());
    let ts: u64 = cfg_delay_zero.unwrap().parse().unwrap();
    assert!(ts > 0);

    // 3. PurgeAfterDelay with large delay (e.g. 1 year = 31536000s)
    let source_3 = b"source_payload_purge_after_large_delay";
    let (digest_3, _) = register_dummy_blob(&store, source_3);

    let res_delay_large = manager
        .get_or_create_derived_layout(
            &digest_3,
            "1.0.0",
            "ime_tile_8x16",
            RetentionPolicy::PurgeAfterDelay { delay_secs: 31_536_000 },
            |_src, dst| {
                fs::write(dst, b"tiled_3").unwrap();
                Ok(7)
            },
        )
        .expect("derive with delay 31536000");
    assert!(matches!(res_delay_large, DerivedResolution::Created { .. }));

    let cfg_delay_large = catalog.get_storage_config(&format!("purge_after:{digest_3}")).unwrap();
    assert!(cfg_delay_large.is_some());
    let ts_large: u64 = cfg_delay_large.unwrap().parse().unwrap();
    assert!(ts_large > ts);
}

#[test]
fn test_stale_recipe_detection_auto_rebuild_and_advisory() {
    let (store, catalog, _tmp) = setup_cas_store();
    let manager = DerivedBlobManager::new(Arc::new(store.clone()));

    let source_content = b"original_source_for_stale_upgrade_testing";
    let (source_digest, source_path) = register_dummy_blob(&store, source_content);

    // 1. Initial build under recipe "1.0.0" for layout "ime_tile_8x16"
    let _initial = manager
        .get_or_create_derived_layout(
            &source_digest,
            "1.0.0",
            "ime_tile_8x16",
            RetentionPolicy::Keep,
            |_src, dst| {
                fs::write(dst, b"derived_v1_bytes").unwrap();
                Ok(16)
            },
        )
        .expect("initial v1 build");

    // 2. Querying when active recipe is "1.0.0" -> UpToDate
    let outcome_uptodate = manager
        .detect_and_handle_stale_recipe(
            &source_digest,
            "1.0.0",
            "ime_tile_8x16",
            "test_model",
            |_src, _dst| panic!("rebuilder must not be called when up-to-date"),
        )
        .expect("check up-to-date");

    assert!(matches!(outcome_uptodate, StaleRecipeOutcome::UpToDate { .. }));

    // 3. Upgraded runtime with active recipe "2.0.0", source weights present -> Rebuilt
    let mut rebuild_called = false;
    let outcome_rebuild = manager
        .detect_and_handle_stale_recipe(
            &source_digest,
            "2.0.0",
            "ime_tile_8x16",
            "test_model",
            |_src, dst| {
                rebuild_called = true;
                fs::write(dst, b"derived_v2_rebuilt_bytes").unwrap();
                Ok(24)
            },
        )
        .expect("stale recipe rebuild");

    assert!(rebuild_called, "rebuilder must have been executed");
    match outcome_rebuild {
        StaleRecipeOutcome::Rebuilt { new_derived_digest, new_derived_path } => {
            assert!(new_derived_path.exists());
            assert_eq!(new_derived_digest, Sha256::digest(b"derived_v2_rebuilt_bytes"));
        }
        _ => panic!("expected StaleRecipeOutcome::Rebuilt"),
    }

    // Verify DuckDB catalog now has "stale" on v1 and "valid" on v2
    let derived_list = catalog.list_derived_for_source(&source_digest).unwrap();
    assert_eq!(derived_list.len(), 2);
    let v1 = derived_list.iter().find(|d| d.recipe_version == "1.0.0").unwrap();
    let v2 = derived_list.iter().find(|d| d.recipe_version == "2.0.0").unwrap();
    assert_eq!(v1.status, "stale");
    assert_eq!(v2.status, "valid");

    // 4. Upgraded runtime to "3.0.0", but source weights are now purged/deleted -> SourcePurgedAdvisory
    fs::remove_file(&source_path).expect("delete source weights");
    assert!(!source_path.exists());

    let outcome_purged = manager
        .detect_and_handle_stale_recipe(
            &source_digest,
            "3.0.0",
            "ime_tile_8x16",
            "my_favorite_qwen_model",
            |_src, _dst| panic!("rebuilder must never be called when source is purged"),
        )
        .expect("stale with purged source");

    match outcome_purged {
        StaleRecipeOutcome::SourcePurgedAdvisory {
            source_digest: s_dig,
            stale_recipe_version,
            active_recipe_version,
            advisory_message,
        } => {
            assert_eq!(s_dig, source_digest);
            assert_eq!(stale_recipe_version, "1.0.0");
            assert_eq!(active_recipe_version, "3.0.0");
            assert!(advisory_message.contains("my_favorite_qwen_model"));
            assert!(advisory_message.contains("Please re-acquire the source model"));
        }
        _ => panic!("expected StaleRecipeOutcome::SourcePurgedAdvisory"),
    }
}

#[test]
fn test_derived_weights_failed_conversion_cleanup() {
    let (store, _catalog, _tmp) = setup_cas_store();
    let manager = DerivedBlobManager::new(Arc::new(store.clone()));

    let source_content = b"original_source_for_failing_conversion";
    let (source_digest, _) = register_dummy_blob(&store, source_content);

    // Initial count of staging files
    let initial_staging = fs::read_dir(store.staging_dir()).unwrap().count();

    // Converter fails midway
    let failing_converter = |_src: &Path, _dst: &Path| -> Result<u64, K3Error> {
        Err(K3Error::CapabilityMismatch {
            architecture: "converter".to_owned(),
            reason: "simulated out of memory or kernel trap".to_owned(),
        })
    };

    let err = manager
        .get_or_create_derived_layout(
            &source_digest,
            "1.0.0",
            "ime_tile_8x16",
            RetentionPolicy::Keep,
            failing_converter,
        )
        .expect_err("failing conversion must return error");

    assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    assert!(err.to_string().contains("simulated out of memory"));

    // Verify staging directory is cleaned up and no leftover .part files exist
    let after_staging = fs::read_dir(store.staging_dir()).unwrap().count();
    assert_eq!(
        initial_staging, after_staging,
        "staging files must be cleaned up on failed conversion"
    );
}

#[test]
fn test_derived_memory_admission_limits() {
    let available_ram = 32 * 1024 * 1024 * 1024; // 32 GiB
    let rlimit_memlock = 28 * 1024 * 1024 * 1024; // 28 GiB

    // Case 1: Within limit
    assert!(
        check_derived_memory_admission(
            8 * 1024 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
            available_ram,
            rlimit_memlock,
        )
        .is_ok()
    );

    // Case 2: Exceeding rlimit_memlock
    let err_memlock = check_derived_memory_admission(
        15 * 1024 * 1024 * 1024,
        15 * 1024 * 1024 * 1024,
        2 * 1024 * 1024 * 1024,
        available_ram,
        rlimit_memlock,
    )
    .expect_err("must reject when exceeding memlock limit");

    assert!(matches!(err_memlock, K3Error::InsufficientLockableMemory { .. }));
    assert_eq!(err_memlock.diagnostic_code(), "insufficient_lockable_memory");

    // Case 3: Exceeding available RAM
    let tight_ram = 16 * 1024 * 1024 * 1024;
    let err_ram = check_derived_memory_admission(
        8 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
        2 * 1024 * 1024 * 1024,
        tight_ram,
        rlimit_memlock,
    )
    .expect_err("must reject when exceeding available RAM");

    assert!(matches!(err_ram, K3Error::InsufficientLockableMemory { .. }));
}

#[test]
fn test_alias_creation_nonexistent_manifest_and_duplicate_alias() {
    let (store, catalog, _tmp) = setup_cas_store();

    // 1. Alias creation pointing to non-existent manifest must fail upfront
    let fake_manifest = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    let err = store
        .create_alias("local", "ns", "model", "v1", fake_manifest)
        .expect_err("nonexistent manifest must be rejected");

    assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    assert!(err.to_string().contains("manifest 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff' not found"));

    // 2. Duplicate alias registration updates/idempotently replaces the existing alias
    let (weight_digest, _) = register_dummy_blob(&store, b"weight_payload_alias_dup");
    let man_rec = register_dummy_manifest(&store, &weight_digest, "synthetic", "1.0.0", "F32");

    let alias1 = store
        .create_alias("local", "ns", "model", "v1", &man_rec.manifest_digest)
        .expect("create alias 1");

    // Second registration with same alias path succeeds via SQL upsert (ON CONFLICT DO UPDATE SET)
    let alias2 = store
        .create_alias("local", "ns", "model", "v1", &man_rec.manifest_digest)
        .expect("create alias 2 overwrite");

    assert_eq!(alias1.alias_id, alias2.alias_id);

    // Verify resolve_alias works cleanly
    let resolved = store.resolve_alias("local/ns/model/v1").expect("resolve alias");
    assert_eq!(resolved.weight_path, store.blobs_dir().join(&weight_digest));

    // Ref count should remain consistent (manifest ref + alias ref = 2)
    let blob = catalog.get_blob(&weight_digest).unwrap().unwrap();
    assert_eq!(blob.ref_count, 2);
}

#[test]
fn test_derived_weights_missing_source_and_multi_target_isolation() {
    let (store, catalog, _tmp) = setup_cas_store();
    let manager = DerivedBlobManager::new(Arc::new(store.clone()));

    // 1. Derivation on nonexistent source blob fails upfront
    let missing_digest = "9999999999999999999999999999999999999999999999999999999999999999";
    let err = manager
        .get_or_create_derived_layout(
            missing_digest,
            "1.0.0",
            "ime_tile_8x16",
            RetentionPolicy::Keep,
            |_src, dst| {
                fs::write(dst, b"derived").unwrap();
                Ok(7)
            },
        )
        .expect_err("missing source weights must be rejected");

    assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    assert!(err.to_string().contains("not found in store; cannot derive layout"));

    // 2. Multi-target layout isolation on same source weights
    let source_content = b"original_source_for_multi_layout_testing";
    let (source_digest, _) = register_dummy_blob(&store, source_content);

    // Layout A: "ime_tile_8x16"
    let res_a = manager
        .get_or_create_derived_layout(
            &source_digest,
            "1.0.0",
            "ime_tile_8x16",
            RetentionPolicy::Keep,
            |_src, dst| {
                fs::write(dst, b"layout_a_ime").unwrap();
                Ok(12)
            },
        )
        .expect("derive layout A");

    // Layout B: "native"
    let res_b = manager
        .get_or_create_derived_layout(
            &source_digest,
            "1.0.0",
            "native",
            RetentionPolicy::Keep,
            |_src, dst| {
                fs::write(dst, b"layout_b_native").unwrap();
                Ok(15)
            },
        )
        .expect("derive layout B");

    let digest_a = match res_a {
        DerivedResolution::Created { derived_digest, .. } => derived_digest,
        _ => panic!("expected Created"),
    };
    let digest_b = match res_b {
        DerivedResolution::Created { derived_digest, .. } => derived_digest,
        _ => panic!("expected Created"),
    };

    assert_ne!(digest_a, digest_b, "different target layouts must produce distinct blobs");

    // Query both layouts independently from catalog
    let record_a = catalog.get_derived_blob(&source_digest, "1.0.0", "ime_tile_8x16").unwrap().unwrap();
    let record_b = catalog.get_derived_blob(&source_digest, "1.0.0", "native").unwrap().unwrap();
    assert_eq!(record_a.derived_digest, digest_a);
    assert_eq!(record_b.derived_digest, digest_b);

    // When upgrading recipe for layout A, layout B should remain unaffected
    let mut rebuild_ran = false;
    let outcome = manager
        .detect_and_handle_stale_recipe(
            &source_digest,
            "2.0.0",
            "ime_tile_8x16",
            "test_model",
            |_src, dst| {
                rebuild_ran = true;
                fs::write(dst, b"layout_a_v2").unwrap();
                Ok(11)
            },
        )
        .expect("upgrade layout A");

    assert!(rebuild_ran);
    assert!(matches!(outcome, StaleRecipeOutcome::Rebuilt { .. }));

    // Verify layout B is still valid and unchanged
    let record_b_after = catalog.get_derived_blob(&source_digest, "1.0.0", "native").unwrap().unwrap();
    assert_eq!(record_b_after.status, "valid");
    assert_eq!(record_b_after.recipe_version, "1.0.0");
}

#[test]
fn test_cas_catalog_extreme_accounting_and_blob_sizes() {
    let (_store, catalog, _tmp) = setup_cas_store();

    // Test large blob sizes (e.g. 70B parameter model = 140 GiB)
    let huge_size: u64 = 140 * 1024 * 1024 * 1024;
    let blob_huge = ModelBlobRecord {
        digest: "huge_model_digest_140gib".to_owned(),
        size_bytes: huge_size,
        blob_type: "raw_weights".to_owned(),
        created_at: 100,
        last_accessed_at: 200,
        ref_count: 0,
    };
    catalog.save_blob(&blob_huge).unwrap();

    let fetched = catalog.get_blob("huge_model_digest_140gib").unwrap().unwrap();
    assert_eq!(fetched.size_bytes, huge_size, "u64 sizes must not truncate in DuckDB");

    let total = catalog.total_blob_bytes().unwrap();
    assert!(total >= huge_size, "total_blob_bytes must handle 64-bit sums correctly");
}

