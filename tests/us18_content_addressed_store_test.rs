//! Scenario and unit verification for US-18: Content-Addressed Model Store & Human-Browsable Registry.
//!
//! Verifies:
//! - Scenario 1: Cross-source weight deduplication without byte duplication.
//! - Scenario 2: Configurable store root location directing assets to dedicated volumes.
//! - Scenario 3: Human-browsable relative symlink alias inspection.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use k3_manager::persistence::DuckDbPersistenceStore;
use k3_manager::store::{ContentAddressedStore, ENV_STORE_ROOT, ModelManifest, Sha256};

static GLOBAL_SEQ: AtomicU64 = AtomicU64::new(1);

struct TestTempDir {
    path: PathBuf,
}

impl TestTempDir {
    fn new(prefix: &str) -> Self {
        let id = GLOBAL_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{id}", std::process::id()));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn create_temp_store(name: &str) -> (ContentAddressedStore, TestTempDir) {
    let temp_dir = TestTempDir::new(&format!("k3_test_us18_{name}"));
    let catalog = DuckDbPersistenceStore::open_in_memory().expect("open duckdb catalog");
    let store = ContentAddressedStore::open(temp_dir.path(), Arc::new(catalog))
        .expect("open content-addressed store");
    (store, temp_dir)
}

fn create_dummy_gguf_bytes(content: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes()); // version 3
    bytes.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    bytes.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata kv
    bytes.extend_from_slice(content);
    bytes
}

#[test]
fn test_scenario1_cross_source_weight_deduplication() {
    let (store, _tmp) = create_temp_store("dedup");

    // 1. Create a model weight file representing Source A (HuggingFace)
    let weight_bytes = create_dummy_gguf_bytes(b"common_model_weights_payload_12345");
    let expected_digest = Sha256::digest(&weight_bytes);

    let (staging1, mut f1) = store.create_staging_file().expect("staging file 1");
    f1.write_all(&weight_bytes).expect("write staging 1");
    f1.flush().expect("flush staging 1");
    drop(f1);

    let promoted_path1 = store
        .promote_staged_blob(&staging1, &expected_digest, "gguf")
        .expect("promote 1");
    assert!(promoted_path1.exists());

    // Register manifest for Source A
    let manifest_a = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: Some("Apache-2.0".to_owned()),
        origin: Some("https://huggingface.co/Qwen/Qwen2.5-7B".to_owned()),
        weight_digest: expected_digest.clone(),
        weight_size_bytes: weight_bytes.len() as u64,
        created_at: 1000,
    };
    let manifest_rec_a = store
        .register_manifest(&manifest_a)
        .expect("register manifest A");

    // Create alias for Source A: huggingface/Qwen/Qwen2.5-7B/latest
    let alias_a = store
        .create_alias(
            "huggingface",
            "Qwen",
            "Qwen2.5-7B",
            "latest",
            &manifest_rec_a.manifest_digest,
        )
        .expect("create alias A");
    assert_eq!(alias_a.weight_digest, expected_digest);

    // Verify initial storage state: exactly 1 physical blob
    let blobs_after_a = store.catalog().list_blobs().expect("list blobs");
    assert_eq!(blobs_after_a.len(), 1);
    assert_eq!(blobs_after_a[0].digest, expected_digest);
    assert_eq!(blobs_after_a[0].ref_count, 2); // 1 manifest + 1 alias

    // 2. Ingest identical weights from Source B (Ollama) under a new tag
    let (staging2, mut f2) = store.create_staging_file().expect("staging file 2");
    f2.write_all(&weight_bytes).expect("write staging 2");
    f2.flush().expect("flush staging 2");
    drop(f2);

    let promoted_path2 = store
        .promote_staged_blob(&staging2, &expected_digest, "gguf")
        .expect("promote 2 (deduplicated)");
    assert_eq!(promoted_path1, promoted_path2);

    // Register manifest for Source B
    let manifest_b = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: Some("Apache-2.0".to_owned()),
        origin: Some("ollama://qwen2.5:7b".to_owned()),
        weight_digest: expected_digest.clone(),
        weight_size_bytes: weight_bytes.len() as u64,
        created_at: 2000,
    };
    let manifest_rec_b = store
        .register_manifest(&manifest_b)
        .expect("register manifest B");

    // Create alias for Source B: ollama/library/qwen2.5/7b
    let alias_b = store
        .create_alias(
            "ollama",
            "library",
            "qwen2.5",
            "7b",
            &manifest_rec_b.manifest_digest,
        )
        .expect("create alias B");
    assert_eq!(alias_b.weight_digest, expected_digest);

    // 3. Assert deduplication invariant:
    // Only one file on disk in blobs/sha256/
    let physical_blob_files: Vec<_> = fs::read_dir(store.blobs_dir())
        .expect("read blobs dir")
        .flatten()
        .collect();
    assert_eq!(
        physical_blob_files.len(),
        1,
        "Storage must retain exactly 1 physical blob file across distinct sources"
    );

    // 2 distinct manifests in DuckDB and on disk
    let manifests = store.catalog().list_manifests().expect("list manifests");
    assert_eq!(manifests.len(), 2);

    // 2 distinct aliases in DuckDB
    let aliases = store.catalog().list_aliases().expect("list aliases");
    assert_eq!(aliases.len(), 2);

    // DuckDB ref count reflects shared usage (2 manifests + 2 aliases = 4 refs)
    let updated_blob = store
        .catalog()
        .get_blob(&expected_digest)
        .expect("get blob")
        .expect("blob must exist");
    assert_eq!(updated_blob.ref_count, 4);
}

#[test]
fn test_scenario2_configurable_store_root_location() {
    let custom_secondary_nvme = TestTempDir::new("secondary_nvme_tier");

    // Set environment variable override K3_STORE_ROOT
    // SAFETY: Single-threaded test execution context setting temporary test env var
    unsafe {
        std::env::set_var(ENV_STORE_ROOT, custom_secondary_nvme.path());
    }

    let catalog = DuckDbPersistenceStore::open_in_memory().expect("open catalog");
    let store = ContentAddressedStore::open_default(Arc::new(catalog))
        .expect("open default store from env");

    assert_eq!(store.root(), custom_secondary_nvme.path());
    assert!(store.blobs_dir().starts_with(custom_secondary_nvme.path()));
    assert!(
        store
            .manifests_dir()
            .starts_with(custom_secondary_nvme.path())
    );
    assert!(store.models_dir().starts_with(custom_secondary_nvme.path()));
    assert!(
        store
            .staging_dir()
            .starts_with(custom_secondary_nvme.path())
    );

    // Verify all file operations happen exclusively under the configured secondary volume
    let (staging, mut f) = store.create_staging_file().expect("staging file");
    assert!(staging.starts_with(custom_secondary_nvme.path()));
    let payload = create_dummy_gguf_bytes(b"nvme_tier_payload");
    let digest = Sha256::digest(&payload);
    f.write_all(&payload).expect("write");
    drop(f);

    let promoted = store
        .promote_staged_blob(&staging, &digest, "gguf")
        .expect("promote");
    assert!(promoted.starts_with(custom_secondary_nvme.path()));

    // Clean up environment variable
    // SAFETY: Restoring test environment variable
    unsafe {
        std::env::remove_var(ENV_STORE_ROOT);
    }
}

#[test]
fn test_scenario3_human_browsable_model_inspection() {
    let (store, _tmp) = create_temp_store("browsable");

    let payload = create_dummy_gguf_bytes(b"human_browsable_test_model");
    let digest = Sha256::digest(&payload);

    let (staging, mut f) = store.create_staging_file().expect("staging");
    f.write_all(&payload).expect("write");
    drop(f);

    store
        .promote_staged_blob(&staging, &digest, "gguf")
        .expect("promote");

    let manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 4096,
        license: Some("MIT".to_owned()),
        origin: Some("local://test".to_owned()),
        weight_digest: digest.clone(),
        weight_size_bytes: payload.len() as u64,
        created_at: 12345,
    };
    let man_rec = store.register_manifest(&manifest).expect("manifest");

    let _alias = store
        .create_alias(
            "local",
            "testing",
            "dummy-model",
            "v1.0",
            &man_rec.manifest_digest,
        )
        .expect("create alias");

    // 1. Verify directory hierarchy on disk
    let alias_dir = store
        .models_dir()
        .join("local")
        .join("testing")
        .join("dummy-model")
        .join("v1.0");
    assert!(alias_dir.exists(), "alias directory must exist on disk");

    let manifest_symlink = alias_dir.join("manifest.json");
    let weights_symlink = alias_dir.join("weights");
    assert!(
        manifest_symlink.exists(),
        "manifest.json symlink must exist"
    );
    assert!(weights_symlink.exists(), "weights symlink must exist");

    // 2. Verify relative symlink resolution
    #[cfg(unix)]
    {
        let manifest_target = fs::read_link(&manifest_symlink).expect("read manifest link");
        let weights_target = fs::read_link(&weights_symlink).expect("read weights link");

        assert!(
            manifest_target
                .to_string_lossy()
                .starts_with("../../../../../manifests/sha256/"),
            "manifest symlink must use relative path pointing inward"
        );
        assert!(
            weights_target
                .to_string_lossy()
                .starts_with("../../../../../blobs/sha256/"),
            "weights symlink must use relative path pointing inward"
        );
    }

    // 3. Resolve alias via store API
    let resolved = store
        .resolve_alias("local/testing/dummy-model/v1.0")
        .expect("resolve alias");
    assert_eq!(resolved.alias_id, "local/testing/dummy-model/v1.0");
    assert_eq!(resolved.manifest.weight_digest, digest);
    assert_eq!(resolved.manifest.context_length, 4096);
    assert_eq!(resolved.weight_path, store.blobs_dir().join(&digest));
    assert!(resolved.weight_path.exists());
    assert!(resolved.manifest_path.exists());
}
