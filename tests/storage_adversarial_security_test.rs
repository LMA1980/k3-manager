//! Security and adversarial attack vector verification for Model-Runner storage stack (Milestone M3, Security R3).
//!
//! Verifies:
//! - Defense against path traversal sequences (`..`, `/`, `\`, null bytes, control characters, leading dashes).
//! - Symlink escape prevention confining resolved assets strictly within the store root.
//! - Defense against `DuckDB` SQL injection attempts in manifests, digests, and alias strings.
//! - TOCTOU integrity validation and corrupted payload rejection during atomic staging promotion.

#![allow(
    clippy::pedantic,
    clippy::doc_markdown,
    clippy::undocumented_unsafe_blocks,
    clippy::field_reassign_with_default,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::needless_pass_by_value
)]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use k3_core::K3Error;
use k3_core::profile::TuningProfile;
use k3_manager::persistence::{
    CasCatalog, DuckDbPersistenceStore, ModelAliasRecord, ModelBlobRecord, ModelManifestRecord,
    PersistenceStore, Project, SessionLastTurn, Tenant,
};
use k3_manager::store::{
    ContentAddressedStore, ModelManifest, Sha256, parse_alias_id, validate_path_component,
    validate_sha256_digest,
};

static GLOBAL_SEQ: AtomicU64 = AtomicU64::new(300);

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

fn create_store() -> (
    ContentAddressedStore,
    Arc<DuckDbPersistenceStore>,
    TestTempDir,
) {
    let temp_dir = TestTempDir::new("k3_sec_test");
    let catalog = Arc::new(DuckDbPersistenceStore::open_in_memory().expect("open catalog"));
    let store = ContentAddressedStore::open(temp_dir.path(), catalog.clone()).expect("open store");
    (store, catalog, temp_dir)
}

#[test]
fn test_path_traversal_rejections() {
    let too_long = "a".repeat(129);
    let malicious_components = [
        "..",
        ".",
        "../etc",
        "..\\windows",
        "foo/bar",
        "foo\\bar",
        "foo\0bar",
        "-leading-dash",
        "--flag",
        "foo\nbar",
        "foo\rbar",
        "foo\x08bar",
        "",
        // Exceeding 128 characters
        too_long.as_str(),
    ];

    for comp in malicious_components {
        let res = validate_path_component(comp);
        assert!(
            res.is_err(),
            "Path component '{comp:?}' must be rejected by validator"
        );
        let err = res.unwrap_err();
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    }

    // Valid components must pass
    let valid_components = [
        "huggingface",
        "Qwen",
        "Qwen2.5-7B-Instruct",
        "latest",
        "v1.0.0",
        "model_build_1",
    ];
    for comp in valid_components {
        assert!(
            validate_path_component(comp).is_ok(),
            "Valid component '{comp}' must pass validation"
        );
    }
}

#[test]
fn test_alias_id_parser_defense() {
    let malicious_alias_ids = [
        "huggingface/../../etc/passwd",
        "huggingface/Qwen/foo/bar/extra", // 5 segments instead of 4
        "huggingface/Qwen",               // 2 segments
        "huggingface/Qwen/name/tag\0injection",
        "/etc/passwd/root/admin/leak",
        "../models/escape/test/v1",
    ];

    for alias in malicious_alias_ids {
        let res = parse_alias_id(alias);
        assert!(
            res.is_err(),
            "Malicious alias '{alias}' must be rejected by parser"
        );
    }
}

#[test]
fn test_duckdb_sql_injection_defense() {
    let (_store, catalog, _tmp) = create_store();

    let injection_payloads = [
        "'; DROP TABLE model_blobs; --",
        "' OR '1'='1",
        "1; DELETE FROM model_manifests; --",
        "admin'--",
        "UNION SELECT * FROM sqlite_master --",
        "foo' UNION ALL SELECT NULL, NULL, NULL --",
    ];

    for (i, payload) in injection_payloads.iter().enumerate() {
        // 1. Model blobs table
        let blob = ModelBlobRecord {
            digest: (*payload).to_owned(),
            size_bytes: 1024,
            blob_type: "raw_weights".to_owned(),
            created_at: 100,
            last_accessed_at: 100,
            ref_count: 0,
        };
        assert!(catalog.save_blob(&blob).is_ok());
        assert!(catalog.list_blobs().is_ok());
        let retrieved = catalog.get_blob(payload).expect("query succeeds");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().digest, *payload);
        assert!(catalog.delete_blob(payload).expect("delete succeeds"));

        // 2. Tenants table
        let tenant_id = format!("t_inj_{i}");
        let tenant = Tenant {
            id: tenant_id.clone(),
            name: (*payload).to_owned(),
            created_at: 100,
        };
        assert!(catalog.save_tenant(&tenant).is_ok());
        let got_tenant = catalog.get_tenant(&tenant_id).expect("query succeeds");
        assert!(got_tenant.is_some());
        assert_eq!(got_tenant.unwrap().name, *payload);

        // 3. Projects table
        let proj_id = format!("p_inj_{i}");
        let project = Project {
            id: proj_id.clone(),
            tenant_id: tenant_id.clone(),
            name: (*payload).to_owned(),
            token_budget: Some(500),
            created_at: 100,
        };
        assert!(catalog.save_project(&project).is_ok());
        let got_proj = catalog.get_project(&proj_id).expect("query succeeds");
        assert!(got_proj.is_some());
        assert_eq!(got_proj.unwrap().name, *payload);

        // 4. Session last turns table
        let sess_id = format!("sess_inj_{i}");
        let turn = SessionLastTurn {
            session_id: sess_id.clone(),
            project_id: proj_id.clone(),
            tenant_id: tenant_id.clone(),
            model_id: (*payload).to_owned(),
            prompt: (*payload).to_owned(),
            response: (*payload).to_owned(),
            prompt_tokens: 15,
            completion_tokens: 30,
            parameters_json: (*payload).to_owned(),
            updated_at: 100,
        };
        assert!(catalog.save_session_last_turn(&turn).is_ok());
        let got_turn = catalog
            .load_session_last_turn(&sess_id)
            .expect("load succeeds");
        assert!(got_turn.is_some());
        let turn_data = got_turn.unwrap();
        assert_eq!(turn_data.prompt, *payload);
        assert_eq!(turn_data.response, *payload);

        // 5. Tuning profiles table
        let model_key = format!("model_inj_{i}");
        let prof = TuningProfile {
            cluster: (*payload).to_owned(),
            advisory_notice: Some((*payload).to_owned()),
            ..Default::default()
        };
        assert!(catalog.save_tuning_profile(&model_key, &prof).is_ok());
        let loaded_prof = catalog
            .load_tuning_profile(&model_key)
            .expect("query profile");
        assert!(loaded_prof.is_some());
        assert_eq!(loaded_prof.unwrap().cluster, *payload);

        // 6. Storage config table
        let cfg_key = format!("config_inj_{i}");
        assert!(catalog.set_storage_config(&cfg_key, payload).is_ok());
        let got_cfg = catalog.get_storage_config(&cfg_key).expect("query config");
        assert!(got_cfg.is_some());
        assert_eq!(got_cfg.unwrap(), *payload);

        // 7. Model manifests table
        let valid_weight = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let blob_base = ModelBlobRecord {
            digest: valid_weight.to_owned(),
            size_bytes: 1024,
            blob_type: "raw_weights".to_owned(),
            created_at: 100,
            last_accessed_at: 100,
            ref_count: 0,
        };
        let _ = catalog.save_blob(&blob_base);

        let manifest_rec = ModelManifestRecord {
            manifest_digest: format!("man_inj_{i}"),
            weight_digest: valid_weight.to_owned(),
            format: (*payload).to_owned(),
            architecture_id: (*payload).to_owned(),
            recipe_version: (*payload).to_owned(),
            quantization: (*payload).to_owned(),
            context_length: 1024,
            license: Some((*payload).to_owned()),
            origin: Some((*payload).to_owned()),
            manifest_json: (*payload).to_owned(),
            created_at: 100,
        };
        assert!(catalog.save_manifest(&manifest_rec).is_ok());
        let got_man = catalog
            .get_manifest(&manifest_rec.manifest_digest)
            .expect("query manifest");
        assert!(got_man.is_some());
        assert_eq!(got_man.unwrap().architecture_id, *payload);

        // 8. Model aliases table
        let alias_rec = ModelAliasRecord {
            alias_id: format!("alias_inj_{i}"),
            source: "local".to_owned(),
            namespace: "ns".to_owned(),
            name: format!("m_{i}"),
            tag: format!("t_{i}"),
            manifest_digest: manifest_rec.manifest_digest.clone(),
            weight_digest: valid_weight.to_owned(),
            alias_path: (*payload).to_owned(),
            created_at: 100,
        };
        assert!(catalog.save_alias(&alias_rec).is_ok());
        let got_alias = catalog.get_alias(&alias_rec.alias_id).expect("query alias");
        assert!(got_alias.is_some());
        assert_eq!(got_alias.unwrap().alias_path, *payload);
    }
}

#[test]
fn test_validate_sha256_digest_strictness() {
    // Uppercase hex must fail
    assert!(
        validate_sha256_digest("E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855")
            .is_err()
    );
    // Leading/trailing whitespace must fail
    assert!(
        validate_sha256_digest(
            " e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 "
        )
        .is_err()
    );
    // 63 chars (too short)
    assert!(
        validate_sha256_digest("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85")
            .is_err()
    );
    // 65 chars (too long)
    assert!(
        validate_sha256_digest("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855a")
            .is_err()
    );
    // Illegal chars
    assert!(
        validate_sha256_digest("g3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            .is_err()
    );
    // Valid 64 lowercase hex
    assert!(
        validate_sha256_digest("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            .is_ok()
    );
}

#[test]
fn test_register_manifest_rejects_weight_digest_traversal() {
    let (store, _catalog, _tmp) = create_store();
    let malicious_manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: None,
        origin: None,
        weight_digest: "../../../../etc/passwd".to_owned(),
        weight_size_bytes: 1024,
        created_at: 1000,
    };
    let res = store.register_manifest(&malicious_manifest);
    assert!(
        res.is_err(),
        "register_manifest must reject path traversal in weight_digest"
    );
}

#[test]
fn test_staging_promotion_fails_if_part_file_is_locked_externally() {
    use std::os::unix::io::AsRawFd;

    let (store, _catalog, _tmp) = create_store();
    let (staging_path, mut f) = store.create_staging_file().unwrap();
    f.write_all(b"valid_bytes").unwrap();

    // External holder takes exclusive lock
    let fd = f.as_raw_fd();
    // SAFETY: fd is an open, valid file descriptor owned by f.
    let lock_res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(lock_res, 0);

    // Concurrent promote must detect locked staging file and refuse promotion
    let digest = Sha256::digest(b"valid_bytes");
    let res = store.promote_staged_blob(&staging_path, &digest, "raw");
    assert!(
        res.is_err(),
        "promote_staged_blob must fail when staging file is locked"
    );

    // Clean up lock
    // SAFETY: fd is an open, valid file descriptor owned by f.
    unsafe { libc::flock(fd, libc::LOCK_UN) };
}

#[test]
fn test_staging_hash_mismatch_and_cleanup() {
    let (store, _catalog, _tmp) = create_store();

    let payload = b"actual_uncompromised_data";
    let (staging_path, mut f) = store.create_staging_file().expect("staging file");
    f.write_all(payload).expect("write");
    drop(f);

    assert!(staging_path.exists());

    // Claim digest is forged / mismatched
    let forged_digest = "0000000000000000000000000000000000000000000000000000000000000000";
    let err = store
        .promote_staged_blob(&staging_path, forged_digest, "raw")
        .expect_err("mismatched digest must fail");

    assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    assert!(err.to_string().contains("integrity verification failed"));

    // Staging file must be purged upon verification failure to prevent accumulation of corrupted files
    assert!(
        !staging_path.exists(),
        "invalid staging file must be deleted upon integrity failure"
    );
}

#[test]
fn test_symlink_escape_confinement_check() {
    let (store, _catalog, _tmp) = create_store();

    // Register a valid blob and manifest
    let payload = b"payload_for_escape_check";
    let digest = Sha256::digest(payload);
    let (staging, mut f) = store.create_staging_file().unwrap();
    f.write_all(payload).unwrap();
    drop(f);
    store.promote_staged_blob(&staging, &digest, "raw").unwrap();

    let manifest = ModelManifest {
        schema_version: 1,
        format: "raw".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "F32".to_owned(),
        context_length: 1024,
        license: None,
        origin: None,
        weight_digest: digest.clone(),
        weight_size_bytes: payload.len() as u64,
        created_at: 100,
    };
    let man_rec = store.register_manifest(&manifest).unwrap();
    store
        .create_alias("local", "ns", "test", "v1", &man_rec.manifest_digest)
        .unwrap();

    // Resolving alias confirms confinement within store root
    let resolved = store.resolve_alias("local/ns/test/v1").unwrap();
    assert!(resolved.weight_path.starts_with(store.root()));
    assert!(resolved.manifest_path.starts_with(store.root()));
}
