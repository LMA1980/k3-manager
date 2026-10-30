//! `DuckDB` operational metadata store & bounded session persistence (US-17, Features 32, 33, 34, ADR-001, ADR-003).
//!
//! Implements relational persistence across daemon restarts for tenants, projects, token budgets,
//! tuning profiles, and cumulative usage counters using `DuckDB`. Provides strictly bounded O(1) single-row
//! last-turn retention per session via SQL upsert (`ON CONFLICT (session_id) DO UPDATE SET`)
//! and diagnostic replay support.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub use duckdb::OptionalExt;
pub use duckdb::OptionalExt as OptionalExtension;
use duckdb::{Connection, params};
use serde::{Deserialize, Serialize};

use k3_core::profile::TuningProfile;

/// Error conditions encountered during persistence operations.
#[derive(Debug)]
pub enum PersistenceError {
    /// `DuckDB` engine error.
    DuckDb(duckdb::Error),
    /// Legacy SQLite engine error variant (preserved for test pattern matching).
    Sqlite(String),
    /// JSON serialization or deserialization failure.
    Serialization(serde_json::Error),
    /// Requested session identifier not found in persistence store.
    SessionNotFound(String),
    /// Requested project identifier not found in persistence store.
    ProjectNotFound(String),
    /// Requested tenant identifier not found in persistence store.
    TenantNotFound(String),
    /// Relational foreign key constraint violation.
    ForeignKeyViolation(String),
    /// Database synchronization lock poisoned.
    LockError(String),
}

impl fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuckDb(e) => write!(f, "duckdb error: {e}"),
            Self::Sqlite(msg) => write!(f, "sqlite error: {msg}"),
            Self::Serialization(e) => write!(f, "serialization error: {e}"),
            Self::SessionNotFound(id) => write!(f, "session '{id}' not found in persistence store"),
            Self::ProjectNotFound(id) => write!(f, "project '{id}' not found in persistence store"),
            Self::TenantNotFound(id) => write!(f, "tenant '{id}' not found in persistence store"),
            Self::ForeignKeyViolation(msg) => write!(f, "foreign key constraint violation: {msg}"),
            Self::LockError(msg) => write!(f, "persistence lock error: {msg}"),
        }
    }
}

impl std::error::Error for PersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DuckDb(e) => Some(e),
            Self::Serialization(e) => Some(e),
            _ => None,
        }
    }
}

impl From<duckdb::Error> for PersistenceError {
    fn from(err: duckdb::Error) -> Self {
        let msg = err.to_string();
        let lower = msg.to_lowercase();
        if lower.contains("foreign key")
            || lower.contains("violates foreign key constraint")
            || lower.contains("constraint error")
        {
            Self::ForeignKeyViolation(msg)
        } else {
            Self::DuckDb(err)
        }
    }
}

impl From<serde_json::Error> for PersistenceError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err)
    }
}

/// Tenant operational definition (ADR-001).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tenant {
    /// Unique tenant identifier.
    pub id: String,
    /// Tenant human-readable name.
    pub name: String,
    /// Unix creation timestamp in seconds.
    pub created_at: u64,
}

/// Project resource and budget definition (ADR-001, AGENTS.md §9).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Project {
    /// Unique project identifier.
    pub id: String,
    /// Owning tenant identifier (foreign key).
    pub tenant_id: String,
    /// Project human-readable name.
    pub name: String,
    /// Token budget in tokens (u32 per AGENTS.md §9).
    pub token_budget: Option<u32>,
    /// Unix creation timestamp in seconds.
    pub created_at: u64,
}

/// Single-row bounded latest turn record for a session (US-17, ADR-003).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionLastTurn {
    /// Unique session identifier (primary key).
    pub session_id: String,
    /// Owning project identifier (foreign key).
    pub project_id: String,
    /// Owning tenant identifier (foreign key).
    pub tenant_id: String,
    /// Logical model identifier used for generation.
    pub model_id: String,
    /// Last input prompt text.
    pub prompt: String,
    /// Last generated response text.
    pub response: String,
    /// Prompt tokens count (u32 per AGENTS.md §9).
    pub prompt_tokens: u32,
    /// Generated completion tokens count (u32 per AGENTS.md §9).
    pub completion_tokens: u32,
    /// Sampling and generation parameters formatted in JSON.
    pub parameters_json: String,
    /// Unix timestamp in seconds of last turn completion.
    pub updated_at: u64,
}

/// Diagnostic replay execution result (US-17 Scenario 3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticReplayResult {
    /// Replayed session identifier.
    pub session_id: String,
    /// Target model identifier.
    pub model_id: String,
    /// Input prompt replayed.
    pub prompt: String,
    /// Expected response recorded in persistence store.
    pub expected_response: String,
    /// Replayed response produced by compute engine.
    pub replayed_response: String,
    /// Whether replayed response matched expected response.
    pub matched: bool,
    /// Replay execution latency in milliseconds.
    pub latency_ms: f64,
}

/// Content-addressed storage blob record in `DuckDB` catalog (US-18, ADR-003, ADR-035).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelBlobRecord {
    /// SHA-256 lowercase hex digest (64 chars).
    pub digest: String,
    /// Physical size on disk in bytes.
    pub size_bytes: u64,
    /// Blob classification: `"raw_weights"` or `"derived_weights"`.
    pub blob_type: String,
    /// Creation timestamp in epoch seconds.
    pub created_at: u64,
    /// Last access timestamp in epoch seconds.
    pub last_accessed_at: u64,
    /// Active reference count across manifests, aliases, and derivations (u32 per AGENTS.md §9).
    pub ref_count: u32,
}

/// Content-addressed model manifest record in `DuckDB` catalog (US-18, ADR-035).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifestRecord {
    /// SHA-256 of canonical manifest JSON.
    pub manifest_digest: String,
    /// Foreign key referencing `model_blobs(digest)`.
    pub weight_digest: String,
    /// Model format: "gguf", "safetensors".
    pub format: String,
    /// Architecture identifier: "synthetic", "qwen3", etc.
    pub architecture_id: String,
    /// Architecture recipe version: "1.0.0".
    pub recipe_version: String,
    /// Quantization format: `"Q4_K_M"`, `"F32"`, etc.
    pub quantization: String,
    /// Trained context length limit in tokens.
    pub context_length: u64,
    /// Optional license identifier.
    pub license: Option<String>,
    /// Optional source origin repository or URL.
    pub origin: Option<String>,
    /// Canonical JSON string of the manifest.
    pub manifest_json: String,
    /// Creation timestamp in epoch seconds.
    pub created_at: u64,
}

/// Human-browsable model alias record in `DuckDB` catalog (US-18, ADR-035).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelAliasRecord {
    /// Canonical alias identifier (e.g. "huggingface/Qwen/Qwen2.5-7B/latest").
    pub alias_id: String,
    /// Source: "huggingface", "ollama", "local", etc.
    pub source: String,
    /// Namespace: "Qwen", "meta-llama", etc.
    pub namespace: String,
    /// Model name: "Qwen2.5-7B-Instruct".
    pub name: String,
    /// Tag: `"latest"`, `"q4_k_m"`.
    pub tag: String,
    /// Foreign key referencing `model_manifests(manifest_digest)`.
    pub manifest_digest: String,
    /// Foreign key referencing `model_blobs(digest)`.
    pub weight_digest: String,
    /// Relative path under `models/` (e.g. "models/huggingface/Qwen/Qwen2.5-7B/latest").
    pub alias_path: String,
    /// Creation timestamp in epoch seconds.
    pub created_at: u64,
}

/// Hardware-repacked or format-converted derived blob record (US-21, ADR-037, ADR-038).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivedBlobRecord {
    /// SHA-256 digest of the derived blob referencing `model_blobs(digest)`.
    pub derived_digest: String,
    /// SHA-256 digest of original source blob referencing `model_blobs(digest)`.
    pub source_digest: String,
    /// Architecture recipe revision string.
    pub recipe_version: String,
    /// Target layout: `"ime_tile_8x16"`, `"native"`.
    pub target_layout: String,
    /// Status: "valid", "stale".
    pub status: String,
    /// Creation timestamp in epoch seconds.
    pub created_at: u64,
}

/// Granular reference record for auditability and cascade safety (US-18, US-22).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobReferenceRecord {
    /// Unique reference UUID.
    pub ref_id: String,
    /// SHA-256 digest referencing `model_blobs(digest)`.
    pub blob_digest: String,
    /// Reference source classification: "manifest", "alias", "derived".
    pub ref_type: String,
    /// Source identifier (`manifest_digest`, `alias_id`, or `derived_digest`).
    pub ref_source: String,
    /// Creation timestamp in epoch seconds.
    pub created_at: u64,
}

/// Interface for Content-Addressed Store (CAS) catalog tracking using `DuckDB`.
#[allow(clippy::missing_errors_doc)]
pub trait CasCatalog: fmt::Debug + Send + Sync {
    // Blob lifecycle
    fn save_blob(&self, blob: &ModelBlobRecord) -> Result<(), PersistenceError>;
    fn get_blob(&self, digest: &str) -> Result<Option<ModelBlobRecord>, PersistenceError>;
    fn list_blobs(&self) -> Result<Vec<ModelBlobRecord>, PersistenceError>;
    fn update_blob_access(&self, digest: &str, timestamp: u64) -> Result<(), PersistenceError>;
    fn delete_blob(&self, digest: &str) -> Result<bool, PersistenceError>;
    fn increment_blob_ref(&self, digest: &str) -> Result<u32, PersistenceError>;
    fn decrement_blob_ref(&self, digest: &str) -> Result<u32, PersistenceError>;

    // Manifest lifecycle
    fn save_manifest(&self, manifest: &ModelManifestRecord) -> Result<(), PersistenceError>;
    fn get_manifest(
        &self,
        manifest_digest: &str,
    ) -> Result<Option<ModelManifestRecord>, PersistenceError>;
    fn list_manifests(&self) -> Result<Vec<ModelManifestRecord>, PersistenceError>;
    fn delete_manifest(&self, manifest_digest: &str) -> Result<bool, PersistenceError>;

    // Alias hierarchy
    fn save_alias(&self, alias: &ModelAliasRecord) -> Result<(), PersistenceError>;
    fn get_alias(&self, alias_id: &str) -> Result<Option<ModelAliasRecord>, PersistenceError>;
    fn find_alias(
        &self,
        source: &str,
        namespace: &str,
        name: &str,
        tag: &str,
    ) -> Result<Option<ModelAliasRecord>, PersistenceError>;
    fn list_aliases(&self) -> Result<Vec<ModelAliasRecord>, PersistenceError>;
    fn delete_alias(&self, alias_id: &str) -> Result<bool, PersistenceError>;

    // Derived layouts
    fn save_derived_blob(&self, derived: &DerivedBlobRecord) -> Result<(), PersistenceError>;
    fn get_derived_blob(
        &self,
        source_digest: &str,
        recipe_version: &str,
        target_layout: &str,
    ) -> Result<Option<DerivedBlobRecord>, PersistenceError>;
    fn list_derived_blobs(&self) -> Result<Vec<DerivedBlobRecord>, PersistenceError>;
    fn list_derived_for_source(
        &self,
        source_digest: &str,
    ) -> Result<Vec<DerivedBlobRecord>, PersistenceError>;
    fn update_derived_status(
        &self,
        derived_digest: &str,
        status: &str,
    ) -> Result<(), PersistenceError>;
    fn delete_derived_blob(&self, derived_digest: &str) -> Result<bool, PersistenceError>;

    // References
    fn add_blob_reference(&self, reference: &BlobReferenceRecord) -> Result<(), PersistenceError>;
    fn remove_blob_reference(&self, ref_id: &str) -> Result<bool, PersistenceError>;
    fn list_blob_references(
        &self,
        blob_digest: &str,
    ) -> Result<Vec<BlobReferenceRecord>, PersistenceError>;

    // Storage config & watermarks
    fn get_storage_config(&self, key: &str) -> Result<Option<String>, PersistenceError>;
    fn set_storage_config(&self, key: &str, value: &str) -> Result<(), PersistenceError>;

    // GC & accounting
    fn recalculate_ref_counts(&self) -> Result<(), PersistenceError>;
    fn list_unreferenced_blobs(&self) -> Result<Vec<ModelBlobRecord>, PersistenceError>;
    fn total_blob_bytes(&self) -> Result<u64, PersistenceError>;
}

/// Core interface for runtime state persistence across reboots.
#[allow(clippy::missing_errors_doc)]
pub trait PersistenceStore: fmt::Debug + Send + Sync {
    // Tenant management
    fn save_tenant(&self, tenant: &Tenant) -> Result<(), PersistenceError>;
    fn get_tenant(&self, id: &str) -> Result<Option<Tenant>, PersistenceError>;
    fn list_tenants(&self) -> Result<Vec<Tenant>, PersistenceError>;

    // Project management
    fn save_project(&self, project: &Project) -> Result<(), PersistenceError>;
    fn get_project(&self, id: &str) -> Result<Option<Project>, PersistenceError>;
    fn list_projects(&self, tenant_id: Option<&str>) -> Result<Vec<Project>, PersistenceError>;

    // Tuning profile persistence
    fn save_tuning_profile(
        &self,
        model_id: &str,
        profile: &TuningProfile,
    ) -> Result<(), PersistenceError>;
    fn load_tuning_profile(
        &self,
        model_id: &str,
    ) -> Result<Option<TuningProfile>, PersistenceError>;
    fn list_tuning_profiles(&self) -> Result<HashMap<String, TuningProfile>, PersistenceError>;

    // Bounded single-row session last-turn management
    fn save_session_last_turn(&self, turn: &SessionLastTurn) -> Result<(), PersistenceError>;
    fn load_session_last_turn(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionLastTurn>, PersistenceError>;
    fn count_session_last_turns(&self) -> Result<usize, PersistenceError>;
    fn delete_session_last_turn(&self, session_id: &str) -> Result<bool, PersistenceError>;

    // Cumulative operational statistics (u32 counters per AGENTS.md §9)
    fn update_cumulative_stat(&self, stat_key: &str, delta: u32) -> Result<u32, PersistenceError>;
    fn get_cumulative_stat(&self, stat_key: &str) -> Result<u32, PersistenceError>;
    fn list_cumulative_stats(&self) -> Result<HashMap<String, u32>, PersistenceError>;
}

/// Production `DuckDB` implementation of [`PersistenceStore`].
#[derive(Debug, Clone)]
pub struct DuckDbPersistenceStore {
    conn: Arc<Mutex<Connection>>,
}

/// Backward compatibility type alias for existing test suites and imports.
pub type SqlitePersistenceStore = DuckDbPersistenceStore;

impl DuckDbPersistenceStore {
    /// Opens an in-memory `DuckDB` database, initializing schema and foreign keys.
    ///
    /// # Errors
    /// Returns [`PersistenceError::DuckDb`] on initialization failure.
    pub fn open_in_memory() -> Result<Self, PersistenceError> {
        let conn = Connection::open_in_memory()?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Opens or creates an on-disk `DuckDB` database with native file WAL persistence.
    ///
    /// # Errors
    /// Returns [`PersistenceError::DuckDb`] on failure to open or migrate.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, PersistenceError> {
        let conn = Connection::open(path)?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Initializes relational database tables, indices, and foreign keys.
    fn init_schema(&self) -> Result<(), PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS tenants (
                id TEXT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL,
                created_at BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY NOT NULL,
                tenant_id TEXT NOT NULL REFERENCES tenants(id),
                name TEXT NOT NULL,
                token_budget BIGINT,
                created_at BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tuning_profiles (
                model_id TEXT PRIMARY KEY NOT NULL,
                profile_json TEXT NOT NULL,
                updated_at BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_last_turns (
                session_id TEXT PRIMARY KEY NOT NULL,
                project_id TEXT NOT NULL REFERENCES projects(id),
                tenant_id TEXT NOT NULL REFERENCES tenants(id),
                model_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                prompt_tokens BIGINT NOT NULL,
                completion_tokens BIGINT NOT NULL,
                parameters_json TEXT NOT NULL,
                updated_at BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cumulative_stats (
                stat_key TEXT PRIMARY KEY NOT NULL,
                stat_value BIGINT NOT NULL,
                updated_at BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_blobs (
                digest TEXT PRIMARY KEY NOT NULL,
                size_bytes BIGINT NOT NULL,
                blob_type TEXT NOT NULL,
                created_at BIGINT NOT NULL,
                last_accessed_at BIGINT NOT NULL,
                ref_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS model_manifests (
                manifest_digest TEXT PRIMARY KEY NOT NULL,
                weight_digest TEXT NOT NULL REFERENCES model_blobs(digest),
                format TEXT NOT NULL,
                architecture_id TEXT NOT NULL,
                recipe_version TEXT NOT NULL,
                quantization TEXT NOT NULL,
                context_length BIGINT NOT NULL,
                license TEXT,
                origin TEXT,
                manifest_json TEXT NOT NULL,
                created_at BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_aliases (
                alias_id TEXT PRIMARY KEY NOT NULL,
                source TEXT NOT NULL,
                namespace TEXT NOT NULL,
                name TEXT NOT NULL,
                tag TEXT NOT NULL,
                manifest_digest TEXT NOT NULL REFERENCES model_manifests(manifest_digest),
                weight_digest TEXT NOT NULL REFERENCES model_blobs(digest),
                alias_path TEXT NOT NULL,
                created_at BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS derived_blobs (
                derived_digest TEXT PRIMARY KEY NOT NULL REFERENCES model_blobs(digest),
                source_digest TEXT NOT NULL REFERENCES model_blobs(digest),
                recipe_version TEXT NOT NULL,
                target_layout TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS blob_references (
                ref_id TEXT PRIMARY KEY NOT NULL,
                blob_digest TEXT NOT NULL REFERENCES model_blobs(digest),
                ref_type TEXT NOT NULL,
                ref_source TEXT NOT NULL,
                created_at BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS storage_config (
                config_key TEXT PRIMARY KEY NOT NULL,
                config_value TEXT NOT NULL,
                updated_at BIGINT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_projects_tenant ON projects(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_session_project ON session_last_turns(project_id);
            CREATE INDEX IF NOT EXISTS idx_manifest_weights ON model_manifests(weight_digest);
            CREATE INDEX IF NOT EXISTS idx_alias_manifest ON model_aliases(manifest_digest);
            CREATE INDEX IF NOT EXISTS idx_alias_lookup ON model_aliases(source, namespace, name, tag);
            CREATE INDEX IF NOT EXISTS idx_derived_lookup ON derived_blobs(source_digest, recipe_version, target_layout);
            CREATE INDEX IF NOT EXISTS idx_blob_refs_digest ON blob_references(blob_digest);",
        )?;
        Ok(())
    }

    /// Returns current Unix epoch timestamp in seconds.
    #[must_use]
    pub fn current_epoch_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

impl PersistenceStore for DuckDbPersistenceStore {
    fn save_tenant(&self, tenant: &Tenant) -> Result<(), PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute(
            "INSERT INTO tenants (id, name, created_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT (id) DO UPDATE SET
                 name = excluded.name,
                 created_at = excluded.created_at;",
            params![tenant.id, tenant.name, tenant.created_at],
        )?;
        Ok(())
    }

    fn get_tenant(&self, id: &str) -> Result<Option<Tenant>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare("SELECT id, name, created_at FROM tenants WHERE id = ?1;")?;
        let res = stmt
            .query_row(params![id], |row| {
                Ok(Tenant {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    created_at: row.get(2)?,
                })
            })
            .optional()?;
        Ok(res)
    }

    fn list_tenants(&self) -> Result<Vec<Tenant>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt =
            conn.prepare("SELECT id, name, created_at FROM tenants ORDER BY created_at ASC;")?;
        let rows = stmt.query_map([], |row| {
            Ok(Tenant {
                id: row.get(0)?,
                name: row.get(1)?,
                created_at: row.get(2)?,
            })
        })?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    fn save_project(&self, project: &Project) -> Result<(), PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;

        // Pre-validate that parent tenant exists to ensure referential integrity
        let tenant_exists: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM tenants WHERE id = ?1;",
                params![project.tenant_id],
                |r| r.get(0),
            )
            .optional()?;
        if tenant_exists.is_none() {
            return Err(PersistenceError::ForeignKeyViolation(format!(
                "FOREIGN KEY constraint failed: tenant '{}' does not exist",
                project.tenant_id
            )));
        }

        conn.execute(
            "INSERT INTO projects (id, tenant_id, name, token_budget, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT (id) DO UPDATE SET
                 name = excluded.name,
                 token_budget = excluded.token_budget,
                 created_at = excluded.created_at;",
            params![
                project.id,
                project.tenant_id,
                project.name,
                project.token_budget,
                project.created_at
            ],
        )?;
        Ok(())
    }

    fn get_project(&self, id: &str) -> Result<Option<Project>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT id, tenant_id, name, token_budget, created_at FROM projects WHERE id = ?1;",
        )?;
        let res = stmt
            .query_row(params![id], |row| {
                Ok(Project {
                    id: row.get(0)?,
                    tenant_id: row.get(1)?,
                    name: row.get(2)?,
                    token_budget: row.get(3)?,
                    created_at: row.get(4)?,
                })
            })
            .optional()?;
        Ok(res)
    }

    fn list_projects(&self, tenant_id: Option<&str>) -> Result<Vec<Project>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut list = Vec::new();
        if let Some(tid) = tenant_id {
            let mut stmt = conn.prepare(
                "SELECT id, tenant_id, name, token_budget, created_at FROM projects WHERE tenant_id = ?1 ORDER BY created_at ASC;",
            )?;
            let rows = stmt.query_map(params![tid], |row| {
                Ok(Project {
                    id: row.get(0)?,
                    tenant_id: row.get(1)?,
                    name: row.get(2)?,
                    token_budget: row.get(3)?,
                    created_at: row.get(4)?,
                })
            })?;
            for r in rows {
                list.push(r?);
            }
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, tenant_id, name, token_budget, created_at FROM projects ORDER BY created_at ASC;",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok(Project {
                    id: row.get(0)?,
                    tenant_id: row.get(1)?,
                    name: row.get(2)?,
                    token_budget: row.get(3)?,
                    created_at: row.get(4)?,
                })
            })?;
            for r in rows {
                list.push(r?);
            }
        }
        Ok(list)
    }

    fn save_tuning_profile(
        &self,
        model_id: &str,
        profile: &TuningProfile,
    ) -> Result<(), PersistenceError> {
        let json_str = serde_json::to_string(profile)?;
        let updated_at = Self::current_epoch_secs();
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute(
            "INSERT INTO tuning_profiles (model_id, profile_json, updated_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT (model_id) DO UPDATE SET
                 profile_json = excluded.profile_json,
                 updated_at = excluded.updated_at;",
            params![model_id, json_str, updated_at],
        )?;
        Ok(())
    }

    fn load_tuning_profile(
        &self,
        model_id: &str,
    ) -> Result<Option<TuningProfile>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt =
            conn.prepare("SELECT profile_json FROM tuning_profiles WHERE model_id = ?1;")?;
        let json_opt: Option<String> = stmt
            .query_row(params![model_id], |row| row.get(0))
            .optional()?;
        match json_opt {
            Some(s) => {
                let prof = serde_json::from_str(&s)?;
                Ok(Some(prof))
            }
            None => Ok(None),
        }
    }

    fn list_tuning_profiles(&self) -> Result<HashMap<String, TuningProfile>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare("SELECT model_id, profile_json FROM tuning_profiles;")?;
        let rows = stmt.query_map([], |row| {
            let mid: String = row.get(0)?;
            let pjson: String = row.get(1)?;
            Ok((mid, pjson))
        })?;
        let mut map = HashMap::new();
        for r in rows {
            let (mid, pjson) = r?;
            if let Ok(prof) = serde_json::from_str::<TuningProfile>(&pjson) {
                map.insert(mid, prof);
            }
        }
        Ok(map)
    }

    /// Saves the latest session turn via O(1) SQL upsert (ADR-003).
    fn save_session_last_turn(&self, turn: &SessionLastTurn) -> Result<(), PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;

        // Pre-validate that parent tenant exists
        let tenant_exists: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM tenants WHERE id = ?1;",
                params![turn.tenant_id],
                |r| r.get(0),
            )
            .optional()?;
        if tenant_exists.is_none() {
            return Err(PersistenceError::ForeignKeyViolation(format!(
                "FOREIGN KEY constraint failed: tenant '{}' does not exist",
                turn.tenant_id
            )));
        }

        // Pre-validate that parent project exists
        let project_exists: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM projects WHERE id = ?1;",
                params![turn.project_id],
                |r| r.get(0),
            )
            .optional()?;
        if project_exists.is_none() {
            return Err(PersistenceError::ForeignKeyViolation(format!(
                "FOREIGN KEY constraint failed: project '{}' does not exist",
                turn.project_id
            )));
        }

        conn.execute(
            "INSERT INTO session_last_turns (
                session_id, project_id, tenant_id, model_id,
                prompt, response, prompt_tokens, completion_tokens,
                parameters_json, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ON CONFLICT (session_id) DO UPDATE SET
                project_id = excluded.project_id,
                tenant_id = excluded.tenant_id,
                model_id = excluded.model_id,
                prompt = excluded.prompt,
                response = excluded.response,
                prompt_tokens = excluded.prompt_tokens,
                completion_tokens = excluded.completion_tokens,
                parameters_json = excluded.parameters_json,
                updated_at = excluded.updated_at;",
            params![
                turn.session_id,
                turn.project_id,
                turn.tenant_id,
                turn.model_id,
                turn.prompt,
                turn.response,
                turn.prompt_tokens,
                turn.completion_tokens,
                turn.parameters_json,
                turn.updated_at,
            ],
        )?;
        Ok(())
    }

    fn load_session_last_turn(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionLastTurn>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT session_id, project_id, tenant_id, model_id,
                    prompt, response, prompt_tokens, completion_tokens,
                    parameters_json, updated_at
             FROM session_last_turns WHERE session_id = ?1;",
        )?;
        let res = stmt
            .query_row(params![session_id], |row| {
                Ok(SessionLastTurn {
                    session_id: row.get(0)?,
                    project_id: row.get(1)?,
                    tenant_id: row.get(2)?,
                    model_id: row.get(3)?,
                    prompt: row.get(4)?,
                    response: row.get(5)?,
                    prompt_tokens: row.get(6)?,
                    completion_tokens: row.get(7)?,
                    parameters_json: row.get(8)?,
                    updated_at: row.get(9)?,
                })
            })
            .optional()?;
        Ok(res)
    }

    fn count_session_last_turns(&self) -> Result<usize, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM session_last_turns;", [], |r| r.get(0))?;
        Ok(usize::try_from(count).unwrap_or(0))
    }

    fn delete_session_last_turn(&self, session_id: &str) -> Result<bool, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let affected = conn.execute(
            "DELETE FROM session_last_turns WHERE session_id = ?1;",
            params![session_id],
        )?;
        Ok(affected > 0)
    }

    fn update_cumulative_stat(&self, stat_key: &str, delta: u32) -> Result<u32, PersistenceError> {
        let updated_at = Self::current_epoch_secs();
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute(
            "INSERT INTO cumulative_stats (stat_key, stat_value, updated_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT (stat_key) DO UPDATE SET
                 stat_value = LEAST(4294967295, cumulative_stats.stat_value + excluded.stat_value),
                 updated_at = excluded.updated_at;",
            params![stat_key, delta, updated_at],
        )?;
        let new_val: i64 = conn.query_row(
            "SELECT stat_value FROM cumulative_stats WHERE stat_key = ?1;",
            params![stat_key],
            |r| r.get(0),
        )?;
        Ok(u32::try_from(new_val).unwrap_or(u32::MAX))
    }

    fn get_cumulative_stat(&self, stat_key: &str) -> Result<u32, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt =
            conn.prepare("SELECT stat_value FROM cumulative_stats WHERE stat_key = ?1;")?;
        let val_opt: Option<i64> = stmt
            .query_row(params![stat_key], |row| row.get(0))
            .optional()?;
        Ok(val_opt.map_or(0, |v| u32::try_from(v).unwrap_or(u32::MAX)))
    }

    fn list_cumulative_stats(&self) -> Result<HashMap<String, u32>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare("SELECT stat_key, stat_value FROM cumulative_stats;")?;
        let rows = stmt.query_map([], |row| {
            let key: String = row.get(0)?;
            let val: i64 = row.get(1)?;
            Ok((key, u32::try_from(val).unwrap_or(u32::MAX)))
        })?;
        let mut map = HashMap::new();
        for r in rows {
            let (k, v) = r?;
            map.insert(k, v);
        }
        Ok(map)
    }
}

impl CasCatalog for DuckDbPersistenceStore {
    fn save_blob(&self, blob: &ModelBlobRecord) -> Result<(), PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let size_i64 = i64::try_from(blob.size_bytes).unwrap_or(i64::MAX);
        let ref_count_i32 = i32::try_from(blob.ref_count).unwrap_or(i32::MAX);
        conn.execute(
            "INSERT INTO model_blobs (digest, size_bytes, blob_type, created_at, last_accessed_at, ref_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT (digest) DO UPDATE SET
                 size_bytes = excluded.size_bytes,
                 blob_type = excluded.blob_type,
                 last_accessed_at = excluded.last_accessed_at,
                 ref_count = excluded.ref_count;",
            params![
                blob.digest,
                size_i64,
                blob.blob_type,
                blob.created_at,
                blob.last_accessed_at,
                ref_count_i32,
            ],
        )?;
        Ok(())
    }

    fn get_blob(&self, digest: &str) -> Result<Option<ModelBlobRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT digest, size_bytes, blob_type, created_at, last_accessed_at, ref_count
             FROM model_blobs WHERE digest = ?1;",
        )?;
        let res = stmt
            .query_row(params![digest], |row| {
                let size_i64: i64 = row.get(1)?;
                let count_i32: i32 = row.get(5)?;
                Ok(ModelBlobRecord {
                    digest: row.get(0)?,
                    size_bytes: u64::try_from(size_i64.max(0)).unwrap_or(0),
                    blob_type: row.get(2)?,
                    created_at: row.get(3)?,
                    last_accessed_at: row.get(4)?,
                    ref_count: u32::try_from(count_i32.max(0)).unwrap_or(0),
                })
            })
            .optional()?;
        Ok(res)
    }

    fn list_blobs(&self) -> Result<Vec<ModelBlobRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT digest, size_bytes, blob_type, created_at, last_accessed_at, ref_count
             FROM model_blobs ORDER BY created_at ASC;",
        )?;
        let rows = stmt.query_map([], |row| {
            let size_i64: i64 = row.get(1)?;
            let count_i32: i32 = row.get(5)?;
            Ok(ModelBlobRecord {
                digest: row.get(0)?,
                size_bytes: u64::try_from(size_i64.max(0)).unwrap_or(0),
                blob_type: row.get(2)?,
                created_at: row.get(3)?,
                last_accessed_at: row.get(4)?,
                ref_count: u32::try_from(count_i32.max(0)).unwrap_or(0),
            })
        })?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    fn update_blob_access(&self, digest: &str, timestamp: u64) -> Result<(), PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute(
            "UPDATE model_blobs SET last_accessed_at = ?2 WHERE digest = ?1;",
            params![digest, timestamp],
        )?;
        Ok(())
    }

    fn delete_blob(&self, digest: &str) -> Result<bool, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let _ = conn.execute(
            "DELETE FROM blob_references WHERE blob_digest = ?1;",
            params![digest],
        );
        let _ = conn.execute(
            "DELETE FROM storage_config WHERE config_key = 'purge_immediate:' || ?1 OR config_key = 'purge_after:' || ?1;",
            params![digest],
        );
        let affected = conn.execute(
            "DELETE FROM model_blobs WHERE digest = ?1;",
            params![digest],
        )?;
        Ok(affected > 0)
    }

    fn increment_blob_ref(&self, digest: &str) -> Result<u32, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute(
            "UPDATE model_blobs SET ref_count = ref_count + 1 WHERE digest = ?1;",
            params![digest],
        )?;
        let count: i32 = conn.query_row(
            "SELECT ref_count FROM model_blobs WHERE digest = ?1;",
            params![digest],
            |r| r.get(0),
        )?;
        Ok(u32::try_from(count.max(0)).unwrap_or(0))
    }

    fn decrement_blob_ref(&self, digest: &str) -> Result<u32, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute(
            "UPDATE model_blobs SET ref_count = GREATEST(0, ref_count - 1) WHERE digest = ?1;",
            params![digest],
        )?;
        let count: i32 = conn.query_row(
            "SELECT ref_count FROM model_blobs WHERE digest = ?1;",
            params![digest],
            |r| r.get(0),
        )?;
        Ok(u32::try_from(count.max(0)).unwrap_or(0))
    }

    fn save_manifest(&self, manifest: &ModelManifestRecord) -> Result<(), PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let ctx_len_i64 = i64::try_from(manifest.context_length).unwrap_or(i64::MAX);
        conn.execute(
            "INSERT INTO model_manifests (
                manifest_digest, weight_digest, format, architecture_id,
                recipe_version, quantization, context_length, license,
                origin, manifest_json, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            ON CONFLICT (manifest_digest) DO UPDATE SET
                weight_digest = excluded.weight_digest,
                format = excluded.format,
                architecture_id = excluded.architecture_id,
                recipe_version = excluded.recipe_version,
                quantization = excluded.quantization,
                context_length = excluded.context_length,
                license = excluded.license,
                origin = excluded.origin,
                manifest_json = excluded.manifest_json,
                created_at = excluded.created_at;",
            params![
                manifest.manifest_digest,
                manifest.weight_digest,
                manifest.format,
                manifest.architecture_id,
                manifest.recipe_version,
                manifest.quantization,
                ctx_len_i64,
                manifest.license,
                manifest.origin,
                manifest.manifest_json,
                manifest.created_at,
            ],
        )?;
        Ok(())
    }

    fn get_manifest(
        &self,
        manifest_digest: &str,
    ) -> Result<Option<ModelManifestRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT manifest_digest, weight_digest, format, architecture_id,
                    recipe_version, quantization, context_length, license,
                    origin, manifest_json, created_at
             FROM model_manifests WHERE manifest_digest = ?1;",
        )?;
        let res = stmt
            .query_row(params![manifest_digest], |row| {
                let ctx_len: i64 = row.get(6)?;
                Ok(ModelManifestRecord {
                    manifest_digest: row.get(0)?,
                    weight_digest: row.get(1)?,
                    format: row.get(2)?,
                    architecture_id: row.get(3)?,
                    recipe_version: row.get(4)?,
                    quantization: row.get(5)?,
                    context_length: u64::try_from(ctx_len.max(0)).unwrap_or(0),
                    license: row.get(7)?,
                    origin: row.get(8)?,
                    manifest_json: row.get(9)?,
                    created_at: row.get(10)?,
                })
            })
            .optional()?;
        Ok(res)
    }

    fn list_manifests(&self) -> Result<Vec<ModelManifestRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT manifest_digest, weight_digest, format, architecture_id,
                    recipe_version, quantization, context_length, license,
                    origin, manifest_json, created_at
             FROM model_manifests ORDER BY created_at ASC;",
        )?;
        let rows = stmt.query_map([], |row| {
            let ctx_len: i64 = row.get(6)?;
            Ok(ModelManifestRecord {
                manifest_digest: row.get(0)?,
                weight_digest: row.get(1)?,
                format: row.get(2)?,
                architecture_id: row.get(3)?,
                recipe_version: row.get(4)?,
                quantization: row.get(5)?,
                context_length: u64::try_from(ctx_len.max(0)).unwrap_or(0),
                license: row.get(7)?,
                origin: row.get(8)?,
                manifest_json: row.get(9)?,
                created_at: row.get(10)?,
            })
        })?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    fn delete_manifest(&self, manifest_digest: &str) -> Result<bool, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let affected = conn.execute(
            "DELETE FROM model_manifests WHERE manifest_digest = ?1;",
            params![manifest_digest],
        )?;
        Ok(affected > 0)
    }

    fn save_alias(&self, alias: &ModelAliasRecord) -> Result<(), PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute(
            "INSERT INTO model_aliases (
                alias_id, source, namespace, name, tag,
                manifest_digest, weight_digest, alias_path, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            ON CONFLICT (alias_id) DO UPDATE SET
                source = excluded.source,
                namespace = excluded.namespace,
                name = excluded.name,
                tag = excluded.tag,
                manifest_digest = excluded.manifest_digest,
                weight_digest = excluded.weight_digest,
                alias_path = excluded.alias_path,
                created_at = excluded.created_at;",
            params![
                alias.alias_id,
                alias.source,
                alias.namespace,
                alias.name,
                alias.tag,
                alias.manifest_digest,
                alias.weight_digest,
                alias.alias_path,
                alias.created_at,
            ],
        )?;
        Ok(())
    }

    fn get_alias(&self, alias_id: &str) -> Result<Option<ModelAliasRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT alias_id, source, namespace, name, tag,
                    manifest_digest, weight_digest, alias_path, created_at
             FROM model_aliases WHERE alias_id = ?1;",
        )?;
        let res = stmt
            .query_row(params![alias_id], |row| {
                Ok(ModelAliasRecord {
                    alias_id: row.get(0)?,
                    source: row.get(1)?,
                    namespace: row.get(2)?,
                    name: row.get(3)?,
                    tag: row.get(4)?,
                    manifest_digest: row.get(5)?,
                    weight_digest: row.get(6)?,
                    alias_path: row.get(7)?,
                    created_at: row.get(8)?,
                })
            })
            .optional()?;
        Ok(res)
    }

    fn find_alias(
        &self,
        source: &str,
        namespace: &str,
        name: &str,
        tag: &str,
    ) -> Result<Option<ModelAliasRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT alias_id, source, namespace, name, tag,
                    manifest_digest, weight_digest, alias_path, created_at
             FROM model_aliases WHERE source = ?1 AND namespace = ?2 AND name = ?3 AND tag = ?4;",
        )?;
        let res = stmt
            .query_row(params![source, namespace, name, tag], |row| {
                Ok(ModelAliasRecord {
                    alias_id: row.get(0)?,
                    source: row.get(1)?,
                    namespace: row.get(2)?,
                    name: row.get(3)?,
                    tag: row.get(4)?,
                    manifest_digest: row.get(5)?,
                    weight_digest: row.get(6)?,
                    alias_path: row.get(7)?,
                    created_at: row.get(8)?,
                })
            })
            .optional()?;
        Ok(res)
    }

    fn list_aliases(&self) -> Result<Vec<ModelAliasRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT alias_id, source, namespace, name, tag,
                    manifest_digest, weight_digest, alias_path, created_at
             FROM model_aliases ORDER BY created_at ASC;",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ModelAliasRecord {
                alias_id: row.get(0)?,
                source: row.get(1)?,
                namespace: row.get(2)?,
                name: row.get(3)?,
                tag: row.get(4)?,
                manifest_digest: row.get(5)?,
                weight_digest: row.get(6)?,
                alias_path: row.get(7)?,
                created_at: row.get(8)?,
            })
        })?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    fn delete_alias(&self, alias_id: &str) -> Result<bool, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let affected = conn.execute(
            "DELETE FROM model_aliases WHERE alias_id = ?1;",
            params![alias_id],
        )?;
        Ok(affected > 0)
    }

    fn save_derived_blob(&self, derived: &DerivedBlobRecord) -> Result<(), PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute(
            "INSERT INTO derived_blobs (
                derived_digest, source_digest, recipe_version, target_layout, status, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT (derived_digest) DO UPDATE SET
                source_digest = excluded.source_digest,
                recipe_version = excluded.recipe_version,
                target_layout = excluded.target_layout,
                status = excluded.status,
                created_at = excluded.created_at;",
            params![
                derived.derived_digest,
                derived.source_digest,
                derived.recipe_version,
                derived.target_layout,
                derived.status,
                derived.created_at,
            ],
        )?;
        Ok(())
    }

    fn get_derived_blob(
        &self,
        source_digest: &str,
        recipe_version: &str,
        target_layout: &str,
    ) -> Result<Option<DerivedBlobRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT derived_digest, source_digest, recipe_version, target_layout, status, created_at
             FROM derived_blobs
             WHERE source_digest = ?1 AND recipe_version = ?2 AND target_layout = ?3;",
        )?;
        let res = stmt
            .query_row(
                params![source_digest, recipe_version, target_layout],
                |row| {
                    Ok(DerivedBlobRecord {
                        derived_digest: row.get(0)?,
                        source_digest: row.get(1)?,
                        recipe_version: row.get(2)?,
                        target_layout: row.get(3)?,
                        status: row.get(4)?,
                        created_at: row.get(5)?,
                    })
                },
            )
            .optional()?;
        Ok(res)
    }

    fn list_derived_blobs(&self) -> Result<Vec<DerivedBlobRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT derived_digest, source_digest, recipe_version, target_layout, status, created_at
             FROM derived_blobs ORDER BY created_at ASC;",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(DerivedBlobRecord {
                derived_digest: row.get(0)?,
                source_digest: row.get(1)?,
                recipe_version: row.get(2)?,
                target_layout: row.get(3)?,
                status: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    fn list_derived_for_source(
        &self,
        source_digest: &str,
    ) -> Result<Vec<DerivedBlobRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT derived_digest, source_digest, recipe_version, target_layout, status, created_at
             FROM derived_blobs WHERE source_digest = ?1 ORDER BY created_at ASC;",
        )?;
        let rows = stmt.query_map(params![source_digest], |row| {
            Ok(DerivedBlobRecord {
                derived_digest: row.get(0)?,
                source_digest: row.get(1)?,
                recipe_version: row.get(2)?,
                target_layout: row.get(3)?,
                status: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    fn update_derived_status(
        &self,
        derived_digest: &str,
        status: &str,
    ) -> Result<(), PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute(
            "UPDATE derived_blobs SET status = ?2 WHERE derived_digest = ?1;",
            params![derived_digest, status],
        )?;
        Ok(())
    }

    fn delete_derived_blob(&self, derived_digest: &str) -> Result<bool, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let affected = conn.execute(
            "DELETE FROM derived_blobs WHERE derived_digest = ?1;",
            params![derived_digest],
        )?;
        Ok(affected > 0)
    }

    fn add_blob_reference(&self, reference: &BlobReferenceRecord) -> Result<(), PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute(
            "INSERT INTO blob_references (ref_id, blob_digest, ref_type, ref_source, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT (ref_id) DO UPDATE SET
                 blob_digest = excluded.blob_digest,
                 ref_type = excluded.ref_type,
                 ref_source = excluded.ref_source,
                 created_at = excluded.created_at;",
            params![
                reference.ref_id,
                reference.blob_digest,
                reference.ref_type,
                reference.ref_source,
                reference.created_at,
            ],
        )?;
        conn.execute(
            "UPDATE model_blobs SET ref_count = (
                SELECT COUNT(*) FROM blob_references WHERE blob_references.blob_digest = ?1
            ) WHERE digest = ?1;",
            params![reference.blob_digest],
        )?;
        Ok(())
    }

    fn remove_blob_reference(&self, ref_id: &str) -> Result<bool, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let blob_opt: Option<String> = conn
            .query_row(
                "SELECT blob_digest FROM blob_references WHERE ref_id = ?1;",
                params![ref_id],
                |r| r.get(0),
            )
            .optional()?;
        let affected = conn.execute(
            "DELETE FROM blob_references WHERE ref_id = ?1;",
            params![ref_id],
        )?;
        if let Some(digest) = blob_opt {
            conn.execute(
                "UPDATE model_blobs SET ref_count = (
                    SELECT COUNT(*) FROM blob_references WHERE blob_references.blob_digest = ?1
                ) WHERE digest = ?1;",
                params![digest],
            )?;
        }
        Ok(affected > 0)
    }

    fn list_blob_references(
        &self,
        blob_digest: &str,
    ) -> Result<Vec<BlobReferenceRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT ref_id, blob_digest, ref_type, ref_source, created_at
             FROM blob_references WHERE blob_digest = ?1 ORDER BY created_at ASC;",
        )?;
        let rows = stmt.query_map(params![blob_digest], |row| {
            Ok(BlobReferenceRecord {
                ref_id: row.get(0)?,
                blob_digest: row.get(1)?,
                ref_type: row.get(2)?,
                ref_source: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    fn get_storage_config(&self, key: &str) -> Result<Option<String>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt =
            conn.prepare("SELECT config_value FROM storage_config WHERE config_key = ?1;")?;
        let res = stmt.query_row(params![key], |row| row.get(0)).optional()?;
        Ok(res)
    }

    fn set_storage_config(&self, key: &str, value: &str) -> Result<(), PersistenceError> {
        let now = Self::current_epoch_secs();
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        conn.execute(
            "INSERT INTO storage_config (config_key, config_value, updated_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT (config_key) DO UPDATE SET
                 config_value = excluded.config_value,
                 updated_at = excluded.updated_at;",
            params![key, value, now],
        )?;
        Ok(())
    }

    fn recalculate_ref_counts(&self) -> Result<(), PersistenceError> {
        let now = Self::current_epoch_secs();
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;

        // 1. Recalculate base reference counts from active references
        conn.execute_batch(
            "UPDATE model_blobs SET ref_count = (
                SELECT COUNT(*) FROM blob_references WHERE blob_references.blob_digest = model_blobs.digest
            );",
        )?;

        // 2. Enforce PurgeImmediate policy: set ref_count = 0 for blobs explicitly marked for immediate purge
        conn.execute(
            "UPDATE model_blobs SET ref_count = 0
             WHERE digest IN (
                 SELECT REPLACE(config_key, 'purge_immediate:', '')
                 FROM storage_config
                 WHERE config_key LIKE 'purge_immediate:%' AND config_value = 'true'
             );",
            [],
        )?;

        // 3. Enforce expired PurgeAfterDelay policy: set ref_count = 0 for blobs whose delay has expired
        conn.execute(
            "UPDATE model_blobs SET ref_count = 0
             WHERE digest IN (
                 SELECT REPLACE(config_key, 'purge_after:', '')
                 FROM storage_config
                 WHERE config_key LIKE 'purge_after:%'
                   AND TRY_CAST(config_value AS BIGINT) IS NOT NULL
                   AND TRY_CAST(config_value AS BIGINT) <= ?1
             );",
            params![now],
        )?;

        Ok(())
    }

    fn list_unreferenced_blobs(&self) -> Result<Vec<ModelBlobRecord>, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT digest, size_bytes, blob_type, created_at, last_accessed_at, ref_count
             FROM model_blobs WHERE ref_count = 0 ORDER BY created_at ASC;",
        )?;
        let rows = stmt.query_map([], |row| {
            let size_i64: i64 = row.get(1)?;
            let count_i32: i32 = row.get(5)?;
            Ok(ModelBlobRecord {
                digest: row.get(0)?,
                size_bytes: u64::try_from(size_i64.max(0)).unwrap_or(0),
                blob_type: row.get(2)?,
                created_at: row.get(3)?,
                last_accessed_at: row.get(4)?,
                ref_count: u32::try_from(count_i32.max(0)).unwrap_or(0),
            })
        })?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    fn total_blob_bytes(&self) -> Result<u64, PersistenceError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| PersistenceError::LockError(e.to_string()))?;
        let sum_opt: Option<i64> = conn
            .query_row(
                "SELECT CAST(COALESCE(SUM(size_bytes), 0) AS BIGINT) FROM model_blobs;",
                [],
                |r| r.get(0),
            )
            .optional()?;
        Ok(sum_opt.map_or(0, |v| u64::try_from(v.max(0)).unwrap_or(0)))
    }
}
