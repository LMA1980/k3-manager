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

            CREATE INDEX IF NOT EXISTS idx_projects_tenant ON projects(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_session_project ON session_last_turns(project_id);",
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
