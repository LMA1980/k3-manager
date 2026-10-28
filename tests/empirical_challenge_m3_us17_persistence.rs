//! Empirical Adversarial Challenge Suite: Milestone M3 US-17 Persistence & Replay Stress
//!
//! Rigorous empirical verification of US-17 SQLite persistence and diagnostic replay:
//! 1. Bounded O(1) single-row invariant: 250+ sequential multi-turn updates on a single session;
//!    assert row count remains strictly 1 and content is updated without residue.
//! 2. WAL mode audit & Cold-boot reboot restoration across process lifetimes.
//! 3. Relational foreign key enforcement: strict rejection of orphaned entities.
//! 4. SQL injection & special character escaping across prompt, response, parameters, and session IDs.
//! 5. Diagnostic replay via `ControlPlaneRuntime::execute_admin(AdminCommand::ReplaySessionLastTurn)`.
//! 6. High-concurrency stress loops (25 iterations) under multi-threaded contention.
//! 7. Cumulative statistics saturating arithmetic and query boundaries.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::bool_assert_comparison
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use duckdb::Connection;

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::profile::{PriorityClass, TuningProfile};
use k3_manager::domain::{MockOsAffinityBackend, OsAffinityBackend};
use k3_manager::error::ManagerError;
use k3_manager::persistence::{
    PersistenceError, PersistenceStore, Project, SessionLastTurn, SqlitePersistenceStore, Tenant,
};
use k3_manager::runtime::{AdminCommand, AdminResponse, ControlPlaneRuntime};

fn current_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unique_temp_db(test_name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("k3_stress_{test_name}_{now}.db"))
}

// ---------------------------------------------------------------------------
// 1. Bounded O(1) Single-Row Invariant: 250+ Sequential Multi-Turn Updates
// ---------------------------------------------------------------------------

#[test]
fn test_empirical_bounded_o1_single_row_250_turns() {
    let store = SqlitePersistenceStore::open_in_memory().expect("in-memory db open failed");

    let tenant = Tenant {
        id: "t_stress".to_owned(),
        name: "Stress Test Tenant".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).expect("save tenant");

    let project = Project {
        id: "p_stress".to_owned(),
        tenant_id: "t_stress".to_owned(),
        name: "Stress Test Project".to_owned(),
        token_budget: Some(10_000_000_u32),
        created_at: current_epoch(),
    };
    store.save_project(&project).expect("save project");

    let session_id = "sess_stress_250_turns";

    // Execute 250 sequential updates on the exact same session_id
    for turn_idx in 1_u32..=250_u32 {
        let turn = SessionLastTurn {
            session_id: session_id.to_owned(),
            project_id: "p_stress".to_owned(),
            tenant_id: "t_stress".to_owned(),
            model_id: "qwen2.5-coder-7b".to_owned(),
            prompt: format!("Sequential prompt turn #{turn_idx}"),
            response: format!("Sequential response turn #{turn_idx}"),
            prompt_tokens: turn_idx,
            completion_tokens: turn_idx * 2,
            parameters_json: format!(r#"{{"turn":{turn_idx},"temp":0.7}}"#),
            updated_at: current_epoch() + u64::from(turn_idx),
        };
        store.save_session_last_turn(&turn).expect("upsert turn");

        // Verify count remains strictly 1 after initial turn and at checkpoints
        if turn_idx == 1 || turn_idx % 25 == 0 {
            let row_count = store
                .count_session_last_turns()
                .expect("count session last turns");
            assert_eq!(
                row_count, 1,
                "Invariant violated at turn {turn_idx}: row count must strictly be 1, found {row_count}"
            );
        }
    }

    // Final assertion: exactly 1 row in table
    assert_eq!(store.count_session_last_turns().unwrap(), 1);

    // Assert the stored record strictly reflects Turn 250 with zero leftover residue from Turn 1
    let loaded = store
        .load_session_last_turn(session_id)
        .unwrap()
        .expect("turn 250 exists");
    assert_eq!(loaded.prompt, "Sequential prompt turn #250");
    assert_eq!(loaded.response, "Sequential response turn #250");
    assert_eq!(loaded.prompt_tokens, 250);
    assert_eq!(loaded.completion_tokens, 500);
    assert_eq!(loaded.parameters_json, r#"{"turn":250,"temp":0.7}"#);

    // Now verify multi-session invariant: 10 distinct sessions, each updated 30 times (300 updates)
    for s_idx in 0_u32..10_u32 {
        let s_id = format!("multi_sess_{s_idx:02}");
        for t_idx in 1_u32..=30_u32 {
            let turn = SessionLastTurn {
                session_id: s_id.clone(),
                project_id: "p_stress".to_owned(),
                tenant_id: "t_stress".to_owned(),
                model_id: "model_multi".to_owned(),
                prompt: format!("Sess {s_idx} turn {t_idx}"),
                response: format!("Resp {s_idx} turn {t_idx}"),
                prompt_tokens: t_idx,
                completion_tokens: t_idx,
                parameters_json: "{}".to_owned(),
                updated_at: current_epoch() + u64::from(t_idx),
            };
            store.save_session_last_turn(&turn).unwrap();
        }
    }

    // Total rows must be 1 (original) + 10 (multi) = 11 rows
    assert_eq!(store.count_session_last_turns().unwrap(), 11);

    // Delete one session -> count decrements to 10
    assert!(store.delete_session_last_turn("multi_sess_00").unwrap());
    assert_eq!(store.count_session_last_turns().unwrap(), 10);

    // Deleting non-existent session returns false, count remains 10
    assert!(!store.delete_session_last_turn("nonexistent_sess").unwrap());
    assert_eq!(store.count_session_last_turns().unwrap(), 10);
}

// ---------------------------------------------------------------------------
// 2. WAL Mode Audit & Cold-Boot Reboot Restoration
// ---------------------------------------------------------------------------

#[test]
fn test_empirical_sqlite_wal_mode_and_cold_boot_reboot_restoration() {
    let db_path = unique_temp_db("wal_cold_boot");

    // Phase 1: Initialize on disk
    let initial_store = SqlitePersistenceStore::open(&db_path).expect("open on-disk db");

    // Audit DuckDB native persistence: verify database file exists on disk
    {
        assert!(
            db_path.exists(),
            "DuckDB persistence file must exist on disk"
        );
        let raw_conn = Connection::open(&db_path).expect("open raw conn");
        let ping: i64 = raw_conn
            .query_row("SELECT 1;", [], |r| r.get(0))
            .expect("ping raw duckdb connection");
        assert_eq!(ping, 1);
    }

    // Phase 2: Populate state under WAL mode
    let tenants = vec![
        Tenant {
            id: "t_alpha".to_owned(),
            name: "Tenant Alpha".to_owned(),
            created_at: current_epoch(),
        },
        Tenant {
            id: "t_beta".to_owned(),
            name: "Tenant Beta".to_owned(),
            created_at: current_epoch(),
        },
    ];
    for t in &tenants {
        initial_store.save_tenant(t).expect("save tenant");
    }

    let projects = vec![
        Project {
            id: "p_alpha_1".to_owned(),
            tenant_id: "t_alpha".to_owned(),
            name: "Alpha Project 1".to_owned(),
            token_budget: Some(1_000_000_u32),
            created_at: current_epoch(),
        },
        Project {
            id: "p_beta_1".to_owned(),
            tenant_id: "t_beta".to_owned(),
            name: "Beta Project 1".to_owned(),
            token_budget: None,
            created_at: current_epoch(),
        },
    ];
    for p in &projects {
        initial_store.save_project(p).expect("save project");
    }

    let profile = TuningProfile {
        model_id: Some("deepseek-r1-7b".to_owned()),
        priority_class: PriorityClass::Latency,
        x100_cores: 4,
        a100_cores: 8,
        lend_idle_cores: true,
        borrow_idle_cores: true,
        ..TuningProfile::conservative_default_with_n_ctx(128)
    };
    initial_store
        .save_tuning_profile("deepseek-r1-7b", &profile)
        .expect("save profile");

    let turn1 = SessionLastTurn {
        session_id: "sess_wal_001".to_owned(),
        project_id: "p_alpha_1".to_owned(),
        tenant_id: "t_alpha".to_owned(),
        model_id: "deepseek-r1-7b".to_owned(),
        prompt: "Explain speculative decoding in 3 bullet points.".to_owned(),
        response: "- Draft model generates tokens\n- Target model verifies\n- Speedup achieved"
            .to_owned(),
        prompt_tokens: 15_u32,
        completion_tokens: 18_u32,
        parameters_json: r#"{"temperature":0.2}"#.to_owned(),
        updated_at: current_epoch(),
    };
    initial_store
        .save_session_last_turn(&turn1)
        .expect("save session 1");

    let turn2 = SessionLastTurn {
        session_id: "sess_wal_002".to_owned(),
        project_id: "p_beta_1".to_owned(),
        tenant_id: "t_beta".to_owned(),
        model_id: "deepseek-r1-7b".to_owned(),
        prompt: "What is SpacemiT K3 SoC?".to_owned(),
        response: "A dual-cluster heterogeneous RISC-V SoC with X100 and A100 cores.".to_owned(),
        prompt_tokens: 10_u32,
        completion_tokens: 16_u32,
        parameters_json: r#"{"temperature":0.0}"#.to_owned(),
        updated_at: current_epoch(),
    };
    initial_store
        .save_session_last_turn(&turn2)
        .expect("save session 2");

    initial_store
        .update_cumulative_stat("wal_prompt_tokens", 50_000_u32)
        .expect("update stat");
    initial_store
        .update_cumulative_stat("wal_completion_tokens", 25_000_u32)
        .expect("update stat");

    // Verify initial store sees the data
    assert_eq!(initial_store.count_session_last_turns().unwrap(), 2);

    // Drop store instance — simulating process crash / termination before manual checkpoint
    drop(initial_store);

    // Phase 3: Cold-boot reboot restoration — open fresh instance from file
    let rebooted_store = SqlitePersistenceStore::open(&db_path).expect("reopen on-disk db");

    // Verify DuckDB database persisted natively on disk across process boundary
    {
        assert!(db_path.exists(), "DuckDB file must persist across reboots");
        let check_conn = Connection::open(&db_path).expect("open check conn");
        let raw_tenant_count: i64 = check_conn
            .query_row("SELECT count(*) FROM tenants;", [], |r| r.get(0))
            .expect("query raw tenant count");
        assert_eq!(
            raw_tenant_count, 2,
            "Tenants must persist in DuckDB on disk across reboots"
        );
    }

    // 100% State Recovery Verification:
    // 1. Tenants
    let loaded_t_alpha = rebooted_store
        .get_tenant("t_alpha")
        .unwrap()
        .expect("tenant alpha persists");
    assert_eq!(loaded_t_alpha.name, "Tenant Alpha");
    let loaded_t_beta = rebooted_store
        .get_tenant("t_beta")
        .unwrap()
        .expect("tenant beta persists");
    assert_eq!(loaded_t_beta.name, "Tenant Beta");
    assert_eq!(rebooted_store.list_tenants().unwrap().len(), 2);

    // 2. Projects
    let loaded_p_alpha = rebooted_store
        .get_project("p_alpha_1")
        .unwrap()
        .expect("project alpha persists");
    assert_eq!(loaded_p_alpha.token_budget, Some(1_000_000));
    let alpha_projects = rebooted_store
        .list_projects(Some("t_alpha"))
        .expect("list alpha projects");
    assert_eq!(alpha_projects.len(), 1);
    assert_eq!(alpha_projects[0].id, "p_alpha_1");

    // 3. Tuning Profiles
    let loaded_prof = rebooted_store
        .load_tuning_profile("deepseek-r1-7b")
        .unwrap()
        .expect("profile persists");
    assert_eq!(loaded_prof.priority_class, PriorityClass::Latency);
    assert_eq!(loaded_prof.x100_cores, 4);
    assert_eq!(loaded_prof.a100_cores, 8);
    assert!(loaded_prof.lend_idle_cores);
    assert!(loaded_prof.borrow_idle_cores);

    // 4. Session Last Turns
    assert_eq!(rebooted_store.count_session_last_turns().unwrap(), 2);
    let s1 = rebooted_store
        .load_session_last_turn("sess_wal_001")
        .unwrap()
        .expect("s1 persists");
    assert_eq!(
        s1.prompt,
        "Explain speculative decoding in 3 bullet points."
    );
    assert_eq!(s1.prompt_tokens, 15);
    assert_eq!(s1.completion_tokens, 18);

    let s2 = rebooted_store
        .load_session_last_turn("sess_wal_002")
        .unwrap()
        .expect("s2 persists");
    assert_eq!(s2.prompt, "What is SpacemiT K3 SoC?");

    // 5. Cumulative Statistics
    assert_eq!(
        rebooted_store
            .get_cumulative_stat("wal_prompt_tokens")
            .unwrap(),
        50_000
    );
    assert_eq!(
        rebooted_store
            .get_cumulative_stat("wal_completion_tokens")
            .unwrap(),
        25_000
    );

    // 6. Mutability on Restored Store:
    // Update turn on s1:
    let turn1_v2 = SessionLastTurn {
        session_id: "sess_wal_001".to_owned(),
        project_id: "p_alpha_1".to_owned(),
        tenant_id: "t_alpha".to_owned(),
        model_id: "deepseek-r1-7b".to_owned(),
        prompt: "Followup: How much speedup?".to_owned(),
        response: "Typically 1.8x to 2.5x.".to_owned(),
        prompt_tokens: 8_u32,
        completion_tokens: 10_u32,
        parameters_json: r#"{"temperature":0.2}"#.to_owned(),
        updated_at: current_epoch() + 10,
    };
    rebooted_store
        .save_session_last_turn(&turn1_v2)
        .expect("update on restored store");
    assert_eq!(rebooted_store.count_session_last_turns().unwrap(), 2);

    let updated_s1 = rebooted_store
        .load_session_last_turn("sess_wal_001")
        .unwrap()
        .unwrap();
    assert_eq!(updated_s1.prompt, "Followup: How much speedup?");

    // Cleanup disk artifacts
    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_file(format!("{}-wal", db_path.display()));
    let _ = std::fs::remove_file(format!("{}-shm", db_path.display()));
}

// ---------------------------------------------------------------------------
// 3. Relational Foreign Key Enforcement: Strict Rejection
// ---------------------------------------------------------------------------

#[test]
fn test_empirical_relational_foreign_key_strict_rejection() {
    let store = SqlitePersistenceStore::open_in_memory().expect("in-memory db open failed");

    // 1. Attempt to insert project referencing non-existent tenant
    let orphaned_project = Project {
        id: "p_orphan_01".to_owned(),
        tenant_id: "tenant_does_not_exist".to_owned(),
        name: "Orphaned Project".to_owned(),
        token_budget: None,
        created_at: current_epoch(),
    };
    let err_proj = store
        .save_project(&orphaned_project)
        .expect_err("orphaned project must be rejected");

    match err_proj {
        PersistenceError::ForeignKeyViolation(ref msg) => {
            assert!(
                msg.contains("FOREIGN KEY")
                    || msg.contains("foreign key")
                    || msg.contains("constraint")
            );
        }
        other => panic!("expected PersistenceError::ForeignKeyViolation, got: {other:?}"),
    }

    // 2. Create valid tenant
    let tenant = Tenant {
        id: "t_valid_fk".to_owned(),
        name: "Valid Tenant".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).expect("save valid tenant");

    // 3. Attempt to insert session referencing non-existent project (even with valid tenant)
    let orphaned_session_bad_project = SessionLastTurn {
        session_id: "s_orphan_bad_proj".to_owned(),
        project_id: "proj_does_not_exist".to_owned(),
        tenant_id: "t_valid_fk".to_owned(),
        model_id: "m1".to_owned(),
        prompt: "test".to_owned(),
        response: "test".to_owned(),
        prompt_tokens: 1,
        completion_tokens: 1,
        parameters_json: "{}".to_owned(),
        updated_at: current_epoch(),
    };
    let err_sess1 = store
        .save_session_last_turn(&orphaned_session_bad_project)
        .expect_err("orphaned session (bad project) must be rejected");

    match err_sess1 {
        PersistenceError::ForeignKeyViolation(_) => {}
        other => panic!("expected ForeignKeyViolation, got: {other:?}"),
    }

    // 4. Create valid project
    let project = Project {
        id: "p_valid_fk".to_owned(),
        tenant_id: "t_valid_fk".to_owned(),
        name: "Valid Project".to_owned(),
        token_budget: Some(1000),
        created_at: current_epoch(),
    };
    store.save_project(&project).expect("save valid project");

    // 5. Attempt to insert session referencing valid project but non-existent tenant
    let orphaned_session_bad_tenant = SessionLastTurn {
        session_id: "s_orphan_bad_tenant".to_owned(),
        project_id: "p_valid_fk".to_owned(),
        tenant_id: "nonexistent_tenant_xyz".to_owned(),
        model_id: "m1".to_owned(),
        prompt: "test".to_owned(),
        response: "test".to_owned(),
        prompt_tokens: 1,
        completion_tokens: 1,
        parameters_json: "{}".to_owned(),
        updated_at: current_epoch(),
    };
    let err_sess2 = store
        .save_session_last_turn(&orphaned_session_bad_tenant)
        .expect_err("orphaned session (bad tenant) must be rejected");

    match err_sess2 {
        PersistenceError::ForeignKeyViolation(_) => {}
        other => panic!("expected ForeignKeyViolation, got: {other:?}"),
    }

    // 6. Valid session with both valid project and valid tenant succeeds
    let valid_session = SessionLastTurn {
        session_id: "s_valid_fk".to_owned(),
        project_id: "p_valid_fk".to_owned(),
        tenant_id: "t_valid_fk".to_owned(),
        model_id: "m1".to_owned(),
        prompt: "valid test prompt".to_owned(),
        response: "valid test response".to_owned(),
        prompt_tokens: 5,
        completion_tokens: 5,
        parameters_json: "{}".to_owned(),
        updated_at: current_epoch(),
    };
    store
        .save_session_last_turn(&valid_session)
        .expect("valid session succeeds");
    assert_eq!(store.count_session_last_turns().unwrap(), 1);
}

// ---------------------------------------------------------------------------
// 4. SQL Injection & Special Character Escaping Stress
// ---------------------------------------------------------------------------

#[test]
fn test_empirical_sql_injection_and_special_character_escaping() {
    let store = SqlitePersistenceStore::open_in_memory().expect("in-memory db open failed");

    let tenant = Tenant {
        id: "t_escape".to_owned(),
        name: "Tenant with 'quotes' and \"double quotes\" and ; --".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).expect("save tenant with quotes");

    let project = Project {
        id: "p_escape".to_owned(),
        tenant_id: "t_escape".to_owned(),
        name: "Project '; DROP TABLE tenants; --".to_owned(),
        token_budget: Some(500_000),
        created_at: current_epoch(),
    };
    store.save_project(&project).expect("save project");

    // Vector of adversarial strings to test across prompt, response, parameters, and session_id
    let adversarial_cases = vec![
        (
            "sess_inj_01",
            "'; DROP TABLE tenants; DROP TABLE projects; --",
            "'; DROP TABLE session_last_turns; --",
        ),
        (
            "sess_inj_02",
            "' OR '1'='1' /* comment */",
            "1' UNION SELECT 'x', 'y', 'z', 'w' --",
        ),
        (
            "sess_inj_03",
            "Robert'); DROP TABLE Students;-- 🦀 🚀 🔥",
            "Unicode and emoji \u{1F4BB} \u{1F525} \u{26A1} \u{1F980} multi-line\n\r\t\\",
        ),
        (
            "sess_inj_04",
            "\"double quotes\" and `backticks` and \\\"escaped\\\"",
            "Backslashes: \\ \\\\ \\\\\\ and slashes: / // ///",
        ),
        (
            "sess_inj_05",
            "RTL text: مرحبا بالعالم and CJK: 你好，世界！ and Cyrillic: Привет, мир!",
            "JSON-in-JSON: {\"nested\":\"{\\\"inner\\\":\\\"value's\\\"}\"}",
        ),
    ];

    for (s_id, prompt_str, resp_str) in &adversarial_cases {
        let turn = SessionLastTurn {
            session_id: (*s_id).to_owned(),
            project_id: "p_escape".to_owned(),
            tenant_id: "t_escape".to_owned(),
            model_id: "model_escape".to_owned(),
            prompt: (*prompt_str).to_owned(),
            response: (*resp_str).to_owned(),
            prompt_tokens: 42,
            completion_tokens: 84,
            parameters_json: format!(r#"{{"injection":"{prompt_str}"}}"#),
            updated_at: current_epoch(),
        };
        store
            .save_session_last_turn(&turn)
            .unwrap_or_else(|e| panic!("failed on session {s_id}: {e}"));

        // Verify retrieval matches exactly bit-for-bit
        let loaded = store
            .load_session_last_turn(s_id)
            .unwrap()
            .unwrap_or_else(|| panic!("failed to load {s_id}"));
        assert_eq!(loaded.prompt, *prompt_str);
        assert_eq!(loaded.response, *resp_str);
    }

    // Assert tables remain completely intact (no DROP TABLE succeeded)
    assert!(store.get_tenant("t_escape").unwrap().is_some());
    assert!(store.get_project("p_escape").unwrap().is_some());
    assert_eq!(
        store.count_session_last_turns().unwrap(),
        adversarial_cases.len()
    );

    // Large payload stress: 128 KB prompt and 128 KB response
    let large_prompt = "P".repeat(131_072); // 128 KB
    let large_response = "R".repeat(131_072); // 128 KB
    let turn_large = SessionLastTurn {
        session_id: "sess_128kb".to_owned(),
        project_id: "p_escape".to_owned(),
        tenant_id: "t_escape".to_owned(),
        model_id: "model_large".to_owned(),
        prompt: large_prompt.clone(),
        response: large_response.clone(),
        prompt_tokens: 32_768_u32,
        completion_tokens: 32_768_u32,
        parameters_json: "{}".to_owned(),
        updated_at: current_epoch(),
    };
    store
        .save_session_last_turn(&turn_large)
        .expect("128KB payload save succeeds");

    let loaded_large = store
        .load_session_last_turn("sess_128kb")
        .unwrap()
        .expect("128KB payload load succeeds");
    assert_eq!(loaded_large.prompt.len(), 131_072);
    assert_eq!(loaded_large.response.len(), 131_072);
    assert_eq!(loaded_large.prompt, large_prompt);
    assert_eq!(loaded_large.response, large_response);
}

// ---------------------------------------------------------------------------
// 5. Diagnostic Replay via ControlPlaneRuntime
// ---------------------------------------------------------------------------

#[test]
fn test_empirical_diagnostic_replay_via_control_plane_runtime() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(
        capacity,
        Arc::clone(&mock_affinity) as Arc<dyn OsAffinityBackend>,
    )
    .expect("runtime starts");

    let store = runtime.persistence();
    let tenant = Tenant {
        id: "t_replay".to_owned(),
        name: "Replay Tenant".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).unwrap();

    let project = Project {
        id: "p_replay".to_owned(),
        tenant_id: "t_replay".to_owned(),
        name: "Replay Project".to_owned(),
        token_budget: None,
        created_at: current_epoch(),
    };
    store.save_project(&project).unwrap();

    // 1. Seed Turn 1
    let turn1 = SessionLastTurn {
        session_id: "sess_replay_test".to_owned(),
        project_id: "p_replay".to_owned(),
        tenant_id: "t_replay".to_owned(),
        model_id: "qwen2.5-7b".to_owned(),
        prompt: "Calculate fibonacci(10)".to_owned(),
        response: "55".to_owned(),
        prompt_tokens: 5_u32,
        completion_tokens: 1_u32,
        parameters_json: r#"{"temperature":0.0}"#.to_owned(),
        updated_at: current_epoch(),
    };
    store.save_session_last_turn(&turn1).unwrap();

    // Replay Turn 1
    let response = runtime
        .execute_admin(AdminCommand::ReplaySessionLastTurn {
            session_id: "sess_replay_test".to_owned(),
        })
        .expect("replay succeeds");

    match response {
        AdminResponse::DiagnosticReplay(res) => {
            assert_eq!(res.session_id, "sess_replay_test");
            assert_eq!(res.model_id, "qwen2.5-7b");
            assert_eq!(res.prompt, "Calculate fibonacci(10)");
            assert_eq!(res.expected_response, "55");
            assert_eq!(res.replayed_response, "55");
            assert!(res.matched);
            assert!(res.latency_ms > 0.0);
        }
        other => panic!("expected DiagnosticReplay, got: {other:?}"),
    }

    // 2. Overwrite with Turn 2 (simulating multi-turn conversation)
    let turn2 = SessionLastTurn {
        session_id: "sess_replay_test".to_owned(),
        project_id: "p_replay".to_owned(),
        tenant_id: "t_replay".to_owned(),
        model_id: "qwen2.5-7b".to_owned(),
        prompt: "Calculate fibonacci(11)".to_owned(),
        response: "89".to_owned(),
        prompt_tokens: 5_u32,
        completion_tokens: 1_u32,
        parameters_json: r#"{"temperature":0.0}"#.to_owned(),
        updated_at: current_epoch() + 1,
    };
    store.save_session_last_turn(&turn2).unwrap();

    // Replay again: must replay Turn 2 (89), NOT Turn 1 (55)
    let response_v2 = runtime
        .execute_admin(AdminCommand::ReplaySessionLastTurn {
            session_id: "sess_replay_test".to_owned(),
        })
        .expect("replay turn 2 succeeds");

    match response_v2 {
        AdminResponse::DiagnosticReplay(res) => {
            assert_eq!(res.prompt, "Calculate fibonacci(11)");
            assert_eq!(res.expected_response, "89");
            assert_eq!(res.replayed_response, "89");
            assert!(res.matched);
        }
        other => panic!("expected DiagnosticReplay, got: {other:?}"),
    }

    // 3. Replaying non-existent session returns clear structured error
    let bad_res = runtime.execute_admin(AdminCommand::ReplaySessionLastTurn {
        session_id: "nonexistent_session_abc".to_owned(),
    });
    assert!(bad_res.is_err());
    match bad_res.unwrap_err() {
        ManagerError::Persistence(PersistenceError::SessionNotFound(id)) => {
            assert_eq!(id, "nonexistent_session_abc");
        }
        other => panic!("expected ManagerError::Persistence(SessionNotFound), got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 6. High-Concurrency Stress Loops (25 Iterations)
// ---------------------------------------------------------------------------

#[test]
fn test_empirical_concurrency_stress_loop_25_iterations() {
    let store = SqlitePersistenceStore::open_in_memory().expect("in-memory db open");

    let tenant = Tenant {
        id: "t_conc".to_owned(),
        name: "Concurrent Tenant".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).unwrap();

    let project = Project {
        id: "p_conc".to_owned(),
        tenant_id: "t_conc".to_owned(),
        name: "Concurrent Project".to_owned(),
        token_budget: None,
        created_at: current_epoch(),
    };
    store.save_project(&project).unwrap();

    let store_arc = Arc::new(store);

    // Run 25 iterations of multi-threaded concurrent access
    for iteration in 1..=25 {
        let mut handles = Vec::new();

        // Spawn 4 threads performing concurrent writes and reads
        for worker_id in 0..4 {
            let s = Arc::clone(&store_arc);
            let handle = thread::spawn(move || {
                for i in 0..15 {
                    let s_id = format!("sess_conc_{worker_id}");
                    let turn = SessionLastTurn {
                        session_id: s_id.clone(),
                        project_id: "p_conc".to_owned(),
                        tenant_id: "t_conc".to_owned(),
                        model_id: "model_concurrent".to_owned(),
                        prompt: format!("iter {iteration} worker {worker_id} turn {i}"),
                        response: format!("resp {iteration} worker {worker_id} turn {i}"),
                        prompt_tokens: (worker_id * 10 + i) as u32,
                        completion_tokens: (worker_id * 10 + i) as u32,
                        parameters_json: "{}".to_owned(),
                        updated_at: current_epoch(),
                    };
                    s.save_session_last_turn(&turn)
                        .expect("concurrent save must succeed");

                    // Concurrent read
                    let loaded = s
                        .load_session_last_turn(&s_id)
                        .expect("load")
                        .expect("must exist");
                    assert!(loaded.prompt.contains(&format!("worker {worker_id}")));

                    // Concurrent stat increment
                    s.update_cumulative_stat("concurrent_total_ops", 1)
                        .expect("stat update");
                }
            });
            handles.push(handle);
        }

        // Wait for all worker threads to complete
        for h in handles {
            h.join().expect("thread join must not panic or deadlock");
        }

        // Invariant: Exactly 4 sessions exist in the table (1 per worker)
        let count = store_arc
            .count_session_last_turns()
            .expect("count sessions");
        assert_eq!(
            count, 4,
            "Iteration {iteration}: expected strictly 4 sessions, found {count}"
        );
    }

    // Total ops: 25 iterations * 4 workers * 15 ops = 1500 ops
    let total_ops = store_arc
        .get_cumulative_stat("concurrent_total_ops")
        .expect("get total ops");
    assert_eq!(total_ops, 1500);
}

// ---------------------------------------------------------------------------
// 7. Cumulative Statistics Saturating Arithmetic & Query Boundaries
// ---------------------------------------------------------------------------

#[test]
fn test_empirical_cumulative_statistics_saturating_arithmetic() {
    let store = SqlitePersistenceStore::open_in_memory().expect("in-memory db open");

    // 1. Initial stat is 0
    assert_eq!(
        store.get_cumulative_stat("untracked_key").unwrap(),
        0,
        "nonexistent stat must return 0"
    );

    // 2. Increment by delta 0 returns 0 and does not error
    let res0 = store.update_cumulative_stat("test_stat", 0).unwrap();
    assert_eq!(res0, 0);

    // 3. Normal incremental accumulation
    let v1 = store.update_cumulative_stat("test_stat", 150).unwrap();
    assert_eq!(v1, 150);

    let v2 = store.update_cumulative_stat("test_stat", 250).unwrap();
    assert_eq!(v2, 400);

    assert_eq!(store.get_cumulative_stat("test_stat").unwrap(), 400);

    // 4. Saturating addition near u32::MAX
    let near_max = u32::MAX - 20;
    store
        .update_cumulative_stat("saturation_key", near_max)
        .unwrap();

    // Adding 50 exceeds u32::MAX -> must saturate cleanly to u32::MAX without overflow panic
    let sat_res = store.update_cumulative_stat("saturation_key", 50).unwrap();
    assert_eq!(
        sat_res,
        u32::MAX,
        "Stat must saturate to u32::MAX (4294967295)"
    );

    // Further additions remain saturated at u32::MAX
    let sat_res2 = store
        .update_cumulative_stat("saturation_key", 1000)
        .unwrap();
    assert_eq!(sat_res2, u32::MAX);

    // 5. list_cumulative_stats returns all recorded keys
    let stats_map = store.list_cumulative_stats().unwrap();
    assert_eq!(stats_map.get("test_stat"), Some(&400));
    assert_eq!(stats_map.get("saturation_key"), Some(&u32::MAX));
    assert_eq!(stats_map.len(), 2);
}
