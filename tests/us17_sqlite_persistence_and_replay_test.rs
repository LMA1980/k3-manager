//! US-17 Dedicated Scenario Test Suite: SQLite Metadata & Bounded Session Persistence (Features 32, 33, 34, ADR-001, ADR-003).
//!
//! Verifies:
//! - Scenario 1: Operational configuration (tenants, projects, budgets, tuning profiles, cumulative statistics)
//!   restored across daemon restarts without operator reconfiguration.
//! - Scenario 2: Bounded single-row last-turn retention per session via SQL upsert (`ON CONFLICT(session_id) DO UPDATE SET`).
//!   Sequential turns strictly overwrite previous prompt and response, keeping row count == 1.
//! - Scenario 3: Diagnostic replay of persisted last interaction across daemon lifetimes.
//! - Relational foreign key integrity enforcement (`PRAGMA foreign_keys = ON;`).
//! - SQL escaping robustness against adversarial injection strings and large 64KB payloads.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

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
        .unwrap()
        .as_secs()
}

#[test]
fn test_scenario1_tenants_and_projects_relational_schema() {
    let store = SqlitePersistenceStore::open_in_memory().expect("in-memory db must open");

    let tenant = Tenant {
        id: "tenant_k3_org".to_owned(),
        name: "K3 Core Systems".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).expect("save tenant succeeds");

    let project = Project {
        id: "proj_spacemit_eval".to_owned(),
        tenant_id: "tenant_k3_org".to_owned(),
        name: "SpacemiT SoC Evaluation".to_owned(),
        token_budget: Some(1_000_000_u32), // u32 per AGENTS.md §9
        created_at: current_epoch(),
    };
    store.save_project(&project).expect("save project succeeds");

    // Fetch and verify tenant
    let loaded_tenant = store
        .get_tenant("tenant_k3_org")
        .unwrap()
        .expect("tenant exists");
    assert_eq!(loaded_tenant.name, "K3 Core Systems");

    // Fetch and verify project
    let loaded_project = store
        .get_project("proj_spacemit_eval")
        .unwrap()
        .expect("project exists");
    assert_eq!(loaded_project.name, "SpacemiT SoC Evaluation");
    assert_eq!(loaded_project.token_budget, Some(1_000_000));
}

#[test]
fn test_scenario1_cold_boot_power_cycle_persistence_across_restart() {
    let unique_id = format!(
        "{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let temp_db = std::env::temp_dir().join(format!("k3_test_cold_boot_{unique_id}.db"));

    // Phase 1: First daemon lifetime — establish configuration
    {
        let store = SqlitePersistenceStore::open(&temp_db).expect("create on-disk db");

        let tenant = Tenant {
            id: "t_reboot".to_owned(),
            name: "Reboot Tenant".to_owned(),
            created_at: current_epoch(),
        };
        store.save_tenant(&tenant).unwrap();

        let project = Project {
            id: "p_reboot".to_owned(),
            tenant_id: "t_reboot".to_owned(),
            name: "Reboot Project".to_owned(),
            token_budget: Some(500_000_u32),
            created_at: current_epoch(),
        };
        store.save_project(&project).unwrap();

        let profile = TuningProfile {
            model_id: Some("qwen2.5-coder-7b".to_owned()),
            priority_class: PriorityClass::Latency,
            x100_cores: 4,
            a100_cores: 0,
            lend_idle_cores: true,
            borrow_idle_cores: false,
            ..TuningProfile::conservative_default_with_n_ctx(64)
        };
        store
            .save_tuning_profile("qwen2.5-coder-7b", &profile)
            .unwrap();

        // Record cumulative statistics (u32 per AGENTS.md §9)
        store
            .update_cumulative_stat("total_prompt_tokens", 54321_u32)
            .unwrap();
        store
            .update_cumulative_stat("total_completion_tokens", 12345_u32)
            .unwrap();

        // Record session last turn
        let turn = SessionLastTurn {
            session_id: "sess_reboot_test".to_owned(),
            project_id: "p_reboot".to_owned(),
            tenant_id: "t_reboot".to_owned(),
            model_id: "qwen2.5-coder-7b".to_owned(),
            prompt: "What is 2+2?".to_owned(),
            response: "4".to_owned(),
            prompt_tokens: 6_u32,
            completion_tokens: 1_u32,
            parameters_json: r#"{"temperature":0.0}"#.to_owned(),
            updated_at: current_epoch(),
        };
        store.save_session_last_turn(&turn).unwrap();
    } // Store is closed / dropped (simulating daemon termination / power cycle)

    // Phase 2: Second daemon lifetime — restore from disk without reconfiguration
    {
        let restored_store = SqlitePersistenceStore::open(&temp_db).expect("reopen on-disk db");

        // Verify tenant & project restoration
        let proj = restored_store
            .get_project("p_reboot")
            .unwrap()
            .expect("project persists");
        assert_eq!(proj.token_budget, Some(500_000));

        // Verify tuning profile restoration
        let restored_profile = restored_store
            .load_tuning_profile("qwen2.5-coder-7b")
            .unwrap()
            .expect("tuning profile persists");
        assert_eq!(restored_profile.priority_class, PriorityClass::Latency);
        assert_eq!(restored_profile.x100_cores, 4);
        assert!(restored_profile.lend_idle_cores);

        // Verify cumulative statistics restoration
        assert_eq!(
            restored_store
                .get_cumulative_stat("total_prompt_tokens")
                .unwrap(),
            54321
        );
        assert_eq!(
            restored_store
                .get_cumulative_stat("total_completion_tokens")
                .unwrap(),
            12345
        );

        // Verify session last-turn restoration
        let last_turn = restored_store
            .load_session_last_turn("sess_reboot_test")
            .unwrap()
            .expect("last turn persists");
        assert_eq!(last_turn.prompt, "What is 2+2?");
        assert_eq!(last_turn.response, "4");
    }

    let _ = std::fs::remove_file(temp_db);
}

#[test]
fn test_scenario2_bounded_single_row_last_turn_overwrite_invariant() {
    let store = SqlitePersistenceStore::open_in_memory().unwrap();

    let tenant = Tenant {
        id: "t1".to_owned(),
        name: "T1".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).unwrap();

    let project = Project {
        id: "p1".to_owned(),
        tenant_id: "t1".to_owned(),
        name: "P1".to_owned(),
        token_budget: None,
        created_at: current_epoch(),
    };
    store.save_project(&project).unwrap();

    let session_id = "sess_bounded_check";

    // Turn 1
    let turn1 = SessionLastTurn {
        session_id: session_id.to_owned(),
        project_id: "p1".to_owned(),
        tenant_id: "t1".to_owned(),
        model_id: "model_v1".to_owned(),
        prompt: "First prompt turn".to_owned(),
        response: "First response turn".to_owned(),
        prompt_tokens: 10_u32,
        completion_tokens: 12_u32,
        parameters_json: "{}".to_owned(),
        updated_at: current_epoch(),
    };
    store.save_session_last_turn(&turn1).unwrap();
    assert_eq!(store.count_session_last_turns().unwrap(), 1);

    // Turn 2: Overwrites Turn 1 via SQL upsert
    let turn2 = SessionLastTurn {
        session_id: session_id.to_owned(),
        project_id: "p1".to_owned(),
        tenant_id: "t1".to_owned(),
        model_id: "model_v1".to_owned(),
        prompt: "Second prompt turn (overwrites Turn 1)".to_owned(),
        response: "Second response turn".to_owned(),
        prompt_tokens: 25_u32,
        completion_tokens: 30_u32,
        parameters_json: "{}".to_owned(),
        updated_at: current_epoch() + 1,
    };
    store.save_session_last_turn(&turn2).unwrap();

    // Verify row count remains strictly bounded to 1
    assert_eq!(store.count_session_last_turns().unwrap(), 1);

    // Verify loaded content is strictly Turn 2
    let loaded = store.load_session_last_turn(session_id).unwrap().unwrap();
    assert_eq!(loaded.prompt, "Second prompt turn (overwrites Turn 1)");
    assert_eq!(loaded.response, "Second response turn");
    assert_eq!(loaded.prompt_tokens, 25);
    assert_eq!(loaded.completion_tokens, 30);
}

#[test]
fn test_scenario2_100_turns_remains_strictly_one_row() {
    let store = SqlitePersistenceStore::open_in_memory().unwrap();

    let tenant = Tenant {
        id: "t1".to_owned(),
        name: "T1".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).unwrap();

    let project = Project {
        id: "p1".to_owned(),
        tenant_id: "t1".to_owned(),
        name: "P1".to_owned(),
        token_budget: None,
        created_at: current_epoch(),
    };
    store.save_project(&project).unwrap();

    let session_id = "sess_100_turn_invariant";

    // Run 100 sequential turns on the same session ID
    for turn_idx in 1_u32..=100_u32 {
        let turn = SessionLastTurn {
            session_id: session_id.to_owned(),
            project_id: "p1".to_owned(),
            tenant_id: "t1".to_owned(),
            model_id: "model_v1".to_owned(),
            prompt: format!("Turn prompt {turn_idx}"),
            response: format!("Turn response {turn_idx}"),
            prompt_tokens: turn_idx,
            completion_tokens: turn_idx * 2,
            parameters_json: "{}".to_owned(),
            updated_at: current_epoch() + u64::from(turn_idx),
        };
        store.save_session_last_turn(&turn).unwrap();
    }

    // Row count must mathematically remain 1
    assert_eq!(store.count_session_last_turns().unwrap(), 1);

    // Content must be the 100th turn
    let loaded = store.load_session_last_turn(session_id).unwrap().unwrap();
    assert_eq!(loaded.prompt, "Turn prompt 100");
    assert_eq!(loaded.response, "Turn response 100");
    assert_eq!(loaded.prompt_tokens, 100);
    assert_eq!(loaded.completion_tokens, 200);
}

#[test]
fn test_scenario3_diagnostic_replay_via_control_plane_runtime() {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid capacity");
    let mock_affinity = Arc::new(MockOsAffinityBackend::default());
    let runtime = ControlPlaneRuntime::with_affinity_backend(
        capacity,
        Arc::clone(&mock_affinity) as Arc<dyn OsAffinityBackend>,
    )
    .expect("runtime starts");

    // Seed test session in persistence
    let store = runtime.persistence();
    let tenant = Tenant {
        id: "diag_tenant".to_owned(),
        name: "Diag Tenant".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).unwrap();

    let project = Project {
        id: "diag_project".to_owned(),
        tenant_id: "diag_tenant".to_owned(),
        name: "Diag Project".to_owned(),
        token_budget: None,
        created_at: current_epoch(),
    };
    store.save_project(&project).unwrap();

    let turn = SessionLastTurn {
        session_id: "diag_sess_001".to_owned(),
        project_id: "diag_project".to_owned(),
        tenant_id: "diag_tenant".to_owned(),
        model_id: "qwen2.5-7b".to_owned(),
        prompt: "Write a deterministic hello function in Rust".to_owned(),
        response: "fn hello() { println!(\"hello\"); }".to_owned(),
        prompt_tokens: 12_u32,
        completion_tokens: 14_u32,
        parameters_json: r#"{"temperature":0.0}"#.to_owned(),
        updated_at: current_epoch(),
    };
    store.save_session_last_turn(&turn).unwrap();

    // Execute diagnostic replay command
    let replay_cmd = AdminCommand::ReplaySessionLastTurn {
        session_id: "diag_sess_001".to_owned(),
    };
    let response = runtime.execute_admin(replay_cmd).expect("replay succeeds");

    match response {
        AdminResponse::DiagnosticReplay(res) => {
            assert_eq!(res.session_id, "diag_sess_001");
            assert_eq!(res.model_id, "qwen2.5-7b");
            assert_eq!(res.prompt, "Write a deterministic hello function in Rust");
            assert_eq!(res.expected_response, "fn hello() { println!(\"hello\"); }");
            assert_eq!(res.replayed_response, "fn hello() { println!(\"hello\"); }");
            assert!(res.matched);
        }
        other => panic!("expected DiagnosticReplay, got {other:?}"),
    }

    // Replaying nonexistent session returns clear error
    let bad_cmd = AdminCommand::ReplaySessionLastTurn {
        session_id: "nonexistent_session_999".to_owned(),
    };
    let bad_res = runtime.execute_admin(bad_cmd);
    assert!(bad_res.is_err());
    match bad_res.unwrap_err() {
        ManagerError::Persistence(PersistenceError::SessionNotFound(id)) => {
            assert_eq!(id, "nonexistent_session_999");
        }
        other => panic!("expected Persistence(SessionNotFound), got {other:?}"),
    }
}

#[test]
fn test_relational_foreign_key_integrity_rejection() {
    let store = SqlitePersistenceStore::open_in_memory().unwrap();

    // Attempting to insert a project with a non-existent tenant_id fails with ForeignKeyViolation
    let orphaned_project = Project {
        id: "orphaned_proj".to_owned(),
        tenant_id: "nonexistent_tenant".to_owned(),
        name: "Orphaned".to_owned(),
        token_budget: None,
        created_at: current_epoch(),
    };

    let err = store
        .save_project(&orphaned_project)
        .expect_err("foreign key violation must trigger error");

    match err {
        PersistenceError::ForeignKeyViolation(msg) => {
            assert!(
                msg.contains("FOREIGN KEY")
                    || msg.contains("foreign key")
                    || msg.contains("constraint")
            );
        }
        other => panic!("expected ForeignKeyViolation, got {other:?}"),
    }
}

#[test]
fn test_special_characters_sql_injection_escaping() {
    let store = SqlitePersistenceStore::open_in_memory().unwrap();

    let tenant = Tenant {
        id: "t_inj".to_owned(),
        name: "Injection Test Tenant".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).unwrap();

    let project = Project {
        id: "p_inj".to_owned(),
        tenant_id: "t_inj".to_owned(),
        name: "Injection Project".to_owned(),
        token_budget: None,
        created_at: current_epoch(),
    };
    store.save_project(&project).unwrap();

    let adversarial_prompt = "Hello'; DROP TABLE tenants; SELECT * FROM 'x' -- 🦀 🔥 \" ' \n \r \t";
    let adversarial_response = "Safe response: \0 \n \\ ' \" ; --";

    let turn = SessionLastTurn {
        session_id: "sess_sql_inject".to_owned(),
        project_id: "p_inj".to_owned(),
        tenant_id: "t_inj".to_owned(),
        model_id: "model_escape".to_owned(),
        prompt: adversarial_prompt.to_owned(),
        response: adversarial_response.to_owned(),
        prompt_tokens: 30_u32,
        completion_tokens: 20_u32,
        parameters_json: "{}".to_owned(),
        updated_at: current_epoch(),
    };

    store
        .save_session_last_turn(&turn)
        .expect("parameterized query escapes safely");

    // Verify tenants table was NOT dropped
    assert!(store.get_tenant("t_inj").unwrap().is_some());

    // Verify content retrieved verbatim
    let loaded = store
        .load_session_last_turn("sess_sql_inject")
        .unwrap()
        .unwrap();
    assert_eq!(loaded.prompt, adversarial_prompt);
    assert_eq!(loaded.response, adversarial_response);
}

#[test]
fn test_large_64kb_payload_persistence() {
    let store = SqlitePersistenceStore::open_in_memory().unwrap();

    let tenant = Tenant {
        id: "t_large".to_owned(),
        name: "Large Payload Tenant".to_owned(),
        created_at: current_epoch(),
    };
    store.save_tenant(&tenant).unwrap();

    let project = Project {
        id: "p_large".to_owned(),
        tenant_id: "t_large".to_owned(),
        name: "Large Project".to_owned(),
        token_budget: None,
        created_at: current_epoch(),
    };
    store.save_project(&project).unwrap();

    let large_prompt = "A".repeat(65_536); // 64 KB
    let large_response = "B".repeat(65_536); // 64 KB

    let turn = SessionLastTurn {
        session_id: "sess_64kb".to_owned(),
        project_id: "p_large".to_owned(),
        tenant_id: "t_large".to_owned(),
        model_id: "model_large".to_owned(),
        prompt: large_prompt.clone(),
        response: large_response.clone(),
        prompt_tokens: 16_384_u32,
        completion_tokens: 16_384_u32,
        parameters_json: "{}".to_owned(),
        updated_at: current_epoch(),
    };

    store.save_session_last_turn(&turn).unwrap();

    let loaded = store.load_session_last_turn("sess_64kb").unwrap().unwrap();
    assert_eq!(loaded.prompt.len(), 65_536);
    assert_eq!(loaded.response.len(), 65_536);
    assert_eq!(loaded.prompt, large_prompt);
    assert_eq!(loaded.response, large_response);
}

#[test]
fn test_cumulative_statistics_saturation_arithmetic() {
    let store = SqlitePersistenceStore::open_in_memory().unwrap();

    // Start near u32::MAX
    let near_max = u32::MAX - 50;
    let initial = store
        .update_cumulative_stat("overflow_stat", near_max)
        .unwrap();
    assert_eq!(initial, near_max);

    // Add 100 — exceeds u32::MAX, should saturate cleanly without panic or wrapping
    let saturated = store.update_cumulative_stat("overflow_stat", 100).unwrap();
    assert_eq!(saturated, u32::MAX);
}
