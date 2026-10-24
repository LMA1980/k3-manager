//! `KvClient` unit and integration tests for memory arena management.

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]

use std::time::Duration;

use k3_core::ipc::layout::{ARENA_MAGIC, ARENA_VERSION};
use k3_core::ipc::protocol::StatusCode;
use k3_kv::server::{KvServer, ServerConfig};
use k3_manager::client::KvClient;
use k3_manager::error::ManagerError;

#[test]
fn test_kv_client_lifecycle_and_arena_allocation() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let pid = std::process::id();
    let socket_path = std::env::temp_dir().join(format!("kv_client_test_{pid}.sock"));

    let server_config = ServerConfig {
        socket_path: socket_path.clone(),
        max_memory_bytes: 1024 * 1024 * 1024,
    };
    let server = KvServer::new(server_config);

    let task = rt.spawn(async move {
        let _ = server.run().await;
    });

    // Wait for server to bind
    let mut client = None;
    for _ in 0..50 {
        if let Ok(c) = KvClient::connect(&socket_path) {
            client = Some(c);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut client = client.expect("connect KvClient");

    // 1. Heartbeat
    let echo = client.heartbeat().expect("heartbeat");
    assert!(echo > 0);

    // 2. Allocate arena
    let alloc = client
        .allocate_arena("test_model", 65536, 64, 4)
        .expect("allocate arena");
    assert_eq!(alloc.status, StatusCode::Success);
    assert!(alloc.arena_bytes >= 65536);
    let arena_id = alloc.arena_id;

    // 3. Attach arena (fetch FD via SCM_RIGHTS and map)
    let arena = client
        .attach_arena(arena_id, alloc.arena_bytes as usize, "test_model_shm")
        .expect("attach arena");
    let hdr = *arena.header().expect("read header");
    assert_eq!(hdr.magic, ARENA_MAGIC);
    assert_eq!(hdr.version, ARENA_VERSION);
    assert_eq!(hdr.n_ctx, 64);

    // 4. Deallocate arena
    let dealloc = client.deallocate_arena(arena_id).expect("deallocate arena");
    assert_eq!(dealloc.status, StatusCode::Success);
    assert_eq!(dealloc.arena_id, arena_id);

    // 5. Querying deallocated arena returns error
    let query_err = client.get_arena_fd(arena_id);
    assert!(query_err.is_err());
    match query_err.unwrap_err() {
        ManagerError::Status { status, .. } => {
            assert_eq!(status, StatusCode::ArenaNotFound);
        }
        other => panic!("expected StatusCode::ArenaNotFound, got {other:?}"),
    }

    // Teardown
    task.abort();
    let _ = std::fs::remove_file(&socket_path);
}
