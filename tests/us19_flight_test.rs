//! US-19 Arrow Flight Tensor Streaming & Memfd Arena Verification Test Suite.
//!
//! Verifies:
//! - Packaging SpacemiT A100 IME-repacked weights (`8×16` `i8` blocks = 128 bytes) into Arrow RecordBatches
//! - Flight streamer metrics & batch validation (128-byte alignment)
//! - Anonymous `memfd` creation & writing at byte offsets
//! - `mlock()` physical memory locking within bounded memlock ceiling
//! - `F_SEAL_WRITE` sealing & write trap enforcement
//! - `FlightReceiver` direct memory population & `receive_to_memfd` interface contract
//! - Unix Domain Socket `SCM_RIGHTS` file descriptor transfer and receiver mapping

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::similar_names
)]

use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;

use arrow_array::{Array, BinaryArray};
use k3_manager::flight::{
    FlightError, FlightReceiver, FlightStream, FlightTensorBatchStreamer, IME_TILE_ELEMENTS,
    MemfdArena,
};

#[test]
fn test_ime_repacked_arrow_batch_construction_and_streaming() {
    let streamer = FlightTensorBatchStreamer::new();
    let schema = streamer.schema();
    assert_eq!(schema.fields().len(), 6);
    assert_eq!(schema.field(0).name(), "tensor_name");
    assert_eq!(schema.field(5).name(), "weights");

    // 1. Valid 2-tile weight payload (256 bytes = 2 * 128)
    let mut weights = vec![0i8; 256];
    for (i, item) in weights.iter_mut().enumerate() {
        *item = (i % 127) as i8;
    }

    let batch = streamer
        .build_repacked_batch("blk.0.attn_q.weight", 0, 0, &weights)
        .unwrap();

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 6);

    // Verify weights column
    let binary_col = batch
        .column(5)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("column 5 must be BinaryArray");
    assert_eq!(binary_col.len(), 2);
    assert_eq!(binary_col.value(0).len(), IME_TILE_ELEMENTS);
    assert_eq!(binary_col.value(1).len(), IME_TILE_ELEMENTS);

    // 2. Unaligned weight payload rejected
    let unaligned = vec![1i8; 127]; // 127 != 128
    let err = streamer.build_repacked_batch("blk.0.weight", 0, 0, &unaligned);
    assert!(err.is_err());
    match err.unwrap_err() {
        FlightError::InvalidBatch(msg) => {
            assert!(msg.contains("not a multiple of IME tile size"));
        }
        other => panic!("expected InvalidBatch, got {other:?}"),
    }

    // 3. Streaming metrics summary
    let summary = streamer
        .stream_tensors_to_endpoint("qwen2.5-7b", &[batch], "k3-rpcbridge.target-node:50051")
        .unwrap();

    assert_eq!(summary.model_id, "qwen2.5-7b");
    assert_eq!(summary.total_batches, 1);
    assert_eq!(summary.total_bytes, 256);
    assert_eq!(summary.tensor_count, 1);
    assert!(summary.elapsed_ms > 0);
}

#[test]
fn test_memfd_arena_creation_and_bounds() {
    let mut arena = MemfdArena::new("test_flight_arena", 64 * 1024).unwrap();
    assert!(arena.len() >= 64 * 1024);
    assert!(!arena.is_sealed());

    // Write data at offset 0
    let test_data = b"TENSOR_WEIGHT_BLOCK_DATA_1234";
    arena.write_at(0, test_data).unwrap();

    // Verify written data
    assert_eq!(&arena.as_slice()[0..test_data.len()], test_data);

    // Write data at page boundary
    arena.write_at(4096, test_data).unwrap();
    assert_eq!(&arena.as_slice()[4096..4096 + test_data.len()], test_data);

    // Bounds breach rejected
    let oob_res = arena.write_at(arena.len() - 5, test_data);
    assert!(oob_res.is_err());
}

#[test]
fn test_memfd_mlock_and_bounded_ceiling() {
    let mut arena = MemfdArena::new("test_mlock_arena", 16 * 1024).unwrap();

    // Lock memory within safe bounds
    let lock_res = arena.mlock(Some(64 * 1024 * 1024));
    assert!(lock_res.is_ok(), "mlock within safe limits must succeed");

    // Ceiling breach rejected
    let mut huge_arena = MemfdArena::new("test_huge_arena", 16 * 1024).unwrap();
    let low_ceiling = 1024; // 1KB ceiling
    let breach_res = huge_arena.mlock(Some(low_ceiling));
    assert!(breach_res.is_err());
    match breach_res.unwrap_err() {
        FlightError::InsufficientLockableMemory(_) => {}
        other => panic!("expected InsufficientLockableMemory, got {other:?}"),
    }
}

#[test]
fn test_memfd_f_seal_write_and_modification_trapping() {
    let mut arena = MemfdArena::new("test_seal_arena", 32 * 1024).unwrap();

    // Write before seal succeeds
    arena.write_at(0, b"BEFORE_SEAL_OK").unwrap();
    assert_eq!(&arena.as_slice()[0..14], b"BEFORE_SEAL_OK");

    // Apply write seal
    arena.seal_write().unwrap();
    assert!(arena.is_sealed());

    // Subsequent write attempt must trap / return MemfdSealed
    let write_res = arena.write_at(0, b"AFTER_SEAL_FAIL");
    assert!(write_res.is_err());
    match write_res.unwrap_err() {
        FlightError::MemfdSealed => {}
        other => panic!("expected MemfdSealed, got {other:?}"),
    }

    // as_mut_slice must also reject
    assert!(arena.as_mut_slice().is_err());
}

#[test]
fn test_flight_receiver_direct_population_and_finalize() {
    let streamer = FlightTensorBatchStreamer::new();

    // Prepare 2 batches of repacked weights
    let weights_layer0 = vec![1i8; 128]; // 1 tile
    let weights_layer1 = vec![2i8; 256]; // 2 tiles

    let batch0 = streamer
        .build_repacked_batch("blk.0.attn_q.weight", 0, 0, &weights_layer0)
        .unwrap();
    let batch1 = streamer
        .build_repacked_batch("blk.1.attn_q.weight", 1, 128, &weights_layer1)
        .unwrap();

    let flight_stream = FlightStream::new(vec![batch0, batch1]);
    assert_eq!(flight_stream.total_bytes, 384);

    // Receive directly to memfd
    let raw_fd = FlightReceiver::receive_to_memfd(flight_stream).unwrap();
    assert!(raw_fd >= 0);

    // Verify descriptor liveness via fcntl(F_GETFD)
    // SAFETY: Querying descriptor flags via fcntl does not mutate state or cause undefined behavior.
    let flags = unsafe { libc::fcntl(raw_fd, libc::F_GETFD) };
    assert!(flags >= 0, "returned RawFd must be active and valid in the kernel (not EBADF)");

    // Safely close descriptor at end of test
    // SAFETY: raw_fd was verified open and is exclusively owned by this test.
    unsafe {
        libc::close(raw_fd);
    }
}

#[test]
fn test_memfd_scm_rights_uds_transfer() {
    let (sock_sender, sock_receiver) = UnixStream::pair().unwrap();

    // 1. Create and populate memfd arena on sender side
    let mut arena = MemfdArena::new("sender_tensor_arena", 16 * 1024).unwrap();
    arena.write_at(0, b"TRANSMITTED_VIA_SCM_RIGHTS_PAYLOAD").unwrap();
    arena.seal_write().unwrap();
    assert!(arena.is_sealed());

    // 2. Transfer sealed FD over UDS via SCM_RIGHTS
    let model_id = "qwen2.5-7b-transferred";
    arena.transfer_via_uds(&sock_sender, model_id.as_bytes()).unwrap();

    // 3. Receive sealed FD on receiver side
    let (payload, received_fd) = MemfdArena::receive_from_uds(&sock_receiver).unwrap();
    assert_eq!(payload, model_id.as_bytes());
    assert!(received_fd.as_raw_fd() >= 0);

    // 4. Verify received FD points to identical memory contents
    // Map received FD using mmap
    let page_size = 16 * 1024;
    // SAFETY: mmap maps the received file descriptor into process memory with MAP_SHARED.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            page_size,
            libc::PROT_READ,
            libc::MAP_SHARED,
            received_fd.as_raw_fd(),
            0,
        )
    };
    assert_ne!(ptr, libc::MAP_FAILED);

    // SAFETY: ptr is valid and mapped with page_size bytes for reading.
    let received_slice = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), 34) };
    assert_eq!(received_slice, b"TRANSMITTED_VIA_SCM_RIGHTS_PAYLOAD");

    // Clean up mapping
    // SAFETY: munmap cleans up the test mapping address range.
    unsafe {
        libc::munmap(ptr, page_size);
    }
}
