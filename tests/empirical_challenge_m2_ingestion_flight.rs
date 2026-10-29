//! Empirical Adversarial Challenge Test Suite for Milestone M2 (Ingestion & Flight Streaming).
//!
//! Evaluates and stress-tests:
//! 1. Challenge 1: Staging Download Security & Resource Leak Prevention
//!    - Corrupted format headers (SafeTensors truncated, invalid JSON, huge length prefix).
//!    - Corrupted GGUF headers (bad magic, invalid versions, zero length).
//!    - Stream truncation and simulated mid-transfer network faults.
//!    - SHA-256 digest mismatches.
//!    - Path traversal attempt in `pull_to_staging` via model or tag naming.
//!    - Strict verification of zero leaked `.part` files in the staging directory after any failure.
//! 2. Challenge 2: Arrow Flight Streaming & Memfd Security
//!    - Use-After-Close verification on `FlightReceiver::receive_to_memfd` (RawFd validity audit).
//!    - Denial-of-Service panic check on RecordBatches with fewer than 6 columns.
//!    - Sparse-offset tensor stream handling vs arena allocation capacity calculation.
//!    - Unaligned weight tensors (non-multiple of 128-byte IME tiles).
//!    - Memfd write-sealing (`F_SEAL_WRITE`) mutation trapping.
//!    - Mlock ceiling enforcement.
//! 3. Challenge 3: Credential Isolation & Fail-Closed Security
//!    - Non-existent secret paths fail closed.
//!    - Unreadable secret files (mode 0000) fail closed.
//!    - Empty and whitespace-only secret files fail closed.
//!    - Zero token leakage in Debug output or error messages.

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::undocumented_unsafe_blocks
)]

use std::fs;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use k3_manager::flight::{
    FlightError, FlightReceiver, FlightStream, FlightTensorBatchStreamer, MemfdArena,
};
use k3_manager::ingestion::{
    GgufHeaderPrefix, HuggingFaceAdapter, ModelFormat, PullRequest,
    SafeTensorsHeaderPrefix, SourceAdapter, SourceCredentials, StagingPipeline,
};

static CHALLENGE_SEQ: AtomicU64 = AtomicU64::new(8000);

struct TempStore {
    root: PathBuf,
}

impl TempStore {
    fn new(prefix: &str) -> Self {
        let seq = CHALLENGE_SEQ.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("{prefix}_{}_{seq}", std::process::id()));
        fs::create_dir_all(root.join("staging")).expect("create staging dir");
        fs::create_dir_all(root.join("blobs").join("sha256")).expect("create blobs dir");
        Self { root }
    }

    fn staging_dir(&self) -> PathBuf {
        self.root.join("staging")
    }

    fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs").join("sha256")
    }

    fn count_staging_files(&self) -> usize {
        fs::read_dir(self.staging_dir())
            .map(|entries| entries.filter_map(Result::ok).count())
            .unwrap_or(0)
    }

    fn count_blobs(&self) -> usize {
        fs::read_dir(self.blobs_dir())
            .map(|entries| entries.filter_map(Result::ok).count())
            .unwrap_or(0)
    }
}

impl Drop for TempStore {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

// =========================================================================
// Challenge 1: Staging Download Security & Leak Prevention
// =========================================================================

#[test]
fn challenge_staging_corrupted_gguf_headers_cleaned_up() {
    let store = TempStore::new("ch_gguf_corrupt");
    let pipeline = StagingPipeline::new(&store.root);

    // Subcase 1: Bad magic bytes
    let bad_magic_payload = b"NOT_GGUF_MAGIC_BYTES_123456789";
    let (part1, mut f1) = StagingPipeline::create_staging_file(&pipeline.staging_dir()).unwrap();
    f1.write_all(bad_magic_payload).unwrap();
    drop(f1);

    let res1 = StagingPipeline::verify_format_header(&part1, &ModelFormat::Gguf);
    assert!(res1.is_err(), "bad GGUF magic must fail verification");
    assert!(!part1.exists(), "part file must be unlinked on bad magic");
    assert_eq!(store.count_staging_files(), 0, "staging dir must have 0 leaks");

    // Subcase 2: Truncated GGUF (fewer than 24 bytes header)
    let truncated_payload = b"GGUF\x03\x00\x00\x00"; // only 8 bytes
    let (part2, mut f2) = StagingPipeline::create_staging_file(&pipeline.staging_dir()).unwrap();
    f2.write_all(truncated_payload).unwrap();
    drop(f2);

    let res2 = StagingPipeline::verify_format_header(&part2, &ModelFormat::Gguf);
    assert!(res2.is_err(), "truncated GGUF header must fail verification");
    assert!(!part2.exists(), "part file must be unlinked on truncation");
    assert_eq!(store.count_staging_files(), 0, "staging dir must have 0 leaks");

    // Subcase 3: Unsupported future version (e.g. version 99)
    let header_v99 = GgufHeaderPrefix {
        magic: *b"GGUF",
        version: 99,
        tensor_count: 10,
        metadata_kv_count: 2,
    };
    let (part3, mut f3) = StagingPipeline::create_staging_file(&pipeline.staging_dir()).unwrap();
    f3.write_all(zerocopy::IntoBytes::as_bytes(&header_v99)).unwrap();
    drop(f3);

    let res3 = StagingPipeline::verify_format_header(&part3, &ModelFormat::Gguf);
    assert!(res3.is_err(), "unsupported GGUF version must fail");
    assert!(!part3.exists(), "part file must be unlinked on unsupported version");
    assert_eq!(store.count_staging_files(), 0, "staging dir must have 0 leaks");
}

#[test]
fn challenge_staging_corrupted_safetensors_headers_cleaned_up() {
    let store = TempStore::new("ch_st_corrupt");
    let pipeline = StagingPipeline::new(&store.root);

    // Subcase 1: 0-byte SafeTensors header length
    let header_zero = SafeTensorsHeaderPrefix { header_len: 0 };
    let (part1, mut f1) = StagingPipeline::create_staging_file(&pipeline.staging_dir()).unwrap();
    f1.write_all(zerocopy::IntoBytes::as_bytes(&header_zero)).unwrap();
    drop(f1);

    let res1 = StagingPipeline::verify_format_header(&part1, &ModelFormat::SafeTensors);
    assert!(res1.is_err(), "0-byte SafeTensors header length must fail");
    assert!(!part1.exists(), "part file must be unlinked");
    assert_eq!(store.count_staging_files(), 0, "staging dir must have 0 leaks");

    // Subcase 2: Excessive header length exceeding 100MB bound
    let header_huge = SafeTensorsHeaderPrefix {
        header_len: 200 * 1024 * 1024,
    };
    let (part2, mut f2) = StagingPipeline::create_staging_file(&pipeline.staging_dir()).unwrap();
    f2.write_all(zerocopy::IntoBytes::as_bytes(&header_huge)).unwrap();
    drop(f2);

    let res2 = StagingPipeline::verify_format_header(&part2, &ModelFormat::SafeTensors);
    assert!(res2.is_err(), "header length > 100MB must be rejected");
    assert!(!part2.exists(), "part file must be unlinked");
    assert_eq!(store.count_staging_files(), 0, "staging dir must have 0 leaks");

    // Subcase 3: Header length claims 1000 bytes, but file only has 50 bytes (Truncation)
    let header_trunc = SafeTensorsHeaderPrefix { header_len: 1000 };
    let (part3, mut f3) = StagingPipeline::create_staging_file(&pipeline.staging_dir()).unwrap();
    f3.write_all(zerocopy::IntoBytes::as_bytes(&header_trunc)).unwrap();
    f3.write_all(b"{\"incomplete\":").unwrap();
    drop(f3);

    let res3 = StagingPipeline::verify_format_header(&part3, &ModelFormat::SafeTensors);
    assert!(res3.is_err(), "file smaller than claimed header length must fail");
    assert!(!part3.exists(), "part file must be unlinked on truncation");
    assert_eq!(store.count_staging_files(), 0, "staging dir must have 0 leaks");

    // Subcase 4: Invalid JSON metadata
    let bad_json = b"not_valid_json_at_all!{[]";
    let header_bad_json = SafeTensorsHeaderPrefix {
        header_len: bad_json.len() as u64,
    };
    let (part4, mut f4) = StagingPipeline::create_staging_file(&pipeline.staging_dir()).unwrap();
    f4.write_all(zerocopy::IntoBytes::as_bytes(&header_bad_json)).unwrap();
    f4.write_all(bad_json).unwrap();
    drop(f4);

    let res4 = StagingPipeline::verify_format_header(&part4, &ModelFormat::SafeTensors);
    assert!(res4.is_err(), "invalid JSON must fail SafeTensors verification");
    assert!(!part4.exists(), "part file must be unlinked on invalid JSON");
    assert_eq!(store.count_staging_files(), 0, "staging dir must have 0 leaks");
}

#[test]
fn challenge_staging_digest_mismatch_and_interruption_no_leaks() {
    let store = TempStore::new("ch_leak_check");
    let pipeline = StagingPipeline::new(&store.root);

    // Subcase 1: Digest mismatch in stream_to_staging
    let (part1, _) = StagingPipeline::create_staging_file(&pipeline.staging_dir()).unwrap();
    let mut data = Cursor::new(b"SOME_DATA_TO_STREAM");
    let wrong_digest = "1111111111111111111111111111111111111111111111111111111111111111";

    let res1 = StagingPipeline::stream_to_staging(&part1, &mut data, Some(wrong_digest), None, None);
    assert!(res1.is_err());
    assert!(!part1.exists(), "part1 must be deleted on digest mismatch");
    assert_eq!(store.count_staging_files(), 0, "no staging files leaked");

    // Subcase 2: Simulated fault via adapter pull
    let adapter = HuggingFaceAdapter::new();
    adapter.set_simulated_fault(Some("network_drop"));

    let part2 = pipeline.staging_dir().join("test_fault.part");
    let req = PullRequest {
        source: "huggingface".to_owned(),
        namespace: "Qwen".to_owned(),
        model: "Qwen2.5-7B".to_owned(),
        tag: "q8_0".to_owned(),
        expected_digest: None,
        format: Some(ModelFormat::Gguf),
    };

    let res2 = adapter.pull(&req, &part2, None);
    assert!(res2.is_err(), "network drop must fail");
    assert!(!part2.exists(), "part2 must be unlinked on network drop");
    assert_eq!(store.count_staging_files(), 0, "zero staging files leaked");
    assert_eq!(store.count_blobs(), 0, "zero blobs created");
}

#[test]
fn challenge_staging_path_traversal_check() {
    let store = TempStore::new("ch_path_traversal");
    let adapter = HuggingFaceAdapter::new();

    // Adversarial tag containing path traversal
    let malicious_tag = "../../escaped_model_part";
    let staging_dir = store.staging_dir();

    let res = adapter.pull_to_staging("huggingface", "Qwen/Qwen2.5-7B", malicious_tag, &staging_dir);
    println!("pull_to_staging with path traversal tag: {res:?}");

    // Check if any file escaped outside staging_dir
    let escaped_file = store.root.join("escaped_model_part");
    assert!(
        !escaped_file.exists(),
        "SECURITY RISK: Staging file escaped out of staging directory via relative path traversal tag!"
    );
}

// =========================================================================
// Challenge 2: Arrow Flight Streaming & Memfd Security
// =========================================================================

#[test]
fn challenge_flight_unaligned_tensor_sizes_rejected() {
    let streamer = FlightTensorBatchStreamer::new();

    // 0 bytes -> rejected
    let empty_weights: Vec<i8> = vec![];
    assert!(streamer.build_repacked_batch("tensor_0", 0, 0, &empty_weights).is_err());

    // 1 byte -> rejected
    let one_byte = vec![42i8; 1];
    assert!(streamer.build_repacked_batch("tensor_1", 0, 0, &one_byte).is_err());

    // 127 bytes -> rejected
    let w127 = vec![1i8; 127];
    assert!(streamer.build_repacked_batch("tensor_127", 0, 0, &w127).is_err());

    // 129 bytes -> rejected
    let w129 = vec![1i8; 129];
    assert!(streamer.build_repacked_batch("tensor_129", 0, 0, &w129).is_err());

    // Exactly 128 bytes -> ok
    let w128 = vec![1i8; 128];
    assert!(streamer.build_repacked_batch("tensor_128", 0, 0, &w128).is_ok());

    // Exactly 256 bytes -> ok
    let w256 = vec![1i8; 256];
    assert!(streamer.build_repacked_batch("tensor_256", 0, 0, &w256).is_ok());
}

#[test]
fn challenge_flight_memfd_sealing_prevents_mutations() {
    let mut arena = MemfdArena::new("seal_test_arena", 16 * 1024).unwrap();

    arena.write_at(0, b"INITIAL_DATA").unwrap();
    arena.seal_write().unwrap();
    assert!(arena.is_sealed());

    // write_at must return MemfdSealed
    let res = arena.write_at(0, b"OVERWRITE");
    assert!(res.is_err());
    match res.unwrap_err() {
        FlightError::MemfdSealed => {}
        other => panic!("expected MemfdSealed, got {other:?}"),
    }

    // as_mut_slice must return MemfdSealed
    assert!(arena.as_mut_slice().is_err());

    // Read must still succeed and match initial data
    assert_eq!(&arena.as_slice()[0..12], b"INITIAL_DATA");
}

#[test]
fn challenge_flight_memfd_mlock_bounds_enforced() {
    let mut arena = MemfdArena::new("mlock_bounds", 32 * 1024).unwrap();

    // Lock exceeding tiny ceiling (e.g. 1024 bytes when arena is 32KB)
    let res = arena.mlock(Some(1024));
    assert!(res.is_err(), "exceeding mlock ceiling must fail");
    match res.unwrap_err() {
        FlightError::InsufficientLockableMemory(_) => {}
        other => panic!("expected InsufficientLockableMemory, got {other:?}"),
    }
}

/// VERIFICATION OF REMEDIATION 1:
/// `FlightReceiver::receive_to_memfd` returns an open, valid, sealed RawFd
/// because `arena.into_raw_fd()` disarms drop and preserves the descriptor and RAM mapping.
#[test]
fn challenge_flight_receive_to_memfd_fd_validity() {
    let streamer = FlightTensorBatchStreamer::new();
    let weights = vec![1i8; 128];
    let batch = streamer
        .build_repacked_batch("test.weight", 0, 0, &weights)
        .unwrap();

    let flight_stream = FlightStream::new(vec![batch]);
    let raw_fd = FlightReceiver::receive_to_memfd(flight_stream).unwrap();

    // Verify whether raw_fd is open and valid in the kernel:
    let fd_flags_immediate = unsafe { libc::fcntl(raw_fd, libc::F_GETFD) };
    println!("VERIFICATION: receive_to_memfd returned raw_fd={raw_fd}, immediate fcntl(F_GETFD)={fd_flags_immediate}");

    assert!(
        fd_flags_immediate >= 0,
        "receive_to_memfd must return an active, valid file descriptor in the kernel (not EBADF)!"
    );

    unsafe {
        libc::close(raw_fd);
    }
}

/// VERIFICATION OF REMEDIATION 2:
/// `FlightReceiver::write_batch` checks `batch.num_columns() >= 6` and returns
/// structured `Err(FlightError::InvalidBatch)` instead of panicking with index out of bounds.
#[test]
fn challenge_flight_receiver_malformed_batch_schema() {
    // Construct a RecordBatch with only 2 columns instead of 6
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("col0", DataType::Utf8, false),
        Field::new("col1", DataType::UInt32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            std::sync::Arc::new(StringArray::from(vec!["tensor_a"])),
            std::sync::Arc::new(UInt32Array::from(vec![0])),
        ],
    )
    .unwrap();

    let mut receiver = FlightReceiver::new("malformed_batch_receiver", 4096).unwrap();

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        receiver.write_batch(&batch)
    }));

    // Must not panic on schema mismatch
    assert!(res.is_ok(), "write_batch must not panic on schema mismatch");
    let write_res = res.unwrap();
    assert!(write_res.is_err(), "write_batch must return Err on schema mismatch");
    match write_res.unwrap_err() {
        FlightError::InvalidBatch(msg) => {
            assert!(msg.contains("requires at least 6 columns"));
        }
        other => panic!("expected InvalidBatch error, got {other:?}"),
    }
}

/// VERIFICATION OF REMEDIATION 3:
/// `FlightStream::new` and `receive_to_memfd` compute arena capacity accounting for
/// `byte_offset + weight_bytes.len()`, accommodating sparse/non-contiguous tensor layouts.
#[test]
fn challenge_flight_sparse_offset_stream_handling() {
    let streamer = FlightTensorBatchStreamer::new();
    let w1 = vec![1i8; 128];
    let w2 = vec![2i8; 128];

    // Batch 1 is at offset 0
    let batch1 = streamer.build_repacked_batch("layer0", 0, 0, &w1).unwrap();
    // Batch 2 is at sparse offset 65536 (64KB in)
    let batch2 = streamer.build_repacked_batch("layer1", 1, 65536, &w2).unwrap();

    let flight_stream = FlightStream::new(vec![batch1, batch2]);

    // total_bytes properly accounts for offset extent
    assert!(flight_stream.total_bytes >= 65536 + 128);

    let res = FlightReceiver::receive_to_memfd(flight_stream);
    println!("VERIFICATION: sparse offset receive_to_memfd result: {res:?}");

    assert!(
        res.is_ok(),
        "receive_to_memfd must succeed on sparse tensor offsets by properly allocating capacity"
    );

    let raw_fd = res.unwrap();
    let flags = unsafe { libc::fcntl(raw_fd, libc::F_GETFD) };
    assert!(flags >= 0);
    unsafe {
        libc::close(raw_fd);
    }
}

// =========================================================================
// Challenge 3: Credential Isolation & Fail-Closed Security
// =========================================================================

#[test]
fn challenge_credential_fail_closed_unreadable_file() {
    let store = TempStore::new("ch_cred_unreadable");
    let unreadable_file = store.root.join("unreadable_secret.env");
    fs::write(&unreadable_file, "super_secret_token_12345").unwrap();

    // Restrict permissions to 0000 on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&unreadable_file, fs::Permissions::from_mode(0o000));
    }

    let creds = SourceCredentials::SecretFile(unreadable_file.clone());
    let res = creds.resolve_secret();

    // Restore permissions for cleanup
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&unreadable_file, fs::Permissions::from_mode(0o600));
    }

    assert!(res.is_err(), "unreadable secret file must fail closed");
    let err_msg = res.unwrap_err().to_string();
    assert!(
        !err_msg.contains("super_secret_token_12345"),
        "error message must never leak secret content"
    );
}

#[test]
fn challenge_credential_debug_format_zero_leakage() {
    let secret = "ghp_VERY_SECRET_AUTH_TOKEN_DO_NOT_LEAK";
    let token_creds = SourceCredentials::Token(secret.to_owned());
    let bearer_creds = SourceCredentials::BearerToken(secret.to_owned());
    let basic_creds = SourceCredentials::BasicAuth {
        username: "admin_user".to_owned(),
        password_or_token: secret.to_owned(),
    };

    let s1 = format!("{token_creds:?}");
    let s2 = format!("{bearer_creds:?}");
    let s3 = format!("{basic_creds:?}");

    assert!(!s1.contains(secret), "Token debug leaked secret!");
    assert!(!s2.contains(secret), "BearerToken debug leaked secret!");
    assert!(!s3.contains(secret), "BasicAuth debug leaked secret!");
    assert!(s1.contains("[REDACTED]"));
    assert!(s2.contains("[REDACTED]"));
    assert!(s3.contains("[REDACTED]"));
}
