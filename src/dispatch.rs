//! Zero-copy forward pass dispatch and Apache Arrow tensor exposure.
//!
//! Enforces zero bulk tensor payload bytes across Unix Domain Sockets:
//! input tokens are placed in-place into shared-memory slots (`wrap_tokens_as_uint32_array`),
//! compact `ForwardPassCommand` descriptors are sent over UDS, and output activations/logits
//! are exposed directly from shared memory as Apache Arrow `Float32Array` columnar views.

use std::os::unix::net::UnixStream;
use std::sync::Arc;

use arrow_array::{Float32Array, UInt32Array};
use k3_core::ipc::arrow::{wrap_floats_as_float32_array, wrap_tokens_as_uint32_array};
use k3_core::ipc::layout::{OutputSlabHeader, TokenInputSlotHeader, SLAB_STATUS_IDLE};
use k3_core::ipc::protocol::{
    recv_msg_frame, send_msg_frame, ForwardPassCommand, ForwardPassCompleted, IpcMessage,
};
use k3_core::ipc::shm::SharedMemoryArena;
use zerocopy::FromBytes;

use crate::error::ManagerError;

/// Input parameters for dispatching a forward pass to a compute worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardPassRequest {
    /// Unique request identifier.
    pub request_id: u64,
    /// Sequence identifier.
    pub sequence_id: u64,
    /// Input prompt or decode token IDs.
    pub tokens: Vec<u32>,
    /// Starting sequence position for this forward pass.
    pub start_position: u32,
    /// 16-token KV cache block indices assigned to this sequence.
    pub block_indices: Vec<u32>,
    /// `true` for prompt prefill, `false` for single-token autoregressive decode.
    pub is_prefill: bool,
}

impl ForwardPassRequest {
    /// Constructs a new prefill request.
    #[must_use]
    pub fn new_prefill(
        request_id: u64,
        sequence_id: u64,
        tokens: Vec<u32>,
        block_indices: Vec<u32>,
    ) -> Self {
        Self {
            request_id,
            sequence_id,
            tokens,
            start_position: 0,
            block_indices,
            is_prefill: true,
        }
    }

    /// Constructs a new decode step request.
    #[must_use]
    pub fn new_decode(
        request_id: u64,
        sequence_id: u64,
        next_token: u32,
        position: u32,
        block_indices: Vec<u32>,
    ) -> Self {
        Self {
            request_id,
            sequence_id,
            tokens: vec![next_token],
            start_position: position,
            block_indices,
            is_prefill: false,
        }
    }
}

/// Result returned from executing a forward pass.
#[derive(Debug, Clone)]
pub struct ForwardPassResult {
    /// Execution status, checkpointed committed tokens, and measured latencies.
    pub completed: ForwardPassCompleted,
    /// Zero-copy Apache Arrow view over output logits in the shared-memory slab.
    pub logits: Float32Array,
}

impl ForwardPassResult {
    /// Returns the sampled or top-logit next token ID.
    #[must_use]
    pub const fn next_token(&self) -> u32 {
        self.completed.next_token
    }

    /// Returns the Apache Arrow logits array.
    #[must_use]
    pub fn logits(&self) -> &Float32Array {
        &self.logits
    }

    /// Returns measured prefill latency in milliseconds.
    #[must_use]
    pub const fn prefill_latency_ms(&self) -> f64 {
        self.completed.prefill_latency_ms
    }

    /// Returns measured decode latency in milliseconds.
    #[must_use]
    pub const fn decode_latency_ms(&self) -> f64 {
        self.completed.decode_latency_ms
    }
}

/// Dispatches a forward pass to a connected compute worker with zero bulk tensor bytes over UDS.
///
/// # Errors
/// Returns [`ManagerError`] on shared memory, IPC communication, or worker execution failure.
#[allow(clippy::cast_ptr_alignment)]
pub fn dispatch_forward_pass(
    stream: &UnixStream,
    arena: &mut SharedMemoryArena,
    req: &ForwardPassRequest,
) -> Result<ForwardPassResult, ManagerError> {
    if req.tokens.is_empty() {
        return Err(ManagerError::Protocol(
            "cannot dispatch forward pass with 0 tokens".to_owned(),
        ));
    }

    // 1. Read header geometry
    let hdr = *arena.header()?;
    let slot_offset = usize::try_from(hdr.token_slots_offset).map_err(|e| {
        ManagerError::Protocol(format!("invalid token_slots_offset: {e}"))
    })?;

    // 2. Write TokenInputSlotHeader (16 bytes) into shared memory
    let slot_hdr_slice = arena.slice_mut(slot_offset, 16)?;
    let slot_hdr = TokenInputSlotHeader::mut_from_bytes(slot_hdr_slice).map_err(|e| {
        ManagerError::Protocol(format!("failed to cast TokenInputSlotHeader: {e}"))
    })?;
    slot_hdr.slot_seq_id = req.sequence_id;
    slot_hdr.token_len = u32::try_from(req.tokens.len()).map_err(|_| {
        ManagerError::Protocol("token count exceeds u32".to_owned())
    })?;
    slot_hdr.start_position = req.start_position;

    // 3. Write tokens array in place into shared memory slot
    let tokens_data_offset = slot_offset + 16;
    let tokens_slice = arena.slice_mut(tokens_data_offset, req.tokens.len() * 4)?;
    for (i, &tok) in req.tokens.iter().enumerate() {
        tokens_slice[i * 4..(i + 1) * 4].copy_from_slice(&tok.to_ne_bytes());
    }

    // 4. Mark output slab as IDLE prior to pass execution
    let slab_offset = usize::try_from(hdr.output_slabs_offset).map_err(|e| {
        ManagerError::Protocol(format!("invalid output_slabs_offset: {e}"))
    })?;
    let slab_hdr_slice = arena.slice_mut(slab_offset, 64)?;
    let slab_hdr = OutputSlabHeader::mut_from_bytes(slab_hdr_slice).map_err(|e| {
        ManagerError::Protocol(format!("failed to cast OutputSlabHeader: {e}"))
    })?;
    slab_hdr.status = SLAB_STATUS_IDLE;

    // 5. Send compact ForwardPassCommand descriptor over UDS (zero tensor bytes over socket)
    let cmd = IpcMessage::ForwardPassCommand(ForwardPassCommand {
        request_id: req.request_id,
        sequence_id: req.sequence_id,
        token_offset: req.start_position,
        token_count: u32::try_from(req.tokens.len()).unwrap_or(0),
        block_indices: req.block_indices.clone(),
        is_prefill: req.is_prefill,
    });

    send_msg_frame(stream, &cmd, None).map_err(|e| {
        ManagerError::WorkerIpc(format!("failed to send ForwardPassCommand to worker: {e}"))
    })?;

    // 6. Receive ForwardPassCompleted from worker
    let (reply, _) = recv_msg_frame(stream).map_err(|e| {
        ManagerError::WorkerIpc(format!("failed to receive ForwardPassCompleted from worker: {e}"))
    })?;

    let completed = match reply {
        IpcMessage::ForwardPassCompleted(c) => {
            if !c.status.is_success() {
                return Err(ManagerError::Status {
                    status: c.status,
                    message: c
                        .error_message
                        .unwrap_or_else(|| format!("forward pass failed with status {:?}", c.status)),
                });
            }
            if c.request_id != req.request_id {
                return Err(ManagerError::Protocol(format!(
                    "request ID mismatch: sent {}, received {}",
                    req.request_id, c.request_id
                )));
            }
            c
        }
        other => {
            return Err(ManagerError::Protocol(format!(
                "expected ForwardPassCompleted from worker, got {other:?}"
            )));
        }
    };

    // 7. Verify slab was written and wrap output logits into zero-copy Apache Arrow Float32Array
    let slab_slice = arena.slice(slab_offset, 64)?;
    let slab_hdr = OutputSlabHeader::ref_from_bytes(slab_slice).map_err(|e| {
        ManagerError::Protocol(format!("failed to cast OutputSlabHeader: {e}"))
    })?;
    let vocab_size = if slab_hdr.vocab_size > 0 {
        slab_hdr.vocab_size as usize
    } else if hdr.vocab_size > 0 {
        hdr.vocab_size as usize
    } else {
        10
    };

    let logits_offset = slab_offset + 64;
    let logits_slice = arena.slice(logits_offset, vocab_size * 4)?;

    let arena_fd_owner = Arc::new(arena.duplicate_fd()?);
    // SAFETY: logits_slice points to vocab_size valid f32 values in memory backed by arena_fd_owner.
    let logits = unsafe {
        wrap_floats_as_float32_array(
            arena_fd_owner,
            "output_logits",
            logits_slice.as_ptr().cast::<f32>(),
            vocab_size,
        )?
    };

    Ok(ForwardPassResult { completed, logits })
}

/// Exposes the input tokens in the shared-memory slot as a zero-copy Apache Arrow [`UInt32Array`].
///
/// # Errors
/// Returns [`ManagerError`] on bounds or layout failure.
#[allow(clippy::cast_ptr_alignment)]
pub fn wrap_input_tokens_as_arrow(
    arena: &SharedMemoryArena,
) -> Result<UInt32Array, ManagerError> {
    let hdr = *arena.header()?;
    let slot_offset = usize::try_from(hdr.token_slots_offset).map_err(|e| {
        ManagerError::Protocol(format!("invalid token_slots_offset: {e}"))
    })?;

    let slot_hdr_slice = arena.slice(slot_offset, 16)?;
    let slot_hdr = TokenInputSlotHeader::ref_from_bytes(slot_hdr_slice).map_err(|e| {
        ManagerError::Protocol(format!("failed to cast TokenInputSlotHeader: {e}"))
    })?;
    let token_count = slot_hdr.token_len as usize;

    let tokens_data_offset = slot_offset + 16;
    let tokens_slice = arena.slice(tokens_data_offset, token_count * 4)?;

    let arena_fd_owner = Arc::new(arena.duplicate_fd()?);
    // SAFETY: tokens_slice points to token_count valid u32 values backed by arena_fd_owner.
    unsafe {
        wrap_tokens_as_uint32_array(
            arena_fd_owner,
            tokens_slice.as_ptr().cast::<u32>(),
            token_count,
        )
        .map_err(ManagerError::from)
    }
}
