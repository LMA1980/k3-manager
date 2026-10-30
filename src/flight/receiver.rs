//! Arrow Flight Receiver & Anonymous Memfd Arena Population (US-19, ADR-037).
//!
//! Receives streamed Apache Arrow Flight tensor batches, writes them directly into
//! an anonymous shared memory arena (`memfd_create`), locks pages in physical RAM (`mlock()`),
//! applies `F_SEAL_WRITE` sealing, and hands off the sealed descriptor to worker processes
//! over Unix Domain Sockets using `SCM_RIGHTS`.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::needless_pass_by_value
)]

use std::os::fd::RawFd;
use std::os::unix::net::UnixStream;

use arrow_array::{Array, BinaryArray, RecordBatch, UInt64Array};
use k3_core::K3Error;

use super::memfd::{FlightError, MemfdArena};

/// Stream container for incoming Flight tensor batches.
#[derive(Debug, Clone)]
pub struct FlightStream {
    /// Ordered list of Arrow tensor batches.
    pub batches: Vec<RecordBatch>,
    /// Total expected memory footprint in bytes.
    pub total_bytes: usize,
}

impl FlightStream {
    /// Creates a new flight stream wrapping a collection of batches.
    #[must_use]
    pub fn new(batches: Vec<RecordBatch>) -> Self {
        let mut max_extent = 0;
        let mut sum_lengths = 0;
        for batch in &batches {
            if batch.num_columns() >= 6 {
                let offsets = batch.column(3).as_any().downcast_ref::<UInt64Array>();
                let weights = batch.column(5).as_any().downcast_ref::<BinaryArray>();

                if let (Some(offsets), Some(weights)) = (offsets, weights) {
                    let rows = offsets.len().min(weights.len());
                    for i in 0..rows {
                        let offset = offsets.value(i) as usize;
                        let len = weights.value(i).len();
                        sum_lengths += len;
                        let extent = offset.saturating_add(len);
                        if extent > max_extent {
                            max_extent = extent;
                        }
                    }
                }
            }
        }
        let total_bytes = max_extent.max(sum_lengths);
        Self {
            batches,
            total_bytes,
        }
    }
}

/// Node bridge receiver that populates anonymous `memfd` arenas from Arrow Flight batches.
#[derive(Debug)]
pub struct FlightReceiver {
    arena: MemfdArena,
    bytes_written: usize,
}

impl FlightReceiver {
    /// Creates a new receiver with an anonymous `memfd` sized to `capacity_bytes`.
    ///
    /// # Errors
    /// Returns [`FlightError::AllocationFailed`] if arena creation fails.
    pub fn new(name: &str, capacity_bytes: usize) -> Result<Self, FlightError> {
        let arena = MemfdArena::new(name, capacity_bytes)?;
        Ok(Self {
            arena,
            bytes_written: 0,
        })
    }

    /// Returns a reference to the underlying `MemfdArena`.
    #[must_use]
    pub fn arena(&self) -> &MemfdArena {
        &self.arena
    }

    /// Returns a mutable reference to the underlying `MemfdArena`.
    pub fn arena_mut(&mut self) -> &mut MemfdArena {
        &mut self.arena
    }

    /// Writes an incoming Arrow [`RecordBatch`] directly into the anonymous `memfd`.
    ///
    /// # Invariant
    /// Extracts `byte_offset` and raw `weights` binary data and copies directly
    /// into the shared memory segment without intermediate allocations.
    ///
    /// # Errors
    /// Returns [`FlightError`] if parsing or writing fails.
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<usize, FlightError> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(0);
        }

        if batch.num_columns() < 6 {
            return Err(FlightError::InvalidBatch(format!(
                "record batch requires at least 6 columns, got {}",
                batch.num_columns()
            )));
        }

        let offsets = batch
            .column(3)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| FlightError::InvalidBatch("column 3 is not UInt64Array".to_owned()))?;

        let weights = batch
            .column(5)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| FlightError::InvalidBatch("column 5 is not BinaryArray".to_owned()))?;

        let mut batch_bytes = 0;

        for i in 0..num_rows {
            let offset = offsets.value(i) as usize;
            let slice = weights.value(i);

            self.arena.write_at(offset, slice)?;
            batch_bytes += slice.len();
        }

        self.bytes_written += batch_bytes;
        Ok(batch_bytes)
    }

    /// Consumes the receiver and returns the underlying open `RawFd`.
    #[must_use]
    pub fn into_raw_fd(self) -> RawFd {
        self.arena.into_raw_fd()
    }

    /// Locks arena memory in RAM (`mlock`) and applies `F_SEAL_WRITE`.
    ///
    /// # Errors
    /// Returns [`FlightError`] if memlock ceiling is breached or sealing fails.
    pub fn finalize(&mut self, ceiling_bytes: Option<u64>) -> Result<(), FlightError> {
        // 1. Lock physical pages in RAM
        self.arena.mlock(ceiling_bytes)?;

        // 2. Apply write seal
        self.arena.seal_write()?;

        Ok(())
    }

    /// Transfers the sealed `memfd` descriptor to a compute worker over UDS via `SCM_RIGHTS`.
    ///
    /// # Errors
    /// Returns [`FlightError::UdsError`] on transfer failure.
    pub fn transfer_to_worker(
        &self,
        stream: &UnixStream,
        model_id: &str,
    ) -> Result<(), FlightError> {
        self.arena.transfer_via_uds(stream, model_id.as_bytes())
    }

    /// Implements the `PROJECT.md` interface contract:
    /// `FlightReceiver::receive_to_memfd(flight_stream: FlightStream) -> Result<RawFd, K3Error>`
    ///
    /// Receives all tensor batches, writes them directly into a sealed `memfd`,
    /// calls `mlock()`, applies `F_SEAL_WRITE`, and returns the raw file descriptor.
    ///
    /// # Errors
    /// Returns [`K3Error`] on failure.
    pub fn receive_to_memfd(flight_stream: FlightStream) -> Result<RawFd, K3Error> {
        let mut max_extent = flight_stream.total_bytes;
        for batch in &flight_stream.batches {
            if batch.num_columns() >= 6 {
                if let (Some(offsets), Some(weights)) = (
                    batch.column(3).as_any().downcast_ref::<UInt64Array>(),
                    batch.column(5).as_any().downcast_ref::<BinaryArray>(),
                ) {
                    let rows = offsets.len().min(weights.len());
                    for i in 0..rows {
                        let extent =
                            (offsets.value(i) as usize).saturating_add(weights.value(i).len());
                        if extent > max_extent {
                            max_extent = extent;
                        }
                    }
                }
            }
        }

        let capacity = max_extent.max(4096);
        let mut receiver = Self::new("flight_received_arena", capacity).map_err(K3Error::from)?;

        for batch in &flight_stream.batches {
            receiver.write_batch(batch).map_err(K3Error::from)?;
        }

        receiver.finalize(None).map_err(K3Error::from)?;

        Ok(receiver.into_raw_fd())
    }
}
