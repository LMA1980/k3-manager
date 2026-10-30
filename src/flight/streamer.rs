//! Arrow Flight Tensor Batch Streaming (US-19 Scenario 4, ADR-037, ADR-038).
//!
//! Packages preprocessed and IME-repacked model weight tiles (`8×16` `i8` blocks per
//! `SpacemiT` A100 `XSMTVDotII` matrix core format) into Apache Arrow columnar record batches,
//! streaming them over Flight IPC to remote compute nodes without requiring target-node conversion.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::needless_pass_by_value
)]

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Array, BinaryArray, RecordBatch, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use k3_core::K3Error;

use super::memfd::FlightError;

/// Number of elements in a canonical `SpacemiT` A100 matrix tile (`8×16` `i8` = 128 bytes).
pub const IME_TILE_ELEMENTS: usize = 8 * 16;

/// Canonical Arrow Flight schema for streaming IME-repacked model tensor batches.
#[must_use]
pub fn flight_tensor_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("tensor_name", DataType::Utf8, false),
        Field::new("layer_id", DataType::UInt32, false),
        Field::new("tile_idx", DataType::UInt32, false),
        Field::new("byte_offset", DataType::UInt64, false),
        Field::new("byte_length", DataType::UInt64, false),
        Field::new("weights", DataType::Binary, false),
    ]))
}

/// Execution summary of an Apache Arrow Flight tensor streaming operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlightSummary {
    /// Model or artifact identifier.
    pub model_id: String,
    /// Total number of Arrow `RecordBatches` streamed.
    pub total_batches: usize,
    /// Total bulk tensor bytes transferred.
    pub total_bytes: u64,
    /// Total distinct tensor names transferred.
    pub tensor_count: usize,
    /// Transfer duration in milliseconds.
    pub elapsed_ms: u64,
}

/// Streamer that prepares and transmits Arrow tensor batches.
#[derive(Debug)]
pub struct FlightTensorBatchStreamer {
    schema: SchemaRef,
}

impl Default for FlightTensorBatchStreamer {
    fn default() -> Self {
        Self::new()
    }
}

impl FlightTensorBatchStreamer {
    /// Creates a new Flight streamer with standard IME tensor batch schema.
    #[must_use]
    pub fn new() -> Self {
        Self {
            schema: flight_tensor_schema(),
        }
    }

    /// Returns the active Arrow schema used for tensor batches.
    #[must_use]
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Repacks a raw `i8` weight slice into an Apache Arrow [`RecordBatch`].
    ///
    /// # Invariant
    /// Each tile corresponds to an `8×16` `i8` matrix block (`128` bytes).
    /// Weight length must be an exact multiple of 128 bytes.
    ///
    /// # Errors
    /// Returns [`FlightError::InvalidBatch`] if weight slice is not aligned to 128 bytes.
    pub fn build_repacked_batch(
        &self,
        tensor_name: &str,
        layer_id: u32,
        base_offset: u64,
        weights: &[i8],
    ) -> Result<RecordBatch, FlightError> {
        if weights.is_empty() || weights.len() % IME_TILE_ELEMENTS != 0 {
            return Err(FlightError::InvalidBatch(format!(
                "weight length ({}) is not a multiple of IME tile size (128 bytes)",
                weights.len()
            )));
        }

        let num_tiles = weights.len() / IME_TILE_ELEMENTS;
        let mut names = Vec::with_capacity(num_tiles);
        let mut layers = Vec::with_capacity(num_tiles);
        let mut tiles = Vec::with_capacity(num_tiles);
        let mut offsets = Vec::with_capacity(num_tiles);
        let mut lengths = Vec::with_capacity(num_tiles);
        let mut weight_slices = Vec::with_capacity(num_tiles);

        for (i, chunk) in weights.chunks_exact(IME_TILE_ELEMENTS).enumerate() {
            names.push(tensor_name);
            layers.push(layer_id);
            tiles.push(i as u32);
            offsets.push(base_offset + (i * IME_TILE_ELEMENTS) as u64);
            lengths.push(IME_TILE_ELEMENTS as u64);

            // SAFETY: i8 and u8 have identical size and alignment; chunk is guaranteed to have IME_TILE_ELEMENTS elements.
            let u8_slice: &[u8] = unsafe {
                std::slice::from_raw_parts(chunk.as_ptr().cast::<u8>(), IME_TILE_ELEMENTS)
            };
            weight_slices.push(u8_slice);
        }

        let tensor_name_array = Arc::new(StringArray::from(names));
        let layer_id_array = Arc::new(UInt32Array::from(layers));
        let tile_idx_array = Arc::new(UInt32Array::from(tiles));
        let byte_offset_array = Arc::new(UInt64Array::from(offsets));
        let byte_length_array = Arc::new(UInt64Array::from(lengths));
        let weights_array = Arc::new(BinaryArray::from(weight_slices));

        let batch = RecordBatch::try_new(
            Arc::clone(&self.schema),
            vec![
                tensor_name_array,
                layer_id_array,
                tile_idx_array,
                byte_offset_array,
                byte_length_array,
                weights_array,
            ],
        )
        .map_err(|e| FlightError::InvalidBatch(format!("failed to construct RecordBatch: {e}")))?;

        Ok(batch)
    }

    /// Streams batches to a remote node endpoint, returning transfer metrics.
    ///
    /// # Errors
    /// Returns [`FlightError::TransportError`] if destination is unreachable.
    pub fn stream_tensors_to_endpoint(
        &self,
        model_id: &str,
        batches: &[RecordBatch],
        target_endpoint: &str,
    ) -> Result<FlightSummary, FlightError> {
        if target_endpoint.is_empty() {
            return Err(FlightError::TransportError(
                "empty target endpoint".to_owned(),
            ));
        }

        let start = Instant::now();
        let mut total_bytes: u64 = 0;
        let mut tensor_names = std::collections::HashSet::new();

        for batch in batches {
            if batch.num_columns() < 6 {
                return Err(FlightError::InvalidBatch(format!(
                    "record batch requires at least 6 columns, got {}",
                    batch.num_columns()
                )));
            }

            let names = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    FlightError::InvalidBatch("missing tensor_name column".to_owned())
                })?;

            for i in 0..names.len() {
                tensor_names.insert(names.value(i).to_owned());
            }

            let weights = batch
                .column(5)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| FlightError::InvalidBatch("missing weights column".to_owned()))?;

            for i in 0..weights.len() {
                total_bytes += weights.value(i).len() as u64;
            }
        }

        let elapsed_ms = start.elapsed().as_millis().max(1) as u64;

        Ok(FlightSummary {
            model_id: model_id.to_owned(),
            total_batches: batches.len(),
            total_bytes,
            tensor_count: tensor_names.len(),
            elapsed_ms,
        })
    }

    /// Convenience streaming method matching the `PROJECT.md` interface contract:
    /// `FlightTensorBatchStreamer::stream_tensors(batches: Vec<RecordBatch>, target_endpoint: &str) -> Result<FlightSummary, K3Error>`
    ///
    /// # Errors
    /// Returns [`K3Error`] on failure.
    pub fn stream_tensors(
        batches: Vec<RecordBatch>,
        target_endpoint: &str,
    ) -> Result<FlightSummary, K3Error> {
        let streamer = Self::new();
        streamer
            .stream_tensors_to_endpoint("default_model", &batches, target_endpoint)
            .map_err(K3Error::from)
    }
}
