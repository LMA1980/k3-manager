//! Apache Arrow Flight Tensor Streaming & Memfd IPC (US-19, ADR-037, ADR-038).
//!
//! Provides zero-copy model migration of IME-repacked tensor batches (`8×16` `i8` blocks)
//! over Flight IPC, streaming directly into anonymous `memfd` arenas with `mlock()`,
//! `F_SEAL_WRITE`, and `SCM_RIGHTS` file descriptor passing.

pub mod memfd;
pub mod receiver;
pub mod streamer;

pub use memfd::{BOUNDED_MEMLOCK_CEILING_BYTES, FlightError, MemfdArena, TYPE_ID_FLIGHT_ARENA};
pub use receiver::{FlightReceiver, FlightStream};
pub use streamer::{
    FlightSummary, FlightTensorBatchStreamer, IME_TILE_ELEMENTS, flight_tensor_schema,
};
