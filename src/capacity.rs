//! Control-plane capacity management, core affinity resolution, and task scheduling.
//!
//! Re-exports foundational capacity types and cluster masks from `k3-core::affinity`,
//! and defines the manager-level [`ControlPlaneScheduler`] for round-robin control task
//! dispatch, health supervisor watchdog core binding, and compute worker placement validation
//! per ADR-007, ADR-008, ADR-041, and Issue #69.

pub use k3_core::affinity::{
    A100_CLUSTER_MASK, A100_DEFAULT_MAX_COMPUTE_THREADS, ActiveControlRotator,
    ControlPlaneCapacityConfig, DEFAULT_A100_MAX_COMPUTE_CORES, DEFAULT_ACTIVE_CONTROL_MASK,
    DEFAULT_MAX_A100_COMPUTE_CORES, DEFAULT_MAX_X100_COMPUTE_CORES, DEFAULT_SPARE_RECOVERY_MASK,
    DEFAULT_X100_COMPUTE_CAP_MASK, DEFAULT_X100_MAX_COMPUTE_CORES, ResolvedControlCapacity,
    SOC_CLUSTER_MASK, X100_CLUSTER_MASK, X100_DEFAULT_MAX_COMPUTE_THREADS,
};

use k3_core::K3Error;

use crate::error::ManagerError;
use crate::supervisor::WorkerPlacement;

/// Extension trait providing manager-specific placement validation on [`ResolvedControlCapacity`].
pub trait ResolvedControlCapacityExt {
    /// Validates a worker placement compute affinity mask against the resolved capacity.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::WorkerSpawn`] if the affinity mask string cannot be parsed,
    /// or [`ManagerError::Affinity`] if the placement encroaches on dedicated active control cores
    /// or exceeds the X100 compute cap.
    fn validate_worker_placement(&self, placement: &WorkerPlacement) -> Result<(), ManagerError>;
}

impl ResolvedControlCapacityExt for ResolvedControlCapacity {
    fn validate_worker_placement(&self, placement: &WorkerPlacement) -> Result<(), ManagerError> {
        if let Some(ref aff_str) = placement.affinity {
            let mask = if let Some(hex_str) = aff_str
                .strip_prefix("0x")
                .or_else(|| aff_str.strip_prefix("0X"))
            {
                u64::from_str_radix(hex_str, 16).map_err(|e| {
                    ManagerError::WorkerSpawn(format!("invalid hex affinity '{aff_str}': {e}"))
                })?
            } else {
                aff_str.parse::<u64>().map_err(|e| {
                    ManagerError::WorkerSpawn(format!("invalid integer affinity '{aff_str}': {e}"))
                })?
            };

            // If placement targets X100, ensure it does not overlap active control cores or exceed compute cap
            if placement.cluster.eq_ignore_ascii_case("x100") {
                if (mask & self.active_control_mask) != 0 {
                    return Err(ManagerError::Affinity(K3Error::AffinityError {
                        requested_mask: mask,
                        requested_threads: mask.count_ones() as usize,
                        reason: format!(
                            "worker placement on '{0}' encroaches on dedicated active control cores 0x{1:04X} (intersection: 0x{2:04X})",
                            placement.model_id,
                            self.active_control_mask,
                            mask & self.active_control_mask
                        ),
                    }));
                }

                if (mask & !self.x100_compute_cap_mask & X100_CLUSTER_MASK) != 0 {
                    return Err(ManagerError::Affinity(K3Error::AffinityError {
                        requested_mask: mask,
                        requested_threads: mask.count_ones() as usize,
                        reason: format!(
                            "worker placement on '{0}' exceeds X100 compute cap mask 0x{1:04X}",
                            placement.model_id, self.x100_compute_cap_mask
                        ),
                    }));
                }
            }
        }
        Ok(())
    }
}

/// Manager-level task scheduler distributing control-plane tasks and binding supervisor watchdogs.
#[derive(Debug, Clone)]
pub struct ControlPlaneScheduler {
    capacity: ResolvedControlCapacity,
}

impl ControlPlaneScheduler {
    /// Creates a new [`ControlPlaneScheduler`] bound to the given resolved control capacity.
    #[must_use]
    pub const fn new(capacity: ResolvedControlCapacity) -> Self {
        Self { capacity }
    }

    /// Returns a reference to the underlying [`ResolvedControlCapacity`].
    #[must_use]
    pub const fn capacity(&self) -> &ResolvedControlCapacity {
        &self.capacity
    }

    /// Dispatches a control-plane task to the next active control core in round-robin sequence.
    #[must_use]
    pub fn schedule_control_task(&self) -> usize {
        self.capacity.next_active_control_core()
    }

    /// Returns the dedicated spare recovery core for health supervisor watchdog binding.
    #[must_use]
    pub fn watchdog_recovery_core(&self) -> Option<usize> {
        self.capacity.spare_recovery_core()
    }

    /// Validates a worker placement compute affinity mask against the resolved capacity.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if the affinity mask is invalid or violates capacity boundaries.
    pub fn validate_worker_placement(
        &self,
        placement: &WorkerPlacement,
    ) -> Result<(), ManagerError> {
        self.capacity.validate_worker_placement(placement)
    }
}
