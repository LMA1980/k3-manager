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

pub use crate::domain::{
    CgroupFs, CgroupV2DomainController, DomainPlacementMode, IsolationDomainController,
    MockCgroupFs, MockOsAffinityBackend, OsAffinityBackend, PlacementOutcome, RealCgroupFs,
    RealOsAffinityBackend, format_cpuset_list, parse_cpuset_list, resolve_worker_compute_mask,
    validate_model_id,
};

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
    #[allow(clippy::too_many_lines)]
    fn validate_worker_placement(&self, placement: &WorkerPlacement) -> Result<(), ManagerError> {
        let cluster = placement.cluster.to_ascii_lowercase();
        if cluster != "x100" && cluster != "a100" {
            return Err(ManagerError::WorkerSpawn(format!(
                "unsupported cluster '{0}'; must be 'x100' or 'a100'",
                placement.cluster
            )));
        }

        // 1. Validate requested thread count bounds upfront
        if let Some(threads) = placement.threads {
            if threads == 0 {
                return Err(ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!(
                        "worker placement on '{0}' requested 0 threads; must be greater than 0",
                        placement.model_id
                    ),
                }));
            }

            if cluster == "x100" && threads as usize > self.x100_max_compute_cores {
                return Err(ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: self.x100_compute_cap_mask,
                    requested_threads: threads as usize,
                    reason: format!(
                        "worker placement on '{0}' requests {threads} threads, exceeding X100 max compute limit {1}",
                        placement.model_id, self.x100_max_compute_cores
                    ),
                }));
            }

            if cluster == "a100" && threads as usize > self.a100_max_compute_cores {
                return Err(ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: A100_CLUSTER_MASK,
                    requested_threads: threads as usize,
                    reason: format!(
                        "worker placement on '{0}' requests {threads} threads, exceeding A100 max compute limit {1}",
                        placement.model_id, self.a100_max_compute_cores
                    ),
                }));
            }
        }

        // 2. Validate explicit affinity specification if provided
        if let Some(ref aff_str) = placement.affinity {
            let trimmed = aff_str.trim();
            if trimmed.starts_with('-') {
                return Err(ManagerError::WorkerSpawn(format!(
                    "invalid integer affinity '{aff_str}': parse error"
                )));
            }

            let mask = if let Some(hex_str) = trimmed
                .strip_prefix("0x")
                .or_else(|| trimmed.strip_prefix("0X"))
            {
                u64::from_str_radix(hex_str, 16).map_err(|e| {
                    ManagerError::WorkerSpawn(format!("invalid hex affinity '{aff_str}': {e}"))
                })?
            } else if trimmed.contains('-') || trimmed.contains(',') {
                if let Ok(val) = parse_cpuset_list(trimmed) {
                    val
                } else {
                    return Err(ManagerError::WorkerSpawn(format!(
                        "invalid integer affinity '{aff_str}': parse error"
                    )));
                }
            } else if let Ok(val) = trimmed.parse::<u64>() {
                if cluster == "a100" && (8..=15).contains(&val) {
                    1_u64 << val
                } else {
                    val
                }
            } else if let Ok(val) = parse_cpuset_list(trimmed) {
                val
            } else {
                return Err(ManagerError::WorkerSpawn(format!(
                    "invalid integer affinity '{aff_str}': parse error"
                )));
            };

            // If placement targets X100, ensure it does not overlap active control cores or exceed compute cap
            if cluster == "x100" {
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

                if (mask & !X100_CLUSTER_MASK) != 0 {
                    return Err(ManagerError::Affinity(K3Error::AffinityError {
                        requested_mask: mask,
                        requested_threads: mask.count_ones() as usize,
                        reason: format!(
                            "worker placement on '{0}' targets cluster 'x100' but specifies cores outside X100 mask 0x{X100_CLUSTER_MASK:04X}",
                            placement.model_id
                        ),
                    }));
                }
            } else if cluster == "a100" {
                if (mask & self.active_control_mask) != 0 {
                    return Err(ManagerError::Affinity(K3Error::AffinityError {
                        requested_mask: mask,
                        requested_threads: mask.count_ones() as usize,
                        reason: format!(
                            "worker placement on '{0}' encroaches on dedicated active control cores 0x{1:04X}",
                            placement.model_id, self.active_control_mask
                        ),
                    }));
                }

                if (mask & !A100_CLUSTER_MASK) != 0 {
                    return Err(ManagerError::Affinity(K3Error::AffinityError {
                        requested_mask: mask,
                        requested_threads: mask.count_ones() as usize,
                        reason: format!(
                            "worker placement on '{0}' targets cluster 'a100' but specifies cores outside A100 mask 0x{A100_CLUSTER_MASK:04X}",
                            placement.model_id
                        ),
                    }));
                }
            }

            if let Some(threads) = placement.threads {
                if mask != 0 && threads as usize > mask.count_ones() as usize {
                    return Err(ManagerError::Affinity(K3Error::AffinityError {
                        requested_mask: mask,
                        requested_threads: threads as usize,
                        reason: format!(
                            "worker placement on '{0}' requests {threads} threads, exceeding {1} available cores in affinity mask 0x{mask:04X}",
                            placement.model_id,
                            mask.count_ones()
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

    /// Places the control plane into its isolated cgroup v2 domain (`.../control`) or OS affinity fallback.
    ///
    /// If `target_pid` is 0, targets the current process ID (`std::process::id()`).
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if domain placement fails.
    pub fn place_control_plane(
        &self,
        target_pid: u32,
        controller: &dyn IsolationDomainController,
    ) -> Result<PlacementOutcome, ManagerError> {
        controller.place_control_plane(target_pid, &self.capacity)
    }
}
