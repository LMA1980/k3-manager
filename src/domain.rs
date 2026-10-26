//! Cgroup v2 leaf domain placement and OS thread/process affinity fallback.
//!
//! Enforces control-plane (`.../control`) and compute worker (`.../engines/<model>`)
//! isolation domains via cgroup v2 `cpuset.cpus` with automatic fallback to OS
//! thread/process affinity (`sched_setaffinity`) per ADR-007, ADR-008, ADR-041,
//! `specs/runtime/isolation.md`, and Issue #70.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use k3_core::K3Error;
use k3_core::affinity::{
    A100_CLUSTER_MASK, DEFAULT_X100_COMPUTE_CAP_MASK, ResolvedControlCapacity, SOC_CLUSTER_MASK,
};

use crate::error::ManagerError;
use crate::supervisor::WorkerPlacement;

// ============================================================================
// Canonical cpuset.cpus list syntax formatter and parser
// ============================================================================

/// Converts a 64-bit physical core bitmask into canonical Linux `cpuset.cpus` list syntax.
///
/// Formats contiguous runs of 2 or more cores as ranges (`start-end`), isolated cores
/// as individual numbers (`core`), and disjoint runs as comma-separated entries
/// (e.g. `0-5`, `6-7`, `8-15`, `0-2,6-7`).
///
/// # Errors
///
/// Returns [`ManagerError::Affinity`] if `mask == 0` or if `mask` specifies cores outside
/// the 16-core `SoC` topology (`0xFFFF`).
pub fn format_cpuset_list(mask: u64) -> Result<String, ManagerError> {
    if mask == 0 {
        return Err(ManagerError::Affinity(K3Error::AffinityError {
            requested_mask: 0,
            requested_threads: 0,
            reason: "core bitmask cannot be zero".to_string(),
        }));
    }
    if (mask & !SOC_CLUSTER_MASK) != 0 {
        return Err(ManagerError::Affinity(K3Error::AffinityError {
            requested_mask: mask,
            requested_threads: mask.count_ones() as usize,
            reason: format!(
                "core bitmask 0x{mask:04X} specifies cores outside valid 16-core SoC mask 0x{SOC_CLUSTER_MASK:04X}"
            ),
        }));
    }

    let cores: Vec<usize> = (0..16).filter(|&i| (mask & (1 << i)) != 0).collect();
    let mut ranges: Vec<String> = Vec::new();
    let mut i = 0;
    while i < cores.len() {
        let start = cores[i];
        let mut end = start;
        while i + 1 < cores.len() && cores[i + 1] == end + 1 {
            end = cores[i + 1];
            i += 1;
        }
        if start == end {
            ranges.push(format!("{start}"));
        } else {
            ranges.push(format!("{start}-{end}"));
        }
        i += 1;
    }

    Ok(ranges.join(","))
}

/// Parses canonical Linux `cpuset.cpus` list syntax into a 64-bit physical core bitmask.
///
/// Accepts comma-separated single core IDs (e.g. `5`) and ranges (e.g. `0-5`, `8-15`).
///
/// # Errors
///
/// Returns [`ManagerError::Affinity`] if the syntax is malformed, contains empty tokens,
/// specifies cores `>= 16`, defines an inverted range (`start > end`), or results in a zero mask.
#[allow(clippy::too_many_lines)]
pub fn parse_cpuset_list(s: &str) -> Result<u64, ManagerError> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err(ManagerError::Affinity(K3Error::AffinityError {
            requested_mask: 0,
            requested_threads: 0,
            reason: "cpuset string cannot be empty".to_string(),
        }));
    }

    let mut mask = 0_u64;
    for token in trimmed.split(',') {
        let t = token.trim();
        if t.is_empty() {
            return Err(ManagerError::Affinity(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads: 0,
                reason: format!("malformed cpuset string '{trimmed}': empty entry between commas"),
            }));
        }

        if let Some((start_str, end_str)) = t.split_once('-') {
            let s_trimmed = start_str.trim();
            let e_trimmed = end_str.trim();
            if s_trimmed.is_empty()
                || !s_trimmed.chars().all(|c| c.is_ascii_digit())
                || e_trimmed.is_empty()
                || !e_trimmed.chars().all(|c| c.is_ascii_digit())
            {
                return Err(ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!("invalid range format '{t}' in cpuset '{trimmed}'"),
                }));
            }

            let start: usize = s_trimmed.parse().map_err(|e| {
                ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!("invalid range start '{start_str}' in cpuset '{trimmed}': {e}"),
                })
            })?;
            let end: usize = e_trimmed.parse().map_err(|e| {
                ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!("invalid range end '{end_str}' in cpuset '{trimmed}': {e}"),
                })
            })?;

            if start > end {
                return Err(ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!(
                        "inverted cpuset range '{t}' in '{trimmed}': start {start} > end {end}"
                    ),
                }));
            }
            if end >= 16 {
                return Err(ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!(
                        "cpuset core {end} in range '{t}' exceeds 16-core SoC boundary (0..15)"
                    ),
                }));
            }

            for core in start..=end {
                mask |= 1 << core;
            }
        } else {
            if !t.chars().all(|c| c.is_ascii_digit()) {
                return Err(ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!(
                        "invalid core '{t}' in cpuset '{trimmed}': non-digit characters"
                    ),
                }));
            }

            let core: usize = t.parse().map_err(|e| {
                ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!("invalid core '{t}' in cpuset '{trimmed}': {e}"),
                })
            })?;

            if core >= 16 {
                return Err(ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!("cpuset core {core} exceeds 16-core SoC boundary (0..15)"),
                }));
            }
            mask |= 1 << core;
        }
    }

    if mask == 0 {
        return Err(ManagerError::Affinity(K3Error::AffinityError {
            requested_mask: 0,
            requested_threads: 0,
            reason: "parsed cpuset resulted in zero mask".to_string(),
        }));
    }

    Ok(mask)
}

/// Resolves the effective compute core bitmask for a [`WorkerPlacement`].
///
/// Parses hex (`0x...`), decimal, or cpuset list syntax if specified in `placement.affinity`,
/// or applies default cluster compute masks:
/// - Cluster `"x100"`: [`DEFAULT_X100_COMPUTE_CAP_MASK`] (`0x003F`, cores 0–5)
/// - Cluster `"a100"`: [`A100_CLUSTER_MASK`] (`0xFF00`, cores 8–15)
///
/// # Errors
///
/// Returns [`ManagerError::WorkerSpawn`] if affinity string cannot be parsed,
/// or [`ManagerError::Affinity`] if the resolved mask is zero or out of bounds.
pub fn resolve_worker_compute_mask(placement: &WorkerPlacement) -> Result<u64, ManagerError> {
    if let Some(ref aff_str) = placement.affinity {
        let trimmed = aff_str.trim();
        if trimmed.is_empty() {
            return Err(ManagerError::WorkerSpawn(format!(
                "invalid integer affinity '{aff_str}': empty string"
            )));
        }
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
        } else if let Ok(val) = trimmed.parse::<u64>() {
            val
        } else if let Ok(val) = parse_cpuset_list(trimmed) {
            val
        } else {
            return Err(ManagerError::WorkerSpawn(format!(
                "invalid integer affinity '{aff_str}': parse error"
            )));
        };

        if mask == 0 {
            return Err(ManagerError::Affinity(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads: 0,
                reason: "worker compute affinity mask cannot be zero".to_string(),
            }));
        }
        if (mask & !SOC_CLUSTER_MASK) != 0 {
            return Err(ManagerError::Affinity(K3Error::AffinityError {
                requested_mask: mask,
                requested_threads: mask.count_ones() as usize,
                reason: format!(
                    "worker compute affinity 0x{mask:04X} specifies cores outside 16-core SoC mask 0x{SOC_CLUSTER_MASK:04X}"
                ),
            }));
        }
        Ok(mask)
    } else if placement.cluster.eq_ignore_ascii_case("a100") {
        Ok(A100_CLUSTER_MASK)
    } else {
        Ok(DEFAULT_X100_COMPUTE_CAP_MASK)
    }
}

/// Validates that a model identifier is safe for cgroup leaf directory naming.
///
/// # Errors
/// Returns [`ManagerError::WorkerSpawn`] if `model_id` is empty, whitespace-only,
/// contains leading/trailing whitespace, path separators (`/`, `\`), control characters, null bytes,
/// or directory traversal (`..`, `.`).
pub fn validate_model_id(model_id: &str) -> Result<(), ManagerError> {
    let trimmed = model_id.trim();
    if trimmed.is_empty() {
        return Err(ManagerError::WorkerSpawn(
            "model_id cannot be empty".to_string(),
        ));
    }
    if model_id != trimmed {
        return Err(ManagerError::WorkerSpawn(format!(
            "invalid model_id '{model_id}': cannot contain leading or trailing whitespace"
        )));
    }
    if trimmed == "."
        || trimmed == ".."
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains("..")
        || trimmed.chars().any(|c| c.is_control() || c.is_whitespace())
    {
        return Err(ManagerError::WorkerSpawn(format!(
            "invalid model_id '{model_id}': cannot contain path separators, whitespace, control characters, or traversal components"
        )));
    }
    Ok(())
}

// ============================================================================
// Placement Outcome Types
// ============================================================================

/// Mode and details under which domain isolation was enforced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomainPlacementMode {
    /// Enforced via cgroup v2 cpuset and proc/thread migration.
    CgroupV2 {
        /// Leaf cgroup path (e.g. `/sys/fs/cgroup/k3.slice/control` or `.../engines/<model>`).
        cgroup_path: PathBuf,
        /// Canonical cpuset list string applied (e.g. `"6-7"` or `"0-5"`).
        cpuset: String,
    },
    /// Enforced via OS thread/process affinity (`sched_setaffinity`).
    OsAffinityFallback {
        /// Core bitmask applied via OS affinity.
        mask: u64,
        /// Diagnostic reason why cgroup v2 placement fell back to OS affinity.
        reason: String,
    },
}

impl DomainPlacementMode {
    /// Returns `true` if isolation was enforced via cgroup v2.
    #[must_use]
    pub const fn is_cgroup_v2(&self) -> bool {
        matches!(self, Self::CgroupV2 { .. })
    }

    /// Returns `true` if isolation was enforced via OS thread/process affinity fallback.
    #[must_use]
    pub const fn is_os_affinity_fallback(&self) -> bool {
        matches!(self, Self::OsAffinityFallback { .. })
    }
}

/// Structured outcome of an isolation domain placement operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementOutcome {
    /// Target process ID or thread ID that was placed.
    pub target_id: u32,
    /// Placement mode and diagnostics.
    pub mode: DomainPlacementMode,
}

// ============================================================================
// Mockable Cgroup Filesystem Backend Abstraction
// ============================================================================

/// Filesystem backend interface for cgroup v2 hierarchy operations.
pub trait CgroupFs: Send + Sync + fmt::Debug {
    /// Creates directory and parent directories.
    ///
    /// # Errors
    /// Returns [`io::Error`] on failure.
    fn create_dir_all(&self, path: &Path) -> io::Result<()>;

    /// Writes byte contents to file.
    ///
    /// # Errors
    /// Returns [`io::Error`] on failure.
    fn write(&self, path: &Path, contents: &[u8]) -> io::Result<()>;

    /// Reads file content into a string.
    ///
    /// # Errors
    /// Returns [`io::Error`] on failure.
    fn read_to_string(&self, path: &Path) -> io::Result<String>;

    /// Returns `true` if path exists.
    fn exists(&self, path: &Path) -> bool;
}

/// Standard host filesystem implementation using `std::fs`.
#[derive(Debug, Default, Clone, Copy)]
pub struct RealCgroupFs;

impl CgroupFs for RealCgroupFs {
    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        std::fs::create_dir_all(path)
    }

    fn write(&self, path: &Path, contents: &[u8]) -> io::Result<()> {
        std::fs::write(path, contents)
    }

    fn read_to_string(&self, path: &Path) -> io::Result<String> {
        std::fs::read_to_string(path)
    }

    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }
}

/// In-memory mock cgroup filesystem backend for deterministic error injection and verification.
#[derive(Debug, Default)]
pub struct MockCgroupFs {
    files: Mutex<HashMap<PathBuf, Vec<u8>>>,
    write_history: Mutex<Vec<(PathBuf, Vec<u8>)>>,
    created_dirs: Mutex<HashSet<PathBuf>>,
    error_injections: Mutex<HashMap<PathBuf, io::ErrorKind>>,
}

#[allow(clippy::missing_panics_doc)]
impl MockCgroupFs {
    /// Creates a new empty [`MockCgroupFs`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            files: Mutex::new(HashMap::new()),
            write_history: Mutex::new(Vec::new()),
            created_dirs: Mutex::new(HashSet::new()),
            error_injections: Mutex::new(HashMap::new()),
        }
    }

    /// Injects a simulated I/O error kind for operations on `path`.
    pub fn inject_error(&self, path: impl Into<PathBuf>, kind: io::ErrorKind) {
        self.error_injections
            .lock()
            .expect("lock error_injections")
            .insert(path.into(), kind);
    }

    /// Clears any injected error for `path`.
    pub fn clear_error(&self, path: impl Into<PathBuf>) {
        self.error_injections
            .lock()
            .expect("lock error_injections")
            .remove(&path.into());
    }

    /// Returns the string content written to `path`, if any.
    #[must_use]
    pub fn written_file(&self, path: impl AsRef<Path>) -> Option<String> {
        self.files
            .lock()
            .expect("lock files")
            .get(path.as_ref())
            .and_then(|bytes| String::from_utf8(bytes.clone()).ok())
    }

    /// Returns the write history as `(path, contents)` pairs in invocation order.
    #[must_use]
    pub fn write_history(&self) -> Vec<(PathBuf, Vec<u8>)> {
        self.write_history.lock().expect("lock history").clone()
    }

    /// Returns the sequence of written file paths in invocation order.
    #[must_use]
    pub fn written_paths(&self) -> Vec<PathBuf> {
        self.write_history
            .lock()
            .expect("lock history")
            .iter()
            .map(|(p, _)| p.clone())
            .collect()
    }

    /// Returns `true` if `path` was created as a directory.
    #[must_use]
    pub fn is_dir_created(&self, path: impl AsRef<Path>) -> bool {
        self.created_dirs
            .lock()
            .expect("lock dirs")
            .contains(path.as_ref())
    }

    /// Returns snapshot of all written files.
    #[must_use]
    pub fn files(&self) -> HashMap<PathBuf, Vec<u8>> {
        self.files.lock().expect("lock files").clone()
    }

    /// Returns snapshot of all created directories.
    #[must_use]
    pub fn created_dirs(&self) -> HashSet<PathBuf> {
        self.created_dirs.lock().expect("lock dirs").clone()
    }

    fn check_error(&self, path: &Path) -> io::Result<()> {
        let errors = self.error_injections.lock().expect("lock error_injections");
        if let Some(&kind) = errors.get(path) {
            return Err(io::Error::new(
                kind,
                format!("mock injected error on '{}'", path.display()),
            ));
        }
        for ancestor in path.ancestors() {
            if let Some(&kind) = errors.get(ancestor) {
                return Err(io::Error::new(
                    kind,
                    format!(
                        "mock injected ancestor error on '{}' for '{}'",
                        ancestor.display(),
                        path.display()
                    ),
                ));
            }
        }
        Ok(())
    }
}

impl CgroupFs for MockCgroupFs {
    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.check_error(path)?;
        let mut dirs = self.created_dirs.lock().expect("lock dirs");
        for ancestor in path.ancestors() {
            dirs.insert(ancestor.to_path_buf());
        }
        Ok(())
    }

    fn write(&self, path: &Path, contents: &[u8]) -> io::Result<()> {
        self.check_error(path)?;
        self.files
            .lock()
            .expect("lock files")
            .insert(path.to_path_buf(), contents.to_vec());
        self.write_history
            .lock()
            .expect("lock history")
            .push((path.to_path_buf(), contents.to_vec()));
        Ok(())
    }

    fn read_to_string(&self, path: &Path) -> io::Result<String> {
        self.check_error(path)?;
        self.files
            .lock()
            .expect("lock files")
            .get(path)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "file not found"))
            .and_then(|bytes| {
                String::from_utf8(bytes.clone())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            })
    }

    fn exists(&self, path: &Path) -> bool {
        self.files.lock().expect("lock files").contains_key(path)
            || self.created_dirs.lock().expect("lock dirs").contains(path)
    }
}

// ============================================================================
// OS Thread/Process Affinity Backend Abstraction
// ============================================================================

/// Backend interface for applying OS thread/process affinity (`sched_setaffinity`).
pub trait OsAffinityBackend: Send + Sync + fmt::Debug {
    /// Applies the physical core bitmask to the target PID (or TID).
    ///
    /// If `pid == 0`, targets the calling process/thread.
    ///
    /// # Errors
    /// Returns [`ManagerError::Affinity`] if OS affinity cannot be applied.
    fn set_affinity(&self, pid: u32, mask: u64) -> Result<(), ManagerError>;
}

/// Real OS affinity implementation using Linux syscalls (`sched_setaffinity` & `/proc/set_ai_thread`).
#[derive(Debug, Default, Clone, Copy)]
pub struct RealOsAffinityBackend;

impl OsAffinityBackend for RealOsAffinityBackend {
    fn set_affinity(&self, pid: u32, mask: u64) -> Result<(), ManagerError> {
        #[cfg(target_os = "linux")]
        {
            // AGENTS.md §6: SpacemiT HMP invariant:
            // Before pinning to Cluster 1 (A100 cores 8–15, mask 0xFF00),
            // conditionally write thread ID to /proc/set_ai_thread if it exists.
            if (mask & A100_CLUSTER_MASK) != 0 {
                let set_ai_thread_path = Path::new("/proc/set_ai_thread");
                if set_ai_thread_path.exists() {
                    let tid_to_write = if pid == 0 {
                        // SAFETY: SYS_gettid syscall takes no arguments and has no side effects.
                        unsafe { libc::syscall(libc::SYS_gettid) }
                    } else {
                        pid as libc::c_long
                    };
                    let _ = std::fs::write(set_ai_thread_path, format!("{tid_to_write}\n"));
                }
            }

            // SAFETY: libc::cpu_set_t zero initialization and CPU_SET macro calls.
            let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            for bit in 0..16 {
                if (mask & (1 << bit)) != 0 {
                    // SAFETY: bit is bounded in 0..16, within standard cpu_set_t bounds (1024 cpus).
                    unsafe { libc::CPU_SET(bit, &mut set) };
                }
            }

            let target = pid as libc::pid_t;
            // SAFETY: Invokes standard sched_setaffinity syscall for target pid.
            let ret =
                unsafe { libc::sched_setaffinity(target, size_of::<libc::cpu_set_t>(), &set) };
            if ret != 0 {
                let err = io::Error::last_os_error();
                return Err(ManagerError::Affinity(K3Error::AffinityError {
                    requested_mask: mask,
                    requested_threads: mask.count_ones() as usize,
                    reason: format!("sched_setaffinity failed for pid {pid}: {err}"),
                }));
            }
            Ok(())
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Non-Linux hosts (e.g. macOS host development): no-op success
            let _ = (pid, mask);
            Ok(())
        }
    }
}

/// Mock OS affinity backend for deterministic verification and call tracing.
#[derive(Debug, Default)]
pub struct MockOsAffinityBackend {
    calls: Mutex<Vec<(u32, u64)>>,
    ai_thread_writes: Mutex<Vec<u32>>,
    failing_pids: Mutex<HashSet<u32>>,
}

#[allow(clippy::missing_panics_doc)]
impl MockOsAffinityBackend {
    /// Creates a new empty [`MockOsAffinityBackend`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            ai_thread_writes: Mutex::new(Vec::new()),
            failing_pids: Mutex::new(HashSet::new()),
        }
    }

    /// Configures `pid` to fail when `set_affinity` is called.
    pub fn fail_pid(&self, pid: u32) {
        self.failing_pids
            .lock()
            .expect("lock failing_pids")
            .insert(pid);
    }

    /// Returns recorded `(pid, mask)` affinity calls.
    #[must_use]
    pub fn calls(&self) -> Vec<(u32, u64)> {
        self.calls.lock().expect("lock calls").clone()
    }

    /// Returns recorded PIDs registered via simulated `/proc/set_ai_thread`.
    #[must_use]
    pub fn ai_thread_writes(&self) -> Vec<u32> {
        self.ai_thread_writes.lock().expect("lock ai").clone()
    }

    /// Returns `true` if `(pid, mask)` was called.
    #[must_use]
    pub fn has_called(&self, pid: u32, mask: u64) -> bool {
        self.calls
            .lock()
            .expect("lock calls")
            .contains(&(pid, mask))
    }
}

impl OsAffinityBackend for MockOsAffinityBackend {
    fn set_affinity(&self, pid: u32, mask: u64) -> Result<(), ManagerError> {
        if self
            .failing_pids
            .lock()
            .expect("lock failing_pids")
            .contains(&pid)
        {
            return Err(ManagerError::Affinity(K3Error::AffinityError {
                requested_mask: mask,
                requested_threads: mask.count_ones() as usize,
                reason: format!("mock sched_setaffinity failure for pid {pid}"),
            }));
        }

        if (mask & A100_CLUSTER_MASK) != 0 {
            self.ai_thread_writes.lock().expect("lock ai").push(pid);
        }

        self.calls.lock().expect("lock calls").push((pid, mask));
        Ok(())
    }
}

// ============================================================================
// Isolation Domain Controller Trait
// ============================================================================

/// Core controller trait managing cgroup v2 domain placement and OS affinity fallback.
pub trait IsolationDomainController: Send + Sync + fmt::Debug {
    /// Places `k3-manager` into the `.../control` leaf domain with reserved control-plane cores.
    ///
    /// If `target_pid == 0`, targets current process (`std::process::id()`).
    ///
    /// # Errors
    /// Returns [`ManagerError`] if mask is invalid or if both cgroup and fallback fail.
    fn place_control_plane(
        &self,
        target_pid: u32,
        capacity: &ResolvedControlCapacity,
    ) -> Result<PlacementOutcome, ManagerError>;

    /// Places a spawned `k3-worker` process into delegated `.../engines/<model>` leaf domain.
    ///
    /// # Errors
    /// Returns [`ManagerError`] if mask is invalid or if both cgroup and fallback fail.
    fn place_worker(
        &self,
        worker_pid: u32,
        model_id: &str,
        compute_mask: u64,
    ) -> Result<PlacementOutcome, ManagerError>;
}

// ============================================================================
// CgroupV2DomainController Implementation
// ============================================================================

/// Cgroup v2 isolation domain controller with automatic OS affinity fallback.
#[derive(Debug, Clone)]
pub struct CgroupV2DomainController {
    base_path: PathBuf,
    fs: Arc<dyn CgroupFs>,
    affinity_backend: Arc<dyn OsAffinityBackend>,
    include_spare_in_control: bool,
}

impl CgroupV2DomainController {
    /// Constructs a domain controller rooted at standard systemd slice (`/sys/fs/cgroup/k3.slice`).
    #[must_use]
    pub fn default_slice() -> Self {
        Self {
            base_path: PathBuf::from("/sys/fs/cgroup/k3.slice"),
            fs: Arc::new(RealCgroupFs),
            affinity_backend: Arc::new(RealOsAffinityBackend),
            include_spare_in_control: true,
        }
    }

    /// Constructs a domain controller rooted at a custom cgroup base path.
    #[must_use]
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
            fs: Arc::new(RealCgroupFs),
            affinity_backend: Arc::new(RealOsAffinityBackend),
            include_spare_in_control: true,
        }
    }

    /// Sets custom cgroup filesystem backend.
    #[must_use]
    pub fn with_fs(mut self, fs: Arc<dyn CgroupFs>) -> Self {
        self.fs = fs;
        self
    }

    /// Sets custom OS affinity backend.
    #[must_use]
    pub fn with_affinity_backend(mut self, backend: Arc<dyn OsAffinityBackend>) -> Self {
        self.affinity_backend = backend;
        self
    }

    /// Sets whether spare recovery core(s) are included in the `.../control` cpuset.
    #[must_use]
    pub const fn with_include_spare(mut self, include_spare: bool) -> Self {
        self.include_spare_in_control = include_spare;
        self
    }

    /// Returns reference to cgroup base path.
    #[must_use]
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Returns whether spare recovery cores are included in control cpuset.
    #[must_use]
    pub const fn include_spare_in_control(&self) -> bool {
        self.include_spare_in_control
    }

    /// Places `k3-manager` into the `.../control` leaf domain with reserved control-plane cores.
    ///
    /// # Errors
    /// Returns [`ManagerError`] if mask is invalid or if both cgroup and fallback fail.
    pub fn place_control_plane(
        &self,
        target_pid: u32,
        capacity: &ResolvedControlCapacity,
    ) -> Result<PlacementOutcome, ManagerError> {
        IsolationDomainController::place_control_plane(self, target_pid, capacity)
    }

    /// Places a spawned `k3-worker` process into delegated `.../engines/<model>` leaf domain.
    ///
    /// # Errors
    /// Returns [`ManagerError`] if mask is invalid or if both cgroup and fallback fail.
    pub fn place_worker(
        &self,
        worker_pid: u32,
        model_id: &str,
        compute_mask: u64,
    ) -> Result<PlacementOutcome, ManagerError> {
        IsolationDomainController::place_worker(self, worker_pid, model_id, compute_mask)
    }

    fn try_cgroup_placement(&self, leaf_dir: &Path, cpuset_str: &str, pid: u32) -> io::Result<()> {
        self.fs.create_dir_all(leaf_dir)?;

        // Enable +cpuset on intermediate cgroup ancestors between base_path and leaf_dir
        // in strictly TOP-DOWN order per Linux cgroup v2 controller delegation constraints.
        let mut ancestors = Vec::new();
        if let Some(parent) = leaf_dir.parent() {
            let mut curr = parent;
            while curr.starts_with(&self.base_path) {
                ancestors.push(curr.to_path_buf());
                if curr == self.base_path {
                    break;
                }
                if let Some(p) = curr.parent() {
                    curr = p;
                } else {
                    break;
                }
            }
        }
        ancestors.reverse(); // Now strictly top-down: base_path down to leaf_dir.parent()

        for anc in ancestors {
            let subtree_file = anc.join("cgroup.subtree_control");
            let _ = self.fs.write(&subtree_file, b"+cpuset\n");
        }

        let cpuset_file = leaf_dir.join("cpuset.cpus");
        self.fs
            .write(&cpuset_file, format!("{cpuset_str}\n").as_bytes())?;

        let procs_file = leaf_dir.join("cgroup.procs");
        let content = format!("{pid}\n");
        match self.fs.write(&procs_file, content.as_bytes()) {
            Ok(()) => Ok(()),
            Err(e) => {
                // If cgroup.procs write fails (e.g. threaded domain), try cgroup.threads
                let threads_file = leaf_dir.join("cgroup.threads");
                self.fs
                    .write(&threads_file, content.as_bytes())
                    .map_err(|_| e)
            }
        }
    }
}

impl Default for CgroupV2DomainController {
    fn default() -> Self {
        Self::default_slice()
    }
}

impl IsolationDomainController for CgroupV2DomainController {
    fn place_control_plane(
        &self,
        target_pid: u32,
        capacity: &ResolvedControlCapacity,
    ) -> Result<PlacementOutcome, ManagerError> {
        if target_pid > (i32::MAX as u32) {
            return Err(ManagerError::WorkerSpawn(format!(
                "invalid target pid {target_pid}: must be 0 (current process) or positive 32-bit process ID (1..={})",
                i32::MAX
            )));
        }

        let control_mask = if self.include_spare_in_control && capacity.spare_recovery_mask != 0 {
            capacity.active_control_mask | capacity.spare_recovery_mask
        } else {
            capacity.active_control_mask
        };

        if control_mask == 0 {
            return Err(ManagerError::Affinity(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads: 0,
                reason: "control plane core mask cannot be zero".to_string(),
            }));
        }
        if (control_mask & !SOC_CLUSTER_MASK) != 0 {
            return Err(ManagerError::Affinity(K3Error::AffinityError {
                requested_mask: control_mask,
                requested_threads: control_mask.count_ones() as usize,
                reason: format!(
                    "control mask 0x{control_mask:04X} specifies invalid cores outside 16-core SoC mask 0x{SOC_CLUSTER_MASK:04X}"
                ),
            }));
        }

        let pid = if target_pid == 0 {
            std::process::id()
        } else {
            target_pid
        };

        let cpuset_str = format_cpuset_list(control_mask)?;
        let control_dir = self.base_path.join("control");

        match self.try_cgroup_placement(&control_dir, &cpuset_str, pid) {
            Ok(()) => Ok(PlacementOutcome {
                target_id: pid,
                mode: DomainPlacementMode::CgroupV2 {
                    cgroup_path: control_dir,
                    cpuset: cpuset_str,
                },
            }),
            Err(cgroup_err) => {
                self.affinity_backend.set_affinity(pid, control_mask)?;
                Ok(PlacementOutcome {
                    target_id: pid,
                    mode: DomainPlacementMode::OsAffinityFallback {
                        mask: control_mask,
                        reason: format!(
                            "cgroup v2 control placement failed on '{}' ({cgroup_err}); applied OS affinity fallback",
                            control_dir.display()
                        ),
                    },
                })
            }
        }
    }

    fn place_worker(
        &self,
        worker_pid: u32,
        model_id: &str,
        compute_mask: u64,
    ) -> Result<PlacementOutcome, ManagerError> {
        validate_model_id(model_id)?;
        if worker_pid == 0 || worker_pid > (i32::MAX as u32) {
            return Err(ManagerError::WorkerSpawn(format!(
                "invalid worker pid {worker_pid}: must be a positive 32-bit process ID (1..={})",
                i32::MAX
            )));
        }
        if compute_mask == 0 {
            return Err(ManagerError::Affinity(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads: 0,
                reason: "compute core mask cannot be zero".to_string(),
            }));
        }
        if (compute_mask & !SOC_CLUSTER_MASK) != 0 {
            return Err(ManagerError::Affinity(K3Error::AffinityError {
                requested_mask: compute_mask,
                requested_threads: compute_mask.count_ones() as usize,
                reason: format!(
                    "compute mask 0x{compute_mask:04X} specifies invalid cores outside 16-core SoC mask 0x{SOC_CLUSTER_MASK:04X}"
                ),
            }));
        }

        let cpuset_str = format_cpuset_list(compute_mask)?;
        let engine_dir = self.base_path.join("engines").join(model_id);

        match self.try_cgroup_placement(&engine_dir, &cpuset_str, worker_pid) {
            Ok(()) => Ok(PlacementOutcome {
                target_id: worker_pid,
                mode: DomainPlacementMode::CgroupV2 {
                    cgroup_path: engine_dir,
                    cpuset: cpuset_str,
                },
            }),
            Err(cgroup_err) => {
                self.affinity_backend
                    .set_affinity(worker_pid, compute_mask)?;
                Ok(PlacementOutcome {
                    target_id: worker_pid,
                    mode: DomainPlacementMode::OsAffinityFallback {
                        mask: compute_mask,
                        reason: format!(
                            "cgroup v2 engine placement failed on '{}' ({cgroup_err}); applied OS affinity fallback",
                            engine_dir.display()
                        ),
                    },
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k3_core::affinity::ControlPlaneCapacityConfig;

    #[test]
    fn test_format_cpuset_list_canonical_ranges() {
        assert_eq!(format_cpuset_list(0x003F).unwrap(), "0-5");
        assert_eq!(format_cpuset_list(0x00C0).unwrap(), "6-7");
        assert_eq!(format_cpuset_list(0x00E0).unwrap(), "5-7");
        assert_eq!(format_cpuset_list(0x0020).unwrap(), "5");
        assert_eq!(format_cpuset_list(0x0080).unwrap(), "7");
        assert_eq!(format_cpuset_list(0xFF00).unwrap(), "8-15");
        assert_eq!(format_cpuset_list(0xFFFF).unwrap(), "0-15");
        assert_eq!(format_cpuset_list(0x00F3).unwrap(), "0-1,4-7");
        assert_eq!(format_cpuset_list(0x00A5).unwrap(), "0,2,5,7");
    }

    #[test]
    fn test_format_cpuset_list_rejects_zero_and_out_of_bounds() {
        assert!(format_cpuset_list(0).is_err());
        assert!(format_cpuset_list(0x1_0000).is_err());
        assert!(format_cpuset_list(0x8000_0000_0000_0000).is_err());
    }

    #[test]
    fn test_parse_cpuset_list_valid_syntax() {
        assert_eq!(parse_cpuset_list("0-5").unwrap(), 0x003F);
        assert_eq!(parse_cpuset_list("6-7").unwrap(), 0x00C0);
        assert_eq!(parse_cpuset_list("5").unwrap(), 0x0020);
        assert_eq!(parse_cpuset_list("8-15").unwrap(), 0xFF00);
        assert_eq!(parse_cpuset_list("0-1,4-7").unwrap(), 0x00F3);
        assert_eq!(parse_cpuset_list(" 0 , 2 , 5 , 7 ").unwrap(), 0x00A5);
    }

    #[test]
    fn test_parse_cpuset_list_rejects_invalid_syntax() {
        assert!(parse_cpuset_list("").is_err());
        assert!(parse_cpuset_list("   ").is_err());
        assert!(parse_cpuset_list("0,,1").is_err());
        assert!(parse_cpuset_list("16").is_err());
        assert!(parse_cpuset_list("0-16").is_err());
        assert!(parse_cpuset_list("5-3").is_err());
        assert!(parse_cpuset_list("abc").is_err());
    }

    #[test]
    fn test_mock_cgroup_v2_happy_path_placement() {
        let fs = Arc::new(MockCgroupFs::new());
        let aff = Arc::new(MockOsAffinityBackend::new());
        let controller = CgroupV2DomainController::new("/mock/cgroup")
            .with_fs(fs.clone())
            .with_affinity_backend(aff.clone());

        let cap = ControlPlaneCapacityConfig::default()
            .resolve_default()
            .unwrap();

        // Control plane placement
        let control_res = controller.place_control_plane(1001, &cap).unwrap();
        assert_eq!(control_res.target_id, 1001);
        assert!(control_res.mode.is_cgroup_v2());
        assert_eq!(
            fs.written_file("/mock/cgroup/control/cpuset.cpus")
                .as_deref(),
            Some("5-7\n")
        );
        assert_eq!(
            fs.written_file("/mock/cgroup/control/cgroup.procs")
                .as_deref(),
            Some("1001\n")
        );
        assert_eq!(aff.calls().len(), 0);

        // Worker placement
        let worker_res = controller.place_worker(2002, "qwen2.5-7b", 0x003F).unwrap();
        assert_eq!(worker_res.target_id, 2002);
        assert!(worker_res.mode.is_cgroup_v2());
        assert_eq!(
            fs.written_file("/mock/cgroup/engines/qwen2.5-7b/cpuset.cpus")
                .as_deref(),
            Some("0-5\n")
        );
        assert_eq!(
            fs.written_file("/mock/cgroup/engines/qwen2.5-7b/cgroup.procs")
                .as_deref(),
            Some("2002\n")
        );
        assert_eq!(aff.calls().len(), 0);
    }

    #[test]
    fn test_mock_cgroup_v2_fallback_to_os_affinity_on_eacces_and_enoent() {
        let fs = Arc::new(MockCgroupFs::new());
        let aff = Arc::new(MockOsAffinityBackend::new());
        let controller = CgroupV2DomainController::new("/mock/cgroup")
            .with_fs(fs.clone())
            .with_affinity_backend(aff.clone());

        // Inject EACCES on worker engine leaf
        fs.inject_error(
            "/mock/cgroup/engines/qwen2.5-7b",
            io::ErrorKind::PermissionDenied,
        );

        let res = controller.place_worker(3003, "qwen2.5-7b", 0x003F).unwrap();
        assert_eq!(res.target_id, 3003);
        assert!(res.mode.is_os_affinity_fallback());
        assert!(aff.has_called(3003, 0x003F));

        // Inject ENOENT on control dir
        fs.inject_error("/mock/cgroup/control", io::ErrorKind::NotFound);
        let cap = ControlPlaneCapacityConfig::default()
            .resolve_default()
            .unwrap();

        let control_res = controller.place_control_plane(4004, &cap).unwrap();
        assert_eq!(control_res.target_id, 4004);
        assert!(control_res.mode.is_os_affinity_fallback());
        assert!(aff.has_called(4004, 0x00E0));
    }

    #[test]
    fn test_mock_cgroup_v2_a100_ai_thread_registration() {
        let fs = Arc::new(MockCgroupFs::new());
        let aff = Arc::new(MockOsAffinityBackend::new());
        let controller = CgroupV2DomainController::new("/mock/cgroup")
            .with_fs(fs.clone())
            .with_affinity_backend(aff.clone());

        // Fail cgroup to force OS affinity fallback
        fs.inject_error(
            "/mock/cgroup/engines/deepseek",
            io::ErrorKind::PermissionDenied,
        );

        let res = controller.place_worker(5005, "deepseek", 0xFF00).unwrap();
        assert!(res.mode.is_os_affinity_fallback());
        assert!(aff.has_called(5005, 0xFF00));
        assert_eq!(aff.ai_thread_writes(), vec![5005]);
    }
}
