//! Anonymous Shared Memory (`memfd_create`), Sealing, and `SCM_RIGHTS` IPC (US-19, ADR-037, ADR-041).
//!
//! Provides an anonymous memory arena for receiving Arrow Flight tensor batches,
//! locking pages in RAM (`mlock()`), applying write seals (`F_SEAL_WRITE`),
//! and transferring the sealed file descriptor to compute workers via `SCM_RIGHTS`.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::manual_div_ceil,
    clippy::match_same_arms
)]

use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use k3_core::K3Error;
use k3_core::ipc::protocol::{recv_frame, send_frame};

static MEMFD_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Bounded memlock ceiling per K3 Workspace Agent Guidelines §5 (28 GiB on 32 GiB hardware).
pub const BOUNDED_MEMLOCK_CEILING_BYTES: u64 = 29_360_128 * 1024; // ~28 GiB

/// Standard IPC type ID for Flight tensor arena FD transfer.
pub const TYPE_ID_FLIGHT_ARENA: u16 = 205;

/// Error type for Flight tensor arena and IPC operations.
#[derive(Debug)]
pub enum FlightError {
    /// Failure allocating or resizing shared memory arena.
    AllocationFailed(String),
    /// Lockable memory ceiling breached or `mlock` rejected.
    InsufficientLockableMemory(K3Error),
    /// Attempted write on a sealed memfd.
    MemfdSealed,
    /// Arrow record batch or layout validation failure.
    InvalidBatch(String),
    /// Transport or streaming failure.
    TransportError(String),
    /// Socket or `SCM_RIGHTS` transfer failure.
    UdsError(String),
    /// Standard I/O error.
    Io(io::Error),
    /// Underlying K3 platform error.
    K3(K3Error),
}

impl std::fmt::Display for FlightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllocationFailed(msg) => write!(f, "[flight allocation error] {msg}"),
            Self::InsufficientLockableMemory(e) => {
                write!(f, "[flight memory lock error] {e}")
            }
            Self::MemfdSealed => {
                write!(f, "[flight error] EPERM: cannot write to F_SEAL_WRITE sealed memfd")
            }
            Self::InvalidBatch(msg) => write!(f, "[flight invalid batch] {msg}"),
            Self::TransportError(msg) => write!(f, "[flight transport error] {msg}"),
            Self::UdsError(msg) => write!(f, "[flight uds error] {msg}"),
            Self::Io(e) => write!(f, "[flight io error] {e}"),
            Self::K3(e) => write!(f, "[flight k3 error] {e}"),
        }
    }
}

impl std::error::Error for FlightError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for FlightError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<K3Error> for FlightError {
    fn from(err: K3Error) -> Self {
        Self::K3(err)
    }
}

impl From<FlightError> for K3Error {
    fn from(err: FlightError) -> Self {
        match err {
            FlightError::K3(k) | FlightError::InsufficientLockableMemory(k) => k,
            FlightError::MemfdSealed => K3Error::CapabilityMismatch {
                architecture: "memfd_sealing".to_owned(),
                reason: "EPERM: write prohibited on sealed memory".to_owned(),
            },
            other => K3Error::CapabilityMismatch {
                architecture: "flight".to_owned(),
                reason: other.to_string(),
            },
        }
    }
}

/// Anonymous shared memory arena backed by Linux `memfd_create` or portable POSIX fallback.
#[derive(Debug)]
pub struct MemfdArena {
    fd: OwnedFd,
    ptr: NonNull<u8>,
    len: usize,
    name: String,
    sealed: bool,
    locked: bool,
}

// SAFETY: MemfdArena exclusively manages its mapped virtual memory and OwnedFd.
unsafe impl Send for MemfdArena {}
// SAFETY: Immutable operations across threads are safe as buffer is invariant once populated.
unsafe impl Sync for MemfdArena {}

impl MemfdArena {
    /// Creates a new anonymous shared memory segment with `ALLOW_SEALING`.
    ///
    /// # Errors
    /// Returns [`FlightError::AllocationFailed`] on OS allocation or mapping error.
    pub fn new(name: &str, capacity_bytes: usize) -> Result<Self, FlightError> {
        // SAFETY: sysconf reads standard system page size.
        let page_size = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(4096);
        let aligned_len = ((capacity_bytes.max(1) + page_size - 1) / page_size) * page_size;

        #[cfg(target_os = "linux")]
        {
            let cname = std::ffi::CString::new(name).map_err(|e| {
                FlightError::AllocationFailed(format!("invalid memfd name: {e}"))
            })?;

            // SAFETY: libc::memfd_create creates an anonymous in-memory file with sealing support.
            let raw_fd = unsafe {
                libc::memfd_create(cname.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
            };

            if raw_fd >= 0 {
                // SAFETY: ftruncate sets the backing capacity.
                let trunc_ret = unsafe { libc::ftruncate(raw_fd, aligned_len as libc::off_t) };
                if trunc_ret == 0 {
                    // SAFETY: mmap maps the anonymous file descriptor into process memory.
                    let ptr = unsafe {
                        libc::mmap(
                            std::ptr::null_mut(),
                            aligned_len,
                            libc::PROT_READ | libc::PROT_WRITE,
                            libc::MAP_SHARED,
                            raw_fd,
                            0,
                        )
                    };

                    if ptr != libc::MAP_FAILED {
                        // SAFETY: raw_fd is valid and ownership is transferred to OwnedFd.
                        let owned_fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
                        let non_null = NonNull::new(ptr.cast::<u8>()).ok_or_else(|| {
                            FlightError::AllocationFailed("null pointer from mmap".to_owned())
                        })?;

                        // SAFETY: zeroing freshly allocated mmap region.
                        unsafe {
                            std::ptr::write_bytes(non_null.as_ptr(), 0, aligned_len);
                        }

                        return Ok(Self {
                            fd: owned_fd,
                            ptr: non_null,
                            len: aligned_len,
                            name: name.to_owned(),
                            sealed: false,
                            locked: false,
                        });
                    }
                }
                // SAFETY: Close raw_fd on initialization error path.
                unsafe { libc::close(raw_fd) };
            }
        }

        // Portable fallback: macOS and systems without memfd_create
        let seq = MEMFD_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let temp_path = std::env::temp_dir().join(format!("k3_flight_{name}_{pid}_{seq}_{nanos}.part"));
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)
            .map_err(|e| FlightError::AllocationFailed(format!("failed to open temp shm file: {e}")))?;

        file.set_len(aligned_len as u64)
            .map_err(|e| FlightError::AllocationFailed(format!("failed to truncate shm file: {e}")))?;

        let raw_fd = file.as_raw_fd();
        // SAFETY: mmap maps the file into process virtual memory with MAP_SHARED.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                aligned_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                raw_fd,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            let _ = std::fs::remove_file(&temp_path);
            let err = io::Error::last_os_error();
            return Err(FlightError::AllocationFailed(format!("mmap failed: {err}")));
        }

        // Unlink immediately from disk so memory is held solely via the open file descriptor
        let _ = std::fs::remove_file(&temp_path);

        let non_null = NonNull::new(ptr.cast::<u8>()).ok_or_else(|| {
            FlightError::AllocationFailed("null pointer from mmap".to_owned())
        })?;

        // SAFETY: zero-initializing anonymous memory mapping.
        unsafe {
            std::ptr::write_bytes(non_null.as_ptr(), 0, aligned_len);
        }

        Ok(Self {
            fd: file.into(),
            ptr: non_null,
            len: aligned_len,
            name: name.to_owned(),
            sealed: false,
            locked: false,
        })
    }

    /// Returns the mapped byte capacity of the arena.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if length is 0.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the logical name of this arena.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the raw file descriptor.
    #[must_use]
    pub fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Consumes the arena, disarming the `Drop` handler so that the virtual memory mapping
    /// remains active and the backing file descriptor remains open and valid.
    ///
    /// Returns the raw file descriptor. The caller is responsible for eventually closing
    /// the file descriptor.
    #[must_use]
    pub fn into_raw_fd(self) -> RawFd {
        let raw_fd = self.as_raw_fd();
        std::mem::forget(self);
        raw_fd
    }

    /// Returns a shared slice over the entire arena.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: self.ptr points to self.len valid mapped bytes while arena lives.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a mutable slice over the entire arena if not sealed.
    ///
    /// # Errors
    /// Returns [`FlightError::MemfdSealed`] if sealed against writes.
    pub fn as_mut_slice(&mut self) -> Result<&mut [u8], FlightError> {
        if self.sealed {
            return Err(FlightError::MemfdSealed);
        }
        // SAFETY: self.ptr points to self.len mapped bytes, &mut self guarantees exclusive access.
        unsafe { Ok(std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)) }
    }

    /// Writes a slice into the arena at `offset`, checking bounds and write seals.
    ///
    /// # Errors
    /// Returns [`FlightError::MemfdSealed`] if sealed, or [`FlightError::AllocationFailed`] on bounds breach.
    pub fn write_at(&mut self, offset: usize, data: &[u8]) -> Result<(), FlightError> {
        if self.sealed {
            return Err(FlightError::MemfdSealed);
        }

        let end = offset.checked_add(data.len()).ok_or_else(|| {
            FlightError::AllocationFailed("offset + data.len() overflowed usize".to_owned())
        })?;

        if end > self.len {
            return Err(FlightError::AllocationFailed(format!(
                "write range [{offset}..{end}] exceeds arena capacity {}",
                self.len
            )));
        }

        // SAFETY: Bounds verified above, source and target are non-overlapping valid memory regions.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr().add(offset), data.len());
        }

        Ok(())
    }

    /// Locks the arena in physical RAM via `libc::mlock`, checking against memlock ceiling.
    ///
    /// # Errors
    /// Returns [`FlightError::InsufficientLockableMemory`] if bounds exceeded or lock fails.
    pub fn mlock(&mut self, ceiling_bytes: Option<u64>) -> Result<(), FlightError> {
        if self.locked {
            return Ok(());
        }

        let ceiling = ceiling_bytes.unwrap_or(BOUNDED_MEMLOCK_CEILING_BYTES);
        let req_bytes = self.len as u64;

        if req_bytes > ceiling {
            return Err(FlightError::InsufficientLockableMemory(
                K3Error::InsufficientLockableMemory {
                    required_bytes: req_bytes,
                    weight_bytes: req_bytes,
                    kv_bytes: 0,
                    available_bytes: 0,
                    rlimit_memlock_bytes: ceiling,
                },
            ));
        }

        // SAFETY: self.ptr points to self.len valid mapped bytes.
        let ret = unsafe { libc::mlock(self.ptr.as_ptr().cast(), self.len) };
        if ret != 0 {
            return Err(FlightError::InsufficientLockableMemory(
                K3Error::InsufficientLockableMemory {
                    required_bytes: req_bytes,
                    weight_bytes: req_bytes,
                    kv_bytes: 0,
                    available_bytes: 0,
                    rlimit_memlock_bytes: ceiling,
                },
            ));
        }

        self.locked = true;
        Ok(())
    }

    /// Applies `F_SEAL_WRITE` seal to the memfd, preventing any subsequent modifications.
    ///
    /// # Errors
    /// Returns [`FlightError::AllocationFailed`] if applying seals fails.
    pub fn seal_write(&mut self) -> Result<(), FlightError> {
        if self.sealed {
            return Ok(());
        }

        #[cfg(target_os = "linux")]
        {
            // SAFETY: fcntl F_ADD_SEALS applies seals to memfd descriptor.
            let ret = unsafe {
                libc::fcntl(
                    self.as_raw_fd(),
                    libc::F_ADD_SEALS,
                    libc::F_SEAL_WRITE | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW,
                )
            };
            if ret != 0 {
                let err = io::Error::last_os_error();
                return Err(FlightError::AllocationFailed(format!(
                    "fcntl F_ADD_SEALS failed: {err}"
                )));
            }
        }

        // Hardware memory protection: make mapped virtual memory read-only on all platforms
        // SAFETY: mprotect marks memory read-only.
        unsafe {
            libc::mprotect(self.ptr.as_ptr().cast(), self.len, libc::PROT_READ);
        }

        self.sealed = true;
        Ok(())
    }

    /// Returns `true` if `seal_write()` has been applied.
    #[must_use]
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// Duplicates the backing file descriptor using `libc::dup`.
    ///
    /// # Errors
    /// Returns [`FlightError::AllocationFailed`] if duplication fails.
    pub fn duplicate_fd(&self) -> Result<OwnedFd, FlightError> {
        // SAFETY: dup duplicates open file descriptor safely.
        let raw = unsafe { libc::dup(self.as_raw_fd()) };
        if raw < 0 {
            return Err(FlightError::AllocationFailed(format!(
                "dup failed: {}",
                io::Error::last_os_error()
            )));
        }
        // SAFETY: raw is verified >= 0 and owned by OwnedFd.
        Ok(unsafe { OwnedFd::from_raw_fd(raw) })
    }

    /// Transfers this arena's file descriptor over a Unix Domain Socket via `SCM_RIGHTS`.
    ///
    /// # Errors
    /// Returns [`FlightError::UdsError`] on socket send failure.
    pub fn transfer_via_uds(&self, stream: &UnixStream, payload: &[u8]) -> Result<(), FlightError> {
        send_frame(
            stream.as_raw_fd(),
            TYPE_ID_FLIGHT_ARENA,
            payload,
            Some(self.as_raw_fd()),
        )
        .map_err(|e| FlightError::UdsError(format!("SCM_RIGHTS transfer failed: {e}")))
    }

    /// Receives a transferred file descriptor from a Unix Domain Socket via `SCM_RIGHTS`.
    ///
    /// # Errors
    /// Returns [`FlightError::UdsError`] if receiving fails or no FD was passed.
    pub fn receive_from_uds(stream: &UnixStream) -> Result<(Vec<u8>, OwnedFd), FlightError> {
        let (_header, payload, opt_fd) = recv_frame(stream.as_raw_fd())
            .map_err(|e| FlightError::UdsError(format!("SCM_RIGHTS recv failed: {e}")))?;

        let owned_fd = opt_fd.ok_or_else(|| {
            FlightError::UdsError("no SCM_RIGHTS file descriptor attached to frame".to_owned())
        })?;

        Ok((payload, owned_fd))
    }
}

impl Drop for MemfdArena {
    fn drop(&mut self) {
        if self.locked {
            // SAFETY: munlock releases memory pin on drop.
            unsafe {
                libc::munlock(self.ptr.as_ptr().cast(), self.len);
            }
        }
        // SAFETY: munmap cleans up address range on drop.
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast(), self.len);
        }
    }
}
