//! Unix Domain Socket client for `k3-kv` context memory holder daemon.

use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use k3_core::ipc::protocol::{
    recv_msg_frame, send_msg_frame, AllocateArenaRequest, AllocateArenaResponse,
    DeallocateArenaRequest, DeallocateArenaResponse, GetArenaFdRequest, GetArenaFdResponse,
    IpcMessage, PingRequest, PingResponse,
};
use k3_core::ipc::shm::SharedMemoryArena;

use crate::error::ManagerError;

/// Client connection to the `k3-kv` memory holder daemon over Unix Domain Sockets.
#[derive(Debug)]
pub struct KvClient {
    stream: UnixStream,
    socket_path: Option<PathBuf>,
}

impl KvClient {
    /// Connects to a running `k3-kv` daemon at `socket_path`.
    ///
    /// # Errors
    /// Returns [`ManagerError::Kv`] if connecting to the socket fails.
    pub fn connect(socket_path: impl AsRef<Path>) -> Result<Self, ManagerError> {
        let path = socket_path.as_ref();
        let stream = UnixStream::connect(path).map_err(|e| {
            ManagerError::Kv(format!(
                "failed to connect to k3-kv socket '{}': {e}",
                path.display()
            ))
        })?;
        Ok(Self {
            stream,
            socket_path: Some(path.to_path_buf()),
        })
    }

    /// Wraps an existing connected [`UnixStream`].
    #[must_use]
    pub fn from_stream(stream: UnixStream) -> Self {
        Self {
            stream,
            socket_path: None,
        }
    }

    /// Returns the configured socket path if connected via path.
    #[must_use]
    pub fn socket_path(&self) -> Option<&Path> {
        self.socket_path.as_deref()
    }

    /// Sends a liveness ping (`PingRequest` / Type ID 107) and returns the echoed nanosecond timestamp.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on communication failure or protocol violation.
    pub fn ping(&mut self) -> Result<u64, ManagerError> {
        let now_ns = u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| ManagerError::Kv(e.to_string()))?
                .as_nanos(),
        )
        .unwrap_or(0);

        let req = IpcMessage::PingRequest(PingRequest {
            timestamp_ns: now_ns,
        });

        send_msg_frame(&self.stream, &req, None).map_err(|e| {
            ManagerError::Kv(format!("failed to send PingRequest to k3-kv: {e}"))
        })?;

        let (reply, _) = recv_msg_frame(&self.stream).map_err(|e| {
            ManagerError::Kv(format!("failed to receive PingResponse from k3-kv: {e}"))
        })?;

        match reply {
            IpcMessage::PingResponse(PingResponse { timestamp_ns }) => Ok(timestamp_ns),
            other => Err(ManagerError::Protocol(format!(
                "expected PingResponse from k3-kv, got {other:?}"
            ))),
        }
    }

    /// Sends a heartbeat request (alias for [`Self::ping`]).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on communication failure.
    pub fn heartbeat(&mut self) -> Result<u64, ManagerError> {
        self.ping()
    }

    /// Requests allocation of a shared-memory arena for a model (`AllocateArenaRequest` / Type ID 101).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on communication failure or non-success status.
    pub fn allocate_arena(
        &mut self,
        model_id: &str,
        arena_bytes: u64,
        n_ctx: u32,
        max_blocks: u32,
    ) -> Result<AllocateArenaResponse, ManagerError> {
        let req = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
            model_id: model_id.to_owned(),
            arena_bytes,
            n_ctx,
            max_blocks,
        });

        send_msg_frame(&self.stream, &req, None).map_err(|e| {
            ManagerError::Kv(format!("failed to send AllocateArenaRequest to k3-kv: {e}"))
        })?;

        let (reply, _) = recv_msg_frame(&self.stream).map_err(|e| {
            ManagerError::Kv(format!("failed to receive AllocateArenaResponse from k3-kv: {e}"))
        })?;

        match reply {
            IpcMessage::AllocateArenaResponse(resp) => {
                if resp.status.is_success() {
                    Ok(resp)
                } else {
                    Err(ManagerError::Status {
                        status: resp.status,
                        message: resp
                            .error_message
                            .unwrap_or_else(|| format!("AllocateArena failed with status {:?}", resp.status)),
                    })
                }
            }
            other => Err(ManagerError::Protocol(format!(
                "expected AllocateArenaResponse from k3-kv, got {other:?}"
            ))),
        }
    }

    /// Requests the file descriptor for an arena passed via `SCM_RIGHTS` (`GetArenaFdRequest` / Type ID 103).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on communication failure, non-success status, or missing FD.
    pub fn get_arena_fd(
        &mut self,
        arena_id: u64,
    ) -> Result<(GetArenaFdResponse, OwnedFd), ManagerError> {
        let req = IpcMessage::GetArenaFdRequest(GetArenaFdRequest { arena_id });

        send_msg_frame(&self.stream, &req, None).map_err(|e| {
            ManagerError::Kv(format!("failed to send GetArenaFdRequest to k3-kv: {e}"))
        })?;

        let (reply, opt_fd) = recv_msg_frame(&self.stream).map_err(|e| {
            ManagerError::Kv(format!("failed to receive GetArenaFdResponse from k3-kv: {e}"))
        })?;

        match reply {
            IpcMessage::GetArenaFdResponse(resp) => {
                if !resp.status.is_success() {
                    return Err(ManagerError::Status {
                        status: resp.status,
                        message: resp
                            .error_message
                            .unwrap_or_else(|| format!("GetArenaFd failed with status {:?}", resp.status)),
                    });
                }
                let fd = opt_fd.ok_or_else(|| {
                    ManagerError::Protocol(
                        "k3-kv returned GetArenaFdResponse with success but no SCM_RIGHTS FD was attached"
                            .to_owned(),
                    )
                })?;
                Ok((resp, fd))
            }
            other => Err(ManagerError::Protocol(format!(
                "expected GetArenaFdResponse from k3-kv, got {other:?}"
            ))),
        }
    }

    /// Fetches the file descriptor for an arena and maps it into process virtual memory as a [`SharedMemoryArena`].
    ///
    /// # Errors
    /// Returns [`ManagerError`] on failure.
    pub fn attach_arena(
        &mut self,
        arena_id: u64,
        arena_bytes: usize,
        name: &str,
    ) -> Result<SharedMemoryArena, ManagerError> {
        let (_resp, fd) = self.get_arena_fd(arena_id)?;
        let arena = SharedMemoryArena::from_existing_fd(fd, arena_bytes, name, false)?;
        Ok(arena)
    }

    /// Allocates an arena and attaches/maps it into local virtual memory in a single convenience call.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on failure.
    pub fn allocate_and_attach(
        &mut self,
        model_id: &str,
        arena_bytes: u64,
        n_ctx: u32,
        max_blocks: u32,
    ) -> Result<(AllocateArenaResponse, SharedMemoryArena), ManagerError> {
        let alloc_resp = self.allocate_arena(model_id, arena_bytes, n_ctx, max_blocks)?;
        let bytes_len = usize::try_from(alloc_resp.arena_bytes).map_err(|_| {
            let _ = self.deallocate_arena(alloc_resp.arena_id);
            ManagerError::Kv("allocated arena_bytes exceeds usize".to_owned())
        })?;
        let arena = match self.attach_arena(alloc_resp.arena_id, bytes_len, model_id) {
            Ok(a) => a,
            Err(e) => {
                let _ = self.deallocate_arena(alloc_resp.arena_id);
                return Err(e);
            }
        };
        Ok((alloc_resp, arena))
    }

    /// Requests deallocation of an arena in `k3-kv` (`DeallocateArenaRequest` / Type ID 105).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on communication failure or non-success status.
    pub fn deallocate_arena(
        &mut self,
        arena_id: u64,
    ) -> Result<DeallocateArenaResponse, ManagerError> {
        let req = IpcMessage::DeallocateArenaRequest(DeallocateArenaRequest { arena_id });

        send_msg_frame(&self.stream, &req, None).map_err(|e| {
            ManagerError::Kv(format!("failed to send DeallocateArenaRequest to k3-kv: {e}"))
        })?;

        let (reply, _) = recv_msg_frame(&self.stream).map_err(|e| {
            ManagerError::Kv(format!("failed to receive DeallocateArenaResponse from k3-kv: {e}"))
        })?;

        match reply {
            IpcMessage::DeallocateArenaResponse(resp) => {
                if resp.status.is_success() {
                    Ok(resp)
                } else {
                    Err(ManagerError::Status {
                        status: resp.status,
                        message: resp
                            .error_message
                            .unwrap_or_else(|| format!("DeallocateArena failed with status {:?}", resp.status)),
                    })
                }
            }
            other => Err(ManagerError::Protocol(format!(
                "expected DeallocateArenaResponse from k3-kv, got {other:?}"
            ))),
        }
    }
}
