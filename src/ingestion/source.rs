//! Secret Resolution and Credential Isolation (Feature 30, ADR-041, AGENTS.md §3).
//!
//! Enforces fail-closed resolution for secrets specified either as literal tokens
//! or file paths (e.g., mode 0600 `/etc/k3/credentials/<source>.env`).
//! Sensitive credential memory is zeroized via `.wipe()` using authentic volatile
//! memory writes (`std::ptr::write_volatile`) upon completion.

use std::path::Path;
use std::sync::atomic::{Ordering, compiler_fence};

use crate::ingestion::adapter::{IngestionError, SourceCredentials};

/// Volatiles zeroizes a byte slice using `std::ptr::write_volatile` and a
/// sequential consistency compiler fence to defeat LLVM Dead Store Elimination (DSE).
#[inline]
pub fn volatile_zeroize_bytes(buf: &mut [u8]) {
    let len = buf.len();
    let ptr = buf.as_mut_ptr();
    for i in 0..len {
        // SAFETY: `ptr` points to `len` contiguous initialized bytes of `buf`.
        // `i < len` ensures `ptr.add(i)` is within allocated buffer bounds and properly aligned.
        unsafe {
            std::ptr::write_volatile(ptr.add(i), 0u8);
        }
    }
    compiler_fence(Ordering::SeqCst);
}

/// Volatiles zeroizes the heap allocation of a `String` up to its allocated capacity
/// using `std::ptr::write_volatile`, then clears its length.
///
/// Defeats compiler optimization (LLVM Dead Store Elimination) when discarding sensitive
/// tokens or passwords before deallocation.
#[inline]
pub fn volatile_zeroize_string(s: &mut String) {
    // SAFETY: We safely access the underlying vector to overwrite all allocated bytes.
    // 1. `as_mut_vec()` provides mutable access to the backing `Vec<u8>`.
    // 2. `as_mut_ptr()` yields a pointer to the allocated allocation.
    // 3. For any non-empty allocation (`cap > 0`), the pointer is valid for writes
    //    up to `cap` bytes, and `u8` requires 1-byte alignment (always aligned).
    // 4. Overwriting bytes with `0u8` (ASCII NUL) leaves valid UTF-8 characters.
    // 5. Volatile writes cannot be reordered or eliminated by compiler DSE optimizations.
    unsafe {
        let vec = s.as_mut_vec();
        let ptr = vec.as_mut_ptr();
        let cap = vec.capacity();
        if cap > 0 && !ptr.is_null() {
            for i in 0..cap {
                std::ptr::write_volatile(ptr.add(i), 0u8);
            }
        }
    }
    compiler_fence(Ordering::SeqCst);
    s.clear();
}

/// Resolves a secret token from either a file path or a literal value.
///
/// # Invariants (AGENTS.md §3)
/// - If `val` appears to be a file path (starts with `/`, `./`, `../`, equals `.` or `..`,
///   or contains path separators `/` or `\\`), it is treated strictly as a file path.
/// - If `val` points to an existing directory (even without path separators), it is treated
///   as an unreadable credential path and **FAILS CLOSED** returning `None`.
/// - If the file cannot be read, does not exist, is a directory, or is empty,
///   this function **FAILS CLOSED** and returns `None`.
/// - **NEVER** falls back to using the file path string itself as the secret token.
#[must_use]
pub fn resolve_secret_value(val: &str) -> Option<String> {
    let trimmed = val.trim();
    if trimmed.is_empty() {
        return None;
    }

    let is_path = trimmed == "."
        || trimmed == ".."
        || trimmed.starts_with('/')
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
        || trimmed.contains('/')
        || trimmed.contains('\\');

    let path = Path::new(trimmed);
    if path.is_dir() {
        // Any path that resolves to an existing directory must FAIL CLOSED!
        return None;
    }

    if path.is_file() {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let secret = content.trim().to_string();
                if secret.is_empty() {
                    None
                } else {
                    Some(secret)
                }
            }
            Err(_) => None, // FAIL CLOSED!
        }
    } else if is_path {
        // Path does not exist, is a directory, or is not a readable file -> FAIL CLOSED!
        None
    } else {
        // Literal non-path string
        Some(trimmed.to_string())
    }
}

impl SourceCredentials {
    /// Constructs `SourceCredentials` from a raw secret string or path, failing closed.
    ///
    /// # Invariants (AGENTS.md §3)
    /// If `val` appears to be a path, attempts to read and trim the file. If reading fails
    /// (file does not exist, is empty, or permission denied), returns
    /// [`IngestionError::CredentialResolutionFailed`]. It never returns the path as a token.
    ///
    /// # Errors
    /// Returns [`IngestionError::CredentialResolutionFailed`] if path resolution fails.
    pub fn from_secret_string(val: &str) -> Result<Self, IngestionError> {
        let trimmed = val.trim();
        if trimmed.is_empty() {
            return Ok(Self::Anonymous);
        }

        match resolve_secret_value(trimmed) {
            Some(token) => Ok(Self::Token(token)),
            None => Err(IngestionError::CredentialResolutionFailed(format!(
                "failed to resolve secret from '{trimmed}': unreadable, empty, or missing file; failing closed"
            ))),
        }
    }

    /// Constructs `SourceCredentials` from an environment variable, failing closed.
    ///
    /// # Errors
    /// Returns [`IngestionError::CredentialResolutionFailed`] if the variable is set but resolution fails.
    pub fn from_env(var_name: &str) -> Result<Self, IngestionError> {
        match std::env::var(var_name) {
            Ok(val) => Self::from_secret_string(&val),
            Err(_) => Ok(Self::Anonymous),
        }
    }

    /// Securely zeroizes in-memory token and secret buffers using `std::ptr::write_volatile`
    /// to defeat LLVM Dead Store Elimination (DSE), and resets to [`SourceCredentials::Anonymous`].
    pub fn wipe(&mut self) {
        match self {
            Self::Token(s) | Self::BearerToken(s) => {
                volatile_zeroize_string(s);
            }
            Self::BasicAuth {
                username,
                password_or_token,
            } => {
                volatile_zeroize_string(username);
                volatile_zeroize_string(password_or_token);
            }
            _ => {}
        }
        *self = Self::Anonymous;
    }
}
