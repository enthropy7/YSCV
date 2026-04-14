//! Common frame-payload types shared between the 3-stage and 5-stage
//! pipelines. Exposes the zero-copy DMA-BUF path so callers can work
//! with camera-exported file descriptors without hand-rolling a
//! wrapper for every pipeline variant.
//!
//! # Ownership rules
//!
//! [`FramePayload::DmaBuf`] stores an **unowned** file descriptor:
//! the camera layer (`V4l2Camera::export_dmabuf`) allocated the backing
//! buffer and holds the authoritative `OwnedFd`. The pipeline borrows
//! the fd for the frame's duration; it must NOT `close()` it. The
//! camera layer is responsible for releasing the fd when the V4L2
//! buffer is re-queued.
//!
//! # Why not `Arc<OwnedFd>`?
//!
//! Real-time pipelines can't tolerate `Arc` refcount traffic on the
//! hot path. The "camera owns the fd, pipeline borrows" convention is
//! enforceable at review time and survives `drop` safely (a dropped
//! `FramePayload::DmaBuf` simply forgets the integer; the camera's
//! buffer is still alive).

use std::fmt;

/// A per-frame byte payload. Either owned (copy-based ingress) or a
/// borrowed DMA-BUF fd (zero-copy ingress, typically from V4L2).
///
/// Constructors:
/// - [`FramePayload::owned`] — wrap an existing `Vec<u8>`.
/// - [`FramePayload::dma_buf`] — adopt a camera-exported fd + length.
///
/// Access:
/// - [`FramePayload::bytes`] — returns `Some(&[u8])` for the `Owned`
///   variant and `None` for DMA-BUF (the fd itself is opaque to the
///   CPU until the caller explicitly maps it or passes it to a
///   zero-copy consumer like [`yscv_kernels::RknnBackend::wrap_fd`]).
/// - [`FramePayload::dma_buf_fd`] / [`FramePayload::dma_buf_len`] —
///   primitive accessors for the DMA-BUF variant, for consumers that
///   need to re-export the fd to a downstream API.
pub enum FramePayload {
    Owned(Vec<u8>),
    DmaBuf {
        /// Camera-owned DMA-BUF fd. Pipeline MUST NOT close this.
        fd: i32,
        /// Payload length in bytes.
        len: usize,
    },
}

impl FramePayload {
    /// Construct an owned-bytes payload. Use for copy-based pipelines
    /// or for test / fixture inputs.
    pub fn owned(bytes: Vec<u8>) -> Self {
        FramePayload::Owned(bytes)
    }

    /// Construct a DMA-BUF payload from a camera-exported fd + length.
    /// See the module-level ownership rules.
    pub fn dma_buf(fd: i32, len: usize) -> Self {
        FramePayload::DmaBuf { fd, len }
    }

    /// Whether this payload is DMA-BUF-backed (zero-copy).
    pub fn is_dma_buf(&self) -> bool {
        matches!(self, FramePayload::DmaBuf { .. })
    }

    /// Whether this payload owns an in-memory `Vec<u8>`.
    pub fn is_owned(&self) -> bool {
        matches!(self, FramePayload::Owned(_))
    }

    /// Payload length in bytes. Constant-time for both variants.
    pub fn len(&self) -> usize {
        match self {
            FramePayload::Owned(v) => v.len(),
            FramePayload::DmaBuf { len, .. } => *len,
        }
    }

    /// True if there's no data in this payload.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Byte slice view, if the payload is owned. Returns `None` for
    /// DMA-BUF payloads — callers that need CPU access on a DMA-BUF
    /// should go through the camera's `buffer_mut()` API (which
    /// returns the mmap'd region) rather than through this payload
    /// abstraction.
    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            FramePayload::Owned(v) => Some(v.as_slice()),
            FramePayload::DmaBuf { .. } => None,
        }
    }

    /// DMA-BUF file descriptor, if applicable.
    pub fn dma_buf_fd(&self) -> Option<i32> {
        match self {
            FramePayload::DmaBuf { fd, .. } => Some(*fd),
            FramePayload::Owned(_) => None,
        }
    }

    /// DMA-BUF byte count, if applicable.
    pub fn dma_buf_len(&self) -> Option<usize> {
        match self {
            FramePayload::DmaBuf { len, .. } => Some(*len),
            FramePayload::Owned(_) => None,
        }
    }
}

impl fmt::Debug for FramePayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FramePayload::Owned(v) => write!(f, "FramePayload::Owned({} bytes)", v.len()),
            FramePayload::DmaBuf { fd, len } => {
                write!(f, "FramePayload::DmaBuf {{ fd: {fd}, len: {len} }}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_roundtrip() {
        let p = FramePayload::owned(vec![1, 2, 3, 4]);
        assert!(p.is_owned());
        assert!(!p.is_dma_buf());
        assert_eq!(p.len(), 4);
        assert!(!p.is_empty());
        assert_eq!(p.bytes(), Some(&[1u8, 2, 3, 4][..]));
        assert_eq!(p.dma_buf_fd(), None);
        assert_eq!(p.dma_buf_len(), None);
    }

    #[test]
    fn dma_buf_accessors() {
        let p = FramePayload::dma_buf(42, 1920 * 1080 * 3 / 2); // 1080p NV12
        assert!(!p.is_owned());
        assert!(p.is_dma_buf());
        assert_eq!(p.dma_buf_fd(), Some(42));
        assert_eq!(p.dma_buf_len(), Some(1920 * 1080 * 3 / 2));
        // Byte slice not exposed for DMA-BUF — caller uses camera buffer_mut.
        assert!(p.bytes().is_none());
    }

    #[test]
    fn empty_owned_payload() {
        let p = FramePayload::owned(vec![]);
        assert!(p.is_empty());
    }

    #[test]
    fn debug_format_identifies_variant() {
        let owned_dbg = format!("{:?}", FramePayload::owned(vec![0; 100]));
        assert!(owned_dbg.contains("Owned"));
        assert!(owned_dbg.contains("100"));
        let dma_dbg = format!("{:?}", FramePayload::dma_buf(7, 2_073_600));
        assert!(dma_dbg.contains("DmaBuf"));
        assert!(dma_dbg.contains("fd: 7"));
    }
}
