use thiserror::Error;

/// Errors from MPP encoder operations.
#[derive(Debug, Error)]
pub enum MppError {
    /// `librockchip_mpp.so` not found on this host.
    #[error("librockchip_mpp.so not found — install Rockchip MPP runtime")]
    LibraryNotFound,

    /// Required MPP function symbol missing from loaded library.
    #[error("MPP symbol `{0}` not found — incompatible MPP version")]
    SymbolMissing(&'static str),

    /// MPP returned non-zero status code from a call.
    #[error("MPP `{op}` failed with status {status}")]
    CallFailed { op: &'static str, status: i32 },

    /// User passed an invalid encoder configuration.
    #[error("MPP encoder config invalid: {0}")]
    InvalidConfig(String),

    /// MPP context not initialised yet.
    #[error("MPP encoder not initialised")]
    NotInitialised,

    /// MPP encoder reported buffer-not-ready for non-blocking call.
    #[error("MPP buffer would block — caller should retry")]
    WouldBlock,
}

pub type MppResult<T> = Result<T, MppError>;
