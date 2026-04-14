use std::path::PathBuf;
use thiserror::Error;

/// Errors raised when loading or validating a [`crate::PipelineConfig`].
#[derive(Debug, Error)]
pub enum ConfigError {
    /// File could not be read or doesn't exist.
    #[error("failed to read config file {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// TOML parse failure.
    #[error("invalid TOML in {path:?}: {source}")]
    Toml {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    /// User asked for an accelerator that isn't compiled in or whose
    /// runtime library is missing on this host.
    #[error(
        "task '{task}' requested {accelerator} but it is not available — \
         either compile with the matching feature flag (`--features {feature_hint}`) \
         or run on a host where the runtime library is present"
    )]
    AcceleratorUnavailable {
        task: String,
        accelerator: String,
        feature_hint: String,
    },

    /// Model file referenced by a task does not exist.
    #[error("task '{task}': model file not found at {path:?}")]
    ModelNotFound { task: String, path: PathBuf },

    /// Model file was found but its contents don't match the expected
    /// format (wrong magic bytes, truncated, or the accelerator
    /// runtime refused to parse it).
    #[error("task '{task}': model at {path:?} is unreadable — {reason}")]
    ModelInvalid {
        task: String,
        path: PathBuf,
        reason: String,
    },

    /// Task graph contains a cycle (A depends on B which depends on A).
    #[error("pipeline graph has a cycle through task '{task}'")]
    CyclicDependency { task: String },

    /// Task references an input that no other task or source produces.
    #[error("task '{task}' references unknown input source '{src_name}'")]
    UnknownSource { task: String, src_name: String },

    /// Field validation failure (e.g. negative width, zero fps).
    #[error("invalid field in config: {0}")]
    InvalidField(String),
}

/// Top-level error type returned from pipeline runtime entry points.
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Config(#[from] ConfigError),

    /// Errors from the underlying yscv-kernels backend.
    #[error("kernel error: {0}")]
    Kernel(String),

    /// I/O failure during pipeline execution.
    #[error("pipeline I/O: {0}")]
    Io(#[from] std::io::Error),

    /// Generic runtime failure with a static message.
    #[error("{0}")]
    Other(String),
}

impl From<yscv_kernels::KernelError> for Error {
    fn from(e: yscv_kernels::KernelError) -> Self {
        Error::Kernel(e.to_string())
    }
}
