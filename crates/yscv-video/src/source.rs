use crate::{Frame, VideoError};

/// Generic frame source abstraction.
pub trait FrameSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, VideoError>;
}

impl<S: FrameSource + ?Sized> FrameSource for Box<S> {
    fn next_frame(&mut self) -> Result<Option<Frame>, VideoError> {
        (**self).next_frame()
    }
}

/// Deterministic in-memory source for tests and reproducible local runs.
#[derive(Debug, Clone)]
pub struct InMemoryFrameSource {
    frames: Vec<Frame>,
    cursor: usize,
}

impl InMemoryFrameSource {
    pub const fn new(frames: Vec<Frame>) -> Self {
        Self { frames, cursor: 0 }
    }
}

impl FrameSource for InMemoryFrameSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, VideoError> {
        if self.cursor >= self.frames.len() {
            return Ok(None);
        }
        let frame = self.frames[self.cursor].clone();
        self.cursor += 1;
        Ok(Some(frame))
    }
}
