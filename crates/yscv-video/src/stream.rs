use crate::{Frame, FrameSource, VideoError};

/// Pull-based frame stream wrapper with deterministic optional limit.
#[derive(Debug)]
pub struct FrameStream<S: FrameSource> {
    source: S,
    max_frames: Option<usize>,
    emitted: usize,
}

impl<S: FrameSource> FrameStream<S> {
    pub const fn new(source: S) -> Self {
        Self {
            source,
            max_frames: None,
            emitted: 0,
        }
    }

    pub const fn with_max_frames(mut self, max_frames: usize) -> Self {
        self.max_frames = Some(max_frames);
        self
    }

    pub fn try_next(&mut self) -> Result<Option<Frame>, VideoError> {
        if let Some(limit) = self.max_frames
            && self.emitted >= limit
        {
            return Ok(None);
        }
        let frame = self.source.next_frame()?;
        if frame.is_some() {
            self.emitted += 1;
        }
        Ok(frame)
    }
}
