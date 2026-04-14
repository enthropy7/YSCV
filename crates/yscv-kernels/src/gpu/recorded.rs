/// A recorded GPU operation for compiled execution replay.
#[non_exhaustive]
pub enum RecordedOp {
    /// Single dispatch in its own compute pass.
    Single {
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        wg: (u32, u32, u32),
    },
    /// Multiple dispatches in one compute pass (batched, no RAW hazards).
    Batch {
        pipeline: wgpu::ComputePipeline,
        bind_groups: Vec<wgpu::BindGroup>,
        wg_sizes: Vec<(u32, u32, u32)>,
    },
}
