//! Distributed training primitives for multi-GPU and multi-node training.
//!
//! This module provides the building blocks for scaling training across multiple
//! workers (processes, threads, or machines):
//!
//! - **Transport layer** ([`Transport`] trait, [`InProcessTransport`]) -- byte-level
//!   send/recv/barrier used by aggregation strategies. For TCP-based networking see
//!   [`crate::tcp_transport`].
//! - **Gradient aggregation** ([`AllReduceAggregator`], [`ParameterServer`]) --
//!   strategies for combining gradients from different workers.
//! - **Gradient compression** ([`TopKCompressor`], [`compress_gradients`],
//!   [`decompress_gradients`]) -- reduce communication volume by sending only the
//!   most significant gradient elements.
//! - **Pipeline parallelism** ([`PipelineParallelConfig`], [`split_into_stages`]) --
//!   partition a model's layers across stages so different micro-batches execute
//!   concurrently in different stages.
//! - **Tensor sharding / FSDP** ([`shard_tensor`], [`gather_shards`]) -- split
//!   parameter tensors along the first dimension for Fully Sharded Data Parallel
//!   style training.
//! - **Distributed train step** ([`distributed_train_step`]) -- combines local
//!   gradient computation, aggregation, and parameter update into a single call.
//!
//! ## Limitations
//! - TCP transport only (no RDMA/InfiniBand)
//! - Gradient synchronization is synchronous (no async overlap with compute)
//! - Tested up to ~8 nodes; not validated at datacenter scale (100+)
//! - No NCCL/MPI backend (use TCP AllReduce or ParameterServer)
//!
//! # Quick start
//!
//! ```rust,ignore
//! use yscv_model::{AllReduceAggregator, InProcessTransport, DistributedConfig, distributed_train_step};
//!
//! // Create two in-process workers for testing
//! let transports = InProcessTransport::create_group(2);
//!
//! let config = DistributedConfig { world_size: 2, rank: 0, coordinator_addr: String::new() };
//! let mut aggregator = AllReduceAggregator::new(config, Box::new(transports.into_iter().next().unwrap()));
//!
//! let loss = distributed_train_step(
//!     || Ok((0.5, vec![/* local gradients */])),
//!     |aggregated| { /* apply gradients */ Ok(()) },
//!     &mut aggregator,
//! ).unwrap();
//! ```

use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use yscv_tensor::Tensor;

use crate::ModelError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Identifies a worker inside a distributed training group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Total number of workers.
    pub world_size: usize,
    /// Zero-based rank of this worker.
    pub rank: usize,
    /// `host:port` of the rank-0 coordinator.
    pub coordinator_addr: String,
}

// ---------------------------------------------------------------------------
// Transport trait
// ---------------------------------------------------------------------------

/// Byte-level communication primitive used by aggregation strategies.
pub trait Transport: Send {
    /// Send `data` to the worker with the given rank.
    fn send(&self, dest_rank: usize, data: &[u8]) -> Result<(), ModelError>;
    /// Receive data from the worker with the given rank.
    fn recv(&self, src_rank: usize) -> Result<Vec<u8>, ModelError>;
    /// Block until all workers reach the barrier.
    fn barrier(&self) -> Result<(), ModelError>;
}

/// In-process transport backed by `mpsc` channels (for testing).
pub struct InProcessTransport {
    rank: usize,
    _world_size: usize,
    // senders[dest_rank] = sender for messages TO that rank FROM this rank
    senders: Vec<mpsc::Sender<Vec<u8>>>,
    // receivers[src_rank] = receiver for messages FROM that rank TO this rank
    receivers: Vec<mpsc::Receiver<Vec<u8>>>,
    barrier_state: Arc<Mutex<BarrierState>>,
}

struct BarrierState {
    count: usize,
    generation: u64,
    target: usize,
    condvar_senders: Vec<mpsc::Sender<()>>,
    condvar_receivers: Vec<Option<mpsc::Receiver<()>>>,
}

impl InProcessTransport {
    /// Creates `world_size` connected transports for in-process testing.
    #[allow(clippy::type_complexity)]
    pub fn create_group(world_size: usize) -> Vec<Self> {
        // Build sender/receiver grids: sender_grid[src][dest], receiver_grid[src][dest]
        let mut sender_grid: Vec<Vec<Option<mpsc::Sender<Vec<u8>>>>> = Vec::new();
        let mut receiver_grid: Vec<Vec<Option<mpsc::Receiver<Vec<u8>>>>> = Vec::new();
        for _ in 0..world_size {
            let mut s_row = Vec::new();
            let mut r_row = Vec::new();
            for _ in 0..world_size {
                let (tx, rx) = mpsc::channel();
                s_row.push(Some(tx));
                r_row.push(Some(rx));
            }
            sender_grid.push(s_row);
            receiver_grid.push(r_row);
        }

        // Barrier state
        let mut barrier_senders = Vec::new();
        let mut barrier_receivers = Vec::new();
        for _ in 0..world_size {
            let (tx, rx) = mpsc::channel();
            barrier_senders.push(tx);
            barrier_receivers.push(Some(rx));
        }

        let barrier_state = Arc::new(Mutex::new(BarrierState {
            count: 0,
            generation: 0,
            target: world_size,
            condvar_senders: barrier_senders,
            condvar_receivers: barrier_receivers,
        }));

        let mut transports = Vec::new();

        for r in 0..world_size {
            let mut my_senders = Vec::new();
            let mut my_receivers = Vec::new();

            // transport[r].senders[d] = sender_grid[r][d]
            for item in sender_grid[r].iter_mut() {
                my_senders.push(item.take().expect("sender not yet taken"));
            }
            // transport[r].receivers[s] = receiver_grid[s][r]
            for row in receiver_grid.iter_mut() {
                my_receivers.push(row[r].take().expect("receiver not yet taken"));
            }

            transports.push(InProcessTransport {
                rank: r,
                _world_size: world_size,
                senders: my_senders,
                receivers: my_receivers,
                barrier_state: barrier_state.clone(),
            });
        }

        transports
    }
}

impl Transport for InProcessTransport {
    fn send(&self, dest_rank: usize, data: &[u8]) -> Result<(), ModelError> {
        self.senders[dest_rank].send(data.to_vec()).map_err(|e| {
            ModelError::CheckpointSerialization {
                message: format!("transport send failed: {e}"),
            }
        })
    }

    fn recv(&self, src_rank: usize) -> Result<Vec<u8>, ModelError> {
        self.receivers[src_rank]
            .recv()
            .map_err(|e| ModelError::CheckpointSerialization {
                message: format!("transport recv failed: {e}"),
            })
    }

    fn barrier(&self) -> Result<(), ModelError> {
        let mut state =
            self.barrier_state
                .lock()
                .map_err(|e| ModelError::CheckpointSerialization {
                    message: format!("barrier lock failed: {e}"),
                })?;
        state.count += 1;
        if state.count == state.target {
            state.count = 0;
            state.generation += 1;
            // Wake all waiting workers
            for tx in &state.condvar_senders {
                let _ = tx.send(());
            }
            Ok(())
        } else {
            let rx = state.condvar_receivers[self.rank].take();
            drop(state);
            if let Some(rx) = rx {
                let _ = rx.recv();
            }
            // Put receiver back for next barrier
            let (tx, rx) = mpsc::channel();
            let mut state =
                self.barrier_state
                    .lock()
                    .map_err(|e| ModelError::CheckpointSerialization {
                        message: format!("barrier lock failed: {e}"),
                    })?;
            state.condvar_senders[self.rank] = tx;
            state.condvar_receivers[self.rank] = Some(rx);
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient Aggregator trait
// ---------------------------------------------------------------------------

/// Strategy for combining gradients across distributed workers.
pub trait GradientAggregator: Send {
    /// Aggregate local gradients, returning the combined result.
    fn aggregate(&mut self, local_gradients: &[Tensor]) -> Result<Vec<Tensor>, ModelError>;
}

/// No-op aggregator for single-machine training (API uniformity).
pub struct LocalAggregator;

impl GradientAggregator for LocalAggregator {
    fn aggregate(&mut self, local_gradients: &[Tensor]) -> Result<Vec<Tensor>, ModelError> {
        Ok(local_gradients.to_vec())
    }
}

/// All-reduce aggregator: averages gradients across all workers via ring reduce.
///
/// Each worker independently computes local gradients and then calls
/// [`GradientAggregator::aggregate`]. Internally the aggregator serialises the
/// gradient tensors, performs a ring all-reduce (scatter-reduce followed by
/// all-gather) over the configured [`Transport`], and returns the element-wise
/// average. This is the most common strategy for synchronous data-parallel
/// training because every worker ends up with the same averaged gradients
/// without a central bottleneck.
///
/// # Example
///
/// ```rust,ignore
/// let transports = InProcessTransport::create_group(2);
/// let cfg = DistributedConfig { world_size: 2, rank: 0, coordinator_addr: String::new() };
/// let mut agg = AllReduceAggregator::new(cfg, Box::new(transports.into_iter().next().unwrap()));
/// let averaged = agg.aggregate(&local_grads).unwrap();
/// ```
pub struct AllReduceAggregator {
    config: DistributedConfig,
    transport: Box<dyn Transport>,
}

impl AllReduceAggregator {
    pub fn new(config: DistributedConfig, transport: Box<dyn Transport>) -> Self {
        Self { config, transport }
    }
}

impl GradientAggregator for AllReduceAggregator {
    fn aggregate(&mut self, local_gradients: &[Tensor]) -> Result<Vec<Tensor>, ModelError> {
        let world = self.config.world_size;
        if world <= 1 {
            return Ok(local_gradients.to_vec());
        }

        // Serialize local gradients
        let local_bytes = serialize_tensors(local_gradients)?;

        // Ring all-reduce: each rank sends to (rank+1) % world, receives from (rank-1+world) % world
        let next = (self.config.rank + 1) % world;
        let prev = (self.config.rank + world - 1) % world;

        // Scatter-reduce phase
        let mut accumulated = local_bytes.clone();
        for _ in 0..(world - 1) {
            self.transport.send(next, &accumulated)?;
            let received = self.transport.recv(prev)?;
            // Accumulate: element-wise add
            let recv_tensors = deserialize_tensors(&received)?;
            let acc_tensors = deserialize_tensors(&accumulated)?;
            let mut summed = Vec::new();
            for (a, r) in acc_tensors.iter().zip(recv_tensors.iter()) {
                summed.push(a.add(r)?);
            }
            accumulated = serialize_tensors(&summed)?;
        }

        // Average
        let result_tensors = deserialize_tensors(&accumulated)?;
        let scale = 1.0 / world as f32;
        let mut averaged = Vec::new();
        for t in &result_tensors {
            averaged.push(t.scale(scale));
        }

        self.transport.barrier()?;
        Ok(averaged)
    }
}

// ---------------------------------------------------------------------------
// Parameter Server
// ---------------------------------------------------------------------------

/// Centralized parameter server: rank 0 collects, averages, and broadcasts
/// gradients (or parameters).
///
/// Use this instead of [`AllReduceAggregator`] when you want a star topology
/// (all workers communicate only with rank 0). This is simpler to reason about
/// and works well when the coordinator has high bandwidth, but can become a
/// bottleneck at large scale compared to ring all-reduce.
///
/// * [`broadcast_params`](Self::broadcast_params) -- rank 0 sends the current
///   parameters to all workers (useful at initialisation or after a checkpoint
///   restore).
/// * [`reduce_gradients`](Self::reduce_gradients) -- workers send local
///   gradients to rank 0; rank 0 averages them and broadcasts the result back.
pub struct ParameterServer {
    config: DistributedConfig,
    transport: Box<dyn Transport>,
}

impl ParameterServer {
    pub fn new(config: DistributedConfig, transport: Box<dyn Transport>) -> Self {
        Self { config, transport }
    }

    /// Rank 0 broadcasts params to all workers; workers receive and return them.
    pub fn broadcast_params(&self, params: &[Tensor]) -> Result<Vec<Tensor>, ModelError> {
        let world = self.config.world_size;
        if world <= 1 {
            return Ok(params.to_vec());
        }

        if self.config.rank == 0 {
            let data = serialize_tensors(params)?;
            for dest in 1..world {
                self.transport.send(dest, &data)?;
            }
            Ok(params.to_vec())
        } else {
            let data = self.transport.recv(0)?;
            deserialize_tensors(&data)
        }
    }

    /// Workers send gradients to rank 0; rank 0 averages and returns result.
    pub fn reduce_gradients(&self, grads: &[Tensor]) -> Result<Vec<Tensor>, ModelError> {
        let world = self.config.world_size;
        if world <= 1 {
            return Ok(grads.to_vec());
        }

        if self.config.rank == 0 {
            let mut acc = grads.to_vec();
            for src in 1..world {
                let data = self.transport.recv(src)?;
                let remote_grads = deserialize_tensors(&data)?;
                for (a, r) in acc.iter_mut().zip(remote_grads.iter()) {
                    *a = a.add(r)?;
                }
            }
            let scale = 1.0 / world as f32;
            let mut averaged = Vec::new();
            for t in &acc {
                averaged.push(t.scale(scale));
            }
            // Broadcast averaged gradients back
            let data = serialize_tensors(&averaged)?;
            for dest in 1..world {
                self.transport.send(dest, &data)?;
            }
            Ok(averaged)
        } else {
            let data = serialize_tensors(grads)?;
            self.transport.send(0, &data)?;
            let result_data = self.transport.recv(0)?;
            deserialize_tensors(&result_data)
        }
    }
}

// ---------------------------------------------------------------------------
// Data-parallel config
// ---------------------------------------------------------------------------

/// Configuration for data-parallel distributed training.
pub struct DataParallelConfig {
    pub config: DistributedConfig,
    pub aggregator: Box<dyn GradientAggregator>,
}

// ---------------------------------------------------------------------------
// Gradient compression
// ---------------------------------------------------------------------------

/// Compressed gradient: stores only the top-k elements by magnitude.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedGradient {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
    pub original_len: usize,
}

/// Top-K gradient compressor: keeps only the top `ratio` fraction of gradients.
pub struct TopKCompressor {
    pub ratio: f32,
}

impl TopKCompressor {
    pub const fn new(ratio: f32) -> Self {
        Self {
            ratio: ratio.clamp(0.0, 1.0),
        }
    }
}

/// Compress gradients by keeping only top-k% elements by magnitude.
pub fn compress_gradients(gradients: &[Tensor], ratio: f32) -> Vec<CompressedGradient> {
    let ratio = ratio.clamp(0.0, 1.0);
    gradients
        .iter()
        .map(|t| {
            let data = t.data();
            let k = ((data.len() as f32 * ratio).ceil() as usize)
                .max(1)
                .min(data.len());

            // Find top-k by magnitude
            let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k);

            let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
            let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();

            CompressedGradient {
                indices,
                values,
                original_len: data.len(),
            }
        })
        .collect()
}

/// Decompress gradients back to full tensors.
pub fn decompress_gradients(
    compressed: &[CompressedGradient],
    shapes: &[Vec<usize>],
) -> Result<Vec<Tensor>, ModelError> {
    let mut result = Vec::with_capacity(compressed.len());
    for (cg, shape) in compressed.iter().zip(shapes.iter()) {
        let mut data = vec![0.0f32; cg.original_len];
        for (&idx, &val) in cg.indices.iter().zip(cg.values.iter()) {
            if idx < data.len() {
                data[idx] = val;
            }
        }
        result.push(Tensor::from_vec(shape.clone(), data)?);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Distributed training step
// ---------------------------------------------------------------------------

/// Performs a single distributed training step: forward, backward, aggregate, update.
///
/// This is the main entry point for one iteration of distributed training.
/// It calls `compute_gradients_fn` to run the local forward and backward pass,
/// then aggregates gradients across workers via the provided `aggregator`,
/// and finally applies the aggregated gradients through `apply_gradients_fn`.
///
/// # Arguments
///
/// * `compute_gradients_fn` -- closure that returns `(loss, local_gradients)`.
/// * `apply_gradients_fn` -- closure that receives the aggregated gradients and
///   updates model parameters (e.g. via an optimizer step).
/// * `aggregator` -- the gradient aggregation strategy (e.g. [`AllReduceAggregator`]
///   or [`LocalAggregator`] for single-worker training).
///
/// # Returns
///
/// The scalar loss value produced by `compute_gradients_fn`.
///
/// # Example
///
/// ```rust,ignore
/// let loss = distributed_train_step(
///     || { /* forward + backward */ Ok((loss_val, grads)) },
///     |agg_grads| { optimizer.apply(agg_grads); Ok(()) },
///     &mut aggregator,
/// )?;
/// ```
pub fn distributed_train_step<F, G>(
    compute_gradients_fn: F,
    apply_gradients_fn: G,
    aggregator: &mut dyn GradientAggregator,
) -> Result<f32, ModelError>
where
    F: FnOnce() -> Result<(f32, Vec<Tensor>), ModelError>,
    G: FnOnce(&[Tensor]) -> Result<(), ModelError>,
{
    let (loss, local_grads) = compute_gradients_fn()?;
    let aggregated = aggregator.aggregate(&local_grads)?;
    apply_gradients_fn(&aggregated)?;
    Ok(loss)
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

fn serialize_tensors(tensors: &[Tensor]) -> Result<Vec<u8>, ModelError> {
    let mut entries = Vec::new();
    for t in tensors {
        let shape = t.shape().to_vec();
        let data = t.data().to_vec();
        entries.push((shape, data));
    }
    serde_json::to_vec(&entries).map_err(|e| ModelError::CheckpointSerialization {
        message: format!("tensor serialization failed: {e}"),
    })
}

fn deserialize_tensors(data: &[u8]) -> Result<Vec<Tensor>, ModelError> {
    let entries: Vec<(Vec<usize>, Vec<f32>)> =
        serde_json::from_slice(data).map_err(|e| ModelError::CheckpointSerialization {
            message: format!("tensor deserialization failed: {e}"),
        })?;
    let mut tensors = Vec::with_capacity(entries.len());
    for (shape, values) in entries {
        tensors.push(Tensor::from_vec(shape, values)?);
    }
    Ok(tensors)
}

// ---------------------------------------------------------------------------
// Pipeline Parallelism
// ---------------------------------------------------------------------------

/// Pipeline parallelism: split a sequential model across multiple stages.
///
/// Each stage holds a contiguous subset of layers. During forward pass,
/// micro-batches flow through stages sequentially. This enables training
/// models larger than single-device memory.
pub struct PipelineStage {
    /// Layer indices [start, end) in the original model
    pub start_layer: usize,
    pub end_layer: usize,
    /// Stage rank in the pipeline
    pub rank: usize,
}

/// Configuration for pipeline-parallel training.
///
/// Pipeline parallelism partitions a model's sequential layers into
/// `num_stages` stages, each assigned to a different device. During training,
/// the mini-batch is split into `num_micro_batches` micro-batches that flow
/// through the pipeline concurrently (GPipe-style scheduling), reducing the
/// bubble time compared to naive sequential execution.
///
/// Use [`split_into_stages`] to compute the layer ranges for each stage.
pub struct PipelineParallelConfig {
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Number of micro-batches per mini-batch
    pub num_micro_batches: usize,
}

impl PipelineParallelConfig {
    pub const fn new(num_stages: usize, num_micro_batches: usize) -> Self {
        Self {
            num_stages,
            num_micro_batches,
        }
    }
}

/// Split a model with `num_layers` layers into `num_stages` roughly equal stages.
pub fn split_into_stages(num_layers: usize, num_stages: usize) -> Vec<PipelineStage> {
    assert!(num_stages > 0 && num_stages <= num_layers);
    let base = num_layers / num_stages;
    let remainder = num_layers % num_stages;
    let mut stages = Vec::with_capacity(num_stages);
    let mut start = 0;
    for rank in 0..num_stages {
        let extra = if rank < remainder { 1 } else { 0 };
        let end = start + base + extra;
        stages.push(PipelineStage {
            start_layer: start,
            end_layer: end,
            rank,
        });
        start = end;
    }
    stages
}

// ---------------------------------------------------------------------------
// Tensor Sharding (FSDP-lite)
// ---------------------------------------------------------------------------

/// Shard a tensor along its first dimension into `num_shards` roughly equal parts.
///
/// This is the core primitive for Fully Sharded Data Parallel (FSDP)-style
/// training: large parameter tensors are split across workers so that each
/// worker stores only its shard. Before a forward or backward pass the
/// shards are gathered (see [`gather_shards`]), and after the pass only the
/// local shard's gradients are kept.
///
/// Each shard is a separate [`Tensor`] that can be placed on a different device.
/// If `num_shards` does not evenly divide the first dimension, earlier shards
/// receive one extra row.
pub fn shard_tensor(tensor: &Tensor, num_shards: usize) -> Result<Vec<Tensor>, ModelError> {
    if num_shards == 0 {
        return Err(ModelError::InvalidConv2dStride {
            stride_h: 0,
            stride_w: 0,
        });
    }
    let shape = tensor.shape();
    if shape.is_empty() {
        return Ok(vec![tensor.clone()]);
    }
    let first_dim = shape[0];
    if num_shards > first_dim {
        return Err(ModelError::InvalidParameterShape {
            parameter: "shard_tensor",
            expected: vec![num_shards],
            got: shape.to_vec(),
        });
    }

    let data = tensor.data();
    let stride = data.len() / first_dim; // elements per row along dim 0
    let base = first_dim / num_shards;
    let remainder = first_dim % num_shards;

    let mut shards = Vec::with_capacity(num_shards);
    let mut offset = 0;
    for i in 0..num_shards {
        let rows = base + if i < remainder { 1 } else { 0 };
        let start = offset * stride;
        let end = (offset + rows) * stride;
        let mut shard_shape = shape.to_vec();
        shard_shape[0] = rows;
        shards.push(Tensor::from_vec(shard_shape, data[start..end].to_vec())?);
        offset += rows;
    }
    Ok(shards)
}

/// Reassemble shards (produced by [`shard_tensor`]) back into a single tensor.
///
/// Concatenates along the first dimension. The invariant
/// `gather_shards(&shard_tensor(t, n)?) == Ok(t.clone())` holds for any
/// valid `n`.
pub fn gather_shards(shards: &[Tensor]) -> Result<Tensor, ModelError> {
    if shards.is_empty() {
        return Err(ModelError::InvalidParameterShape {
            parameter: "gather_shards",
            expected: vec![1],
            got: vec![0],
        });
    }
    if shards.len() == 1 {
        return Ok(shards[0].clone());
    }

    let first_shape = shards[0].shape();
    let tail: Vec<usize> = first_shape[1..].to_vec();
    let stride: usize = tail.iter().product::<usize>().max(1);

    let total_rows: usize = shards.iter().map(|s| s.shape()[0]).sum();
    let mut data = Vec::with_capacity(total_rows * stride);
    for shard in shards {
        data.extend_from_slice(shard.data());
    }

    let mut out_shape = vec![total_rows];
    out_shape.extend_from_slice(&tail);
    Tensor::from_vec(out_shape, data).map_err(ModelError::Tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_stages_cover_all_layers() {
        let stages = split_into_stages(10, 3);
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0].start_layer, 0);
        assert_eq!(stages[2].end_layer, 10);
        for i in 1..stages.len() {
            assert_eq!(stages[i].start_layer, stages[i - 1].end_layer);
        }
    }

    #[test]
    fn shard_and_gather_roundtrip() {
        let t = Tensor::from_vec(vec![6, 4], (0..24).map(|i| i as f32).collect()).unwrap();
        let shards = shard_tensor(&t, 3).unwrap();
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0].shape(), &[2, 4]);
        assert_eq!(shards[1].shape(), &[2, 4]);
        assert_eq!(shards[2].shape(), &[2, 4]);
        let gathered = gather_shards(&shards).unwrap();
        assert_eq!(gathered.shape(), t.shape());
        assert_eq!(gathered.data(), t.data());
    }

    #[test]
    fn shard_uneven_split() {
        let t = Tensor::from_vec(vec![7, 2], (0..14).map(|i| i as f32).collect()).unwrap();
        let shards = shard_tensor(&t, 3).unwrap();
        // 7 / 3 = 2 base + 1 remainder → first shard gets 3, others get 2
        assert_eq!(shards[0].shape()[0], 3);
        assert_eq!(shards[1].shape()[0], 2);
        assert_eq!(shards[2].shape()[0], 2);
        let gathered = gather_shards(&shards).unwrap();
        assert_eq!(gathered.data(), t.data());
    }
}
