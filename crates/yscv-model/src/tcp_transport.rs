//! TCP-based transport for multi-node gradient exchange.
//!
//! One node acts as coordinator (server), others connect as workers.
//! Protocol: simple length-prefixed f32 array exchange.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use crate::ModelError;

/// TCP-based transport for multi-node gradient exchange.
///
/// One node acts as **coordinator** (rank 0, TCP server) while all other
/// nodes act as **workers** (TCP clients). The coordinator calls
/// [`TcpTransport::coordinator`] which binds to a socket and blocks until
/// `world_size - 1` workers have connected. Each worker calls
/// [`TcpTransport::worker`] with the coordinator's address and its own rank.
///
/// # Wire protocol
///
/// Every message is a length-prefixed `f32` array:
///
/// 1. 4-byte little-endian `u32` element count.
/// 2. `count * 4` bytes of little-endian `f32` values.
///
/// # Rank coordination
///
/// Workers announce their rank as a 4-byte LE `u32` immediately after the
/// TCP handshake so the coordinator can place each connection in the correct
/// slot.
///
/// # Usage
///
/// For single-machine testing use [`loopback_pair`] which creates a
/// coordinator + worker pair over `127.0.0.1` on a random port.
pub struct TcpTransport {
    #[allow(dead_code)]
    role: NodeRole,
    peers: Vec<Arc<Mutex<TcpStream>>>,
    rank: usize,
    world_size: usize,
}

/// Describes whether this node is the coordinator (rank 0) or a worker.
///
/// The coordinator binds a TCP listener and waits for workers to connect.
/// Workers initiate connections to the coordinator. After the initial
/// handshake every peer can send and receive gradient data symmetrically.
#[derive(Debug, Clone)]
pub enum NodeRole {
    /// The coordinator listens on this address for incoming worker connections.
    /// Always corresponds to rank 0.
    Coordinator { bind_addr: String },
    /// A worker connects to the coordinator at this address.
    /// Rank must be in `1..world_size`.
    Worker { coordinator_addr: String },
}

impl TcpTransport {
    /// Create a coordinator node that listens for worker connections.
    ///
    /// Blocks until `world_size - 1` workers have connected.
    /// The coordinator is always rank 0.
    pub fn coordinator(bind_addr: &str, world_size: usize) -> Result<Self, ModelError> {
        if world_size == 0 {
            return Err(ModelError::TransportError(
                "world_size must be > 0".to_string(),
            ));
        }

        let listener = TcpListener::bind(bind_addr).map_err(|e| {
            ModelError::TransportError(format!("coordinator failed to bind {bind_addr}: {e}"))
        })?;

        let mut peers = Vec::with_capacity(world_size);
        // Slot 0 is unused (self), but we keep it to index by rank.
        // We'll fill it with a dummy that is never used.
        // Actually, store peers in order of connection; index by worker rank.
        // Workers send their rank as a u32 LE upon connecting.

        // Pre-allocate with None so we can insert by rank.
        let mut peer_slots: Vec<Option<Arc<Mutex<TcpStream>>>> =
            (0..world_size).map(|_| None).collect();

        for _ in 1..world_size {
            let (mut stream, _addr) = listener.accept().map_err(|e| {
                ModelError::TransportError(format!("coordinator accept failed: {e}"))
            })?;

            stream
                .set_nodelay(true)
                .map_err(|e| ModelError::TransportError(format!("set_nodelay failed: {e}")))?;

            // Read the worker's rank.
            let mut rank_buf = [0u8; 4];
            stream.read_exact(&mut rank_buf).map_err(|e| {
                ModelError::TransportError(format!("failed to read worker rank: {e}"))
            })?;
            let worker_rank = u32::from_le_bytes(rank_buf) as usize;

            if worker_rank == 0 || worker_rank >= world_size {
                return Err(ModelError::TransportError(format!(
                    "invalid worker rank {worker_rank}"
                )));
            }

            peer_slots[worker_rank] = Some(Arc::new(Mutex::new(stream)));
        }

        // Build the peers vec. Index 0 is self (coordinator) -- store a placeholder.
        for slot in peer_slots.iter_mut() {
            if let Some(s) = slot.take() {
                peers.push(s);
            }
        }

        Ok(Self {
            role: NodeRole::Coordinator {
                bind_addr: bind_addr.to_string(),
            },
            peers,
            rank: 0,
            world_size,
        })
    }

    /// Create a worker node that connects to the coordinator.
    pub fn worker(coordinator_addr: &str, rank: usize) -> Result<Self, ModelError> {
        if rank == 0 {
            return Err(ModelError::TransportError(
                "worker rank must be > 0; use coordinator() for rank 0".to_string(),
            ));
        }

        let mut stream = TcpStream::connect(coordinator_addr).map_err(|e| {
            ModelError::TransportError(format!(
                "worker rank {rank} failed to connect to {coordinator_addr}: {e}"
            ))
        })?;

        stream
            .set_nodelay(true)
            .map_err(|e| ModelError::TransportError(format!("set_nodelay failed: {e}")))?;

        // Announce our rank.
        stream
            .write_all(&(rank as u32).to_le_bytes())
            .map_err(|e| ModelError::TransportError(format!("failed to send rank: {e}")))?;

        let peers = vec![Arc::new(Mutex::new(stream))];

        Ok(Self {
            role: NodeRole::Worker {
                coordinator_addr: coordinator_addr.to_string(),
            },
            peers,
            rank,
            world_size: 0, // will be unknown until protocol exchange; for now unused
        })
    }

    /// Send a tensor's data to a specific peer.
    ///
    /// `peer` is an index into the peers list (0-based).
    pub fn send(&self, peer: usize, data: &[f32]) -> Result<(), ModelError> {
        if peer >= self.peers.len() {
            return Err(ModelError::TransportError(format!(
                "peer index {peer} out of range (have {} peers)",
                self.peers.len()
            )));
        }
        let stream = &self.peers[peer];
        let mut stream = stream
            .lock()
            .map_err(|_| ModelError::TransportError("lock poisoned".into()))?;
        let len = data.len() as u32;
        stream
            .write_all(&len.to_le_bytes())
            .map_err(|e| ModelError::TransportError(e.to_string()))?;
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        stream
            .write_all(&bytes)
            .map_err(|e| ModelError::TransportError(e.to_string()))?;
        Ok(())
    }

    /// Receive tensor data from a specific peer.
    ///
    /// `peer` is an index into the peers list (0-based).
    pub fn recv(&self, peer: usize) -> Result<Vec<f32>, ModelError> {
        if peer >= self.peers.len() {
            return Err(ModelError::TransportError(format!(
                "peer index {peer} out of range (have {} peers)",
                self.peers.len()
            )));
        }
        let stream = &self.peers[peer];
        let mut stream = stream
            .lock()
            .map_err(|_| ModelError::TransportError("lock poisoned".into()))?;
        let mut len_buf = [0u8; 4];
        stream
            .read_exact(&mut len_buf)
            .map_err(|e| ModelError::TransportError(e.to_string()))?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len * 4];
        stream
            .read_exact(&mut buf)
            .map_err(|e| ModelError::TransportError(e.to_string()))?;
        let data: Vec<f32> = buf
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(data)
    }

    /// All-reduce: sum gradients across all nodes.
    ///
    /// Implements a simple butterfly all-reduce pattern: the coordinator
    /// collects data from all workers, computes the element-wise sum, and
    /// broadcasts the result back.
    pub fn allreduce_sum(&self, data: &mut [f32]) -> Result<(), ModelError> {
        let n = self.peers.len() + 1; // total nodes = peers + self
        if n <= 1 {
            return Ok(());
        }

        if self.rank == 0 {
            // Coordinator: receive from each worker, sum, broadcast result.
            let mut acc: Vec<f32> = data.to_vec();
            for peer_idx in 0..self.peers.len() {
                let remote = self.recv(peer_idx)?;
                if remote.len() != acc.len() {
                    return Err(ModelError::TransportError(format!(
                        "allreduce length mismatch: expected {}, got {}",
                        acc.len(),
                        remote.len()
                    )));
                }
                for (a, r) in acc.iter_mut().zip(remote.iter()) {
                    *a += r;
                }
            }
            // Broadcast summed result back to all workers.
            for peer_idx in 0..self.peers.len() {
                self.send(peer_idx, &acc)?;
            }
            // Update local data in place.
            data.copy_from_slice(&acc);
        } else {
            // Worker: send local data to coordinator, receive summed result.
            self.send(0, data)?;
            let result = self.recv(0)?;
            if result.len() != data.len() {
                return Err(ModelError::TransportError(format!(
                    "allreduce length mismatch: expected {}, got {}",
                    data.len(),
                    result.len()
                )));
            }
            data.copy_from_slice(&result);
        }

        Ok(())
    }

    /// Returns the rank of this transport instance.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the world size (total number of nodes).
    pub const fn world_size(&self) -> usize {
        // For coordinator, world_size is set during construction.
        // For workers, world_size = peers.len() + 1 (self).
        if self.world_size > 0 {
            self.world_size
        } else {
            self.peers.len() + 1
        }
    }
}

/// Wrapper that uses a [`TcpTransport`] for gradient aggregation.
pub struct TcpAllReduceAggregator {
    transport: TcpTransport,
}

impl TcpAllReduceAggregator {
    /// Create a new aggregator wrapping the given TCP transport.
    pub const fn new(transport: TcpTransport) -> Self {
        Self { transport }
    }

    /// All-reduce (sum) raw f32 slices across all connected nodes.
    pub fn allreduce_sum(&self, data: &mut [f32]) -> Result<(), ModelError> {
        self.transport.allreduce_sum(data)
    }
}

/// Create a loopback TCP transport pair for testing.
///
/// Starts a coordinator on a random port on localhost and connects a single
/// worker to it. Returns `(coordinator, worker)`.
pub fn loopback_pair() -> Result<(TcpTransport, TcpTransport), ModelError> {
    // Bind to port 0 to get a random available port.
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| {
        ModelError::TransportError(format!("failed to bind loopback listener: {e}"))
    })?;
    let port = listener
        .local_addr()
        .map_err(|e| ModelError::TransportError(format!("failed to get local addr: {e}")))?
        .port();

    // We need to accept in a separate thread because coordinator() blocks.
    // Instead, build the transports manually using the raw listener.

    // Spawn the worker connection in a thread.
    let addr = format!("127.0.0.1:{port}");
    let addr_clone = addr.clone();
    let worker_handle = std::thread::spawn(move || -> Result<TcpTransport, ModelError> {
        TcpTransport::worker(&addr_clone, 1)
    });

    // Accept the single worker connection as coordinator.
    // Build coordinator manually from the existing listener.
    let (mut stream, _) = listener
        .accept()
        .map_err(|e| ModelError::TransportError(format!("loopback accept failed: {e}")))?;
    stream
        .set_nodelay(true)
        .map_err(|e| ModelError::TransportError(format!("set_nodelay failed: {e}")))?;

    // Read the worker's rank announcement.
    let mut rank_buf = [0u8; 4];
    stream
        .read_exact(&mut rank_buf)
        .map_err(|e| ModelError::TransportError(format!("failed to read worker rank: {e}")))?;

    let coordinator = TcpTransport {
        role: NodeRole::Coordinator { bind_addr: addr },
        peers: vec![Arc::new(Mutex::new(stream))],
        rank: 0,
        world_size: 2,
    };

    let worker = worker_handle
        .join()
        .map_err(|_| ModelError::TransportError("worker thread panicked".to_string()))??;

    Ok((coordinator, worker))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcp_loopback_send_recv() {
        let (coord, worker) = loopback_pair().unwrap();

        let send_data: Vec<f32> = vec![1.0, 2.5, -3.0, 0.0];

        // Coordinator sends to worker (peer index 0 = the single worker).
        coord.send(0, &send_data).unwrap();

        // Worker receives from coordinator (peer index 0 = the coordinator).
        let received = worker.recv(0).unwrap();

        assert_eq!(received, send_data);
    }

    #[test]
    fn tcp_loopback_allreduce() {
        let (coord, worker) = loopback_pair().unwrap();

        // Coordinator has [1.0, 2.0, 3.0], worker has [4.0, 5.0, 6.0].
        // After allreduce_sum, both should have [5.0, 7.0, 9.0].

        let coord_handle = std::thread::spawn(move || -> Result<Vec<f32>, ModelError> {
            let mut data = vec![1.0, 2.0, 3.0];
            coord.allreduce_sum(&mut data)?;
            Ok(data)
        });

        let worker_handle = std::thread::spawn(move || -> Result<Vec<f32>, ModelError> {
            let mut data = vec![4.0, 5.0, 6.0];
            worker.allreduce_sum(&mut data)?;
            Ok(data)
        });

        let coord_result = coord_handle.join().unwrap().unwrap();
        let worker_result = worker_handle.join().unwrap().unwrap();

        assert_eq!(coord_result, vec![5.0, 7.0, 9.0]);
        assert_eq!(worker_result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn tcp_transport_rank_world_size() {
        let (coord, worker) = loopback_pair().unwrap();

        assert_eq!(coord.rank(), 0);
        assert_eq!(coord.world_size(), 2);

        assert_eq!(worker.rank(), 1);
        assert_eq!(worker.world_size(), 2);
    }
}
