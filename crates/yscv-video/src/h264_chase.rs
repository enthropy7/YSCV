//! Threaded in-loop deblocking for the H.264 decoder ("shadow chase").
//!
//! The decode thread reconstructs macroblock rows into its own planes and
//! never filters them — intra prediction therefore always reads unfiltered
//! neighbour samples, exactly as clause 8.3 requires. As each macroblock row
//! completes, its samples and deblock metadata are copied into a message and
//! sent to a worker thread, which assembles a shadow copy of the frame and
//! runs the clause 8.7 filter over it sequentially (identical maths, identical
//! order — bit-exact with the single-threaded filter). When the frame
//! finishes, the filtered shadow planes come back and simply swap places with
//! the unfiltered ones: the shadow becomes the reference picture, the decode
//! buffers become the next shadow. Ownership moves through channels, so the
//! two threads never share mutable state.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use super::h264_deblock::{DeblockInfo, deblock_mb_row};
use super::h264_motion::{padded_plane_geometry, replicate_plane_edges};

/// One frame's worth of padded YUV planes, circulating between the decoder
/// (as decode targets / references) and the worker (as shadow buffers).
pub(crate) struct PlaneSet {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// Frame-constant deblock parameters, sent once per picture.
#[derive(Clone, Copy)]
pub(crate) struct FrameJob {
    pub w: usize,
    pub h: usize,
    pub cw: usize,
    pub ch: usize,
    pub mb_w: usize,
    pub mb_h: usize,
    pub chroma_qp_index_offset: i32,
}

/// One completed macroblock row: the padded-stride pixel rows plus the
/// metadata the boundary-strength derivation reads for this row.
pub(crate) struct RowMsg {
    pub mby: usize,
    /// 16 luma rows, `16 * stride_y` bytes starting at the row's origin-based
    /// offset (includes the horizontal padding between rows).
    pub y: Vec<u8>,
    /// 8 chroma rows each, `8 * stride_c` bytes.
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    /// Luma nnz for the row's four 4x4-grid rows (`4 * grid_w4`).
    pub nnz: Vec<u8>,
    /// L0 motion field rows (`4 * grid_w4` each).
    pub mvx: Vec<i16>,
    pub mvy: Vec<i16>,
    pub refi: Vec<i8>,
    /// L1 motion field rows (B slices only; empty otherwise, so the worker
    /// leaves that row's L1 grid at the reset "no reference" value).
    pub mvx1: Vec<i16>,
    pub mvy1: Vec<i16>,
    pub refi1: Vec<i8>,
    /// Per-macroblock QP for this row (`mb_w`).
    pub qp: Vec<i32>,
    /// Per-macroblock 8x8-transform flag for this row (`mb_w`): the deblocker
    /// leaves the internal 4-sample luma edges of an 8x8-transform MB unfiltered.
    pub tr8x8: Vec<bool>,
    /// Slice deblock control for this row.
    pub filter_on: bool,
    pub alpha_c0_offset: i32,
    pub beta_offset: i32,
}

impl RowMsg {
    fn empty() -> Box<Self> {
        Box::new(Self {
            mby: 0,
            y: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            nnz: Vec::new(),
            mvx: Vec::new(),
            mvy: Vec::new(),
            refi: Vec::new(),
            mvx1: Vec::new(),
            mvy1: Vec::new(),
            refi1: Vec::new(),
            qp: Vec::new(),
            tr8x8: Vec::new(),
            filter_on: true,
            alpha_c0_offset: 0,
            beta_offset: 0,
        })
    }
}

enum ChaseMsg {
    /// Begin a new picture: allocate/resize the shadow and metadata arrays.
    Start(FrameJob),
    Row(Box<RowMsg>),
    /// Picture complete: optionally replicate the padding ring (reference
    /// pictures), then hand the filtered shadow back.
    Finish { replicate: bool },
    /// Donate a plane set (the swapped-out unfiltered buffers) to the
    /// worker's shadow pool.
    Recycle(PlaneSet),
    /// Abandon the in-flight picture (corrupt stream): drop its shadow back
    /// into the pool without replying.
    Abort,
}

/// Decoder-side handle: channels plus the worker join handle.
pub(crate) struct ChaseHandle {
    tx: Option<Sender<ChaseMsg>>,
    reply_rx: Receiver<PlaneSet>,
    row_rx: Receiver<Box<RowMsg>>,
    join: Option<JoinHandle<()>>,
}

impl ChaseHandle {
    pub fn spawn() -> Self {
        let (tx, rx) = channel::<ChaseMsg>();
        let (reply_tx, reply_rx) = channel::<PlaneSet>();
        let (row_tx, row_rx) = channel::<Box<RowMsg>>();
        let join = std::thread::Builder::new()
            .name("h264-deblock".into())
            .spawn(move || worker_loop(rx, reply_tx, row_tx))
            .ok();
        Self {
            tx: Some(tx),
            reply_rx,
            row_rx,
            join,
        }
    }

    fn send(&self, msg: ChaseMsg) -> bool {
        self.tx.as_ref().is_some_and(|tx| tx.send(msg).is_ok())
    }

    pub fn start_frame(&self, job: FrameJob) -> bool {
        self.send(ChaseMsg::Start(job))
    }

    /// A recycled (or fresh) row-message buffer to fill in.
    pub fn row_buf(&self) -> Box<RowMsg> {
        self.row_rx.try_recv().unwrap_or_else(|_| RowMsg::empty())
    }

    pub fn send_row(&self, msg: Box<RowMsg>) -> bool {
        self.send(ChaseMsg::Row(msg))
    }

    /// Completes the picture: waits for the filtered shadow planes.
    pub fn finish_frame(&self, replicate: bool) -> Option<PlaneSet> {
        if !self.send(ChaseMsg::Finish { replicate }) {
            return None;
        }
        self.reply_rx.recv().ok()
    }

    pub fn recycle(&self, planes: PlaneSet) {
        let _ = self.send(ChaseMsg::Recycle(planes));
    }

    pub fn abort_frame(&self) {
        let _ = self.send(ChaseMsg::Abort);
    }
}

impl Drop for ChaseHandle {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// Worker-side state: the shadow frame being assembled plus full-frame copies
/// of the deblock metadata, filled row by row from the messages.
struct WorkerState {
    job: Option<FrameJob>,
    shadow: Option<PlaneSet>,
    pool: Vec<PlaneSet>,
    nnz: Vec<u8>,
    mvx: Vec<i16>,
    mvy: Vec<i16>,
    refi: Vec<i8>,
    mvx1: Vec<i16>,
    mvy1: Vec<i16>,
    refi1: Vec<i8>,
    qp: Vec<i32>,
    tr8x8: Vec<bool>,
}

fn worker_loop(rx: Receiver<ChaseMsg>, reply_tx: Sender<PlaneSet>, row_tx: Sender<Box<RowMsg>>) {
    let mut st = WorkerState {
        job: None,
        shadow: None,
        pool: Vec::new(),
        nnz: Vec::new(),
        mvx: Vec::new(),
        mvy: Vec::new(),
        refi: Vec::new(),
        mvx1: Vec::new(),
        mvy1: Vec::new(),
        refi1: Vec::new(),
        qp: Vec::new(),
        tr8x8: Vec::new(),
    };
    while let Ok(msg) = rx.recv() {
        match msg {
            ChaseMsg::Start(job) => {
                let (_, _, y_sz) = padded_plane_geometry(job.w, job.h);
                let (_, _, c_sz) = padded_plane_geometry(job.cw, job.ch);
                let mut shadow = st.pool.pop().unwrap_or(PlaneSet {
                    y: Vec::new(),
                    u: Vec::new(),
                    v: Vec::new(),
                });
                shadow.y.resize(y_sz, 128);
                shadow.u.resize(c_sz, 128);
                shadow.v.resize(c_sz, 128);
                let grid = job.mb_w * 4 * job.mb_h * 4;
                st.nnz.clear();
                st.nnz.resize(grid, 0);
                st.mvx.clear();
                st.mvx.resize(grid, 0);
                st.mvy.clear();
                st.mvy.resize(grid, 0);
                st.refi.clear();
                st.refi.resize(grid, -1);
                st.mvx1.clear();
                st.mvx1.resize(grid, 0);
                st.mvy1.clear();
                st.mvy1.resize(grid, 0);
                st.refi1.clear();
                st.refi1.resize(grid, -1);
                st.qp.clear();
                st.qp.resize(job.mb_w * job.mb_h, 0);
                st.tr8x8.clear();
                st.tr8x8.resize(job.mb_w * job.mb_h, false);
                st.shadow = Some(shadow);
                st.job = Some(job);
            }
            ChaseMsg::Row(msg) => {
                if let (Some(job), Some(shadow)) = (st.job, st.shadow.as_mut()) {
                    apply_row(
                        &mut st.nnz, &mut st.mvx, &mut st.mvy, &mut st.refi, &mut st.mvx1,
                        &mut st.mvy1, &mut st.refi1, &mut st.qp, &mut st.tr8x8, shadow, &job, &msg,
                    );
                }
                let _ = row_tx.send(msg);
            }
            ChaseMsg::Finish { replicate } => {
                if let (Some(job), Some(mut shadow)) = (st.job.take(), st.shadow.take()) {
                    if replicate {
                        replicate_plane_edges(&mut shadow.y, job.w, job.h);
                        replicate_plane_edges(&mut shadow.u, job.cw, job.ch);
                        replicate_plane_edges(&mut shadow.v, job.cw, job.ch);
                    }
                    if reply_tx.send(shadow).is_err() {
                        return;
                    }
                }
            }
            ChaseMsg::Recycle(planes) => st.pool.push(planes),
            ChaseMsg::Abort => {
                if let Some(shadow) = st.shadow.take() {
                    st.pool.push(shadow);
                }
                st.job = None;
            }
        }
    }
}

/// Copies one macroblock row into the shadow frame and metadata arrays, then
/// filters it (the row above is already in its final filtered state, so this
/// reproduces the sequential whole-frame filter order exactly).
#[allow(clippy::too_many_arguments)]
fn apply_row(
    nnz: &mut [u8],
    mvx: &mut [i16],
    mvy: &mut [i16],
    refi: &mut [i8],
    mvx1: &mut [i16],
    mvy1: &mut [i16],
    refi1: &mut [i8],
    qp: &mut [i32],
    tr8x8: &mut [bool],
    shadow: &mut PlaneSet,
    job: &FrameJob,
    msg: &RowMsg,
) {
    let (stride_y, origin_y, _) = padded_plane_geometry(job.w, job.h);
    let (stride_c, origin_c, _) = padded_plane_geometry(job.cw, job.ch);
    let grid_w4 = job.mb_w * 4;
    let r = msg.mby;

    let y_off = origin_y + r * 16 * stride_y;
    shadow.y[y_off..y_off + 16 * stride_y].copy_from_slice(&msg.y);
    let c_off = origin_c + r * 8 * stride_c;
    shadow.u[c_off..c_off + 8 * stride_c].copy_from_slice(&msg.u);
    shadow.v[c_off..c_off + 8 * stride_c].copy_from_slice(&msg.v);

    let g_off = r * 4 * grid_w4;
    let g_len = 4 * grid_w4;
    nnz[g_off..g_off + g_len].copy_from_slice(&msg.nnz);
    mvx[g_off..g_off + g_len].copy_from_slice(&msg.mvx);
    mvy[g_off..g_off + g_len].copy_from_slice(&msg.mvy);
    refi[g_off..g_off + g_len].copy_from_slice(&msg.refi);
    // L1 motion is present only for B rows; for other rows the grid keeps its
    // reset "no reference" (-1) values, so the boundary strength reduces to the
    // uni-directional case.
    if !msg.refi1.is_empty() {
        mvx1[g_off..g_off + g_len].copy_from_slice(&msg.mvx1);
        mvy1[g_off..g_off + g_len].copy_from_slice(&msg.mvy1);
        refi1[g_off..g_off + g_len].copy_from_slice(&msg.refi1);
    } else {
        refi1[g_off..g_off + g_len].fill(-1);
    }
    qp[r * job.mb_w..(r + 1) * job.mb_w].copy_from_slice(&msg.qp);
    tr8x8[r * job.mb_w..(r + 1) * job.mb_w].copy_from_slice(&msg.tr8x8);

    if msg.filter_on {
        let info = DeblockInfo {
            nnz_y: nnz,
            mvx4: mvx,
            mvy4: mvy,
            ref4: refi,
            mvx4_l1: mvx1,
            mvy4_l1: mvy1,
            ref4_l1: refi1,
            grid_w4,
            mb_qp: qp,
            tr8x8,
            mb_w: job.mb_w,
            chroma_qp_index_offset: job.chroma_qp_index_offset,
            alpha_c0_offset: msg.alpha_c0_offset,
            beta_offset: msg.beta_offset,
        };
        deblock_mb_row(
            &mut shadow.y[origin_y..],
            &mut shadow.u[origin_c..],
            &mut shadow.v[origin_c..],
            stride_y,
            stride_c,
            &info,
            r,
        );
    }
}
