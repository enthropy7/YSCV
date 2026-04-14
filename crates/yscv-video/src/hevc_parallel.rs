// Parallel tile and WPP decode for HEVC.
//
// Separated into its own compilation unit to avoid LLVM codegen
// interference with the recursive `decode_coding_tree_cabac` in the
// sequential path. The `rayon::scope` closures here don't share a
// compilation unit with the main decode loop.

use super::hevc_decoder::{DecodedCu, HevcPps, HevcSliceType, HevcSps, pps_tile_rects};
use super::hevc_filter::SaoParams;
use super::hevc_inter::{HevcMvField, SendMutPtr};
use super::hevc_syntax::{self, HevcSliceCabacState, TileBounds};

/// Decode CTUs in parallel across tiles using rayon.
///
/// Each tile gets its own CABAC decoder starting at the tile's entry-point
/// byte offset. Tiles write to disjoint rectangles of the shared recon
/// buffers via raw pointers (safety: rectangles are non-overlapping by
/// tile geometry).
pub(crate) fn decode_tiles_parallel(
    payload: &[u8],
    sps: &HevcSps,
    pps: &HevcPps,
    slice_type: HevcSliceType,
    slice_qp: i32,
    w: usize,
    h: usize,
    ctu_size_log2: u8,
    max_depth: u8,
    entry_point_offsets: &[usize],
    recon_luma: &mut Vec<i16>,
    recon_cb: &mut Vec<i16>,
    recon_cr: &mut Vec<i16>,
    cus: &mut Vec<DecodedCu>,
    dpb: &super::hevc_inter::HevcDpb,
    mv_field: &mut Vec<HevcMvField>,
    sao_list: &mut [SaoParams],
) {
    let ctu_size = 1usize << ctu_size_log2;
    let pic_w_ctb = w.div_ceil(ctu_size) as u32;
    let pic_h_ctb = h.div_ceil(ctu_size) as u32;
    let tile_rects = pps_tile_rects(pps, pic_w_ctb, pic_h_ctb);

    if tile_rects.len() <= 1 || entry_point_offsets.is_empty() {
        // Single tile or no entry points — fall back to sequential in caller
        return;
    }

    // Build absolute byte offsets for each tile's CABAC start
    let mut tile_starts = Vec::with_capacity(tile_rects.len());
    tile_starts.push(0usize); // first tile at offset 0
    let mut acc = 0usize;
    for &off in entry_point_offsets.iter().take(tile_rects.len() - 1) {
        acc += off;
        tile_starts.push(acc);
    }

    // Raw pointers for shared buffer access
    let luma_ptr = SendMutPtr(recon_luma.as_mut_ptr());
    let cb_ptr = SendMutPtr(recon_cb.as_mut_ptr());
    let cr_ptr = SendMutPtr(recon_cr.as_mut_ptr());
    let mv_ptr = SendMutPtr(mv_field.as_mut_ptr());
    let luma_len = recon_luma.len();
    let cb_len = recon_cb.len();
    let cr_len = recon_cr.len();
    let mv_len = mv_field.len();

    let tile_results = std::sync::Mutex::new(vec![
        (
            Vec::<DecodedCu>::new(),
            Vec::<(usize, SaoParams)>::new()
        );
        tile_rects.len()
    ]);

    rayon::scope(|s| {
        for (tile_idx, tile_rect) in tile_rects.iter().enumerate() {
            let byte_offset = tile_starts.get(tile_idx).copied().unwrap_or(0);
            let results_ref = &tile_results;

            // Force closure to capture SendMutPtr wrappers (Send+Sync),
            // not raw pointer fields (edition 2024 precise captures).
            let lp = luma_ptr;
            let cp = cb_ptr;
            let crp = cr_ptr;
            let mp = mv_ptr;
            s.spawn(move |_| {
                // Extract raw pointers via method to avoid edition-2024
                // precise field capture of the non-Send `*mut T` field.
                let lp_raw = lp.as_ptr();
                let cp_raw = cp.as_ptr();
                let crp_raw = crp.as_ptr();
                let mp_raw = mp.as_ptr();
                // SAFETY: each tile writes to a disjoint CTU rectangle.
                // SendMutPtr guarantees Send+Sync; slice lifetimes bounded by scope.
                let tile_recon_luma = unsafe { std::slice::from_raw_parts_mut(lp_raw, luma_len) };
                let tile_recon_cb = unsafe { std::slice::from_raw_parts_mut(cp_raw, cb_len) };
                let tile_recon_cr = unsafe { std::slice::from_raw_parts_mut(crp_raw, cr_len) };
                let tile_mv_field = unsafe { std::slice::from_raw_parts_mut(mp_raw, mv_len) };
                let mut tile_cabac =
                    HevcSliceCabacState::new_at_offset(payload, slice_qp, byte_offset);

                let mut tile_cus = Vec::new();
                let mut tile_saos = Vec::new();

                let tile_bounds = TileBounds {
                    col_start: tile_rect.col_start as usize * ctu_size,
                    col_end: tile_rect.col_end as usize * ctu_size,
                    row_start: tile_rect.row_start as usize * ctu_size,
                    row_end: tile_rect.row_end as usize * ctu_size,
                };

                let mut ctu_y = tile_rect.row_start as usize * ctu_size;
                let tile_y_end =
                    ctu_y + (tile_rect.row_end - tile_rect.row_start) as usize * ctu_size;
                while ctu_y < tile_y_end.min(h) {
                    let mut ctu_x = tile_rect.col_start as usize * ctu_size;
                    let tile_x_end =
                        ctu_x + (tile_rect.col_end - tile_rect.col_start) as usize * ctu_size;
                    while ctu_x < tile_x_end.min(w) {
                        if tile_cabac.cabac.bytes_remaining() < 1 {
                            break;
                        }
                        if sps.sample_adaptive_offset_enabled {
                            let left_avail = ctu_x > tile_bounds.col_start;
                            let above_avail = ctu_y > tile_bounds.row_start;
                            let sao = super::hevc_filter::parse_sao_params(
                                &mut tile_cabac.cabac,
                                left_avail,
                                above_avail,
                            );
                            let cy = ctu_y / ctu_size;
                            let cx = ctu_x / ctu_size;
                            let idx = cy * pic_w_ctb as usize + cx;
                            tile_saos.push((idx, sao));
                        }

                        hevc_syntax::decode_coding_tree_cabac(
                            &mut tile_cabac,
                            ctu_x,
                            ctu_y,
                            ctu_size_log2,
                            0,
                            max_depth,
                            sps,
                            pps,
                            slice_type,
                            w,
                            h,
                            tile_recon_luma,
                            tile_recon_cb,
                            tile_recon_cr,
                            &mut tile_cus,
                            dpb,
                            tile_mv_field,
                            None,
                        );
                        ctu_x += ctu_size;
                    }
                    ctu_y += ctu_size;
                }

                // Prevent Vec wrappers from freeing the shared buffers
                // Slices are borrowed, no ownership to release.

                if let Ok(mut results_vec) = results_ref.lock()
                    && tile_idx < results_vec.len()
                {
                    results_vec[tile_idx] = (tile_cus, tile_saos);
                }
            });
        }
    });

    // Merge per-tile results
    if let Ok(results_vec) = tile_results.into_inner() {
        for (tile_cus, tile_saos) in results_vec {
            cus.extend(tile_cus);
            for (idx, sao) in tile_saos {
                if idx < sao_list.len() {
                    sao_list[idx] = sao;
                }
            }
        }
    }
}

/// Decode CTU rows in parallel using WPP (Wavefront Parallel Processing).
///
/// Row R starts after row R-1 completes CTU(1) and publishes its CABAC
/// context snapshot. Each row writes to a disjoint horizontal stripe of
/// the recon buffer.
pub(crate) fn decode_wpp_parallel(
    payload: &[u8],
    sps: &HevcSps,
    pps: &HevcPps,
    slice_type: HevcSliceType,
    slice_qp: i32,
    w: usize,
    h: usize,
    ctu_size_log2: u8,
    max_depth: u8,
    entry_point_offsets: &[usize],
    recon_luma: &mut Vec<i16>,
    recon_cb: &mut Vec<i16>,
    recon_cr: &mut Vec<i16>,
    cus: &mut Vec<DecodedCu>,
    dpb: &super::hevc_inter::HevcDpb,
    mv_field: &mut Vec<HevcMvField>,
    sao_list: &mut [SaoParams],
) {
    let ctu_size = 1usize << ctu_size_log2;
    let pic_h_ctb = h.div_ceil(ctu_size);
    let pic_w_ctb = w.div_ceil(ctu_size);

    if pic_h_ctb <= 2 || entry_point_offsets.is_empty() {
        return;
    }

    // Build absolute byte offsets per row
    let mut row_starts = Vec::with_capacity(pic_h_ctb);
    row_starts.push(0usize);
    let mut acc = 0usize;
    for &off in entry_point_offsets.iter().take(pic_h_ctb - 1) {
        acc += off;
        row_starts.push(acc);
    }

    let luma_ptr = SendMutPtr(recon_luma.as_mut_ptr());
    let cb_ptr = SendMutPtr(recon_cb.as_mut_ptr());
    let cr_ptr = SendMutPtr(recon_cr.as_mut_ptr());
    let mv_ptr = SendMutPtr(mv_field.as_mut_ptr());
    let luma_len = recon_luma.len();
    let cb_len = recon_cb.len();
    let cr_len = recon_cr.len();
    let mv_len = mv_field.len();

    use std::sync::{Arc, Condvar, Mutex};
    type Snapshot = Vec<super::hevc_cabac::ContextModel>;

    // Per-row barrier: row R-1 publishes its CABAC snapshot after CTU(1)
    let barriers: Vec<Arc<(Mutex<Option<Snapshot>>, Condvar)>> = (0..pic_h_ctb)
        .map(|_| Arc::new((Mutex::new(None), Condvar::new())))
        .collect();

    let row_results = Mutex::new(vec![
        (
            Vec::<DecodedCu>::new(),
            Vec::<(usize, SaoParams)>::new()
        );
        pic_h_ctb
    ]);

    rayon::scope(|s| {
        for row in 0..pic_h_ctb {
            let barrier_prev = if row > 0 {
                Some(Arc::clone(&barriers[row - 1]))
            } else {
                None
            };
            let barrier_cur = Arc::clone(&barriers[row]);
            let byte_offset = row_starts.get(row).copied().unwrap_or(0);
            let results_ref = &row_results;
            let lp = luma_ptr;
            let cp = cb_ptr;
            let crp = cr_ptr;
            let mp = mv_ptr;

            s.spawn(move |_| {
                // Wait for previous row's context snapshot
                let mut row_cabac = if let Some(ref prev) = barrier_prev {
                    let (lock, cvar) = prev.as_ref();
                    let guard = lock.lock().expect("barrier lock");
                    let guard = cvar
                        .wait_while(guard, |snap| snap.is_none())
                        .expect("barrier wait");
                    let snap = guard.as_ref().expect("snapshot must exist");
                    let mut state =
                        HevcSliceCabacState::new_at_offset(payload, slice_qp, byte_offset);
                    state.restore_contexts(snap);
                    state
                } else {
                    HevcSliceCabacState::new(payload, slice_qp)
                };

                // SAFETY: each row writes to a disjoint horizontal stripe.
                // SendMutPtr guarantees Send+Sync; slice lifetimes bounded by scope.
                let lp_raw = lp.as_ptr();
                let cp_raw = cp.as_ptr();
                let crp_raw = crp.as_ptr();
                let mp_raw = mp.as_ptr();
                let row_recon_luma = unsafe { std::slice::from_raw_parts_mut(lp_raw, luma_len) };
                let row_recon_cb = unsafe { std::slice::from_raw_parts_mut(cp_raw, cb_len) };
                let row_recon_cr = unsafe { std::slice::from_raw_parts_mut(crp_raw, cr_len) };
                let row_mv_field = unsafe { std::slice::from_raw_parts_mut(mp_raw, mv_len) };

                let _tile_bounds = TileBounds::full(w, h);
                let mut row_cus = Vec::new();
                let mut row_saos = Vec::new();
                let ctu_y = row * ctu_size;
                let mut ctu_x = 0;
                let mut ctu_idx = 0;

                while ctu_x < w {
                    if row_cabac.cabac.bytes_remaining() < 1 {
                        break;
                    }
                    if sps.sample_adaptive_offset_enabled {
                        let left_avail = ctu_x > 0;
                        let above_avail = ctu_y > 0;
                        let sao = super::hevc_filter::parse_sao_params(
                            &mut row_cabac.cabac,
                            left_avail,
                            above_avail,
                        );
                        let idx = row * pic_w_ctb + ctu_x / ctu_size;
                        row_saos.push((idx, sao));
                    }

                    hevc_syntax::decode_coding_tree_cabac(
                        &mut row_cabac,
                        ctu_x,
                        ctu_y,
                        ctu_size_log2,
                        0,
                        max_depth,
                        sps,
                        pps,
                        slice_type,
                        w,
                        h,
                        row_recon_luma,
                        row_recon_cb,
                        row_recon_cr,
                        &mut row_cus,
                        dpb,
                        row_mv_field,
                        None,
                    );

                    // After CTU(1), publish snapshot for next row
                    if ctu_idx == 1 {
                        let snap = row_cabac.snapshot_contexts();
                        let (lock, cvar) = barrier_cur.as_ref();
                        *lock.lock().expect("snapshot lock") = Some(snap);
                        cvar.notify_all();
                    }

                    ctu_x += ctu_size;
                    ctu_idx += 1;
                }

                // If row had only 1 CTU, publish snapshot anyway
                if ctu_idx == 1 {
                    let snap = row_cabac.snapshot_contexts();
                    let (lock, cvar) = barrier_cur.as_ref();
                    *lock.lock().expect("snapshot lock") = Some(snap);
                    cvar.notify_all();
                }

                if let Ok(mut results_vec) = results_ref.lock()
                    && row < results_vec.len()
                {
                    results_vec[row] = (row_cus, row_saos);
                }
            });
        }
    });

    if let Ok(results_vec) = row_results.into_inner() {
        for (row_cus, row_saos) in results_vec {
            cus.extend(row_cus);
            for (idx, sao) in row_saos {
                if idx < sao_list.len() {
                    sao_list[idx] = sao;
                }
            }
        }
    }
}
