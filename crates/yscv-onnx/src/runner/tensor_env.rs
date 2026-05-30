//! TensorEnv: the id-indexed execution environment (slots, layout flags,
//! weight references) and its construction from a model + runtime inputs.

use super::*;

/// A tensor environment backed by a `Vec<Option<Tensor>>` for O(1) lookups
/// by integer index. Tensor names are mapped to dense integer IDs during
/// construction, eliminating string hashing in the hot execution loop.
///
/// Model initializers (weights) are referenced without cloning. Only when
/// mutation is needed (get_mut/remove) is a clone-on-write performed.
pub(crate) struct TensorEnv<'m, 'i> {
    static_name_to_id: &'m HashMap<String, usize>,
    pub(crate) use_counts_by_id: &'m [usize],
    dynamic_name_to_id: HashMap<String, usize>,
    slots: Vec<Option<Tensor>>,
    quant_slots: Vec<Option<QuantTensor>>,
    /// Per-slot flag: true if the tensor is stored in NHWC layout.
    nhwc_flags: Vec<bool>,
    /// Per-slot NCHWc block size (0 = not NCHWc-resident; 8 or 16 = block
    /// size). Used by the M3 enclave to skip nhwc↔nchwc conversions when
    /// adjacent FusedDwPw_AVX512 ops share NCHWc layout.
    nchwc_block_flags: Vec<u8>,
    /// Slot IDs whose tensors have been pre-permuted from OIHW to KHWC.
    khwc_weights: &'m HashSet<usize>,
    /// Slot IDs whose depthwise weights were pre-permuted [O,1,KH,KW] → [KH,KW,C,dm].
    dw_khwc_weights: &'m HashSet<usize>,
    /// Slot IDs whose grouped-conv weights were pre-permuted [O,I/G,KH,KW] → [O,KH,KW,I/G].
    group_khwc_weights: &'m HashSet<usize>,
    /// Pre-packed blocked-GEMM B-matrices keyed by weight tensor name. Built
    /// once at model load (`build_runtime_index`). Hot path looks up by the
    /// Conv/MatMul's weight input name and hands the shared `Arc<PackedB>`
    /// straight into the GEMM layer, skipping fingerprint cache + repack.
    prepacked_weights: &'m HashMap<String, std::sync::Arc<yscv_kernels::PackedB>>,
    /// Symmetric-int8 RHS matrices packed once at model load for
    /// QLinearConv/QLinearMatMul/MatMulInteger fast paths.
    prepacked_i8_weights: &'m HashMap<String, std::sync::Arc<yscv_kernels::PackedI8B>>,
    /// QLinear depthwise 3x3/5x5 weights packed once as KHWC i8.
    prepacked_i8_depthwise: &'m HashMap<String, std::sync::Arc<Vec<i8>>>,
    scratch_i8_a: Vec<i8>,
    scratch_i32: Vec<i32>,
    /// Counter for dynamically allocated temporary names that were not in
    /// the pre-built mapping (e.g., "__qa", "__qb_mat").
    next_dynamic: usize,
    /// Reference to model initializers for zero-copy weight access.
    initializers: &'m HashMap<String, Tensor>,
    /// Side-table of MatMul/Gemm weights packed as symmetric INT4 with
    /// per-group fp32 scales. Hot-path GEMV dispatch consults this map
    /// before falling back to the fp32 path.
    pub(crate) packed_int4_weights: &'m HashMap<String, crate::quantize::PackedInt4Weight>,
    /// Shared reference to the loader-computed set of Reshape output tensor
    /// names whose single consumer is a `FusedTransposeMatMul` (perm=[0,2,1]).
    /// Used by
    /// `execute_node_with_layout_kind_inner` to gate the NHWC-passthrough
    /// fast path.
    pub(crate) reshape_nhwc_passthrough_safe: &'m HashSet<String>,
    /// Optional borrowed runtime inputs for zero-copy inference entry.
    runtime_inputs: Option<RuntimeInputs<'i>>,
}

#[derive(Clone, Copy)]
pub(crate) enum RuntimeInputs<'i> {
    Map(&'i HashMap<String, Tensor>),
    Slice(&'i [(&'i str, &'i Tensor)]),
}

impl<'m, 'i> TensorEnv<'m, 'i> {
    /// Build from the model, pre-allocating a slot for every known tensor name.
    /// Holds a reference to model initializers for zero-copy weight access.
    pub(crate) fn from_model(model: &'m OnnxModel) -> Self {
        Self::from_model_with_inputs(model, None)
    }

    /// Build from the model with optional borrowed runtime inputs.
    pub(crate) fn from_model_with_inputs(
        model: &'m OnnxModel,
        runtime_inputs: Option<RuntimeInputs<'i>>,
    ) -> Self {
        let num_slots = model.runtime_index.name_to_id.len();
        TensorEnv {
            next_dynamic: num_slots,
            static_name_to_id: &model.runtime_index.name_to_id,
            use_counts_by_id: &model.runtime_index.use_counts_by_id,
            dynamic_name_to_id: HashMap::new(),
            slots: vec![None; num_slots],
            quant_slots: vec![None; num_slots],
            nhwc_flags: vec![false; num_slots],
            nchwc_block_flags: vec![0u8; num_slots],
            khwc_weights: &model.runtime_index.khwc_weight_ids,
            dw_khwc_weights: &model.runtime_index.dw_khwc_weight_ids,
            group_khwc_weights: &model.runtime_index.group_khwc_weight_ids,
            prepacked_weights: &model.runtime_index.prepacked_weights,
            prepacked_i8_weights: &model.runtime_index.prepacked_i8_weights,
            prepacked_i8_depthwise: &model.runtime_index.prepacked_i8_depthwise,
            scratch_i8_a: Vec::new(),
            scratch_i32: Vec::new(),
            initializers: &model.initializers,
            packed_int4_weights: &model.packed_int4_weights,
            reshape_nhwc_passthrough_safe: &model.runtime_index.reshape_nhwc_passthrough_safe,
            runtime_inputs,
        }
    }

    /// Returns the pre-packed B-matrix for the given weight tensor name, if
    /// one was built at model-load time. Callers can hand the resulting
    /// `&PackedB` to the blocked-GEMM fast path.
    #[inline]
    pub(crate) fn prepacked_b(&self, weight_name: &str) -> Option<&yscv_kernels::PackedB> {
        self.prepacked_weights
            .get(weight_name)
            .map(|arc| arc.as_ref())
    }

    #[inline]
    pub(crate) fn prepacked_i8_b(&self, weight_name: &str) -> Option<&yscv_kernels::PackedI8B> {
        self.prepacked_i8_weights
            .get(weight_name)
            .map(|arc| arc.as_ref())
    }

    #[inline]
    pub(crate) fn prepacked_i8_depthwise(
        &self,
        weight_name: &str,
    ) -> Option<std::sync::Arc<Vec<i8>>> {
        self.prepacked_i8_depthwise
            .get(weight_name)
            .map(std::sync::Arc::clone)
    }

    #[inline]
    pub(crate) fn take_i8_scratch_a(&mut self, len: usize) -> Vec<i8> {
        let mut scratch = std::mem::take(&mut self.scratch_i8_a);
        scratch.resize(len, 0);
        scratch
    }

    #[inline]
    pub(crate) fn put_i8_scratch_a(&mut self, mut scratch: Vec<i8>) {
        scratch.clear();
        self.scratch_i8_a = scratch;
    }

    #[inline]
    pub(crate) fn take_i32_scratch(&mut self, len: usize) -> Vec<i32> {
        let mut scratch = std::mem::take(&mut self.scratch_i32);
        scratch.resize(len, 0);
        scratch
    }

    #[inline]
    pub(crate) fn put_i32_scratch(&mut self, mut scratch: Vec<i32>) {
        scratch.clear();
        self.scratch_i32 = scratch;
    }

    #[inline]
    fn runtime_input(&self, name: &str) -> Option<&Tensor> {
        match self.runtime_inputs {
            Some(RuntimeInputs::Map(inputs)) => inputs.get(name),
            Some(RuntimeInputs::Slice(inputs)) => inputs
                .iter()
                .find_map(|(n, t)| if *n == name { Some(*t) } else { None }),
            None => None,
        }
    }

    #[inline]
    pub(crate) fn resolve_id(&self, name: &str) -> Option<usize> {
        self.dynamic_name_to_id
            .get(name)
            .copied()
            .or_else(|| self.static_name_to_id.get(name).copied())
    }

    /// Look up a tensor by name. Falls back to model initializers if the
    /// slot is empty (zero-copy access to weights).
    #[inline]
    pub(crate) fn get(&self, name: &str) -> Option<&Tensor> {
        let id = self.resolve_id(name)?;
        self.slots[id]
            .as_ref()
            .or_else(|| self.initializers.get(name))
            .or_else(|| self.runtime_input(name))
    }

    /// Direct slot removal by pre-resolved ID — O(1), no HashMap lookup.
    #[inline]
    pub(crate) fn remove_by_id(&mut self, id: usize) {
        if id < self.slots.len() {
            self.slots[id] = None;
            if id < self.quant_slots.len() {
                self.quant_slots[id] = None;
            }
            if id < self.nhwc_flags.len() {
                self.nhwc_flags[id] = false;
            }
            if id < self.nchwc_block_flags.len() {
                self.nchwc_block_flags[id] = 0;
            }
        }
    }

    /// Insert a tensor by name. If the name is unknown, a new slot is
    /// allocated dynamically (this handles temporary names created by
    /// quantization ops, etc.).
    #[inline]
    pub(crate) fn insert(&mut self, name: String, tensor: Tensor) {
        crate::quantize::calibrate::record_activation(&name, &tensor);
        if let Some(id) = self.resolve_id(&name) {
            self.slots[id] = Some(tensor);
            if id < self.quant_slots.len() {
                self.quant_slots[id] = None;
            }
            if id < self.nhwc_flags.len() {
                self.nhwc_flags[id] = false;
            }
            if id < self.nchwc_block_flags.len() {
                self.nchwc_block_flags[id] = 0;
            }
        } else {
            let id = self.next_dynamic;
            self.next_dynamic += 1;
            self.dynamic_name_to_id.insert(name, id);
            self.slots.push(Some(tensor));
            self.quant_slots.push(None);
            self.nhwc_flags.push(false);
            self.nchwc_block_flags.push(0);
        }
    }

    /// Reverse name lookup for a slot id. O(N) over the static map; only
    /// invoked when calibration is active. Returns the static name (with
    /// the model's `'m` lifetime) if present, falling back to dynamic
    /// names (temporaries like `__qa`).
    fn name_for_id(&self, id: usize) -> Option<&'m str> {
        if let Some(name) = self
            .static_name_to_id
            .iter()
            .find_map(|(n, &i)| if i == id { Some(n.as_str()) } else { None })
        {
            return Some(name);
        }
        // Dynamic names live in `self.dynamic_name_to_id` (owned by the
        // env, lifetime 'static within this env). They are short-lived
        // temporaries; record under their name so calibrators can filter.
        // We can't return the `&str` here without borrowing self mutably
        // through the dynamic map, so skip dynamic-name records — they
        // are temporaries (`__qa`, `__qb_mat`) of no interest for PTQ
        // anyway.
        let _ = id;
        None
    }

    /// direct slot insertion by pre-resolved ID, skipping
    /// the HashMap lookup inside `resolve_id`. Caller (runner hot path)
    /// passes the slot ID from `node_output_ids` table. Mirrors
    /// `remove_by_id` for output path.
    #[inline]
    pub(crate) fn insert_by_id(&mut self, id: usize, tensor: Tensor) {
        if crate::quantize::calibrate::calibration_active()
            && let Some(name) = self.name_for_id(id)
        {
            // Static map has lifetime `'m`, so the &str outlives the
            // following mutable slot write under NLL.
            crate::quantize::calibrate::record_activation(name, &tensor);
        }
        if id < self.slots.len() {
            self.slots[id] = Some(tensor);
            if id < self.quant_slots.len() {
                self.quant_slots[id] = None;
            }
            if id < self.nhwc_flags.len() {
                self.nhwc_flags[id] = false;
            }
            if id < self.nchwc_block_flags.len() {
                self.nchwc_block_flags[id] = 0;
            }
        } else {
            // Grow the slots vec if needed. This should not happen in
            // practice — node_output_ids is built at load time from
            // name_to_id which sized slots at construction.
            while self.slots.len() <= id {
                self.slots.push(None);
                self.quant_slots.push(None);
                self.nhwc_flags.push(false);
                self.nchwc_block_flags.push(0);
            }
            self.slots[id] = Some(tensor);
            self.quant_slots[id] = None;
        }
    }

    #[inline]
    pub(crate) fn insert_quant_i8(&mut self, name: String, tensor: QuantTensor) {
        note_quant_i8_store();
        if let Some(id) = self.resolve_id(&name) {
            self.slots[id] = None;
            if id >= self.quant_slots.len() {
                self.quant_slots.resize_with(id + 1, || None);
            }
            if id >= self.nhwc_flags.len() {
                self.nhwc_flags.resize(id + 1, false);
            }
            if id >= self.nchwc_block_flags.len() {
                self.nchwc_block_flags.resize(id + 1, 0);
            }
            self.nhwc_flags[id] = tensor.nhwc;
            self.nchwc_block_flags[id] = 0;
            self.quant_slots[id] = Some(tensor);
        } else {
            let id = self.next_dynamic;
            self.next_dynamic += 1;
            self.dynamic_name_to_id.insert(name, id);
            self.slots.push(None);
            self.nhwc_flags.push(tensor.nhwc);
            self.nchwc_block_flags.push(0);
            self.quant_slots.push(Some(tensor));
        }
    }

    #[inline]
    pub(crate) fn get_quant_i8(&self, name: &str) -> Option<&QuantTensor> {
        let id = self.resolve_id(name)?;
        self.quant_slots.get(id).and_then(Option::as_ref)
    }

    #[inline]
    pub(crate) fn take_quant_i8(&mut self, name: &str) -> Option<QuantTensor> {
        let id = self.resolve_id(name)?;
        if id < self.nhwc_flags.len() {
            self.nhwc_flags[id] = false;
        }
        if id < self.nchwc_block_flags.len() {
            self.nchwc_block_flags[id] = 0;
        }
        self.quant_slots.get_mut(id)?.take()
    }

    #[inline]
    pub(crate) fn static_use_count(&self, name: &str) -> usize {
        self.resolve_id(name)
            .and_then(|id| self.use_counts_by_id.get(id).copied())
            .unwrap_or(0)
    }

    pub(crate) fn materialize_quant_i8_raw(&mut self, name: &str) -> Result<(), OnnxError> {
        let Some(q) = self.get_quant_i8(name).cloned() else {
            return Ok(());
        };
        note_quant_i8_materialize();
        let data: Vec<f32> = q.data.iter().map(|&v| v as f32).collect();
        let out = Tensor::from_vec(q.shape, data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        self.insert(name.to_string(), out);
        if q.nhwc {
            self.mark_nhwc(name);
        }
        Ok(())
    }

    /// direct slot NHWC mark by ID.
    #[inline]
    pub(crate) fn mark_nhwc_by_id(&mut self, id: usize) {
        if id < self.nhwc_flags.len() {
            self.nhwc_flags[id] = true;
        }
        if id < self.nchwc_block_flags.len() {
            self.nchwc_block_flags[id] = 0;
        }
    }

    /// Mark the tensor at `name` as being in NCHWc layout with the given
    /// block size. Also clears the NHWC flag (a tensor is either NHWC or
    /// NCHWc, not both). Used by the M3 enclave to leave NCHWc outputs
    /// for downstream FusedDwPw_AVX512 consumers.
    #[inline]
    pub(crate) fn mark_nchwc(&mut self, name: &str, block: u8) {
        if let Some(id) = self.resolve_id(name) {
            if id < self.nchwc_block_flags.len() {
                self.nchwc_block_flags[id] = block;
            }
            if id < self.nhwc_flags.len() {
                self.nhwc_flags[id] = false;
            }
        }
    }

    /// Returns the NCHWc block size if the tensor at `name` is stored in
    /// NCHWc layout, or `None` otherwise.
    #[inline]
    pub(crate) fn nchwc_block(&self, name: &str) -> Option<u8> {
        let id = self.resolve_id(name)?;
        let b = self.nchwc_block_flags.get(id).copied().unwrap_or(0);
        if b == 0 { None } else { Some(b) }
    }

    /// Get a mutable reference to a tensor by name.
    /// Clone-on-write: if the tensor is only in initializers, clone it into
    /// the slot first.
    #[inline]
    pub(crate) fn get_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        let id = self.resolve_id(name)?;
        if self.slots[id].is_none()
            && let Some(t) = self.initializers.get(name)
        {
            self.slots[id] = Some(t.clone());
        } else if self.slots[id].is_none()
            && let Some(t) = self.runtime_input(name)
        {
            self.slots[id] = Some(t.clone());
        }
        self.slots[id].as_mut()
    }

    /// Fork env for tower-parallel execution. Shares all read-only metadata
    /// (initializers, name tables, pre-permute flags) and clones the slot
    /// state so each branch has its own mutable view. Only live tensors are
    /// copied — Tensor is `Arc<AlignedVec>`, so the clone is a refcount bump,
    /// not a data copy.
    pub(crate) fn fork(&self) -> Self {
        TensorEnv {
            static_name_to_id: self.static_name_to_id,
            use_counts_by_id: self.use_counts_by_id,
            dynamic_name_to_id: self.dynamic_name_to_id.clone(),
            slots: self.slots.clone(),
            quant_slots: self.quant_slots.clone(),
            nhwc_flags: self.nhwc_flags.clone(),
            nchwc_block_flags: self.nchwc_block_flags.clone(),
            khwc_weights: self.khwc_weights,
            dw_khwc_weights: self.dw_khwc_weights,
            group_khwc_weights: self.group_khwc_weights,
            prepacked_weights: self.prepacked_weights,
            prepacked_i8_weights: self.prepacked_i8_weights,
            prepacked_i8_depthwise: self.prepacked_i8_depthwise,
            scratch_i8_a: Vec::new(),
            scratch_i32: Vec::new(),
            next_dynamic: self.next_dynamic,
            initializers: self.initializers,
            packed_int4_weights: self.packed_int4_weights,
            reshape_nhwc_passthrough_safe: self.reshape_nhwc_passthrough_safe,
            runtime_inputs: self.runtime_inputs,
        }
    }

    /// Transfer new tensors from `other` back into `self`. A slot that is
    /// Some in `other` but None (or stale) in `self` gets moved over; other
    /// slots in `self` stay intact. Used after tower-parallel execution to
    /// reunite branches into one env before running the merge tail.
    pub(crate) fn merge_from(&mut self, mut other: TensorEnv<'m, 'i>) {
        let n = self.slots.len().min(other.slots.len());
        for id in 0..n {
            if let Some(t) = other.slots[id].take() {
                self.slots[id] = Some(t);
                if id < self.quant_slots.len() {
                    self.quant_slots[id] = None;
                }
                if id < self.nhwc_flags.len() && id < other.nhwc_flags.len() {
                    self.nhwc_flags[id] = other.nhwc_flags[id];
                }
                if id < self.nchwc_block_flags.len() && id < other.nchwc_block_flags.len() {
                    self.nchwc_block_flags[id] = other.nchwc_block_flags[id];
                }
            } else if id < other.quant_slots.len()
                && let Some(q) = other.quant_slots[id].take()
            {
                if id >= self.quant_slots.len() {
                    self.quant_slots.resize_with(id + 1, || None);
                }
                self.quant_slots[id] = Some(q);
                if id < self.slots.len() {
                    self.slots[id] = None;
                }
                if id < self.nhwc_flags.len() && id < other.nhwc_flags.len() {
                    self.nhwc_flags[id] = other.nhwc_flags[id];
                }
                if id < self.nchwc_block_flags.len() && id < other.nchwc_block_flags.len() {
                    self.nchwc_block_flags[id] = other.nchwc_block_flags[id];
                }
            }
        }
        // Any dynamic names added in the forked env need to be propagated so
        // the merge-branch nodes can resolve them.
        for (name, id) in other.dynamic_name_to_id.drain() {
            self.dynamic_name_to_id.insert(name, id);
        }
        self.next_dynamic = self.next_dynamic.max(other.next_dynamic);
    }

    /// Remove a tensor by name (sets the slot to `None`).
    /// If the tensor is only in initializers, clone and return it.
    #[inline]
    pub(crate) fn remove(&mut self, name: &str) -> Option<Tensor> {
        let id = self.resolve_id(name)?;
        if id < self.nhwc_flags.len() {
            self.nhwc_flags[id] = false;
        }
        if id < self.nchwc_block_flags.len() {
            self.nchwc_block_flags[id] = 0;
        }
        if id < self.quant_slots.len() {
            self.quant_slots[id] = None;
        }
        self.slots[id]
            .take()
            .or_else(|| self.initializers.get(name).cloned())
            .or_else(|| self.runtime_input(name).cloned())
    }

    /// Returns true if the tensor at `name` is stored in NHWC layout.
    #[inline]
    pub(crate) fn is_nhwc(&self, name: &str) -> bool {
        self.resolve_id(name)
            .map(|id| self.nhwc_flags.get(id).copied().unwrap_or(false))
            .unwrap_or(false)
    }

    /// Mark the tensor at `name` as being in NHWC layout.
    #[inline]
    pub(crate) fn mark_nhwc(&mut self, name: &str) {
        if let Some(id) = self.resolve_id(name)
            && id < self.nhwc_flags.len()
        {
            self.nhwc_flags[id] = true;
        }
    }

    /// Returns true if the tensor has been pre-permuted to KHWC format.
    #[inline]
    pub(crate) fn is_khwc_weight(&self, name: &str) -> bool {
        self.resolve_id(name)
            .is_some_and(|id| self.khwc_weights.contains(&id))
    }

    /// Returns true if the depthwise conv weight has been pre-permuted to
    /// `[KH, KW, C, depth_multiplier]` format at load time.
    #[inline]
    pub(crate) fn is_dw_khwc_weight(&self, name: &str) -> bool {
        self.resolve_id(name)
            .is_some_and(|id| self.dw_khwc_weights.contains(&id))
    }

    /// Returns true if grouped conv weight has been pre-permuted to
    /// `[O, KH, KW, I/G]` format at load time.
    #[inline]
    pub(crate) fn is_group_khwc_weight(&self, name: &str) -> bool {
        self.resolve_id(name)
            .is_some_and(|id| self.group_khwc_weights.contains(&id))
    }

    /// Create a zero-copy alias: remap `alias_name` to the same slot as
    /// `target_name`. No tensor data is cloned — both names point to the
    /// identical storage. Safe because ONNX outputs are write-once.
    #[inline]
    pub(crate) fn alias(&mut self, alias_name: &str, target_name: &str) {
        let target_id = match self.resolve_id(target_name) {
            Some(id) => id,
            None => return,
        };
        // If the target lives only in `initializers`, materialize it into the
        // slot so the alias name can resolve via `get()` — which otherwise
        // would fall back to `initializers.get(alias_name)` and miss.
        if self.slots[target_id].is_none()
            && let Some(t) = self.initializers.get(target_name)
        {
            self.slots[target_id] = Some(t.clone());
        } else if self.slots[target_id].is_none()
            && let Some(t) = self.runtime_input(target_name)
        {
            self.slots[target_id] = Some(t.clone());
        }
        // Point alias_name to the same slot ID as target_name.
        self.dynamic_name_to_id
            .insert(alias_name.to_string(), target_id);
    }
}
