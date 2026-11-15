/*
 * Copyright 2026 Teenygrad
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! FoundationStereo's `FlashMultiheadAttention` (`core/submodule.py`), as it actually
//! computes:
//!
//! ```python
//! Q = q_proj(x).view(B*H*W, D, heads, head_dim)          # likewise K, V
//! out = F.scaled_dot_product_attention(Q, K, V)          # attends over dim -2: heads
//! ```
//!
//! `scaled_dot_product_attention` treats all but the last two axes as batch, so with
//! no transpose to `(.., heads, D, head_dim)` each voxel's `heads` vectors of
//! `head_dim` channels attend to one another, and nothing mixes along the disparity
//! axis. The port reproduces that: attention is per position, over heads, with scale
//! `head_dim^-0.5`.
//!
//! This kernel: softmax over `j` and the weighted sum of values, fused as in
//! [`super::xca_apply`] (a max pass, then one pass accumulating the exponentials and
//! the product, dividing at the end). **Layout**: logits `[B, heads * heads, S]`,
//! `v` `[B, C, S]`, output `[B, C, S]`. Grid: `B * C * ceil(S / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, i*hd + e, s] = sum_j softmax_j(attn[b, i*heads + j, s]) * v[b, j*hd + e, s]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn head_attn_apply_forward<T: Triton, D: Float, const BLOCK: i32>(
    attn_ptr: InPtr<T::Pointer<D>>,
    v_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    HEADS: i32,
    HD: i32,
    S: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = HEADS * HD;
    let tiles = (S + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let row = pid / tiles; // b * C + i * HD + e
    let ch = row % c;
    let b = row / c;
    let i = ch / HD;
    let e = ch % HD;
    let s = T::arange(0, BLOCK) + tile * BLOCK;
    let m = s.lt(S);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let logits = (b * HEADS + i) * HEADS;

    let mut mx = T::full(&[BLOCK], D::from_f64(-3.0e38));
    for j in 0..HEADS {
        let l = T::load(
            attn_ptr.add_offsets(s + (logits + j) * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        mx = T::maximum(mx, l);
    }
    // An exact identity op between the loops: teenyc's RISC-V backend panics on
    // back-to-back loops (teenygrad/teenygrad#156).
    let mx = mx + zeros;
    let mut sum = T::zeros::<D>(&[BLOCK]);
    let mut acc = T::zeros::<D>(&[BLOCK]);
    for j in 0..HEADS {
        let l = T::load(
            attn_ptr.add_offsets(s + (logits + j) * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let v = T::load(
            v_ptr.add_offsets(s + (b * c + j * HD + e) * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let x = T::exp(l - mx);
        sum = sum + x;
        acc = acc + x * v;
    }
    T::store(
        out_ptr.add_offsets(s + row * S),
        acc / sum,
        Some(m),
        &[],
        None,
        None,
    );
}

// ---------------------------------------------------------------------------
// Graph-level op
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use std::any::Any;
use std::sync::Arc;
use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomOp, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};

/// Graph-level head-attention output: logits `[B, heads * heads, ...]` and `v`
/// `[B, C, ...]` to `[B, C, ...]`.
pub struct HeadAttnApplyOp<D: FloatBytes + Send + Sync + 'static> {
    /// Heads.
    pub heads: usize,
    /// Channels per head.
    pub head_dim: usize,
    /// Launch block size.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> HeadAttnApplyOp<D> {
    /// Creates the op.
    pub fn new(heads: usize, head_dim: usize, block: i32) -> Self {
        Self {
            heads,
            head_dim,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for HeadAttnApplyOp<D> {
    fn name(&self) -> &str {
        "stereo.head_attn_apply"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[1].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = HeadAttnApplyForward::<D>::new(self.block);
        Some((
            "head_attn_apply_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(HeadAttnApplyRuntimeOp::<D> {
                heads: self.heads,
                head_dim: self.head_dim,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`HeadAttnApplyOp`].
pub struct HeadAttnApplyRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    heads: usize,
    head_dim: usize,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for HeadAttnApplyRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.heads as i32);
        visitor.visit_i32(self.head_dim as i32);
        visitor.visit_i32(output_shape[2..].iter().product::<usize>() as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let s: usize = output_shape[2..].iter().product();
        [
            (output_shape[0] * output_shape[1] * s.div_ceil(self.block as usize)) as u32,
            1,
            1,
        ]
    }
}
