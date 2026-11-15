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
//! This kernel: the logits. **Layout**: `q`, `k` `[B, C, S]` (`C = heads * head_dim`,
//! channel `h * head_dim + e`, `S` flattening the trailing axes), output
//! `[B, heads * heads, S]` with channel `i * heads + j`. Grid:
//! `B * heads * heads * ceil(S / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, i*heads + j, s] = scale * sum_e q[b, i*hd + e, s] * k[b, j*hd + e, s]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn head_attn_logits_forward<T: Triton, D: Float, const BLOCK: i32>(
    q_ptr: InPtr<T::Pointer<D>>,
    k_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    HEADS: i32,
    HD: i32,
    S: i32,
    scale: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let tiles = (S + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let row = pid / tiles; // (b * HEADS + i) * HEADS + j
    let j = row % HEADS;
    let i = (row / HEADS) % HEADS;
    let b = row / (HEADS * HEADS);
    let c = HEADS * HD;
    let s = T::arange(0, BLOCK) + tile * BLOCK;
    let m = s.lt(S);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let mut acc = T::zeros::<D>(&[BLOCK]);
    for e in 0..HD {
        let q = T::load(
            q_ptr.add_offsets(s + (b * c + i * HD + e) * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let k = T::load(
            k_ptr.add_offsets(s + (b * c + j * HD + e) * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + q * k;
    }
    let f = T::full(&[BLOCK], D::from_f64(scale as f64));
    T::store(
        out_ptr.add_offsets(s + row * S),
        acc * f,
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

/// Graph-level head-attention logits: `q`, `k` `[B, C, ...]` to
/// `[B, heads * heads, ...]`.
pub struct HeadAttnLogitsOp<D: FloatBytes + Send + Sync + 'static> {
    /// Heads.
    pub heads: usize,
    /// Channels per head.
    pub head_dim: usize,
    /// Launch block size.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> HeadAttnLogitsOp<D> {
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

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for HeadAttnLogitsOp<D> {
    fn name(&self) -> &str {
        "stereo.head_attn_logits"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let mut s = input_shapes[0].clone();
        s[1] = Some(self.heads * self.heads);
        s
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = HeadAttnLogitsForward::<D>::new(self.block);
        Some((
            "head_attn_logits_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(HeadAttnLogitsRuntimeOp::<D> {
                heads: self.heads,
                head_dim: self.head_dim,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`HeadAttnLogitsOp`].
pub struct HeadAttnLogitsRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    heads: usize,
    head_dim: usize,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for HeadAttnLogitsRuntimeOp<D> {
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
        assert_eq!(
            inputs[0].1[1],
            self.heads * self.head_dim,
            "head_attn: channels"
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.heads as i32);
        visitor.visit_i32(self.head_dim as i32);
        visitor.visit_i32(output_shape[2..].iter().product::<usize>() as i32);
        visitor.visit_f32((self.head_dim as f64).powf(-0.5) as f32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let s: usize = output_shape[2..].iter().product();
        [
            (output_shape[0] * self.heads * self.heads * s.div_ceil(self.block as usize)) as u32,
            1,
            1,
        ]
    }
}
