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

//! Token self-attention logits, the first half of DINOv2's `Attention`:
//!
//! ```python
//! qkv = self.qkv(x).reshape(B, N, 3, heads, d).permute(2, 0, 3, 1, 4)   # [3, B, heads, N, d]
//! attn = (q * scale) @ k.transpose(-2, -1)                              # [B, heads, N, N]
//! ```
//!
//! On channel-first tokens the 1x1 `qkv` conv lays `q`, `k`, `v` out as
//! `[B, 3C, N]` with channel `t*C + h*d + i`, so a key row is contiguous in `N`.
//! Each program computes one query's logits against a tile of keys, walking the `d`
//! head channels with the query value broadcast across the lanes. Grid:
//! `B * heads * N * ceil(N / BLOCK)`. Output `[B, heads, N, N]`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `attn[b, h, i, j] = scale * sum_e q[b, h*d + e, i] * k[b, h*d + e, j]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn token_attn_logits_forward<T: Triton, D: Float, const BLOCK: i32>(
    qkv_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    HEADS: i32,
    N: i32,
    scale: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let d = C / HEADS;
    let tiles = (N + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let row = pid / tiles; // (b * HEADS + h) * N + i
    let i = row % N;
    let bh = row / N;
    let h = bh % HEADS;
    let b = bh / HEADS;

    let j = T::arange(0, BLOCK) + tile * BLOCK;
    let m = j.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let zero_idx = T::zeros::<i32>(&[BLOCK]);
    let q_base = (b * 3 * C + h * d) * N + i;
    let k_base = (b * 3 * C + C + h * d) * N;
    let mut acc = T::zeros::<D>(&[BLOCK]);
    for e in 0..d {
        let q = T::load(
            qkv_ptr.add_offsets(zero_idx + (q_base + e * N)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let k = T::load(
            qkv_ptr.add_offsets(j + (k_base + e * N)),
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
    let s = T::full(&[BLOCK], D::from_f64(scale as f64));
    T::store(
        out_ptr.add_offsets(j + row * N),
        acc * s,
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

/// Graph-level token attention logits: `qkv` `[B, 3C, 1, N]` to `[B, heads, N, N]`.
pub struct TokenAttnLogitsOp<D: FloatBytes + Send + Sync + 'static> {
    /// Channels `C` (the input has `3C`).
    pub c: usize,
    /// Attention heads.
    pub heads: usize,
    /// Logit scale, `d^-0.5` in DINOv2.
    pub scale: f32,
    /// Launch block size along keys.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> TokenAttnLogitsOp<D> {
    /// Creates the op.
    pub fn new(c: usize, heads: usize, scale: f32, block: i32) -> Self {
        Self {
            c,
            heads,
            scale,
            block,
            _phantom: PhantomData,
        }
    }
}

fn n_of(shape: &[usize]) -> usize {
    shape[2..].iter().product::<usize>().max(1)
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for TokenAttnLogitsOp<D> {
    fn name(&self) -> &str {
        "stereo.token_attn_logits"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let s = input_shapes[0];
        let n = s[2..].iter().try_fold(1, |a, d| d.map(|d| a * d));
        vec![s[0], Some(self.heads), n, n]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = TokenAttnLogitsForward::<D>::new(self.block);
        Some((
            "token_attn_logits_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(TokenAttnLogitsRuntimeOp::<D> {
                c: self.c,
                heads: self.heads,
                scale: self.scale,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`TokenAttnLogitsOp`].
pub struct TokenAttnLogitsRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    c: usize,
    heads: usize,
    scale: f32,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for TokenAttnLogitsRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let x = inputs[0].1;
        assert_eq!(x[1], 3 * self.c, "token_attn_logits: qkv channels");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.c as i32);
        visitor.visit_i32(self.heads as i32);
        visitor.visit_i32(n_of(x) as i32);
        visitor.visit_f32(self.scale);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n = output_shape[3];
        let tiles = n.div_ceil(self.block as usize);
        [(output_shape[0] * self.heads * n * tiles) as u32, 1, 1]
    }
}
