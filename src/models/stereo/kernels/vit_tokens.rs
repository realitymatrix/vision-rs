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

//! DINOv2's `prepare_tokens_with_masks` after the patch embedding:
//!
//! ```python
//! x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)     # [B, 1 + h*w, C]
//! x = x + interpolate_pos_encoding(x, H, W)
//! ```
//!
//! `interpolate_pos_encoding` resizes the `M x M` grid of patch position embeddings
//! to `h x w` with `F.interpolate(mode="bicubic", scale_factor=((h + 0.1) / M,
//! (w + 0.1) / M))` (the `interpolate_offset` kludge, no antialiasing), and the class
//! token keeps `pos_embed[0]`. With a scale factor given, PyTorch maps output pixel
//! `o` to source `(o + 0.5) / scale - 0.5`, so the inverse scales are kernel
//! arguments. Taps use the cubic convolution with `A = -0.75`, clamped to the grid
//! edge, as `upsample_bicubic2d` does.
//!
//! **Layout**: tokens stay channel-first, `[B, C, 1, 1 + h*w]`, so the transformer's
//! LayerNorms are channel LayerNorms and its linears 1x1 convs. Grid:
//! `B * C * ceil((1 + h*w) / BLOCK)`, lanes on tokens; token 0 is the class token.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, c, 0, t]`, as above.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn vit_tokens_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    cls_ptr: InPtr<T::Pointer<D>>,
    pos_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    H: i32,
    W: i32,
    M: i32,
    inv_sy: f32,
    inv_sx: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let n_tok = 1 + H * W;
    let tiles = (n_tok + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let row = pid / tiles; // b * C + c
    let c = row % C;

    let t = T::arange(0, BLOCK) + tile * BLOCK;
    let m = t.lt(n_tok);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let izero = T::zeros::<i32>(&[BLOCK]);
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let half = T::full(&[BLOCK], D::from_f64(0.5));
    let a = T::full(&[BLOCK], D::from_f64(-0.75));

    // Patch index and its row and column (the class token reads patch 0, discarded).
    // Row by float floor, exact at these sizes, avoiding tensor integer division.
    let p = T::maximum(t + (-1), izero);
    let pf = T::cast::<i32, D>(p, None, false);
    let py = T::cast::<D, i32>(
        T::floor((pf + half) * T::full(&[BLOCK], D::from_f64(1.0 / W as f64))),
        None,
        false,
    );
    let px = p + py * (-W);

    // Source coordinates, whole parts and fractions.
    let fy = T::full(&[BLOCK], D::from_f64(inv_sy as f64))
        * (T::cast::<i32, D>(py, None, false) + half)
        - half;
    let fx = T::full(&[BLOCK], D::from_f64(inv_sx as f64))
        * (T::cast::<i32, D>(px, None, false) + half)
        - half;
    let fy0 = T::floor(fy);
    let fx0 = T::floor(fx);
    let ty = fy - fy0;
    let tx = fx - fx0;
    let iy = T::cast::<D, i32>(fy0, None, false);
    let ix = T::cast::<D, i32>(fx0, None, false);

    let mut acc = T::zeros::<D>(&[BLOCK]);
    for k in 0..16 {
        let ka = k / 4;
        let kb = k % 4;
        // Taps 1 and 2 are within one of the sample (cubic_convolution1), 0 and 3
        // between one and two (cubic_convolution2).
        let sa = T::cast::<i32, D>(izero + (ka * (3 - ka) + 1) / 2, None, false);
        let sb = T::cast::<i32, D>(izero + (kb * (3 - kb) + 1) / 2, None, false);
        let da = T::abs(ty - T::cast::<i32, D>(izero + (ka - 1), None, false));
        let db = T::abs(tx - T::cast::<i32, D>(izero + (kb - 1), None, false));
        let wa1 = ((a + one + one) * da - (a + one + one + one)) * da * da + one;
        let wa2 = ((a * da - a * T::full(&[BLOCK], D::from_f64(5.0))) * da
            + a * T::full(&[BLOCK], D::from_f64(8.0)))
            * da
            - a * T::full(&[BLOCK], D::from_f64(4.0));
        let wb1 = ((a + one + one) * db - (a + one + one + one)) * db * db + one;
        let wb2 = ((a * db - a * T::full(&[BLOCK], D::from_f64(5.0))) * db
            + a * T::full(&[BLOCK], D::from_f64(8.0)))
            * db
            - a * T::full(&[BLOCK], D::from_f64(4.0));
        let wa = wa1 * sa + wa2 * (one - sa);
        let wb = wb1 * sb + wb2 * (one - sb);
        let yc = T::maximum(T::minimum(iy + (ka - 1), izero + (M - 1)), izero);
        let xc = T::maximum(T::minimum(ix + (kb - 1), izero + (M - 1)), izero);
        let p = T::load(
            pos_ptr.add_offsets((yc * M + xc + 1) * C + c),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + wa * wb * p;
    }

    let x = T::load(
        x_ptr.add_offsets(p + row * (H * W)),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let cls = T::load(
        cls_ptr.add_offsets(izero + c),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let pos0 = T::load(
        pos_ptr.add_offsets(izero + c),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let v = T::where_(t.ge(1), x + acc, cls + pos0);
    T::store(
        out_ptr.add_offsets(t + row * n_tok),
        v,
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

/// Graph-level token preparation: patch embeddings `[B, C, h, w]` to tokens
/// `[B, C, 1, 1 + h*w]`. Weight slots `cls_token` `[C]` and `pos_embed`
/// `[1 + M*M, C]` (PyTorch's `[1, 1, C]` and `[1, 1 + M*M, C]`, same data).
pub struct VitTokensOp<D: FloatBytes + Send + Sync + 'static> {
    /// Embedding width `C`.
    pub c: usize,
    /// Side of the pretrained position grid `M` (37 for DINOv2 at 518 px).
    pub m: usize,
    /// DINOv2's `interpolate_offset`.
    pub offset: f64,
    /// Launch block size along `w`.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> VitTokensOp<D> {
    /// Creates the op.
    pub fn new(c: usize, m: usize, offset: f64, block: i32) -> Self {
        Self {
            c,
            m,
            offset,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for VitTokensOp<D> {
    fn name(&self) -> &str {
        "stereo.vit_tokens"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let s = input_shapes[0];
        let n = match (s[2], s[3]) {
            (Some(h), Some(w)) => Some(1 + h * w),
            _ => None,
        };
        vec![s[0], Some(self.c), Some(1), n]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = VitTokensForward::<D>::new(self.block);
        Some((
            "vit_tokens_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(VitTokensRuntimeOp::<D> {
                c: self.c,
                m: self.m,
                offset: self.offset,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`VitTokensOp`].
pub struct VitTokensRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    c: usize,
    m: usize,
    offset: f64,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for VitTokensRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.c], vec![1 + self.m * self.m, self.c]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["cls_token", "pos_embed"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let x = inputs[0].1;
        assert_eq!(x[1], self.c, "vit_tokens: channels");
        let (h, w) = (x[2], x[3]);
        // PyTorch's `1.0 / scale_factor`, in double, then narrowed to the float
        // accumulation type.
        let m = self.m as f64;
        let inv = |n: usize| (1.0 / ((n as f64 + self.offset) / m)) as f32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.c as i32);
        visitor.visit_i32(h as i32);
        visitor.visit_i32(w as i32);
        visitor.visit_i32(self.m as i32);
        visitor.visit_f32(inv(h));
        visitor.visit_f32(inv(w));
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = output_shape[3].div_ceil(self.block as usize);
        [(output_shape[0] * self.c * tiles) as u32, 1, 1]
    }
}
