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

//! 2-D bicubic resize, `F.interpolate(x, size=(OH, OW), mode="bicubic",
//! align_corners=False)`, as FoundationStereo's `Feature.forward` resizes the image to
//! a multiple of 112 for the DINOv2 branch.
//!
//! PyTorch's `upsample_bicubic2d`: output pixel `o` samples source
//! `s = scale * (o + 0.5) - 0.5` with `scale = IN / OUT` (computed in f32 on the host,
//! as PyTorch does), taking the 4x4 taps at `floor(s) - 1 ..= floor(s) + 2`, clamped to
//! the edge, weighted by the cubic convolution with `A = -0.75` of the fraction
//! `s - floor(s)`. No antialiasing (PyTorch's default).
//!
//! **Layout**: input `[B, C, IH, IW]`, output `[B, C, OH, OW]`. Grid:
//! `B * C * OH * ceil(OW / BLOCK)`, lanes on columns of one output row; the 16 taps are
//! one flat loop.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The bicubic resize above.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn interpolate_bicubic_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    IH: i32,
    IW: i32,
    OH: i32,
    OW: i32,
    scale_y: f32,
    scale_x: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let tiles = (OW + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let oy = r1 % OH;
    let plane = r1 / OH; // b * C + c

    let ox = T::arange(0, BLOCK) + tile * BLOCK;
    let m = ox.lt(OW);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let izero = T::zeros::<i32>(&[BLOCK]);
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let half = T::full(&[BLOCK], D::from_f64(0.5));
    let a = T::full(&[BLOCK], D::from_f64(-0.75));
    let a5 = T::full(&[BLOCK], D::from_f64(-0.75 * 5.0));
    let a8 = T::full(&[BLOCK], D::from_f64(-0.75 * 8.0));
    let a4 = T::full(&[BLOCK], D::from_f64(-0.75 * 4.0));
    let a2 = T::full(&[BLOCK], D::from_f64(-0.75 + 2.0));
    let a3 = T::full(&[BLOCK], D::from_f64(-0.75 + 3.0));

    let fy = T::full(&[BLOCK], D::from_f64(scale_y as f64))
        * (T::cast::<i32, D>(izero + oy, None, false) + half)
        - half;
    let fx = T::full(&[BLOCK], D::from_f64(scale_x as f64))
        * (T::cast::<i32, D>(ox, None, false) + half)
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
        // Taps 1 and 2 lie within one of the sample (cubic_convolution1), taps 0 and 3
        // between one and two (cubic_convolution2).
        let sa = T::cast::<i32, D>(izero + (ka * (3 - ka) + 1) / 2, None, false);
        let sb = T::cast::<i32, D>(izero + (kb * (3 - kb) + 1) / 2, None, false);
        let da = T::abs(ty - T::cast::<i32, D>(izero + (ka - 1), None, false));
        let db = T::abs(tx - T::cast::<i32, D>(izero + (kb - 1), None, false));
        let wa1 = (a2 * da - a3) * da * da + one;
        let wa2 = ((a * da - a5) * da + a8) * da - a4;
        let wb1 = (a2 * db - a3) * db * db + one;
        let wb2 = ((a * db - a5) * db + a8) * db - a4;
        let wa = wa1 * sa + wa2 * (one - sa);
        let wb = wb1 * sb + wb2 * (one - sb);
        let yc = T::maximum(T::minimum(iy + (ka - 1), izero + (IH - 1)), izero);
        let xc = T::maximum(T::minimum(ix + (kb - 1), izero + (IW - 1)), izero);
        let v = T::load(
            x_ptr.add_offsets(yc * IW + xc + plane * IH * IW),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + wa * wb * v;
    }
    T::store(
        out_ptr.add_offsets(ox + (plane * OH + oy) * OW),
        acc,
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

/// Graph-level bicubic resize to a fixed `(OH, OW)`: `[B, C, IH, IW]` to
/// `[B, C, OH, OW]`.
pub struct InterpolateBicubicOp<D: FloatBytes + Send + Sync + 'static> {
    /// Output size.
    pub size: (usize, usize),
    /// Launch block size along output columns.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> InterpolateBicubicOp<D> {
    /// Creates the op resizing to `size`.
    pub fn new(size: (usize, usize), block: i32) -> Self {
        Self {
            size,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for InterpolateBicubicOp<D> {
    fn name(&self) -> &str {
        "stereo.interpolate_bicubic"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let x = input_shapes[0];
        vec![x[0], x[1], Some(self.size.0), Some(self.size.1)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = InterpolateBicubicForward::<D>::new(self.block);
        Some((
            "interpolate_bicubic_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(InterpolateBicubicRuntimeOp::<D> {
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`InterpolateBicubicOp`].
pub struct InterpolateBicubicRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for InterpolateBicubicRuntimeOp<D> {
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
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let x = inputs[0].1;
        let (ih, iw, oh, ow) = (x[2], x[3], output_shape[2], output_shape[3]);
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(ih as i32);
        visitor.visit_i32(iw as i32);
        visitor.visit_i32(oh as i32);
        visitor.visit_i32(ow as i32);
        // PyTorch's `area_pixel_compute_scale` without a scale factor: in f32.
        visitor.visit_f32(ih as f32 / oh as f32);
        visitor.visit_f32(iw as f32 / ow as f32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = output_shape[3].div_ceil(self.block as usize);
        [
            (output_shape[0] * output_shape[1] * output_shape[2] * tiles) as u32,
            1,
            1,
        ]
    }
}
