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

//! Trilinear upsampling by an integer factor, `F.interpolate(x, scale_factor=F,
//! mode="trilinear", align_corners=False)`, as the `hourglass` brings its disparity
//! transformer's output back to the 1/4 volume (factor 4).
//!
//! Output `o` samples source `max(0, (o + 0.5) / F - 0.5)` along each axis (the scale
//! factor given, PyTorch uses `1 / F`; linear modes clamp below at 0), interpolating
//! between `floor` and `floor + 1` clamped to the edge. **Layout**: `[B, C, D, H, W]`
//! to `[B, C, F*D, F*H, F*W]`. Grid: `B * C * F*D * F*H * ceil(F*W / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The upsampling above.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn trilinear_upsample_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    DD: i32,
    HH: i32,
    WW: i32,
    F: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let OD = DD * F;
    let OH = HH * F;
    let OW = WW * F;
    let tiles = (OW + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let oh = r1 % OH;
    let r2 = r1 / OH;
    let od = r2 % OD;
    let plane = r2 / OD; // b * C + c

    let ow = T::arange(0, BLOCK) + tile * BLOCK;
    let m = ow.lt(OW);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let izero = T::zeros::<i32>(&[BLOCK]);
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let half = T::full(&[BLOCK], D::from_f64(0.5));
    let inv = T::full(&[BLOCK], D::from_f64(1.0 / F as f64));
    // No closures in kernels: the source mapping is written out per axis.
    let sd = T::maximum(
        (T::cast::<i32, D>(izero + od, None, false) + half) * inv - half,
        zeros,
    );
    let sh = T::maximum(
        (T::cast::<i32, D>(izero + oh, None, false) + half) * inv - half,
        zeros,
    );
    let sw = T::maximum(
        (T::cast::<i32, D>(ow, None, false) + half) * inv - half,
        zeros,
    );
    let fd = T::floor(sd);
    let fh = T::floor(sh);
    let fw = T::floor(sw);
    let td = sd - fd;
    let th = sh - fh;
    let tw = sw - fw;
    let d0 = T::cast::<D, i32>(fd, None, false);
    let h0 = T::cast::<D, i32>(fh, None, false);
    let w0 = T::cast::<D, i32>(fw, None, false);

    let mut acc = T::zeros::<D>(&[BLOCK]);
    for k in 0..8 {
        let a = k / 4;
        let bb = (k / 2) % 2;
        let cc = k % 2;
        let di = T::minimum(d0 + a, izero + (DD - 1));
        let hi = T::minimum(h0 + bb, izero + (HH - 1));
        let wi = T::minimum(w0 + cc, izero + (WW - 1));
        let fa = T::cast::<i32, D>(izero + a, None, false);
        let fb = T::cast::<i32, D>(izero + bb, None, false);
        let fc = T::cast::<i32, D>(izero + cc, None, false);
        let wgt = (fa * td + (one - fa) * (one - td))
            * (fb * th + (one - fb) * (one - th))
            * (fc * tw + (one - fc) * (one - tw));
        let v = T::load(
            x_ptr.add_offsets(wi + ((di + plane * DD) * HH + hi) * WW),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + wgt * v;
    }
    T::store(
        out_ptr.add_offsets(ow + ((plane * OD + od) * OH + oh) * OW),
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

/// Graph-level trilinear upsampling by `factor` (`align_corners=False`).
pub struct TrilinearUpsampleOp<D: FloatBytes + Send + Sync + 'static> {
    /// The integer scale factor.
    pub factor: usize,
    /// Launch block size along output columns.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> TrilinearUpsampleOp<D> {
    /// Creates the op.
    pub fn new(factor: usize, block: i32) -> Self {
        Self {
            factor,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for TrilinearUpsampleOp<D> {
    fn name(&self) -> &str {
        "stereo.trilinear_upsample"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let x = input_shapes[0];
        let f = self.factor;
        let up = |d: Option<usize>| d.map(|n| f * n);
        vec![x[0], x[1], up(x[2]), up(x[3]), up(x[4])]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = TrilinearUpsampleForward::<D>::new(self.block);
        Some((
            "trilinear_upsample_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(TrilinearUpsampleRuntimeOp::<D> {
                factor: self.factor,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`TrilinearUpsampleOp`].
pub struct TrilinearUpsampleRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    factor: usize,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for TrilinearUpsampleRuntimeOp<D> {
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
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(x[2] as i32);
        visitor.visit_i32(x[3] as i32);
        visitor.visit_i32(x[4] as i32);
        visitor.visit_i32(self.factor as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = output_shape[4].div_ceil(self.block as usize);
        [
            (output_shape[..4].iter().product::<usize>() * tiles) as u32,
            1,
            1,
        ]
    }
}
