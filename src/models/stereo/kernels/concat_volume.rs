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

//! FoundationStereo's concatenation volume, `build_concat_volume`
//! (`core/submodule.py`):
//!
//! ```python
//! volume[:, :C, d, :, :] = l                    # every disparity sees the full left map
//! volume[:, C:, d, :, d:] = r[..., :-d]         # right pixel x - d, zero where x < d
//! ```
//!
//! **Layout**: features `[B, C, H, W]`, output `[B, 2C, D, H, W]`. Grid:
//! `B * 2C * D * H * ceil(W / BLOCK)`, lanes on columns. Whether an output channel
//! reads left or right is arithmetic (`c / C`), not a branch: the right half shifts
//! by `d` and masks, the left half shifts by zero.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, c, d, y, x]`, as above.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn concat_volume_forward<T: Triton, D: Float, const BLOCK: i32>(
    l_ptr: InPtr<T::Pointer<D>>,
    r_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    DISP: i32,
    H: i32,
    W: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let tiles = (W + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let y = r1 % H;
    let r2 = r1 / H;
    let d = r2 % DISP;
    let r3 = r2 / DISP;
    let oc = r3 % (2 * C);
    let b = r3 / (2 * C);
    let right = oc / C; // 0 left, 1 right
    let c = oc % C;
    let shift = d * right;

    let x = T::arange(0, BLOCK) + tile * BLOCK;
    let m = x.lt(W);
    let xs = x + (-shift);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let xs_c = T::maximum(xs, T::zeros::<i32>(&[BLOCK]));
    let row = ((b * C + c) * H + y) * W;
    let lv = T::load(
        l_ptr.add_offsets(xs_c + row),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let rv = T::load(
        r_ptr.add_offsets(xs_c + row),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let sel = T::full(&[BLOCK], D::from_f64(right as f64));
    let v = rv * sel + lv * (T::full(&[BLOCK], D::from_f64(1.0)) - sel);
    let v = T::where_(xs.ge(0), v, zeros);
    T::store(
        out_ptr.add_offsets(x + (((b * 2 * C + oc) * DISP + d) * H + y) * W),
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

/// Graph-level concatenation volume: left and right features `[B, C, H, W]` to
/// `[B, 2C, disps, H, W]`.
pub struct ConcatVolumeOp<D: FloatBytes + Send + Sync + 'static> {
    /// Disparities (`max_disp // 4`).
    pub disps: usize,
    /// Launch block size along columns.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ConcatVolumeOp<D> {
    /// Creates the op.
    pub fn new(disps: usize, block: i32) -> Self {
        Self {
            disps,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ConcatVolumeOp<D> {
    fn name(&self) -> &str {
        "stereo.concat_volume"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let x = input_shapes[0];
        vec![x[0], x[1].map(|c| 2 * c), Some(self.disps), x[2], x[3]]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ConcatVolumeForward::<D>::new(self.block);
        Some((
            "concat_volume_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ConcatVolumeRuntimeOp::<D> {
                disps: self.disps,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ConcatVolumeOp`].
pub struct ConcatVolumeRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    disps: usize,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ConcatVolumeRuntimeOp<D> {
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
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let x = inputs[0].1;
        assert_eq!(x, inputs[1].1, "concat_volume: left and right shapes");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(x[1] as i32);
        visitor.visit_i32(self.disps as i32);
        visitor.visit_i32(x[2] as i32);
        visitor.visit_i32(x[3] as i32);
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
