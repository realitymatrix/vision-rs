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

//! FoundationStereo's group-wise correlation volume, `build_gwc_volume`
//! (`core/submodule.py`):
//!
//! ```python
//! volume[:, :, d, :, d:] = (F.normalize(l[..., d:], dim=2) * F.normalize(r[..., :-d], dim=2)).sum(dim=2)
//! ```
//!
//! with features split into `G` groups of `C / G` channels: the cosine similarity,
//! per group, of left pixel `x` and right pixel `x - d`, zero where `x < d`.
//! `F.normalize` clamps each norm at `1e-12`; the kernel divides after the dot
//! product rather than normalising first, the same up to rounding.
//!
//! **Layout**: features `[B, C, H, W]`, output `[B, G, D, H, W]`. Grid:
//! `B * G * D * H * ceil(W / BLOCK)`, lanes on columns; one walk over the group's
//! channels accumulates `l.l`, `r.r` and `l.r`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, g, d, y, x]`, as above.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn gwc_volume_forward<T: Triton, D: Float, const BLOCK: i32>(
    l_ptr: InPtr<T::Pointer<D>>,
    r_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    G: i32,
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
    let cpg = C / G;
    let tiles = (W + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let y = r1 % H;
    let r2 = r1 / H;
    let d = r2 % DISP;
    let r3 = r2 / DISP;
    let g = r3 % G;
    let b = r3 / G;

    let x = T::arange(0, BLOCK) + tile * BLOCK;
    let m = x.lt(W);
    let xr = x + (-d);
    let mr = xr.ge(0);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let izero = T::zeros::<i32>(&[BLOCK]);
    let xr_c = T::maximum(xr, izero);
    let mut ll = T::zeros::<D>(&[BLOCK]);
    let mut rr = T::zeros::<D>(&[BLOCK]);
    let mut lr = T::zeros::<D>(&[BLOCK]);
    for c in 0..cpg {
        let row = ((b * C + g * cpg + c) * H + y) * W;
        let l = T::load(
            l_ptr.add_offsets(x + row),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let r = T::load(
            r_ptr.add_offsets(xr_c + row),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        ll = ll + l * l;
        rr = rr + r * r;
        lr = lr + l * r;
    }
    let eps = T::full(&[BLOCK], D::from_f64(1e-12));
    let cost = lr / (T::maximum(T::sqrt(ll), eps) * T::maximum(T::sqrt(rr), eps));
    let cost = T::where_(mr, cost, zeros);
    T::store(
        out_ptr.add_offsets(x + (((b * G + g) * DISP + d) * H + y) * W),
        cost,
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

/// Graph-level group-wise correlation volume: left and right features
/// `[B, C, H, W]` to `[B, groups, disps, H, W]`.
pub struct GwcVolumeOp<D: FloatBytes + Send + Sync + 'static> {
    /// Correlation groups (`cv_group`, 8 in FoundationStereo).
    pub groups: usize,
    /// Disparities (`max_disp // 4`).
    pub disps: usize,
    /// Launch block size along columns.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> GwcVolumeOp<D> {
    /// Creates the op.
    pub fn new(groups: usize, disps: usize, block: i32) -> Self {
        Self {
            groups,
            disps,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for GwcVolumeOp<D> {
    fn name(&self) -> &str {
        "stereo.gwc_volume"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let x = input_shapes[0];
        vec![x[0], Some(self.groups), Some(self.disps), x[2], x[3]]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = GwcVolumeForward::<D>::new(self.block);
        Some((
            "gwc_volume_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(GwcVolumeRuntimeOp::<D> {
                groups: self.groups,
                disps: self.disps,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`GwcVolumeOp`].
pub struct GwcVolumeRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    groups: usize,
    disps: usize,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for GwcVolumeRuntimeOp<D> {
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
        assert_eq!(x, inputs[1].1, "gwc_volume: left and right shapes");
        assert_eq!(
            x[1] % self.groups,
            0,
            "gwc_volume: channels not divisible by groups"
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(x[1] as i32);
        visitor.visit_i32(self.groups as i32);
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
