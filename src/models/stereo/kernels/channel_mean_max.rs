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

//! The channel mean and max of FoundationStereo's `SpatialAttentionExtractor`
//! (`core/submodule.py`):
//!
//! ```python
//! x = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1)
//! ```
//!
//! **Layout**: input `[B, C, H, W]`, output `[B, 2, H, W]` (mean, then max). Grid:
//! `B * ceil(H*W / BLOCK)`, lanes on positions; one walk over channels keeps both.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, 0, s] = mean_c x[b, c, s]`, `out[b, 1, s] = max_c x[b, c, s]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn channel_mean_max_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    S: i32,
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
    let b = pid / tiles;
    let s = T::arange(0, BLOCK) + tile * BLOCK;
    let m = s.lt(S);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let mut sum = T::zeros::<D>(&[BLOCK]);
    let mut mx = T::full(&[BLOCK], D::from_f64(-3.0e38));
    for c in 0..C {
        let v = T::load(
            x_ptr.add_offsets(s + (b * C + c) * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        sum = sum + v;
        mx = T::maximum(mx, v);
    }
    let mean = sum * T::full(&[BLOCK], D::from_f64(1.0 / C as f64));
    T::store(
        out_ptr.add_offsets(s + b * 2 * S),
        mean,
        Some(m),
        &[],
        None,
        None,
    );
    T::store(
        out_ptr.add_offsets(s + (b * 2 + 1) * S),
        mx,
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

/// Graph-level channel mean and max: `[B, C, H, W]` to `[B, 2, H, W]`.
pub struct ChannelMeanMaxOp<D: FloatBytes + Send + Sync + 'static> {
    /// Launch block size.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ChannelMeanMaxOp<D> {
    /// Creates the op.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ChannelMeanMaxOp<D> {
    fn name(&self) -> &str {
        "stereo.channel_mean_max"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let x = input_shapes[0];
        vec![x[0], Some(2), x[2], x[3]]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ChannelMeanMaxForward::<D>::new(self.block);
        Some((
            "channel_mean_max_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ChannelMeanMaxRuntimeOp::<D> {
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ChannelMeanMaxOp`].
pub struct ChannelMeanMaxRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ChannelMeanMaxRuntimeOp<D> {
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
        visitor.visit_i32(x[1] as i32);
        visitor.visit_i32((x[2] * x[3]) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = (output_shape[2] * output_shape[3]).div_ceil(self.block as usize);
        [(output_shape[0] * tiles) as u32, 1, 1]
    }
}
