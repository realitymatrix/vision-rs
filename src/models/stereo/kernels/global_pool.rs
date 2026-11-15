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

//! Global average and max pooling over the spatial axes, `nn.AdaptiveAvgPool2d(1)` and
//! `nn.AdaptiveMaxPool2d(1)`, as FoundationStereo's `ChannelAttentionEnhancement`
//! (`core/submodule.py`) pools its input.
//!
//! **Layout**: input `[B, C, H, W]`, output `[B, C, 1, 1]`. Grid: `B * C`; each program
//! walks its plane in `BLOCK`-wide tiles, then reduces across the lanes and writes
//! from lane 0. `MAX` selects the reduction (a compile-time constant, so neither
//! kernel branches).

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, c] = mean (MAX = 0) or max (MAX = 1) of x[b, c, :, :]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn global_pool_forward<T: Triton, D: Float, const BLOCK: i32, const MAX: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    S: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK);
    let lowest = T::full(&[BLOCK], D::from_f64(-3.0e38));
    let zeros = T::zeros::<D>(&[BLOCK]);
    // MAX selects which accumulator is kept; both are computed.
    let mut sum = T::zeros::<D>(&[BLOCK]);
    let mut mx = T::full(&[BLOCK], D::from_f64(-3.0e38));
    for t in 0..(S + BLOCK - 1) / BLOCK {
        let s = offs + t * BLOCK;
        let m = s.lt(S);
        let v = T::load(
            x_ptr.add_offsets(s + row * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        sum = sum + v;
        let vm = T::load(
            x_ptr.add_offsets(s + row * S),
            Some(m),
            Some(lowest),
            &[],
            None,
            None,
            None,
            false,
        );
        mx = T::maximum(mx, vm);
    }
    let mean = T::sum(sum, Some(0), true) * T::full(&[BLOCK], D::from_f64(1.0 / S as f64));
    let top = T::max(mx, Some(0), true);
    let sel = T::full(&[BLOCK], D::from_f64(MAX as f64));
    let v = top * sel + mean * (T::full(&[BLOCK], D::from_f64(1.0)) - sel);
    // Broadcast across the lanes, written by lane 0 only.
    T::store(
        out_ptr.add_offsets(offs + row),
        v,
        Some(offs.lt(1)),
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

/// Graph-level global pooling: `[B, C, H, W]` to `[B, C, 1, 1]`, average or max.
pub struct GlobalPoolOp<D: FloatBytes + Send + Sync + 'static> {
    /// Max pooling if true, average otherwise.
    pub max: bool,
    /// Launch block size.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> GlobalPoolOp<D> {
    /// `nn.AdaptiveAvgPool2d(1)`.
    pub fn avg(block: i32) -> Self {
        Self {
            max: false,
            block,
            _phantom: PhantomData,
        }
    }

    /// `nn.AdaptiveMaxPool2d(1)`.
    pub fn max(block: i32) -> Self {
        Self {
            max: true,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for GlobalPoolOp<D> {
    fn name(&self) -> &str {
        if self.max {
            "stereo.global_max_pool"
        } else {
            "stereo.global_avg_pool"
        }
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let x = input_shapes[0];
        vec![x[0], x[1], Some(1), Some(1)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = GlobalPoolForward::<D>::new(self.block, self.max as i32);
        Some((
            "global_pool_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(GlobalPoolRuntimeOp::<D> {
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`GlobalPoolOp`].
pub struct GlobalPoolRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for GlobalPoolRuntimeOp<D> {
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
        visitor.visit_i32((x[2] * x[3]) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(output_shape[0] * output_shape[1]) as u32, 1, 1]
    }
}
