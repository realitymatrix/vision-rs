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

//! FoundationStereo's initial disparity (`core/foundation_stereo.py`):
//!
//! ```python
//! prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)   # (B, D, H, W)
//! init_disp = disparity_regression(prob, max_disp // 4)               # sum_d d * prob
//! ```
//!
//! The softmax is fused with the regression (`prob` has no other use): one walk over
//! disparities for the maximum, a second accumulating `sum_d exp(l_d - max)` and
//! `sum_d d * exp(l_d - max)`, dividing at the end. **Layout**: logits `[B, 1, D, H, W]`,
//! output `[B, 1, H, W]`. Grid: `B * ceil(H*W / BLOCK)`, lanes on pixels.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, s] = sum_d d * softmax_d(x[b, d, s])`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn disparity_regression_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    DISP: i32,
    S: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
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
    let izero = T::zeros::<i32>(&[BLOCK]);
    let mut mx = T::full(&[BLOCK], D::from_f64(-3.0e38));
    for d in 0..DISP {
        let l = T::load(
            x_ptr.add_offsets(s + (b * DISP + d) * S),
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
    for d in 0..DISP {
        let l = T::load(
            x_ptr.add_offsets(s + (b * DISP + d) * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let e = T::exp(l - mx);
        sum = sum + e;
        acc = acc + e * T::cast::<i32, D>(izero + d, None, false);
    }
    T::store(
        out_ptr.add_offsets(s + b * S),
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

/// Graph-level softmax-and-regression: logits `[B, 1, D, H, W]` to `[B, 1, H, W]`.
pub struct DisparityRegressionOp<D: FloatBytes + Send + Sync + 'static> {
    /// Launch block size.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> DisparityRegressionOp<D> {
    /// Creates the op.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for DisparityRegressionOp<D> {
    fn name(&self) -> &str {
        "stereo.disparity_regression"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let x = input_shapes[0];
        vec![x[0], Some(1), x[3], x[4]]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = DisparityRegressionForward::<D>::new(self.block);
        Some((
            "disparity_regression_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(DisparityRegressionRuntimeOp::<D> {
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`DisparityRegressionOp`].
pub struct DisparityRegressionRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for DisparityRegressionRuntimeOp<D> {
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
        assert_eq!(x[1], 1, "disparity_regression: expects one channel");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(x[2] as i32);
        visitor.visit_i32((x[3] * x[4]) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let s = output_shape[2] * output_shape[3];
        [
            (output_shape[0] * s.div_ceil(self.block as usize)) as u32,
            1,
            1,
        ]
    }
}
