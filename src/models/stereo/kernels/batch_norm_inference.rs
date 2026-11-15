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

//! Inference batch normalisation over any rank,
//! `y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]`, for
//! FoundationStereo's `BatchNorm3d` layers. teeny's `Op::BatchNorm3d` lowers to a
//! kernel with no runtime implementation (teeny-kernels 0.3.1).
//!
//! **Layout**: `[B, C, S]`, `S` flattening the trailing axes. Grid:
//! `B * C * ceil(S / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The normalisation above.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn batch_norm_inference_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    mean_ptr: InPtr<T::Pointer<D>>,
    var_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    S: i32,
    eps: f32,
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
    let row = pid / tiles; // b * C + c
    let c = row % C;
    let s = T::arange(0, BLOCK) + tile * BLOCK;
    let m = s.lt(S);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let ci = T::zeros::<i32>(&[BLOCK]) + c;
    let w = T::load(
        w_ptr.add_offsets(ci),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(ci),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let mean = T::load(
        mean_ptr.add_offsets(ci),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let var = T::load(
        var_ptr.add_offsets(ci),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let x = T::load(
        x_ptr.add_offsets(s + row * S),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let rstd = T::rsqrt(var + T::full(&[BLOCK], D::from_f64(eps as f64)));
    T::store(
        out_ptr.add_offsets(s + row * S),
        (x - mean) * rstd * w + b,
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

/// Graph-level inference BatchNorm for any rank. Weight slots `weight`, `bias`,
/// `running_mean`, `running_var`, each `[C]`.
pub struct BatchNormInferenceOp<D: FloatBytes + Send + Sync + 'static> {
    /// Channels.
    pub c: usize,
    /// Variance epsilon.
    pub eps: f32,
    /// Launch block size.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> BatchNormInferenceOp<D> {
    /// Creates the op.
    pub fn new(c: usize, eps: f32, block: i32) -> Self {
        Self {
            c,
            eps,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for BatchNormInferenceOp<D> {
    fn name(&self) -> &str {
        "stereo.batch_norm_inference"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = BatchNormInferenceForward::<D>::new(self.block);
        Some((
            "batch_norm_inference_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(BatchNormInferenceRuntimeOp::<D> {
                c: self.c,
                eps: self.eps,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`BatchNormInferenceOp`].
pub struct BatchNormInferenceRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    c: usize,
    eps: f32,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for BatchNormInferenceRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.c]; 4]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "bias", "running_mean", "running_var"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        assert_eq!(inputs[0].1[1], self.c, "batch_norm: channels");
        visitor.visit_ptr(inputs[0].0);
        for p in &params[..4] {
            visitor.visit_ptr(*p);
        }
        visitor.visit_ptr(output);
        visitor.visit_i32(self.c as i32);
        visitor.visit_i32(output_shape[2..].iter().product::<usize>() as i32);
        visitor.visit_f32(self.eps);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let s: usize = output_shape[2..].iter().product();
        [
            (output_shape[0] * self.c * s.div_ceil(self.block as usize)) as u32,
            1,
            1,
        ]
    }
}
