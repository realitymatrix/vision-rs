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

//! The sinusoidal disparity embedding of FoundationStereo's
//! `CostVolumeDisparityAttention` (`PositionalEmbedding`, `core/submodule.py`), added
//! to every voxel of a `[B, C, D, H, W]` volume along its disparity axis:
//!
//! ```python
//! div = exp(arange(0, C, 2) * -(log(10000) / C))
//! pe[d, 2i] = sin(d * div[i]);  pe[d, 2i + 1] = cos(d * div[i])
//! x = x + pe[d, c]
//! ```
//!
//! The table is computed in the kernel (the log factor passed from the host), in the volume's precision (PyTorch builds it in
//! f32). Grid: `B * C * D * ceil(H*W / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, c, d, s] = x[b, c, d, s] + pe[d, c]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn disp_pos_embed_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    DISP: i32,
    S: i32,
    neg_log_over_c: f32,
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
    let row = pid / tiles; // (b * C + c) * DISP + d
    let d = row % DISP;
    let c = (row / DISP) % C;
    let s = T::arange(0, BLOCK) + tile * BLOCK;
    let m = s.lt(S);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let izero = T::zeros::<i32>(&[BLOCK]);
    let pair = T::cast::<i32, D>(izero + 2 * (c / 2), None, false);
    let div = T::exp(pair * T::full(&[BLOCK], D::from_f64(neg_log_over_c as f64)));
    let ang = T::cast::<i32, D>(izero + d, None, false) * div;
    // Even channels take sin, odd cos: selected arithmetically.
    let odd = T::cast::<i32, D>(izero + c % 2, None, false);
    let pe = T::sin(ang) * (T::full(&[BLOCK], D::from_f64(1.0)) - odd) + T::cos(ang) * odd;
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
    T::store(
        out_ptr.add_offsets(s + row * S),
        x + pe,
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

/// Graph-level disparity embedding on `[B, C, D, H, W]`.
pub struct DispPosEmbedOp<D: FloatBytes + Send + Sync + 'static> {
    /// Launch block size.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> DispPosEmbedOp<D> {
    /// Creates the op.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for DispPosEmbedOp<D> {
    fn name(&self) -> &str {
        "stereo.disp_pos_embed"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = DispPosEmbedForward::<D>::new(self.block);
        Some((
            "disp_pos_embed_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(DispPosEmbedRuntimeOp::<D> {
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`DispPosEmbedOp`].
pub struct DispPosEmbedRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for DispPosEmbedRuntimeOp<D> {
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
        assert_eq!(x[1] % 2, 0, "disp_pos_embed: odd channel count");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(x[1] as i32);
        visitor.visit_i32(x[2] as i32);
        visitor.visit_i32((x[3] * x[4]) as i32);
        // `-(log(10000) / C)`: a double in PyTorch, narrowed to f32 when it scales the
        // f32 `arange`. Passed in because kernels have no `ln`.
        visitor.visit_f32((-(10000.0f64).ln() / x[1] as f64) as f32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let s = output_shape[3] * output_shape[4];
        [
            (output_shape[..3].iter().product::<usize>() * s.div_ceil(self.block as usize)) as u32,
            1,
            1,
        ]
    }
}
