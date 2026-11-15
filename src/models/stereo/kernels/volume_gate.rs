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

//! The gate of FoundationStereo's `FeatureAtt` (`core/submodule.py`):
//!
//! ```python
//! cv = torch.sigmoid(feat_att(feat).unsqueeze(2)) * cv     # (B, C, 1, H, W) over (B, C, D, H, W)
//! ```
//!
//! An image-feature map gates every disparity of a cost volume. The sigmoid is fused.
//! **Layout**: `cv` `[B, C, D, H, W]`, `g` `[B, C, H, W]` (pre-sigmoid), output as `cv`.
//! Grid: `B * C * D * ceil(H*W / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, c, d, s] = sigmoid(g[b, c, s]) * cv[b, c, d, s]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn volume_gate_forward<T: Triton, D: Float, const BLOCK: i32>(
    cv_ptr: InPtr<T::Pointer<D>>,
    g_ptr: InPtr<T::Pointer<D>>,
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
    let r1 = pid / tiles; // (b * C + c) * DISP + d
    let bc = r1 / DISP;
    let s = T::arange(0, BLOCK) + tile * BLOCK;
    let m = s.lt(S);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let x = T::load(
        cv_ptr.add_offsets(s + r1 * S),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let g = T::load(
        g_ptr.add_offsets(s + bc * S),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let gate = one / (one + T::exp(zeros - g));
    T::store(
        out_ptr.add_offsets(s + r1 * S),
        gate * x,
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

/// Graph-level `sigmoid(g).unsqueeze(2) * cv`: inputs `cv` `[B, C, D, H, W]` and `g`
/// `[B, C, H, W]`.
pub struct VolumeGateOp<D: FloatBytes + Send + Sync + 'static> {
    /// Launch block size.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> VolumeGateOp<D> {
    /// Creates the op.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for VolumeGateOp<D> {
    fn name(&self) -> &str {
        "stereo.volume_gate"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = VolumeGateForward::<D>::new(self.block);
        Some((
            "volume_gate_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(VolumeGateRuntimeOp::<D> {
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`VolumeGateOp`].
pub struct VolumeGateRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for VolumeGateRuntimeOp<D> {
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
        let (cv, g) = (inputs[0].1, inputs[1].1);
        assert_eq!(
            [cv[0], cv[1], cv[3], cv[4]],
            [g[0], g[1], g[2], g[3]],
            "volume_gate: gate shape"
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(cv[2] as i32);
        visitor.visit_i32((cv[3] * cv[4]) as i32);
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
