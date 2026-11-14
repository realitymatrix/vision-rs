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

//! The patch grid of channel-first ViT tokens, dropping the class token: DINOv2's
//! `out[:, 1:]` followed by the DPT head's `permute(0, 2, 1).reshape(B, C, h, w)`.
//!
//! `[B, C, 1, 1 + h*w]` to `[B, C, h, w]`, a shifted copy of each channel row. Grid:
//! `B * C * ceil(h*w / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, c, p] = tokens[b, c, 1 + p]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn vit_patches_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    P: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let tiles = (P + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let row = pid / tiles;
    let p = T::arange(0, BLOCK) + tile * BLOCK;
    let m = p.lt(P);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let x = T::load(
        x_ptr.add_offsets(p + (row * (P + 1) + 1)),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        out_ptr.add_offsets(p + row * P),
        x,
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

/// Graph-level patch grid: tokens `[B, C, 1, 1 + h*w]` to `[B, C, h, w]`.
pub struct VitPatchesOp<D: FloatBytes + Send + Sync + 'static> {
    /// Patch rows `h`.
    pub h: usize,
    /// Patch columns `w`.
    pub w: usize,
    /// Launch block size.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> VitPatchesOp<D> {
    /// Creates the op for an `h x w` patch grid.
    pub fn new(h: usize, w: usize, block: i32) -> Self {
        Self {
            h,
            w,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for VitPatchesOp<D> {
    fn name(&self) -> &str {
        "stereo.vit_patches"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let s = input_shapes[0];
        if let Some(n) = s[3] {
            assert_eq!(n, 1 + self.h * self.w, "vit_patches: token count");
        }
        vec![s[0], s[1], Some(self.h), Some(self.w)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = VitPatchesForward::<D>::new(self.block);
        Some((
            "vit_patches_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(VitPatchesRuntimeOp::<D> {
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`VitPatchesOp`].
pub struct VitPatchesRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for VitPatchesRuntimeOp<D> {
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
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32((output_shape[2] * output_shape[3]) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let p = output_shape[2] * output_shape[3];
        let tiles = p.div_ceil(self.block as usize);
        [(output_shape[0] * output_shape[1] * tiles) as u32, 1, 1]
    }
}
