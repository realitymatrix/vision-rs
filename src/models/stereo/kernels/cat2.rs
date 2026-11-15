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

//! Channel concatenation of two tensors of any rank, `torch.cat([a, b], dim=1)`.
//!
//! teeny's `Op::ChannelCat` sizes its copy from axes 1..=3, so it handles only 4-D
//! tensors; FoundationStereo concatenates 5-D cost volumes
//! (`torch.cat([gwc_volume, concat_volume], dim=1)`). **Layout**: `a` `[B, CA, ...]`,
//! `b` `[B, CB, ...]` with equal trailing axes flattened to `S`; output
//! `[B, CA + CB, ...]`. Grid: `B * (CA + CB) * ceil(S / BLOCK)`. Which input a channel
//! reads is arithmetic, not a branch: both are loaded, each masked to its own rows.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `out[b, c] = a[b, c]` for `c < CA`, else `b[b, c - CA]`.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn cat2_forward<T: Triton, D: Float, const BLOCK: i32>(
    a_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    CA: i32,
    CB: i32,
    S: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let ct = CA + CB;
    let tiles = (S + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let c = r1 % ct;
    let b = r1 / ct;
    // 0 for a's channels (c < CA), 1 for b's: c + ct - CA lies in [0, ct) or [ct, 2ct).
    let in_b = (c + ct - CA) / ct;
    let ca = c * (1 - in_b);
    let cb = (c - CA) * in_b;
    let s = T::arange(0, BLOCK) + tile * BLOCK;
    // Limit each load to its own input: a read only if in_b == 0, b only if 1.
    let la = s.lt(S * (1 - in_b));
    let lb = s.lt(S * in_b);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let va = T::load(
        a_ptr.add_offsets(s + (b * CA + ca) * S),
        Some(la),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let vb = T::load(
        b_ptr.add_offsets(s + (b * CB + cb) * S),
        Some(lb),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        out_ptr.add_offsets(s + (b * ct + c) * S),
        va + vb,
        Some(s.lt(S)),
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

/// Graph-level `torch.cat([a, b], dim=1)` for any rank.
pub struct Cat2Op<D: FloatBytes + Send + Sync + 'static> {
    /// Launch block size.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> Cat2Op<D> {
    /// Creates the op.
    pub fn new(block: i32) -> Self {
        Self {
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for Cat2Op<D> {
    fn name(&self) -> &str {
        "stereo.cat2"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let (a, b) = (input_shapes[0], input_shapes[1]);
        let mut s = a.clone();
        s[1] = match (a[1], b[1]) {
            (Some(x), Some(y)) => Some(x + y),
            _ => None,
        };
        s
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = Cat2Forward::<D>::new(self.block);
        Some((
            "cat2_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(Cat2RuntimeOp::<D> {
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`Cat2Op`].
pub struct Cat2RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for Cat2RuntimeOp<D> {
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
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let (a, b) = (inputs[0].1, inputs[1].1);
        assert_eq!(a[2..], b[2..], "cat2: trailing axes differ");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(a[1] as i32);
        visitor.visit_i32(b[1] as i32);
        visitor.visit_i32(output_shape[2..].iter().product::<usize>() as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let s: usize = output_shape[2..].iter().product();
        [
            (output_shape[0] * output_shape[1] * s.div_ceil(self.block as usize)) as u32,
            1,
            1,
        ]
    }
}
