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

//! `nn.ConvTranspose2d(cin, cout, kernel_size=S, stride=S, padding=0)`: an exact
//! `S`x upsample whose kernel windows do not overlap, as the Depth Anything DPT head
//! uses for its first two `resize_layers` (`S` = 4 and 2).
//!
//! Each output pixel takes exactly one input pixel, `(oy / S, ox / S)`, through kernel
//! tap `(oy % S, ox % S)`:
//! `out[b, o, oy, ox] = bias[o] + sum_i x[b, i, oy / S, ox / S] * w[i, o, oy % S, ox % S]`.
//! A gather, so no atomics. **Layout**: input `[B, CIN, H, W]`, weight
//! `[CIN, COUT, S, S]` (PyTorch's), output `[B, COUT, S*H, S*W]`. Grid:
//! `B * COUT * S*H * ceil(S*W / BLOCK)`, lanes on columns of one output row; the
//! column's source by float floor (exact at these sizes), avoiding tensor integer
//! division.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The transposed convolution above.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn conv_transpose_patch_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    CIN: i32,
    COUT: i32,
    H: i32,
    W: i32,
    S: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let OH = H * S;
    let OW = W * S;
    let tiles = (OW + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let oy = r1 % OH;
    let r2 = r1 / OH;
    let o = r2 % COUT;
    let b = r2 / COUT;
    let iy = oy / S;
    let ky = oy % S;

    let ox = T::arange(0, BLOCK) + tile * BLOCK;
    let m = ox.lt(OW);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let izero = T::zeros::<i32>(&[BLOCK]);
    let half = T::full(&[BLOCK], D::from_f64(0.5));
    let ix = T::cast::<D, i32>(
        T::floor(
            (T::cast::<i32, D>(ox, None, false) + half)
                * T::full(&[BLOCK], D::from_f64(1.0 / S as f64)),
        ),
        None,
        false,
    );
    let kx = ox + ix * (-S);

    let mut acc = T::load(
        b_ptr.add_offsets(izero + o),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    for i in 0..CIN {
        let x = T::load(
            x_ptr.add_offsets(ix + ((b * CIN + i) * H + iy) * W),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let w = T::load(
            w_ptr.add_offsets(kx + ((i * COUT + o) * S + ky) * S),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + x * w;
    }
    T::store(
        out_ptr.add_offsets(ox + ((b * COUT + o) * OH + oy) * OW),
        acc,
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

/// Graph-level `ConvTranspose2d(cin, cout, s, stride=s)` with bias: `[B, CIN, H, W]` to
/// `[B, COUT, s*H, s*W]`. Weight slots `weight` `[CIN, COUT, s, s]` and `bias` `[COUT]`.
pub struct ConvTransposePatchOp<D: FloatBytes + Send + Sync + 'static> {
    /// Input channels.
    pub cin: usize,
    /// Output channels.
    pub cout: usize,
    /// Kernel size and stride.
    pub s: usize,
    /// Launch block size along output columns.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ConvTransposePatchOp<D> {
    /// Creates the op.
    pub fn new(cin: usize, cout: usize, s: usize, block: i32) -> Self {
        Self {
            cin,
            cout,
            s,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ConvTransposePatchOp<D> {
    fn name(&self) -> &str {
        "stereo.conv_transpose_patch"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let x = input_shapes[0];
        let s = self.s;
        vec![
            x[0],
            Some(self.cout),
            x[2].map(|h| h * s),
            x[3].map(|w| w * s),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ConvTransposePatchForward::<D>::new(self.block);
        Some((
            "conv_transpose_patch_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ConvTransposePatchRuntimeOp::<D> {
                cin: self.cin,
                cout: self.cout,
                s: self.s,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ConvTransposePatchOp`].
pub struct ConvTransposePatchRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    cin: usize,
    cout: usize,
    s: usize,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ConvTransposePatchRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.cin, self.cout, self.s, self.s], vec![self.cout]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "bias"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let x = inputs[0].1;
        assert_eq!(x[1], self.cin, "conv_transpose_patch: input channels");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.cin as i32);
        visitor.visit_i32(self.cout as i32);
        visitor.visit_i32(x[2] as i32);
        visitor.visit_i32(x[3] as i32);
        visitor.visit_i32(self.s as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = output_shape[3].div_ceil(self.block as usize);
        [
            (output_shape[0] * self.cout * output_shape[2] * tiles) as u32,
            1,
            1,
        ]
    }
}
