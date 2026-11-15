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

//! `nn.ConvTranspose3d(cin, cout, 4, stride=2, padding=1, bias=False)`: an exact 2x
//! upsample of a cost volume, as the `hourglass`'s `conv{1,2,3}_up` use it
//! (`core/foundation_stereo.py`).
//!
//! Written as a gather: output `o` along an axis takes input `i = (o + 1 - k) / 2` for
//! the taps `k` where `o + 1 - k` is even and `i` in range, two of the four. The loop
//! visits only those parity-matched taps, `CIN * 8` per output instead of `CIN * 64`.
//! **Layout**: input `[B, CIN, D, H, W]`, weight `[CIN, COUT, 4, 4, 4]` (PyTorch's),
//! output `[B, COUT, 2D, 2H, 2W]`. Grid: `B * COUT * 2D * 2H * ceil(2W / BLOCK)`, lanes
//! on output columns; out-of-range taps are weighted 0.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The transposed convolution above.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn conv_transpose3d_k4s2p1_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    CIN: i32,
    COUT: i32,
    DD: i32,
    HH: i32,
    WW: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let OD = 2 * DD;
    let OH = 2 * HH;
    let OW = 2 * WW;
    let tiles = (OW + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let oh = r1 % OH;
    let r2 = r1 / OH;
    let od = r2 % OD;
    let r3 = r2 / OD;
    let o = r3 % COUT;
    let b = r3 / COUT;

    let ow = T::arange(0, BLOCK) + tile * BLOCK;
    let m = ow.lt(OW);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let izero = T::zeros::<i32>(&[BLOCK]);
    let half = T::full(&[BLOCK], D::from_f64(0.5));

    // Only taps whose parity matches the output reach it: along depth and row that is
    // kd = kd0 + 2a, kh = kh0 + 2b (scalars per program), along columns kw = kw0 + 2c
    // with kw0 per lane. Input index = (o + 1 - k0) / 2 - {a, b, c}.
    let kd0 = (od + 1) % 2;
    let kh0 = (oh + 1) % 2;
    let qd = (od + 1 - kd0) / 2;
    let qh = (oh + 1 - kh0) / 2;
    let nw = ow + 1;
    let qw = T::cast::<D, i32>(
        T::floor(T::cast::<i32, D>(nw, None, false) * half),
        None,
        false,
    );
    let kw0 = nw + qw * (-2); // 0 or 1 per lane

    let mut acc = T::zeros::<D>(&[BLOCK]);
    for t in 0..CIN * 8 {
        let i = t / 8;
        let a = (t / 4) % 2;
        let bb = (t / 2) % 2;
        let c = t % 2;
        let idt = izero + (qd - a);
        let iht = izero + (qh - bb);
        let iw = qw + (-c);
        let idc = T::maximum(T::minimum(idt, izero + (DD - 1)), izero);
        let ihc = T::maximum(T::minimum(iht, izero + (HH - 1)), izero);
        let iwc = T::maximum(T::minimum(iw, izero + (WW - 1)), izero);
        let x = T::load(
            x_ptr.add_offsets(iwc + ((idc + (b * CIN + i) * DD) * HH + ihc) * WW),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let x = T::where_(idt.ge(0), x, zeros);
        let x = T::where_(idt.lt(DD), x, zeros);
        let x = T::where_(iht.ge(0), x, zeros);
        let x = T::where_(iht.lt(HH), x, zeros);
        let x = T::where_(iw.ge(0), x, zeros);
        let x = T::where_(iw.lt(WW), x, zeros);
        let kd = kd0 + 2 * a;
        let kh = kh0 + 2 * bb;
        let w = T::load(
            w_ptr.add_offsets(kw0 + ((i * COUT + o) * 64 + kd * 16 + kh * 4 + 2 * c)),
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
        out_ptr.add_offsets(ow + (((b * COUT + o) * OD + od) * OH + oh) * OW),
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

/// Graph-level bias-free `ConvTranspose3d(cin, cout, 4, stride=2, padding=1)`. Weight
/// slot `weight` `[cin, cout, 4, 4, 4]`.
pub struct ConvTranspose3dK4s2p1Op<D: FloatBytes + Send + Sync + 'static> {
    /// Input channels.
    pub cin: usize,
    /// Output channels.
    pub cout: usize,
    /// Launch block size along output columns.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ConvTranspose3dK4s2p1Op<D> {
    /// Creates the op.
    pub fn new(cin: usize, cout: usize, block: i32) -> Self {
        Self {
            cin,
            cout,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ConvTranspose3dK4s2p1Op<D> {
    fn name(&self) -> &str {
        "stereo.conv_transpose3d_k4s2p1"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let x = input_shapes[0];
        let dbl = |d: Option<usize>| d.map(|n| 2 * n);
        vec![x[0], Some(self.cout), dbl(x[2]), dbl(x[3]), dbl(x[4])]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ConvTranspose3dK4s2p1Forward::<D>::new(self.block);
        Some((
            "conv_transpose3d_k4s2p1_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ConvTranspose3dK4s2p1RuntimeOp::<D> {
                cin: self.cin,
                cout: self.cout,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ConvTranspose3dK4s2p1Op`].
pub struct ConvTranspose3dK4s2p1RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    cin: usize,
    cout: usize,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ConvTranspose3dK4s2p1RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.cin, self.cout, 4, 4, 4]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight"]
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
        assert_eq!(x[1], self.cin, "conv_transpose3d: input channels");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.cin as i32);
        visitor.visit_i32(self.cout as i32);
        visitor.visit_i32(x[2] as i32);
        visitor.visit_i32(x[3] as i32);
        visitor.visit_i32(x[4] as i32);
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
