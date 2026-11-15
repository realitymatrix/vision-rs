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

//! Direct 3-D convolution, `nn.Conv3d(c_in, c_out, (KD, KH, KW), stride, padding,
//! groups, bias)`, for FoundationStereo's cost-volume filtering. teeny's `Op::Conv3d`
//! lowers to a kernel with no runtime implementation (teeny-kernels 0.3.1).
//!
//! One kernel covers every shape the filtering uses: 1x1x1 and 3x3x3, the separable
//! `(1, k, k)` and `(17, 1, 1)` of `Conv3dNormActReduced`, stride 2, the classifier's
//! 7x7x7 and the depthwise `(4, 4, 4)` stride-4 `conv_patch` (`groups = c`).
//!
//! **Layout**: input `[B, CIN, D, H, W]`, weight `[COUT, CIN / G, KD, KH, KW]`
//! (PyTorch's), bias `[COUT]` (zeros when the layer has none), output
//! `[B, COUT, OD, OH, OW]`. Grid: `B * COUT * OD * OH * ceil(OW / BLOCK)`, lanes on
//! output columns; the `CIN / G * KD * KH * KW` taps are one flat loop (no nested
//! loops, teenygrad/teenygrad#156), each tap's weight broadcast across the lanes.
//!
//! When `COUT / G` is a multiple of 4, `conv3d_direct4_forward` runs instead: each
//! program computes four consecutive output channels, so every input load, its index
//! arithmetic and its bounds masks are shared by four accumulators (grid divided by 4).

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The convolution above.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn conv3d_direct_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    CIN: i32,
    COUT: i32,
    G: i32,
    DD: i32,
    HH: i32,
    WW: i32,
    OD: i32,
    OH: i32,
    OW: i32,
    KER: i32,
    STR: i32,
    PAD: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // Kernel size, stride and padding arrive packed as `d << 10 | h << 5 | w` (kernels
    // take at most 20 arguments).
    let KD = KER / 1024;
    let KH = (KER / 32) % 32;
    let KW = KER % 32;
    let SD = STR / 1024;
    let SH = (STR / 32) % 32;
    let SW = STR % 32;
    let PD = PAD / 1024;
    let PH = (PAD / 32) % 32;
    let PW = PAD % 32;
    let cin_g = CIN / G;
    let cout_g = COUT / G;
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
    let grp = o / cout_g;

    let ow = T::arange(0, BLOCK) + tile * BLOCK;
    let m = ow.lt(OW);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let izero = T::zeros::<i32>(&[BLOCK]);
    let iw0 = ow * SW + (-PW);

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
    let ktaps = KD * KH * KW;
    for t in 0..cin_g * ktaps {
        let ci = t / ktaps;
        let k = t % ktaps;
        let kd = k / (KH * KW);
        let kh = (k / KW) % KH;
        let kw = k % KW;
        let id = od * SD - PD + kd;
        let ih = oh * SH - PH + kh;
        // Bounds as lane tensors: teenyc compiles no scalar branches or comparisons.
        let idt = izero + id;
        let iht = izero + ih;
        let iw = iw0 + kw;
        let idc = T::maximum(T::minimum(idt, izero + (DD - 1)), izero);
        let ihc = T::maximum(T::minimum(iht, izero + (HH - 1)), izero);
        let iwc = T::maximum(T::minimum(iw, izero + (WW - 1)), izero);
        let c = grp * cin_g + ci;
        let x = T::load(
            x_ptr.add_offsets(iwc + ((idc + (b * CIN + c) * DD) * HH + ihc) * WW),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let x = T::where_(iw.ge(0), x, zeros);
        let x = T::where_(iw.lt(WW), x, zeros);
        let x = T::where_(idt.ge(0), x, zeros);
        let x = T::where_(idt.lt(DD), x, zeros);
        let x = T::where_(iht.ge(0), x, zeros);
        let x = T::where_(iht.lt(HH), x, zeros);
        let w = T::load(
            w_ptr.add_offsets(izero + ((o * cin_g + ci) * ktaps + k)),
            None,
            None,
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

/// The convolution above, four output channels per program (`COUT / G` a multiple
/// of 4): every input load and bounds check feeds four accumulators.
// `#[kernel]` requires the `Triton` bound inline on the type parameter, while the
// `I32Tensor == Tensor<i32>` equality can only be written in the where clause.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn conv3d_direct4_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    CIN: i32,
    COUT: i32,
    G: i32,
    DD: i32,
    HH: i32,
    WW: i32,
    OD: i32,
    OH: i32,
    OW: i32,
    KER: i32,
    STR: i32,
    PAD: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // Kernel size, stride and padding arrive packed as `d << 10 | h << 5 | w` (kernels
    // take at most 20 arguments).
    let KD = KER / 1024;
    let KH = (KER / 32) % 32;
    let KW = KER % 32;
    let SD = STR / 1024;
    let SH = (STR / 32) % 32;
    let SW = STR % 32;
    let PD = PAD / 1024;
    let PH = (PAD / 32) % 32;
    let PW = PAD % 32;
    let cin_g = CIN / G;
    let cout_g = COUT / G;
    let tiles = (OW + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let tile = pid % tiles;
    let r1 = pid / tiles;
    let oh = r1 % OH;
    let r2 = r1 / OH;
    let od = r2 % OD;
    let r3 = r2 / OD;
    let o = (r3 % (COUT / 4)) * 4;
    let b = r3 / (COUT / 4);
    let grp = o / cout_g;

    let ow = T::arange(0, BLOCK) + tile * BLOCK;
    let m = ow.lt(OW);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let izero = T::zeros::<i32>(&[BLOCK]);
    let iw0 = ow * SW + (-PW);

    let mut acc0 = T::load(
        b_ptr.add_offsets(izero + o),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let mut acc1 = T::load(
        b_ptr.add_offsets(izero + (o + 1)),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let mut acc2 = T::load(
        b_ptr.add_offsets(izero + (o + 2)),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let mut acc3 = T::load(
        b_ptr.add_offsets(izero + (o + 3)),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let ktaps = KD * KH * KW;
    for t in 0..cin_g * ktaps {
        let ci = t / ktaps;
        let k = t % ktaps;
        let kd = k / (KH * KW);
        let kh = (k / KW) % KH;
        let kw = k % KW;
        let id = od * SD - PD + kd;
        let ih = oh * SH - PH + kh;
        // Bounds as lane tensors: teenyc compiles no scalar branches or comparisons.
        let idt = izero + id;
        let iht = izero + ih;
        let iw = iw0 + kw;
        let idc = T::maximum(T::minimum(idt, izero + (DD - 1)), izero);
        let ihc = T::maximum(T::minimum(iht, izero + (HH - 1)), izero);
        let iwc = T::maximum(T::minimum(iw, izero + (WW - 1)), izero);
        let c = grp * cin_g + ci;
        let x = T::load(
            x_ptr.add_offsets(iwc + ((idc + (b * CIN + c) * DD) * HH + ihc) * WW),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let x = T::where_(iw.ge(0), x, zeros);
        let x = T::where_(iw.lt(WW), x, zeros);
        let x = T::where_(idt.ge(0), x, zeros);
        let x = T::where_(idt.lt(DD), x, zeros);
        let x = T::where_(iht.ge(0), x, zeros);
        let x = T::where_(iht.lt(HH), x, zeros);
        let w0 = T::load(
            w_ptr.add_offsets(izero + ((o * cin_g + ci) * ktaps + k)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        acc0 = acc0 + x * w0;
        let w1 = T::load(
            w_ptr.add_offsets(izero + (((o + 1) * cin_g + ci) * ktaps + k)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        acc1 = acc1 + x * w1;
        let w2 = T::load(
            w_ptr.add_offsets(izero + (((o + 2) * cin_g + ci) * ktaps + k)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        acc2 = acc2 + x * w2;
        let w3 = T::load(
            w_ptr.add_offsets(izero + (((o + 3) * cin_g + ci) * ktaps + k)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        acc3 = acc3 + x * w3;
    }
    T::store(
        out_ptr.add_offsets(ow + (((b * COUT + o) * OD + od) * OH + oh) * OW),
        acc0,
        Some(m),
        &[],
        None,
        None,
    );
    T::store(
        out_ptr.add_offsets(ow + (((b * COUT + o + 1) * OD + od) * OH + oh) * OW),
        acc1,
        Some(m),
        &[],
        None,
        None,
    );
    T::store(
        out_ptr.add_offsets(ow + (((b * COUT + o + 2) * OD + od) * OH + oh) * OW),
        acc2,
        Some(m),
        &[],
        None,
        None,
    );
    T::store(
        out_ptr.add_offsets(ow + (((b * COUT + o + 3) * OD + od) * OH + oh) * OW),
        acc3,
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

/// 3-D convolution geometry.
#[derive(Clone, Copy, Debug)]
pub struct Conv3dSpec {
    /// Input channels.
    pub cin: usize,
    /// Output channels.
    pub cout: usize,
    /// Kernel `(d, h, w)`.
    pub kernel: [usize; 3],
    /// Stride `(d, h, w)`.
    pub stride: [usize; 3],
    /// Padding `(d, h, w)`.
    pub padding: [usize; 3],
    /// Groups.
    pub groups: usize,
    /// Whether the layer has a bias (a `bias` slot; otherwise zeros are bound).
    pub bias: bool,
}

impl Conv3dSpec {
    /// A cubic `k` kernel, stride 1, padding `k / 2`, one group.
    pub fn cube(cin: usize, cout: usize, k: usize, bias: bool) -> Self {
        Self {
            cin,
            cout,
            kernel: [k; 3],
            stride: [1; 3],
            padding: [k / 2; 3],
            groups: 1,
            bias,
        }
    }

    /// Whether the four-channel kernel applies: `cout / groups` a multiple of 4.
    fn blocked(&self) -> bool {
        (self.cout / self.groups).is_multiple_of(4)
    }

    fn out_size(&self, i: usize, n: usize) -> usize {
        (n + 2 * self.padding[i] - self.kernel[i]) / self.stride[i] + 1
    }
}

/// Graph-level 3-D convolution: `[B, CIN, D, H, W]` to `[B, COUT, OD, OH, OW]`. Weight
/// slots `weight` `[COUT, CIN / groups, KD, KH, KW]` and, with a bias, `bias`.
pub struct Conv3dDirectOp<D: FloatBytes + Send + Sync + 'static> {
    /// The geometry.
    pub spec: Conv3dSpec,
    /// Launch block size along output columns.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> Conv3dDirectOp<D> {
    /// Creates the op.
    pub fn new(spec: Conv3dSpec, block: i32) -> Self {
        assert!(
            spec.cin.is_multiple_of(spec.groups) && spec.cout.is_multiple_of(spec.groups),
            "conv3d groups"
        );
        Self {
            spec,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for Conv3dDirectOp<D> {
    fn name(&self) -> &str {
        "stereo.conv3d_direct"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let x = input_shapes[0];
        let s = &self.spec;
        vec![
            x[0],
            Some(s.cout),
            x[2].map(|n| s.out_size(0, n)),
            x[3].map(|n| s.out_size(1, n)),
            x[4].map(|n| s.out_size(2, n)),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let (name, source) = if self.spec.blocked() {
            let k = Conv3dDirect4Forward::<D>::new(self.block);
            ("conv3d_direct4_forward", k.source)
        } else {
            let k = Conv3dDirectForward::<D>::new(self.block);
            ("conv3d_direct_forward", k.source)
        };
        Some((
            name.to_string(),
            source,
            "entry_point".to_string(),
            Arc::new(Conv3dDirectRuntimeOp::<D> {
                spec: self.spec,
                block: self.block,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// `[d, h, w]` as `d << 10 | h << 5 | w`, the kernel's packed form.
fn pack(v: [usize; 3]) -> i32 {
    assert!(
        v.iter().all(|&x| x < 32),
        "conv3d: kernel, stride or padding >= 32"
    );
    ((v[0] << 10) | (v[1] << 5) | v[2]) as i32
}

/// Runtime dispatch for [`Conv3dDirectOp`].
pub struct Conv3dDirectRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    spec: Conv3dSpec,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for Conv3dDirectRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        let s = &self.spec;
        let w = vec![
            s.cout,
            s.cin / s.groups,
            s.kernel[0],
            s.kernel[1],
            s.kernel[2],
        ];
        // A bias-free layer still binds a `bias` buffer: unnamed, so no weight loads
        // into it, and zero-filled by `param_init_data`.
        vec![w, vec![s.cout]]
    }

    fn param_init_data(&self, param_idx: usize) -> Option<Vec<u8>> {
        (param_idx == 1 && !self.spec.bias)
            .then(|| vec![0u8; self.spec.cout * core::mem::size_of::<D>()])
    }

    fn param_names(&self) -> &'static [&'static str] {
        if self.spec.bias {
            &["weight", "bias"]
        } else {
            &["weight", ""]
        }
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
        let x = inputs[0].1;
        let s = &self.spec;
        assert_eq!(x[1], s.cin, "conv3d: input channels");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(output);
        for v in [s.cin, s.cout, s.groups, x[2], x[3], x[4]] {
            visitor.visit_i32(v as i32);
        }
        for v in &output_shape[2..5] {
            visitor.visit_i32(*v as i32);
        }
        for v in [s.kernel, s.stride, s.padding] {
            visitor.visit_i32(pack(v));
        }
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = output_shape[4].div_ceil(self.block as usize);
        let per = if self.spec.blocked() { 4 } else { 1 };
        [
            (output_shape[..4].iter().product::<usize>() / per * tiles) as u32,
            1,
            1,
        ]
    }
}
