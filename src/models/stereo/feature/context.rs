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

//! FoundationStereo's context network (`ContextNetDino`, `core/extractor.py`, with its
//! default `norm_fn='batch'`) and the post-processing its forward applies
//! (`core/foundation_stereo.py`):
//!
//! ```python
//! cnet_list = self.cnet(image1, vit_feat=vit_feat)      # (1/4, 1/8, 1/16)
//! net_list = [torch.tanh(x[0]) for x in cnet_list]      # GRU hidden states
//! inp_list = [torch.relu(x[1]) for x in cnet_list]      # context
//! inp_list = [self.cam(x) * x for x in inp_list]        # channel attention
//! att = [self.sam(x) for x in inp_list]                 # spatial attention
//! ```
//!
//! The network: a 7x7 conv, `layer1..3` of `ResidualBlock`s to 1/4 (for
//! `n_downsample=2`), the ViT feature concatenated in and fused by `conv2`
//! (`BasicConv`: bias-free 3x3, BatchNorm, LeakyReLU), then per level a head per output
//! (`ResidualBlock` + 3x3 conv at 1/4 and 1/8, a 3x3 conv at 1/16), with `layer4`,
//! `layer5` halving between levels. BatchNorm runs in inference mode on its running
//! statistics. Its unused `down` and `norm` modules are not built.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::{CustomData, SymTensor},
    name_scope::name_scope,
    nn::{Layer, batchnorm::BatchNorm2d, conv2d::Conv2d},
};

use crate::models::stereo::kernels::{
    channel_gate::ChannelGateOp, channel_mean_max::ChannelMeanMaxOp, global_pool::GlobalPoolOp,
};
use crate::models::stereo::update::ops::{add, cat, leaky, relu, sigmoid, tanh};

const BLOCK: i32 = 256;

/// Context network sizes.
#[derive(Clone, Copy, Debug)]
pub struct ContextDims {
    /// Output channels per level (1/4, 1/8, 1/16) of the hidden-state head
    /// (`hidden_dims`).
    pub hidden: [usize; 3],
    /// Output channels per level of the context head (`context_dims`).
    pub context: [usize; 3],
    /// `vit_feat` channels.
    pub vit: usize,
    /// `n_downsample`: the 1/4 map is reached by `layer2` and `layer3` at 2, and
    /// `conv1` too at 3.
    pub downsample: usize,
}

impl ContextDims {
    /// FoundationStereo's released configuration: `hidden_dims = [128] * 3`,
    /// `context_dims = hidden_dims`, `vit_size: vitl` (128 ViT channels),
    /// `n_downsample: 2`.
    pub fn foundation_stereo() -> Self {
        Self {
            hidden: [128; 3],
            context: [128; 3],
            vit: 128,
            downsample: 2,
        }
    }
}

/// `nn.Conv2d(c_in, c_out, k, stride, padding=k // 2)` under `name`.
fn conv_s<D: Float>(
    name: &str,
    x: &SymTensor,
    (c_in, c_out, k, stride): (usize, usize, usize, usize),
    bias: bool,
) -> SymTensor {
    let _g = name_scope(name);
    let p = k / 2;
    Conv2d::<D, SymTensor, SymTensor, 4>::new(c_in, c_out, (k, k), (stride, stride), (p, p), bias)
        .call(x.clone())
}

/// `nn.BatchNorm2d(c)` in inference mode, under `name`.
fn bn<D: Float>(name: &str, x: &SymTensor, c: usize) -> SymTensor {
    let _g = name_scope(name);
    BatchNorm2d::<D, SymTensor, SymTensor, 4>::new(c)
        .with_eps(1e-5)
        .call(x.clone())
}

/// `ResidualBlock(c_in, c_out, norm_fn='batch', stride)`:
/// `relu(down(x) + relu(bn2(conv2(relu(bn1(conv1(x)))))))`, with `down` a 1x1 strided
/// conv and BatchNorm (`downsample.0`, `downsample.1`) when the shape changes.
pub fn residual_block<D: Float>(
    x: &SymTensor,
    c_in: usize,
    c_out: usize,
    stride: usize,
) -> SymTensor {
    let y = relu::<D>(bn::<D>(
        "norm1",
        &conv_s::<D>("conv1", x, (c_in, c_out, 3, stride), true),
        c_out,
    ));
    let y = relu::<D>(bn::<D>(
        "norm2",
        &conv_s::<D>("conv2", &y, (c_out, c_out, 3, 1), true),
        c_out,
    ));
    let x = if stride == 1 && c_in == c_out {
        x.clone()
    } else {
        let _g = name_scope("downsample");
        bn::<D>(
            "1",
            &conv_s::<D>("0", x, (c_in, c_out, 1, stride), true),
            c_out,
        )
    };
    relu::<D>(add(&x, &y))
}

/// `_make_layer`: two `ResidualBlock`s, the first with `stride`, under `name`.
fn layer<D: Float>(
    name: &str,
    x: &SymTensor,
    c_in: usize,
    c_out: usize,
    stride: usize,
) -> SymTensor {
    let _g = name_scope(name);
    let x = {
        let _b = name_scope("0");
        residual_block::<D>(x, c_in, c_out, stride)
    };
    let _b = name_scope("1");
    residual_block::<D>(&x, c_out, c_out, 1)
}

/// `outputs04` / `outputs08` heads: `ResidualBlock(128, 128)` then a 3x3 conv.
fn head<D: Float>(name: String, x: &SymTensor, c_out: usize) -> SymTensor {
    let _g = name_scope(name);
    let y = {
        let _b = name_scope("0");
        residual_block::<D>(x, 128, 128, 1)
    };
    conv_s::<D>("1", &y, (128, c_out, 3, 1), true)
}

/// `ContextNetDino.forward(image, vit_feat)`: per level (1/4, 1/8, 1/16), the hidden
/// and context heads' outputs, before activation.
pub fn context_net<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: &ContextDims,
    image: &SymTensor,
    vit_feat: &SymTensor,
) -> [[SymTensor; 2]; 3] {
    let s = |on: bool| 1 + on as usize;
    let x = conv_s::<D>("conv1", image, (3, 64, 7, s(dims.downsample > 2)), true);
    let x = relu::<D>(bn::<D>("norm1", &x, 64));
    let x = layer::<D>("layer1", &x, 64, 64, 1);
    let x = layer::<D>("layer2", &x, 64, 96, s(dims.downsample > 1));
    let x = layer::<D>("layer3", &x, 96, 128, s(dims.downsample > 0));
    let x = cat(&[&x, vit_feat]);
    let x = {
        let _g = name_scope("conv2");
        let y = conv_s::<D>("conv", &x, (128 + dims.vit, 128, 3, 1), false);
        leaky::<D>(&bn::<D>("bn", &y, 128))
    };
    let outs = [dims.hidden, dims.context];
    let o04 = [0, 1].map(|i| head::<D>(format!("outputs04.{i}"), &x, outs[i][0]));
    let y = layer::<D>("layer4", &x, 128, 128, 2);
    let o08 = [0, 1].map(|i| head::<D>(format!("outputs08.{i}"), &y, outs[i][1]));
    let z = layer::<D>("layer5", &y, 128, 128, 2);
    let o16 =
        [0, 1].map(|i| conv_s::<D>(&format!("outputs16.{i}"), &z, (128, outs[i][2], 3, 1), true));
    [o04, o08, o16]
}

/// `ChannelAttentionEnhancement` (`cam`) applied as `cam(x) * x`:
/// `sigmoid(fc(avg_pool(x)) + fc(max_pool(x)))` gates each channel, `fc` a bias-free
/// 1x1 conv to `c / 16`, ReLU, and a 1x1 conv back (`fc.0`, `fc.2`).
pub fn channel_attention<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c: usize,
) -> SymTensor {
    let fc = |p: &SymTensor| {
        let _g = name_scope("fc");
        let h = relu::<D>(conv_s::<D>("0", p, (c, c / 16, 1, 1), false));
        conv_s::<D>("2", &h, (c / 16, c, 1, 1), false)
    };
    let avg = x.record_custom(CustomData::new(GlobalPoolOp::<D>::avg(BLOCK)), &[], None);
    let max = x.record_custom(CustomData::new(GlobalPoolOp::<D>::max(BLOCK)), &[], None);
    let g = sigmoid::<D>(add(&fc(&avg), &fc(&max)));
    x.record_custom(CustomData::new(ChannelGateOp::<D>::new(BLOCK)), &[&g], None)
}

/// `SpatialAttentionExtractor` (`sam`): `sigmoid(samconv(cat(mean_c x, max_c x)))`, a
/// bias-free 7x7 conv to one channel.
pub fn spatial_attention<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
) -> SymTensor {
    let s = x.record_custom(
        CustomData::new(ChannelMeanMaxOp::<D>::new(BLOCK)),
        &[],
        None,
    );
    sigmoid::<D>(conv_s::<D>("samconv", &s, (2, 1, 7, 1), false))
}

/// The context network's outputs as the refinement loop takes them: per level,
/// `net = tanh(hidden)`, `inp = cam(relu(context)) * relu(context)` and
/// `att = sam(inp)`. Records `cnet`, `cam` and `sam` under those scopes.
pub struct Context {
    /// GRU hidden states per level.
    pub net: [SymTensor; 3],
    /// Context features per level.
    pub inp: [SymTensor; 3],
    /// Spatial attention per level.
    pub att: [SymTensor; 3],
}

/// Builds [`Context`] from the left image and `vit_feat`.
pub fn context<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: &ContextDims,
    image: &SymTensor,
    vit_feat: &SymTensor,
) -> Context {
    let levels = {
        let _g = name_scope("cnet");
        context_net::<D>(dims, image, vit_feat)
    };
    let net = [0, 1, 2].map(|l| tanh::<D>(levels[l][0].clone()));
    let inp = [0, 1, 2].map(|l| {
        let x = relu::<D>(levels[l][1].clone());
        let _g = name_scope("cam");
        channel_attention::<D>(&x, dims.context[l])
    });
    let att = [0, 1, 2].map(|l| {
        let _g = name_scope("sam");
        spatial_attention::<D>(&inp[l])
    });
    Context { net, inp, att }
}
