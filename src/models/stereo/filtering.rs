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

//! FoundationStereo's cost-volume filtering (`core/foundation_stereo.py`), from the
//! combined volume to the geometry-encoding volume:
//!
//! ```python
//! comb_volume = self.corr_stem(comb_volume)                         # 32 -> 28 channels
//! comb_volume = self.corr_feature_att(comb_volume, features_left[0])
//! ```
//!
//! `corr_stem` is a 1x1x1 conv (with bias), a `BasicConv` 3-D (bias-free 3x3x3 conv,
//! BatchNorm3d, LeakyReLU) and two `ResnetBasicBlock3D`s (bias-free 3x3x3 convs with
//! BatchNorm3d, `relu(x + bn2(conv2(relu(bn1(conv1(x))))))`). `FeatureAtt` gates every
//! disparity with `sigmoid` of a 2-D map of the image features (`BasicConv` 1x1 to half
//! width, then a 1x1 conv with bias to the volume's channels). BatchNorm runs on its
//! running statistics. The 3-D layers use this crate's own conv and BatchNorm kernels:
//! teeny-kernels 0.3.1 lowers `Op::Conv3d` and `Op::BatchNorm3d` to kernels with no
//! runtime implementation.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::{CustomData, SymTensor},
    name_scope::name_scope,
    nn::Layer,
};

use crate::models::stereo::kernels::{
    batch_norm_inference::BatchNormInferenceOp,
    cat2::Cat2Op,
    channel_layer_norm::ChannelLayerNormOp,
    conv_transpose3d_k4s2p1::ConvTranspose3dK4s2p1Op,
    conv3d_direct::{Conv3dDirectOp, Conv3dSpec},
    disp_pos_embed::DispPosEmbedOp,
    disparity_regression::DisparityRegressionOp,
    head_attn_apply::HeadAttnApplyOp,
    head_attn_logits::HeadAttnLogitsOp,
    trilinear_upsample::TrilinearUpsampleOp,
    volume_gate::VolumeGateOp,
};
use crate::models::stereo::update::ops::{add, conv, conv_no_bias, gelu, leaky, relu};

/// FoundationStereo's `volume_dim`, the filtered volume's channels.
pub const VOLUME_DIM: usize = 28;

/// `nn.Conv3d(c_in, c_out, k, padding=k // 2)` under `name`.
pub fn conv3d<D: FloatBytes + Send + Sync + 'static>(
    name: &str,
    x: &SymTensor,
    c_in: usize,
    c_out: usize,
    k: usize,
    bias: bool,
) -> SymTensor {
    conv3d_spec::<D>(name, x, Conv3dSpec::cube(c_in, c_out, k, bias))
}

/// A 3-D convolution of any geometry under `name`.
pub fn conv3d_spec<D: FloatBytes + Send + Sync + 'static>(
    name: &str,
    x: &SymTensor,
    spec: Conv3dSpec,
) -> SymTensor {
    let _g = name_scope(name);
    x.record_custom(
        CustomData::new(Conv3dDirectOp::<D>::new(spec, 32)),
        &[],
        None,
    )
}

/// `nn.BatchNorm3d(c)` in inference mode under `name`.
pub fn bn3d<D: FloatBytes + Send + Sync + 'static>(
    name: &str,
    x: &SymTensor,
    c: usize,
) -> SymTensor {
    let _g = name_scope(name);
    x.record_custom(
        CustomData::new(BatchNormInferenceOp::<D>::new(c, 1e-5, 256)),
        &[],
        None,
    )
}

/// `BasicConv(c_in, c_out, is_3d=True, kernel_size=k, padding=k // 2)`: bias-free conv,
/// BatchNorm3d, LeakyReLU (`conv`, `bn`).
pub fn basic_conv3d<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c_in: usize,
    c_out: usize,
    k: usize,
) -> SymTensor {
    let y = conv3d::<D>("conv", x, c_in, c_out, k, false);
    leaky::<D>(&bn3d::<D>("bn", &y, c_out))
}

/// `ResnetBasicBlock3D(c, c, kernel_size=3, padding=1)`.
pub fn resnet_basic_block3d<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c: usize,
) -> SymTensor {
    let y = relu::<D>(bn3d::<D>(
        "bn1",
        &conv3d::<D>("conv1", x, c, c, 3, false),
        c,
    ));
    let y = bn3d::<D>("bn2", &conv3d::<D>("conv2", &y, c, c, 3, false), c);
    relu::<D>(add(&y, x))
}

/// `corr_stem` on the combined volume (`c_in` channels) to `c` channels.
pub fn corr_stem<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c_in: usize,
    c: usize,
) -> SymTensor {
    let x = conv3d::<D>("0", x, c_in, c, 1, true);
    let x = {
        let _g = name_scope("1");
        basic_conv3d::<D>(&x, c, c, 3)
    };
    let x = {
        let _g = name_scope("2");
        resnet_basic_block3d::<D>(&x, c)
    };
    let _g = name_scope("3");
    resnet_basic_block3d::<D>(&x, c)
}

/// `FeatureAtt(cv_chan, feat_chan)` on volume `cv` with image features `feat`
/// (`feat_att.0` the `BasicConv`, `feat_att.1` the conv).
pub fn feature_att<D: Float + FloatBytes + Send + Sync + 'static>(
    cv: &SymTensor,
    feat: &SymTensor,
    cv_chan: usize,
    feat_chan: usize,
) -> SymTensor {
    let g = {
        let _f = name_scope("feat_att");
        let h = {
            let _b = name_scope("0");
            let h = conv_no_bias::<D>("conv", feat_chan, feat_chan / 2, 1)(feat);
            let _n = name_scope("bn");
            let bn = teeny_core::nn::batchnorm::BatchNorm2d::<D, SymTensor, SymTensor, 4>::new(
                feat_chan / 2,
            )
            .with_eps(1e-5);
            leaky::<D>(&bn.call(h))
        };
        conv::<D>("1", feat_chan / 2, cv_chan, 1)(&h)
    };
    cv.record_custom(CustomData::new(VolumeGateOp::<D>::new(256)), &[&g], None)
}

/// `BasicConv(c_in, c_out, is_3d=True, kernel_size=k, stride, padding=k // 2)`.
fn basic_conv3d_s<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c_in: usize,
    c_out: usize,
    k: usize,
    stride: usize,
) -> SymTensor {
    let spec = Conv3dSpec {
        stride: [stride; 3],
        ..Conv3dSpec::cube(c_in, c_out, k, false)
    };
    let y = conv3d_spec::<D>("conv", x, spec);
    leaky::<D>(&bn3d::<D>("bn", &y, c_out))
}

/// `BasicConv(c_in, c_out, deconv=True, is_3d=True, kernel_size=4, stride=2,
/// padding=1)`: bias-free transposed conv, BatchNorm3d, LeakyReLU.
fn basic_deconv3d<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c_in: usize,
    c_out: usize,
) -> SymTensor {
    let y = {
        let _g = name_scope("conv");
        x.record_custom(
            CustomData::new(ConvTranspose3dK4s2p1Op::<D>::new(c_in, c_out, 32)),
            &[],
            None,
        )
    };
    leaky::<D>(&bn3d::<D>("bn", &y, c_out))
}

/// `Conv3dNormActReduced(c_in, c_out, kernel_size=3, kernel_disp=17)`: a `(1, 3, 3)`
/// conv then a `(17, 1, 1)` conv, each with bias, BatchNorm3d and ReLU.
pub fn conv3d_reduced<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c_in: usize,
    c_out: usize,
) -> SymTensor {
    let spatial = Conv3dSpec {
        kernel: [1, 3, 3],
        padding: [0, 1, 1],
        ..Conv3dSpec::cube(c_in, c_out, 1, true)
    };
    let disp = Conv3dSpec {
        kernel: [17, 1, 1],
        padding: [8, 0, 0],
        ..Conv3dSpec::cube(c_out, c_out, 1, true)
    };
    let x = {
        let _g = name_scope("conv1");
        relu::<D>(bn3d::<D>("1", &conv3d_spec::<D>("0", x, spatial), c_out))
    };
    let _g = name_scope("conv2");
    relu::<D>(bn3d::<D>("1", &conv3d_spec::<D>("0", &x, disp), c_out))
}

/// One `FlashAttentionTransformerEncoderLayer` on a volume (post-norm, GELU, as
/// FoundationStereo runs it): the attention is per voxel across heads (see
/// [`HeadAttnLogitsOp`]), every linear a 1x1x1 conv, every LayerNorm over channels.
pub fn transformer_layer<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c: usize,
    heads: usize,
    ffn: usize,
) -> SymTensor {
    let ln = |name: &str, t: &SymTensor| {
        let _g = name_scope(name);
        t.record_custom(
            CustomData::new(ChannelLayerNormOp::<D>::new(c, 1e-5, 256)),
            &[],
            None,
        )
    };
    let a = {
        let _g = name_scope("self_attn");
        let q = conv3d::<D>("q_proj", x, c, c, 1, true);
        let k = conv3d::<D>("k_proj", x, c, c, 1, true);
        let v = conv3d::<D>("v_proj", x, c, c, 1, true);
        let hd = c / heads;
        let l = q.record_custom(
            CustomData::new(HeadAttnLogitsOp::<D>::new(heads, hd, 256)),
            &[&k],
            None,
        );
        let o = l.record_custom(
            CustomData::new(HeadAttnApplyOp::<D>::new(heads, hd, 256)),
            &[&v],
            None,
        );
        conv3d::<D>("out_proj", &o, c, c, 1, true)
    };
    let x = ln("norm1", &add(x, &a));
    let h = conv3d::<D>("linear1", &x, c, ffn, 1, true);
    let h = conv3d::<D>("linear2", &gelu::<D>(&h), ffn, c, 1, true);
    ln("norm2", &add(&x, &h))
}

/// `CostVolumeDisparityAttention(d_model=c, nhead=4, dim_feedforward=c,
/// num_transformer=4)`: the disparity embedding, then the layers (`sa.{i}`).
pub fn disparity_attention<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c: usize,
) -> SymTensor {
    let mut x = x.record_custom(CustomData::new(DispPosEmbedOp::<D>::new(256)), &[], None);
    for i in 0..4 {
        let _g = name_scope(format!("sa.{i}"));
        x = transformer_layer::<D>(&x, c, 4, c);
    }
    x
}

/// `hourglass(in_channels=c, feat_dims)` (`cost_agg`) on volume `x` with the left
/// features at 1/4, 1/8, 1/16 and 1/32 (`feat_dims` channels).
pub fn hourglass<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    features: [&SymTensor; 4],
    c: usize,
    feat_dims: [usize; 4],
) -> SymTensor {
    let down = |name: &str, x: &SymTensor, ci: usize, co: usize| {
        let _g = name_scope(name);
        let y = {
            let _b = name_scope("0");
            basic_conv3d_s::<D>(x, ci, co, 3, 2)
        };
        let _r = name_scope("1");
        conv3d_reduced::<D>(&y, co, co)
    };
    let fatt = |name: &str, cv: &SymTensor, f: &SymTensor, cc: usize, fc: usize| {
        let _g = name_scope(name);
        feature_att::<D>(cv, f, cc, fc)
    };
    let agg = |name: &str, x: &SymTensor, ci: usize, co: usize| {
        let _g = name_scope(name);
        let y = {
            let _b = name_scope("0");
            basic_conv3d_s::<D>(x, ci, co, 1, 1)
        };
        let y = {
            let _r = name_scope("1");
            conv3d_reduced::<D>(&y, co, co)
        };
        let _r = name_scope("2");
        conv3d_reduced::<D>(&y, co, co)
    };
    let up = |name: &str, x: &SymTensor, ci: usize, co: usize| {
        let _g = name_scope(name);
        basic_deconv3d::<D>(x, ci, co)
    };
    let cat = |a: &SymTensor, b: &SymTensor| {
        a.record_custom(CustomData::new(Cat2Op::<D>::new(256)), &[b], None)
    };

    let conv1 = down("conv1", x, c, 2 * c);
    let conv1 = fatt("feature_att_8", &conv1, features[1], 2 * c, feat_dims[1]);
    let conv2 = down("conv2", &conv1, 2 * c, 4 * c);
    let conv2 = fatt("feature_att_16", &conv2, features[2], 4 * c, feat_dims[2]);
    let conv3 = down("conv3", &conv2, 4 * c, 6 * c);
    let conv3 = fatt("feature_att_32", &conv3, features[3], 6 * c, feat_dims[3]);

    let conv3_up = up("conv3_up", &conv3, 6 * c, 4 * c);
    let conv2 = agg("agg_0", &cat(&conv3_up, &conv2), 8 * c, 4 * c);
    let conv2 = fatt(
        "feature_att_up_16",
        &conv2,
        features[2],
        4 * c,
        feat_dims[2],
    );
    let conv2_up = up("conv2_up", &conv2, 4 * c, 2 * c);
    let conv1 = agg("agg_1", &cat(&conv2_up, &conv1), 4 * c, 2 * c);
    let conv1 = fatt("feature_att_up_8", &conv1, features[1], 2 * c, feat_dims[1]);
    let conv = up("conv1_up", &conv1, 2 * c, c);

    let patch = {
        let _g = name_scope("conv_patch");
        let spec = Conv3dSpec {
            kernel: [4; 3],
            stride: [4; 3],
            padding: [0; 3],
            groups: c,
            ..Conv3dSpec::cube(c, c, 4, true)
        };
        bn3d::<D>("1", &conv3d_spec::<D>("0", x, spec), c)
    };
    let att = {
        let _g = name_scope("atts.4");
        disparity_attention::<D>(&patch, c)
    };
    let att = att.record_custom(
        CustomData::new(TrilinearUpsampleOp::<D>::new(4, 32)),
        &[],
        None,
    );
    let conv = add(&conv, &att);
    let _g = name_scope("conv_out");
    let conv = {
        let _r = name_scope("0");
        conv3d_reduced::<D>(&conv, c, c)
    };
    let _r = name_scope("1");
    conv3d_reduced::<D>(&conv, c, c)
}

/// `classifier` on the filtered volume (`c` channels): `BasicConv` 3-D to `c / 2`, a
/// `ResnetBasicBlock3D`, and a 7x7x7 conv with bias to one channel, `[B, 1, D, H, W]`.
pub fn classifier<D: Float + FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    c: usize,
) -> SymTensor {
    let h = c / 2;
    let x = {
        let _g = name_scope("0");
        basic_conv3d::<D>(x, c, h, 3)
    };
    let x = {
        let _g = name_scope("1");
        resnet_basic_block3d::<D>(&x, h)
    };
    conv3d::<D>("2", &x, h, 1, 7, true)
}

/// `disparity_regression(F.softmax(logits.squeeze(1), dim=1))`: the initial
/// disparity `[B, 1, H, W]` at 1/4 resolution from the classifier's logits.
pub fn init_disp<D: FloatBytes + Send + Sync + 'static>(logits: &SymTensor) -> SymTensor {
    logits.record_custom(
        CustomData::new(DisparityRegressionOp::<D>::new(256)),
        &[],
        None,
    )
}
