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
    conv3d_direct::{Conv3dDirectOp, Conv3dSpec},
    volume_gate::VolumeGateOp,
};
use crate::models::stereo::update::ops::{add, conv, conv_no_bias, leaky, relu};

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
