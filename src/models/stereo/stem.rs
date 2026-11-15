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

//! FoundationStereo's `stem_2` (`core/foundation_stereo.py`), the 1/2-resolution image
//! features that the final upsampling (`upsample_disp`) concatenates as `stem_2x`:
//!
//! ```python
//! stem_2 = nn.Sequential(
//!     BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),   # bias-free conv, IN, LeakyReLU
//!     nn.Conv2d(32, 32, 3, 1, 1, bias=False),
//!     nn.InstanceNorm2d(32), nn.ReLU())
//! ```
//!
//! The InstanceNorms have no affine parameters (PyTorch's defaults).

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::SymTensor,
    name_scope::name_scope,
    nn::{Layer, conv2d::Conv2d},
};

use crate::models::stereo::feature::decoder::instance_norm;
use crate::models::stereo::update::ops::{conv_no_bias, leaky, relu};

/// `stem_2` channels.
pub const STEM_2_DIM: usize = 32;

/// `stem_2(image)`: `[B, 3, H, W]` to `[B, 32, ceil(H / 2), ceil(W / 2)]`, weights
/// under the caller's scope (`0.conv`, `1`).
pub fn stem_2<D: Float + FloatBytes + Send + Sync + 'static>(image: &SymTensor) -> SymTensor {
    let c = STEM_2_DIM;
    let x = {
        let _g = name_scope("0");
        let _c = name_scope("conv");
        Conv2d::<D, SymTensor, SymTensor, 4>::new(3, c, (3, 3), (2, 2), (1, 1), false)
            .call(image.clone())
    };
    let x = leaky::<D>(&instance_norm::<D>(&x));
    let x = conv_no_bias::<D>("1", c, c, 3)(&x);
    relu::<D>(instance_norm::<D>(&x))
}
