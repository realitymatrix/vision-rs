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

//! The Depth Anything V2 DPT head of FoundationStereo's DINOv2 branch
//! (`Feature.dino.depth_anything.depth_head`, `DPTHead` with `use_clstoken=False`), up
//! to the `out` that `Feature` keeps, and `Feature`'s resize of it to `vit_feat`:
//!
//! ```python
//! x_l = resize_layers[l](projects[l](patches_l))      # 4x, 2x, 1x, 1/2x the patch grid
//! r_l = scratch.layer{l}_rn(x_l)                      # 3x3, no bias, to `features`
//! path_4 = refinenet4(r_4, size=r_3.shape[2:])
//! path_3 = refinenet3(path_4, r_3, size=r_2.shape[2:])
//! path_2 = refinenet2(path_3, r_2, size=r_1.shape[2:])
//! path_1 = refinenet1(path_2, r_1)                    # scale_factor=2
//! out = F.interpolate(output_conv1(path_1), (14 h, 14 w), mode="bilinear", align_corners=True)
//! vit_feat = F.interpolate(out, (H // 4, W // 4), mode="bilinear", align_corners=True)
//! ```
//!
//! Each `refinenet` is a `FeatureFusionBlock`: the skip input through a
//! `ResidualConvUnit` (`x + conv2(relu(conv1(relu(x))))`) added to the path, a second
//! unit, a bilinear `align_corners=True` resize, and a 1x1 `out_conv`. The depth output
//! (`output_conv2`) is not built: `Feature` discards it.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::{CustomData, SymTensor},
    name_scope::name_scope,
    nn::{Layer, conv2d::Conv2d},
};

use crate::models::stereo::kernels::{
    conv_transpose_patch::ConvTransposePatchOp, interpolate_bilinear::InterpolateBilinearOp,
};
use crate::models::stereo::update::ops::{add, conv, relu};

const BLOCK: i32 = 64;

/// DPT head sizes.
#[derive(Clone, Copy, Debug)]
pub struct DptDims {
    /// ViT width (the `projects` input).
    pub vit: usize,
    /// `features`, the fusion width.
    pub features: usize,
    /// `out_channels` per level.
    pub out_channels: [usize; 4],
    /// ViT patch size.
    pub patch: usize,
}

impl DptDims {
    /// FoundationStereo's `vitl` configuration (`DepthAnythingFeature.model_configs`).
    pub fn vitl() -> Self {
        Self {
            vit: 1024,
            features: 256,
            out_channels: [256, 512, 1024, 1024],
            patch: 14,
        }
    }

    /// Channels of `out` and `vit_feat` (`features / 2`).
    pub fn out_dim(&self) -> usize {
        self.features / 2
    }
}

fn resize<D: FloatBytes + Send + Sync + 'static>(x: &SymTensor, size: (usize, usize)) -> SymTensor {
    x.record_custom(
        CustomData::new(InterpolateBilinearOp::<D>::to_size(true, size, 16)),
        &[],
        None,
    )
}

/// `ResidualConvUnit` without batch norm, under `name`.
fn residual_conv_unit<D: Float + FloatBytes + Send + Sync + 'static>(
    name: &str,
    x: &SymTensor,
    c: usize,
) -> SymTensor {
    let _g = name_scope(name);
    let h = conv::<D>("conv1", c, c, 3)(&relu::<D>(x.clone()));
    let h = conv::<D>("conv2", c, c, 3)(&relu::<D>(h));
    add(&h, x)
}

/// `FeatureFusionBlock`: `path` (plus `skip` through `resConfUnit1`, if given), then
/// `resConfUnit2`, a resize to `size` and `out_conv`.
pub fn fusion_block<D: Float + FloatBytes + Send + Sync + 'static>(
    c: usize,
    path: &SymTensor,
    skip: Option<&SymTensor>,
    size: (usize, usize),
) -> SymTensor {
    let x = match skip {
        Some(s) => add(path, &residual_conv_unit::<D>("resConfUnit1", s, c)),
        None => path.clone(),
    };
    let x = residual_conv_unit::<D>("resConfUnit2", &x, c);
    conv::<D>("out_conv", c, c, 1)(&resize::<D>(&x, size))
}

/// `projects[l]` then `resize_layers[l]` for level `l` on a patch grid.
fn reassemble<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: &DptDims,
    l: usize,
    x: &SymTensor,
) -> SymTensor {
    let c = dims.out_channels[l];
    let x = {
        let _g = name_scope(format!("projects.{l}"));
        Conv2d::<D, SymTensor, SymTensor, 4>::new(dims.vit, c, (1, 1), (1, 1), (0, 0), true)
            .call(x.clone())
    };
    let _g = name_scope(format!("resize_layers.{l}"));
    match l {
        0 | 1 => x.record_custom(
            CustomData::new(ConvTransposePatchOp::<D>::new(c, c, 4 >> l, BLOCK)),
            &[],
            None,
        ),
        2 => x,
        _ => Conv2d::<D, SymTensor, SymTensor, 4>::new(c, c, (3, 3), (2, 2), (1, 1), true).call(x),
    }
}

/// The head on the encoder's four patch grids `[B, vit, h, w]`, returning `out`
/// `[B, features / 2, patch * h, patch * w]`.
pub fn dpt_head<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: &DptDims,
    patches: &[SymTensor; 4],
    (h, w): (usize, usize),
) -> SymTensor {
    let f = dims.features;
    // Spatial sizes of the reassembled levels: 4x, 2x, 1x and (3x3, stride 2, pad 1) 1/2x.
    let sizes = [
        (4 * h, 4 * w),
        (2 * h, 2 * w),
        (h, w),
        (h.div_ceil(2), w.div_ceil(2)),
    ];
    let r: Vec<SymTensor> = (0..4)
        .map(|l| {
            let x = reassemble::<D>(dims, l, &patches[l]);
            let _s = name_scope("scratch");
            let _g = name_scope(format!("layer{}_rn", l + 1));
            Conv2d::<D, SymTensor, SymTensor, 4>::new(
                dims.out_channels[l],
                f,
                (3, 3),
                (1, 1),
                (1, 1),
                false,
            )
            .call(x)
        })
        .collect();
    let _s = name_scope("scratch");
    let mut path = {
        let _g = name_scope("refinenet4");
        fusion_block::<D>(f, &r[3], None, sizes[2])
    };
    for (l, size) in [(2, sizes[1]), (1, sizes[0]), (0, (8 * h, 8 * w))] {
        let _g = name_scope(format!("refinenet{}", l + 1));
        path = fusion_block::<D>(f, &path, Some(&r[l]), size);
    }
    let out = conv::<D>("output_conv1", f, f / 2, 3)(&path);
    resize::<D>(&out, (dims.patch * h, dims.patch * w))
}

/// `Feature`'s `vit_feat`: `out` resized to the 1/4-resolution map `(H / 4, W / 4)`.
pub fn vit_feat<D: FloatBytes + Send + Sync + 'static>(
    out: &SymTensor,
    quarter: (usize, usize),
) -> SymTensor {
    resize::<D>(out, quarter)
}
