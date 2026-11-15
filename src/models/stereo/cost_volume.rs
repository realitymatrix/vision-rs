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

//! FoundationStereo's cost-volume construction (`core/foundation_stereo.py`), from the
//! 1/4-resolution left and right features:
//!
//! ```python
//! gwc_volume = build_gwc_volume(fl, fr, max_disp // 4, cv_group)       # [B, 8, D, H, W]
//! concat_volume = build_concat_volume(proj_cmb(fl), proj_cmb(fr), max_disp // 4)   # [B, 24, D, H, W]
//! comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)          # [B, 32, D, H, W]
//! ```
//!
//! `proj_cmb` is a 1x1 conv with bias to 12 channels, shared by both images. The
//! volume's filtering (`corr_stem`, `corr_feature_att`, `cost_agg`) is not here.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::{CustomData, SymTensor},
};

use crate::models::stereo::kernels::{
    cat2::Cat2Op, concat_volume::ConcatVolumeOp, gwc_volume::GwcVolumeOp,
};
use crate::models::stereo::update::ops::conv;

const BLOCK: i32 = 64;

/// Cost-volume sizes.
#[derive(Clone, Copy, Debug)]
pub struct CostVolumeDims {
    /// Feature channels at 1/4 (`Feature.d_out[0]`).
    pub feat: usize,
    /// Correlation groups (`cv_group`).
    pub groups: usize,
    /// `proj_cmb` output channels.
    pub proj: usize,
    /// Disparities at 1/4 (`max_disp // 4`).
    pub disps: usize,
}

impl CostVolumeDims {
    /// FoundationStereo's released configuration: 224 feature channels, 8 groups,
    /// `proj_cmb` to 12, `max_disp: 416` (104 disparities at 1/4).
    pub fn foundation_stereo() -> Self {
        Self {
            feat: 224,
            groups: 8,
            proj: 12,
            disps: 104,
        }
    }

    /// Channels of the combined volume, `groups + 2 * proj`.
    pub fn channels(&self) -> usize {
        self.groups + 2 * self.proj
    }
}

/// `build_gwc_volume(fl, fr, disps, groups)`.
pub fn gwc_volume<D: FloatBytes + Send + Sync + 'static>(
    fl: &SymTensor,
    fr: &SymTensor,
    groups: usize,
    disps: usize,
) -> SymTensor {
    fl.record_custom(
        CustomData::new(GwcVolumeOp::<D>::new(groups, disps, BLOCK)),
        &[fr],
        None,
    )
}

/// `build_concat_volume(l, r, disps)`.
pub fn concat_volume<D: FloatBytes + Send + Sync + 'static>(
    l: &SymTensor,
    r: &SymTensor,
    disps: usize,
) -> SymTensor {
    l.record_custom(
        CustomData::new(ConcatVolumeOp::<D>::new(disps, BLOCK)),
        &[r],
        None,
    )
}

/// `comb_volume` from the 1/4 left and right features (the `proj_cmb` slots).
pub fn comb_volume<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: &CostVolumeDims,
    fl: &SymTensor,
    fr: &SymTensor,
) -> SymTensor {
    let gwc = gwc_volume::<D>(fl, fr, dims.groups, dims.disps);
    let proj = conv::<D>("proj_cmb", dims.feat, dims.proj, 1);
    let cat = concat_volume::<D>(&proj(fl), &proj(fr), dims.disps);
    gwc.record_custom(CustomData::new(Cat2Op::<D>::new(256)), &[&cat], None)
}
