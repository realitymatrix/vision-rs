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

//! FoundationStereo end to end (`FoundationStereo.forward(test_mode=True)`,
//! `core/foundation_stereo.py`), as three graphs and a host loop:
//!
//! 1. **prelude** ([`prelude`]), once per image pair: `Feature` on both images, the
//!    combined cost volume and its filtering, the classifier's initial disparity,
//!    the context network with `cam` / `sam`, the geometry pyramid and `stem_2`;
//! 2. **step** ([`super::update::refinement_loop::build_step`]), once per iteration,
//!    driven by [`super::update::refinement_loop::run_refinement`];
//! 3. **upsample** ([`super::update::upsample::upsample_disp`]), once, on the final
//!    disparity and mask.
//!
//! The prelude's outputs are named as the step graph's inputs ([`PRELUDE_OUTPUTS`]).
//! Images are taken normalised (`normalize_image`: `(x / 255 - mean) / std` per RGB
//! channel, see [`normalize_image`]), with sides a multiple of 32, as FoundationStereo
//! pads them. `Feature` runs once per image rather than on the stacked pair: with
//! BatchNorm in eval mode and no cross-batch op the two are identical (FoundationStereo's
//! own `OSN_FS_SEQ_ENCODER` path does the same).

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::SymTensor,
    name_scope::name_scope,
};

use super::cost_volume::{CostVolumeDims, comb_volume};
use super::feature::{
    context::{ContextDims, context},
    decoder::{DecoderDims, decoder},
    dinov2::{VitDims, branch_input_size, encoder, resize_input},
    dpt::{DptDims, dpt_head, vit_feat},
    edgenext::backbone,
};
use super::filtering::{VOLUME_DIM, classifier, corr_stem, feature_att, hourglass, init_disp};
use super::geometry::{correlation_volume, geo_pyramid};
use super::stem::stem_2;

/// Pyramid levels (`corr_levels`).
pub const CORR_LEVELS: usize = 2;
/// Lookup radius (`corr_radius`).
pub const CORR_RADIUS: usize = 4;

/// ImageNet mean and standard deviation, as `normalize_image` applies them.
pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// See [`MEAN`].
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// `normalize_image` on a planar `[3, H, W]` RGB image in 0..=255, in place.
pub fn normalize_image(rgb: &mut [f32]) {
    let n = rgb.len() / 3;
    for (c, plane) in rgb.chunks_mut(n).enumerate() {
        for v in plane {
            *v = (*v / 255.0 - MEAN[c]) / STD[c];
        }
    }
}

/// Names of [`prelude`]'s outputs, in order: the step graph's inputs (see
/// [`super::update::refinement_loop`]) for two pyramid levels, then `stem_2x`.
pub const PRELUDE_OUTPUTS: [&str; 14] = [
    "net0", "net1", "net2", "inp0", "inp1", "inp2", "att0", "att1", "att2", "geo0", "geo1",
    "corr0", "corr1", "disp",
];

/// `Feature.forward(image)` for an `h x w` image: `[x4, x8, x16, x32]` (224, 192, 320
/// and 304 channels) and `vit_feat` (128 at 1/4), weights under the caller's scope.
pub fn feature<D: Float + FloatBytes + Send + Sync + 'static>(
    image: &SymTensor,
    (h, w): (usize, usize),
) -> ([SymTensor; 4], SymTensor) {
    let vit = {
        let _g = name_scope("dino");
        let _d = name_scope("depth_anything");
        let size = branch_input_size(h, w);
        let x = resize_input::<D>(image, size);
        let vd = VitDims::vitl14();
        let patches = {
            let _p = name_scope("pretrained");
            encoder::<D>(&vd, &x, size, &VitDims::VITL_LAYERS)
        };
        let patches: [SymTensor; 4] = match patches.try_into() {
            Ok(p) => p,
            Err(_) => unreachable!(),
        };
        let out = {
            let _h = name_scope("depth_head");
            dpt_head::<D>(
                &DptDims::vitl(),
                &patches,
                (size.0 / vd.patch, size.1 / vd.patch),
            )
        };
        vit_feat::<D>(&out, (h / 4, w / 4))
    };
    let maps = backbone::<D>(image);
    let feats = decoder::<D>(
        DecoderDims::foundation_stereo(),
        [&maps[0], &maps[1], &maps[2], &maps[3]],
        &vit,
    );
    (feats, vit)
}

/// The prelude graph on normalised `left` and `right` images `[1, 3, h, w]`: the step
/// graph's fixed inputs, its initial state and `stem_2x`, as `(name, tensor)` in
/// [`PRELUDE_OUTPUTS`] order followed by `("stem_2x", …)`. Weights under the model's
/// `state_dict` keys (`feature.*`, `cnet.*`, `cost_agg.*`, …).
pub fn prelude<D: Float + FloatBytes + Send + Sync + 'static>(
    left: &SymTensor,
    right: &SymTensor,
    (h, w): (usize, usize),
) -> Vec<(&'static str, SymTensor)> {
    assert!(
        h % 32 == 0 && w % 32 == 0,
        "pad the images to a multiple of 32"
    );
    let (fl, vit_l) = {
        let _g = name_scope("feature");
        feature::<D>(left, (h, w))
    };
    let (fr, _) = {
        let _g = name_scope("feature");
        feature::<D>(right, (h, w))
    };
    let feat_dims = DecoderDims::foundation_stereo().d_out();

    let cv = comb_volume::<D>(&CostVolumeDims::foundation_stereo(), &fl[0], &fr[0]);
    let cv = {
        let _g = name_scope("corr_stem");
        corr_stem::<D>(
            &cv,
            CostVolumeDims::foundation_stereo().channels(),
            VOLUME_DIM,
        )
    };
    let cv = {
        let _g = name_scope("corr_feature_att");
        feature_att::<D>(&cv, &fl[0], VOLUME_DIM, feat_dims[0])
    };
    let geo = {
        let _g = name_scope("cost_agg");
        hourglass::<D>(&cv, [&fl[0], &fl[1], &fl[2], &fl[3]], VOLUME_DIM, feat_dims)
    };
    let disp = {
        let _g = name_scope("classifier");
        init_disp::<D>(&classifier::<D>(&geo, VOLUME_DIM))
    };

    let ctx = context::<D>(&ContextDims::foundation_stereo(), left, &vit_l);
    let corr = correlation_volume::<D>(&fl[0], &fr[0], feat_dims[0].next_power_of_two() as i32);
    let pyr = geo_pyramid::<D>(&geo, &corr, CORR_LEVELS);
    let stem = {
        let _g = name_scope("stem_2");
        stem_2::<D>(left)
    };

    let [n0, n1, n2] = ctx.net;
    let [i0, i1, i2] = ctx.inp;
    let [a0, a1, a2] = ctx.att;
    let mut t = vec![n0, n1, n2, i0, i1, i2, a0, a1, a2];
    t.extend(pyr.geo);
    t.extend(pyr.corr);
    t.push(disp);
    t.push(stem);
    PRELUDE_OUTPUTS
        .into_iter()
        .chain(["stem_2x"])
        .zip(t)
        .collect()
}
