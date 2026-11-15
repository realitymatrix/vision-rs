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

//! The DINOv2 ViT encoder of FoundationStereo's Depth Anything branch
//! (`Feature.dino.depth_anything.pretrained`, `dinov2_vitl14`), up to
//! `get_intermediate_layers(x, layers, return_class_token=True)`.
//!
//! Tokens are channel-first, `[B, C, 1, 1 + h*w]` with the class token first, so
//! every `nn.Linear` is a 1x1 conv (the `[out, in]` weight loads unchanged) and every
//! LayerNorm is [`ChannelLayerNormOp`]. Each block is
//!
//! ```python
//! x = x + ls1(attn(norm1(x)))
//! x = x + ls2(mlp(norm2(x)))         # fc1, exact GELU, fc2
//! ```
//!
//! and the outputs are the final `norm` applied to the chosen blocks' tokens, as patch
//! grids `[B, C, h, w]`. The class tokens are dropped: FoundationStereo's DPT head is
//! built with `use_clstoken=False` and never reads them.

use teeny_core::{
    dtype::{Float, FloatBytes},
    graph::{CustomData, SymTensor},
    name_scope::name_scope,
    nn::{Layer, conv2d::Conv2d},
};

use crate::models::stereo::kernels::{
    channel_layer_norm::ChannelLayerNormOp, interpolate_bicubic::InterpolateBicubicOp,
    token_attn_apply::TokenAttnApplyOp, token_attn_logits::TokenAttnLogitsOp,
    vit_patches::VitPatchesOp, vit_tokens::VitTokensOp,
};
use crate::models::stereo::update::ops::{add, channel_scale, conv, gelu};

/// DINOv2's LayerNorm epsilon.
pub const EPS: f32 = 1e-6;
/// DINOv2's `interpolate_offset`.
pub const INTERPOLATE_OFFSET: f64 = 0.1;

const BLOCK: i32 = 256;

/// ViT sizes.
#[derive(Clone, Copy, Debug)]
pub struct VitDims {
    /// Embedding width.
    pub dim: usize,
    /// Transformer blocks.
    pub depth: usize,
    /// Attention heads.
    pub heads: usize,
    /// MLP hidden width.
    pub mlp: usize,
    /// Patch size.
    pub patch: usize,
    /// Side of the pretrained position-embedding grid.
    pub pos_grid: usize,
}

impl VitDims {
    /// `dinov2_vitl14`, as FoundationStereo's `vit_size: vitl` loads it.
    pub fn vitl14() -> Self {
        Self {
            dim: 1024,
            depth: 24,
            heads: 16,
            mlp: 4096,
            patch: 14,
            pos_grid: 37,
        }
    }

    /// The blocks Depth Anything V2 reads for `vitl` (`intermediate_layer_idx`).
    pub const VITL_LAYERS: [usize; 4] = [4, 11, 17, 23];
}

fn layer_norm<D: FloatBytes + Send + Sync + 'static>(
    name: &str,
    x: &SymTensor,
    c: usize,
) -> SymTensor {
    let _g = name_scope(name);
    x.record_custom(
        CustomData::new(ChannelLayerNormOp::<D>::new(c, EPS, BLOCK)),
        &[],
        None,
    )
}

/// `patch_embed` plus the class token and interpolated position embedding:
/// image `[B, 3, H, W]` (multiples of the patch size) to tokens.
pub fn embed<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: &VitDims,
    x: &SymTensor,
) -> SymTensor {
    let p = dims.patch;
    let proj = Conv2d::<D, SymTensor, SymTensor, 4>::new(3, dims.dim, (p, p), (p, p), (0, 0), true);
    let patches = {
        let _g = name_scope("patch_embed.proj");
        proj.call(x.clone())
    };
    patches.record_custom(
        CustomData::new(VitTokensOp::<D>::new(
            dims.dim,
            dims.pos_grid,
            INTERPOLATE_OFFSET,
            BLOCK,
        )),
        &[],
        None,
    )
}

/// DINOv2's `Attention` on tokens (`qkv` and `proj` slots).
pub fn attention<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: &VitDims,
    x: &SymTensor,
) -> SymTensor {
    let c = dims.dim;
    let scale = ((c / dims.heads) as f64).powf(-0.5) as f32;
    let _g = name_scope("attn");
    let qkv = conv::<D>("qkv", c, 3 * c, 1)(x);
    let logits = qkv.record_custom(
        CustomData::new(TokenAttnLogitsOp::<D>::new(c, dims.heads, scale, BLOCK)),
        &[],
        None,
    );
    let out = logits.record_custom(
        CustomData::new(TokenAttnApplyOp::<D>::new(c, dims.heads, BLOCK)),
        &[&qkv],
        None,
    );
    conv::<D>("proj", c, c, 1)(&out)
}

/// One transformer block with layer scale (`norm1`, `attn`, `ls1`, `norm2`, `mlp`,
/// `ls2`).
pub fn block<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: &VitDims,
    x: &SymTensor,
) -> SymTensor {
    let c = dims.dim;
    let a = attention::<D>(dims, &layer_norm::<D>("norm1", x, c));
    let a = {
        let _g = name_scope("ls1");
        channel_scale::<D>("gamma", c)(&a)
    };
    let x = add(x, &a);
    let n = layer_norm::<D>("norm2", &x, c);
    let h = {
        let _g = name_scope("mlp");
        let h = conv::<D>("fc1", c, dims.mlp, 1)(&n);
        conv::<D>("fc2", dims.mlp, c, 1)(&gelu::<D>(&h))
    };
    let h = {
        let _g = name_scope("ls2");
        channel_scale::<D>("gamma", c)(&h)
    };
    add(&x, &h)
}

/// `get_intermediate_layers(x, layers)` for an image `[B, 3, H, W]` whose sides are
/// multiples of the patch size: the final-normed patch grids `[B, dim, H/p, W/p]` of
/// the blocks in `layers` (ascending), stopping after the last of them.
pub fn encoder<D: Float + FloatBytes + Send + Sync + 'static>(
    dims: &VitDims,
    x: &SymTensor,
    (h, w): (usize, usize),
    layers: &[usize],
) -> Vec<SymTensor> {
    assert!(
        h % dims.patch == 0 && w % dims.patch == 0,
        "image not a multiple of the patch size"
    );
    let last = *layers.last().expect("no layers");
    assert!(last < dims.depth, "layer {last} out of range");
    let (ph, pw) = (h / dims.patch, w / dims.patch);
    let mut t = embed::<D>(dims, x);
    let mut out = Vec::new();
    for i in 0..=last {
        t = {
            let _g = name_scope(format!("blocks.{i}"));
            block::<D>(dims, &t)
        };
        if layers.contains(&i) {
            let n = layer_norm::<D>("norm", &t, dims.dim);
            out.push(n.record_custom(
                CustomData::new(VitPatchesOp::<D>::new(ph, pw, BLOCK)),
                &[],
                None,
            ));
        }
    }
    out
}

/// `Feature.forward`'s resize divider, `lcm(patch 14, 16)`.
pub const RESIZE_DIVIDER: usize = 112;
/// `Feature.forward`'s `max_H` and `max_W`.
pub const RESIZE_MAX: usize = 1344;

/// The size `Feature.forward` resizes an `h x w` image to for the ViT:
/// FoundationStereo's `get_resize_keep_aspect_ratio(h, w, divider=112, max_H=1344,
/// max_W=1344)`. Each side rounds up to a multiple of 112; if either then exceeds 1344,
/// the longer becomes 1344 and the other scales with it (rounded up again).
pub fn branch_input_size(h: usize, w: usize) -> (usize, usize) {
    let (d, max) = (RESIZE_DIVIDER, RESIZE_MAX);
    let round = |x: f64| ((x / d as f64).ceil() as usize) * d;
    let (hr, wr) = (round(h as f64), round(w as f64));
    if hr <= max && wr <= max {
        (hr, wr)
    } else if hr > wr {
        (max, round(wr as f64 * max as f64 / hr as f64))
    } else {
        (round(hr as f64 * max as f64 / wr as f64), max)
    }
}

/// `F.interpolate(x, size, mode="bicubic", align_corners=False)`: the image resize in
/// front of the ViT, to [`branch_input_size`].
pub fn resize_input<D: FloatBytes + Send + Sync + 'static>(
    x: &SymTensor,
    size: (usize, usize),
) -> SymTensor {
    x.record_custom(
        CustomData::new(InterpolateBicubicOp::<D>::new(size, 64)),
        &[],
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::branch_input_size;

    #[test]
    fn branch_input_size_matches_foundation_stereo() {
        // Values from FoundationStereo's `get_resize_keep_aspect_ratio`.
        assert_eq!(branch_input_size(224, 320), (224, 336));
        assert_eq!(branch_input_size(480, 640), (560, 672));
        assert_eq!(branch_input_size(1080, 1920), (784, 1344));
        assert_eq!(branch_input_size(2000, 1500), (1344, 1120));
        assert_eq!(branch_input_size(1344, 1344), (1344, 1344));
    }
}
