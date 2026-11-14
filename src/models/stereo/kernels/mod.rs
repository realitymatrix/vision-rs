//! Kernels specific to stereo matching.

pub mod avg_pool2d_padded;
pub mod bilinear_sampler_1d;
pub mod channel_layer_norm;
pub mod channel_slice;
pub mod channel_softmax;
pub mod context_upsample;
pub mod conv_transpose2d_k4s2p1;
pub mod conv_transpose_patch;
pub mod correlation;
pub mod gated_blend;
pub mod gelu_erf;
pub mod geo_lookup;
pub mod instance_norm3d;
pub mod interpolate_bilinear;
pub mod l2_normalize;
pub mod leaky_relu;
pub mod pool_pairs;
pub mod pos_fourier;
pub mod scale;
pub mod token_attn_apply;
pub mod token_attn_logits;
pub mod unfold3x3;
pub mod vit_patches;
pub mod vit_tokens;
pub mod xca_apply;
pub mod xca_gram;
