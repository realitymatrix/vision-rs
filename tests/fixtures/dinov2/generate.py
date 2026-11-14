#!/usr/bin/env python3
"""Fixtures for tests/test_dinov2.rs: DINOv2's ViT (`DinoVisionTransformer`, as
FoundationStereo loads `dinov2_vitl14` from torch hub) at reduced width, through
`get_intermediate_layers(x, layers, return_class_token=True)`.

Weights are seeded random (no download), with layer scales and LayerNorm affines
randomised. The reference is computed in FP64 on the CPU (DINOv2 interpolates its
position embedding in FP32 regardless); inputs, weights and outputs are written as
little-endian f32, outputs as patch grids `[B, C, h, w]`. Weight keys carry the
`pretrained.` prefix they have under Depth Anything (a root-scope graph op has no
named slot).

Writes, under this directory:
  vit_down/   64 wide, 3 blocks, 4 heads, 37x37 position grid (as vitl14) shrunk to
              the 4x6 patches of a 56x84 image; outputs of blocks 0 and 2
  vit_up/     48 wide, 1 block, 3 heads, 4x4 position grid enlarged to the 7x10
              patches of a 98x140 image; output of block 0

Run with FoundationStereo's environment (torch hub's dinov2 checkout in
~/.cache/torch/hub):
    python generate.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
sys.path.insert(0, HUB)
from dinov2.models.vision_transformer import DinoVisionTransformer  # noqa: E402


def randomise(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("gamma"):
                p.copy_(torch.rand(p.shape, generator=g) * 0.5 + 0.5)
            elif "norm" in n:
                if n.endswith("weight"):
                    p.copy_(1 + 0.2 * torch.randn(p.shape, generator=g))
                else:
                    p.copy_(0.1 * torch.randn(p.shape, generator=g))
            elif n in ("cls_token", "pos_embed", "mask_token"):
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                fan_in = p[0].numel() if p.dim() > 1 else 1
                p.copy_(torch.randn(p.shape, generator=g) / np.sqrt(fan_in))


def write(d, name, t):
    os.makedirs(d, exist_ok=True)
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def dump(name, img_size, dim, depth, heads, hw, layers, seed):
    d = os.path.join(HERE, name)
    for f in os.listdir(d) if os.path.isdir(d) else []:
        os.remove(os.path.join(d, f))
    m = DinoVisionTransformer(img_size=img_size, patch_size=14, embed_dim=dim,
                              depth=depth, num_heads=heads, mlp_ratio=4,
                              init_values=1.0, block_chunks=0)
    randomise(m, seed)
    m = m.double().eval()
    for k, v in m.state_dict().items():
        if k != "mask_token":
            write(d, f"w__pretrained.{k}.bin", v)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(1, 3, *hw, generator=g, dtype=torch.float64)
    write(d, "x.bin", x)
    h, w = hw[0] // 14, hw[1] // 14
    with torch.no_grad():
        outs = m.get_intermediate_layers(x, layers, return_class_token=True)
    for l, (o, _cls) in zip(layers, outs):
        y = o.permute(0, 2, 1).reshape(1, dim, h, w)
        write(d, f"expected_l{l}.bin", y)
    print(name, [tuple(o.shape) for o, _ in outs])


def main():
    dump("vit_down", 518, 64, 3, 4, (56, 84), [0, 2], 0)
    dump("vit_up", 56, 48, 1, 3, (98, 140), [0], 10)


if __name__ == "__main__":
    main()
