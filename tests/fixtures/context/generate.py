#!/usr/bin/env python3
"""Fixtures for tests/test_context.rs: pieces of FoundationStereo's context path, with
seeded random weights (BatchNorm running statistics and affines randomised too), in
FP64 on the CPU. Written as little-endian f32. BatchNorm's `num_batches_tracked` and
`ResidualBlock.norm3` (the same module as `downsample.1`) are not written.
  resblock/   ResidualBlock(64, 96, norm_fn='batch', stride=2) in eval mode,
              input [1, 64, 12, 18]
  attention/  inp = cam(x) * x, att = sam(inp) (ChannelAttentionEnhancement(32),
              SpatialAttentionExtractor()), input [1, 32, 9, 13]; outputs inp, att

    python generate.py --fs ~/Documents/osn/FoundationStereo
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))


def write(d, name, t):
    os.makedirs(d, exist_ok=True)
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def randomise(m, g):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if ("norm" in n or "downsample.1" in n) and p.dim() == 1:
                p.copy_((1 + 0.2 * torch.randn(p.shape, generator=g)) if n.endswith("weight")
                        else 0.1 * torch.randn(p.shape, generator=g))
            else:
                fan_in = p[0].numel() if p.dim() > 1 else 1
                p.copy_(torch.randn(p.shape, generator=g) / np.sqrt(fan_in))
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.copy_(0.2 * torch.randn(mod.num_features, generator=g))
                mod.running_var.copy_(0.5 + torch.rand(mod.num_features, generator=g))


def dump_weights(d, prefix, m):
    for k, v in m.state_dict().items():
        if k.endswith("num_batches_tracked") or k.startswith("norm3."):
            continue
        write(d, f"w__{prefix}{k}.bin", v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.fs)
    from core.extractor import ResidualBlock
    from core.submodule import ChannelAttentionEnhancement, SpatialAttentionExtractor

    g = torch.Generator().manual_seed(0)

    d = os.path.join(HERE, "resblock")
    blk = ResidualBlock(64, 96, norm_fn="batch", stride=2)
    randomise(blk, g)
    blk = blk.double().eval()
    x = torch.randn(1, 64, 12, 18, generator=g, dtype=torch.float64)
    write(d, "x.bin", x)
    dump_weights(d, "blk.", blk)
    with torch.no_grad():
        write(d, "expected.bin", blk(x))

    d = os.path.join(HERE, "attention")
    cam, sam = ChannelAttentionEnhancement(32), SpatialAttentionExtractor()
    randomise(cam, g)
    randomise(sam, g)
    cam, sam = cam.double().eval(), sam.double().eval()
    x = torch.relu(torch.randn(1, 32, 9, 13, generator=g, dtype=torch.float64))
    write(d, "x.bin", x)
    dump_weights(d, "cam.", cam)
    dump_weights(d, "sam.", sam)
    with torch.no_grad():
        inp = cam(x) * x
        write(d, "expected_inp.bin", inp)
        write(d, "expected_att.bin", sam(inp))
    print("ok")


if __name__ == "__main__":
    main()
