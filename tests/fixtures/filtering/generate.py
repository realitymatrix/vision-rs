#!/usr/bin/env python3
"""Fixtures for tests/test_filtering.rs: FoundationStereo's `corr_stem` and
`corr_feature_att` (built from its own modules as `FoundationStereo.__init__` builds
them), with seeded random weights and BatchNorm running statistics, in FP64 on the
CPU, eval mode. Written as little-endian f32 under stem_att/:
    cv.bin                  combined volume [1, 32, 6, 5, 7]
    feat.bin                image features [1, 24, 5, 7]
    w__corr_stem.*, w__corr_feature_att.*
    expected_stem.bin       corr_stem(cv), [1, 28, 6, 5, 7]
    expected.bin            corr_feature_att(corr_stem(cv), feat)
and under hourglass/:
    cv.bin                  volume [1, 8, 16, 8, 16]
    f{1,2,3}.bin            left features at 1/2, 1/4, 1/8 of the volume's H, W:
                            [1, 12, 4, 8], [1, 16, 2, 4], [1, 20, 1, 2]
    w__cost_agg.*           `hourglass(cfg={'max_disp': 64}, in_channels=8,
                            feat_dims=[8, 12, 16, 20])`, LayerNorm affines randomised
    expected.bin            [1, 8, 16, 8, 16]

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
        for mod in m.modules():
            if isinstance(mod, nn.LayerNorm):
                mod.weight.copy_(1 + 0.2 * torch.randn(mod.weight.shape, generator=g))
                mod.bias.copy_(0.1 * torch.randn(mod.bias.shape, generator=g))
            elif isinstance(mod, nn.Linear):
                mod.weight.copy_(torch.randn(mod.weight.shape, generator=g)
                                 / np.sqrt(mod.weight.shape[1]))
                mod.bias.copy_(0.1 * torch.randn(mod.bias.shape, generator=g))
            elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm3d)):
                mod.weight.copy_(1 + 0.2 * torch.randn(mod.num_features, generator=g))
                mod.bias.copy_(0.1 * torch.randn(mod.num_features, generator=g))
                mod.running_mean.copy_(0.2 * torch.randn(mod.num_features, generator=g))
                mod.running_var.copy_(0.5 + torch.rand(mod.num_features, generator=g))
            elif isinstance(mod, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
                mod.weight.copy_(torch.randn(mod.weight.shape, generator=g)
                                 / np.sqrt(mod.weight[0].numel()))
                if mod.bias is not None:
                    mod.bias.copy_(0.1 * torch.randn(mod.bias.shape, generator=g))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.fs)
    from core.submodule import BasicConv, FeatureAtt, ResnetBasicBlock3D

    vd = 28
    stem = nn.Sequential(
        nn.Conv3d(32, vd, kernel_size=1),
        BasicConv(vd, vd, kernel_size=3, padding=1, is_3d=True),
        ResnetBasicBlock3D(vd, vd, kernel_size=3, stride=1, padding=1),
        ResnetBasicBlock3D(vd, vd, kernel_size=3, stride=1, padding=1),
    )
    att = FeatureAtt(vd, 24)
    g = torch.Generator().manual_seed(0)
    randomise(stem, g)
    randomise(att, g)
    stem, att = stem.double().eval(), att.double().eval()
    cv = torch.randn(1, 32, 6, 5, 7, generator=g, dtype=torch.float64)
    feat = torch.randn(1, 24, 5, 7, generator=g, dtype=torch.float64)
    d = os.path.join(HERE, "stem_att")
    for f in os.listdir(d) if os.path.isdir(d) else []:
        os.remove(os.path.join(d, f))
    write(d, "cv.bin", cv)
    write(d, "feat.bin", feat)
    for prefix, m in [("corr_stem", stem), ("corr_feature_att", att)]:
        for k, v in m.state_dict().items():
            if not k.endswith("num_batches_tracked"):
                write(d, f"w__{prefix}.{k}.bin", v)
    with torch.no_grad():
        s = stem(cv)
        write(d, "expected_stem.bin", s)
        write(d, "expected.bin", att(s, feat))
    from core.foundation_stereo import hourglass
    hg = hourglass(cfg={"max_disp": 64}, in_channels=8, feat_dims=[8, 12, 16, 20])
    randomise(hg, g)
    hg = hg.double().eval()
    d = os.path.join(HERE, "hourglass")
    for f in os.listdir(d) if os.path.isdir(d) else []:
        os.remove(os.path.join(d, f))
    cv = torch.randn(1, 8, 16, 8, 16, generator=g, dtype=torch.float64)
    feats = [None] + [torch.randn(1, c, 8 >> l, 16 >> l, generator=g, dtype=torch.float64)
                      for l, c in [(1, 12), (2, 16), (3, 20)]]
    write(d, "cv.bin", cv)
    for l in (1, 2, 3):
        write(d, f"f{l}.bin", feats[l])
    for k, v in hg.state_dict().items():
        if not k.endswith("num_batches_tracked"):
            write(d, f"w__cost_agg.{k}.bin", v)
    with torch.no_grad():
        write(d, "expected.bin", hg(cv, feats))
    print("ok")


if __name__ == "__main__":
    main()
