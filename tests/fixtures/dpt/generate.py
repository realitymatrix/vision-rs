#!/usr/bin/env python3
"""Fixtures for tests/test_dpt.rs: the Depth Anything V2 DPT head of FoundationStereo's
DINOv2 branch (`depth_anything.dpt.DPTHead`, `use_clstoken=False`) at reduced width,
up to the `out` that `Feature` keeps, and `Feature`'s bilinear `align_corners=True`
resize of it to `vit_feat`.

Weights are seeded random (no download). Two modules that never run are not written:
`scratch.output_conv2` (the depth output, discarded by `Feature`) and
`scratch.refinenet4.resConfUnit1` (refinenet4 gets no skip input). Inputs are four random patch grids
`[1, 64, 5, 7]` (odd, so the stride-2 level rounds up). The reference is computed in
FP64 on the CPU; everything is written as little-endian f32 under dpt_small/:
    l{0..3}.bin                 patch grids
    w__depth_head.<key>.bin     weights
    expected_out.bin            `out`, [1, 16, 70, 98]
    expected_vit_feat.bin       `out` resized to [1, 16, 17, 25]

    python generate.py --fs ~/Documents/osn/FoundationStereo
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
UNUSED = ("scratch.output_conv2", "scratch.refinenet4.resConfUnit1")
VIT, FEATURES, OUT_CH, HW, QUARTER = 64, 32, [16, 32, 64, 64], (5, 7), (17, 25)


def write(d, name, t):
    os.makedirs(d, exist_ok=True)
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.fs)
    from depth_anything.dpt import DPTHead

    head = DPTHead(1, VIT, FEATURES, False, out_channels=OUT_CH, use_clstoken=False)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, q in head.named_parameters():
            fan_in = q[0].numel() if q.dim() > 1 else 1
            if "resize_layers.0" in n or "resize_layers.1" in n:
                fan_in = q.shape[0]   # ConvTranspose2d weight is [in, out, k, k]
            q.copy_(torch.randn(q.shape, generator=g) / np.sqrt(fan_in))
    head = head.double().eval()

    d = os.path.join(HERE, "dpt_small")
    for f in os.listdir(d) if os.path.isdir(d) else []:
        os.remove(os.path.join(d, f))
    for k, v in head.state_dict().items():
        if not k.startswith(UNUSED):
            write(d, f"w__depth_head.{k}.bin", v)
    h, w = HW
    grids = [torch.randn(1, VIT, h, w, generator=g, dtype=torch.float64) for _ in range(4)]
    for i, x in enumerate(grids):
        write(d, f"l{i}.bin", x)
    feats = [(x.flatten(2).transpose(1, 2), None) for x in grids]
    with torch.no_grad():
        out = head(feats, h, w, return_intermediate=True)[0]
    vit = F.interpolate(out, size=QUARTER, mode="bilinear", align_corners=True)
    write(d, "expected_out.bin", out)
    write(d, "expected_vit_feat.bin", vit)
    print(tuple(out.shape), tuple(vit.shape))


if __name__ == "__main__":
    main()
