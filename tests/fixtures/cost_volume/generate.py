#!/usr/bin/env python3
"""Fixtures for tests/test_cost_volume.rs: FoundationStereo's cost-volume construction
(`build_gwc_volume`, `proj_cmb`, `build_concat_volume`, their concatenation) with
seeded random features and `proj_cmb` weights, in FP64 on the CPU (the correlation
itself in FP32, as FoundationStereo computes it). Written as
little-endian f32 under small/:
    fl.bin, fr.bin          features [1, 16, 4, 9]
    w__proj_cmb.*.bin       1x1 conv 16 -> 3
    expected_gwc.bin        [1, 4, 11, 4, 9]   4 groups; 11 disparities, more than
                            the 9 columns (as the real 104 exceed 80)
    expected_comb.bin       [1, 10, 11, 4, 9]

    python generate.py --fs ~/Documents/osn/FoundationStereo
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))


def write(d, name, t):
    os.makedirs(d, exist_ok=True)
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.fs)
    from core.submodule import build_concat_volume, build_gwc_volume

    g = torch.Generator().manual_seed(0)
    fl = torch.randn(1, 16, 4, 9, generator=g, dtype=torch.float64)
    fr = torch.randn(1, 16, 4, 9, generator=g, dtype=torch.float64)
    proj = nn.Conv2d(16, 3, 1).double()
    with torch.no_grad():
        proj.weight.copy_(torch.randn(proj.weight.shape, generator=g) / 4)
        proj.bias.copy_(0.1 * torch.randn(3, generator=g))
        # groupwise_correlation computes in FP32 whatever its input (`.float()`,
        # autocast off), as FoundationStereo runs it; the FP64 volume only stores it.
        gwc = build_gwc_volume(fl, fr, 11, 4)
        cat = build_concat_volume(proj(fl), proj(fr), 11)
        comb = torch.cat([gwc.double(), cat], dim=1)
    d = os.path.join(HERE, "small")
    write(d, "fl.bin", fl)
    write(d, "fr.bin", fr)
    for k, v in proj.state_dict().items():
        write(d, f"w__proj_cmb.{k}.bin", v)
    write(d, "expected_gwc.bin", gwc)
    write(d, "expected_comb.bin", comb)
    print(tuple(gwc.shape), gwc.dtype, tuple(comb.shape))


if __name__ == "__main__":
    main()
