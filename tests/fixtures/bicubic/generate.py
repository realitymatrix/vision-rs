#!/usr/bin/env python3
"""Fixtures for tests/test_interpolate_bicubic.rs: `F.interpolate(x, size,
mode="bicubic", align_corners=False)`, as FoundationStereo's `Feature.forward` resizes
the image for the DINOv2 branch. The reference runs in FP32 on the CPU, as `Feature`
runs it: PyTorch computes the source scale `in / out` in the input's type, and an FP64
reference's exact scale moves sample positions by ~1e-6 px, more than the f32 rounding
of the 16-tap sum. Written as little-endian f32:
  up/      [1, 3, 30, 44] -> [1, 3, 37, 61]   enlarging both sides, uneven ratios
  down/    [1, 3, 30, 44] -> [1, 3, 19, 27]   shrinking (no antialiasing, as PyTorch's default)
  branch/  [1, 3, 56, 80] -> [1, 3, 56, 84]   the branch's pattern at 224x320 -> 224x336
                                              scaled by 1/4: height kept, width widened

    python generate.py
"""
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def write(d, name, t):
    os.makedirs(d, exist_ok=True)
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def main():
    g = torch.Generator().manual_seed(0)
    for name, shape, size in [("up", (30, 44), (37, 61)), ("down", (30, 44), (19, 27)),
                              ("branch", (56, 80), (56, 84))]:
        x = torch.randn(1, 3, *shape, generator=g, dtype=torch.float64).float()
        y = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
        d = os.path.join(HERE, name)
        write(d, "x.bin", x)
        write(d, "expected.bin", y)
        print(name, tuple(y.shape))


if __name__ == "__main__":
    main()
