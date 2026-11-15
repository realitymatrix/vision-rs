#!/usr/bin/env python3
"""Fixtures for tests/test_stem.rs: FoundationStereo's `stem_2`, built from its own
modules as `FoundationStereo.__init__` builds it, in FP64 on the CPU. Written as
little-endian f32.
  small/   seeded random weights, random image [1, 3, 30, 46] -> [1, 32, 15, 23]
  with --run and --out, also <out>/stem_real/: the real checkpoint's `stem_2` on the
           sample left image as the other real fixtures prepare it (bilinear to
           224x320, normalised) -> [1, 32, 112, 160]

    python generate.py --fs ~/Documents/osn/FoundationStereo \\
        [--run pretrained_models/23-51-11 --out ~/.cache/vision-rs/fs_fixtures]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def write(d, name, t):
    os.makedirs(d, exist_ok=True)
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def dump(d, stem, x):
    for f in os.listdir(d) if os.path.isdir(d) else []:
        os.remove(os.path.join(d, f))
    write(d, "image.bin", x)
    for k, v in stem.state_dict().items():
        write(d, f"w__stem_2.{k}.bin", v)
    with torch.no_grad():
        y = stem(x)
    write(d, "expected.bin", y)
    print(d, tuple(y.shape))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    p.add_argument("--run")
    p.add_argument("--out")
    a = p.parse_args()
    sys.path.insert(0, a.fs)
    from core.submodule import BasicConv_IN

    def build():
        return nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU())

    g = torch.Generator().manual_seed(0)
    stem = build()
    with torch.no_grad():
        for q in stem.parameters():
            q.copy_(torch.randn(q.shape, generator=g) / np.sqrt(q[0].numel()))
    x = torch.randn(1, 3, 30, 46, generator=g, dtype=torch.float64)
    dump(os.path.join(HERE, "small"), stem.double().eval(), x)

    if a.run and a.out:
        ckpt = torch.load(os.path.join(a.fs, a.run, "model_best_bp2.pth"),
                          weights_only=False, map_location="cpu")["model"]
        stem = build()
        stem.load_state_dict({k[len("stem_2."):]: v for k, v in ckpt.items()
                              if k.startswith("stem_2.")})
        img = np.array(Image.open(os.path.join(a.fs, "assets/left.png")).convert("RGB"))
        x = torch.from_numpy(img).permute(2, 0, 1)[None].double()
        x = F.interpolate(x, size=(224, 320), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64).view(1, 3, 1, 1)
        x = ((x / 255.0 - mean) / std).float().double()
        dump(os.path.join(a.out, "stem_real"), stem.double().eval(), x)


if __name__ == "__main__":
    main()
