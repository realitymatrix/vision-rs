#!/usr/bin/env python3
"""Real-checkpoint fixtures for `comb_volume_matches_real_checkpoint`
(tests/test_cost_volume.rs): FoundationStereo's cost-volume construction with a real
checkpoint's `proj_cmb`, on the real 1/4 features of the sample stereo pair.

Both images are prepared as the other real fixtures prepare them (bilinear to 224x320,
normalised) and run through the real `Feature` in FP64; its x4 maps, rounded to f32,
are the test inputs. Writes to <out>/cost_volume_real/:
    fl.bin, fr.bin              [1, 224, 56, 80]
    w__proj_cmb.*.bin
    expected_comb.bin           [1, 32, 104, 56, 80] (max_disp 416 / 4)

    python gen_real.py --fs ~/Documents/osn/FoundationStereo \\
        --run pretrained_models/23-51-11 --out ~/.cache/vision-rs/fs_fixtures
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def write(d, name, t):
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def sub(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load(fs, name):
    img = np.array(Image.open(os.path.join(fs, "assets", name)).convert("RGB"))
    x = torch.from_numpy(img).permute(2, 0, 1)[None].double()
    x = F.interpolate(x, size=(224, 320), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64).view(1, 3, 1, 1)
    return ((x / 255.0 - mean) / std).float().double()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.fs)
    from core.extractor import Feature
    from core.submodule import build_concat_volume, build_gwc_volume

    ckpt = torch.load(os.path.join(a.fs, a.run, "model_best_bp2.pth"),
                      weights_only=False, map_location="cpu")["model"]
    feat = Feature(SimpleNamespace(vit_size="vitl"))
    feat.load_state_dict(sub(ckpt, "feature."))
    feat = feat.double().eval()
    proj = nn.Conv2d(224, 12, 1)
    proj.load_state_dict(sub(ckpt, "proj_cmb."))
    proj = proj.double().eval()

    with torch.no_grad():
        fl = feat(load(a.fs, "left.png"))[0][0].float().double()
        fr = feat(load(a.fs, "right.png"))[0][0].float().double()
        gwc = build_gwc_volume(fl, fr, 104, 8)
        cat = build_concat_volume(proj(fl), proj(fr), 104)
        comb = torch.cat([gwc, cat], dim=1)

    d = os.path.join(a.out, "cost_volume_real")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    write(d, "fl.bin", fl)
    write(d, "fr.bin", fr)
    for k, v in proj.state_dict().items():
        write(d, f"w__proj_cmb.{k}.bin", v)
    write(d, "expected_comb.bin", comb)
    print(tuple(fl.shape), tuple(comb.shape), float(comb.abs().max()))


if __name__ == "__main__":
    main()
