#!/usr/bin/env python3
"""Real-checkpoint fixtures for `dpt_head_matches_real_checkpoint` (tests/test_dpt.rs):
FoundationStereo's Depth Anything DPT head (`feature.dino.depth_anything.depth_head`)
on the real ViT-L features of the sample left image, and `Feature`'s resize to
`vit_feat`, in FP64.

The image is prepared as for tests/fixtures/dinov2/gen_real.py (224x320, normalised,
bicubic to 224x336); the ViT runs in FP64 and its four patch grids, rounded to f32,
are the test inputs. Writes to <out>/dpt_real/:
    l{0..3}.bin                 patch grids [1, 1024, 16, 24]
    w__depth_head.<key>.bin     the head's weights (without the unused
                                `scratch.output_conv2`, `scratch.refinenet4.resConfUnit1`)
    expected_out.bin            `out` [1, 128, 224, 336]
    expected_vit_feat.bin       `vit_feat` [1, 128, 56, 80]

    python gen_real.py --fs ~/Documents/osn/FoundationStereo \\
        --run pretrained_models/23-51-11 --out ~/.cache/vision-rs/fs_fixtures
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

LAYERS = [4, 11, 17, 23]
UNUSED = ("scratch.output_conv2", "scratch.refinenet4.resConfUnit1")


def write(d, name, t):
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.fs)
    from core.extractor import DepthAnythingFeature
    from Utils import get_resize_keep_aspect_ratio

    ckpt = torch.load(os.path.join(a.fs, a.run, "model_best_bp2.pth"),
                      weights_only=False, map_location="cpu")["model"]
    prefix = "feature.dino.depth_anything."
    sd = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}

    img = np.array(Image.open(os.path.join(a.fs, "assets/left.png")).convert("RGB"))
    x = torch.from_numpy(img).permute(2, 0, 1)[None].double()
    x = F.interpolate(x, size=(224, 320), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64).view(1, 3, 1, 1)
    x = ((x / 255.0 - mean) / std).float()
    H, W = x.shape[-2:]
    hr, wr = get_resize_keep_aspect_ratio(H, W, divider=int(np.lcm(14, 16)),
                                          max_H=1344, max_W=1344)
    x_in = F.interpolate(x, size=(hr, wr), mode="bicubic", align_corners=False)

    dino = DepthAnythingFeature(encoder="vitl")
    dino.depth_anything.load_state_dict(sd)
    da = dino.depth_anything.double().eval()
    h, w = hr // 14, wr // 14

    d = os.path.join(a.out, "dpt_real")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    with torch.no_grad():
        outs = da.pretrained.get_intermediate_layers(x_in.double(), LAYERS,
                                                     return_class_token=True)
        grids = [o.permute(0, 2, 1).reshape(1, -1, h, w).float().double()
                 for o, _ in outs]
        feats = [(g.flatten(2).transpose(1, 2), None) for g in grids]
        out = da.depth_head(feats, h, w, return_intermediate=True)[0]
        vit = F.interpolate(out, size=(H // 4, W // 4), mode="bilinear",
                            align_corners=True)
    for i, g in enumerate(grids):
        write(d, f"l{i}.bin", g)
    for k, v in da.depth_head.state_dict().items():
        if not k.startswith(UNUSED):
            write(d, f"w__depth_head.{k}.bin", v)
    write(d, "expected_out.bin", out)
    write(d, "expected_vit_feat.bin", vit)
    print(tuple(out.shape), float(out.abs().max()), tuple(vit.shape))


if __name__ == "__main__":
    main()
