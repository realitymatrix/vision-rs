#!/usr/bin/env python3
"""Real-checkpoint fixtures for `vit_encoder_matches_real_checkpoint`
(tests/test_dinov2.rs): FoundationStereo's frozen DINOv2 ViT-L
(`feature.dino.depth_anything.pretrained`, `dinov2_vitl14`) on the sample left image,
through `get_intermediate_layers(x, [4, 11, 17, 23], return_class_token=True)`, in FP64.

The image is prepared as `Feature.forward` prepares it for the branch: resized to
224x320 and normalised (as gen_real.py of the decoder does), then bicubic-resized to
224x336, the next multiple of lcm(14, 16) = 112. That resize is not yet ported, so its
output is the test input. Writes to <out>/dinov2_real/:
    x.bin                       the resized image [1, 3, 224, 336], f32
    w__pretrained.<key>.bin     the ViT's weights
    expected_l{4,11,17,23}.bin  normed patch grids [1, 1024, 16, 24]

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
    hr, wr = get_resize_keep_aspect_ratio(224, 320, divider=int(np.lcm(14, 16)),
                                          max_H=1344, max_W=1344)
    x_in = F.interpolate(x, size=(hr, wr), mode="bicubic", align_corners=False)

    dino = DepthAnythingFeature(encoder="vitl")
    dino.depth_anything.load_state_dict(sd)
    vit = dino.depth_anything.pretrained.double().eval()

    d = os.path.join(a.out, "dinov2_real")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    write(d, "x.bin", x_in)
    for k, v in vit.state_dict().items():
        if k != "mask_token":
            write(d, f"w__pretrained.{k}.bin", v)
    with torch.no_grad():
        outs = vit.get_intermediate_layers(x_in.double(), LAYERS, return_class_token=True)
    h, w = hr // 14, wr // 14
    for l, (o, _cls) in zip(LAYERS, outs):
        y = o.permute(0, 2, 1).reshape(1, -1, h, w)
        write(d, f"expected_l{l}.bin", y)
        print(l, tuple(y.shape), float(y.abs().max()))


if __name__ == "__main__":
    main()
