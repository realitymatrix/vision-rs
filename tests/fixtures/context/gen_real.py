#!/usr/bin/env python3
"""Real-checkpoint fixtures for `context_matches_real_checkpoint`
(tests/test_context.rs): FoundationStereo's context path with a real checkpoint
(`cnet`, `cam`, `sam`) on the sample left image at 224x320, in FP64.

The image is prepared as the other real fixtures prepare it (bilinear to 224x320,
normalised). `vit_feat` comes from the real DINOv2 branch (`feature.dino`) run in
FP64 as `Feature.forward` runs it, rounded to f32 as the test feeds it. Writes to
<out>/context_real/:
    image.bin, vit.bin              inputs [1, 3, 224, 320], [1, 128, 56, 80]
    w__<key>.bin                    `cnet.*` (without the unused `down`, `norm`, the
                                    duplicate `norm3` and `num_batches_tracked`),
                                    `cam.*`, `sam.*`
    expected_{net,inp,att}{0,1,2}.bin

    python gen_real.py --fs ~/Documents/osn/FoundationStereo \\
        --run pretrained_models/23-51-11 --out ~/.cache/vision-rs/fs_fixtures
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def write(d, name, t):
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def sub(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.fs)
    from core.extractor import ContextNetDino, DepthAnythingFeature
    from core.submodule import ChannelAttentionEnhancement, SpatialAttentionExtractor
    from Utils import get_resize_keep_aspect_ratio

    ckpt = torch.load(os.path.join(a.fs, a.run, "model_best_bp2.pth"),
                      weights_only=False, map_location="cpu")["model"]

    img = np.array(Image.open(os.path.join(a.fs, "assets/left.png")).convert("RGB"))
    x = torch.from_numpy(img).permute(2, 0, 1)[None].double()
    x = F.interpolate(x, size=(224, 320), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64).view(1, 3, 1, 1)
    x = ((x / 255.0 - mean) / std).float().double()
    H, W = x.shape[-2:]

    dino = DepthAnythingFeature(encoder="vitl")
    dino.load_state_dict(sub(ckpt, "feature.dino."))
    dino = dino.double().eval()
    args = SimpleNamespace(vit_size="vitl")
    cnet = ContextNetDino(args, output_dim=[[128] * 3, [128] * 3], downsample=2)
    cnet.load_state_dict(sub(ckpt, "cnet."))
    cnet = cnet.double().eval()
    cam = ChannelAttentionEnhancement(128)
    cam.load_state_dict(sub(ckpt, "cam."))
    sam = SpatialAttentionExtractor()
    sam.load_state_dict(sub(ckpt, "sam."))
    cam, sam = cam.double().eval(), sam.double().eval()

    with torch.no_grad():
        hr, wr = get_resize_keep_aspect_ratio(H, W, divider=int(np.lcm(14, 16)),
                                              max_H=1344, max_W=1344)
        x_in = F.interpolate(x, size=(hr, wr), mode="bicubic", align_corners=False)
        vit = dino(x_in)["out"]
        vit = F.interpolate(vit, size=(H // 4, W // 4), mode="bilinear",
                            align_corners=True).float().double()
        levels = cnet(x, vit_feat=vit)
        net = [torch.tanh(l[0]) for l in levels]
        inp = [torch.relu(l[1]) for l in levels]
        inp = [cam(t) * t for t in inp]
        att = [sam(t) for t in inp]

    d = os.path.join(a.out, "context_real")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    write(d, "image.bin", x)
    write(d, "vit.bin", vit)
    skip = ("down.", "norm.")
    for k, v in cnet.state_dict().items():
        if k.startswith(skip) or ".norm3." in k or k.endswith("num_batches_tracked"):
            continue
        write(d, f"w__cnet.{k}.bin", v)
    for prefix, m in [("cam.", cam), ("sam.", sam)]:
        for k, v in m.state_dict().items():
            write(d, f"w__{prefix}{k}.bin", v)
    for name, ts in [("net", net), ("inp", inp), ("att", att)]:
        for l, t in enumerate(ts):
            write(d, f"expected_{name}{l}.bin", t)
            print(name, l, tuple(t.shape), float(t.abs().max()))


if __name__ == "__main__":
    main()
