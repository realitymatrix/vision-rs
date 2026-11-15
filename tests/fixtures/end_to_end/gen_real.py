#!/usr/bin/env python3
"""Real-checkpoint fixtures for tests/test_end_to_end.rs: the whole of
`FoundationStereo.forward` on the sample stereo pair (bilinear to `--size`, 224x320 by
default) with a real
checkpoint, in FP64 on the CPU (`mixed_precision` off, and the loop's `.float()` casts
made casts to double), `iters` refinement steps.

Both images are bilinear-resized to `--size` (0..255) and fed to the model, which
normalises them; the test is fed the same images normalised in f32. The checkpoint is
split by graph, each directory holding exactly the weights its graph loads:
    <out>/<name>/                   `end_to_end_real` by default
        left.bin, right.bin         normalised images [1, 3, H, W]
        prelude/w__<key>.bin        feature, proj_cmb, corr_stem, corr_feature_att,
                                    cost_agg, classifier, cnet, cam, sam, stem_2
                                    (without modules that never run)
        step/w__update_block.*      the refinement step
        upsample/w__spx_*           upsample_disp
        expected_init_disp.bin      [1, 1, H/4, W/4]
        expected_disp_it{i}.bin     the upsampled disparity after step i, [1, 1, H, W]

    python gen_real.py --fs ~/Documents/osn/FoundationStereo \\
        --run pretrained_models/23-51-11 --out ~/.cache/vision-rs/fs_fixtures [--iters 4]
    # the smaller case for RISC-V under qemu:
    python gen_real.py ... --size 128 160 --iters 2 --name end_to_end_small
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PRELUDE = ("feature.", "proj_cmb.", "corr_stem.", "corr_feature_att.", "cost_agg.",
           "classifier.", "cnet.", "cam.", "sam.", "stem_2.")
NEVER_RUN = (
    "feature.dino.depth_anything.depth_head.scratch.output_conv2.",   # discarded depth
    "feature.dino.depth_anything.depth_head.scratch.refinenet4.resConfUnit1.",
    "feature.dino.depth_anything.pretrained.mask_token",
    "cnet.down.", "cnet.norm.",
)


def write(d, name, t):
    os.makedirs(d, exist_ok=True)
    t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(d, name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--size", type=int, nargs=2, default=[224, 320], metavar=("H", "W"),
                   help="image size, a multiple of 32")
    p.add_argument("--name", default="end_to_end_real", help="output directory name")
    a = p.parse_args()
    sys.path.insert(0, a.fs)
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo

    cfg = OmegaConf.load(os.path.join(a.fs, a.run, "cfg.yaml"))
    cfg.mixed_precision = False
    cfg.vit_size = "vitl"   # set by run_demo.py's arguments, not in cfg.yaml
    ckpt = torch.load(os.path.join(a.fs, a.run, "model_best_bp2.pth"),
                      weights_only=False, map_location="cpu")["model"]
    model = FoundationStereo(cfg)
    model.load_state_dict(ckpt)
    model = model.double().eval()

    def load(name):
        img = np.array(Image.open(os.path.join(a.fs, "assets", name)).convert("RGB"))
        x = torch.from_numpy(img).permute(2, 0, 1)[None].double()
        return F.interpolate(x, size=tuple(a.size), mode="bilinear", align_corners=False)

    raw_l, raw_r = load("left.png"), load("right.png")
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64).view(1, 3, 1, 1)
    # The f32 images the port is fed; the model gets the matching 0..255 values.
    norm_l = ((raw_l / 255.0 - mean) / std).float().double()
    norm_r = ((raw_r / 255.0 - mean) / std).float().double()
    raw_l, raw_r = (norm_l * std + mean) * 255.0, (norm_r * std + mean) * 255.0

    # `forward` casts the geometry volumes and disparity to `.float()` for the loop;
    # in an FP64 run that clashes with the FP64 update block. Keep FP64 throughout by
    # making `.float()` a cast to double for the call.
    to_float = torch.Tensor.float
    torch.Tensor.float = lambda self, *args, **kw: self.double()
    try:
        with torch.no_grad():
            init_disp, preds = model(raw_l, raw_r, iters=a.iters, test_mode=False)
    finally:
        torch.Tensor.float = to_float

    d = os.path.join(a.out, a.name)
    if os.path.isdir(d):
        for root, _, files in os.walk(d):
            for f in files:
                os.remove(os.path.join(root, f))
    write(d, "left.bin", norm_l)
    write(d, "right.bin", norm_r)
    for k, v in model.state_dict().items():
        if k.endswith("num_batches_tracked") or ".norm3." in k or k.startswith(NEVER_RUN):
            continue
        if k.startswith(PRELUDE):
            write(os.path.join(d, "prelude"), f"w__{k}.bin", v)
        elif k.startswith("update_block."):
            write(os.path.join(d, "step"), f"w__{k}.bin", v)
        elif k.startswith(("spx_2_gru.", "spx_gru.")):
            write(os.path.join(d, "upsample"), f"w__{k}.bin", v)
    write(d, "expected_init_disp.bin", init_disp)
    for i, t in enumerate(preds):
        write(d, f"expected_disp_it{i}.bin", t)
    print(tuple(init_disp.shape), [round(float(t.mean()), 3) for t in preds])


if __name__ == "__main__":
    main()
