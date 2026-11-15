#!/usr/bin/env python3
"""Real-checkpoint fixtures for `filtering_matches_real_checkpoint`
(tests/test_filtering.rs): FoundationStereo's cost-volume filtering with a real
checkpoint (`corr_stem`, `corr_feature_att`, `cost_agg`), on the real combined volume
of the sample stereo pair, in FP64.

Both images are prepared as the other real fixtures prepare them (bilinear to 224x320,
normalised) and run through the real `Feature`; the combined volume is built from its
x4 maps with the real `proj_cmb`. Inputs are rounded to f32 as the test feeds them.
Writes to <out>/filtering_real/:
    cv.bin                  combined volume [1, 32, 104, 56, 80]
    f{0,1,2,3}.bin          left features [1, 224, 56, 80], [1, 192, 28, 40],
                            [1, 320, 14, 20], [1, 304, 7, 10]
    w__<key>.bin            `corr_stem.*`, `corr_feature_att.*`, `cost_agg.*`
    expected_att.bin        corr_feature_att(corr_stem(cv), f0), [1, 28, 104, 56, 80]
    expected.bin            cost_agg(that, features)

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


def f32(t):
    return t.float().double()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.fs)
    from core.extractor import Feature
    from core.foundation_stereo import hourglass
    from core.submodule import (BasicConv, FeatureAtt, ResnetBasicBlock3D,
                                build_concat_volume, build_gwc_volume)

    ckpt = torch.load(os.path.join(a.fs, a.run, "model_best_bp2.pth"),
                      weights_only=False, map_location="cpu")["model"]
    feat = Feature(SimpleNamespace(vit_size="vitl"))
    feat.load_state_dict(sub(ckpt, "feature."))
    feat = feat.double().eval()
    proj = nn.Conv2d(224, 12, 1)
    proj.load_state_dict(sub(ckpt, "proj_cmb."))
    vd = 28
    stem = nn.Sequential(
        nn.Conv3d(32, vd, kernel_size=1),
        BasicConv(vd, vd, kernel_size=3, padding=1, is_3d=True),
        ResnetBasicBlock3D(vd, vd, kernel_size=3, stride=1, padding=1),
        ResnetBasicBlock3D(vd, vd, kernel_size=3, stride=1, padding=1),
    )
    stem.load_state_dict(sub(ckpt, "corr_stem."))
    att = FeatureAtt(vd, 224)
    att.load_state_dict(sub(ckpt, "corr_feature_att."))
    agg = hourglass(cfg={"max_disp": 416}, in_channels=vd, feat_dims=feat.d_out)
    agg.load_state_dict(sub(ckpt, "cost_agg."))
    proj, stem, att, agg = (m.double().eval() for m in (proj, stem, att, agg))

    with torch.no_grad():
        fl = [f32(t) for t in feat(load(a.fs, "left.png"))[0]]
        fr = f32(feat(load(a.fs, "right.png"))[0][0])
        cv = f32(torch.cat([build_gwc_volume(fl[0], fr, 104, 8),
                            build_concat_volume(proj(fl[0]), proj(fr), 104)], dim=1))
        v = att(stem(cv), fl[0])
        out = agg(v, fl)

    d = os.path.join(a.out, "filtering_real")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    write(d, "cv.bin", cv)
    for i, t in enumerate(fl):
        write(d, f"f{i}.bin", t)
    for prefix, m in [("corr_stem", stem), ("corr_feature_att", att), ("cost_agg", agg)]:
        for k, t in m.state_dict().items():
            if not k.endswith("num_batches_tracked"):
                write(d, f"w__{prefix}.{k}.bin", t)
    write(d, "expected_att.bin", v)
    write(d, "expected.bin", out)
    print([tuple(t.shape) for t in fl], tuple(out.shape), float(out.abs().max()))


if __name__ == "__main__":
    main()
