#!/usr/bin/env python3
"""Converts a FoundationStereo checkpoint (`model_best_bp2.pth`) to the safetensors
file the `foundation_stereo` example loads: every `state_dict` tensor as f32 under its
PyTorch key. Dropped: BatchNorm's integer `num_batches_tracked` counters, and
`ResidualBlock.norm3`, an alias of `downsample.1` (the same tensors, which safetensors
will not store twice).

    python convert.py ~/Documents/osn/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \\
        ~/.cache/vision-rs/models/foundation_stereo/23-51-11.safetensors

Needs torch and safetensors (FoundationStereo's environment has both).
"""
import os
import sys

import torch
from safetensors.torch import save_file


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1:]
    sd = torch.load(src, weights_only=False, map_location="cpu")["model"]
    out = {k: v.float().contiguous() for k, v in sd.items()
           if not k.endswith("num_batches_tracked") and ".norm3." not in k}
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    save_file(out, dst)
    print(f"{len(out)} tensors, {sum(v.numel() for v in out.values()) / 1e6:.1f}M parameters -> {dst}")


if __name__ == "__main__":
    main()
