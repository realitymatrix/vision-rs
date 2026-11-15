# Reproducing the FoundationStereo port

How to check vision-rs's FoundationStereo port against PyTorch, on CUDA and on RISC-V
(RVV under `qemu-riscv64`). You'll need about 5 GB of free disk for the fixtures, and
an NVIDIA GPU for the CUDA runs.

The tests that use made-up weights run with a plain `cargo test`, because their fixtures
are committed. The tests that use the real checkpoint are `#[ignore]`d. Their fixtures
add up to about 4.7 GB, so they aren't committed, and you build them yourself with the
steps below.

## 1. Toolchain

Set up vision-rs as the README's "Installing the toolchain" section describes, then run
`cp .env.dev .env` and adjust the paths for your machine.

For RISC-V you also need these packages (Debian/Ubuntu names):

```bash
sudo apt install qemu-user-static gcc-riscv64-linux-gnu
```

## 2. FoundationStereo and its checkpoint

1. Clone upstream FoundationStereo, then set up its Python environment as its README
   describes.

   ```bash
   git clone https://github.com/NVlabs/FoundationStereo ~/FoundationStereo
   ```

   Every reference is generated from upstream's own code. Nothing has to be patched,
   because the generators set what they need at run time.
2. Download the `23-51-11` checkpoint folder: it's the ViT-L model, and the link is in
   upstream's README. Put the whole folder at
   `~/FoundationStereo/pretrained_models/23-51-11/`. It holds `cfg.yaml` and
   `model_best_bp2.pth`.

The generators also use upstream's `assets/left.png` and `assets/right.png` as the stereo
pair.

## 3. Build the fixtures

Run these from vision-rs, in FoundationStereo's Python environment:

```bash
FS=~/FoundationStereo
RUN=pretrained_models/23-51-11
OUT=~/.cache/vision-rs/fs_fixtures

for g in edgenext feature_decoder dinov2 dpt context cost_volume filtering end_to_end; do
  python tests/fixtures/$g/gen_real.py --fs $FS --run $RUN --out $OUT
done
python tests/fixtures/stem/generate.py --fs $FS --run $RUN --out $OUT
python tests/fixtures/end_to_end/gen_real.py --fs $FS --run $RUN --out $OUT \
  --size 128 160 --iters 2 --name end_to_end_small
python tests/fixtures/gen_update_block_real.py --fs $FS --run $RUN --out $OUT
```

These commands write the following directories under `$OUT`:

| Directory | Written by | Used by |
|---|---|---|
| `edgenext_real` | `edgenext/gen_real.py` | `backbone_matches_real_checkpoint` |
| `feature_real` | `feature_decoder/gen_real.py` | `feature_decoder_matches_real_checkpoint` |
| `dinov2_real` | `dinov2/gen_real.py` | `vit_encoder_matches_real_checkpoint`, `bicubic_resize_matches_real_image` |
| `dpt_real` | `dpt/gen_real.py` | `dpt_head_matches_real_checkpoint` |
| `context_real` | `context/gen_real.py` | `context_matches_real_checkpoint` |
| `cost_volume_real` | `cost_volume/gen_real.py` | `comb_volume_matches_real_checkpoint` |
| `filtering_real`, `classifier_real` | `filtering/gen_real.py` | `filtering_matches_real_checkpoint`, `classifier_matches_real_checkpoint` |
| `stem_real` | `stem/generate.py` | `stem_2_matches_real_checkpoint` |
| `end_to_end_real` | `end_to_end/gen_real.py` | `foundation_stereo_matches_real_checkpoint` (224×320, 4 steps) |
| `end_to_end_small` | `end_to_end/gen_real.py --size 128 160 --iters 2 --name end_to_end_small` | `foundation_stereo_small_matches_real_checkpoint` |
| `update_block_real_it{0,8}` | `gen_update_block_real.py` | `update_block_matches_real_checkpoint` |

Each script's docstring says exactly what it writes.

Most references run in FP64 on the CPU. `end_to_end/gen_real.py` runs the whole of
`FoundationStereo.forward` in FP64: it sets the `vit_size` that `cfg.yaml` lacks, and
keeps the refinement loop's `.float()` casts in FP64. So the generators take a while.
`gen_update_block_real.py` expects a GPU for the forward pass that captures its inputs.

## 4. Run on CUDA

```bash
source .env
export FS_REAL_FIXTURES=~/.cache/vision-rs/fs_fixtures
cargo test --test test_end_to_end -- --ignored --nocapture
```

That runs both whole-model tests, at 224×320 and at 128×160, in about 20 s.

To run every stage test as well, including the ones with made-up weights:

```bash
cargo test --test test_edgenext --test test_feature_decoder --test test_dinov2 \
  --test test_dpt --test test_interpolate_bicubic --test test_context \
  --test test_cost_volume --test test_filtering --test test_stem \
  --test test_refinement_head --test test_end_to_end -- --include-ignored
```

Expected results on CUDA, all against PyTorch:

| Test | Worst error |
|---|---|
| whole model, 224×320, 4 steps: initial disparity | 2.7e-5 px |
| whole model, 224×320, 4 steps: full-res disparity, steps 1–4 (≤ 45 px) | 6.5e-4 / 9.7e-4 / 1.9e-3 / 1.3e-3 px |
| whole model, 128×160, 2 steps: initial / step 1 / step 2 (≤ 22 px) | 1.5e-5 / 2.0e-4 / 1.6e-4 px |
| EdgeNeXt backbone | ≤ 3.8e-6 rel. |
| feature decoder x4 / x8 / x16 | 1.3e-6 / 4.0e-7 / 1.9e-7 rel. |
| DINOv2 ViT-L l4 / l11 / l17 / l23 | 2.1e-6 / 2.1e-6 / 2.5e-5 / 1.7e-5 rel. |
| DPT head `out` / `vit_feat` | 3.7e-6 / 2.5e-6 rel. |
| bicubic resize, real image | 6.8e-6 rel. |
| context `net` 1/4, 1/8, 1/16 | 2.5e-5 / 1.2e-5 / 3.1e-6 abs. |
| combined cost volume | 4.2e-7 rel. |
| filtering, gated / out | 6.6e-7 / 1.2e-6 rel. |
| classifier logits / init_disp | 1.4e-6 / 1.6e-6 rel. |
| `stem_2` | 1.1e-6 rel. |
| update block `net[0]`, iterations 0 / 8 | 1.8e-5 / 2.6e-5 abs. |

The tests themselves check these against tolerances; the numbers above are what we
measured. Each test prints its errors with `--nocapture`.

## 5. Run on RISC-V (RVV under qemu)

Build into a separate target directory, so the RISC-V build doesn't overwrite the CUDA
one:

```bash
source .env
export FS_REAL_FIXTURES=~/.cache/vision-rs/fs_fixtures
export TEENYC_QEMU_RISCV64=/usr/bin/qemu-riscv64-static
export CARGO_TARGET_DIR=target/riscv-tests
nice cargo test --no-default-features --features qemu --test test_end_to_end small \
  -- --ignored --nocapture
```

This runs the whole model at 128×160 with 2 steps. It took 3 h 02 m on our machine,
because every kernel launch is a separate single-threaded qemu process. We measured:

| Output | Worst error vs PyTorch |
|---|---|
| initial disparity | 1.4e-5 px |
| full-res disparity, step 1 | 2.3e-4 px |
| full-res disparity, step 2 | 2.0e-4 px |

The stage tests run the same way, with `--test <file> <name>`. Under qemu:

- the ViT-L test (`test_dinov2`) takes about an hour;
- `context` takes about 10 minutes;
- the small tests take seconds.

The 224×320 whole-model test and `filtering_matches_real_checkpoint` are far too slow
under qemu, so skip them there. The 128×160 test covers the same code.

## 6. The demo on your own images

First, convert the checkpoint to safetensors in FoundationStereo's environment:

```bash
python examples/foundation_stereo/convert.py \
  ~/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
  ~/.cache/vision-rs/models/foundation_stereo/23-51-11.safetensors
```

Then run the demo:

```bash
cargo run --release --example foundation_stereo --features cuda -- \
  --left ~/FoundationStereo/assets/left.png --right ~/FoundationStereo/assets/right.png \
  --weights ~/.cache/vision-rs/models/foundation_stereo/23-51-11.safetensors \
  --out disparity.png
```

Useful flags:

- `--scale` resizes both images first. It works as in upstream's `run_demo.py`, and the
  default is 1.0.
- `--iters` sets the number of refinement steps. The default is 32.
- `--raw <file>` also writes the disparity as little-endian f32.

The demo reuses buffers between operations, so a 960×540 pair fits in about 7 GB of
GPU memory.
