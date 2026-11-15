/*
 * Copyright 2026 Teenygrad
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! FoundationStereo on a rectified stereo pair: disparity for the left image.
//!
//! ```text
//! python examples/foundation_stereo/convert.py \
//!     ~/Documents/osn/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
//!     ~/.cache/vision-rs/models/foundation_stereo/23-51-11.safetensors
//! source .env
//! cargo run --release --example foundation_stereo --features cuda -- \
//!     --left left.png --right right.png \
//!     --weights ~/.cache/vision-rs/models/foundation_stereo/23-51-11.safetensors
//! ```
//!
//! As FoundationStereo's `scripts/run_demo.py`: the images are scaled by `--scale`,
//! padded to a multiple of 32 by edge replication (`InputPadder`, centred), normalised,
//! and run through `forward(test_mode=True)` for `--iters` refinement steps; the
//! disparity is cropped back to the scaled image. Writes a colourised PNG (`--out`)
//! and, with `--raw`, the disparity as little-endian f32 (`H x W`, row-major).
//!
//! The model runs as three graphs ([`vision_rs::models::stereo::pipeline`]): the prelude
//! once, the refinement step once per iteration, the upsampling once. Every
//! intermediate buffer stays allocated (teeny-test's `RuntimeModel` frees nothing), so
//! memory grows quickly with the image: the default `--scale 0.4` runs FoundationStereo's
//! 960x540 sample at 384x224, peaking at 13.1 GB on CUDA; at 0.5 (480x288) it does not
//! fit in 16 GB.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use teeny_core::graph::{DtypeRepr, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_core::name_scope::name_scope;
use teeny_data::safetensors::SafeTensors;
use teeny_kernels::graph::TritonLowering;
use teeny_test::model::{RuntimeModel, runtime};
use vision_rs::models::stereo::filtering::VOLUME_DIM;
use vision_rs::models::stereo::pipeline::{
    CORR_LEVELS, CORR_RADIUS, PRELUDE_OUTPUTS, normalize_image, prelude,
};
use vision_rs::models::stereo::update::{
    ops::{input_like, torch_key},
    refinement_loop::{LoopFixed, LoopState, StepShape, build_step, run_refinement, step_inputs},
    update_block::UpdateBlockDims,
    upsample::upsample_disp,
};

/// FoundationStereo's `max_disp // 4` for the released checkpoints (`max_disp: 416`).
const DISPS: usize = 104;

#[derive(Parser)]
#[command(about = "FoundationStereo disparity for a rectified stereo pair")]
struct Args {
    /// Left image.
    #[arg(long)]
    left: PathBuf,
    /// Right image (same size).
    #[arg(long)]
    right: PathBuf,
    /// Checkpoint converted by examples/foundation_stereo/convert.py.
    #[arg(long)]
    weights: PathBuf,
    /// Colourised disparity.
    #[arg(long, default_value = "disparity.png")]
    out: PathBuf,
    /// Also write the raw disparity, little-endian f32, `H x W`.
    #[arg(long)]
    raw: Option<PathBuf>,
    /// Refinement iterations (`valid_iters`; 32 in the released configuration).
    #[arg(long, default_value_t = 32)]
    iters: usize,
    /// Scale applied to both images first (FoundationStereo's `--scale`).
    #[arg(long, default_value_t = 0.4)]
    scale: f32,
}

/// A compiled graph and the DAG nodes of its outputs.
struct Graph<'d> {
    model: RuntimeModel<'d>,
    outputs: Vec<usize>,
}

impl<'d> Graph<'d> {
    /// Traces `build` over inputs of `shapes`, compiles it and loads every weight slot
    /// from `weights` by its PyTorch key.
    fn new(
        device: &'d runtime::Device<'d>,
        label: &str,
        shapes: &[Vec<usize>],
        weights: &dyn Fn(&str) -> Result<Vec<f32>>,
        build: impl FnOnce(&[SymTensor]) -> Vec<(&'static str, SymTensor)>,
    ) -> Result<Self> {
        let t = Instant::now();
        let shape = |s: &[usize]| s.iter().map(|&d| Some(d)).collect::<Vec<_>>();
        let (first, graph) = SymTensor::input(DtypeRepr::F32, shape(&shapes[0]));
        let mut syms = vec![first.clone()];
        for s in &shapes[1..] {
            syms.push(input_like(&first, shape(s)));
        }
        let outs = build(&syms);
        let lowering = TritonLowering::new();
        let (_, graph_to_dag) =
            lowering.lower_with_mapping(&graph.borrow(), LoweringMode::Inference)?;
        let target = runtime::default_target(device)?;
        let mut model = RuntimeModel::compile(
            device,
            &graph.borrow(),
            &lowering,
            LoweringMode::Inference,
            &target,
            shapes[0][0],
            false,
        )
        .with_context(|| format!("compiling {label}"))?;
        let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
        for (slot, node, idx) in &slots {
            let key = torch_key(slot);
            let data = weights(&key).with_context(|| format!("{label}: weight {key}"))?;
            model.load_param_f32(*node, *idx, &data)?;
        }
        println!(
            "{label}: compiled, {} weight slots loaded ({:.1} s)",
            slots.len(),
            t.elapsed().as_secs_f32()
        );
        Ok(Self {
            model,
            outputs: outs.iter().map(|(_, t)| graph_to_dag[t.node_id]).collect(),
        })
    }

    fn run(&mut self, device: &runtime::Device<'d>, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        self.model.run(device, inputs, &self.outputs)
    }
}

/// An RGB image as planar `[3, h, w]` floats in 0..=255, scaled by `scale` (bilinear).
fn load(path: &Path, scale: f32) -> Result<(Vec<f32>, usize, usize)> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_rgb8();
    let img = if (scale - 1.0).abs() > f32::EPSILON {
        let w = ((img.width() as f32 * scale).round() as u32).max(1);
        let h = ((img.height() as f32 * scale).round() as u32).max(1);
        image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut planar = vec![0.0f32; 3 * h * w];
    for (x, y, p) in img.enumerate_pixels() {
        for c in 0..3 {
            planar[(c * h + y as usize) * w + x as usize] = p[c] as f32;
        }
    }
    Ok((planar, h, w))
}

/// FoundationStereo's `InputPadder(divis_by=32)`: `(top, bottom, left, right)` padding
/// that brings `(h, w)` to a multiple of 32, split as evenly as possible.
fn padding(h: usize, w: usize) -> [usize; 4] {
    let ph = (32 - h % 32) % 32;
    let pw = (32 - w % 32) % 32;
    [ph / 2, ph - ph / 2, pw / 2, pw - pw / 2]
}

/// Pads a planar `[c, h, w]` image by edge replication (`F.pad(mode='replicate')`).
fn pad(img: &[f32], c: usize, h: usize, w: usize, [t, b, l, r]: [usize; 4]) -> Vec<f32> {
    let (hp, wp) = (h + t + b, w + l + r);
    let mut out = vec![0.0f32; c * hp * wp];
    for ch in 0..c {
        for y in 0..hp {
            let sy = y.saturating_sub(t).min(h - 1);
            for x in 0..wp {
                let sx = x.saturating_sub(l).min(w - 1);
                out[(ch * hp + y) * wp + x] = img[(ch * h + sy) * w + sx];
            }
        }
    }
    out
}

/// A perceptual colour map from dark blue (near) to yellow-red (far): the
/// polynomial approximation of Google's Turbo.
fn turbo(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0) as f64;
    let poly = |c: [f64; 6]| {
        let v = c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    [
        poly([
            0.13572138,
            4.61539260,
            -42.66032258,
            132.13108234,
            -152.94239396,
            59.28637943,
        ]),
        poly([
            0.09140261,
            2.19418839,
            4.84296658,
            -14.18503333,
            4.27729857,
            2.82956604,
        ]),
        poly([
            0.10667330,
            12.64194608,
            -60.58204836,
            110.36276771,
            -89.90310912,
            27.34824973,
        ]),
    ]
}

fn main() -> Result<()> {
    dotenv::dotenv().ok();
    let args = Args::parse();
    let (left, h, w) = load(&args.left, args.scale)?;
    let (right, hr, wr) = load(&args.right, args.scale)?;
    ensure!(
        (h, w) == (hr, wr),
        "left {w}x{h} and right {wr}x{hr} differ in size"
    );
    let pads = padding(h, w);
    let (hp, wp) = (h + pads[0] + pads[1], w + pads[2] + pads[3]);
    let prep = |img: &[f32]| {
        let mut x = pad(img, 3, h, w, pads);
        normalize_image(&mut x);
        x
    };
    let (left, right) = (prep(&left), prep(&right));
    println!("images {w}x{h}, padded to {wp}x{hp}");

    let st = SafeTensors::from_pretrained(&args.weights)
        .with_context(|| format!("opening {}", args.weights.display()))?;
    let tensors = st.tensors()?;
    let weights = |key: &str| -> Result<Vec<f32>> {
        let view = tensors
            .tensor(key)
            .map_err(|_| anyhow::anyhow!("not in the checkpoint"))?;
        Ok(view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    };
    let device = runtime::open()?;

    let img = vec![1, 3, hp, wp];
    let mut pre = Graph::new(&device, "prelude", &[img.clone(), img], &weights, |i| {
        prelude::<f32>(&i[0], &i[1], (hp, wp))
    })?;
    let dims = UpdateBlockDims::foundation_stereo();
    let shape = StepShape {
        batch: 1,
        h: hp / 4,
        w: wp / 4,
        groups: VOLUME_DIM,
        disps: DISPS,
        w2: wp / 4,
        levels: CORR_LEVELS,
    };
    let step_shapes: Vec<Vec<usize>> = step_inputs(&dims, &shape)
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    let mut step = Graph::new(&device, "step", &step_shapes, &weights, |i| {
        let _g = name_scope("update_block");
        build_step::<f32>(dims, CORR_RADIUS, CORR_LEVELS, i)
    })?;
    let c = dims.mask_out;
    let mut up = Graph::new(
        &device,
        "upsample",
        &[
            vec![1, 1, hp / 4, wp / 4],
            vec![1, c, hp / 4, wp / 4],
            vec![1, c, hp / 2, wp / 2],
        ],
        &weights,
        |i| vec![("", upsample_disp::<f32>(c)(&i[0], &i[1], &i[2]))],
    )?;

    let t = Instant::now();
    let out = pre.run(&device, &[&left, &right])?;
    println!("prelude: {:.0} ms", t.elapsed().as_secs_f32() * 1e3);
    let get = |name: &str| {
        let i = PRELUDE_OUTPUTS
            .iter()
            .chain(&["stem_2x"])
            .position(|n| *n == name)
            .unwrap();
        &out[i]
    };
    let fixed = LoopFixed {
        inp: [get("inp0"), get("inp1"), get("inp2")],
        att: [get("att0"), get("att1"), get("att2")],
        geo: &[get("geo0"), get("geo1")],
        corr: &[get("corr0"), get("corr1")],
    };
    let state = LoopState {
        net: [
            get("net0").clone(),
            get("net1").clone(),
            get("net2").clone(),
        ],
        disp: get("disp").clone(),
        mask: Vec::new(),
    };
    let t = Instant::now();
    let state = run_refinement(
        args.iters,
        &fixed,
        state,
        |feed| step.run(&device, feed),
        |_, _| {},
    )?;
    println!(
        "refinement: {} iterations, {:.1} ms each",
        args.iters,
        t.elapsed().as_secs_f32() * 1e3 / args.iters.max(1) as f32
    );
    let t = Instant::now();
    let full = up
        .run(&device, &[&state.disp, &state.mask, get("stem_2x")])?
        .remove(0);
    println!("upsample: {:.0} ms", t.elapsed().as_secs_f32() * 1e3);

    // Crop the padding back off.
    let disp: Vec<f32> = (0..h)
        .flat_map(|y| {
            let row = (y + pads[0]) * wp + pads[2];
            full[row..row + w].to_vec()
        })
        .collect();
    let (lo, hi) = disp
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    println!("disparity {lo:.2}..{hi:.2} px");

    if let Some(raw) = &args.raw {
        let bytes: Vec<u8> = disp.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(raw, bytes).with_context(|| format!("writing {}", raw.display()))?;
    }
    let span = (hi - lo).max(f32::EPSILON);
    let vis = image::RgbImage::from_fn(w as u32, h as u32, |x, y| {
        image::Rgb(turbo((disp[y as usize * w + x as usize] - lo) / span))
    });
    vis.save(&args.out)
        .with_context(|| format!("writing {}", args.out.display()))?;
    println!("wrote {}", args.out.display());
    Ok(())
}
