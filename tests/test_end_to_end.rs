// FoundationStereo end to end with a real checkpoint: the prelude graph (features,
// cost volume and filtering, initial disparity, context, geometry pyramid, stem), the
// refinement step graph run in the host loop, and the upsampling graph, against the
// model's own `forward` in FP64 on the sample stereo pair, at 224x320 and, small enough
// for RISC-V under qemu, at 128x160.
//
// Build the fixtures with tests/fixtures/end_to_end/gen_real.py, point
// `FS_REAL_FIXTURES` at the directory holding `end_to_end_real/` and run with
// `--ignored`.
//
// Runs on whichever backend teeny-runtime was built for: CUDA, or RISC-V under qemu
// (`--no-default-features --features qemu`, with
// `TEENYC_QEMU_RISCV64=/usr/bin/qemu-riscv64-static` if only the static qemu exists).

#![cfg(any(feature = "cuda", feature = "qemu"))]

mod common;

use common::stereo_block::{Block, compare, fixture};
use serial_test::serial;
use teeny_core::name_scope::name_scope;
use vision_rs::models::stereo::filtering::VOLUME_DIM;
use vision_rs::models::stereo::pipeline::{CORR_LEVELS, CORR_RADIUS, PRELUDE_OUTPUTS, prelude};
use vision_rs::models::stereo::update::{
    refinement_loop::{LoopFixed, LoopState, StepShape, build_step, run_refinement, step_inputs},
    update_block::UpdateBlockDims,
    upsample::upsample_disp,
};

/// `FoundationStereo.forward(left, right, iters=4)` on the sample pair at 224x320: the
/// initial disparity at 1/4 resolution and the upsampled full-resolution disparity after
/// every step. Full-resolution disparities reach ~45 px, so the tolerances are absolute
/// (5e-3 px, ~1e-4 relative).
#[test]
#[serial]
#[ignore = "needs FS_REAL_FIXTURES (real-checkpoint fixtures, not committed)"]
fn foundation_stereo_matches_real_checkpoint() -> anyhow::Result<()> {
    run("end_to_end_real", 224, 320, 4)
}

/// The same at 128x160 with 2 steps (`gen_real.py --size 128 160 --iters 2 --name
/// end_to_end_small`): the whole model, small enough to finish under qemu on RISC-V.
#[test]
#[serial]
#[ignore = "needs FS_REAL_FIXTURES (real-checkpoint fixtures, not committed)"]
fn foundation_stereo_small_matches_real_checkpoint() -> anyhow::Result<()> {
    run("end_to_end_small", 128, 160, 2)
}

/// Runs the three graphs on fixture directory `name` (an `h x w` pair, `iters` steps)
/// and checks the initial disparity and every step's upsampled disparity.
#[allow(non_snake_case)]
fn run(name: &str, H: usize, W: usize, ITERS: usize) -> anyhow::Result<()> {
    let root = std::env::var("FS_REAL_FIXTURES")
        .expect("set FS_REAL_FIXTURES to the real-checkpoint fixture directory");
    let dir = format!("{root}/{name}");
    let f = |n: &str| fixture(&dir, &format!("{n}.bin"));
    let device = teeny_test::model::runtime::open()?;

    // Prelude, once per pair.
    let img: &[usize] = &[1, 3, H, W];
    let mut pre = Block::new(
        &device,
        &format!("{name}/prelude"),
        &format!("{dir}/prelude"),
        &[("left", img), ("right", img)],
        |i| prelude::<f32>(&i[0], &i[1], (H, W)),
    )?;
    let out = pre.run(&[&f("left"), &f("right")])?;
    let get = |out_name: &str| {
        let i = PRELUDE_OUTPUTS
            .iter()
            .chain(&["stem_2x"])
            .position(|n| *n == out_name)
            .unwrap();
        &out[i]
    };
    compare(
        &format!("{name}/init_disp"),
        get("disp"),
        &f("expected_init_disp"),
        1e-3,
    );

    // Refinement step, loaded once.
    let dims = UpdateBlockDims::foundation_stereo();
    let shape = StepShape {
        batch: 1,
        h: H / 4,
        w: W / 4,
        groups: VOLUME_DIM,
        disps: 104,
        w2: W / 4,
        levels: CORR_LEVELS,
    };
    let names = step_inputs(&dims, &shape);
    let inputs: Vec<(&str, &[usize])> = names
        .iter()
        .map(|(n, s)| (n.as_str(), s.as_slice()))
        .collect();
    let mut step = Block::new(
        &device,
        &format!("{name}/step"),
        &format!("{dir}/step"),
        &inputs,
        |i| {
            let _g = name_scope("update_block");
            build_step::<f32>(dims, CORR_RADIUS, CORR_LEVELS, i)
        },
    )?;

    // Upsampling, run after every step to check each iteration's output.
    let c = dims.mask_out;
    let mut up = Block::new(
        &device,
        &format!("{name}/upsample"),
        &format!("{dir}/upsample"),
        &[
            ("disp", &[1, 1, H / 4, W / 4]),
            ("mask", &[1, c, H / 4, W / 4]),
            ("stem", &[1, c, H / 2, W / 2]),
        ],
        |i| vec![("", upsample_disp::<f32>(c)(&i[0], &i[1], &i[2]))],
    )?;

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
    let stem = get("stem_2x");
    let mut worst = 0.0f32;
    run_refinement(
        ITERS,
        &fixed,
        state,
        |feed| step.run(feed),
        |it, s| {
            let disp = up
                .run(&[&s.disp, &s.mask, stem])
                .expect("upsample")
                .remove(0);
            worst = worst.max(compare(
                &format!("{name}/disp_it{it}"),
                &disp,
                &f(&format!("expected_disp_it{it}")),
                5e-3,
            ));
        },
    )?;
    println!("{name}: worst |disp - torch| over {ITERS} steps = {worst}");
    Ok(())
}
