// The FoundationStereo example's pooled executor (examples/foundation_stereo/pooled.rs),
// which shares activation buffers between nodes with disjoint lifetimes, against
// teeny-test's `RuntimeModel`, which gives every node its own: on the same graph and
// weights the outputs must be bit-identical (same kernels, same order; only the
// buffers differ). Uses the committed `cost_agg` hourglass fixture, whose skip
// connections and up/down levels keep buffers alive across long stretches.
//
// Runs on whichever backend teeny-runtime was built for: CUDA, or RISC-V under qemu
// (`--no-default-features --features qemu`, with
// `TEENYC_QEMU_RISCV64=/usr/bin/qemu-riscv64-static` if only the static qemu exists).

#![cfg(any(feature = "cuda", feature = "qemu"))]

mod common;
#[path = "../examples/foundation_stereo/pooled.rs"]
#[allow(dead_code)]
mod pooled;

use common::stereo_block::fixture;
use pooled::PooledModel;
use serial_test::serial;
use teeny_core::graph::{DtypeRepr, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_core::name_scope::name_scope;
use teeny_kernels::graph::TritonLowering;
use teeny_test::model::{RuntimeModel, runtime};
use vision_rs::models::stereo::filtering::hourglass;
use vision_rs::models::stereo::update::ops::{input_like, torch_key};

#[test]
#[serial]
fn pooled_model_matches_runtime_model_bitwise() -> anyhow::Result<()> {
    const DIR: &str = "filtering/hourglass";
    let shapes: [&[usize]; 4] = [
        &[1, 8, 16, 8, 16],
        &[1, 12, 4, 8],
        &[1, 16, 2, 4],
        &[1, 20, 1, 2],
    ];
    let shape = |s: &[usize]| s.iter().map(|&d| Some(d)).collect::<Vec<_>>();
    let (first, graph) = SymTensor::input(DtypeRepr::F32, shape(shapes[0]));
    let mut i = vec![first.clone()];
    for s in &shapes[1..] {
        i.push(input_like(&first, shape(s)));
    }
    let out = {
        let _g = name_scope("cost_agg");
        hourglass::<f32>(&i[0], [&i[1], &i[1], &i[2], &i[3]], 8, [8, 12, 16, 20])
    };

    let device = runtime::open()?;
    let target = runtime::default_target(&device)?;
    let lowering = TritonLowering::new();
    let (_, graph_to_dag) =
        lowering.lower_with_mapping(&graph.borrow(), LoweringMode::Inference)?;
    let out_idx = graph_to_dag[out.node_id];
    let inputs: Vec<Vec<f32>> = ["cv", "f1", "f2", "f3"]
        .iter()
        .map(|n| fixture(DIR, &format!("{n}.bin")))
        .collect();
    let feed: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();

    let mut plain = RuntimeModel::compile(
        &device,
        &graph.borrow(),
        &lowering,
        LoweringMode::Inference,
        &target,
        1,
        false,
    )?;
    let slots: Vec<(String, usize, usize)> = plain.param_info_named().collect();
    for (slot, node, idx) in &slots {
        plain.load_param_f32(
            *node,
            *idx,
            &fixture(DIR, &format!("w__{}.bin", torch_key(slot))),
        )?;
    }
    let want = plain.run(&device, &feed, &[out_idx])?.remove(0);

    let mut pooled = PooledModel::compile(&device, &graph.borrow(), &lowering, &target, 1)?;
    let pslots = pooled.param_info_named();
    assert_eq!(pslots, slots, "weight slots differ");
    for (slot, node, idx) in &pslots {
        pooled.load_param_f32(
            *node,
            *idx,
            &fixture(DIR, &format!("w__{}.bin", torch_key(slot))),
        )?;
    }
    println!(
        "pooled: activations {} B (one buffer per node: {} B)",
        pooled.pooled_bytes, pooled.unpooled_bytes
    );
    assert!(
        pooled.pooled_bytes < pooled.unpooled_bytes,
        "no buffer was shared"
    );
    // Twice: the second run starts from buffers holding the first run's stale data.
    for run in 0..2 {
        let got = pooled.run(&device, &feed, &[out_idx])?.remove(0);
        assert!(
            got.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "run {run}: pooled output differs from RuntimeModel's"
        );
    }
    Ok(())
}
