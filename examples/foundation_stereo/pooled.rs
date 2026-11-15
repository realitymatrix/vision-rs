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

//! A graph executor like teeny-test's `RuntimeModel` (same lowering, compilation, weight
//! slots and one-kernel-at-a-time launches on whichever backend `teeny-runtime` was built
//! for), but with activation buffers **shared** between nodes whose lifetimes do not
//! overlap. `RuntimeModel` gives every node its own output buffer for the model's
//! lifetime, so memory is the sum of all activations; here it is close to the largest set
//! live at any one step.
//!
//! **Plan** (at compile time, for the fixed topological order): a node's buffer is busy
//! from its launch until its last consumer's launch; graph inputs are busy from the start,
//! since `run` uploads them all first. Walking the order, each node takes the smallest
//! free buffer that fits, or a new one; buffers free up after their last reader runs.
//! All buffers are allocated once, zeroed, after planning.
//!
//! Reuse means a node's output does not survive the run, so [`PooledModel::run`] copies
//! each requested output to the host as soon as its node has run. Kernels must write
//! their whole output (they all do in the forward pass: teeny-kernels' accumulating
//! `atomic_add` kernels are backward-only), since a reused buffer holds stale data.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use teeny_core::device::Device as _;
use teeny_core::device::buffer::Buffer as _;
use teeny_core::graph::{DtypeRepr, Graph};
use teeny_core::model::{Lowering, LoweringMode, RawPtr, RuntimeOp};
use teeny_test::ExecKernel;
use teeny_test::model::runtime::{self, Buffer, Device, Program, Target};
use teeny_test::model::{PackedArgs, PackedKernel};

fn dtype_bytes(dtype: DtypeRepr) -> usize {
    match dtype {
        DtypeRepr::Bool | DtypeRepr::I8 | DtypeRepr::U8 => 1,
        DtypeRepr::I16 | DtypeRepr::U16 | DtypeRepr::F16 | DtypeRepr::BF16 => 2,
        DtypeRepr::I32 | DtypeRepr::U32 | DtypeRepr::F32 => 4,
        DtypeRepr::I64 | DtypeRepr::U64 | DtypeRepr::F64 => 8,
    }
}

/// `n` elements of `dtype` in f32 words (rounded up, at least one).
fn words(n: usize, dtype: DtypeRepr) -> usize {
    (n * dtype_bytes(dtype)).div_ceil(4).max(1)
}

fn zeroed<'a>(device: &Device<'a>, words: usize) -> Result<Buffer<'a, f32>> {
    let mut buf = device.buffer::<f32>(words)?;
    buf.to_device(&vec![0.0f32; words])?;
    Ok(buf)
}

fn ptr(buf: &Buffer<'_, f32>) -> RawPtr {
    buf.as_device_ptr() as RawPtr
}

/// A padded output buffer, for ops whose kernel writes rows with a stride wider than the
/// tight layout; private to its node, as in `RuntimeModel`.
struct Padded<'a> {
    buf: Buffer<'a, f32>,
    natural_stride: usize,
    required_stride: usize,
    n_rows: usize,
}

struct NodeOp<'a> {
    program: Rc<Program<'static, PackedKernel>>,
    runtime_op: Arc<dyn RuntimeOp>,
    params: Vec<Buffer<'a, f32>>,
    padded: Option<Padded<'a>>,
}

struct Node<'a> {
    /// `None` for graph inputs.
    op: Option<NodeOp<'a>>,
    shape: Vec<usize>,
    /// The pooled buffer holding this node's output.
    slot: usize,
}

/// A lowered graph compiled for the `teeny-runtime` backend, with pooled activations.
pub struct PooledModel<'a> {
    nodes: Vec<Node<'a>>,
    parents: Vec<Vec<usize>>,
    topo: Vec<usize>,
    names: HashMap<usize, String>,
    slots: Vec<Buffer<'a, f32>>,
    /// Bytes of the pooled buffers, and what one buffer per node would have taken.
    pub pooled_bytes: usize,
    /// See [`Self::pooled_bytes`].
    pub unpooled_bytes: usize,
}

impl<'a> PooledModel<'a> {
    /// Lowers `graph`, compiles every op's forward kernel for `target`, allocates the
    /// parameters, and plans and allocates the shared activation buffers.
    pub fn compile<'l, L: Lowering<'l>>(
        device: &Device<'a>,
        graph: &Graph,
        lowering: &L,
        target: &Target,
        batch_size: usize,
    ) -> Result<Self> {
        let (dag, graph_to_dag) = lowering.lower_with_mapping(graph, LoweringMode::Inference)?;
        let n = dag.len();
        let topo = dag.topological_sort();
        let parents: Vec<Vec<usize>> = (0..n).map(|i| dag.node(i).parents.clone()).collect();
        let mut names: HashMap<usize, String> = graph
            .names
            .iter()
            .filter_map(|(&g, name)| Some((*graph_to_dag.get(g)?, name.clone())))
            .collect();
        for (d, name) in lowering.extra_dag_names(graph, &graph_to_dag) {
            names.entry(d).or_insert(name);
        }

        let mut programs: HashMap<String, Rc<Program<'static, PackedKernel>>> = HashMap::new();
        let mut shapes: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut ops: Vec<Option<NodeOp<'a>>> = (0..n).map(|_| None).collect();
        let mut sizes = vec![0usize; n];
        for &idx in &topo {
            let op = dag.node(idx).value.as_ref();
            let raw: Vec<usize> = op
                .output_shape()
                .iter()
                .map(|d| d.unwrap_or(batch_size))
                .collect();
            let dtype = op.output_dtype();
            let Some(runtime_op) = op.runtime_op().filter(|_| !op.is_input()) else {
                sizes[idx] = words(raw.iter().product(), dtype);
                shapes[idx] = raw;
                continue;
            };
            let parent_shapes: Vec<&[usize]> =
                parents[idx].iter().map(|&p| shapes[p].as_slice()).collect();
            let shape = runtime_op.compute_concrete_output_shape(&parent_shapes, &raw);
            ensure!(
                !op.forward_kernel_source().is_empty(),
                "no forward kernel source for op {}",
                op.name()
            );
            let path = runtime::compile_kernel(&ExecKernel(op), target, false, false)
                .with_context(|| format!("compiling node {idx} ({})", op.name()))?;
            let program = match programs.get(&path) {
                Some(p) => Rc::clone(p),
                None => {
                    let p = Rc::new(
                        runtime::load_program::<PackedKernel>(&path)
                            .with_context(|| format!("loading {path}"))?,
                    );
                    programs.insert(path, Rc::clone(&p));
                    p
                }
            };
            let param_shapes = runtime_op.param_shapes(&parent_shapes, &shape);
            let mut params = Vec::with_capacity(param_shapes.len());
            for (pi, ps) in param_shapes.iter().enumerate() {
                let mut buf = zeroed(device, words(ps.iter().product(), dtype))?;
                if let Some(bytes) = runtime_op.param_init_data(pi) {
                    let w: Vec<f32> = bytes
                        .chunks(4)
                        .map(|c| {
                            let mut b = [0u8; 4];
                            b[..c.len()].copy_from_slice(c);
                            f32::from_ne_bytes(b)
                        })
                        .collect();
                    buf.to_device(&w)?;
                }
                params.push(buf);
            }
            let n_elems: usize = shape.iter().product();
            let natural = runtime_op.forward_output_row_elems(&shape);
            let required = runtime_op.forward_output_row_stride(&shape);
            let padded = if required > natural {
                let n_rows = n_elems / natural.max(1);
                Some(Padded {
                    buf: zeroed(device, words(n_rows * required, dtype))?,
                    natural_stride: natural,
                    required_stride: required,
                    n_rows,
                })
            } else {
                None
            };
            sizes[idx] = words(n_elems, dtype);
            shapes[idx] = shape;
            ops[idx] = Some(NodeOp {
                program,
                runtime_op,
                params,
                padded,
            });
        }

        // Lifetimes in topological steps: inputs are born at step 0 (uploaded before any
        // launch), other nodes at their own step; each dies after its last reader.
        let step_of: Vec<usize> = {
            let mut s = vec![0; n];
            for (i, &idx) in topo.iter().enumerate() {
                s[idx] = i;
            }
            s
        };
        let birth: Vec<usize> = (0..n)
            .map(|i| if ops[i].is_none() { 0 } else { step_of[i] })
            .collect();
        let mut death = birth.clone();
        for (child, ps) in parents.iter().enumerate() {
            for &p in ps {
                death[p] = death[p].max(step_of[child]);
            }
        }

        // Best-fit assignment, inputs first so they never share a buffer with each other.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (birth[i], ops[i].is_some(), step_of[i]));
        let mut caps: Vec<usize> = Vec::new();
        let mut busy_until: Vec<Option<usize>> = Vec::new(); // death of the current holder
        let mut slot_of = vec![0usize; n];
        for &i in &order {
            let free = |s: usize, busy: &[Option<usize>]| match busy[s] {
                None => true,
                Some(d) => d < birth[i],
            };
            let best = (0..caps.len())
                .filter(|&s| free(s, &busy_until) && caps[s] >= sizes[i])
                .min_by_key(|&s| caps[s]);
            let s = match best {
                Some(s) => s,
                None => {
                    caps.push(sizes[i]);
                    busy_until.push(None);
                    caps.len() - 1
                }
            };
            busy_until[s] = Some(death[i]);
            slot_of[i] = s;
        }
        let slots = caps
            .iter()
            .map(|&w| zeroed(device, w))
            .collect::<Result<Vec<_>>>()?;

        let nodes = ops
            .into_iter()
            .zip(shapes)
            .enumerate()
            .map(|(i, (op, shape))| Node {
                op,
                shape,
                slot: slot_of[i],
            })
            .collect();
        Ok(Self {
            nodes,
            parents,
            topo,
            names,
            slots,
            pooled_bytes: caps.iter().sum::<usize>() * 4,
            unpooled_bytes: sizes.iter().sum::<usize>() * 4,
        })
    }

    /// `(scoped slot name, node, parameter index)` for every named weight slot, as
    /// `RuntimeModel::param_info_named`.
    pub fn param_info_named(&self) -> Vec<(String, usize, usize)> {
        let mut out = Vec::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            let (Some(op), Some(name)) = (&node.op, self.names.get(&idx)) else {
                continue;
            };
            let slots = op.runtime_op.param_names();
            for pi in 0..op.params.len() {
                let slot = slots.get(pi).copied().unwrap_or("");
                if !slot.is_empty() {
                    out.push((format!("{name}.{slot}"), idx, pi));
                }
            }
        }
        out
    }

    /// Uploads `data` into parameter `pi` of node `idx`.
    pub fn load_param_f32(&mut self, idx: usize, pi: usize, data: &[f32]) -> Result<()> {
        let op = self.nodes[idx]
            .op
            .as_mut()
            .ok_or_else(|| anyhow!("node {idx} has no parameters"))?;
        op.params[pi].to_device(data)?;
        Ok(())
    }

    fn read(&self, idx: usize) -> Result<Vec<f32>> {
        let node = &self.nodes[idx];
        // Buffers copy a prefix: the slot may be larger than this node's output.
        let mut out = vec![0.0f32; node.shape.iter().product()];
        self.slots[node.slot].to_host(&mut out)?;
        Ok(out)
    }

    /// Runs one forward pass. `inputs` feed the graph inputs in topological order; returns
    /// the outputs of the nodes in `outputs`, in that order, each copied to the host as
    /// soon as its node has run.
    pub fn run(
        &mut self,
        device: &Device<'a>,
        inputs: &[&[f32]],
        outputs: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let mut got: Vec<Option<Vec<f32>>> = vec![None; outputs.len()];
        let mut cursor = 0;
        for &idx in &self.topo {
            if self.nodes[idx].op.is_none() {
                let data = inputs
                    .get(cursor)
                    .ok_or_else(|| anyhow!("too few inputs: needed >{cursor}"))?;
                let slot = self.nodes[idx].slot;
                self.slots[slot].to_device(data)?;
                cursor += 1;
            }
        }
        for (o, &idx) in outputs.iter().enumerate() {
            if self.nodes[idx].op.is_none() {
                got[o] = Some(self.read(idx)?);
            }
        }
        for step in 0..self.topo.len() {
            let idx = self.topo[step];
            if self.nodes[idx].op.is_none() {
                continue;
            }
            self.launch(device, idx)?;
            for (o, &out) in outputs.iter().enumerate() {
                if out == idx {
                    got[o] = Some(self.read(idx)?);
                }
            }
        }
        Ok(got
            .into_iter()
            .map(|g| g.expect("every output node runs"))
            .collect())
    }

    fn launch(&mut self, device: &Device<'a>, idx: usize) -> Result<()> {
        let node = &self.nodes[idx];
        let op = node.op.as_ref().expect("launch of an input node");
        let act: Vec<(RawPtr, &[usize])> = self.parents[idx]
            .iter()
            .map(|&p| {
                let pn = &self.nodes[p];
                (ptr(&self.slots[pn.slot]), pn.shape.as_slice())
            })
            .collect();
        let input_shapes: Vec<&[usize]> = act.iter().map(|(_, s)| *s).collect();
        let params: Vec<RawPtr> = op.params.iter().map(ptr).collect();
        let (out_ptr, stride) = match &op.padded {
            Some(p) => (ptr(&p.buf), p.required_stride),
            None => (
                ptr(&self.slots[node.slot]),
                op.runtime_op.forward_output_row_stride(&node.shape),
            ),
        };
        let launches = op.runtime_op.n_launches();
        for li in 0..launches {
            let mut args = PackedArgs::default();
            let grid = if launches == 1 {
                op.runtime_op.pack_args(
                    &act,
                    &params,
                    out_ptr,
                    &node.shape,
                    stride as i32,
                    &mut args,
                );
                op.runtime_op.grid(&node.shape)
            } else {
                op.runtime_op.pack_args_for_launch(
                    li,
                    &act,
                    &params,
                    out_ptr,
                    &node.shape,
                    stride as i32,
                    &mut args,
                );
                op.runtime_op
                    .grid_for_launch(li, &input_shapes, &node.shape)
            };
            let cfg = runtime::launch_config_grid(grid, &op.program);
            device.launch(&op.program, &cfg, args).with_context(|| {
                format!(
                    "launch {li}/{launches} failed at node {idx} ({}, shape {:?})",
                    self.names.get(&idx).map_or("unnamed", |s| s.as_str()),
                    node.shape
                )
            })?;
        }
        if let Some(p) = &op.padded {
            let mut padded = vec![0.0f32; p.n_rows * p.required_stride];
            p.buf.to_host(&mut padded)?;
            let tight: Vec<f32> = padded
                .chunks(p.required_stride)
                .flat_map(|row| &row[..p.natural_stride])
                .copied()
                .collect();
            let slot = node.slot;
            self.slots[slot].to_device(&tight)?;
        }
        Ok(())
    }
}
