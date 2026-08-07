# #651c — Can prefill run as a CPU+GPU pipeline split? Feasibility verdict

Target shape (user, 2026-08-07): the laptop has **one CPU and one GPU sharing
RAM**. Prefill is compute-bound, so use **both as an uneven PP layer split**
with most layers on the GPU; **decode on the GPU only**; speculation on.

Verdict: **not achievable on this tree.** Four independent blockers, three of
them requiring real code work. Below: what blocks it, what the CPU stage could
be worth even if unblocked (a hard ceiling that does not need measuring), and
what IS achievable today.

Status: **[DESK-PROVEN by code reading]** at `6c1e5cafb7` plus arithmetic from
the checkpoint. Nothing measured on hardware.

---

## 1. The ceiling: how much could a CPU prefill stage ever be worth?

Worth settling first, because it bounds how much the blockers are worth fixing.

In a balanced 2-stage pipeline the stage times are equal, so with
`R = R_gpu / R_cpu` (prefill throughput ratio) the balanced split is
`L_cpu = N/(R+1)` layers and the **best possible** prefill throughput gain over
GPU-alone is:

```
speedup_max = 1 + 1/R          ( = 1 + R_cpu/R_gpu )
```

i.e. exactly the CPU's share of total system compute. No layer assignment can
beat it.

| R = GPU/CPU prefill | balanced CPU layers (of 40) | **max prefill speedup** | dense bf16 RAM those layers cost |
|---:|---:|---:|---:|
| 5x  | 6.7 | **+20 %** | 10.7 GiB |
| 10x | 3.6 | **+10 %** | 5.7 GiB |
| 20x | 1.9 | **+5 %**  | 3.0 GiB |
| 50x | 0.8 | **+2 %**  | 1.3 GiB |

The RAM column is the killer and is explained in §2.1: a CPU stage cannot hold
GGUF K-quants, so its layers must be materialized **dense**, at **3.17x** the
quantized size (506.2 MiB -> 1604.2 MiB per layer, measured from the file).

So on a shared-RAM machine the trade is: **spend several GiB of the same RAM
pool the GPU is competing for, to gain 2-10 % of prefill** — and only when
several chunked-prefill chunks or concurrent requests are in flight, because PP
pipelines microbatches and does **not** speed up a single prefill (stages run in
sequence; single-request latency is the *sum* of stage times, minimized by
putting every layer on the fastest device).

A 35B-A3B MoE activates ~3B params per token, which favours the CPU more than a
dense 35B would, so `R` may land at the friendlier end. But even `R = 5` — a
very generous assumption for a laptop CPU against any discrete-class GPU — caps
the gain at +20 % while costing 10.7 GiB of dense weights.

**Conclusion: the CPU prefill stage is not worth building for throughput.** It
would only make sense as a *memory* fallback (letting layers live somewhere when
the GPU cannot hold them) — and for that purpose it is strictly worse than
weight offload, which keeps compute on the GPU (§4).

---

## 1.5 Shared RAM: the prefill->decode reshard is logically real, physically free

User, 2026-08-07: *"zwischen prefill und decode läuft logisch ein reshard auf die
gpu, allerdings liegt praktisch sowieso immer alles im gleichen ram."*

This is correct and it **removes one of the three costs** I charged against the
split above. Worth stating precisely, because it changes which objections
survive.

On a discrete-GPU rig the prefill->decode handover is a genuine data movement:
KV rows are filtered by `owned_ordinals` and pushed over RDMA/TCP by mooncake
(`disaggregation/mooncake/conn.py:1502-1558`), and that transfer is a real cost
that has shaped the whole PD design. On a machine where CPU and GPU share one
physical RAM, the same handover is a **layout/ownership reinterpretation over
bytes that never move**. The reshard stays logically necessary — the decode side
wants a different row ownership than the prefill side produced — but its
bandwidth cost approaches zero.

**What this changes:**

- The PP stage-boundary activation transfer (§3, gloo path) and the PD KV
  handover are both effectively free. Any cost model for this hardware that
  charges bus bandwidth for them is wrong.
- It makes the **PD-pair shape genuinely attractive on this hardware** — the
  usual reason to avoid splitting prefill from decode on one machine is the
  handover, and here there isn't one. If the #631a refusal (§2.4) is ever
  lifted, this laptop is an unusually favourable place to run that shape.
- It probably dissolves the "GPU-addressable memory" question in §4: with true
  unified memory there is no carve-out to overflow, so the 21,784 MiB of weights
  fit whenever system RAM is adequate, and no offload is needed.
- There is a real optimization latent here that nobody has built: on unified
  memory the PD KV transfer could be **zero-copy** (hand over ownership, do not
  memcpy). Today the code copies unconditionally. Out of scope for #651, but
  worth its own ticket if this hardware becomes a target.

**What it does NOT change** — the three objections that survive, none of which
are about data movement:

1. **The compute ceiling `1 + 1/R` (§1) is untouched.** It is a statement about
   FLOPs, not bytes. If the CPU is 20x slower at prefill, the maximum gain is
   +5 % even with a perfectly free handover.
2. **The 3.17x dequantization penalty (§2.1) is untouched, and shared RAM makes
   it worse, not better.** The CPU stage's layers must be materialized dense
   because there is no CPU K-quant kernel — and on a shared-RAM machine those
   extra GiB come out of the *same* pool the GPU is drawing from. On a discrete
   rig, host RAM spent on a CPU stage at least would not compete with VRAM; here
   it does, directly.
3. **The two code blockers (§2.2 per-rank device, §2.4 PP+spec refusal) are
   unaffected** — they are refusals in the argument handling, not costs.

So the shared-RAM property is a genuine and favourable fact about this machine.
It improves the case for the *phase-split* shape once it is buildable. It does
not rescue the *CPU-as-compute-stage* idea, which is capped by arithmetic (§1)
and taxed by the missing CPU kernel (§2.1).

---

## 2. The blockers

### 2.1 GGUF K-quant has no CPU kernel — fatal, and it fails late

`layers/quantization/gguf.py:34-79` imports the ggml ops only under `_is_cuda`
(from `sgl_kernel`), `_is_musa`, or `_is_npu`. The ops themselves are registered
for the CUDA dispatch key only — `sgl-kernel/csrc/common_extension.cc:431-441`:

```cpp
m.impl("ggml_dequantize",     torch::kCUDA, &ggml_dequantize);
m.impl("ggml_mul_mat_vec_a8", torch::kCUDA, &ggml_mul_mat_vec_a8);
m.impl("ggml_mul_mat_a8",     torch::kCUDA, &ggml_mul_mat_a8);
```

`get_quant_method` (`gguf.py:165-192`) has an `_is_npu` branch and **no `_is_cpu`
branch**, so a CPU stage still gets `GGUFLinearMethod`, whose `apply` calls
`fused_mul_mat_gguf` -> `ggml_mul_mat_a8`.

**A trap worth naming**: `_is_cuda` is a *build/platform* probe
(`torch.cuda.is_available() and torch.version.cuda is not None`), **not**
`server_args.device`. On a laptop that has CUDA, `_is_cuda` is True even in a
process launched `--device cpu`. So the "Only CUDA, MUSA and NPU support GGUF
quantization" warning at `gguf.py:79` **never fires**, the CUDA kernels import
cleanly, and the failure arrives later as a dispatcher
`NotImplementedError: no dispatchable fallback` on the first CPU matmul. Do not
read the absence of that warning as support.

The only CPU-adjacent route in the tree is the Ascend helper
`ggml_dequantize_ascend` (`gguf.py:1557-1581`), which dequantizes with the numpy
reference implementation at load time. Generalizing it to CPU is the realistic
path, and it costs the **3.17x expansion** above — which removes the reason to
run a quantized checkpoint on that stage at all.

### 2.2 Device type is global, not per rank — blocking

`--device` is one string for the whole server (`server_args.py:1623-1626`); the
scheduler processes all receive the same `server_args`, and the only per-rank
value is an integer **GPU index** (`entrypoints/engine.py:698-748`,
`server_args.gpu_id_for_rank` at `:8794-8816`, `model_runner.py:494`).
`--rank-gpu-id` takes CUDA ordinals only, and PP stages must occupy **disjoint
GPU groups** (`server_args.py:9066-9098`).

Worse, the collective layer picks its device from a platform probe rather than
from `server_args.device` (`distributed/parallel_state.py:656-668`:
`if is_cuda_alike(): self.device = torch.device(f"cuda:{device_id}")`), so on a
CUDA machine every rank's `GroupCoordinator` is CUDA regardless.

`--device cpu` therefore puts **both** stages on CPU. Making one stage CPU and
one GPU needs a new per-rank device vector threaded through
`server_args.device` -> `ModelRunner.device` -> `GroupCoordinator.device` ->
`get_default_distributed_backend` (`parallel_state.py:2761`).

### 2.3 PP forbids speculation in one server — blocking for the "+ spec" half

`server_args.py:16264-16269`:

```python
if self.pp_size > 1:
    assert self.disable_overlap_schedule and self.speculative_algorithm is None, (
        "Pipeline parallelism is not compatible with overlap schedule, speculative decoding"
    )
```

`docs/dev/DESIGN_625.md:81-91` names this **B1**: a server booted `pp_size > 1`
has no MTP/NEXTN decode, and "the target picture wants TP+spec decode AND a PP
prefill in ONE server, which this assert forbids."

### 2.4 The two-process escape also refuses speculation — blocking

The natural workaround is a PD pair (prefill arm `pp_size=2`, decode arm
`pp_size=1`), and that pairing *is* explicitly supported
(`disaggregation/common/conn.py:574-582`; geometry is deliberately excluded from
the handshake at `:464-501`). **But both PD arms refuse speculation** —
`arg_groups/pd_disaggregation_hook.py:194-229` raises for any
`--speculative-algorithm` in disaggregation mode, and the only env escape
(`SGLANG_PD_AUTO_DISABLE_SPEC=1`) *disables* spec rather than enabling it.

`docs/dev/DESIGN_631b_draft_kv_wiring.md:7` — "Status: specification only.
Nothing here is built. The #631a refusal stays in force."

**So both routes to "PP prefill + speculative decode" are closed today.** That is
precisely the gap the sibling Route A (#631) strand exists to open; it is
unbuilt, and it is a prerequisite for this half of the #651 goal — on any
hardware, not just the laptop.

---

## 3. What IS built and good

- **Uneven PP layer split is merged, tested and documented.** `--pp-layer-ratio`
  (explicit counts, `server_args.py:1435-1454`), `--pp-stage-ratio` (relative
  capability scores -> `derive_pp_layer_split`, full-attention-aware for hybrids,
  `distributed/utils.py:1481-1560`), and `SGLANG_PP_LAYER_PARTITION`
  (`distributed/utils.py:1442-1478`). The runbook has worked examples
  (`--pp-layer-ratio 44,20` for a 5090+3080). So the *weighting idea itself* is
  directly expressible — it just cannot address a CPU stage.
  The weights are conveniently uniform here (506.2 MiB mean; GDN 506.1, attention
  506.6), so a layer-count ratio is also a memory ratio.
- **PP p2p has a working gloo/CPU path** — `parallel_state.py:2334-2364`,
  `comm_group = metadata_group if tensor.is_cpu else group`, with a gloo
  `cpu_group` created unconditionally for every group (`:701-717`). So the
  boundary transfer is not the blocker. (Caveat: the receiver allocates on the
  *sender's* device type, `:238-246` and `:2395`, so a GPU stage receiving from a
  CPU stage gets a CPU tensor and nothing inserts the `.to(device)`.)

### Two gotchas that will bite whoever boots PP here

- **`--pp-layer-ratio` must sum to the backbone depth (40), not the GGUF
  `block_count` (41).** The extra block is the MTP/NEXTN draft.
- **`max_total_num_tokens` is min-reduced across the world group**, so the
  tightest stage caps capacity for every stage (DESIGN_625 B5).
- **Keep hierarchical/disk HiCache OFF on a PP arm** — #630 is a silent wedge at
  warmup, health 503 forever. And ensure the tree carries the #633 PP
  weight-update-group deadlock fix (upstream sglang#33934).

---

## 4. The honest restatement of the goal

On **one GPU + one CPU sharing RAM**, with this tree:

- **TP is impossible** — TP needs ≥2 GPUs.
- **PP is impossible** — the only candidate second stage is the CPU, blocked by
  §2.1 and §2.2; and even unblocked it is capped at +2-20 % prefill (§1) while
  costing several GiB of the shared pool.
- **Speculation is achievable, but only WITHOUT PP** (§2.3, §2.4).

**So the achievable bring-up is: GGUF Q4 35B, TP=1, PP=1, GPU-only, with NEXTN
speculation.** That is stages a and b of `docs/dev/651/boot.sh` and it is what
the laptop side should target.

Whether it fits is one number — how much of the shared RAM the GPU can address:

| GPU-addressable memory | Verdict |
|---|---|
| >= ~24 GiB | Fits directly. 21,784 MiB weights + KV (20.00 KiB/token fp16, 10.00 fp8) + GDN state (61.9 MiB/sequence fp32, 31.9 bf16). Drop the 818 MiB vision tower (#651b) for headroom. |
| ~16-23 GiB | Needs weight offload. Try `--cpu-offload-gb` (`server_args.py:4273`, `utils/offloader.py:98-140`) — it parks parameters in host RAM and copies them back per forward, so **compute stays on the GPU**. On a shared-RAM machine that copy is within one physical pool, which is a genuinely favourable property here. **Must be tested against GGUF**: the MoE expert-offload family is walled off for GGUF by #123 (`GGUFUninitializedParameter` only materializes in loader postprocess), and whether the generic offloader composes with GGUF params is unverified. |
| < ~16 GiB | 86 % of the model is routed experts (18,726 of 21,784 MiB), so anything shedding more than ~3 GiB must shed experts — the #123 wall. That is a rebuild, not a flag. |

### If the CPU is to be used at all, use it as a memory tier, not a compute stage

`--cpu-offload-gb` gives the CPU's RAM to the model while compute stays on the
GPU. That captures the real benefit of shared RAM without paying the 3.17x
dequantization penalty, without the per-rank-device work, and without losing
speculation to the PP assert. It is strictly better than a CPU pipeline stage
for this hardware.

---

## 5. What would have to be built for the original shape

Ordered by size, for whoever prices it later:

1. **CPU GGUF K-quant kernels** (§2.1) — either vendor llama.cpp-style CPU
   kernels into `sgl-kernel`, or generalize the Ascend dequant-at-load path and
   accept 3.17x memory on the CPU stage. Largest item by far.
2. **Per-rank device type** (§2.2) — a `--rank-device` vector threaded through
   the model runner, the group coordinator and backend selection.
3. **Lift the PP-vs-speculation refusal** (§2.3/§2.4) — this is Route A (#631),
   already an open workstream; `DESIGN_631b` is specification-only.
4. **PP boundary device transfer** (small) — insert the `.to(device)` the PP
   receive path does not do.
5. **CPU graph runner asserts `pp_size == 1`** (`cpu_graph_runner.py:609-610`) —
   the CPU stage would have to run eager.

Given §1 caps the payoff at +2-20 % prefill, items 1, 2 and 5 are hard to
justify on throughput grounds alone. Item 3 is worth doing regardless, because
it blocks PP+spec on **every** machine, not just this one.
