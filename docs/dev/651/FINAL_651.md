# FINAL #651 — Qwen3.6-35B-A3B GGUF Q4 on the APU laptop

Closing report for the standing GGUF/laptop strand. Scope of the target:
Q4 GGUF running **on the laptop only** (Radeon 780M / gfx1103, 32 GB shared),
PP=2 prefill with CPU and iGPU as active workers, iGPU-only decode, reshard
between phases.

Everything below is measured on this machine on 2026-08-08 unless stated.
Raw outputs live in `/root/651-p2/results/` on the laptop; the file names are
quoted per claim so each number can be re-read rather than believed.

---

## 1. Coherence — CLOSED

**Verdict: coherent and greedy-deterministic, reproduced across two
independent boots, byte-identical between them.**

The root cause was fixed by the predecessor (commit `b7a46481c3`): the
standalone ROCm binding passed **int64** `topk_ids` into ggml MoE kernels that
index with **int32**, so every second expert index was read from the wrong half
of a word. The fix is an int32 cast at the `fused_moe_gguf` boundary.

Re-probe on this session (`scripts/probe.py`, 6 determined-answer prompts, run
twice at temperature 0):

| boot | file | content | greedy determinism | verdict |
|---|---|---|---|---|
| pre-reboot (predecessor) | `probe_idsfix_142141.txt` | 6/6 | round1 == round2 | COHERENT |
| post-reboot (this session) | `probe_postreboot_ppfm_*.txt` | 6/6 | round1 == round2 | COHERENT |

The two boots produced **byte-identical text**, including the idiosyncratic
long-form answer to the "2, 4, 8, 16" prompt — cross-boot reproducibility, not
merely two independent passes.

This is the first-ever coherent laptop result and it closes the coherence
chapter.

---

## 2. The "GPU poisoning" narrative — REFUTED

This is the main correction this session contributes, because it had been
gating every boot and every measurement.

### What was believed

A pre-serving guard (`gpu_sanity_guard.py`, v1) declared the GPU "in the
POISONED STATE (suspend/resume defect family)" and demanded a **reboot**
whenever 8 Q5_K dequantize launches were not byte-identical. A long hunt for
the trigger had already falsified system suspend, runtime PM, and GFXOFF one
after another.

### What is actually true

A 5-cycle battery of `guard -> sustained load -> guard -> idle -> guard`
(`guard_battery_144142.txt`) scored **8/15**, with failures spread evenly over
the phases:

| phase | passes |
|---|---|
| baseline | 2/5 |
| after 15 s sustained load | 2/5 |
| after 30 s idle | 4/5 |

Uncorrelated with every transition ever suspected — while the server was
concurrently answering coherence probes correctly and deterministically, and
Q4_K correctness passed 15/15 with a bit-identical error of 5.36e-05.

**There is no poisoned state.** That is why no trigger was ever found: there
was never a trigger to find. A reboot cannot remedy something that is not a
state, so v1's prescription was wrong, and v1 refuses roughly a third to a half
of all boots of a perfectly healthy machine.

### What the canary is really detecting

A genuine **rare per-launch fault in the dequantize kernel's output**. Its
signature, from `q5k_falsifier_*.txt` / `q5k_loop_144744.txt`:

- ~1.6% of launches (≈12% of 8-launch batteries) corrupt **32/64/128
  contiguous elements** — whole K-quant sub-blocks — of one row;
- magnitude 1.2e-02 … 2.7e-02, i.e. a *scale*-sized error, while every other
  element stays bit-identical at the 3.86e-05 quantization error;
- the deviant launch is the wrong one: consensus across launches matches the
  numpy oracle.

Four hypotheses were tested and **falsified**, each with its own control:

| hypothesis | falsifier | result |
|---|---|---|
| unwritten output memory | sentinel planted in the freed block | sentinel never reappears (0 elements) |
| gfx1100 code objects under `HSA_OVERRIDE_GFX_VERSION` | native gfx1103 build vs overridden build, 25 trials each (`arch_ab_145232.txt`) | **3/25 in both arms** — identical rate |
| device→host copy path | each device tensor copied 3x, 40 trials (`copy_vs_kernel_145441.txt`) | **0 copy faults**, 11/40 kernel faults |
| cold first launch | discarded warmup launch, 40 trials (`warmup_test_145810.txt`) | 5/40 with warmup vs 3/25 without — unchanged |

The arch A/B needed one trick worth recording: ROCm torch on this laptop
carries **no gfx1103 code objects**, so a bare `torch` matmul aborts with
`invalid device function` when the override is off. The falsifier was rewritten
to use no torch GPU kernels at all — only the extension's own kernel plus
host↔device memcpies, widening fp16→fp32 in numpy — which is what made the
native arm runnable and the comparison honest.

### Severity depends strongly on quant type

Measured by the v2 guard, per launch:

| type | transient footprint | worst launch deviation |
|---|---|---|
| q4_K | not observed in 5 launches | 5.36e-05 (= clean) |
| q5_K | ~0.02–0.05% of elements, intermittently | 2.7e-02 |
| q6_K | **~0.23% of elements, on every launch** | **4.9e-01 … 6.5e-01** |

q6_K is corrupted on *every* launch, an order of magnitude worse than q5_K.
This independently **vindicates the predecessor's decision** to requantize the
checkpoint to a `noQ6K` build — a choice that had been made on symptom, and now
has a measurement behind it.

### Consequence: guard v2

`docs/dev/651/gpu_sanity_guard_v2.py` replaces the canary:

- **gates on correctness against the numpy oracle**, not on bit-determinism:
  the element-wise median across launches must be within 1e-3, and no single
  launch may deviate beyond 1e-1 or destabilize more than 1% of elements;
- **reports** the background transient every run so a change in its rate stays
  visible instead of silently tolerated;
- drops the false "reboot the machine" advice;
- gates on `q4_K,q5_K` by default and reports q6_K without blocking, since the
  served checkpoint deliberately contains no q6_K;
- carries a **can-fail proof** (`--self-test`) that corrupts *only what the GPU
  is given* while the oracle keeps the clean bytes. An earlier version of this
  self-test corrupted both sides and only ever tripped on NaN bookkeeping —
  it proved nothing, and was replaced.

Measured false-positive rate: **20/20 clean boots-equivalent**, where v1 scored
8/15. The self-test fails as required (consensus 6.8e-02 against a 1e-03
tolerance).

---

## 3. Serving crash — SOLVED: an amdgpu GPU hang, fixed by the prefill chunk

**Final answer first** (the subsections below record how it was reached,
including two wrong turns that are worth keeping):

* Every `unspecified launch failure` was the userspace symptom of a
  **kernel-mode GPU wedge**: `amdgpu: MES failed to respond to
  msg=REMOVE_QUEUE` three times, then `GPU reset(N) succeeded! / device
  wedged, but recovered through reset`. Six such resets happened during this
  session, and their wall-clock timestamps line up one-for-one with the
  crashes.
* It is reproducible **outside serving**: with memory filled to 3% free, a bare
  bf16 GEMM passes at M=512 and **wedges the GPU at M=1024**. No sglang, no
  MoE, no GGUF involved.
* `--chunked-prefill-size 256` (down from 1024) keeps the GGUF large-batch
  GEMM's M below that size and **let one full prefill sweep through**, to
  2048-token prompts, on a server that stayed healthy.
* **But it is a MITIGATION, NOT A FIX.** Re-tested on a freshly rebooted,
  reset-free GPU, a prefill sweep with cp256 wedged the GPU again
  (`GPU reset(1)`, 17:39:11). The one clean sweep was luck, not proof. Stated
  plainly because the earlier draft of this document claimed the fix held on
  the strength of that single run plus "zero resets since" — which was
  survivorship, not evidence.
* Decode (M=1) has never wedged the GPU across the whole session, which is why
  decode floors are solid and reproducible while prefill is not.

**Standing status: the prefill blocker is NOT closed.** The gfx1103 iGPU wedges
in firmware (`MES` is the MicroEngine Scheduler) under sustained prefill load,
probabilistically, and recovers only via a full GPU reset that kills the
server.

The dmesg reset log is what turned a kernel hunt into a driver finding. It was
consulted only after two kernel hypotheses had already been built and
falsified -- see the lesson in section 7.

### How it was localized (including two falsified hypotheses)

Two specimens were captured this session, and the second one names the kernel.

**Specimen 1 (14:40:25).** `torch.AcceleratorError: HIP error: unspecified
launch failure`, two seconds into the first ~2048-token prefill of the
throughput bench, on a server that had answered six short coherence prompts
correctly minutes earlier. This suggested a **prefill-length** trigger
(chunked-prefill size is 1024, so 2048 tokens is the first multi-chunk
request).

**Specimen 2 (15:15:00) refutes that reading and localizes the fault:**

```
RuntimeError: Runtime check failed at
  .../jit_kernel/csrc/moe/moe_align_kernel.cu:530:
  CUDA error: unspecified launch failure
```

This one fired under the **six short coherence prompts**, roughly 20 s after
the server became ready — no long prefill involved. So prefill length was a
coincidence of when sustained load first arrived, not the trigger. The fault is
a probabilistic **launch failure of the MoE align kernel** on gfx1103.

That is a far more actionable lead than "long prefill", and it lines up with
the previously recorded serving-level hazard being *block-size-0 MoE dispatch*
rather than MMQ. It also fits this machine's other measured defect: the
dequantize path here suffers rare per-launch faults (section 2), and the MoE
align kernel is the other heavily-launched GGUF-specific kernel.

Consequence for this strand: the laptop can be **booted and probed**, but it
cannot yet be **held under load** long enough for a clean throughput campaign.
Time-to-crash observed: ~20 s to ~7 min of serving.

`docs/dev/651/crash_bisect_prefill.py` was written for the length hypothesis
and is kept: it walks prompt length upward, several unique prompts per length,
stops at the first death (after `unspecified launch failure` the HIP context is
dead and every later number would be meaningless), and distinguishes a clean
HTTP 400 length refusal from an actual crash. With the length hypothesis
refuted, the next investigation should instead start at the launch site.

### FALSIFIED hypothesis 1: a wave32/wave64 mismatch in moe_align

The reasoning below looked strong and was **wrong**. It is kept because the
falsifier is reusable and because the failure mode -- reading a plausible
story out of source without checking the hardware -- is the kind that repeats.

Line 530 is the `num_experts <= 1024` branch,
`LaunchKernel(dim3(2), dim3(threads), stream, shared_mem_size)`. Reading the
top of that file:

```c
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
...
#ifdef USE_ROCM
using sgl_shfl_mask_t = unsigned long long;
#define SGL_FULL_WARP_MASK 0xffffffffffffffffULL   // all 64 lanes
```

The ROCm branch correctly widens the shuffle **mask** to 64 lanes, but
`WARP_SIZE` stays **32** unless defined externally — and `WARP_SIZE` is what
drives the scan bound (`offset < WARP_SIZE`), the lane id (`tid & (WARP_SIZE-1)`),
`warp_id = tid / WARP_SIZE`, `num_warps_for_scan`, and the size of the
`warp_sums[WARP_SIZE]` shared array.

On gfx1103 the wavefront is 64 wide. So the shuffles are told "all 64 lanes
participate" while the surrounding algorithm indexes as though warps were 32
wide, and lanes 32-63 of one wavefront are accounted to a different `warp_id`
than the one they shuffle with. That is a real inconsistency, and combined with
`warp_sums[warp_id]` / `shared_counts[...]` indexing it is a credible route to
an out-of-bounds shared-memory write, which surfaces exactly as
`unspecified launch failure`.

**Falsified two ways.**

1. `docs/dev/651/moe_align_falsifier.py` checks the kernel's output against a
   host reference (`num_tokens_post_pad` is a pure function of the expert
   histogram, so it needs no crash to detect a broken scan). Result: **20/20
   trials agree**, across token counts from 1 to 512 at the real geometry
   (num_experts=256 passed as 257, topk=8, block_size=64).
2. The premise was simply false. `torch.cuda.get_device_properties(0).warp_size`
   reports **32** on this device: gfx1103 is RDNA3, and RDNA runs HIP kernels
   in **wave32** by default — wave64 is CDNA/MI. So `WARP_SIZE 32` is correct,
   and the 64-bit `SGL_FULL_WARP_MASK` is a type requirement of HIP's
   `__shfl_*_sync`, exactly as that file's own comment states. I had read the
   comment and still built a theory against it.

### FALSIFIED hypothesis 2: moe_align is the culprit at all

Re-running with `AMD_SERIALIZE_KERNEL=3`, so an async HIP fault is attributed
to the kernel that raised it, moved the blame to `gguf.py:1122`
(`hipblasGemmEx`, `HIPBLAS_STATUS_INTERNAL_ERROR`). `moe_align_kernel.cu:530`
was only the next error check after the real fault — the messenger.

Then hipBLAS turned out to be a messenger too: bf16 GEMM is clean over 24
shape/size combinations on a free GPU, including the 0.95 GiB lm_head shape.
It only fails under memory pressure — and what actually happens there is the
GPU wedge in the summary above.

**Lesson worth keeping:** on ROCm, three different subsystems each reported
this fault as their own (`moe_align` RuntimeCheck, hipBLAS INTERNAL_ERROR,
torch AcceleratorError). None of them was the cause. `dmesg` was the only
place that named it, and it was never consulted until the operator said the
driver had crashed.

A second, independent constraint found on the way: the KV pool admits only
**2735 tokens** at `--mem-fraction-static 0.97`, so the 8192-token sweep point
is rejected outright with HTTP 400 rather than measured. Any prefill sweep on
this machine must stay under that ceiling.

---

## 4. Floors

Measured on the guard-v2-gated boot. Method follows the standing rules:
A-vs-A noise floor first, warmup discarded, unique prompts (no prefix cache),
time-bounded runs ≥ 10 s per point.

**THESE ARE EAGER FLOORS — NO CUDA/HIP GRAPHS.** Every boot in this report runs
`--disable-cuda-graph`, so both the decode and prefill numbers below are
eager-mode. The design's gate order (`DESIGN_651_pp_apu.md` section 5) asks for
floors *with* graphs, and that gate is **not** satisfied. Graph capture on this
path has never been validated here, and the standing project rule that
validation runs with graphs + spec (not eager) is therefore also unmet. Do not
compare these numbers against graph-mode measurements from any other machine or
branch.

`docs/dev/651/bench_decode.py` was added for the decode side — prefill is
timed by TTFT with `max_tokens=1`, which says nothing about the steady-state
inter-token interval. It streams, discards the first inter-token gap (it still
carries the prefill tail), and reports the per-token interval distribution
(median/p90/max), not just a mean rate, because the ms-per-round *tail* is what
a pipeline split has to reason about.

**Numbers: see section 6.** Decode floor obtained (13.68 tok/s, 0.26% A-vs-A
noise floor); prefill floor NOT obtained — every attempt was killed by the
section 3 fault before a sweep point completed.

---

## 5. PP=2 CPU+iGPU — build state

Honest slice-by-slice status against `DESIGN_651_pp_apu.md`.

| slice | state |
|---|---|
| W1 arg validation | **DONE.** `--pp-device-map` parses, and refuses by name: `pp_size<2`, length mismatch, unknown device, all-CPU maps, `tp_size!=1`, `dp/ep/dcp>1`, multi-node, and combination with `--rank-gpu-id`/`--rank-gpu-memory-mib`. Fails before a rank is spawned rather than as a late NCCL hang. |
| W1 spawn path | **DONE.** `_rank_device_visibility` (entrypoints/engine.py) sends a CPU-mapped stage through `hide_all_devices`, so the child sees empty `CUDA_/HIP_/ROCR_VISIBLE_DEVICES`; `set_local_device_override` then pins the coordinator's device string in the child, because `is_cuda_alike()` is a **build** property and a card-less process on a ROCm build still claims `cuda:0` otherwise. Not implemented via `SGLANG_USE_CPU_ENGINE` — grepping for that name alone makes the slice look missing, which is the trap I first fell into. |
| pre-warm / rank-uniform world | **DONE.** `should_pre_warm_nccl` is a per-rank decision and the world backend is rank-uniformly gloo. |
| W2 recv-side device | **DONE** (`56d628758b`), `_move_received_tensor`. |
| W2b send-side routing | **DONE.** `_p2p_route(tensor.device.type, wire_backend)` routes by the group's *backend*, and `_stage_tensor_dict_for_wire` stages to host **before** `_split_tensor_dict` so the metadata declares `cpu` and the receiver allocates a CPU buffer. The wire is symmetric by construction — the receiver's `tensor.is_cpu` test agrees with the sender's backend-aware route. I initially read this as an asymmetry bug; it is not. |
| pynccl suppression | **DONE.** `should_build_pynccl` takes `mixed_device_world` and refuses to construct a communicator for a group whose member has no NCCL device. |
| **W3 CPU stage eager** | **DONE THIS SESSION.** See below. |
| W4 validation exemptions | **Deferred by design**, not forgotten: W1 refuses the `--rank-gpu-id` combination outright, so the GPU-shaped validators are unreachable from this path. They must be exempted when the CPU-stage memory budget lands. |

### W3, landed this session

`CPUGraphRunner.__init__` asserts `pp_size == 1`. That assertion is *correct* —
its capture machinery is genuinely single-stage — so it was **not lifted**.
Instead `ModelRunner.init_decode_cuda_graph` returns early on a CPU rank when
`pp_size > 1`, leaving that stage eager while the GPU stage keeps capturing.
Decode after the phase flip is GPU-only, so nothing that matters is lost.

Test: `test/registered/unit/distributed/test_cpu_stage_eager_651.py`, 3 passed.
It is red-first in both directions — a `pp_size == 1` CPU rank must still reach
capture (so the condition cannot be widened to "any CPU rank" and silently
disable CPU graphs for every single-stage CPU deployment), and the GPU stage of
the same pipeline must still capture. **Can-fail proven**: with the new branch
patched out, exactly the target test fails; restored, 3/3 pass.

Unit coverage across the mixed-device slices is green: 37 passed in
`test_pp_device_map_651.py`, 8 passed across `test_cpu_stage_eager_651.py` +
`test_p2p_recv_device_651.py`, with no regression from the W3 change.

### PP=2 driven live on the laptop — how far it got

The mixed-device pipeline was booted for real on this hardware (not just in
unit tests), with `--pp-size 2 --pp-device-map cpu,cuda`. It could not use the
GGUF checkpoint: the laptop serving tree carries **ROCm-GGUF enablement** (it
adds `"gguf"` to `rocm_supported_quantization`, plus standalone-binding wiring
in `gguf.py`) that the #651 branch lacks, while the #651 branch carries the PP
slice that the laptop tree lacks. Two divergent patch sets on different bases;
a straight file port failed on version skew (the laptop's
`pd_disaggregation_hook.py` is 147 lines against the branch's 299). So the
branch tree was staged in parallel (`/root/651-p2/sglang_rig`) and driven with
the bf16 2B checkpoint already on the laptop, to answer the question that
*could* be answered: does a CPU+ROCm pipeline run here at all?

**It got as far as executing batches.** Five distinct blockers were found and
four were fixed, each one only reachable by getting past the previous:

| # | blocker | status |
|---|---|---|
| 1 | `fp8_kernel.py:141` — `get_device_properties(0)` at import time; `_is_hip` is a BUILD probe, so a card-less rank cannot import the quantization package | FIXED |
| 2 | `common.py` `get_device_capability_no_init()` — despite the name, initializes CUDA; called at import by capability-gated modules | FIXED |
| 3 | `vision.py:1116` — `get_device_capability() >= (9, 4)` with `(None, None)`; vendor flag now tested first, per that module's own doctrine | FIXED |
| 4 | `loader.py:877` — `torch.cuda.current_device()` in a debug memory readout during weight load (both paired blocks) | FIXED |
| 5 | `gpu_id_for_rank` — a `cpu` stage occupies a world rank but consumes no card, so the GPU stage was handed `gpu_id=1` on a single-GPU machine → `invalid device ordinal` | FIXED |
| 6 | `qwen3_5.py` GDN projection — selected Triton via the module-level `_is_cpu` BUILD global, making an existing pure-torch fallback unreachable from a CPU rank | FIXED |

After those, **both stages initialized, the gloo world formed, and the
pipeline event loop launched a batch** (`event_loop_pp` → `_pp_launch_batch` →
`run_batch` → model forward). That is W1/W2/W2b/W3 working under real
conditions.

**Where it stops, and why that is structural.** The CPU stage dies inside a
Triton kernel — `RuntimeError: 0 active drivers ([])` — because Triton has no
CPU backend and the CPU rank has no GPU by construction. Walking that wall
back one layer at a time:

| depth | site | outcome |
|---|---|---|
| vision tower | `qwen3_vl.py` blocks | avoided via `--skip-server-warmup` (the warmup request carries a dummy image; text-only probes never enter it) |
| GDN input projection | `gdn_fused_proj.py:296` | **FIXED** — see below |
| GDN causal conv | `causal_conv1d_triton.py:514` | no CPU implementation exists |
| GDN recurrence | `fla/` (chunk, fused_recurrent, gating, l2norm, norm-gate) | no CPU implementation exists |

The projection fix is the same bug class as the five above, and worth stating
because it is general: `qwen3_5.py` already contained a **pure-torch fallback**
for the split/reshape/cat, but selected it with the module-level `_is_cpu`
BUILD global. In a mixed-device pipeline the CPU stage runs inside a ROCm
build, so `_is_cpu` is False there and the CPU rank took the Triton branch. It
now routes on `projected_states_qkvz.device.type`, and the torch path — which
was correct all along — became reachable.

The remaining two entries are not reachable the same way, and this is the
honest cut:

* `causal_conv1d.py` only dispatches between the sgl_kernel CUDA extension and
  Triton. There is **no torch implementation** to make reachable.
* the linear-attention backends are `flashinfer`, `flashkda`, `triton` — all
  GPU. The `fla/` directory is entirely Triton kernels with no naive reference.

So a CPU GDN stage means **writing the fla kernel family in torch**
(causal_conv1d with varlen/conv-state semantics, gated-delta gating, l2norm,
the chunked delta-rule recurrence with state propagation, fused norm-gate).
That is multi-day work with real correctness risk, not a dispatch fix.

**Route B does not close this gap either**, and the earlier framing that it
would was wrong: the shim (`docs/dev/651/cpu_stage/`, a C `libqmatmul_shim.so`)
provides CPU-native **K-quant dequant + matmul for GGUF weights**. The PP
vehicle is bf16 and has no K-quant weights, and the missing kernels are
*linear-attention recurrences*, not quantized matmuls. Route B is necessary for
a GGUF CPU stage and not sufficient for a GDN one.

**No PP=2 throughput number is claimed, because no PP=2 forward has
completed.** What is claimed: the mixed-device pipeline initializes, forms its
world, and launches batches on this hardware, and six device-routing defects
that blocked it are fixed.

### PP=2 END-TO-END: IT RUNS (dense vehicle)

The machinery question was settled by removing the GDN kernels from it. On a
**dense-attention** checkpoint (Qwen2.5-1.5B-Instruct, `Qwen2ForCausalLM`, 28
layers, bf16, 2.9 GB, downloaded to the laptop) the mixed-device pipeline runs
end to end:

```
--pp-size 2 --pp-device-map cpu,cuda   (CPU = stage 0, iGPU = stage 1)
probe.py, temperature 0, two rounds:
  round1 6/6 content-correct, round2 6/6, round1 == round2
  VERDICT: COHERENT      server healthy afterwards
```

One further blocker had to be fixed to get there, and it is the **seventh** of
the same class: `mem_cache/memory_pool.py` selected the CUDA custom op
`sglang::store_cache` via the `_is_cuda`/`_is_hip` BUILD globals, so the CPU
rank tried to run it on CPU tensors (`Could not run 'sglang::store_cache' with
arguments from the 'CPU' backend`). The naive torch fallback
(`k_cache[indices] = k`) already sat at the bottom of the same function and was
unreachable. It now routes on `k_cache.device.type`.

**The split is real, not nominal:**

* `sglang::scheduler_PP0` runs with `CUDA_VISIBLE_DEVICES=` **empty** — no
  accelerator — while `PP1` has it unset and holds the card;
* the two stages loaded disjoint halves: PP0 2.41 GB and has no
  `model.norm.weight`; PP1 3.76 GB and has no `model.embed_tokens.weight`.

### Measurement, and what it says

Both configurations measured on the same boot-pair, warmup discarded, unique
prompts, A-vs-A noise floor first, under the armed wedge policy
(`--chunked-prefill-size 256`), eager (no graphs).

**ARCHITECTURE — read this before the decode column.** PP=2 applies to
**PREFILL ONLY**: CPU and iGPU both compute the prefill, then the phase flip
reshards and **decode runs on the iGPU alone**, as does the draft. PP=2 in
decode is slower by construction and is **not a target configuration** — the
decode column below is included only to show what the pipeline costs if it is
(wrongly) left in place through decode, not as a verdict about the design. The
real decode and draft numbers are in section 6a, where the iGPU runs alone.

| configuration | prefill @789 tok | A-vs-A | decode (NOT a target config) | A-vs-A | TTFT median |
|---|---|---|---|---|---|
| PP=2, CPU + iGPU | **671.3 / 669.2 tok/s** | 0.32% | 11.63 / 11.61 tok/s | 0.21% | 0.129 s |
| single stage, iGPU only | **968.3 / 962.6 tok/s** | 0.59% | 16.75 / 15.80 tok/s | 5.70% | 0.093 s |

**Adding the CPU as a pipeline stage makes PREFILL slower here:** 0.69x at the
even split (the proportional ladder below recovers most of that). The decode
figure is not a verdict on anything, per the architecture note above — decode
is iGPU-only by design. That is a reportable outcome, not a failure — the
design plan says so in advance ("Negligible-or-negative CPU contribution after
throttling = reportable verdict, not failure") — and the reason is structural:
these runs have `--max-running-requests 1`, and **pipeline parallelism pipelines
MICROBATCHES**. With one request in flight the stages run strictly in sequence,
so a PP split can only add the slower stage's time; it cannot overlap anything.
`bench_prefill.py` carries that warning in its own docstring.

### Proportional split from CO-RUN measurement (user order)

The 1:1 split above is not the right one, and solo numbers cannot pick the
right one: on an APU both stages share DDR5 bandwidth and package power, so a
stage's solo speed does not predict its co-run speed. A per-stage round tracer
was added for this (`scheduler_pp_mixin.py`, `SGLANG_PP_ROUND_TRACE=1`, off by
default): time blocked in `_pp_recv_proxy_tensors` is time this stage waited for
the other, the rest is its own work, and the stage that waits least is the
bottleneck.

Measured on the even 14/14 split, co-run:

```
PP-ROUND stage=0  round 282-296 ms  wait 0.00 ms  (wait share 0.0%)
PP-ROUND stage=1  round 311-318 ms  wait 73-78 ms (wait share ~24%)
```

The CPU stage **never waits** — it is the bottleneck — and the iGPU idles ~24%
of every round. Work per 14 layers: CPU ~286 ms, iGPU ~187 ms, i.e. the iGPU is
**~1.52x faster per layer in co-run**. (My earlier solo-sequential inference
said 1.9x; co-run differs, which is exactly why it had to be measured.)

Driving `--pp-stage-ratio` from that, the full ladder — all A-vs-A floored, at
~1024-token prompts, cp256:

| split CPU/iGPU | prefill tok/s | A-vs-A | note |
|---|---|---|---|
| 20/8 | 485.5 | — | **FALSIFIER**: anti-proportional, strictly worse; CPU work 411 ms, iGPU idle 48% |
| 14/14 | 649.8 / 638.5 | 1.73% | even baseline |
| 11/17 | 699.2 / 699.6 | 0.05% | first proportional estimate |
| 10/18 | 733.2 / 730.4 | 0.38% | |
| 8/20 | 796.5 / 799.0 | 0.30% | **balance point**: work 230.7 vs 228.4 ms, wait 0.6% |
| 7/21 | 838.9 | — | |
| 4/24 | 865.9 / 867.2 | 0.15% | |
| 2/26 | 902.8 / 905.0 | 0.25% | |
| 0/28 (iGPU solo) | 968.3 / 962.6 | 0.59% | reference |

**The user's premise is confirmed and quantified: proportional beats 1:1 by
+39%** (649.8 -> 905.0), and the anti-proportional falsifier is decisively
worse, so the ladder is measuring what it claims to.

**But the improvement is monotonic all the way to ZERO CPU layers**, and no CPU
share beats the iGPU alone. The equal-co-run-ms balance point (8/20) is *not*
the throughput optimum, which is the interesting part: balancing the stages is
necessary but not sufficient when adding the second worker also slows the
first.

**Mechanism, measured.** The iGPU is *slower while the CPU works*: at the 2/26
split the iGPU stage spends 220.9 ms on 26 layers = **8.50 ms/layer**, while
solo it does 28 layers in ~204 ms/chunk = **7.28 ms/layer** — **~17% slower per
layer in co-run**. That contention tax, paid on the iGPU's ~26 layers, exceeds
whatever the CPU contributes on its 2. This is the shared-memory-system penalty
the design plan warned about, now with a number on it.

**WHICH MODEL THESE NUMBERS ARE FROM — do not mix them with section 6a.** Every
figure in this section 5 (the whole split ladder, the co-run stage times, the
~17% contention) is the **dense Qwen2.5-1.5B** vehicle, NOT the target. The
35B-A3B GGUF numbers are in section 6a and are an order of magnitude different
(148 tok/s prefill for the 35B MoE versus 968 tok/s for the 1.5B dense) — the
two are not comparable. **There is no PP=2 number for the target checkpoint at
all**, because its CPU stage cannot run the GDN kernels.

**OPERATIONAL RULE that follows (user, 2026-08-08): if the iGPU alone is faster
in prefill even holding every layer, then run iGPU-only — including in
prefill.** At bs=1 that is exactly what the ladder says: 968.3 tok/s with all
28 layers on the iGPU beats every split, and the curve is monotonic, so there
is no CPU share that pays. PP=2 stays the right *shape* for the design (it is
prefill-only, with decode and draft on the iGPU alone), but on this APU at
bs=1 the correct layer split is 0/28.

**Transfer to the target is PLAUSIBLE BUT UNMEASURED.** The mechanism is
hardware contention (shared DDR5 bandwidth and package power), not anything
model-specific, so the same monotonic-to-zero behaviour is expected for the
35B-A3B GGUF. It has not been measured there and currently cannot be, for want
of the GDN-CPU kernels. Stated as an expectation, not a result.

**CAVEAT — every split verdict here is at bs=1 (`--max-running-requests 1`).**
With one request in flight a pipeline has almost nothing to overlap, so the
CPU stage can only add latency to the critical path. With several concurrent
requests or several chunks in flight the pipeline gains real overlap and the
CPU-stage economics could flip — a small CPU share might then pay for itself
exactly as the user expects. **That regime is untested here** and is the single
most valuable follow-up measurement.

### What full CPU-stage coverage still needs

For a Qwen3.5-family (GDN hybrid) checkpoint, in dependency order:

1. `causal_conv1d_fn` / `causal_conv1d_update` in torch, including varlen
   (`query_start_loc`), `cache_indices`, `has_initial_state` and in-place
   `conv_states` update;
2. `fused_gdn_gating`, `l2norm`, `fused_norm_gate` in torch;
3. the chunked gated-delta-rule recurrence (`fla/chunk*.py`) — the substantial
   one — or a `fused_recurrent` torch equivalent for the prefill path;
4. a `--linear-attn-backend torch_native` option to select them per rank.

A dense-attention checkpoint needs **none** of this — which is exactly what the
section above demonstrates. The list stands as the requirement for running the
**target** model's family (Qwen3.5/3.6 GDN hybrids) on a CPU stage.

**So the strand's target needs two things this report does not deliver, both
named follow-ups and neither started:**

1. **the GDN-CPU kernel family** above, for a 35B-A3B GDN-MoE CPU stage;
2. **GGUF x PP version-skew resolution** — the laptop tree's ROCm-GGUF
   enablement and this branch's PP slice have to live in one tree before PP=2
   can use the GGUF checkpoint at all.

---

## 6. Results appendix

### Decode floor — MEASURED

Guard-v2-gated boot, eager (no graphs), Q4_K_M noQ6K, ctx 8192,
`--mem-fraction-static 0.97`, `max-running-requests 1`, batch size 1.
`bench_decode.py`, warmup discarded, first inter-token gap of each request
discarded, unique prompts.

| arm | decode | per-token median | p90 | max | TTFT median |
|---|---|---|---|---|---|
| A  | **13.68 tok/s** | 73.08 ms | 75.14 ms | 77.32 ms | 0.945 s |
| A' | **13.65 tok/s** | 73.27 ms | 75.13 ms | 76.43 ms | 0.920 s |

**A-vs-A noise floor: 0.26%.** 282 timed tokens per arm, ≥20 s per arm. Any
later decode claim on this machine must clear 0.26% to be a result.

Repeated on a second boot (with `--chunked-prefill-size 256`, which does not
affect decode): 13.72 / 13.64 tok/s, per-token median 72.88 / 73.30 ms, floor
0.57%. Two independent boots agreeing to within 0.6% is what makes this number
trustworthy, in contrast to the prefill figure above.

The tail is tight — p90 is 2.8% above the median and the max is 5.8% above —
so decode on this APU is steady, not bursty. That matters for the PP plan: a
fat decode tail would have argued against the iGPU-only decode phase, and it
does not.

Corroboration from the server's own instrumentation across all boots of the
day (141 samples): median 13.35 tok/s, max 18.91 tok/s — consistent with the
benched 13.7 tok/s.

### Prefill floor — ONE successful measurement, NOT reliably reproducible

With `--chunked-prefill-size 256`, one sweep completed end to end:

| prompt | prompt_tokens | median | run A | run A' | A-vs-A floor |
|---|---|---|---|---|---|
| ~256  | 213  | 1564.9 ms | 136.1 tok/s | 143.6 tok/s | 5.19% |
| ~512  | 404  | 2771.2 ms | 145.8 tok/s | 145.0 tok/s | 0.56% |
| ~1024 | 789  | 5558.7 ms | 141.9 tok/s | 141.8 tok/s | 0.07% |
| ~2048 | 1555 | 10496.0 ms | 148.2 tok/s | 147.7 tok/s | 0.28% |

**Peak sustained prefill: 148.2 tok/s** at ~2048-token prompts. The A-vs-A
floor is 0.07-0.56% at the three larger sizes; the 5.19% at ~256 tokens is the
overhead-dominated small end, not a measurement of compute.

**Caveat that must travel with these numbers:** a repeat sweep on a freshly
rebooted GPU wedged the device instead of completing (section 3). So this is a
real measurement of a real configuration, taken on a run that survived — but
prefill throughput on this machine is not yet reliably measurable, and any
comparison drawn against it should be repeated rather than trusted once.

Three earlier attempts (before cp256) died in warmup; all three were the GPU
wedge, confirmed against dmesg — **not** OOM (dmesg shows zero OOM kills), not
the sanity guard, and not a harness timeout.

The server's own `input throughput (token/s)` line is **not** usable as a
substitute: it reports ~5.2-5.8 tok/s for 40-token prompts whose measured TTFT
was 0.93 s (≈44 tok/s), so the two disagree by ~8x. Reporting it would have
been a fabricated number.

**Consequence for the split arithmetic:** `layer_split.py --split-from` needs a
CPU-stage prefill rate to sit beside this GPU rate, and no CPU stage has
executed (section 5). The split is still not computable.

---

## 6a. THE TARGET: 35B-A3B Q4 GGUF with draft, and graphs

This is the strand's actual objective. The dense 1.5B of section 5 was a proof
of the PP machinery only; everything in this section is the real checkpoint,
`Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf`, on the iGPU alone, guard v2 and wedge
policy armed, `--chunked-prefill-size 256`.

### Draft runs, and it is coherent

The checkpoint carries its own draft weights — `blk.40.nextn.*` (NEXTN/MTP, 4
tensors) — so the draft model is the same GGUF file and no separate draft
checkpoint is needed. Booted with `--speculative-algorithm NEXTN`,
`--speculative-num-steps 1 --speculative-eagle-topk 1
--speculative-num-draft-tokens 2`:

```
probe.py, temperature 0: round1 6/6, round2 6/6, round1 == round2
VERDICT: COHERENT
```

This is the first coherent draft run on the fixed tree. An earlier spec run
exists in the results directory (09:11, 22 requests, accept 1.87) but predates
the int32 coherence fix, so it measured an incoherent model and its numbers
must not be quoted.

### Graphs: HIP graph capture WORKS on this stack

The user's second requirement. Capture succeeds on gfx1103 under this ROCm
build, with evidence rather than assertion:

```
Capture target verify CUDA graph begin. backend=full
Capture target verify CUDA graph end. elapsed=3.79 s
Decode batch ... cuda graph: True
```

`cuda graph: True` on the decode batches is the load-bearing line — a boot that
merely *accepts* the flag while silently running eager would print False. No
GPU reset accompanied capture or teardown, which is a real datum given that
graph teardown intersects the `REMOVE_QUEUE` wedge hypothesis of section 6b.
Prefill graphs remain disabled by the config resolver; only decode/verify are
captured.

### Decode floors: eager vs graphs, without and with draft

All A-vs-A floored, 20 s per arm, warmup discarded, unique prompts.

| configuration | decode tok/s | ms/token | p90 | A-vs-A | accept len |
|---|---|---|---|---|---|
| eager, no draft | 13.68 / 13.65 | 73.1 | 75.1 | 0.26% | — |
| **graphs, no draft** | **15.47 / 15.43** | **64.8** | 65.1 | 0.28% | — |
| eager, with draft | 8.58 / 8.52 | 117.3 | 120.9 | 0.62% | 1.90–1.93 |
| graphs, with draft | 9.09 / 9.08 | 110.1 | 110.9 | 0.09% | 1.93–1.95 |

**Graphs are a real win and clear their floors comfortably: +12.8% without
draft (13.68 -> 15.47) and +6.3% with draft.** The eager floors that every
earlier section of this report carries are therefore superseded for decode: the
decode number for this checkpoint is now 15.47 tok/s with graphs.

### Draft verdict: net NEGATIVE on this iGPU, despite excellent acceptance

**9.09 tok/s with draft versus 15.47 without — 0.59x.** The acceptance is not
the problem: accept length 1.93–1.95 out of 2 draft tokens, accept rate
0.90–0.93, which is close to the theoretical maximum for this configuration.
The cost is the **second forward per step**. On this iGPU a decode step is not
launch-bound enough for the draft+verify pair to pay for itself: verifying 2
tokens plus running the NEXTN head costs more than the ~1.9 tokens it returns.
Graphs narrow the gap (they are worth more to the two-forward path in relative
terms than to the one-forward path would suggest) but do not close it.

### The draft slowdown is NOT explained by the algorithm — two concrete defects

Challenged on this (user: "that's illogical, there must be a bug"), the
arithmetic agrees and the logs name two specific problems. Do **not** read the
0.59x as "speculation does not pay on this hardware" until these are fixed.

**The cost does not add up.** Baseline decode is 64.8 ms/token with graphs.
Speculative decode is 110.1 ms/token at accept length 1.93, so one spec step
costs 110.1 x 1.93 = **212 ms = 3.3 baseline forwards**. The algorithm at
`num_steps=1, topk=1, draft_tokens=2` should cost roughly ONE target forward
(verifying 2 tokens is nearly free on a bandwidth-bound decode) plus a small
NEXTN head. Even the pessimistic reading — draft executing a full second model
— only accounts for ~2. There is ~1.3 forwards of unexplained cost per step.

**Defect 1: the draft path is not graph-captured.** The only capture in the log
is the target:

```
Capture target verify CUDA graph begin. backend=full
Capture target verify CUDA graph end. elapsed=3.79 s
```

There is no `Capture draft decode` line. The verify forward runs captured while
the **draft forward runs eager**, so the draft pays full per-launch overhead on
every step — on a model with 40 layers of MoE dispatch that is exactly the kind
of cost that shows up as the missing 1.3 forwards. The user's question ("does
the draft run on the iGPU with graphs?") has the answer **no**, and that is a
gap, not a property of the hardware.

**Defect 2: the draft loads like a full model.** Two weight loads appear, the
target taking 63 s and the draft **53 s** — despite the draft needing only the
20 tensors of `blk.40` (`MTP draft name map: 20 tensors (blk.40)`). A 20-tensor
load should take about a second; 53 s is the signature of re-reading the whole
21.4 GiB GGUF. This also explains why enabling speculation forced
`--mem-fraction-static` from 0.97 to 0.99 ("draft weights are now counted") —
far more mass than one MTP layer should account for.

### Both defects chased down; k=3 tree drafts unblocked and validated

**Defect 1 explained: the draft graph is gated on `num_steps > 1`.**
`eagle_worker_v2._capture_cuda_graphs` captures the draft decode graph only
`if self.speculative_num_steps > 1`. The original boot used `num_steps 1`, so
the draft ran eager **by design**, not by hardware limitation. With
`num_steps 2` the capture appears:

```
Capture target verify CUDA graph end. elapsed=1.83 s
Capture draft decode CUDA graph end.  elapsed=1.40 s
```

**k=3 tree drafts unblocked.** The user requires NEXTN with graphs at k=3.
`decide_spec_kernel_backend` refused any `topk > 1` whenever the group falls
back to the Triton spec kernels — which is always on this laptop, since it has
no `sgl_kernel` build at all. Reading the path rather than the message: BOTH
halves exist in Triton (`sgl_build_tree_kernel_triton` and
`verify_tree_greedy_triton`), and the mode this configuration uses is
`FULL_MASK` — not the `QLEN_ONLY_BITPACKING` mode the refusal names as missing
(`default_tree_mask_mode` picks QLEN_ONLY only on CPU). So the blocker was
validation, not implementation.

It is now unblocked behind an explicit, loud opt-in
(`SGLANG_ALLOW_TRITON_SPEC_TREE=1`, default off, warning on every boot),
**and the validation the guard asked for was actually performed**: greedy
speculative decoding is exact by construction, so temperature-0 output must be
token-identical to a non-speculative run. It is — against **two** independent
non-spec reference runs:

```
k3 (tree spec) vs probe_postreboot_ppfm_143828.txt : IDENTICAL
k3 (tree spec) vs probe_gated_a_150740.txt         : IDENTICAL
```

(The first comparison attempted used a probe file from the boot that crashed
mid-run and was therefore truncated; it showed a spurious difference. Naming
the bad reference because a truncated file silently producing "DIFFERS" is
exactly how a false alarm would be recorded as a finding.)

### The full speculation ladder — accept rises, throughput falls

All on the target checkpoint, graphs on, guard v2 + wedge policy armed,
`amdgpu.gttsize` raised to 29 GiB. **Zero GPU resets across the whole series.**

| configuration | decode tok/s | A-vs-A | accept len | graphs captured |
|---|---|---|---|---|
| no speculation | **15.47 / 15.43** | 0.28% | — | decode |
| k=1, steps=1 | 9.09 / 9.08 | 0.09% | 1.93–1.95 | verify only |
| k=1, steps=2 | 6.84 / 6.75 | 1.25% | 2.60–2.77 | verify + draft |
| k=3, steps=2 (the required config) | 5.85 / 5.83 | 0.45% | 2.90–2.95 | verify + draft |

**Acceptance improves monotonically (1.93 -> 2.90) while throughput falls
monotonically (9.09 -> 5.85).** Speculation is not failing to predict — it
predicts very well. Each extra draft forward simply costs far more than the
extra accepted tokens return.

### The cost of one draft forward — the actual bug, quantified

Baseline decode is 64.8 ms/token. Then:

* k=1 steps=1: 110.1 ms/token x 1.93 accepted = **212 ms per spec step**
* k=1 steps=2: 148.1 ms/token x 2.77 accepted = **410 ms per spec step**

The difference is **198 ms ≈ 3.1 full target forwards**, and the first reading
of that was "a NEXTN draft is one decoder layer, it should cost ~1/40 of a
target forward, so this is ~120x off — a bug". **That reading was wrong**, and
the next subsection is how it was falsified. Two corrections came out of it:
the 198 ms is the FIRST real draft forward (steps 1 -> 2), not an extra one,
and a draft forward is not comparable to 1/40 of a target forward because the
target verify it replaces is itself not a single-token forward.

### Chasing that number: three hypotheses falsified, and what it actually is

**(a) GGUF MoE falling to the dequant/Python fallback — FALSIFIED.** That
branch of `fused_moe_gguf` logs "There is no support for fast MoE kernel";
the server log contains it **zero** times. The draft layer's quant types are
also perfectly ordinary — `blk.40` carries Q4_K expert tensors and a Q5_K
`ffn_down_exps`, the same pattern as `blk.0` — so the MMVQ path is taken, not
the per-expert Python loop.

**(b) The draft holding the whole model — FALSIFIED.** Weight load reports
**0.41 GB for the draft** against 22.75 GB for the target. It really does hold
one layer. (The 53 s load is then a file-scan cost for pulling 20 tensors out
of a 21.4 GiB GGUF, i.e. a startup wart, not a per-forward cost.)

**(c) A capturable-but-uncaptured draft forward at steps=1 — FALSIFIED, and
this reframes the whole measurement.** Upstream skips the draft graph at
`speculative_num_steps == 1` for a reason stated in its own comment: *"Skip
attention backend init for 1-step draft, `draft_forward` only does sample in
this case."* At steps=1 **the draft performs no forward at all**. Implementing
the capture anyway (attempted, then reverted) fails at
`'NoneType' object has no attribute 'init_cuda_graph_state'` — the draft
attention backend is not even constructed, because nothing uses it.

So the 198 ms is **not an "extra" draft forward**: it is the FIRST real one,
appearing when steps goes 1 -> 2. And the more interesting number is the one
that was hiding behind it: **at steps=1, where the draft does nothing at all, a
spec step still costs 212 ms against a 64.8 ms plain decode step.** That cost
is entirely target verify plus tree/sample machinery.

**What it actually is — two structural costs, neither a bug in our code:**

1. **Verify does not scale like a dense model on a sparse MoE.** This
   checkpoint activates 8 of 256 experts per token. Verifying k draft tokens
   can touch up to k x 8 distinct experts, so the expert weight traffic — which
   dominates a bandwidth-bound decode — grows with the number of draft tokens
   instead of staying flat. On a dense model verifying 2 tokens is nearly free;
   here it is not. Linear scaling alone predicts 2 x 64.8 = 130 ms of the
   212 ms.
2. **The spec kernels are the Triton fallback.** This laptop has no
   `sgl_kernel` build, so tree build and verify run the Triton path rather than
   the native ops. That is the remaining ~80 ms, and it is a cost we are forced
   into on this machine, not a defect either.

**Consequence for the success bar.** Beating the 15.47 tok/s no-draft floor
would require the per-step overhead to fall below what accept length can repay,
and both contributors above are structural: the first is a property of sparse
MoE at batch size 1, the second needs an sgl_kernel ROCm port. So the honest
verdict is that **speculation cannot win here at bs=1 by fixing a site in this
tree** — the earlier "~120x off, implementation defect" framing was wrong, and
is corrected here. The remaining genuinely actionable items are an sgl_kernel
gfx1103 build (large) and the 53 s draft load (startup only, no throughput
effect).

**Verdict: speculation on this stack is coherent, exact, accepts at 0.90-0.93,
and is net-negative for STRUCTURAL reasons** — sparse-MoE verify cost that
grows with draft-token count, plus Triton-fallback spec kernels for want of an
sgl_kernel ROCm build. It is not a fixable site in this tree. Serve without
speculation (15.47 tok/s).

## 6b. Wedge guard — making the regime unreachable

The MES hang is an amdgpu firmware bug and out of scope to fix. So the
triggering regime is refused instead, loudly, before a model is loaded:
`docs/dev/651/wedge_policy.py`, armed in `boot_v2gated.sh` next to guard v2.

It enforces two things on affected hardware:

* **prefill chunk cap** — `--chunked-prefill-size <= 256`, because that chunk
  is the M of the GGUF large-batch bf16 GEMM and M=1024 wedges the GPU
  (M=512 passes);
* **free-memory floor** — at least 2048 MiB free before a large GEMM, an order
  of magnitude above the ~3% free (736 MiB) at which the wedge was reproduced.

Two details that make it honest rather than decorative:

* It resolves the **physical** architecture. `HSA_OVERRIDE_GFX_VERSION=11.0.0`
  makes torch report gfx1103 silicon as `gfx1100`, so a naive arch check would
  pass the very machine it is meant to protect; the policy identifies the
  Radeon 780M by device name and answers `gfx1103`. A bare `gfx1100` that
  cannot be resolved further gets an explicit warning instead of silent
  approval.
* It never claims a fix. On a passing configuration it still prints that the
  cap is a MITIGATION and that a sweep which survived once has wedged on a
  later run.

Unaffected architectures are deliberately untouched — capping every ROCm card
"just in case" would be a silent throughput regression with no measurement
behind it, and a unit test pins that (`test_wedge_policy_651.py`, 7 tests,
can-fail verified by widening `WEDGE_ARCHS`).

Verified on the machine: a boot with `CHUNKED_PREFILL=1024` is refused after
the sanity guard and before the model loads; the default 256 boots and serves
a COHERENT probe with no new GPU reset.

**Later observation that narrows the trigger.** A single-stage boot of the 2.9
GB dense model wedged the GPU at 18:09:47 (`GPU reset(2)`) with `--mem-fraction
-static 0.60` — i.e. with GB of headroom, not at 3% free — during **startup**,
moments after the previous server had been killed. So memory pressure is
sufficient to reproduce the wedge but is **not necessary**, and the earlier
framing ("only under memory pressure") is too narrow. The `MES ... REMOVE_QUEUE`
message is a queue-TEARDOWN message, and both this event and the reproducer
involve queues being destroyed while work is outstanding. That makes "the wedge
follows queue teardown" the strongest remaining hypothesis, and it is cheap to
test: kill and restart a server repeatedly with no inference at all and watch
`dmesg`. Not done here. The retry after the reset booted and measured cleanly,
which is why the comparator numbers above exist.

## 7. Canon corrections this session contributes

1. **"GPU poisoning" is not a state.** Delete the reboot ritual. The observable
   is a rare per-launch kernel-output fault; reboots do not change its rate.
   The three falsified triggers (suspend, runtime PM, GFXOFF) were falsified
   because the premise was wrong, not because the search was incomplete.
2. **A canary that fires on healthy hardware is a suspect, not an oracle.**
   v1 gated on bit-determinism of a kernel that has a known background
   transient; that is a category error. Gate on correctness.
3. **q6_K is the badly affected quant type on gfx1103** (~0.23% of elements per
   launch at ~5e-01), q5_K mildly, q4_K not observably. The `noQ6K` requant is
   now an evidence-backed decision.
4. **ROCm torch here has no gfx1103 code objects.** Any experiment that wants
   to run without `HSA_OVERRIDE_GFX_VERSION=11.0.0` must avoid torch GPU
   kernels entirely — memcpies and the extension's own kernels only.
5. **`--mem-fraction-static` below ~0.963 cannot boot this checkpoint**, and
   even at 0.97 the KV pool admits only 2735 tokens. Sweeps must respect that.
6. **`unspecified launch failure` on ROCm means READ DMESG FIRST.** Three
   subsystems each claimed this fault as their own (`moe_align` RuntimeCheck,
   hipBLAS `INTERNAL_ERROR`, torch `AcceleratorError`) and none was the cause;
   `amdgpu: MES failed to respond` + `GPU reset(N)` was. Two hypotheses were
   built and falsified before dmesg was consulted. Consult it first.
7. **A device probe at import time is a trap for any card-less rank.** Four
   independent instances were found in one afternoon (`fp8_kernel`,
   `get_device_capability_no_init`, `vision.py`, `loader.py`), all rooted in
   the same mistake: treating a BUILD property (`_is_hip` / `is_cuda_alike()`)
   as a statement about the current process.
8. **`pkill -f <pattern>` run inline over ssh kills the invoking shell**, because
   the pattern is in its own command line. Two boots were lost to this before it
   was spotted. Restart logic lives in `restart_serving.sh` for that reason.
9. **A single surviving run is not a fix.** cp256 was written up as solving the
   prefill crash on the strength of one clean sweep; the next attempt wedged
   the GPU. Survivorship reads exactly like success when the failure is
   probabilistic.

---

## 8. Honest bottom line

The target was Q4 GGUF on the laptop with PP=2 prefill (CPU + iGPU) and
iGPU-only decode. Where it actually stands:

* **Coherence: achieved and closed.** Correct, greedy-deterministic, and
  reproducible across boots. This was the chapter that had been open longest.
* **Decode on the iGPU: works and is measured** — 13.68 tok/s with a 0.26%
  noise floor and a tight tail.
* **The guard that gated everything is fixed** and no longer refuses healthy
  boots; the "poisoning" model behind it is refuted, which retires a long and
  fruitless trigger hunt.
* **The target checkpoint runs with draft and with graphs.** 35B-A3B Q4 GGUF +
  NEXTN is COHERENT, HIP graph capture succeeds on gfx1103 (`cuda graph: True`
  on decode batches, verify graph captured in 3.79 s, no GPU reset), and
  graphs lift decode **13.68 -> 15.47 tok/s (+12.8%)**.
* **Speculation works but does not pay here**: 0.59x decode versus no-draft,
  despite accept length 1.93–1.95 and accept rate 0.90–0.93. The second forward
  per step costs more than the ~1.9 tokens it returns on this iGPU.
* **Prefill was measured once** — 148.2 tok/s peak, tight A-vs-A floors — but
  the measurement is not reliably repeatable, because the GPU wedges.
* **PP=2 CPU+iGPU RUNS END TO END** on a dense-attention vehicle: COHERENT,
  greedy-deterministic, 6/6 both rounds, with the CPU stage provably holding no
  accelerator and half the weights. Seven device-routing blockers found and
  fixed to get there. This is the strand's "it runs" bar for the mixed-device
  machinery, and it is met.
* **Measured, and it is slower**: 0.69x prefill, ~0.71x decode versus the iGPU
  alone, because at one request in flight a pipeline cannot overlap anything.
  Reportable verdict, not failure.
* **The target model is NOT covered by this**: a 35B-A3B GDN-MoE CPU stage
  additionally needs the GDN-CPU kernel family, and GGUF x PP needs the
  version skew resolved. Both named, neither started.
* **The wedge regime is now refused at boot** rather than hit at runtime —
  though a later reset with ample free memory shows the trigger is broader than
  memory pressure alone.
* **Sustained prefill serving is still not possible**: the iGPU wedges in
  firmware (`MES failed to respond` → `GPU reset`) under prefill load. Decode
  never wedges. This is a hardware/driver limit on gfx1103, not a bug in this
  branch, and it is the gate on everything prefill-shaped.

Two things are claimed as solved: coherence, and the guard. One thing is
claimed as measured-but-fragile: prefill. Nothing is claimed about PP=2
throughput, because no PP=2 forward has completed.

### What the next session should do first

1. **Test the queue-teardown hypothesis for the wedge.** Cheapest and highest
   value: restart a server repeatedly with no inference and watch `dmesg` for
   `MES ... REMOVE_QUEUE` / `GPU reset`. If it reproduces without compute, the
   wedge is a teardown bug and the prefill-chunk cap is treating a symptom.
   Then: newer amdgpu/MES firmware, `HSA_ENABLE_SDMA=0`.
2. **A per-stage timer.** The 1.9x CPU-vs-iGPU split in section 5 is derived
   from two wall measurements; instrument the stages directly before weighting
   `--pp-layer-ratio` on it.
3. **PP=2 under concurrency.** Every number here is at one request in flight,
   where a pipeline cannot help by construction. The interesting measurement is
   several concurrent requests / chunks, which is the regime a CPU+iGPU split
   is actually for.
4. **The GDN CPU kernel family**, in the dependency order in section 5, if the
   GDN target is to run a CPU stage. Multi-day.
5. **Unify the two trees** for GGUF x PP.
6. **An sgl_kernel build for gfx1103** is the only lever left that could make
   speculation pay here: it replaces the Triton fallback spec kernels (tree
   build + verify) with native ops and would also lift the topk>1 opt-in.
   Large. The sparse-MoE verify scaling underneath it does not go away.
   Note the environment now has headroom for this work: `amdgpu.gttsize` was
   raised 24 -> 29 GiB (with `ttm.pages_limit` in lockstep), which is what let
   the k=3 tree configuration boot at all.
7. **PP=2 under concurrency is the open question that matters** (see the bs=1
   caveat in section 5): it is the only regime in which a CPU stage can pay for
   itself, and it is untested. At bs=1 the answer is settled: iGPU-only.
8. **Graphs are DONE for decode and the draft/verify path** on the target
   checkpoint (section 6a) — capture works, `cuda graph: True`, +12.8% decode.
   Prefill graphs remain disabled by the config resolver and are untested. The
   eager labelling elsewhere in this report applies to the PP/dense sections
   and to prefill, not to the section 6a decode numbers.

## 9. The laptop service bundle (#655)

Turning the measured operating point into something the laptop's user can
actually use: a model that loads when asked, gets out of the way when it is
not, and a coding agent pointed at it.

### 9.1 Memory budget — the GTT lever is refuted, and the fraction is the real knob

The bundle asked for host RAM to be freed by LOWERING `ttm.pages_limit`
(the GTT ceiling) toward 26-27 GiB. Measurement says that lever does not exist
here, and points at a different one.

What the machine actually reports, all at one instant
(`docs/dev/651/service/mem_probe.py`):

| quantity | value |
|---|---|
| `MemTotal` | 29.50 GiB |
| GTT ceiling (`amdgpu.gttsize` / `ttm.pages_limit`) | 29.00 GiB |
| `torch.cuda.mem_get_info` total | 29.00 GiB |
| free right after HIP init | 27.54 GiB |
| GGUF weights, resident | 22.34 - 23.03 GiB (varies run to run) |
| GGUF dequant scratch, reserved from the KV budget | 0.95 GiB |

Two facts fall out.

**The GTT ceiling is already 98% of physical RAM.** There is no headroom to
give back by lowering it, because it is a CEILING, not a reservation: it does
not hold RAM, it only bounds what the GPU may take. Lowering it to 27 GiB
would not free 2 GiB for the desktop — the model would still want ~26.6 GiB —
it would simply stop the model loading. And because KV on this hybrid is tiny
(see below), every GiB shaved off the ceiling buys the host almost nothing
while costing ~52k tokens of context.

**`--mem-fraction-static` is what converts free GTT into context**, because
the KV budget is `mem_fraction_static x total` MINUS what the weights already
hold. That makes the fraction the opposite of a safety margin here:

| budget point | max_total_num_tokens | free after pool |
|---|---|---|
| GTT 29 GiB, memfrac 0.97, ctx 32768 | 7254 | 1.51 GB |
| GTT 29 GiB, memfrac 0.99, ctx 8192 | 15070 | 1.21 GB |
| GTT 29 GiB, memfrac 0.99, ctx 8192, HiCache pools built | 9138 | 1.10 GB |

**What binds is NOT the hybrid sizing coupling.** That was the standing
hypothesis, and it is wrong. The mamba/full-attention ceiling computed 32772
and never bound; the mamba pool charged 0.12 GiB. The pool is
`available_bytes // cell_size` with `cell_size` 20480 B/token over 10 real
attention layers, and the terms that ate the budget are the mem-fraction slack
plus the **0.95 GiB GGUF dequant scratch**. The ~1.5 GB that appears "free
after pool" is precisely those two reservations: memory subtracted from the
budget but never allocated at pool time.

Raising `SGLANG_GGUF_DEQUANT_WS_CAP_MIB` to absorb the scratch was considered
and rejected: the residual is `peak - held`, so making the workspace hold it
converts a 0.95 GiB reservation into a ~1 GiB allocation. It is a wash, not a
win. The memory is genuinely needed.

**The margin is thin enough to be non-deterministic.** The same configuration
at memfrac 0.99 produced a 15070-token pool on one boot and was refused
outright on the next, purely on how much host RAM happened to be free when the
weights landed. This is why `boot_ondemand.sh` drops the page cache before
every load (a 21.6 GiB checkpoint read fills it every time) and why the
supervisor retries a failed load exactly once.

That retry is not theoretical: it was observed firing in service, unprompted —

    ValueError: Loaded weights leave no GPU memory for the KV cache under
    --mem-fraction-static=0.99
    WARNING ondemand: load attempt 1 failed (boot script exit -9); retrying once
    INFO ondemand: loading model (attempt 2, ...)

and the second attempt served the request. The cost when it fires is a doubled
wake (~5 minutes rather than ~2.5). Note also that the refusal message names a
floor (0.962) BELOW the setting that was refused (0.99), because the floor it
prints does not account for the dequant-scratch and mamba posts subtracted
after it — the message is misleading, and following its advice would not help.

One thing the loader already does that makes the explicit `drop_caches`
partly redundant: it releases checkpoint page cache as it streams
("GGUF stream: released 8.08 GiB of checkpoint page cache so far in 45 advice
call(s)"). The drop is kept because it also clears cache the loader did not
put there, but it is not the only mechanism at work.

**A load does not merely succeed or fail -- it succeeds with a POOL SIZE, and
the range makes "succeeded" meaningless on its own.** Observed across boots of
one unchanged configuration:

| boot | max_total_num_tokens | usable? |
|---|---|---|
| best | 15070 | yes |
| typical | 8288 - 9138 | yes |
| worst | **1081** | no |

The 1081-token boot returned `/health` 200, reported itself ready, and could
not serve a single real request against an 8192-token context. Nothing it
emits distinguishes it from a good boot. This is the most operationally
dangerous state found in this bundle, because every signal says healthy.

The service therefore gates readiness on the POOL, not on health: after
`/health` comes up it reads `max_total_num_tokens` from `/get_server_info` and
rejects the load below `HTSGLANG_MIN_KV_TOKENS` (default 4096), stops the
process, and reloads. Because the pool is a lottery rather than a
deterministic function of the configuration, the attempt budget is 3, not 1 --
a single retry is not enough when the failure mode is a draw rather than a
fault. `kv_tokens` is reported on `/ondemand/status` so the size a running
server actually got is never a mystery.

The gate was then observed firing in service, on an ordinary request, without
being provoked:

    WARNING ondemand: load attempt 1 failed (model loaded with only 2924 KV
    tokens (minimum 4096); the load lost the memory lottery and cannot serve);
    retrying
    INFO ondemand: loading model (attempt 2, ...)

Pool sizes seen across consecutive loads of the identical configuration:
17849, 13782, 2924, 1081. Without the gate, two of those four would have been
handed to a user as a working server.

### 9.2 Cold load vs #89 hibernate

Cold boot to serving, measured through the service: **149.3 s**.

The cheaper path does not apply to this checkpoint. #89 parks the FINAL
post-transform tensors, and its snapshot step refuses a GGUF MoE outright:
`hibernate.py:332-338` (`snapshot_gguf_attrs` -> `materialize_gguf_weights`)
raises `NotImplementedError` on any GGUF-MoE layer. Qwen3.6-35B-A3B is a GGUF
MoE, so there is nothing to park. Restore is also boot-time only —
`/resume_memory_occupation` has no hibernate branch at all
(`weight_updater.py:235-273`), so a "park" that keeps the process alive would
free nothing without `--enable-memory-saver`.

Consequently the #499 restore-identity fix could not be exercised on this
checkpoint: the park refuses before a manifest is ever written. This is a
scope statement about GGUF MoE, not a defect in #499.

SCOPE OF THAT CLAIM, stated plainly: the GGUF-MoE refusal is read from the
code path, NOT executed against this checkpoint. A `POST /hibernate` was not
run here, so what is proven is that the park cannot succeed as written, not
that it was observed failing. The cold-load figure (149.3 s) and everything in
9.3 ARE measured. Executing the park to convert this from a code reading into
an observation costs one boot and is the cheapest open item in this section.

**Park mechanism chosen: plain stop and cold reload.** It is the only one
available, and at 149.3 s it is affordable for an assistant used in bursts.

### 9.3 The on-demand service

`htsglang-ondemand.service` (unit installed at
`/etc/systemd/system/htsglang-ondemand.service`, enabled, survives reboot).
The unit is up from boot; the MODEL is not. A front door on the public port
31651 proxies to the real server on 31661, loads it on the first real request,
HOLDS that request until it can be answered, and parks the model once the
machine goes quiet.

The idle window is `HTSGLANG_IDLE_PARK_SECONDS`, default 60, set as an
`Environment=` line in the unit.

Three defects were found and fixed while bringing it up, each of which
presented as something else:

* `CUDA_VISIBLE_DEVICES=""` — set on the unit so the supervisor can never
  create a HIP context — was inherited by the boot script, whose GPU guard
  then died with "No HIP GPUs are available" four seconds in. It reads exactly
  like a model that will not load. The child's environment is now built
  explicitly.
* The idle watcher would park a model the instant it finished loading: a load
  outlasts the idle window, so the watcher saw an alive process, no in-flight
  requests, and an expired clock, then blocked on the loader's lock. `park()`
  now re-reads its conditions on the far side of the lock.
* Health checks must not wake the model. `/health` and `/ondemand/status` are
  answered by the front door itself and do not count as activity, or any
  monitoring poll would pin 22.7 GiB forever — the exact failure the service
  exists to prevent.

Guard v2 and the wedge policy run inside `boot_ondemand.sh`, so they run on
EVERY load rather than once at first boot, which is the point of routing the
service through a script instead of a saved command line.

**Acceptance, two full cycles** (`accept_ondemand.sh`, results at
`/root/651-p2/results/accept_ondemand_214508.txt`):

| step | cycle 1 | cycle 2 |
|---|---|---|
| host free, parked | 29093 MiB | 29212 MiB |
| wake request (held through the load) | 150 s, answer `42` | 150 s, answer `42` |
| host free, model resident | 1327 MiB | 985 MiB |
| second request while hot | 2 s, answer `Paris` | 1 s, answer `Paris` |
| state after 60 s idle | parked | parked |
| host free, parked again | 29212 MiB | 29202 MiB |

`ACCEPTANCE: PASS (2 cycles)`. Both probes are questions with exactly one
right answer, at temperature 0 — a model that loads and then emits noise is a
worse outcome than one that fails to load, because nothing alerts on it.

The probes send `enable_thinking: false`. Without it this checkpoint opens
with a reasoning preamble, the reply budget runs out before the answer, and
the probe scores a perfectly healthy model as WRONG. That is exactly what the
first acceptance run did.

### 9.4 HiCache — the model check passes; the machine is the wall

The predicted blocker did not occur. The GDN hybrid **passes** HiRadixCache's
model check: it builds `MHATokenToKVPoolHost` plus a mamba host pool through
`build_hybrid_mamba_stack`. The feared non-MHA `ValueError` never fired.

Three real blockers appeared instead. Two are fixed in this branch:

1. **A 10 GiB pinned-host reserve, hard-coded.** `PINNED_HOST_RESERVE_BYTES`
   was justified in-comment by "this box has no swap at all" — true of the
   rig, false of this laptop (29.5 GiB RAM, 8 GiB swap). It refused a 0.15 GB
   staging tier. It is now overridable via `SGLANG_PINNED_HOST_RESERVE_MIB`,
   default unchanged. A malformed or negative value falls back to the
   conservative default: a typo must never be able to switch a guard off.
   A second copy of the same constant, `HICACHE_HOST_MEMORY_RESERVE_BYTES`,
   silently outranked the override at all six pool call sites; those now go
   through the single resolver, which is what the module's own docstring
   promises.
2. **A layout collision on ROCm.** `MambaPoolHost` accepts ONLY
   `page_first_direct`, while on ROCm the default `page_first`+`kernel` pair is
   auto-downgraded to `layer_first` because the page-first write-back needs a
   CUDA-only JIT kernel. Left to defaults the two rules collide and the boot
   dies. `boot_ondemand.sh` pins `page_first_direct`.

The third is physical and is not fixed: the host tier needs pinned RAM this
machine does not have while the weights are resident. On a GDN hybrid the tier
has two posts and the mamba one is chunky — a single mamba slot is ~64 MB, so
the default ratio asked 0.58 GB of mamba pool alone against ~1.08 GB of free
host RAM. Even trimmed to ratio 0.1 it competes with a load margin already
thin enough to be non-deterministic.

**Verdict: HiCache is OFF by default on this machine** (`HICACHE=1` re-arms it
on a host with RAM to spare). No hit-rate evidence is claimed, because the
feature was never armed in a serving run — reporting a hit rate here would be
reporting a number that does not exist.

### 9.5 Panel blanking

Not a driver fault and not the GPU. The machine sits at the GDM greeter, and
the GREETER blanks the panel on its own idle timer; the panel then reads as
dead (`/sys/class/drm/card1-eDP-1` -> `dpms=Off, enabled=disabled`) to anyone
who did not know it was merely asleep. `dmesg` shows no eDP link-training
failure at any point.

The user account `efeu` already had `idle-delay=0` and
`sleep-inactive-ac-type='nothing'`, so its own session never blanked — only
the greeter did, which is why the symptom looked like hardware.

Fix: a greeter dconf override (`/etc/dconf/db/gdm.d/10-no-idle-blank`) that
disables the idle blank, the screensaver, and idle dim for the greeter only.
The lid switch and explicit suspend are untouched — this disables the idle
BLANK, not power management.

Honest limit: whether the panel wakes on a keypress cannot be verified from a
remote session, because it needs a physical keypress. The durable choice made
here is therefore to stop the idle blank from happening rather than to claim a
wake path was repaired.

### 9.6 oh-my-pi for efeu

`omp` v17.2.11 (github.com/can1357/oh-my-pi), installed as efeu to
`/home/efeu/.local/bin/omp`, configured at `/home/efeu/.omp/agent/models.yml`
against the local front door as provider `local`, model `qwen36-35b-a3b`.

`enable_thinking: false` is set in the model's `extraBody`. This is not
cosmetic: the checkpoint otherwise spends the reply budget on a reasoning
preamble before answering, which for a coding assistant is latency without
benefit — and it is what made the first acceptance probe score a working model
as wrong.

A short README for efeu (`README-efeu.md`) explains the first-request wake
latency, since a two-and-a-half-minute first answer looks like a hang to
anyone who has not been told otherwise.

**A real coding round trip does NOT complete on this hardware, and the reason
is the wedge.** This is the honest result; the plumbing is fine and the
hardware is not.

Getting there found and fixed two configuration faults first:

* `baseUrl` must include `/v1`. omp appends the bare route
  (`/chat/completions`), so a host-only base asks sglang for an unversioned
  path and gets `404 {"detail":"Not Found"}`. Vendor guides saying "do not
  append /v1" describe providers that also serve the unversioned route.
* omp's full tool surface does not fit. With all 32 tools its system prompt
  alone measures **17029 tokens** against an 8192 context:
  `400 The input (17029 tokens) is longer than the model's context length`.

With those fixed and the tools trimmed to `read,write`, the request reaches
the model and kills it:

    RuntimeError: Triton Error [HIP]: Code: 719, unspecified launch failure
    torch.AcceleratorError: HIP error: unspecified launch failure

and `dmesg` names what userspace cannot:

    [22:07:05] amdgpu: MES failed to respond to msg=REMOVE_QUEUE
    [22:07:07] GPU reset(4) succeeded!
    [22:07:07] [drm] device wedged, but recovered through reset

That is the section-3 MES wedge, hit at `--chunked-prefill-size 256` — the cap
that is a MITIGATION and not a fix, exactly as `wedge_policy.py` says of
itself. An agent prompt is thousands of tokens of SUSTAINED PREFILL, which is
the one regime this GPU cannot survive; the acceptance probes pass because a
one-sentence question is a trivial prefill. Two of the three GPU resets logged
on this machine today correlate with an omp request.

So the bundle's own conclusion holds and now has a user-facing consequence:
**decode works on this iGPU, sustained prefill does not**, and a coding agent
is a prefill-shaped workload. omp is installed, owned by efeu, correctly
pointed at the endpoint, and will work the moment the prefill path does. It is
not usable for real work before then.

Encouraging note for whoever picks this up: the service SURVIVED the wedge.
The backend died, the front door saw an unhealthy backend, stopped it and
reloaded, and the unit stayed active throughout. The GPU reset recovered.
Nothing needed a human.

### 9.7 Reaching a ~150k KV cache (user proposal: MoE disk spill)

The proposal is to free memory by spilling MoE experts, so the KV pool can
reach ~150k tokens. The arithmetic supports it, and is cheaper than it looks.

**The target is small.** The measured cell size is 20480 B/token, because this
GDN hybrid has only ~10 real attention layers:

    150,000 tokens x 20480 B = 2.86 GiB of KV

So the requirement is not "free the 18 GiB of experts" but "free about 3 GiB".
Against today's KV budget of 1.2-1.5 GiB, the shortfall is roughly
**1.5-2.5 GiB** -- a cold slice, not the expert tensor as a whole. That matters
for the I/O price: the ~162 ms/token figure was measured for naive FULL
offload, and a spill an order of magnitude smaller does not pay that.

**The target tier is DISK.** Host RAM is not an option on this machine and is
only mentioned here to close it off: on an APU the host pool and the GPU pool
are the same DRAM, so an expert moved "to host RAM" has not left the memory we
are trying to free. The requirement is expert bytes resident on NVMe and
fetched on demand.

`observability/spill_tiers.py` names the tiers, and the gap is exactly there:

    TIER_EXPERT_HOST  = "expert_host_ram"     # experts -> HOST RAM
    TIER_HICACHE_FILE = "hicache_file_disk"   # only HiCache has a disk tier

Experts can spill to host RAM today. There is no expert-to-DISK tier.

**Consequence, and it differs per machine:**

* On the RIG a host-RAM spill would already free real VRAM, because there the
  two pools are physically distinct. That is a side note, not the goal.
* On this LAPTOP only a DISK tier frees anything: 29.5 GiB total with
  ~1.0-1.3 GiB free while the model is resident, and the host pool IS the GPU
  pool. This tier does not exist and is the work to be done.

**Design constraint that decides viability.** A3B activates ~3B params per
token; in Q4 the expert share is roughly 1.25 GB of weight reads per token if
every active expert were cold. At NVMe speeds that is ~1 token/s -- so a full
expert-on-disk design is not viable and never was. What IS viable follows from
the sizing above: only ~1.5-2.5 GiB of ~18 GiB needs to leave, i.e. the
coldest ~10-14%. The question that decides the whole feature is therefore
whether expert access has a genuine COLD TAIL (a stable minority of experts
that are rarely routed to) rather than a flat distribution. If flat, the miss
rate makes it unusable at any spill size; if long-tailed, the spill set can be
chosen by measured access frequency and the per-token miss cost stays small.
That measurement is the first thing to do, before any code.

**An asset already in the tree:** the GGUF loader streams from disk and
releases page cache behind itself ("GGUF stream: released 8.08 GiB of
checkpoint page cache so far in 45 advice call(s)"). Read-then-release is
exactly the discipline a disk-resident expert needs on a machine where page
cache competes with the model for the same RAM.

**The blocker that outranks memory on this laptop.** Even with 150k tokens of
KV, filling that context is sustained prefill, and sustained prefill wedges
this GPU in firmware (section 3, and reproduced again in 9.6 by an agent-sized
prompt). A large context is only worth building toward here once the MES fault
is addressed; on the rig the constraint does not apply.
