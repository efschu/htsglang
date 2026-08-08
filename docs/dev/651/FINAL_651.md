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

| configuration | prefill @789 tok | A-vs-A | decode | A-vs-A | TTFT median |
|---|---|---|---|---|---|
| PP=2, CPU + iGPU | **671.3 / 669.2 tok/s** | 0.32% | **11.63 / 11.61 tok/s** | 0.21% | 0.129 s |
| single stage, iGPU only | **968.3 / 962.6 tok/s** | 0.59% | 16.75 / 15.80 tok/s | 5.70% | 0.093 s |

**Adding the CPU as a pipeline stage makes this slower, not faster:** 0.69x on
prefill and ~0.71x on decode. That is a reportable verdict, not a failure — the
design plan says so in advance ("Negligible-or-negative CPU contribution after
throttling = reportable verdict, not failure") — and the reason is structural:
these runs have `--max-running-requests 1`, and **pipeline parallelism pipelines
MICROBATCHES**. With one request in flight the stages run strictly in sequence,
so a PP split can only add the slower stage's time; it cannot overlap anything.
`bench_prefill.py` carries that warning in its own docstring.

**Per-stage wall split** (derived, with its assumption stated): for the same
789-token prompt, PP=2 takes 1175 ms and the iGPU alone takes 820 ms for *all*
28 layers. Assuming the default even 14/14 layer split and strictly sequential
stages, the iGPU half is ~410 ms, so the CPU half is ~1175 − 410 ≈ **765 ms** —
roughly **1.9x slower than the iGPU for the same layer share**. This is an
inference from two wall measurements, not per-stage instrumentation; a direct
per-stage timer is the honest way to confirm it and was not built here.

The practical consequence for the strand's target: a CPU+iGPU prefill split
only pays off with several chunks or concurrent requests in flight, and at a
layer ratio weighted by that ~1.9x — not the even split used here.
`layer_split.py --split-from` can now be fed real numbers for a dense model.

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
6. **Graphs.** Every floor here is eager; nothing is known about this path
   under graph capture.
