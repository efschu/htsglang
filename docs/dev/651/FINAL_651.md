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

## 3. Serving crash — localized to the MoE align kernel

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

### Strongest lead for the next session: a wave32/wave64 mismatch

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

**This is a hypothesis, not a validated result** — it was found by reading the
launch site after the second specimen named it, and no fix was built or tested
in this session. It is written down because it is cheap to test (build with
`-DWARP_SIZE=64` under ROCm, or derive it from `warpSize`, and re-run the
coherence probe under sustained load) and because it would also explain why the
failure is probabilistic rather than deterministic: whether the OOB write lands
on a mapped address depends on the launch's shared-memory layout.

A second, independent constraint found on the way: the KV pool admits only
**2735 tokens** at `--mem-fraction-static 0.97`, so the 8192-token sweep point
is rejected outright with HTTP 400 rather than measured. Any prefill sweep on
this machine must stay under that ceiling.

---

## 4. Floors

Measured on the guard-v2-gated boot. Method follows the standing rules:
A-vs-A noise floor first, warmup discarded, unique prompts (no prefix cache),
time-bounded runs ≥ 10 s per point.

**These are EAGER floors.** The design's gate order asks for floors *with*
graphs, but the coherent boot recipe runs `--disable-cuda-graph`; graphs on
this path are not yet trustworthy. Labelled accordingly rather than quietly
compared against graph-mode numbers.

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

### What blocks end-to-end PP=2 on this laptop

Not the plumbing — W1, W2, W2b and W3 are all in, and the mixed world forms in
a real 2-process gloo test. The blockers are:

1. **The serving crash at multi-chunk prefill** (section 3). A CPU+iGPU split
   is a *prefill* feature, so single-device prefill has to survive past one
   chunk before a split of it means anything.
2. **The CPU stage's weights.** The CPU stage must compute K-quant-native via
   the Route B ggml-cpu shim; that shim is proven at kernel level (~1e-2 vs
   the numpy oracle, MoE dispatch +6-16%) but has not been driven as a live
   pipeline stage under a real checkpoint.

Claiming a PP=2 prefill throughput number before those two is closed would be
a fiction, so this report does not claim one.

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

The tail is tight — p90 is 2.8% above the median and the max is 5.8% above —
so decode on this APU is steady, not bursty. That matters for the PP plan: a
fat decode tail would have argued against the iGPU-only decode phase, and it
does not.

Corroboration from the server's own instrumentation across all boots of the
day (141 samples): median 13.35 tok/s, max 18.91 tok/s — consistent with the
benched 13.7 tok/s.

### Prefill floor — NOT OBTAINED, and why

Three attempts, all killed by the `moe_align_kernel` fault of section 3 before
a single sweep point completed:

| attempt | lengths | outcome |
|---|---|---|
| 1 | 256,1024,2048 | died in warmup (2048) |
| 2 | 256,1024,2048 | died in warmup |
| 3 | 256,512 | died in warmup (512) |

Prefill load reaches the fault far faster than decode load: the decode bench
survived a full warmup + two 20 s arms (~60 s of continuous generation), while
every prefill sweep died within seconds. Whether that is because prefill
launches the MoE align kernel with larger/more varied token counts, or simply
launches it more often per unit time, is not established here.

The server's own `input throughput (token/s)` line is **not** usable as a
substitute: it reports ~5.2-5.8 tok/s for 40-token prompts whose measured TTFT
was 0.93 s (≈44 tok/s), so the two disagree by ~8x and the metric's definition
would have to be established before trusting it. Reporting it as a prefill
floor would have been a fabricated number.

**So: no prefill floor is claimed.** The gate order in `DESIGN_651_pp_apu.md`
puts floors before any CPU/GPU split arithmetic, and the prefill half of that
gate is not passed. `layer_split.py --split-from` must not be run until it is.

---

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
6. **The serving blocker is `moe_align_kernel.cu:530`, not prefill length.**
   Three specimens, one of them under six short prompts. Lead: `WARP_SIZE`
   stays 32 while the ROCm shuffle mask is widened to 64 lanes.
7. **`pkill -f <pattern>` run inline over ssh kills the invoking shell**, because
   the pattern is in its own command line. Two boots were lost to this before it
   was spotted. Restart logic lives in `restart_serving.sh` for that reason.

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
* **PP=2 plumbing is complete through W3** with green unit coverage, but has
  never been run end-to-end on this laptop.
* **Sustained serving is not yet possible**, because of a MoE-align-kernel
  launch failure that kills the server in 20 s to 7 min under load. This is the
  one thing standing between the current state and a real PP=2 prefill
  measurement, and it now has a named file, a named line, and a concrete
  hypothesis to test first.

No PP=2 throughput number is claimed, and no prefill floor is claimed, because
neither was measured.
