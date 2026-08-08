# DESIGN #651: CPU+iGPU PP=2 prefill, iGPU-only decode, reshard between — APU laptop

Status 2026-08-08: desk design, falsifier-first. Serving-coherence isolation
(HANDOFF §12) gates ON-LAPTOP execution of the measured parts, NOT this design
work. Each work item names its falsifier; per the desk-written-never-executed
doctrine nothing below counts as validated until its falsifier has run.

The user's design (verbatim rationale in HANDOFF §6): prefill is
compute-bound — ADD both compute pools as two PP stages (CPU is an ACTIVE
stage); decode is bandwidth-bound on one DDR5 — iGPU alone computes; reshard
between phases; on shared memory the reshard is an ownership/view flip,
bytes never move.

## 1. CPU stage: K-quant-native compute (route B, primary)

Decision (HANDOFF §6.0.1) stands: the CPU stage computes on PACKED GGUF
blocks. New concretization — a **minimal ggml-cpu shim**, not a port of the
GGUF surface:

- Artifact: `docs/dev/651/cpu_stage/` — `libqmatmul_shim.so` wrapping ggml's
  `MUL_MAT(quantized W, f32 X)` at caller-chosen thread count, plus the
  build recipe against CPU-only llama.cpp libs. Falsifier: real checkpoint
  slices vs the numpy oracle, with a can-fail proof (bit-flip must turn the
  test red); determinism 3-run; throughput probe (rig numbers are recipe
  validation ONLY — the laptop re-measures; ms-per-round doctrine applies).
- Integration point in the fork: a `CpuGGUFLinearMethod.apply()` sibling that
  calls the shim when the layer's device is CPU, and a CPU-side
  `fused_moe_gguf` equivalent that loops active experts through the shim
  (top-k * tokens gemms; llama.cpp does exactly this on CPU). The CPU stage
  holds its layer share's PACKED bytes only (~k * quant size; no 3.17x).
- The gfx1103 Q6_K/IQ hazards do NOT apply on the CPU stage: ggml-cpu is the
  validated reference implementation family. If the CPU stage owns
  blk.34/38/39, the Q6_K containment shrinks.
- Zen4 note: the 8840HS has AVX-512; ggml-cpu autodetects. The shim build on
  the laptop must NOT set -march flags that fight the runtime dispatch.

## 2. Mixed-device world: one PP group, rank0=CPU, rank1=ROCm

From HANDOFF §6.0.5, turned into ordered work items with falsifiers:

| # | Item | Falsifier |
|---|---|---|
| W1 | Per-rank device: `--pp-device-map cpu,cuda` (new, PP-only, len==pp_size; refuse any other parallelism combined) threaded ModelRunner.device -> GroupCoordinator.device -> get_default_distributed_backend (gloo for the CPU rank; the unconditional gloo cpu_group already exists, parallel_state.py:701-717) | 2-process CPU-only unit test: world boots, one rank forced device=cpu, groups form, barrier passes (rig, CUDA_VISIBLE_DEVICES=99 for the CPU rank; GPU rank needs the laptop or a rig card) |
| W2 | Recv-side device fix: receiver allocates on ITS OWN device and `.to(device)` after gloo recv (parallel_state.py:238-246, :2334-2364, :2395) | same 2-process test sends a CPU tensor to a "cuda" rank; today it lands wrong — test must be RED first on the unpatched tree |
| W3 | CPU stage runs eager (cpu_graph_runner.py:609-610 asserts pp_size==1 — relax to skip-graphs-on-cpu-stage, not a general lift) | boot smoke |
| W4 | Scheduler: PP stages must occupy disjoint GPU groups (server_args.py:9066-9098) — the CPU stage needs an exemption from GPU-group accounting | arg-validation unit test |

Zero-copy stage handoff (gloo memcpy replaced by shared-memory tensor views)
is EXPLICITLY DEFERRED: on this machine the p2p payload per round is
activations of one chunk (1024 tok x 2048 hidden x 2 B = 4 MiB), gloo memcpy
of 4 MiB over DDR5 is ~0.1-0.2 ms — noise against multi-second rounds. The
latent optimization stays ticketed, not built (same class as the PD KV copy,
HANDOFF §5).

## 3. The phase flip and PP+spec exclusivity

Decode after the flip is single-worker; `server_args.py:16264` blocks spec
whenever pp_size>1 AT BOOT. Route A (#631) owns making that phase-aware.
Laptop degenerate case (bytes never move) per HANDOFF §6.0.4: check Route A's
#297-envelope state BEFORE building anything here — the only laptop-specific
piece is the GDN/mamba state-cache ownership flip, which on shared memory is
a pointer handover. DO NOT build flip machinery in #651; consume Route A's.

## 4. Disk-park inventory (user directive, HANDOFF §12.6)

Phase-exclusive state parks on DISK (named files, never swap) because GTT
and host RAM are the same DDR5 — every parked byte is reshard headroom.
Decision rule per item: measured ms among (a) keep resident, (b) disk-park +
sequential reload, (c) drop + reconstruct. NVMe on this laptop reads
~3.5 GB/s sequential -> reload cost ~0.29 ms/MiB; reconstruct cost is
item-specific. Inventory (sizes measured where known):

| Item | Lives in | Size | Phase exclusivity | Park candidate route |
|---|---|---|---|---|
| HIP graph pools (decode graphs) | GTT | O(100 MiB), measure at capture | decode-only | (c) drop + re-capture (~seconds) or (b); measure both |
| Dequant scratch `_DEQUANT_WS` | GTT | prefill-batch dependent (§9.3, unpriced) | prefill-only | (c) drop — grow-only workspace rebuilds itself |
| CPU-stage packed weights (its layer share k) | host RAM | k * 21.6 GiB | prefill-only | (a) resident is DEFAULT — they are the reshard's own payload; park only if decode ctx needs the room |
| Draft/NEXTN state (when spec lands) | GTT | ~0.5 GiB class | decode-only | (b) park during prefill phase |
| Vision tower (never used, text-only ckpt) | GTT | 818 MiB | NEVER used | drop at load (#651b) — not a park, a delete |
| Mamba/GDN state cache | GTT | 61.9 MiB/seq fp32 | phase-SHARED | never parked — it is the live sequence state the flip hands over |
| KV cache (10 full-attn layers) | GTT | 20 KiB/tok | phase-shared | never parked |

Flip budget = park-write ms + reload ms, charged per FLIP not per request;
flips are regime-wide and rare (HANDOFF §12.6). Reuse #286 offload register
classes + #89/#456 sparse VRAM-to-disk writer; NO new spill path.

## 5. Measurement plan (unchanged doctrine, restated as the gate order)

1. Coherence gate (HANDOFF §12) — no measured number counts before it.
2. Floors on the coherent base WITH graphs (post-graphs baseline per
   coordinator steer): decode ms/round, prefill ms/round restated from
   tok/s, bandwidth floor (done: 80.0 GB/s read).
3. CPU-stage solo ms/round on the laptop via the shim (diagnostic bracket).
4. CO-RUN ms/round per stage, split compute vs wait; equal-co-run-ms sets
   `--pp-layer-ratio`; sweep 2-3 splits; runs >= 10 s; under-load pins arm
   each session (§12.8). Clocks/power never a basis.
5. Negligible-or-negative CPU contribution after throttling = reportable
   verdict, not failure.
