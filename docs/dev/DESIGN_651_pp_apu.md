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
| W1 | Per-rank device `--pp-device-map cpu,cuda` (PP-only, len==pp_size, refuse other parallelism). SPAWN-PATH MAP (subagent audit 2026-08-08, file:line in its report): plumb like `--rank-gpu-id` (parent resolves, per-child env via the `maybe_reindex_device_id` slot in entrypoints/engine.py:733-753 + configure_scheduler_process); the CPU child needs CUDA/HIP/ROCR_VISIBLE_DEVICES="" + SGLANG_USE_CPU_ENGINE=1 (a new branch — the existing helper cannot express "no device"). CRITICAL correction: `is_hip()` is a BUILD probe, so on the laptop a GPU-less process still probes cuda-alike — GroupCoordinator.device (parallel_state.py:671-683) MUST take the per-rank device string, not the platform probe. WORLD backend must be RANK-UNIFORMLY gloo in a mixed world (per-process derivation would deadlock init); subgroups inherit it (fine at tp=1). `pre_warm_nccl` must become per-rank (model_runner.py:1849 crashes a GPU-less rank under pp>1). attention/sampling backend resolution per rank (server_args._handle_cpu_backends is server-global). Pool layer needs NOTHING: ~40 sites already parameterize on self.device; sizing collectives already run on cpu_group. | 2-proc gloo world test, one rank device=cpu: groups form, barrier passes, pool sizing min-reduce completes; laptop is the cuda-side proof |
| W2 | DONE (56d628758b): recv lands on the receiving group's device via `_move_received_tensor`; red-first proven with a meta-device receiver. | done |
| W2b | Send-side routing gap (found by the W1 audit): `comm_group = metadata_group if tensor.is_cpu else group` (parallel_state.py:2377) sends a CUDA tensor over the gloo device_group in a mixed world — gloo cannot. Route by group backend: if device_group backend is gloo and tensor is non-CPU, stage through `.cpu()` (correctness first; zero-copy later falls out of the shared-memory deferral). | helper-level unit test (backend-aware route choice) + the W1 world test |
| W3 | CPU stage runs eager (cpu_graph_runner.py:609-610 asserts pp_size==1 — relax to skip-graphs-on-cpu-stage, not a general lift) | boot smoke |
| W4 | Validation exemptions for a CPU rank: `_validate_pp_stage_gpu_groups` treats every world-rank as a physical card; `--rank-gpu-id` co-requisites are GPU-shaped (rank_gpu_memory_mib/NVML); `apply_rank_memory_budget` indexes rank_gpu_id unconditionally. CPU rank budgets host RAM instead. Known bite: `max_total_num_tokens` is min-reduced world-wide (DESIGN_625 B5) — the CPU stage's capacity caps the GPU stage; set the CPU stage's token capacity explicitly generous. | arg-validation unit tests |

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
