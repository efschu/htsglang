# Multi-rank-per-GPU co-location (TP emulation)

`--rank-gpu-id` maps each tensor-parallel rank to a **physical** GPU index.
When the list contains **duplicates**, several ranks share one physical GPU —
letting a machine with *N* cards emulate a `--tp-size` **larger than N**.

```bash
# TP=5 on 3 cards: 3 ranks co-located on physical GPU 0, one each on GPU 1 and 2.
--tp 5 --rank-gpu-id 0,0,0,1,2
```

> **This is an EMULATION, not real N-card performance.** Co-located ranks
> share one GPU's SMs and memory bandwidth; with CUDA MPS they run
> concurrently but still contend, so throughput is **lower** than the same TP
> degree spread across that many physical cards. Use it to validate
> correctness, sharding geometry, and distributed/graph-capture behavior at a
> higher TP degree than the hardware natively provides — not to measure
> production tok/s.

## Requirements

### 1. CUDA MPS must be running
Without MPS, co-located ranks are **time-sliced**: their kernels never overlap
and NCCL collectives busy-spin waiting for the peer rank (>20x slowdown).
Start the control daemon before launching:

```bash
nvidia-cuda-mps-control -d
```

The launcher warns if MPS is absent. The check probes the control daemon
directly (`nvidia-cuda-mps-control` reply), **not** the mere existence of the
`/tmp/nvidia-mps` pipe directory — that directory lingers after a crashed
daemon and is not a liveness signal.

MPS is fragile under co-location: if a boot shows the processes present but
0% GPU utilization persistently, restart MPS cleanly (`echo quit |
nvidia-cuda-mps-control`, then re-launch the daemon) rather than debugging the
server — a wedged MPS server is the usual cause.

### 2. NCCL >= 2.30 for the co-located communicator
NCCL must map several ranks onto one physical GPU. This needs
`NCCL_MULTI_RANK_GPU_ENABLE=1`, which is only honored by **NCCL >= 2.30**.
Older NCCL rejects the communicator with:

```
Duplicate GPU detected : rank 0 and rank 1 both on CUDA device <busid>
ncclInvalidUsage
```

When duplicates are detected, the launcher auto-sets (each only if you have
not set it yourself):

- `NCCL_MULTI_RANK_GPU_ENABLE=1` — enable co-located ranks.
- `NCCL_NVLS_ENABLE=0` — NVLS (NVLink SHARP) is invalid when two ranks share a
  device; disabled explicitly instead of relying on NCCL's silent fallback.
- `NCCL_MAX_CTAS=max(1, 8 // max_colocated)` — cap CTAs so co-located ranks'
  NCCL kernels leave SM headroom for each other (override by exporting it).

If your environment ships an older NCCL (e.g. the torch wheel bundles 2.28.x),
side-load a newer one **without modifying the shared venv** via `LD_PRELOAD`:

```bash
# torch pins libnccl through DT_RPATH, so LD_LIBRARY_PATH does NOT override it;
# LD_PRELOAD is searched first and interposes torch's NCCL calls process-wide.
export LD_PRELOAD=/path/to/nccl-2.30/lib/libnccl.so.2
```

Verification caveat: `torch.cuda.nccl.version()` returns torch's **build-time**
NCCL header constant, so it keeps reporting the old version even when the
runtime library is newer. Confirm real uptake instead via (a) the co-located
communicator building at all (older NCCL throws "Duplicate GPU"), (b) the
newer `libnccl.so.2` appearing in a worker's `/proc/<pid>/maps`, or (c)
`NCCL_DEBUG=VERSION` in the log. Note the scheduler workers **fork** from the
launcher, so the `LD_PRELOAD`-loaded library is inherited even though the
`LD_PRELOAD` env var itself is not present in the workers' minimal env.

## Memory: per-GPU reserve for co-located ranks

`--rank-tp-ratio auto` sizes each rank's weight/KV budget proportional to its
GPU memory. On a co-located GPU that budget is split across the co-located
ranks. The per-GPU **reserve** (`--rank-auto-reserve-mib`) is what is held back
from the KV pool for **graph capture + CUDA context**, and each co-located rank
captures its **own** graphs, so a shared GPU needs roughly *N×* the
single-rank capture headroom.

Two practical points learned on real hardware:

- Auto budgets are derived from the **NVML total**, but CUDA exposes slightly
  less and each MPS context consumes more — so a reserve tuned for a
  single-rank GPU under-provisions a 3-rank GPU and OOMs at **graph capture**
  (not at compute). Give co-located GPUs a generous reserve.
- Pass reserve as a per-rank list aligned with `--rank-gpu-id` (per-physical-GPU
  max wins), e.g. `--rank-auto-reserve-mib 11500,11500,11500,3500,3500` for the
  `0,0,0,1,2` layout, and add
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to recover fragmentation
  during the multi-rank capture.

The only hard, enforced constraint is a **physical-impossibility check**: for
every physical GPU, `sum(co-located rank budgets) <= NVML total`. Leaving
enough headroom above that for context + fragmentation is the user's
responsibility.

## Validated example (5090 + 2x 3080, 32 + 20 + 20 GB)

Dense Qwen3.6-27B GGUF (Q4_K_XL), full-perf (CUDA graphs ON + NEXTN/MTP),
NCCL 2.30.7 side-loaded, MPS on:

```bash
LD_PRELOAD=/path/to/nccl-2.30/lib/libnccl.so.2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.launch_server --model-path <gguf> --tokenizer-path <dir> \
  --load-format gguf --quantization gguf \
  --tp 5 --rank-gpu-id 0,0,0,1,2 --rank-tp-ratio auto \
  --rank-auto-reserve-mib 11500,11500,11500,3500,3500 \
  --cuda-graph-max-bs-decode 16 \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --speculative-algorithm NEXTN --speculative-draft-model-path <gguf> \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

Result: 3 ranks co-located on the 5090 (~7 GB budget each) + one rank per 3080
(~17 GB each). All five ranks build the NCCL communicator, capture
target-verify + draft-decode + draft-extend graphs, and serve. Output is
coherent, retrieves a needle from a ~15k-token context, and is bit-identical
across two boots. Decode throughput is intentionally **not** representative of
a real 5-card deployment (three ranks time-slice one card via MPS).

### MoE (Qwen3.6-35B-A3B) at TP=5 — kv-boundary-aware auto split (#116)

The MoE A3B GGUF co-locates the same way (swap the dense model + tokenizer
for the A3B `-MTP-GGUF` paths, keep `--rank-tp-ratio auto`):

```bash
  --tp 5 --rank-gpu-id 0,0,0,1,2 --rank-tp-ratio auto \
  --rank-auto-reserve-mib 11500,11500,11500,3500,3500 \
```

A3B has `num_key_value_heads = 2` for `tp = 5`, so its full-attention layers
run the **REPLICATED-KV** geometry (every rank holds all kv heads; the q heads
split by the plan). Under `--rank-tp-ratio auto` the memory-proportional weight
vector could previously derive a q-head split whose per-rank packets **straddle
a global kv-head-group boundary** (e.g. q-heads `[2,2,2,6,4]`, rank 3 crossing
the boundary at head 8) — which the current-chunk ragged attention kernel
cannot represent, so it failed fast (#105 guard). The only workaround was an
explicit kv-aligned `--rank-tp-ratio`, which is **mutually exclusive** with
`--rank-auto-reserve-mib` (auto derives the budgets itself), so MoE TP=5
co-location was effectively blocked.

The auto planner is now **kv-boundary-aware** (#116): when `num_kv_heads < tp`,
the Q-dimension split is constrained to whole kv-head groups (repairing the
example to `[4,2,2,4,4]`), so `--rank-tp-ratio auto` composes with
`--rank-auto-reserve-mib` and boots directly. The split stays byte-identical
whenever the raw proportional split was already aligned (even TP, `kv >= tp`,
and explicit kv-aligned ratios are unchanged), and the #105 guard no longer
fires on any auto config — a general robustness win for **every** `kv < tp`
auto split, not just MoE TP=5.

## Scope

Single-node **pure Tensor Parallelism** only. Co-location is rejected in
combination with Pipeline / Data / Expert parallelism. Length of
`--rank-gpu-id` must equal `--tp-size`.
