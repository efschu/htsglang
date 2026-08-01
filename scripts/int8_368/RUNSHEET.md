# Task #368 -- card runsheet: INT8 W8A8 decode-path microbench

Desk work is done. This sheet is everything the card operator needs: two
short windows, one per architecture, and a decision tree that turns the
numbers into the next task (or into closing #368).

## 0. What is being decided

`CompressedTensorsW8A8Int8.apply_weights` runs exactly two kernels per
linear layer and nothing else:

```
python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py
  :213   x_q, x_scale = per_token_quant_int8(x)          # Triton, one program per token row
  :215   int8_scaled_mm(x_q, layer.weight, x_scale,      # CUTLASS, sgl_kernel
                        layer.weight_scale, out_dtype=x.dtype, bias=bias)
```

There is no Triton config lookup anywhere on this path (#370), so there is
no #255-style config-generation remedy. INT8 loses 4-10 % decode against
FP8 at small batch (#354); -6.5 % accept length accounts for part of it.
This run prices the two kernels separately at the real per-rank shapes and
at decode batch sizes, so the remaining gap is attributed rather than
guessed.

The FP8 side is measured twice, because the FP8 comparator is not one
kernel. `Qwen3.6-27B-FP8` carries `weight_block_size [128,128]`, so
`Fp8LinearMethod` takes its `block_quant` branch (`fp8.py:1132`) into
`w8a8_block_fp8_linear` -- group-128 activation quant, block-scaled GEMM.
That is the lane #354 measured (`fp8_block_fused`). The compressed-tensors
FP8 scheme (`fp8_ct_fused`) is the structural twin of INT8 -- per-token
activation scale, per-channel weight scale, CUTLASS scaled-mm -- and is the
one that isolates the arithmetic question.

## 1. Desk pre-flight -- no lock, no card, run this first

```bash
cd /spinning/wt-368-prep
export PYTHONPATH=/spinning/wt-368-prep/python
PY=/spinning/htsglang-gpu/.venv/bin/python

CUDA_VISIBLE_DEVICES="" $PY scripts/int8_368/microbench.py --check-imports
CUDA_VISIBLE_DEVICES="" $PY scripts/int8_368/microbench.py --shapes-only
CUDA_VISIBLE_DEVICES="" $PY scripts/int8_368/microbench.py --dry-run
```

`--check-imports` resolves the six real kernel entry points without
touching a card and exits non-zero if any is missing. `--dry-run` runs every
lane on CPU stand-ins. Both were green on the desk before this sheet was
written; re-run them after any rebase, because a harness that has never
executed is not a harness.

## 2. Shape table -- rank 0, TP=3, BOTH deployed vectors

Two shard-plan vectors are in production and the final INT8-vs-FP8
comparison uses both: **decode runs on the auto vector `[30,17,17]`,
prefill runs on the phase-optimal vector of the `16,1,1` class** (#354's
`--rank-mlp-ratio`, which puts almost the whole MLP on the 5090). The bench
derives and measures both; identical GEMMs are measured once and attributed
to every plan that produces them.

All derived from the checkpoint config, cross-checked against
`sglang.srt.distributed.utils._partition_units_raw`.

| shape | N (out) | K (in) | layers/token | plans | module |
|---|---:|---:|---:|---|---|
| `auto_attn_qkv` | 7168 | 5120 | 16 | auto, prefopt, ref255 | `qkv_proj` (12 q x2 output gate + 2x2 kv, head_dim 256) |
| `auto_attn_o` | 5120 | 3072 | 16 | auto (also `auto/gdn_out` x48), prefopt, ref255 | `o_proj` |
| `auto_gdn_in_qkvz` | 8192 | 5120 | 48 | auto | `in_proj_qkvz` (2x8x128 + 2x24x128) |
| `auto_mlp_gate_up` | 16320 | 5120 | 64 | auto | `gate_up_proj`, local intermediate 8160 |
| `auto_mlp_down` | 5120 | 8160 | 64 | auto | `down_proj` |
| `prefopt_gdn_in_qkvz` | 14336 | 5120 | 48 | prefopt | `in_proj_qkvz` (2x14x128 + 2x42x128) |
| `prefopt_gdn_out` | 5120 | 5376 | 48 | prefopt | `out_proj` |
| `prefopt_mlp_gate_up` | 30944 | 5120 | 64 | prefopt | `gate_up_proj`, local intermediate 15472 |
| `prefopt_mlp_down` | 5120 | 15472 | 64 | prefopt | `down_proj` |
| `ref255_gdn_out` | 5120 | 2688 | -- | reference | #255 tuner queue, older auto vector |
| `ref370_fp8block_mlp_gate_up` | 30976 | 5120 | -- | reference | #370 tuner queue, FP8 block family |
| `ref370_fp8block_mlp_down` | 5120 | 15488 | -- | reference | #370 tuner queue, FP8 block family |

256 INT8 linear layers per decoded token on rank 0 under either plan = 512
kernel launches.

### Why the prefopt MLP appears twice, 16 elements apart

`IDLE_TUNER_QUEUE_370.md`
(`/spinning/gpu-battery-results/2026-08-01_370_sm120_configs/`) queues
`30976x5120` and `5120x15488`. **Those are the `16,1,1` family seen through
the FP8 checkpoint's unit family, and they are correct for that purpose** --
the queue tunes `--block-n 128 --block-k 128`, and `Qwen3.6-27B-FP8` carries
`weight_block_size [128,128]`, so its dense MLP coarsens to `17408/128 = 136`
units; `partition_units(136, [16,1,1]) = [121,8,7]` -> rank 0 holds
`121x128 = 15488`, gate_up `30976`.

`Qwen3.6-27B-INT8-W8A8` is channel/token quantized with no weight block, so
its family is `17408/gcd(17408,16) = 1088` units of 16;
`partition_units(1088, [16,1,1]) = [967,61,60]` -> rank 0 holds `15472`,
gate_up `30944`. Same vector, different unit family. Both rows are carried:
the `prefopt_*` ones are the INT8 shapes, the `ref370_*` ones the FP8 shapes,
so the two checkpoints can be read against each other at the shapes each
actually runs.

### Not INT8 at all

By the checkpoint's own ignore list: `linear_attn.in_proj_b` / `in_proj_a`
(so the merged `in_proj_ba` stays bf16), `lm_head`, and `re:.*mtp.*` --
**the whole MTP draft model runs unquantized**. Worth holding on to when
reading #354: the draft steps are not on the INT8 path at all, so a
kernel-level INT8 deficit can only show up in the verify step.

## 3. Arbitration (rig-runbook 7.1, per-card locks v2)

GPU windows are rotating under other agents. Do not skip any of this.

```bash
# Which physical card is which -- confirm by NAME every time, the
# torch-vs-NVML order trap is on file.
nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv,noheader
```

* One lock per card: `mkdir /tmp/gpu-card-<NVML-idx>.lock`, then write an
  `info` file inside it with owner, purpose (`#368 int8 microbench`),
  `acquired`, `nvml_index`, `uuid`, `pci_bus_id`. Atomic `mkdir` is the
  claim; a pre-existing directory means the card is taken -- wait, never
  break it.
* This measurement is latency-critical, so also take the quiet flag:
  `mkdir /tmp/gpu-quiet.lock` with an `info` carrying the expected duration
  (5 min). Remove it the moment the run ends.
* Only one card at a time is needed. Take it right before the run, release
  right after. Never hold both.
* A rig-wide `/tmp/gpu-owner.lock` means all cards are taken.
* Card-is-free check goes by `memory.used` (~0-10 MiB expected), never by an
  empty compute-apps list -- host-side processes are invisible from inside
  the container.

Release, always, including on failure:

```bash
rm -rf /tmp/gpu-quiet.lock /tmp/gpu-card-<NVML-idx>.lock
```

## 4. Process discipline

* **Never `pkill -f` or `pgrep -f`.** Other agents' servers match almost any
  pattern you would write, and killing one is unrecoverable for them.
* This harness starts no server and holds no port. It runs in the foreground
  and exits on its own. If it has to be launched detached, write a pidfile
  and kill only that PID:

```bash
setsid $PY scripts/int8_368/microbench.py ... > /tmp/int8_368.<card>.log 2>&1 &
echo $! > /tmp/int8_368.<card>.pid
# stop, if ever needed:
py-spy dump --pid "$(cat /tmp/int8_368.<card>.pid)"   # look before you kill
kill "$(cat /tmp/int8_368.<card>.pid)"
```

* After any kill, check `nvidia-smi --query-compute-apps=pid,used_memory`
  for leftovers and kill those **by PID**.

## 5. The two runs

Pick the card by UUID, not by ordinal -- `CUDA_VISIBLE_DEVICES` accepts a
`GPU-...` UUID and that is immune to the enumeration-order trap. The script
records `device_name` and `capability` in its JSON, so the arch is verifiable
after the fact.

```bash
cd /spinning/wt-368-prep
export PYTHONPATH=/spinning/wt-368-prep/python
PY=/spinning/htsglang-gpu/.venv/bin/python

# Arm A -- RTX 5090, sm120
CUDA_VISIBLE_DEVICES=<uuid-of-5090> $PY scripts/int8_368/microbench.py \
    --out /tmp/int8_368.sm120.json

# Arm B -- RTX 3080, sm86
CUDA_VISIBLE_DEVICES=<uuid-of-3080> $PY scripts/int8_368/microbench.py \
    --out /tmp/int8_368.sm86.json
```

Defaults: `--tp-size 3 --rank 0`,
`--plans "auto=30,17,17;prefopt=16,1,1"`, `M = 1,2,4,8,16,2048`, 8 lanes,
each duplicated as its own A-vs-A twin, 9 rounds, ~20 ms per burst.

Budget: 36 operating points x 16 timed streams x 9 rounds x 20 ms is about
2 min of pure burst time; with calibration and tensor construction expect
**5-8 min per card**, inside a 15 min window with room. Peak VRAM is under
3 GiB (largest weight pair is `30944x5120`, held in bf16 + int8 + fp8
per-channel + fp8 block, twice) -- the VRAM corridor rule does not bind
here, but nothing else should be resident during the window anyway.

If a window is tight, in this order:
`--plans auto=30,17,17` measures the decode vector only and roughly halves
the run; `--m 1,2,4,8` drops the prefill-shaped point; `--rounds 5` halves
again. Do not cut the A-vs-A twins -- without the noise floor the run
answers nothing.

Copy both JSONs out of `/tmp` when done.

## 6. Reading the result

Every point in the JSON carries `lanes` (median / p5 / p95 / min / max per
burst), `noise_floor` (the A-vs-A spread of each lane, `a_vs_a_rel`), and
`derived`:

| field | meaning |
|---|---|
| `int8_quant_share_of_fused` | fraction of the serving path spent in the Triton activation quant |
| `int8_gemm_share_of_fused` | fraction spent in CUTLASS |
| `int8_fused_minus_parts_ms` | fused cost minus the two isolated parts: the launch-gap a fusion could recover |
| `int8_over_fp8_block` | INT8 serving path / the block-fp8 lane the deployed FP8 checkpoint runs (what #354 measured) |
| `int8_over_fp8_ct` | INT8 / the compressed-tensors FP8 twin |
| `int8_over_bf16` | INT8 / unquantized `F.linear` |

**Read the noise floor first.** Any difference smaller than the larger of
the two lanes' `a_vs_a_rel` at that point is not reportable, full stop.

### Decision tree

Evaluate the **auto-vector shapes** on the decode points (`M` in 1,2,4,8)
and the **prefopt shapes** on `M = 2048` -- that is how each vector is
actually deployed. Weight shapes by their `layers` count: `mlp_gate_up` /
`mlp_down` occur 64x per token, `gdn_*` 48x, `attn_*` only 16x, so a
regression in the MLP pair is worth four times an equal one in attention.
Each shape row in the JSON lists every plan/module it serves, with its
layer count, so the weighting is read off the output rather than guessed.

1. **`int8_quant_share_of_fused` is the larger share at M<=8**
   -> *fusion candidate*. The activation quant is a one-row Triton launch
   at M=1; it is a fixed cost paid 256 times per token. Next task: fold the
   quant into the producer's epilogue (RMSNorm / residual add already write
   that tensor) or into the GEMM prologue. Report the projected saving as
   `int8_quant median x 256 x layers-weight` in ms/round, next to the
   implementation effort -- gain AND effort, as a pair, never a bare percent.

2. **`int8_gemm` dominates AND `int8_over_bf16` > 1 at M<=8**
   -> *kernel/dispatch candidate*. CUTLASS INT8 is losing to a plain bf16
   matmul in the memory-bound regime, i.e. the tile shape it picks for
   M<=8 is wrong. Next task: a GEMV-shaped path or an M-threshold dispatch,
   in the shape of the existing `--gemv-decode-divisor` work. Record which
   shapes and which M it holds for; a threshold is only defensible where
   the crossover is measured, not assumed.

3. **`int8_fused_minus_parts_ms` is large relative to both parts**
   -> the cost is between the kernels, not inside them: launch gap,
   allocator, or the `torch.empty_like` in `per_token_quant_int8`. Cheapest
   remedy first (buffer reuse), fusion second.

4. **All three above are inside the noise floor, and
   `int8_over_fp8_block` is ~1 at M<=8**
   -> the per-GEMM path is not the deficit. Then #354's remaining gap is
   accept-length physics plus whatever the MTP draft (which is bf16 in BOTH
   checkpoints) contributes, and **#368 closes with this evidence**: the two
   JSONs, the noise floor, and the statement that no INT8 kernel remedy
   exists at these shapes. That is a real result, not a failure -- it
   unblocks the INT8-vs-FP8 graph comparison, which is gated on this plus
   the FP8 idle-tuner configs.

Cases 1-3 are not exclusive. If two fire, order the follow-up tasks by
gain/effort, and say so explicitly.

### Cross-arch reading

sm120 and sm86 can land in different branches of the tree; the 5090 has
INT8 tensor cores the 3080 also has, but very different launch overhead and
memory bandwidth. Report each arch separately. A remedy that helps only one
arch is still a remedy -- gate it on capability, do not average the two.

## 7. What to hand back

* both JSONs, and where they were copied to;
* the noise floor at M=1 and M=8 for `int8_fused` on each card (one number
  each) -- everything else is read against it;
* which branch of the decision tree fired, per arch, with the numbers that
  decided it;
* if a follow-up task is proposed: the projected gain in ms/round AND the
  implementation effort, as a pair.
