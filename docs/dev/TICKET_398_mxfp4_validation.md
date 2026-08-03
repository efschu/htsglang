# TICKET #398 — numerical gates and payoff boots for the native GGUF MXFP4 path

Status: **GPU-PENDING.** Everything below is desk work plus a full CPU-only
build (`CUDA_VISIBLE_DEVICES=99`). No `/spinning/gpu-arb/` window was taken and
no kernel has executed. Nothing in this document may be cited as a measurement
until the corresponding section is filled in from a real run.

What #398 shipped (source): a complete GGUF MXFP4 (ggml type 39) kernel set in
`sgl-kernel/csrc/quantization/gguf/` — dequantize, dense MMVQ, dense MMQ, MoE
MMVQ, MoE MMQ, plus the `ggml_moe_get_block_size` registration and a
`ggml_mxfp4_native` capability marker. Python side: MXFP4 enters
`DEQUANT_TYPES` / `MMVQ_QUANT_TYPES` / `MMQ_QUANT_TYPES` iff the installed
wheel carries the marker, and the load-time MXFP4→Q5_0 repack
(`gguf_mxfp4_repack`) becomes a no-op in that case.

---

## 0 — Before anything: which card is which

Physical index ≠ torch ordinal on this rig (catalog §11). Resolve through the
IdentityMap, never by assumption:

```bash
CUDA_VISIBLE_DEVICES= $V/bin/python -c "
from sglang.srt.registry.nvml import identity_map
for c in identity_map(allow_cuda_init=True):
    print(c.nvml_index, c.cuda_ordinal, c.name, c.uuid, c.pci_bus_id)"
```

Both arches are separate gates:

* **sm86** — RTX 3080. The MMVQ/MMQ launch parameters (`MMQ_X/Y_MXFP4 = 4/32`,
  `NWARPS_MXFP4 = 4`) are the CUDA branch, shared with every other type.
* **sm120** — RTX 5090. Same branch, different occupancy; and this is the only
  card where an MXFP4 path (`Mxfp4MarlinMoEMethod`, safetensors route) existed
  before, so an sm120-only green result would not prove the GGUF path.

---

## 1 — Gate A: numerical correctness (both arches)

```bash
V=/spinning/htsglang-gpu/.venv
cd /spinning/wt-398-mxfp4
CUDA_VISIBLE_DEVICES=<sm86_ordinal> PYTHONPATH=$PWD/python \
  $V/bin/python -m pytest test/registered/unit/quantization/test_gguf_mxfp4_cuda.py -q
CUDA_VISIBLE_DEVICES=<sm120_ordinal> PYTHONPATH=$PWD/python \
  $V/bin/python -m pytest test/registered/unit/quantization/test_gguf_mxfp4_cuda.py -q
```

14 tests, currently all `SKIPPED: no CUDA device`. What each one gates:

| test | reference | gate |
|---|---|---|
| `test_exact_against_the_host_reference_fp32` | python host decoder | **exact**, zero tolerance |
| `..._half_and_bfloat16` | host decoder cast to the dtype | **exact** |
| `test_real_tensor_rows` | host decoder over shipped `blk.26.ffn_down_exps` bytes | **exact** |
| `test_shard_that_is_a_multiple_of_32_but_not_of_256` | host decoder, 3x96 | exact + no NaN leaked past the guard |
| `test_odd_block_offsets_are_read_correctly` | host decoder, 2048-wide row | exact (this is the `get_int_b1` claim) |
| `TestMXFP4MMVQ` (3) | dequant + fp32 matmul | atol 1.5 / rtol 3e1 (bf16, the existing GGUF suite's numbers) |
| `TestMXFP4MMQ::test_matches_...` | dequant + fp32 matmul | atol 1.5 / rtol 1e4 |
| `TestMXFP4MMQ::test_agrees_with_mmvq...` | MMVQ, same activations | **relmax < 1e-3** — the meaningful gate |
| `TestMXFP4MMQ::test_need_check_path...` | dequant + fp32 matmul | rows=100, exercises `need_check=true` |
| `TestMXFP4MoE` (3) | per-expert dequant + matmul | as above, plus `block_size > 0` |

Exactness on the dequant tests is not optimism: the kernel is a table lookup
and one multiply by a power of two, and `ggml_cuda_e8m0_to_fp32_half` is
bit-identical to the host's `e8m0_to_fp32_half`. Any delta is a bug.

**Record here:** per arch, the pytest summary line verbatim, plus
`nvidia-smi --query-gpu=name,uuid --format=csv` for the card that ran it.

```
sm86  : PENDING
sm120 : PENDING
```

### 1b — Gate A': the falsifier, on the real wheel

Hermetically the flip is already executed
(`test_gguf_mxfp4_native.py::TestDispatchFlip`, red and green in one process).
On the GPU, confirm the wheel actually carries it:

```bash
$V/bin/python -c "
import torch, sgl_kernel
print('marker:', hasattr(torch.ops.sgl_kernel,'ggml_mxfp4_native'))
from sglang.srt.layers.quantization.gguf import MXFP4_NATIVE, MMVQ_QUANT_TYPES
print('MXFP4_NATIVE:', MXFP4_NATIVE, '| in MMVQ set:', 39 in {int(t) for t in MMVQ_QUANT_TYPES})"
# expect: True / True / True
SGLANG_GGUF_MXFP4_NATIVE=0 $V/bin/python -c "...same..."   # expect False / False / False
```

---

## 2 — Gate B: the #479 replacement, on the ACTIVE quant

The UD-IQ3_XXS export we serve carries exactly **two** MXFP4 tensors:
`blk.26.ffn_down_exps.weight` (shard 3) and `blk.42.ffn_down_exps.weight`
(shard 4), each `[2048, 4096, 256]` = 1 140 850 688 B, **2.125 GiB together**.
Until #398 they were carried by the Q5_0 repack, which grows them 22/17:

| | bytes | delta |
|---|---|---|
| MXFP4 as shipped | 2.125 GiB | — |
| Q5_0 repack | 2.750 GiB | **+0.625 GiB** |

So the native path returns **0.625 GiB** on this checkpoint, spread over host
RAM and (for the resident share) VRAM. #479 traces what the untraced fallback
for these two tensors currently costs; whatever it finds, these kernels
replace it — the tensors now take the same dequant/MMVQ/MMQ route as every
other GGUF type.

**Boot:** the standard DSV4-GGUF TP=3 recipe (rig-runbook §4.5.4b), once with
`SGLANG_GGUF_MXFP4_NATIVE=0` (repack arm, today's behaviour) and once without
(native arm). Same boot, A-vs-A floor first (catalog §10 canon).

**Record:** per arm — the `GGUF MXFP4: ... run NATIVELY` / `GGUF MXFP4->Q5_0
load-time repack` log line, load time, resident VRAM per rank, free-VRAM
corridor (>= 400 MiB, memory rule), ms/verify and ms/prefill per rank.

```
repack arm : PENDING
native arm : PENDING
delta      : PENDING
```

If #479 finds a load-time dequant for these two tensors, the VRAM delta is
larger than 0.625 GiB by whatever that dequant materialized; quantify it here
rather than in #479, since only this ticket has both arms.

---

## 3 — Gate C: the DSpark head, native (feeds TICKET_470 Boot B)

Be precise about which DSpark artifact this unlocks, because two different
things are called "the MXFP4 DSpark head":

| artifact | format | who runs it before #398 | who runs it after |
|---|---|---|---|
| `DeepSeek-V4-Flash-0731-dspark-head[-filtered]` | **safetensors** MXFP4 | `Mxfp4MarlinMoEMethod` — **sm90/sm120 only** | unchanged; #398 is a GGUF path and does not touch it |
| `alessandrobologna/…-DSpark-Drafter-GGUF` (MXFP4-Q8_0, 10.149 GiB) | **GGUF** type 39 | nothing — MXFP4 was in no GGUF type set | **these kernels**, on sm86 AND sm120 |
| `am17an/DeepseekV4-Flash-20260731-DSpark` (10.148 GiB) | GGUF, MXFP4 experts | nothing | these kernels |

The second row is the actual new capability: a DSpark draft head that is not
confined to the 5090. TICKET_470 Boot B places the draft head solo on the
5090 precisely because marlin-MXFP4 is sm120-only (ANALYSE_463 §2, R1). With a
GGUF MXFP4 head that constraint is no longer structural, which reopens
`split` placement for the draft — the thing R3 was going to buy with a whole
GPTQ requant pipeline.

**This is a claim about reachability, not about speed.** The GGUF MMVQ/MMQ
path is not marlin; on sm120 the safetensors route may well stay faster. Both
arms must be measured before anything is recommended.

**Record:** boot of the GGUF DSpark head on sm86 (does it load and generate at
all — the hard question), then A/B against Boot B's safetensors arm on sm120.

```
GGUF head on sm86  : PENDING
GGUF vs safetensors on sm120 : PENDING
```

Prerequisite that is NOT discharged by #398: the loader side. Those files use
the `deepseek_v4_flash_dspark_draft` / `dflash` arch strings, and only `dflash`
has a consumer (ANALYSE_463 §2.1). #398 makes the TENSORS executable; the arch
string, name map and sibling config are a separate piece of work and belong to
#470's python integration, not here.

---

## 4 — Gate D: bullerwins lossless MXFP4_MOE images — feasibility, honestly

| image | size | experts |
|---|---|---|
| `MXFP4_MOE-BF16` | 150.7 GiB | MXFP4 |
| `MXFP4_MOE-Q8_0` | 145.6 GiB | MXFP4 |

Both were blocked **solely** by the missing type-39 kernels — everything else
in them is a type this stack already dispatches on. #398 removes that block.

Whether they are *servable on this rig* is a different question and the answer
is "at the edge, probably not resident":

* host RAM: **104 GiB total**, ~90 GiB available with nothing else running.
* VRAM: 32 (5090) + 20 + 20 = 72 GiB, of which the runtime keeps a corridor.
* 145.6 GiB of weights therefore cannot be RAM-resident, let alone
  RAM+VRAM-resident. Serving them means the GGUF page-cache stream
  (rig-runbook §4.5.5) plus expert offload from disk, i.e. the NVMe expert
  tier (`ANALYSE_389_nvme_expert_tier.md`), not the pinned pool.
* the currently served UD-IQ3_XXS is ~101 GiB on disk and already leans on
  that stream.

**Verdict to record after a boot attempt, not before:** the honest expectation
is that Q8_0 (145.6 GiB) is a disk-streaming configuration whose decode rate is
set by NVMe, and that the interesting comparison is quality-per-token against
UD-IQ3_XXS at a much lower token rate — not a throughput win. Do not start
this before gates A–C are green.

```
load attempt (Q8_0) : PENDING
```

---

## 5 — Gate E: ms/token vs the Q5_0 repack

The bandwidth argument, stated exactly:

| | bytes/32 values | bpw | read bytes rel. |
|---|---|---|---|
| MXFP4 | 17 | 4.250 | 1.000 |
| Q5_0 (repack) | 22 | 5.500 | 1.294 |

So the repack reads **29.4 % more bytes** for the same weights; equivalently
the native path reads **22.7 % fewer**. On the down-projections that is the
whole tensor, and MMVQ decode is weight-bandwidth-bound, so this is the one
place a measurable ms/verify delta should appear. On UD-IQ3_XXS only 2 of 45
layers' down tensors are MXFP4, so the whole-model effect is small by
construction — the honest headline is per-layer, and the whole-model number is
whatever it is.

Method (catalog §10 / memory "Benchmark-Harness-Pflichten"): same boot for
both arms is impossible (the flag is read at load), so run A-vs-A on each arm
first to establish that boot's floor, fix the clock, interleave the arms across
windows, and report **ms/verify and ms/prefill per rank**, not tok/s. Nothing
below the same-boot floor gets reported.

```
A-vs-A floor, repack arm : PENDING
A-vs-A floor, native arm : PENDING
ms/verify delta          : PENDING
```

---

## 6 — What would falsify #398

* any dequant test that is close-but-not-equal → the E8M0 conversion or the
  nibble order is wrong, not "rounding";
* MMVQ and MMQ agreeing with the fp32 reference but not with each other beyond
  1e-3 relmax → the MMQ tile's `need_sum=false` scale handling is wrong;
* a misaligned-address fault → something reached `qs` with a 2-byte-aligned
  loader instead of `get_int_b1` (the 17-byte stride);
* a shard whose element count is a multiple of 32 but not 256 producing
  garbage at the tail → the dequant guard is not doing its job.
