# TICKET #398 — numerical gates and payoff boots for the native GGUF MXFP4 path

Status: **GPU-PENDING.** Everything below is desk work plus a full CPU-only
build (`CUDA_VISIBLE_DEVICES=99`). No `/spinning/gpu-arb/` window was taken and
no kernel has executed. Nothing in this document may be cited as a measurement
until the corresponding section is filled in from a real run.

> **Correction 2026-08-04 — the wheel IS installed.** The #398 merge message
> and the header above both say the wheel was built but not installed. That
> stopped being true on 2026-08-03 12:37. Proof, not inference:
> `/spinning/htsglang-gpu/.venv/.../sglang_kernel-0.4.4.dist-info/direct_url.json`
> reads `file:///spinning/wt-398-wheel/...whl` with
> `sha256=67f03cfa755efa01498c7732bd6ae015ec5673feffe9a51452fefdbe0dcd4664`,
> i.e. exactly the pinned #398 wheel, and
> `CUDA_VISIBLE_DEVICES=99 python -c "import sgl_kernel, torch;
> print(hasattr(torch.ops.sgl_kernel,'ggml_mxfp4_native'))"` prints `True`.
> Consequence: `MXFP4_NATIVE` is True in the serving venv, so MXFP4 is in
> `DEQUANT_/MMVQ_/MMQ_QUANT_TYPES` and the load-time repack is already a no-op
> unless a boot sets `SGLANG_GGUF_MXFP4_NATIVE=0`. Sections 1-5 stay PENDING —
> installed is not measured — but the "repack arm" in §2/§5 is no longer the
> default and must be produced with the env kill switch, not by doing nothing.
>
> **The gate could not run, and that was invisible.** With the wheel installed,
> `test_gguf_mxfp4_native.py` did not fail — it *aborted the interpreter*
> (`Fatal Python error: Aborted`) at the first `_FakeNativeOp` block. The
> helper probed `hasattr(torch.ops.sgl_kernel, "ggml_mxfp4_native")` before
> anything had imported `sgl_kernel`; torch registers ops when the extension
> `.so` loads, not when the namespace is touched, so the probe answered
> "absent" on a wheel that carries the op, the fake schema was defined, and the
> real `.so` loading afterwards registered the same schema twice — a C++ abort,
> not a catchable exception. Fixed by importing `sgl_kernel` before the probe
> (`test_gguf_mxfp4_native.py:179-192`). Falsifier: abort unfixed / 16 passed
> fixed, and the collision reproduces in four lines outside pytest
> (define fake -> `import sgl_kernel` -> abort).

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
SGLANG_GGUF_MXFP4_NATIVE=0 $V/bin/python -c "...same..."   # expect True / False / False
```

**Why the second arm is True/False/False and not False/False/False (#519).**
The first line is `hasattr(torch.ops.sgl_kernel, 'ggml_mxfp4_native')` — a
property of the WHEEL, i.e. of which objects are on disk. The environment
variable cannot change it, and it must not: it is the marker that says the
build carries the kernels. `SGLANG_GGUF_MXFP4_NATIVE` is read one level up, in
`layers/quantization/gguf.mxfp4_native()`, which returns False before it ever
reaches the `hasattr` (`if os.environ.get("SGLANG_GGUF_MXFP4_NATIVE", "1")[:1]
== "0": return False`). So the lever flips the two DERIVED answers and leaves
the marker alone. An arm that expected all three to flip would have been red on
a correct build — the expectation was the defect, not the code.

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

---

## 7 — Continuation note (2026-08-04, branch `feat/398-mxfp4-native`)

Written so a successor can continue without this session's memory.

### What stands

Both stages of the original two-stage plan are **already merged** on
`integration/r3-probe-next2` (merge `08bde23da7`); do not rebuild them.

| piece | where | state |
|---|---|---|
| `block_mxfp4` struct, E8M0 + 32x fp4-e2m1 | `sgl-kernel/csrc/quantization/gguf/ggml-common.h` | done |
| `dequantize_block_mxfp4` | `.../dequantize.cuh` | done |
| `vec_dot_mxfp4_q8_1`, MMVQ | `.../vecdotq.cuh`, `.../mmvq.cuh` | done (stage 2) |
| MMQ tile (`MMQ_X/Y_MXFP4 = 4/32`, `NWARPS_MXFP4 = 4`) | `.../mmq.cuh` | done (stage 2) |
| MoE MMVQ / MMQ | `.../moe_vec.cuh`, `.../moe.cuh` | done (stage 2) |
| capability marker `ggml_mxfp4_native` | `.../common_extension.cc` | done |
| python gate `MXFP4_NATIVE` | `python/sglang/srt/layers/quantization/gguf.py:250-285` | done |
| repack hand-off (identity on native) | `python/sglang/srt/model_loader/gguf_mxfp4_repack.py` | done |

The gate is a pure registration probe — `hasattr(torch.ops.sgl_kernel,
"ggml_mxfp4_native")` at `gguf.py:276`, never a call. That matters twice: it is
why #518's dispatch-key defect does **not** block MXFP4 (only ops that are
actually *called* with no tensor were affected), and it is why the probe must
never run before `sgl_kernel` is imported (see the correction at the top).

### What is next, in order

1. **Gate A on GPU** (§1) — 14 CUDA tests, both arches, in window #537. They
   are the only thing standing between "installed" and "proven". Currently
   `SKIPPED: no CUDA device` off-GPU.
2. **Gate B** (§2) — native vs repack on the two live IQ3_XXS MXFP4 tensors.
   The repack arm now needs `SGLANG_GGUF_MXFP4_NATIVE=0` explicitly.
3. **#512/#518 rebuild** — `TICKET_511_kernel_bundle_wheel.md`, still NOT
   BUILT. Preconditions were re-checked 2026-08-04 and are **not met**: the
   translator/DSV4F boot is live on all three cards and 5 processes map
   `sgl_kernel`. Neither defect blocks MXFP4 (#512 needs a >2 GiB per-rank
   per-layer expert tensor, out of reach at TP=3; #518 is worked around by the
   `_ggml_moe_get_block_size` python mirror), so this is not on the critical
   path for #398 — but the runbook §2.1 pin still names the pre-#512/#518
   wheel, and that pin is what a fresh install would restore.
4. Gates C-E (§3-§5) only after A and B are green.

### Reference values and fixtures that already exist

* `test/registered/unit/quantization/mxfp4_real_rows.npy` — real
  `blk.26.ffn_down_exps` bytes; the block layout was taken from the actual GGUF
  file, not from documentation (the #372 lesson), and `test_real_tensor_rows`
  gates the kernel against a host decode of exactly those bytes.
* Host reference decoder + `e8m0_to_fp32_half` live in
  `test_gguf_mxfp4_native.py`; the dequant gates are **exact**, zero tolerance —
  a table lookup times a power of two has no rounding to hide behind.
* Derived (not inherited) MMVQ/MMQ tolerances: `activation_quant_sigma()` at
  `SIGMA_MULTIPLIER=8`, from #511. The old atol=1.5/rtol=3e1 predicate accepted
  an all-zero and a sign-flipped output; two tests now execute that refuted
  predicate so the regression cannot come back quietly.
* Byte accounting, already pinned by a test: MXFP4 17 B/32 values (4.250 bpw)
  vs Q5_0 repack 22 B (5.500 bpw) — repack reads 29.4 % more, native 22.7 %
  fewer.

### Build state

No rebuild was performed or needed on this branch: the change is test-only, and
the installed wheel (sha `67f03cfa`) already carries every #398 kernel. Hermetic
results, `CUDA_VISIBLE_DEVICES=99`, `PYTHONPATH=<worktree>/python`:
`test_gguf_mxfp4_native.py` 16 passed (was: interpreter abort);
`test_gguf_mxfp4_cuda.py` + `test_gguf_mxfp4_dsv4f_moe_479.py` +
`test_gguf_mxfp4_bridge.py` 30 passed, 24 subtests, 14 skipped (all 14 are the
GPU-only Gate-A cases). `ruff` and `codespell` clean on the changed file.
