# NOTE 440 — the C4 indexer head fold: refused, with the derivation it omits

**Verdict: not adopted on any path in this fork. The identity it rests on is
not the operator this fork computes.** The head axis of the DSpark (C4) indexer
is irreducible, and the reason is one term the upstream derivation does not
mention. This note records the check, the numbers, and the guard that keeps the
proposal from being adopted later by someone reading only the algebra.

Nothing in the production code changed. The deliverable is
`test/registered/unit/layers/attention/test_dsv4_indexer_head_fold_440.py`
(18 tests, 19 subtests, CPU-only) plus the two audits below.

## Source

sgl-project/sglang#33246, comment
[5159510149](https://github.com/sgl-project/sglang/issues/33246#issuecomment-5159510149)
(Hakureirm, 2026-08-02) and its follow-up
[5159716067](https://github.com/sgl-project/sglang/issues/33246#issuecomment-5159716067).
The implementation is **sgl-project/sglang#33271** (open, Apache-2.0).

The derivation is public and the credit is due: the linearity argument, the MQA
observation, the cost table, and the two implementation traps recorded below all
come from that thread. What follows is the check against this fork's code and
against that PR's diff, not a dispute of the reasoning as stated.

## The proposal

For an indexer whose logits are

```
logits[i, j] = sum_h w[i, h] * (sum_d q[i, h, d] * k[j, d]) * k_scale[j]
```

`w` is indexed `(row, head)` only and `k[j, d]` carries no head index, because
the C4 indexer is MQA — one shared KV. So the head sum moves inside the dot
product and one folded query replaces the per-head stage:

```
q_eff[i, :] = sum_h w[i, h] * q[i, h, :]        cost S*H*D
logits[i, j] = (q_eff[i, :] . k[j, :]) * k_scale[j]   cost S*KV*D
```

The quadratic term loses the `index_n_heads = 64` factor. Upstream measured
37x / 78x / 99x at KV 16K / 32K / 64K on 8xA800, Triton fold vs per-head torch.

## Why it does not apply here

**The C4 indexer applies a per-head ReLU between the dot product and the
weighted head sum.** The operator is

```
logits[i, j] = sum_h w[i, h] * relu(sum_d q[i, h, d] * k[j, d]) * k_scale[j]
```

`relu` is not linear, so the head sum cannot cross it. This is not a property of
this fork's implementation — it is in every implementation of the operator that
could be read, upstream included:

| implementation | file:line | ReLU |
|---|---|---|
| torch, production path since #417 Cut 3 | `python/sglang/srt/layers/attention/dsv4/indexer.py:349` | `F.relu(score)` before `* weight_row`, `sum(dim=2)` |
| torch, reference twin | `python/sglang/srt/layers/attention/dsv4/indexer.py:178` | `F.relu(scores)` before `* weight`, `sum(dim=2)` |
| DeepGEMM reference used by our kernel test | `test/registered/kernels/test_deepgemm_paged_mqa_logits.py:63` | `torch.relu(qk)` before the weighted head sum |
| TileLang | `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py:245` | `relu(fp32 logits) * q_s -> logits_sum` |
| CuTe DSL SM100 | `python/sglang/jit_kernel/cutedsl_fp8_paged_mqa_logits.py:180-184`, `1478`, `1541` | `relu2_fma_f32x2`, ReLU folded into the per-head FMA accumulation |
| upstream `main`, both torch paths | upstream `.../dsv4/indexer.py:111`, `:243` | `F.relu` in the same position |

The MQA premise the upstream comment calls load-bearing **is** satisfied here —
the index cache is `[pages, 64, 1, head_dim + 4]`, one shared K per position
(`indexer.py:310`, `:859`), and `weight` is `[batch, num_heads]`
(`indexer.py:311`). MQA is necessary for the fold and it is present; it is just
not sufficient, and the condition that fails is linearity.

## Applicability verdict, per path

| path | file:line | verdict |
|---|---|---|
| torch `_sm120` — the sm86 and sm120 production path since #417 Cut 3 | `indexer.py:263-364`, dispatched at `indexer.py:113` | **not applicable** — per-head ReLU at `:349` |
| torch reference twin `fp8_paged_mqa_logits_torch` | `indexer.py:120-203` | **not applicable** — per-head ReLU at `:178` |
| Triton twin of the logits | — | **does not exist in this fork.** The only `triton.jit` in `dsv4/indexer.py` is `_fused_scale_kernel` (`:461`), which scales `weight * q_scale` and computes no logits. Upstream carries an XPU `fp8_paged_mqa_logits_triton` branch that this fork's `select_paged_mqa_logits_fn` does not have. |
| non-paged `deep_gemm.fp8_mqa_logits` | `indexer.py:709-737` | **not applicable** — DeepGEMM kernel, and it relus internally like every other implementation |
| TileLang / AITER / FP4 | opt-in backends | **not applicable** — same operator |

## Measured, on this fork's own functions

CPU float32, synthetic tensors in the production cache layout, 64 heads,
`fp8_paged_mqa_logits_torch_sm120` called directly.

| comparison | max abs | relative |
|---|---|---|
| folded vs the **linear** operator (ReLU removed) | 2.29e-05 | **3.59e-07** |
| folded vs the **actual** operator (ReLU present) | 94.97 | **0.942** |

The algebra is right — against a relu-free operator the fold is exact to fp32
rounding. Against the operator this fork runs it is a different function.
49.7% of the per-head products on this fixture are negative, i.e. the ReLU is
not a corner case here; it discards about half the terms.

Top-k overlap against the production path, the metric that actually matters
because these logits exist only to select pages:

| KV | selection | ratio | overlap with the fold |
|---|---|---|---|
| 4096 | top-512 | 12.5% | 0.549 |
| 4096 | top-64 | 1.6% | 0.563 |
| 4096 | top-32 | 0.8% | 0.406 |
| 16384 | top-512 | 3.1% | 0.525 |

Moving the ReLU to *after* the fold (`relu(q_eff . k)`) changes nothing —
0.549 / 0.563 at the same two ratios — as expected, since ReLU is monotone and
`k_scale > 0`, so a trailing ReLU cannot reorder a top-k.

## The reference implementation confirms it at file level

The fold is implemented in **sgl-project/sglang#33271** (Hakureirm, open,
6 files, +593/-170): "Make DeepSeek-V4 serve on SM80: compile, dispatch, kernel,
and the indexer's oversized intermediate". Reading the diff removes the
inference from the argument above — the folded kernel does not relu.

`_folded_paged_logits`, the whole of its arithmetic:

```python
acc = _tl.dot(q, _tl.trans(k))
acc = _tl.where(offs_k[None, :] < sl[:, None], acc, float("-inf"))
_tl.store(o_ptr, acc, mask=o_mask)
```

`q` is the folded `q_eff` (`einsum("bhd,bh->bd")` in the wrapper), `k` carries
`k_scale` pre-multiplied (`kk = (kdat * ksc[:, None]).to(bfloat16)`). There is
no ReLU, and there cannot be one: the per-head products the ReLU would clamp are
never formed.

The same PR keeps the per-head path as its fallback, twenty lines below, and
that fallback still relus:

```python
_sc = torch.bmm(_kvv, q.transpose(1, 2))
_sc = F.relu(_sc)
_sc = _sc * weight.unsqueeze(1)
_sc = _sc.sum(dim=2) * _kvs
```

So within one file the fast path and the slow path compute different functions,
and the fast path is selected by default on every SM8x card
(`SGLANG_DSV4_TRITON_FOLDED_INDEXER`, defaulting to `1` when
`get_device_capability()[0] == 8`). This is a correctness observation about an
open upstream PR, read from its diff, not an inference from ours.

## The open question this leaves, stated honestly

Upstream reports **0.9967** top-k overlap at a 0.8% selection ratio and needle
4/4 at 32K/128K/256K on a real long-context task. Against a relu'd operator on
random tensors we measure **0.41**. Both cannot describe the same comparison,
and now that the kernel is readable, only two explanations remain:

1. the offline harness's per-head baseline also omitted the ReLU, so the 0.9967
   compares two relu-free forms and agrees with our 3.59e-07 row; or
2. on the real DeepSeek-V4-Flash checkpoint the per-head products are
   predominantly positive, the ReLU is nearly inert on trained tensors, and the
   fold is a good *approximation* in practice.

The needle 4/4 at 256K is real evidence for (2) — it was produced by the actual
relu-free kernel on a real checkpoint — but needle retrieval is a weak
instrument and 4/4 is not a precision figure.

If (2) holds the fold could still be worth having — but as a data-dependent
approximation with no error bound, not as the exact restatement the derivation
claims. That distinction is the whole of this fork's position: an identity that
holds only when the data cooperates is not an identity, and shipping it as one
would put an unbounded, silent, input-dependent error on the page-selection path
of every long-context prefill. Deciding between (1) and (2) is a measurement,
and it is the GPU arm below.

## What IS worth adopting from #33271 — and is not the fold

The same diff carries an optimisation that preserves the ReLU exactly and is
independent of the fold: **gather the KV once when every query row shares a page
table.** Single-sequence chunked prefill has an identical `page_table` on all
`B` query rows, so the per-row gather this fork does at `indexer.py:336`
(`kvcache_flat[page_ids[:, page_start:page_stop]]`) copies the same pages `B`
times. Upstream's `_shared` branch gathers once and reshapes the `bmm` into a
single `[n, D] @ [D, B*NH]` GEMM, keeping `relu` -> `* weight` -> `sum(dim=2)`
untouched. At `B = 8192` with a 256K context that is the difference between one
copy of the KV span and 8192 of them.

This is the recommended next slice, and it is deliberately **not** taken here:

- it changes the production prefill path, and the GEMM reshape is not obviously
  bit-identical to the per-row `bmm` (same values, different accumulation
  tiling), so it has to be checked against the #425 golden pins before it lands;
- it needs the same page-table uniformity check, hence the same D2H sync, hence
  the same capture guard — the trap upstream hit and documented;
- its payoff is memory traffic, which cannot be measured at a desk.

Landing it inside a task whose verdict is "refuse the fold" would mix a
validated refusal with an unvalidated production change. It belongs to the GPU
arm, with the uniformity check and capture guard written from
`indexer.py:604`/`:622`-style structural conditions where possible rather than
from a runtime probe.

## Collision check against this fork's open upstream PRs

- **#33266** (ours, open, "Fix DeepSeek DSA top-k v2 defaulting on below SM90"):
  **no collision.** It adds the missing pre-Hopper arm for
  `SGLANG_OPT_USE_TOPK_V2` in `server_args.py`, because
  `topk_transform_512_v2` uses `cg::this_cluster()`. #33271 touches neither
  `server_args.py` nor the top-k kernel — the two are disjoint in files and in
  blocker. Thematic overlap only ("make DSV4 serve on SM8x"); both are needed.
- **#33272** (Hakureirm, closed **unmerged**, "Classify packed DeepSeek layers by
  what was built, not by quantization name"): folded into #33271, which carries
  the same change to `deepseek_v2.py`. **This fork carries the unfixed pattern**
  at `python/sglang/srt/models/deepseek_v2.py:846-849` and `:1880` — the
  `{"awq", "awq_marlin", "moe_wna16"}` name match, which misses gptq and
  auto_round and then reads `.weight.dtype` on a layer it failed to classify.
  Latent `AttributeError` at init for a packed non-AWQ DSV4 checkpoint. Reported
  here for orientation; out of scope for #440, worth its own task.

## Item 4 — the `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` coupling

**Verdict: the coupling exists textually and is already defanged. No change.**

Upstream `_can_use_nonpaged_indexer` disables the non-paged DeepGEMM fast path
whenever `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` is set (upstream
`.../dsv4/indexer.py:506`, inside the block at `:490-508`), with **no**
architecture guard ahead of it. That is the coupling worth breaking upstream:
on Ampere that env was the only way to reach a working paged path, so asking for
it also cost the non-paged prefill route.

This fork carries the same clause at `indexer.py:612` — but #417 Cut 3 inserted
an architecture guard **before** it, at `indexer.py:596`:

```python
if not deepgemm_indexer_supported():
    return False
```

The guard upstream's decoupling would need is therefore already in place, and it
runs first. Consequences, all pinned by tests:

- an sm8x/sm12x rank is refused the non-paged branch **architecturally**, with
  the env at either setting, so it can never be handed a route into
  `deep_gemm.fp8_mqa_logits` (which has no torch twin and would reject the
  card);
- the env clause can only bite on a card where DeepGEMM exists (sm90/sm100);
- **nobody has to set the env on a card without DeepGEMM any more.**
  `resolve_paged_mqa_logits_backend` (`indexer_arch.py:68-83`) picks
  `BACKEND_TORCH` from the capability. The upstream coupling has no victim here
  because the workaround that created the victim was removed.

What remains is `indexer_arch.py`'s documented doctrine — an explicit env
selection is a statement about the launch, not a probe (`indexer_arch.py:71-73`)
— so "do not use DeepGEMM for indexer logits" is honoured on both routes. That
is deliberate and stays.

## Items 1 and 2 — the two traps, audited against this fork

Upstream's folded kernel reuses row 0's page table for a whole tile to save
registers; correct for single-sequence chunked prefill, silently wrong KV for a
batch of different requests. The uniformity check added to guard it was a
device-to-host sync and invalidated CUDA graph capture on first call
(`cudaErrorStreamCaptureInvalidated`, server dead at startup).

This fork makes the same row-0 assumption, in a different place:

- `indexer.py:690` — `request_page_table = page_table[:1].contiguous()`, the
  non-paged plan built from row 0 alone.

It is guarded **structurally**, not by a runtime probe, which is why neither
trap is reachable:

- `indexer.py:604` — `forward_batch.batch_size != 1` refuses any multi-request
  batch before the plan is built;
- `indexer.py:622` — `return not torch.cuda.is_current_stream_capturing()`
  refuses the branch during capture outright;
- `indexer.py:655-660` — `to_cpu_int_list` returns `None` for any tensor whose
  device is not CPU, so the plan bails instead of pulling metadata across.

No `.item()` on a device tensor runs anywhere in the eligibility check or the
plan builder. Pinned by `TestRowZeroPageTableReuseIsGuarded`, including a
`torch.Tensor.item` interception that asserts zero calls and a `meta`-device
falsifier that shows the bail is real.

## Tests

`test/registered/unit/layers/attention/test_dsv4_indexer_head_fold_440.py`,
18 tests / 19 subtests, `CUDA_VISIBLE_DEVICES=99`, ~11 s.

| class | pins |
|---|---|
| `TestFoldAlgebraIsCorrect` | the fold reproduces a relu-free operator to fp32 rounding; the cost claim in the shapes each form contracts |
| `TestMqaPremiseHolds` | shared K per position, `(row, head)` weights, and a per-head K cache is *rejected* rather than silently reduced |
| `TestOperatorIsNotLinearInTheHeadAxis` | both torch paths match the per-head definition; the fold is outside any dtype tolerance (>0.1 relative); the top-k it feeds is lost |
| `TestReluIsTheWholeDifference` | the can-fail — neuter `F.relu` and the fold becomes exact again; plus a guard-the-guard that the fixture really has negative products |
| `TestNonPagedCouplingIsAlreadyArchGuarded` | the arch guard decides ahead of the env clause, on both sides |
| `TestRowZeroPageTableReuseIsGuarded` | batch>1 and capture never reach the row-0 plan; no D2H sync; meta-device metadata bails |

Can-fail, both executed:

- **remove `F.relu` from `indexer.py:349`** → 4 tests fail, and the folded-vs-
  actual relative divergence collapses from 0.942 to 3.59e-07. The guard is
  driven by exactly the term the derivation omits, and it fires the day the ReLU
  disappears from the production path.
- **break the fold order** (drop the weights from `q_eff`) → 2 tests fail.

Neither the #425 golden pins nor any production code were touched.

## GPU measurement arm (spec — not run; desk task, `CUDA_VISIBLE_DEVICES=99`)

The arm is **not** a speed arm. Measuring the speed of a wrong operator answers
nothing. The question worth GPU time is the one above: is the ReLU inert on the
real checkpoint?

**A. Sign statistics of the real per-head products.** Decides everything else.

- Boot DeepSeek-V4-Flash on the rig, same configuration as
  `docs/dev/BENCH_394_v4flash_club3090.md` (read the boot log for the flags;
  do not reconstruct them from memory). Hold `/spinning/gpu-arb/` with a
  heartbeat, stop the heartbeat before releasing.
- Instrument `fp8_paged_mqa_logits_torch_sm120` behind a debug env, recording
  per call, per DSpark layer (40-42):
  - `frac_neg` — fraction of `score < 0` before the ReLU;
  - `mass_clipped` — `sum(w * relu(s)) / sum(w * s)` over the finite span, the
    share of the weighted signal the ReLU actually removes;
  - the top-k overlap of `_folded_reference` against the real output at the
    production `index_topk = 512`.
- Prefill-heavy, at 32K / 128K / 256K, real long-context prompts, not random
  tokens — the sign statistics are the property under test and synthetic input
  is exactly what cannot answer it.
- **Decision rule, fixed in advance.** `frac_neg < 0.02` and overlap `>= 0.99`
  at every layer and every length → explanation (2) holds, reopen the fold as an
  explicitly-labelled approximation behind an off-by-default flag, with the
  error bound stated as "none, data-dependent". Anything else → the refusal in
  this note is final and no speed arm follows.

**B. Speed arm, conditional on A passing.** Only then, and only against the
per-head path in the same boot:

- ms/prefill-round per rank, not tok/s, per the standing measurement rule.
- Same-boot A-vs-A floor first: run the per-head path twice and report its own
  spread before any A/B delta. Upstream saw 157 s vs 275 s on two passes over
  the same lengths and declined to quote a number — that variance is the floor
  this arm has to beat before it may claim anything.
- Warm-up discipline: first-call Triton/`torch.compile` compilation is the
  named suspect for that spread. Discard the first two iterations at every
  shape, and record compile time separately rather than inside the measurement.
- The rig's 3080s (sm86) and the 5090 (sm120) all take the torch path, so this
  is the production path on every card here, not a fallback — which is the same
  reason #426 chunked it.

## Related upstream work

- **#33246** — the issue: the `[B, S, H]` bmm product OOMs at long context.
  This fork already carries the sequence-axis chunk fix (#426,
  `indexer.py:329-358`).
- **#33259, #33288** — open, bound the logits *allocation* (#33288 ports the DSA
  budget detection plus query-axis chunking to DSV4 and routes to the varlen
  non-paged path). Complementary to the fold, which attacks the constant rather
  than the allocation, and unaffected by this refusal. Noted for orientation
  only; no action here.
