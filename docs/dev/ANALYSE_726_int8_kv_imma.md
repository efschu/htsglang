# ANALYSE 726 — INT8 KV with IMMA QK

Desk only, 2026-08-17. No boot, no GPU, no model load. **No kernel designed**
and none proposed for build here; §5 says why, and it is not the reason the
brief expected.

## 0 — Three premise corrections, before anything is designed on top of them

**(a) There is no int8 KV dtype in this fork, "tiering to Triton" or
otherwise.** `--kv-cache-dtype` admits `auto | fp8_e5m2 | fp8_e4m3 | bf16 |
bfloat16 | fp4_e2m1` (`server_args.py:1008`). The harvest says the same
("Ours: **no INT8 KV at all**", ANALYSE_NINFER §3.2) and so does the plan:
*"int8 KV is unbuilt in our fork … zero int8-KV code tree-wide"*
(`PLAN_PERF_PIPELINE_2026-08-16.md` §(e)).

**(b) "int8-PTH" is the name of a PROPOSAL, not of a feature.** It is the
title of `docs/ANALYSE_489_int8_kv.md` @ `272b408980` — *"int8-PTH KV cache
dtype as a performance candidate"*, status *"survey → evaluation, desk only.
No build, no GPU, no load."* Nothing tiers to Triton in our tree because
nothing exists to tier.

**(c) The Triton inversion is #489's DECLINE REASON, and it is external.**
The published shape is +76–81% decode at short context but **−72% at 58K**,
because choosing int8 selects a *backend tier*: fp8 keeps FlashInfer, int8
falls to Triton-only. Our rig runs `attention_backend='flashinfer'` on the
text path, and our 327,680-token regime sits **~5.6× past** the 58K inversion.
#489 therefore closed it as **evaluated-and-declined**.

So this ticket is not a fresh design problem. It is a **re-open request against
a documented decline**, and the only question that matters is whether the new
evidence bears on the reason for that decline.

## 1 — It does, and this is the whole substance of #726

#489 declined on the assumption that *int8 ⇒ Triton ⇒ slow*. The harvest
supplies the fact that assumption did not have:

- QK runs on **native `m16n8k32.s8` IMMA tensor cores**. K stays int8 in cache
  and is read straight into shared memory **with no dequant at all**; the int32
  MMA output is rescaled per 64-group; only V is dequantized once to bf16
  (ANALYSE_NINFER §3.2, from `gqa_attention_decode_i8.cuh:6-14`).
- `m16n8k32.s8` is an **sm80-era** instruction. It was verified to assemble on
  our CUDA 12.9 for **both `sm_86` and `sm_120a`** (probe compiled to cubin, no
  GPU used).
- **Our 3080s have native IMMA and no fp8 tensor-core path whatsoever.**

That changes the comparison on two of three cards. It is not "fast fp8
FlashInfer versus slow Triton int8"; on sm_86 it is "fp8 dequantised to bf16
math" versus "native int8 tensor-core math with half the KV bytes". The
published −72% was measured against a Triton lane doing dequant-and-bf16, which
is precisely the lane IMMA-QK replaces.

**This does not overturn the decline.** It removes the ground the decline stood
on, which is exactly the condition #489 wrote for itself:

> *"Close as evaluated-and-declined for this rig, **unless the (c) microbench
> contradicts the published inversion on sm_86**."*

## 2 — The disconfirmer already exists and is unscheduled

#489 §(c) is a written ticket spec, not a plan to write one: kernel-level
microbench, no model load, context sweep 1K/8K/32K/58K/128K/327K at batch 1
and 4, A-vs-A first against the rig's 14.1% floor, log which backend each arm
actually selected rather than inferring it, and a stated **kill condition** —
*"if the 58K point reproduces the published inversion on sm_86, stop."*

The plan records it as *"the only cheap disconfirmer but **not scheduled**"*.
Before #726 it had no motivation; the IMMA finding is that motivation.

**The cheapest correct next step is to schedule #489 §(c), not to design a
kernel family.** Designing first would price a kernel against an inversion we
have not reproduced on our own silicon, on cards where the mechanism that
produced it does not obviously apply.

## 3 — What was BUILT here (hermetic, useful under either outcome)

`test/registered/unit/attention/test_int8_kv_codec_726.py` — 16 pins, 4.7 s,
no CUDA.

**The codec oracle.** Pure-torch transcription of NInfer's codec
(`gqa_attention_kv_quant.cuh:19-55`): symmetric, no zero point, per-token
per-64-group absmax, **fp16-stored** scale, RNE, clamp to ±127. The fp16
round-trip happens at quantize time, not at the end, because the kernel reads
an fp16 scale and a reference must quantize against the value the kernel will
use. Any IMMA-QK kernel needs something to be validated against, and
"validated against itself" is not a gate.

**Measured quantization error** (CPU-sampled inputs; the CUDA generator is not
bit-comparable across architectures):

| input | max rel. error | rel. RMS |
|---|---:|---:|
| normal | 0.38% | **0.59%** |
| heavy-tailed | 0.39% | **1.24%** |
| with a 1000× outlier in one group | 0.39% | 0.60% |

Two readings worth carrying into the quality gate: group-64 **does** isolate
outliers (the 1000× case leaves RMS unmoved, which is the design's claim), and
**heavy-tailed activations roughly double the RMS error** — that is the
distribution shape the gate must actually test, not Gaussian noise.

**The 1.94× footprint, as arithmetic rather than a quoted number.** At
head_dim 256: 256 codes + 4×2 B fp16 scales = **264 B**/token/KV-head/plane
against 512 B in bf16 → **1.939×**. Pinned so a later layout change that
doubles the scale cost cannot keep claiming 1.94×.

## 4 — The key-collision surface the brief expected: ALREADY CLOSED

The brief asked where the HiCache keys "must learn the new dtype". They
already have. `compute_model_identity_hash` (`mem_cache/hicache_storage.py:59`)
folds `str(server_args.kv_cache_dtype or "auto").lower()` into the identity,
and its docstring says why in the #241/#513 terms exactly:

> *"a later run that shares the served_model_name and storage location but
> differs in e.g. `--kv-cache-dtype` would silently read pages written in
> another byte format. Incorporating this hash into the key suffix turns that
> silent wrong hit into a clean miss."*

So adding an `int8` value to the dtype choices separates its pages
automatically; **no key work is required**. Pinned anyway
(`TestTheDtypeKeyAlreadySeparatesFormats`): int8 vs fp8 hashes differ, all six
dtypes are mutually distinct, and a source pin names where the separation
comes from so a behavioural pass cannot be produced by an incidental salt.

One caveat, unresolved: the hash is optional
(`model_identity_hash: Optional[str] = None`, "None keeps legacy key layout").
A deployment on the legacy layout has no dtype in its key. Whether any of our
tiers run legacy is **NOT ESTABLISHED** here.

## 5 — Verdict

**No kernel design, and no build proposed.** Not because the idea is weak — of
every item in the harvest this is the only one that helps all three cards — but
because the next honest step is cheaper than a design and could close the
ticket outright.

The order is: schedule #489 §(c) on sm_86 → if the inversion reproduces, #489's
kill condition fires and #726 closes on our own silicon; if it does not, the
decline's ground is gone and a design pass is then worth its cost, with the
codec oracle and the footprint arithmetic already in hand.

Writing ANALYSE_726 as a from-scratch design would also have duplicated #489,
which already carries the support survey, the inversion pricing for this rig,
the per-rank hetero twist, and the three-part lossy quality gate.

## 6 — Hermetic vs GPU-window, as asked

**Hermetic and DONE:** codec oracle, round-trip error pricing, scale-layout
pins, footprint arithmetic, dtype-key collision pins.

**Needs a GPU window:** #489 §(c) microbench (kernel-level, no model load,
rides any window under a gpu-arb claim) — and it is the ONLY window item
required before the reopen decision.

**Needs a window only if the reopen succeeds:** the three-part quality gate
(lossy; needs a model, and per §3 must include heavy-tailed activations, not
Gaussian), and any kernel bring-up.

## 7 — What this note does not establish

Which attention backends could carry an IMMA-QK variant, and what the pool
layout change costs in code, are **not** mapped here — a delegated map was
still running when the verdict became clear, and the verdict does not depend on
it: if #489 §(c) fires its kill condition, that map is work nobody needs. It is
the first thing to produce if the microbench comes back favourable.

Whether V should stay int8 (NInfer keeps it int8 in cache and dequantizes once
to bf16, since PV is not a key-contracted int8 accumulation) versus K-only
quantization is a real quality/bandwidth fork and is **not** decided here.

---

## 8 — Backend and pool map (folded in after §7 was written)

The delegated map completed after the verdict above and independently
reproduced all three premise corrections in §0 — including the zero-hits grep
for int8 KV and the absence of `int8_pth` anywhere in the fork. It also
supplied four things that change what a build would have to do. The two
load-bearing ones I verified directly rather than adopting on trust.

**(1) Today's fp8 KV scale mechanism is hard-coded to PER-TENSOR, and says so.**
`layers/quantization/kv_cache.py:76-79`:

```python
if not isinstance(k_scale, float) or not isinstance(v_scale, float):
    raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")
```

So a group-64 per-token scale layout is not an extension of the fp8 path; that
path actively refuses anything finer than one float per layer. **VERIFIED.**

**(2) The one existing group-scale precedent dequantizes EAGERLY — which is
exactly what this design refuses.** `MHATokenToKVPoolFP4`
(`mem_cache/memory_pool.py:3302-3350`) keeps a separate per-layer
`k_scale_buffer` with one scale per 16-element block — structurally the right
shape to imitate — but `_get_key_buffer` then calls
`BlockFP4KVQuantizeUtil.batched_dequantize(...)`, materialising full precision
before any backend sees it. **VERIFIED.**

That is the trap. Copying the fp4 pattern would deliver the **VRAM** saving and
NOT the **bandwidth** saving, and bandwidth is the entire point of IMMA-QK —
NInfer explicitly refuse a standalone quant/dequant kernel because it "would
defeat the halved-bandwidth goal" (`gqa_attention_kv_quant.cuh:5-8`). A build
that reused the fp4 machinery would look like it had landed the feature while
delivering half of it.

**(3) The closest carrier is the grouped Triton kernel, and its scale handling
is the gap.** `_fwd_grouped_kernel_stage1`
(`python/sglang/kernels/ops/attention/decode_attention.py:352+`) already casts
Q to K's dtype and does `qk = tl.dot(q_k, k)` — a real `tl.dot`, which Triton
can lower to tensor-core MMA. But `k_scale` is folded into `sm_scale` as a
**single scalar before the kernel is entered** (`:899-969`), so there is no
per-token/per-group descale inside the kernel at all. The non-grouped MHA path
is further away still: it does `qk = tl.sum(q[None, :] * k, 1)`, not a dot.

**(4) IMMA is already in-tree, in the wrong place.**
`sgl-kernel/csrc/gemm/int8_gemm_kernel.cu` runs
`mma.sync.aligned.m16n8k32.s8` via CUTLASS for W8A8 **linear layers**, and is
reached from `layers/quantization/w8a8_int8.py` — never from the KV pool or any
attention backend. `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu` is a
group-size-templated per-token quant kernel and is a plausible reusable
primitive, also not applied to KV today.

Read together: the instruction is proven present and callable in this tree, and
the arithmetic shape of a group-64 scale buffer has a precedent. What does not
exist is a fused read path — and the fp4 precedent is a worked example of
solving the layout while missing the point.

## 9 — Correction to §4: the collision surface is closed only for the DTYPE STRING

§4 said the key-collision surface is already closed. That is right for the
question asked and **incomplete** for the question a build would raise.

`compute_model_identity_hash` hashes the dtype **string**, not the byte layout
within it. Once `int8` exists as a value, a later change to its internal layout
— group-64 → group-128, or an fp16 scale becoming bf16 — keeps the same
`--kv-cache-dtype=int8` string and therefore the same identity hash, while the
pages on disk mean something different. Persistent tiers outlive the process,
so that is a silent wrong hit of exactly the kind this hash was built to
prevent, and exactly the #513 class it already documents for uneven-TP bytes.

**Named as a build requirement, not pinned.** A pin asserting that two int8
layouts produce different keys cannot fail today, because neither layout
exists; writing one would encode a green state that does not exist, which this
strand has refused twice already. The requirement is: if int8 KV is built, its
layout parameters (group size, scale dtype) must enter the identity, not just
the dtype name.

This does not change §5. It is one more item on the build's bill, and the build
is not the next step.
