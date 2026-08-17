# #700 — ReplaySSM: the wiring contradiction, the net count, and the identity arm

Desk halves of #700, following #325. Tree `/spinning/wt-602-slot2`, branch
`fix/602-fill-side`. **No GPU was touched.** Byte figures continue the table in
`NOTE_325_gdn_fusion_audit.md` §2 (Qwen3.8-27B-INT8 dims).

---

## 1 — The wiring contradiction: the KERNEL HEADER is stale

`layers/attention/fla/fused_recurrent_linear_replayssm.py:10-12` claims:

> This is a STANDALONE increment: kernel + wrapper only. It is NOT yet wired
> into the memory pool / radix cache / scheduler / backend dispatch.

**That is false as of this tree.** The wiring is present, deliberate, and
seam-aware:

| layer | evidence |
|---|---|
| ring allocation | `memory_pool.py:469-477` (documented layout), `:611-633` (allocated only under the flag; `None` otherwise so the legacy path stays byte-identical) |
| slot alloc | `memory_pool.py:1530-1531` — cursor reset on (re)alloc |
| slot copy | `memory_pool.py:941-968` — invariant documented, dst cursor reset, debug assert |
| flush / clear | `memory_pool.py:908-909`, and `:865-875` explains recycled slots must not leak a stale cursor |
| offload / restore | `memory_pool.py:1017-1018`, `:1049-1051`, `:1075-1076` — `write_pos` travels in the state blob |
| **radix donate** | `mamba_radix_cache.py:599-608` |
| unified cache | `unified_cache_components/mamba_component.py:409-417` (mirrors the above) |
| backend + CUDA graph | `hybrid_linear_attn_backend.py` (54 references, incl. per-bs static buffers and the cursor advance at `:139-145`) |
| dispatch | `gdn_backend.py:497-537` |
| tests | `test_mamba_checkpoint_interval.py::TestFlushResetsMambaPool` |

### 1.1 The dangerous seam is handled, not unwired

The specific hazard — a prefix-cache donate or copy mid-window, where the ring
holds updates for a state the pool is about to swap — is addressed head-on.
`mamba_radix_cache.py:600-608`, verbatim:

> ReplaySSM (no_buffer): `temporal[slot]` lags the live state by the slot's
> unflushed ring depth (`write_pos`), so cap the donate to the last flush
> boundary (where temporal is current) and reset the cursor, keeping the
> donated checkpoint consistent with its key length.

and `MambaPool.copy_from` (`memory_pool.py:944-951`) states the invariant and
enumerates every caller's compliance: COW copies radix checkpoints;
`cache_unfinished_req` copies an active slot only during prefill (ring empty);
`cache_finished_req` caps to the last flush boundary.

**So #700 does not close as refused-until-fixed.** There is no correctness hole
of the kind the ticket anticipated. The defect is the stale header, which cost
this audit a wrong "irreducible" claim in #325 revision 1.

### 1.2 Two real findings that remain

**(a) The `copy_from` invariant is unenforced in production.** The assert at
`memory_pool.py:952-960` is gated on `self.debug_memory_pool`, off by default.
The invariant therefore rests entirely on the caller discipline listed in a
docstring. A fourth caller added later violates it **silently** — a wrong-answer
mode, not a hang. This is the #624 drift class: a documented contract with no
live guard. Hardening candidate, not a bug today.

**(b) The pool tests cannot run hermetically.** `_build_tree`'s docstring says
"MambaRadixCache + pools **on CPU**", but construction reaches
`utils/common.py:1322`, which raises
`RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) ... is
available.` under `CUDA_VISIBLE_DEVICES=""`. This affects the **pre-existing**
`TestFlushResetsMambaPool` identically, so it is not introduced here. A second
comment that does not match its code.

### 1.3 What execution did and did not settle

The ticket asked to resolve the contradiction **by execution**. That was
attempted and is **blocked hermetically** by 1.2(b): the pool cannot be built
without an accelerator, so the pool/radix half of the claim is window-gated.
The determination above is therefore made by reading, and the executable proof
is committed and waiting for a GPU window:
`test_mamba_checkpoint_interval.py::TestReplaySsmRadixSeam` — four tests
covering ring allocation, dst-cursor reset, the can-fail proof that the
documented invariant really trips under the debug guard, and one that pins
1.2(a) as a fact: with the guard off, an un-flushed source is copied silently.
**These are not reported green; they have never run.**

## 2 — The net count left open in #325

Ring records, from `memory_pool.py:472-474` and `:611-633`. Note `replayssm_k`
is per **K**-head (16), not per V-head:

* `d`: `[layers, slots, HV, L, V]` ssm dtype -> 48 x 128 x 4 B = 24,576 B/step
* `k`: `[layers, slots, H,  L, K]` ssm dtype -> 16 x 128 x 4 B = 8,192 B/step
* `g`: `[layers, slots, HV, L]` fp32 -> 48 x 4 B = 192 B/step
* **append R = 32,960 B/step**; state `S = 3,145,728 B`

Per step per layer: `S0` is read every step, one record is appended, the buffer
is re-read on reconstruction (mean depth `(L-1)/2`), and the full state is
written once per `L`:

```
total(L) = S + R + ((L-1)/2)*R + S/L
```

| L | reconstruction | flush/step | chain total | saved vs baseline | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 3,145,728 | 6,640,000 | **-0.5 %** | 1.00x |
| 4 | 49,440 | 786,432 | 4,330,144 | 34.5 % | 1.53x |
| 8 | 115,360 | 393,216 | 4,002,848 | 39.4 % | 1.65x |
| **14** | 214,240 | 224,695 | 3,933,207 | **40.5 %** | 1.68x |
| **16 (default)** | 247,200 | 196,608 | 3,938,080 | **40.4 %** | **1.68x** |
| 24 | 379,040 | 131,072 | 4,004,384 | 39.4 % | 1.65x |
| 64 | 1,038,240 | 49,152 | 4,581,664 | 30.7 % | 1.44x |

**Answer: 40.4 % of the chain's bytes at the default L=16, a 1.68x reduction.**

Three things this count settles:

* It is **below** the header's "roughly halved" (47.6 % would be the
  write-elimination ceiling). The gap is the reconstruction re-read, which the
  header acknowledges qualitatively and prices at zero.
* The optimum is `L* = sqrt(2S/R) = 13.8`; **the default L=16 is within 0.1
  percentage points of optimal**, and the curve is flat from L=8 to L=24. The
  shipped default is well chosen and does not need tuning.
* `L=1` is **worse than baseline** (-0.5 %) — the expected sanity result, since
  flushing every step is the baseline plus ring overhead. A count that did not
  reproduce that would be wrong.

Against #325's stated >10 % gate this clears by **4x**, versus 0.99 % for the
best fusion. ReplaySSM, not fusion, is the lever on this chain.

## 3 — The byte-identity arm (window-gated)

**Purpose: measure divergence, not assert equality.** The flag text claims only
"numerically correct". The reconstruction sums buffered rank-1 updates in a
different order than the sequential update, so bit-identity should be assumed
**absent** until measured. The arm's job is to say how far off, and whether it
changes emitted tokens.

* **Inputs**: sampled on CPU and moved to device — `torch.randn` on-GPU is not
  arch-identical, so a GPU-sampled input cannot support an identity claim.
* **A-vs-A floor FIRST**: baseline against itself, back-to-back, warmup
  discarded. Until A-vs-A is bit-identical, no A-vs-B difference means anything.
* **Keep the probe SHORT.** GDN prefill is non-reproducible above ~109 tokens
  (upstream, pre-existing). A byte gate longer than that measures the known
  non-determinism, not ReplaySSM. Decode-only, short prompt.
* **Arms**: control `--enable-linear-replayssm` off; treatment on at default
  `--linear-replayssm-cache-len 16`. GDN scalar-gate model only — the flag text
  states KDA decode is *slower* than the packed baseline, so KDA is not an arm.
* **Report**: (1) bit-identity of emitted token ids; (2) if not identical, max
  abs/rel divergence of logits and the first divergent step; (3) only then
  ms/round split compute vs wait, at batch >= 64 where the help text places the
  win.
* **Expectation**: upstream `sgl-project/sglang#28511` reports ~2.3 % end-to-end
  TPOT on a MoE model. Our checkpoint is MoE, so plan against that, not the
  1.2-1.5x kernel figure.
* `layers/attention/fla/bench_gdn_replayssm_decode.py` already exists and must
  be read before any harness is written.

**Gate order is not negotiable:** #700 sits on the lossless queue, so the
identity result decides whether the perf arm is even worth running. If the
divergence changes emitted tokens, ReplaySSM is a lossy feature and goes last
by standing policy, whatever the 1.68x says.
