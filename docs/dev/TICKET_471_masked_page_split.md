# TICKET #471 — masked SM120 SWA page-split: the GPU numbers

Status: **PORTED AND PINNED AT THE DESK, NOT MEASURED HERE.** No
`/spinning/gpu-arb/` window was taken; no SM120 launch of the ported kernel has
happened in this fork. Nothing below the "Recipe" heading may be cited as a
measurement until it is filled in from a real run.

## What was ported

Upstream sglang **#32320**, commit `204e0fbac0` ("[SM120] Only split touched
SWA pages in FlashMLA page-split kernel"), merged upstream. Upstream's file
lives at `python/sglang/kernels/ops/attention/flash_mla_sm120.py`; this fork's
copy is `python/sglang/srt/layers/attention/flash_mla_sm120.py`, so this is a
port, not a cherry-pick — the surrounding module has diverged (per-call arch
dispatch via `flash_mla_arch`, the `sparse_mla_sm120_decode_dsv4` entry point).

The page-split region itself had NOT diverged: it was byte-identical to
upstream's pre-#32320 state, and after the port it is byte-identical to
upstream's post-#32320 state (verified by diffing the region against
`204e0fbac0^` and `204e0fbac0`). The only fork-specific adaptation is in
`_flash_mla_flashinfer`, where `idx` is now computed before the split so it can
be passed as `touched_indices`.

Mechanism: a persistent per-device int8 mask, one byte per source page, zeroed
per call; `_page_mark_kernel` sets `mask[token // src_pbs] = 1` for every valid
token index (-1 skipped); `_page_split_kernel` returns early for an unmarked
page. Untouched destination pages keep their stale bytes, which is sound
because the caller only ever reads pages the same indices address.

## What upstream measured (upstream's numbers, not ours)

`_page_split_kernel` 16.8 % -> 2.2 % of GPU time; median ITL -20 %; TPOT -18 %
on DeepSeek-V4-Flash; `max_abs_diff` 0.0; CUDA-graph safe. These are quoted
from the upstream PR and have **not** been reproduced on this rig.

## What is pinned here

* `test/registered/unit/layers/attention/test_flash_mla_page_split_mask_471.py`
  — 10 hermetic tests that execute the REAL `_page_mark_kernel` and
  `_page_split_kernel` through Triton's interpreter (`TRITON_INTERPRET=1`) on
  CPU tensors. Coverage (every named token's destination page is byte-identical
  to the unmasked split), fewer splits (the mask is exactly
  `{token // src_pbs}`), the persistent-buffer contract (untouched pages and
  alignment tails keep their bytes), mask zeroing between steps, int64 indices,
  the empty-index fallback, and neutrality with `touched_indices=None`.
  Four executed can-fail arms:

  | mutation | result |
  |---|---|
  | `_page_split_kernel` ignores the mask | 3 red |
  | `_page_mark_kernel` marks at `// 64` instead of `// src_pbs` | 5 red |
  | `mask.zero_()` removed | 6 red |
  | `use_mask` forced False (pre-port behaviour) | 7 red |

* `test/registered/kernels/test_flash_mla_backends.py::TestTouchedPageSplit` —
  upstream's SM120 test, ported to this tree's module paths. Skipped without an
  SM120 card; it is the arm that proves the compiled kernel does what the
  interpreted one does.

## Recipe for the measurement window (what this ticket still owes)

Card: the 5090 (sm120) — resolve its index through the IdentityMap, never by
assumption (catalog §11). Hold `/spinning/gpu-arb/` for the duration.

1. **Confirm the path is live.** The masked split only runs when the FlashMLA
   fallback resolves to `flashinfer` and the SWA pool page size is not already
   64. Read the resolved backend at boot before trusting any delta.
2. **A-vs-A floor first, per arm, same boot** (catalog §10 canon;
   `s12_prefill_kurve.py --floor-draws N` runs the draws back to back in one
   process, §16). A number below its own floor is not reported.
3. **Arms.** Pre-port and post-port are two different builds, so they are two
   boots: interleave them across the window rather than running A then B.
   The pre-port arm is reproducible in-tree by forcing `use_mask = False` in
   `_split_kv_pages_to_64` — that is exactly can-fail arm D above, so the
   control arm is a one-line, test-covered change rather than a rebuild.
4. **Numbers to record**, per rank, per arm: ms/verify and ms/prefill (never
   tok/s — memory "ms/Runde als Messlatte"), plus the `_page_split_kernel`
   share of GPU time from a profile, which is the term upstream's 16.8 -> 2.2 %
   claim is about. Whole-model ITL/TPOT deltas are secondary and are only
   meaningful against the same-boot floor.
5. **Every boot carries `--enable-metrics`** (CLAUDE.md, no exceptions).
6. **Correctness in the window**: greedy decode, same prompt, both arms; the
   texts must be identical. Upstream reported `max_abs_diff` 0.0; on this rig
   the equivalent gate is the byte gate on a short output (catalog §10 — GDN
   prefill beyond ~109 tokens is upstream-nondeterministic, so keep the gate
   output short).

```
A-vs-A floor, masked arm   : PENDING
A-vs-A floor, unmasked arm : PENDING
ms/verify delta            : PENDING
_page_split_kernel share   : PENDING
byte gate (short greedy)   : PENDING
```

## Risk the desk cannot close

The interpreter executes the kernel's arithmetic, not its concurrency. Two
programs marking the same page byte concurrently is the one property only a
real launch exercises; upstream's comment argues it is safe because both stores
write the same value 1. That argument is sound for a byte-sized store, but it
is an argument, not a measurement — the SM120 arm above is what settles it.
