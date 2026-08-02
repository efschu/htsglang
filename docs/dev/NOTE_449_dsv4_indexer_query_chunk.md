# Task #449 — query-axis chunking for the DSV4 torch indexer logits

Adoption candidate **B** of `ANALYSE_447_llamacpp_dsv4_harvest.md` section 4
(lever **L3** of section 2.3). Desk task, no GPU window taken, `CUDA_VISIBLE_DEVICES=99`
throughout.

## 1 — The three claims, checked at the tree

### Claim 1 — `dsa/` has query-axis chunking, `dsv4/` chunks only the KV axis

**VERIFIED, with one correction to what "wire it in" can mean.**

`dsa/` has it: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py:1081-1096`
(`_should_chunk_mqa_logits`, a need-chunk decision plus a byte budget) and
`:1242-1299` (a `while start < q_offset` loop that slices `q_fp8`, `weights`,
`ks`, `ke` and runs `metadata.topk_transform` per chunk).

`dsv4/` did not: the only loop in `fp8_paged_mqa_logits_torch_sm120` walked KV
pages (`layers/attention/dsv4/indexer.py:331-358` at
`e65b75be1c`), with a single trailing top-k at `:905-931`. Every intermediate in
that loop was `batch_size` tall.

The correction: `dsa/`'s implementation is **not importable into `dsv4/`**, and
the difference is not stylistic.

* It is a method on the DSA indexer bound to `deep_gemm.fp8_mqa_logits`, the
  **ragged non-paged** op. `dsv4/`'s torch path is **paged** and gathers through
  a page table; there is no shared call signature.
* Its budget comes from `torch.cuda.mem_get_info`
  (`dsa_indexer.py:1074`), which the file's own comment records as a host
  synchronization, guarded by a `get_is_capture_mode()` early return at `:1068`.
  Putting a host sync on the `dsv4/` torch path is the trap `NOTE_440` recorded
  (`cudaErrorStreamCaptureInvalidated`, server dead at startup); the torch path
  is the production path on every card of this rig since #417 Cut 3.
* It drives a per-chunk `topk_transform`; `dsv4/`'s three top-k producers take
  the full `[B, S]` logits and its metadata is built for that shape.

So what is wired is the **mechanism** (a budgeted loop over the query axis)
under the **#395 budget discipline** the task asked for, not the `dsa/` code.
Nothing in `dsa/` was touched.

### Claim 2 — the B-fold duplicates the KV gather

**The duplication is VERIFIED. Eliminating it is REFUTED as a desk item and was
not attempted; it is BOUNDED instead.**

Verified: `layers/attention/dsv4/attn_metadata_kernels.py:309-311` (torch twin)
and `:352-358` (triton twin) build the page table as one row per query token, and
both fill row `i` from `req_to_token[req_pool_indices_repeated[i]]` — the same
slice for every row of one request, at full width, not truncated per row. So for
a single-sequence chunked prefill the rows are byte-identical, and the gather at
`indexer.py:336` (base) materialized `B` copies:
`2048 x 128 pages x 64 x 132 = 2 214 592 512` bytes = 2.06 GiB at B=2048 with the
default 8192-position KV chunk, matching ANALYSE_447's 2.2 GB. Pinned as a fact
by `TestTheDuplicationIsBoundedNotRemoved` (the fixture's page-table rows and the
gathered block are asserted equal to their own row 0).

Not eliminated, for three independently sufficient reasons:

1. **Uniformity is not guaranteed.** A batch of different requests has genuinely
   different rows. The structural guard that would license row-0 reuse is
   `batch_size == 1`, which the paged call site does not have; the runtime check
   is exactly the device-to-host uniformity probe `NOTE_440:255-282` records
   invalidating CUDA-graph capture.
2. **The bit-identity could not be established without a GPU.** Deduplicating
   means handing `torch.bmm` a batch-stride-0 expanded operand. Whether cuBLAS
   selects the same algorithm at a different batch count, and whether PyTorch
   even preserves stride 0 rather than calling `.contiguous()` internally (which
   would restore the B-fold silently while the test still passed), is a CUDA
   backend question. A CPU test would pin the wrong backend. Shipping that on a
   desk argument is the `desk-written-never-executed` failure mode.
3. **ANALYSE_447 itself files it separately** — candidate **C**, "medium-large",
   shared with the SWA path (`deepseek_v4_backend.py:1693-1726`) — from candidate
   **B**, which is what this task's own scope paragraph 1 describes. C stays open.

What #449 does instead is the bound ANALYSE_447 predicted for B: the duplication
factor of the gather drops from `B` to the budgeted row count, so the block stops
following the query count at all. Measured in the tests: 4.1 MiB -> 1 MiB at
B=64, and identical peaks at B=32 and B=64.

One genuinely bit-identical data-movement saving *was* taken inside the loop: the
gathered block is reinterpreted in place (`view(dtype=...)` on the whole
contiguous row, then slice) instead of copying each half out with `.contiguous()`
first. That removes two full-size intermediates per step — a `head_dim`-wide fp8
copy and a 4-byte-wide scale copy — leaving the fp32 dequantization as the only
copy. Byte reinterpretation and an elementwise cast, no arithmetic touched.

### Claim 3 — scope exclusions

Honoured. No fused indexer kernel (ANALYSE_447 L2 / candidate D). No head-fold
code touched — `test_dsv4_indexer_head_fold_440.py` runs unchanged and green
(18 tests). Nothing in `dsa/` changed; it is cited, not imported.

## 2 — What was wired

`python/sglang/srt/layers/attention/dsv4/indexer.py`

* `_gather_pages(kvcache_flat, page_ids)` — the duplicated gather, given a name
  so its size is observable from a test rather than only from a profiler.
* `_indexer_logits_step_bytes(chunk_seq, num_heads, head_dim)` — the transient
  bytes one query row costs in one `(query chunk x KV chunk)` step, enumerated
  buffer by buffer.
* `_indexer_logits_chunk_rows(...)` — MiB budget -> this rank's row count.
* `fp8_paged_mqa_logits_torch_sm120` — the KV-page loop is now the **inner** loop
  of a query-row loop. It composes with #426 rather than replacing it: the peak
  is bounded in both axes at once.

`python/sglang/srt/environ.py`

* `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB = EnvInt(2048)` — a per-rank MiB budget,
  0 disables (one pass over the query axis, the pre-#449 shape).

The reference twin `fp8_paged_mqa_logits_torch` is deliberately left single-pass
on both axes, as #426 left it: it is the oracle the byte gate compares against.

### Why MiB and not a row count (#395 discipline)

The bytes a query row costs scale with the head count and with the KV chunk
width, so a flat row knob means a different peak on every geometry — the same
argument `attn_scratch_token_threshold`
(`models/deepseek_common/attention_forward_methods/forward_mha.py:98-126`) makes
for the chunked-prefix threshold. The invariance contract is pinned generalized
(head counts 1..128 and KV chunk widths 512..8192), not as one instance: the
same budget must cap the same byte count on every geometry, tightly — one more
row must not fit.

A geometry note worth recording, because it cuts against the obvious assumption:
the C4 indexer's heads are **replicated, not TP-sharded**
(`indexer.py:980` sets `n_local_heads = n_heads`, and `wq_b` / `weights_proj` are
`ReplicatedLinear` at `:981-996`), unlike the main attention which goes through
`tp_partition_size` (`models/deepseek_v4.py:411`). So under uneven TP the head
term of the per-row cost is currently rank-invariant. The row count is not:
DP-attention shards the query axis per rank (see the comment at
`environ.py:1570-1572`). Chunk counts therefore do diverge between ranks, which
is why the collective audit below is load-bearing rather than decorative.

## 3 — Collective audit (Rank-lokaler-Test-vor-Kollektiv)

The chunk-width decision reads only its arguments and one env var; no group
state, no device tensor, no `.item()`, no `mem_get_info`. The loop body is torch
tensor ops only — `grep` for `torch.distributed|all_reduce|all_gather|broadcast|
barrier` over `dsv4/indexer.py` returns nothing; the only parallel-context read
in the file is `get_parallel().attn_cp_size` at `:749`, inside the *non-paged*
eligibility check, outside both loops, and it reads a config scalar.

Pinned by execution, not by inspection alone:
`TestChunkingIsRankLocalAndCollectiveFree` patches fourteen `torch.distributed`
entry points to raise and runs a chunked call; it patches `torch.Tensor.item`
and asserts zero calls; and two guard-the-guard tests fire both interceptions on
purpose so a silently-inert patch cannot pass for a clean audit. A fourth test
pins that ranks really can disagree on the chunk count (by query rows and by head
count), so the audit is not vacuously true.

## 4 — Tests

`test/registered/unit/layers/attention/test_dsv4_indexer_query_chunk_449.py`,
22 tests (plus subtests), CPU-only, `CUDA_VISIBLE_DEVICES=99`, ~0.3 s.

| class | pins |
|---|---|
| `TestQueryPeakIsBoundedByTheBudget` | gathered block inside the budget; peak does not follow the query count; bmm product bounded in both axes; can-fail arm with the budget off |
| `TestQueryChunkingIsExact` | atol=0/rtol=0 against the unchunked run, against the single-pass reference, across a budget x KV-chunk sweep, at a query count that no chunk width divides, and on a uniform page table — every case also asserting the run really chunked |
| `TestBudgetToRowsConversion` | disabled/oversized mean one pass; the invariance contract across head counts and KV chunk widths, tight; never below one row; the default leaves the #425/#426 pin shapes single-pass |
| `TestChunkingIsRankLocalAndCollectiveFree` | no collective, no host sync, both guards proven able to fire, rank divergence is real |
| `TestTheDuplicationIsBoundedNotRemoved` | the gathered block still holds `rows` copies and they are byte-identical — candidate C is open, not closed |

**Falsifier, executed against the base tree** (production files reverted to
`e65b75be1c`, test file kept): **18 of 22 fail**, all on their own terms, no
import or attribute errors — the probe falls back to deriving the gather shape
from `torch.bmm` when the named seam is absent. The substantive numbers:
`4325376 not less than or equal to 1048576` (a 4.1 MiB gather against a 1 MiB
budget), `2162688 != 4325376` (the peak tracking the query count, 32 vs 64 rows),
`16 not less than 16` (full-height duplication).

**Byte-identity can-fail, both executed** on the wired tree, then reverted:

* `q_t_rows = q_t[row_start:row_stop]` -> `q_t[0:rows]` (every chunk scored with
  the first chunk's queries): **21 failures, 1 error**.
* `logits[row_start:row_stop, ...]` -> `logits[0:rows, ...]` (chunk results
  written at the wrong row offset): **21 failures, 1 error**.

Both perturb chunk-boundary handling only and both are caught by the exactness
class, so the byte gate is real.

**Suite state**, `test/registered/unit/layers/attention/`, the seven dsv4 files
plus the new one: **128 tests, 6 errors, 1 skipped**. Base (same seven files,
before the change): **106 tests, 6 errors, 1 skipped**. The failure set is
identical — all six are `test_dsv4_fp8_triton_compat_417` cases that need a GPU
and fail hermetically at base too. Also green: `test_dsv4_nonpaged_indexer` (5),
`test_dsa_indexer` + `test_dsa_metadata` (19, all skipped without a GPU).

`ruff format --check`, `ruff check` and `codespell` are clean on the two new/edited
files. `indexer.py` carries two pre-existing `E712` findings and one pre-existing
formatting difference on lines this change does not touch (confirmed by running
both against `git show HEAD:...` of the same file).

## 5 — GPU measurement arm — BOOT-PENDING, not run

No number is claimed. What the next window must run, in this order, in ONE boot:

1. **Boot.** DeepSeek-V4-Flash-0731 in the `BENCH_394_v4flash_club3090.md`
   configuration — read the flags out of that document and the boot log, do not
   reconstruct them. TP=3 uneven across the 5090 and the two 3080s. Hold
   `/spinning/gpu-arb/` with a heartbeat; stop the heartbeat before releasing.
2. **A-vs-A floor first.** Run the same arm twice with the query budget at its
   default and report its own spread as ms/prefill-round per rank before any A/B
   delta is computed. `NOTE_440:335-348` records upstream seeing 157 s vs 275 s
   across two passes over the same lengths; that variance is the floor this arm
   has to beat before it may claim anything. Discard the first two iterations at
   each shape (Triton/`torch.compile` first-call compilation) and record compile
   time separately.
3. **The A/B.** ms/prefill-round per rank at **8K and 32K** context,
   `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB` at `0` (off, the pre-#449 shape) versus
   the default `2048`, interleaved A/B/A/B rather than blocked, clock fixed.
   Per the standing rule this is ms per round, not tok/s, and it is per rank —
   the query axis is what DP-attention shards, so the ranks are not
   interchangeable.
4. **The memory reading is the primary result, not the speed.** Record peak
   allocated VRAM per rank (`forward_peak.py`) in both arms. The lever is a
   memory bound; the honest success criterion is "same or better ms/round at a
   materially lower peak", and the follow-on question the number answers is
   whether `--chunked-prefill-size` can be raised above the 512 the runbook
   currently forces.
5. **Launch-count counter-check.** The loop trades allocation for launches
   (ANALYSE_447 L2 is about launch overhead). If ms/round regresses beyond the
   A-vs-A floor, sweep the budget upward (4096, 8192 MiB) before concluding
   anything about the mechanism — the default is a ceiling picked at desk, not a
   tuned value.
