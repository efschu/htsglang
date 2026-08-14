# ANALYSE SWEEP5 — upstream adoption sweep 5 (2026-08-14)

Fork: /spinning/htsglang (branch integration/r2 is the merged line).
Reading tree: /spinning/wt-485-cert (worktree checkout of integration/r2),
`python/` subtree = "our tree" in everything below.
Worktree HEAD at time of this sweep: `858c58c91c8cb54e762d7306f46e968638ffca6e`
(2026-08-14 07:15:22 +0000). Note: the task brief named `cb8da83774` as the
worktree's commit; the worktree had moved on by the time I read it. I used
the on-disk state as-is (per the brief's own instruction to read code from
that worktree) rather than checking out the older SHA. Flagging this as a
minor provenance note, not treated as an error.

Upstream: https://github.com/sgl-project/sglang.
Previous sweep: docs/dev/ANALYSE_611_adoption_sweep4.md, commit `d8195e2560`,
dated 2026-08-06 (recovered via `git show d8195e2560:...`, saved locally at
/tmp/desk485/ANALYSE_611_prev.md). This document follows its format at
reduced implementation depth: this is a DRAFT/ANALYSIS deliverable, not a
port — no code was changed, no tests were run, findings are graded by
grep/read confidence, not by pytest evidence.

> **DESK VERIFICATION NOTE (operator, 2026-08-14).** The upstream retrieval
> below was performed by a delegated agent against the GitHub API. Upstream
> PR/issue numbers, titles and dates are therefore AGENT-RETRIEVED and have
> not been re-fetched by hand; treat them as leads, not as citations, and
> re-verify before any of this is quoted outside the fork. What I checked
> MYSELF, in this tree, is the OUR-TREE half of item 1 — the only ADOPT:
> `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` contains
> exactly **3** occurrences of `tl.int64` across the whole file, while the
> pointer arithmetic at lines 92-94 and 176-179 multiplies
> `sequence_start_index` (and `idx_tokens_last`) by `stride_x_token` with no
> widening cast. The exposure is therefore real in our tree independently of
> whether the upstream PR is numbered as reported. Every other item is
> UNVERIFIED by me and carries the agent's own confidence grading.
>
> Serving was down for this shift, so no local-model lane was available to
> cross-check; that is why this note is narrower than usual.

## 1. Sweep metadata

- Window: 2026-08-06 (day after sweep 4) through 2026-08-14, extended back
  to 2026-08-01 for anything material.
- Method: WebFetch against `api.github.com/search/issues` with
  `q=repo:sgl-project/sglang ... in:title:<keyword> updated:>DATE`, narrowed
  per lens family (DCP, mamba/GDN, cuda graph, DSV4, HiCache, PP, dspark,
  eagle/spec). No GPU/build/test tooling was used — CPU-only grep/Read
  against the worktree plus WebFetch/WebSearch against GitHub.
- What worked: narrow, `in:title`-scoped queries reliably returned complete,
  usable result sets (5-15 items) with correct `total_count`.
- What did NOT work / was not attempted:
  - Broad (non-`in:title`) queries: WebFetch's underlying summarizer
    reports an accurate `total_count` but silently truncates to ~10-13
    items regardless of `per_page`, even when 100+ results exist (e.g. the
    "dspark in:title" query below: `total_count: 140`, 13 shown). Any
    finding from a broad query should be read as "at least this many
    exist", never as an exhaustive list.
  - `api.github.com/repos/sgl-project/sglang/commits?since=...` — the
    method the task brief suggested as a supplement — was **not** queried
    this sweep. Everything below rests on the issues/PR search endpoint
    only. This is the single biggest coverage gap of this sweep.
  - MoE/expert-offload and tree-spec/adaptive-draft lens families got only
    thin, incidental coverage (nothing surfaced that competed for top-5
    slots); do not read their absence below as "clean", read it as
    "under-swept this round".

## 2. Ranked adoption/monitor list

| # | Upstream ref | What it changes | Our file:line | Verdict | Effort | Yield |
|---|---|---|---|---|---|---|
| 1 | [#33665](https://github.com/sgl-project/sglang/pull/33665) "Fix int32 overflow in causal_conv1d pointer arithmetic" (merged 2026-08-07) | Casts `sequence_start_index`/other token-axis offsets to `tl.int64` before multiplying by `stride_x_token`/`stride_o_token` in the Triton causal-conv1d kernels; at long context length with wide strides (fused QKV pitch) the int32 product overflows and reads/writes the wrong row. | `/spinning/wt-485-cert/python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` — unguarded sites at lines 93-94 (`x_base = ... sequence_start_index * stride_x_token ...`), 176, 217-218, 248-249, 271, 279-296, 308, 334, 342, 353, 375 (kernel 1) and 727, 730, 834, 885, 913, 921, 932, 971 (kernel 2, second `_causal_conv1d` function). Only `conv_state_batch_coord` (line 99) and two unrelated offsets (651, 656) already carry `tl.int64`. | **ADOPT** — confirmed exposed, unfixed, matches upstream's own diagnosis exactly (their fix note explicitly says "the neighboring index is already int64, this one wasn't"). | S — mechanical cast insertion at each pointer-arith site, same pattern already present at line 99. | H — this is exactly the long-context regime the standing capacity spec (bs1+YaRN over 262144 tokens) targets; an int32 overflow here is a silent-corruption bug, not a crash, so it can currently be producing wrong mamba-layer output at long context without any visible signal. |
| 2 | [#34760](https://github.com/sgl-project/sglang/pull/34760) / [#34780](https://github.com/sgl-project/sglang/pull/34780) / [#34808](https://github.com/sgl-project/sglang/pull/34808) — three competing, all-open, none-merged fixes for "Mamba checkpoint depth vs DCP-widened radix page" | Under DCP, the radix-tree page is widened (`allocator_page_size * dcp_size`) but Mamba/GDN recurrent-state tracking stays on the narrow (chunk_size) grid; donating a checkpoint at a non-representable depth causes deterministic logprob divergence on cache resume. #34760: skip donation when unaligned + restore a legacy assertion. #34780 (WIP): validate checkpoint depth, fall back to KV-only insert for finished requests. #34808 (the most recent, from a core maintainer): pick donation depth on `lcm(chunk_size, tree_page)` so donated states land on representable depths without losing prefix reuse. | `/spinning/wt-485-cert/python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py` lines 56-57 (`page_size == 1` asserted when `mamba_extra_buffer` disabled), 92-96 (`aligned_seqlen = (sum(...) // chunk_size) * chunk_size` — chunk-size-only alignment, no reference to the DCP-widened tree page), 406-409 ("page_size is asserted == 1, so no realign") and 461-498 (the actual donate call sites). | **MONITOR, falsify before trusting** — our fork already has *a* chunk-size alignment step (lines 92-96), but it aligns to `mamba_cache_chunk_size` only, never to the DCP page width (`allocator_page_size * dcp_size` per FEATURES_VS_UPSTREAM item 2). The comment at 406-409 suggests the donate path's safety net is "page_size==1 is asserted", which does not describe our own uneven/weighted-DCP mode where page_size > 1. Upstream hasn't converged on a fix either (three competing PRs as of today), so there is no drop-in port yet — but the underlying defect class is one we are structurally exposed to. | M to falsify, L-M to port once upstream converges (likely #34808's `lcm` approach). | H — this is a *silent* correctness bug (wrong logprobs on cache resume, not a crash), directly in a mechanism (uneven-DCP + mamba hybrids) our fork has invested heavily in (Draft-KV-DCP #108, GDN-Prefill-Nichtdeterminismus history). |
| 3 | [#33614](https://github.com/sgl-project/sglang/pull/33614) "Fix Dspark state divergence across TP rank" (open, fixes [#33289](https://github.com/sgl-project/sglang/issues/33289), filed by MiaAI-Lab) | Under TP>1, DSpark let each rank make independent sampling decisions (draft proposal tokens, verify outcomes, prefill next-token ids), causing KV/sequence drift and an eventual NCCL-collective deadlock. Fix: a new `DsparkTpSync` helper broadcasts rank-0's decisions to all ranks via the TP group's PyNCCL communicator, in both eager and CUDA-graph-captured paths. | `/spinning/wt-485-cert/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` — has its own `_solo_mirror` publish/broadcast machinery (`publish_normed` line 320/343, `broadcast_round` line 364/372) wired into **both** `_forward_prefill` (line 533) and `_forward_decode` via a shared `on_publish` callback threaded through `forward_batch_generation` (line 516-531). `dspark_verify.py` itself (716 lines) has **zero** `tp_rank`/`broadcast`/`dist.` references — accept/verify math (`correct_len`, `bonus`, `cap_trim_lens`) runs redundantly per rank, which is only correct if every tensor feeding it is already rank-synced upstream. | **MONITOR, likely not exposed but unverified end-to-end** — the architecture looks like an independently-built equivalent to `DsparkTpSync` (publish-once-broadcast pattern, same shape as the general hetero-TP sampling fix noted in the fork's own history). But I did not trace every decode-step branch (CUDA-graph-captured vs eager, solo-rank draft placement at line 472) to confirm no per-rank-independent sampling escapes the broadcast boundary. | M to falsify, S if a gap is found (the sync primitive already exists, a missed call site is a one-line fix). | M — a deadlock here is loud (matches the root issue's own title), not silent, so risk is "we might hang under TP>1 DSpark", not "we're quietly wrong". |
| 4 | Hakureirm's DSV4 cluster: [#33245](https://github.com/sgl-project/sglang/issues/33245) (packed-weight AttributeError on `.weight`), [#33246](https://github.com/sgl-project/sglang/issues/33246) (oversized `[B,S,H]` indexer intermediate OOM), [#33247](https://github.com/sgl-project/sglang/issues/33247) (indexer masks padding with `0.0` instead of `-inf`), all self-closed unfixed; superset [#33271](https://github.com/sgl-project/sglang/pull/33271) (closed, unmerged, CI red both runs) | Same author's PRs to make DeepSeek-V4 run on SM80: Triton `fp8e4nv` compile failure, `is_sm120_supported()` dispatch bypass, packed-quant checkpoint loading, indexer chunking/masking, reverted a lossy "head fold". | `/spinning/wt-485-cert/python/sglang/srt/layers/attention/dsv4/indexer.py`: `-inf` masking already present in both `fp8_paged_mqa_logits_torch` (~line 189-196) and `fp8_paged_mqa_logits_torch_sm120` (~506, 569-571); query/sequence-axis chunking already present (`_indexer_logits_query_chunk` ~303-422, gated by `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB`), with a docstring explicitly describing the exact overshoot #33246 reports. `/spinning/wt-485-cert/python/sglang/srt/models/deepseek_v4.py` has hard-fail guards around `_WQKV_A_PROJECTIONS = ("wq_a","wkv")` (~line 3362) that error loudly ("Set SGLANG_OPT_FUSE_WQA_WKV=0...") rather than silently dropping packed weights — matches #33245's fix suggestion #2, not fix suggestion #1. | **IGNORE (already hardened)** for #33246/#33247 — high confidence, read the actual masking/chunking code. **IGNORE, medium confidence** for #33245 — grep/structural evidence only (hard-fail guard exists), did not trace a full packed-checkpoint load end to end. | — | — (informational: confirms prior hardening held up, no action needed) |
| 5 | [#33639](https://github.com/sgl-project/sglang/pull/33639) "[HiCache][2/2] Support Mamba branching in Unified Radix Cache with HiCache" (merged 2026-08-10) | Lets Mamba state get backed up to HiCache incrementally/component-only, rebuilt from host-cached Full KV, instead of requiring a full KV re-copy on every Mamba backup (benchmark: cache-hit rate 1/10 -> 10/10). | Cross-references sweep-4 item 1: our `unified_radix_cache.py` `write_backup` path currently *couples* the `FULL` and `MAMBA` component backup decision (the `.backuped` predicate sweep-4 found is FULL-only). Sweep 4 concluded this coupling meant we were NOT exposed to upstream's then-#33713 bug (MAMBA silently pruned instead of downgraded) specifically *because* our backup path does not decouple FULL from MAMBA. | **MONITOR — do not adopt standalone.** #33639 is exactly the kind of decoupling sweep-4's finding warned about. If we ever port #33639-style incremental Mamba-only HiCache backup, we MUST re-audit/port sweep-4's #33713 `.backuped`-predicate fix in the same change, or we reopen the exact bug sweep-4 closed out. | S to note, M-L if actually adopting the feature (must be bundled with the predicate fix). | M — currently a non-issue since we don't have the feature; becomes H the moment anyone starts porting #33639 without reading this. |
| 6 | [#27010](https://github.com/sgl-project/sglang/pull/27010) "[HiCache] Fix PP inconsistency with HiCache L3" (open, unmerged, still updated 2026-08-14 — active) | PP ranks can diverge on shared L3 (disk-tier) HiCache state; fix adds a `pp_sync` scheduler-thread step plus an `all_reduce(MIN)` on the prefetch thread to agree on shared L3 sequence length across PP ranks. | Not checked this sweep — I did not locate/read our PP+HiCache interaction code (relevant to the standing PP-prefill/TP-decode strand, #631, and the Disk HiCache tier). | **MONITOR, unverified** — flagged for the next sweep or for whoever owns #631/HiCache-disk-tier to check directly; I ran out of budget to trace it this round. | — | — (severity unknown until checked; PP+HiCache is an active strand area, so treat as higher priority to check than its "unverified" tag suggests) |
| 7 | [#33431](https://github.com/sgl-project/sglang/pull/33431) (test-only, adds coverage for the padding-sentinel fix, all-padded-batch DP-idle-rank case) | No new fix — adds a regression test for the already-merged #33810 (GDN chunked-extend `-1` sentinel OOB, which we ported in sweep 4). | Our sweep-4 port of #33810 stands. | **IGNORE** — informational confirmation only. | — | — |

## 3. "Silently wrong in our tree?" — falsifiers

For every item above where the fork could already be quietly incorrect, a
concrete, cheap check that would prove it:

- **#1 (causal_conv1d int32 overflow).** Falsifier: construct a single-rank
  CPU-side arithmetic check (no GPU needed) — compute
  `sequence_start_index * stride_x_token` for a synthetic long-context case
  (e.g. seq_len > 262144, fused QKV stride typical of our checkpoints) in
  plain Python int32 vs int64 and diff the two. If they diverge for a
  reachable (seqlen, stride) combination in our launch configs, the bug is
  live, no GPU run required to prove the arithmetic class exists. A GPU-side
  falsifier (when a strand has a window) would be: run a causal-conv1d
  mamba/GDN model at context length beyond the int32-overflow threshold for
  the observed stride, diff output against a chunked/short-context run that
  cannot overflow.
- **#2 (DCP+mamba checkpoint depth).** Falsifier: with weighted/uneven DCP
  active (page_size = `allocator_page_size * dcp_size` > 1) and a mamba/GDN
  hybrid model, run a prefill that crosses a chunk boundary not aligned to
  the DCP page width, evict, then resume from the radix-cache checkpoint;
  diff logprobs of the resumed continuation against a page_size=1 (DCP
  disabled) baseline for the same prompt. A mismatch confirms exposure —
  this is exactly the shape of test the three competing upstream PRs are
  adding (kpham-sgl's #34780 explicitly adds "DCP-aware strategy selection
  + regression tests" for this).
- **#3 (DSpark TP-rank divergence).** Falsifier: launch DSpark under TP=2
  (mixed or matched GPUs), run a batch of decode steps, and assert bit-exact
  match of `correct_len`/`bonus`/`cap_trim_lens`/sampled token ids across
  both ranks' local `dspark_verify.py` computations at every step, including
  at least one CUDA-graph-captured decode step and one step through the
  solo-rank draft placement branch (`dspark_worker_v2.py` line 472,
  `tp_rank == 0` special-cased draft sampling). A divergence anywhere
  confirms the sync boundary is incomplete.
- **#5 (HiCache Mamba branching / #33713 coupling).** Falsifier: this one
  is a process falsifier, not a runtime one — before merging any port of
  #33639-style Mamba-only incremental HiCache backup, grep the diff for
  whether it touches the `.backuped` predicate coupling FULL and MAMBA
  commits in `unified_radix_cache.py`; if the PR decouples them without also
  carrying sweep-4's #33713 fix, that is the falsifier firing (reject the
  port as incomplete).
- **#6 (PP+HiCache L3).** No falsifier constructed yet — this item needs a
  first read of our own PP+HiCache code before a falsifier can even be
  designed. Explicitly marked as the weakest-grounded item in this sweep.

## 4. Watch items status

- **#33271** — Hakureirm's "Make DeepSeek-V4 serve on SM80" PR. **Closed,
  unmerged, 2026-08-04**, both CI runs red, author noted needing a `run-ci`
  label they couldn't self-apply. Superset of the #33245/#33246/#33247
  cluster (absorbs closed #33272, relates to follow-up #33297). Reported
  strong accuracy numbers on 8xA800 (GSM8K 0.946) and a decode-kernel
  speedup (~320ms -> ~7ms) that never landed anywhere upstream because the
  PR died in CI. Our tree already independently carries the masking/
  chunking fixes (see item 4 above); the SM80-dispatch-bypass and Triton
  `fp8e4nv` compile-failure parts of this PR were **not** cross-checked
  against our tree this sweep (our cards are SM86/SM120, not SM80 Ampere
  datacenter — relevance to us is plausible via shared dispatch-name
  handling but unconfirmed).
- **Hakureirm** — author of #33271, #33245, #33246, #33247, all closed
  unfixed/unmerged upstream, all DSV4/indexer/SM80-adjacent. No further
  activity found in this window beyond that cluster. Our tree's independent
  hardening (item 4 above) means this author's unmerged work is low-risk to
  us regardless of upstream's decision.
- **MiaAI-Lab** — exactly one item found this window:
  [#33289](https://github.com/sgl-project/sglang/issues/33289), "Multi-node
  TP rank-divergence deadlock: one rank wedges in NCCL proxy append (logits
  all-gather)...", filed 2026-08-02, open, still updated 2026-08-13. This is
  the root-cause issue that #33614 (item 3 above) fixes. No PRs from this
  author found.
- **DSpark competition** — confirmed real and intense: an `in:title:dspark`
  search returned **140 total PRs** touching DSpark (only 13 visible due to
  WebFetch truncation, see metadata section). The 13 visible span 12
  distinct authors (hnyls2002, JustinTong0323, 2044145178, qq1060,
  JackZeng0208, ormandj, shenxiul, b8zhong, QAQEthan, zhangxiaolei123456,
  AliceChenyy, yhyang201) covering perf fixes, an NPU port, speculators-
  format checkpoint support, EP1 perf regression, non-blocking H2D copies,
  a mask-filling draft convention, quantized-lm_head draft-logits fix,
  logprobs support, PD disaggregation, SM120 SWA width, and a Kimi-K3
  default-backend change — in the last ~10 days alone. Read this as: DSpark
  is becoming a first-class heavily-contested upstream feature (validates
  our own DSpark investment), but also as a fast-moving target where any
  snapshot of "what upstream does" goes stale within days — do not treat
  any single DSpark PR read this sweep as durable without rechecking at
  next sweep.

## 5. Top-5 summary

1. **ADOPT** — causal_conv1d Triton kernel int32 pointer-offset overflow
   (upstream #33665, merged 2026-08-07); confirmed exposed, unfixed in our
   `causal_conv1d_triton.py` at 15+ sites; silent long-context corruption
   risk, cheap mechanical fix.
2. **MONITOR/falsify** — Mamba checkpoint-depth vs DCP-widened radix page
   misalignment; three competing unmerged upstream PRs (#34760/#34780/
   #34808), none converged yet; our own `mamba_component.py` aligns to
   chunk_size only, not to the DCP page width — plausible silent logprob
   divergence on cache resume under our own uneven-DCP feature.
3. **MONITOR/falsify** — DSpark TP-rank state divergence (upstream #33614,
   fixing MiaAI-Lab's deadlock issue #33289); our tree has its own
   `_solo_mirror` broadcast machinery wired into prefill+decode but not
   traced end-to-end for gaps.
4. **IGNORE (confirmed already hardened)** — Hakureirm's DSV4 indexer
   masking/OOM cluster (#33245/#33246/#33247, closed unfixed upstream); our
   `indexer.py` and `deepseek_v4.py` already carry the equivalent fixes.
5. **MONITOR (coupled-adoption warning)** — HiCache Mamba-branching feature
   (#33639, merged 2026-08-10) must never be ported standalone: it decouples
   FULL/MAMBA HiCache backups, which is exactly the precondition sweep-4's
   #33713 finding relied on us NOT having.

Not retrievable / not attempted this sweep: the `commits?since=` GitHub
endpoint (never queried — biggest gap); full listings behind any broad
(non-`in:title`) search past ~10-13 items (WebFetch truncates, `total_count`
is accurate but the item list is not); PP+HiCache L3 desync (#27010) was
found but never cross-checked against our own tree; MoE/expert-offload and
tree-spec/adaptive-draft lens families got only incidental coverage this
round, nothing there was pushed to top-5 but that reflects thin search, not
a clean bill of health.
