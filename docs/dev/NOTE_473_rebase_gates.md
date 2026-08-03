# #473 -- sweep-3 reconciliation: indexer arch guard, rebase gates, draft-KV-DCP granularity, EAGLE 2x reserve

Desk-only (`CUDA_VISIBLE_DEVICES=99`, no `/spinning/gpu-arb` hold, no GPU
touched). Worktree tip `cf940b16c8` (merge of #472,
`origin/integration/r3-probe-next2`). Source: `SWEEP3_upstream_2026-08-03.md`
(items (f) "#440 head-folding env coupling", (C) rebase hazards,
`#33348`/`#32828` DCP/draft-KV, `#32459`/`#32574` EAGLE 2x reserve). Catalog
sections read: `docs/dev/FEATURE_CATALOG.md` §1 (uneven parallelism / DCP /
draft-KV), §4 (speculative decoding), §15 (model bring-ups / indexer), §16
(measurement infra). Also read `docs/dev/NOTE_440_c4_indexer_head_fold.md`,
`docs/dev/ANALYSE_442_upstream_adoption_sweep.md` §3, `docs/dev/
TASK_108_DRAFT_KV_DCP.md`, `docs/dev/TICKET_470_dspark_boots.md` §6,
`planner/rejected.py` (checked for a matching discarded-approach entry on
each of the four items below -- none found; nothing here duplicates a
rejected register entry).

## (a) #440 indexer arch guard vs upstream #33288's `_varlen_arch_ok`

**Verdict: ours is stricter, already known -- no new PRIO.**

Upstream (`sgl-project/sglang#33288`, commit `16299f22`, fetched via
`gh`/`.diff` against the public repo) adds, at `indexer.py` (their line
numbers, `@@ -1026,11 +1026,19 @@`):

```python
# deep_gemm's varlen kernels (fp8_mqa_logits / fp8_fp4_mqa_logits)
# assert arch_major >= 9 (Hopper/Blackwell).  Ampere (sm80/sm89) can
# import deep_gemm but the kernel refuses at runtime, so the varlen
# routing must not fire there.  See sgl-project/sglang#33246.
_varlen_arch_ok = is_cuda() and torch.cuda.get_device_capability()[0] >= 9
```

Our tree's guard is `deepgemm_indexer_supported()` /
`_DEEPGEMM_MAJORS = (9, 10)` at
`python/sglang/srt/layers/attention/dsv4/indexer_arch.py:51,62-71`, checked
structurally ahead of every env-based routing decision in
`_can_use_nonpaged_indexer` (`python/sglang/srt/layers/attention/dsv4/
indexer.py:712-748`, arch check at line 731 precedes the
`SGLANG_OPT_USE_TILELANG_INDEXER` / `SGLANG_OPT_USE_AITER_INDEXER` /
`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` env clause) -- exactly the "arch guard
decides ahead of the env clause" property pinned by
`TestNonPagedCouplingIsAlreadyArchGuarded` in `NOTE_440`'s test file
(`test/registered/unit/layers/attention/test_dsv4_indexer_head_fold_440.py`).
That is the #440 note's "502-508 clause" as it existed at the commit the
note was written against; the arch guard was already in place and already
tested by the time #440 landed, so the note's "needs an arch guard before
removal" caution was resolved in-tree, not left open.

**The two guards are not equivalent, and ours is the stricter one.**
Upstream's `capability[0] >= 9` is unbounded above -- it treats consumer
Blackwell (sm120, capability major 12) as "arch ok" and would route a 5090
into `deep_gemm.fp8_mqa_logits`'s varlen path, where it is caught only by
deep_gemm's own internal assertion (`arch_major ∈ {9,10}`, referenced in the
sweep and consistent with `indexer_arch.py`'s comment "DeepGEMM upstream
declined SM12x support (deepgemm PR #318)") -- a crash inside the kernel,
not a graceful reroute. Upstream's own new test file
(`test_dsv4_nonpaged_indexer.py`) only exercises sm80 (excluded) and sm90
(included); there is no sm120 case. Our `_DEEPGEMM_MAJORS = (9, 10)` is an
explicit finite tuple that excludes major 12 up front and routes straight to
`BACKEND_TORCH`, so a 5090 never reaches deep_gemm's assertion at all. This
exact defect in a *ported* #33288 was already caught and named before this
sweep, in `ANALYSE_442_upstream_adoption_sweep.md:254-256` ("defect 1... it
would route a 5090 into a kernel #417 established is absent there. Replace
with `deepgemm_indexer_supported(device_id)`"). Nothing new to fix here;
this NOTE just closes the loop the sweep asked us to check. No catalog
change -- §1/§15's existing statements about the indexer arch dispatch
already describe this correctly.

## (b) Rebase gates for any future rebase past upstream main

Hazards the sweep found, none of which are live exposures on our current
branch (we are not on upstream main), all of which must be re-checked at
rebase time:

1. **#33312 (open)** -- DSV4 DSpark shared-expert loading is silently broken
   on upstream main: #33013 made load-time overrides config-only, so the
   draft worker copies a pristine `ServerArgs` without DSV4's resolved
   shared-expert fusion policy and every `mtp.*.ffn.shared_experts.*` tensor
   is dropped (logged only as "unexpected weight"). Already tracked in
   `TICKET_470_dspark_boots.md` §6 -- cross-referenced here, not duplicated.
   Do not rebase past this without either #33312 landing or a local
   equivalent that keeps the resolved fusion policy on the draft's
   `ServerArgs` copy.
2. **The 5 ServerArgs-burndown merges (#33334-#33338, ch-wan)**, incl. "spec:
   build every draft worker from a draft ServerArgs copy" and "retire the
   last process-global config field reads". This stack is the proximate
   cause of #33312 -- it is the reason the draft's `ServerArgs` copy stopped
   carrying the resolved fusion policy. Any rebase crossing this stack
   inherits #33312's failure mode unless #33312 (or its equivalent) is taken
   at the same time.
3. **`flash_mla_sm120.py` moved upstream** from
   `python/sglang/srt/layers/attention/` to
   `python/sglang/kernels/ops/attention/` (as part of #33297, "Fuse the DSV4
   paged indexer scoring on the portable path"). Our copy is still at
   `python/sglang/srt/layers/attention/flash_mla_sm120.py` (confirmed present
   in this worktree). Any of our sm120 patches to this file -- including the
   B1 adopt candidate from this same sweep (#32320's masked SWA page-split)
   -- are a port to the new path, not a clean cherry-pick, once we rebase
   past the move.
4. **#33350 (pending)** -- extracts V3.2/DSA logic into a new
   `DeepseekV32Mixin` (568+/257- across four files), behavior-preserving by
   intent. Touches `deepseek_v2.py` / `deepseek_nextn.py`, both of which we
   patch (confirmed present at `python/sglang/srt/models/deepseek_v2.py` and
   `python/sglang/srt/models/deepseek_nextn.py`). Re-diff our patches against
   the extracted mixin after this lands, do not assume line-offset porting.
5. **#33298 (merged upstream)** -- moves the DSPARK graph-folded sampler into
   a new `dspark_draft_sampler.py` module and adds in-graph philox sampling.
   Already tracked in `TICKET_470_dspark_boots.md` §6 ("`dspark_draft.py`
   moves... re-check `DraftBlockProposer.propose`'s fold predicate and the
   solo embed hoist after that rebase") -- cross-referenced here, not
   duplicated.

## (c) #33348 (page-size/allocator-granularity mismatch) vs our #108 draft-KV-DCP layout

**Verdict: our REPLICATED draft-KV fallback does not share the root cause --
cited, not merely asserted. One narrow, lower-confidence side note flagged
below, not claimed as a confirmed PRIO.**

Upstream #33348 (kpham-sgl): after #32828, the shared allocator pages the
virtual loc space in `page_size * dcp_size` units; the replicated draft KV
pool kept its `page_size` at the per-rank (unscaled) value, so the
allocator's last page is only partly backed by pool rows and a decode-time
CUDA-graph write into the tail goes out of bounds (`dcp_size=8`: 64 of 512
slots backed). Fix: `python/sglang/srt/mem_cache/kv_cache_configurator.py`,
new `loc_space_scale` / `pool_page_size` accessors on `KVCacheConfigurator`
(`dcp_size if is_draft_worker and dcp_size > 1 else 1`, then
`get_schedule().page_size * loc_space_scale`), applied at every pool-building
call site. No-op at `dcp_size == 1`.

Our tree has no `KVCacheConfigurator` class (different structure -- pools
are built per-`ModelRunner` instance in
`python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`), so the
port does not apply mechanically; the question is whether the same
*mismatch* -- pool storage sized/paged differently than the addressing
scheme that indexes it -- exists under a different name. It does not, on the
path our catalog names as the REPLICATED (DEGRADED) draft-KV layout:

* `draft_pool_is_replicated()` (`python/sglang/srt/layers/dcp/owner.py:102-124`)
  is the single predicate gating this; `model_runner_kv_cache_mixin.py:3554`
  binds it to `_draft_non_dcp`.
* When `_draft_non_dcp` is true, `_hybrid_pool_size = self.max_total_num_tokens`
  (line 3633) -- the full, UNSCALED per-rank token count, never multiplied by
  `dcp_size`.
* The pool that stores it, `HybridLinearKVPool` (line 3738), is built with
  `page_size=self.page_size` (line 3739) -- the natural, UNSCALED page size.
* This is by explicit design, not by omission: the comment at
  `model_runner_kv_cache_mixin.py:3966-3974` states the rule for the whole
  uneven-DCP/weightless lane -- "the paged page_size stays NATURAL (base
  page_size), never inflated by the factor. Inflating it (the stock
  `page_size * dcp_size`) forces the radix cache's page granularity to the
  DCP factor, which collides with `mamba_cache_chunk_size`..." -- and
  `_dcp_token_sharded_pool_rows`'s docstring (lines 2750-2767) documents the
  same natural-vs-inflated split for the plain-MHA pool.

Since neither the replicated draft pool's storage nor its addressing ever
adopts the `page_size * dcp_size` convention, the precondition #33348 needs
(pool paged at one granularity, addressed at another) is structurally absent
on this path -- there is nothing here for the mismatch to attach to. This is
a **cited "no gap"**, not an assumption of safety by architecture-name alone.

**One loose end, flagged for a follow-up glance, not raised as a confirmed
PRIO**: the "stock even-DCP (unchanged)" branch at
`model_runner_kv_cache_mixin.py:4001-4011` (reached only for non-hybrid-SWA
models under a bare `--dcp-size N` with neither `--rank-tp-ratio` nor
weightless-KV active) *does* build its allocator with
`page_size=self.page_size * self.dcp_size`, upstream's original convention,
carried over unaudited by #108 (`TASK_108_DRAFT_KV_DCP.md` never mentions
`page_size`). Whether a draft worker with a REPLICATED (default,
non-`--draft-kv-layout dcp`) pool can actually reach this branch -- i.e.
whether stock even-DCP + speculative decoding + a non-hybrid model is a
combination that occurs on any model this fork boots -- was not established
here; DCP in this fork is otherwise only exercised against hybrid-SWA
(DSV4-class) models, where the `_draft_non_dcp` handling above applies
instead. Recommend: a follow-up task write a falsifier that boots the
narrowest reachable case (or shows the branch is dead for any
speculative-decoding configuration this fork supports) before this line item
is closed either way. No catalog change -- §1's "above TP>kv_heads,
replicated is the DEGRADED layout" statement is unaffected; degraded, not
broken, remains accurate.

## (d) #32459/#32574 EAGLE 2x-KV double buffer vs our tree

**Verdict: we carry the identical mechanism, unmodified from upstream, and
it is currently UNCOUNTED in our VRAM-corridor bookkeeping. PRIO finding for
a follow-up measurement task -- reported here, not fixed.**

Upstream #32574's root cause: `req.kv_committed_len` lags one verify step
behind the true row count in overlap mode, so `get_alloc_reserve_per_decode`
reserves a 2x KV double-buffer per request at every decode step, depleting
the pool and forcing extra eviction -- measured as radix prefix reuse
collapsing 97% to 40-53% on multi-turn traffic.

Our tree has the exact same function, same doubling, same justification, at
`python/sglang/srt/mem_cache/common.py:290-296`:

```python
def get_alloc_reserve_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    """KV length reserved per request at each decode step.

    The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap
    mode; see eagle_utils.eagle_prepare_for_decode.
    """
    return 2 * get_alloc_len_per_decode(server_args)
```

It is live on both spec paths that matter to us:

* `python/sglang/srt/managers/schedule_batch.py:2745-2754`
  (`_new_tokens_required_next_decode_spec_v2`) calls it directly to compute
  `new_tokens_required_next_decode`, which feeds `check_decode_mem` ->
  `evict_from_tree_cache` at line 2758 -- i.e. it directly drives eviction
  pressure against the shared radix tree on every decode step under any
  speculative algorithm.
* `python/sglang/srt/speculative/eagle_utils.py:1220-1233`
  (`eagle_prepare_for_decode`) reads `double_alloc =
  get_alloc_reserve_per_decode()` directly. This is the shared prepare-for-
  decode path our NEXTN/MTP standard config (steps=3, topk=1, draft=4) and
  our DFLASH/DSpark draft-solo paths all route through -- the doubling is
  not gated on `topk > 1`; `get_alloc_len_per_decode` (lines 270-287) only
  changes its *formula* based on `page_size`/`topk`, but
  `get_alloc_reserve_per_decode` doubles whatever that formula returns
  unconditionally. So our topk=1 standard path (the only topk value we ever
  run in production -- topk>1 under DCP is hard-gated per #76) still pays
  the 2x reserve.

**Corridor bookkeeping check**: grepped `docs/dev/*.md` for
`get_alloc_reserve_per_decode` / `get_alloc_len_per_decode` /
`kv_committed_len` -- zero hits. No design doc, no ANALYSE, no BENCH file
mentions this reserve. It has never been counted as a line item against the
VRAM corridor (`forward_peak.py`, §16) or against the Option-A waste rule.
Unlike a fixed idle-VRAM block, this is not bytes sitting unused at peak --
it is a per-request KV-slot reservation that reduces the number of
concurrently-fittable requests and, per upstream's measurement, collapses
achievable prefix-cache reuse. The GiB-equivalent of that reservation on our
own boot shapes (NEXTN/MTP steps=3, and the DFLASH/DSpark solo lanes) has
not been measured here -- this NOTE is the falsifier-not-yet-run flag, not
the measurement. **Recommend a follow-up task**: measure
`get_alloc_reserve_per_decode()`'s actual token cost at our standard
boot config, express it as an equivalent MiB reserve against
`max_total_num_tokens`, and check whether it is large enough to register
under the Option-A rule (>1.5 GiB net) on the 5090/3080 boot shapes we
actually run. Do not fix or tune this reserve in that follow-up without a
same-boot A-vs-A floor first (Full-Perf-Testen / ms-pro-runde rules apply).
