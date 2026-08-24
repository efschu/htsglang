# NOTE 856 — the flip carries no KV: design, blockers, retirement inventory

## BUILD STATE (2026-08-24) — what is built, and what is not

BUILT AND HERMETICALLY PINNED:

* **Pending priced by residency** (`scheduler.uncached_prompt_tokens`,
  `Req.reset_for_retract` stamps the fill boundary before clearing it). The
  prerequisite: without it a cutover hands the policy the retracted prompts at
  cold-prefill price, which is #731's "six cutovers, nothing served" by a
  route #731's own dedup cannot catch.
* **The seam order as a law** (`release_residents_for_cutover`): retract
  STRICTLY before reset, refusing if either step is absent. The FATAL order is
  reproduced hermetically against a faithful model of `dec_lock_ref`'s walk,
  so if it ever stops raising the model has drifted from the crash.
* **The retirement**: residents retracted and the tree dropped after the
  fence, then the transfer plan rebuilt EMPTY, so the wave loop iterates over
  nothing. Done by emptying the mover's INPUT rather than deleting a loop
  whose extent bookkeeping still has to run.
* **`_seam_reserve_bytes`**, with `wave_peak` retired from the ask.
  `_staging_bytes` keeps its meaning and all its measured pins — two
  different questions, two names.
* **Launch refusal**: `--enable-phase-flip` without
  `--enable-hierarchical-cache` is refused at parse time (#806 pattern), with
  no mover fallback offered and a test asserting the message says so.
* **Old-contract tests rewritten**: `TestTheFlipCarriesNoKv` asserts the
  seam's transient storage is EXACTLY 0 in both directions and that every rank
  drops its tree once; the wedging request that used to be unaffordable now
  is; waving no longer changes the price.

BUILT BUT ONLY HALF-WIRED, said plainly:

* **The warm-up ledger** (`managers/warmup_latency.py`). `note_cutover` is
  wired at the cutover. The REQUEST feed is not: request latency is assembled
  in `tokenizer_manager`, a different process, so it is a cross-process
  integration rather than a line. For W27 the warm-up cost must therefore be
  read from the CLIENT side, and if it is not collected it is UNMEASURED —
  never "no warm-up cost observed".

NOT BUILT:

* Deletion of the now-dead wave-mover code. It is inert (empty input) but
  still present; deleting it is a separate cleanup with its own diff.
* `hicache_demotion` is still off by default; turning it on for flip
  deployments is the evict-before-persist coverage and is not yet wired.
* `phase_flip_rebind_hicache` is still `False`. The tree drop makes lookups
  miss, which is what correctness needs; arming the rebind is the separate
  #847 refusal-conversion step.

KNOWN VACUOUS: `TestMoverLiveSetIsBounded` now passes trivially (peak 0 is
below any bound) and can no longer fail. It is subsumed by
`test_the_seam_moves_no_kv_at_all` and must not be cited as evidence.


USER DECISION (2026-08-24, binding, verbatim): *"das kv soll niemals vom layer
flip her stammen, einfach aus dem hicache laden fertig."* — with the context
that HiCache/Radix was always the intended cross-layout KV home and the flip
mover was built past that intent.

This is a DESIGN DECISION, not a measurement outcome. The ledger
(`NOTE_856_seam_cost_ledger.md`) is validation, not a vote.

## The target shape

    phase ends -> layout stops
      -> FENCE: persist the un-written writeback tail + the GDN/mamba delta
         since the last anchor
      -> cutover + weights refill
      -> new layout serves IMMEDIATELY; KV arrives via HiCache/Radix
         read-through as the scheduler admits work

**No carry, no pre-warm.** The fence is only about not LOSING data. There is no
eager re-load of the resident set: warm-up happens by read-through exactly as
it does for any queued request. Mamba/GDN follow the same rule — resume from
HiCache anchors (#745/#747), no eager restore.

**No fallbacks.** Every gap is fixed INSIDE the HiCache route. A conditional
revival of the mover is forbidden.

## The tree already agrees with the premise, in its own words

`mem_cache/hicache_flip_writeback.py:21-23`:

> "a prefix's only way across the flip is the geometry-free STORE (#706): the
> disk tier, whose keys carry content alone and whose pages are cut at read
> time for whichever geometry asks."

and the same docstring states exactly why a mover exists today:

> "device rows survive the flip, because the live row set (radix tree values
> UNION parked requests' rows) is relocated between the two phase pools BY ROW
> ID."

That relocation IS the wave mover. Removing it means the device tier must MISS
and read through — which is what #719's rebind and #718's disarm exist for.

## THE BLOCKER that shapes the build

`managers/phase_flip_resident_carry.py:64-76`:

> "No KV row, no `req_to_token` row, and no mamba slot id is rewritten here...
> both layouts key on the same global slot ids. Every per-request handle a
> carried `Req` holds ... therefore stays valid across the layout swap by
> construction. **The bytes behind those ids are what the KV and GDN movers
> relocate.**"

So `PHASE-FLIP-CARRY` is NOT a KV mover — it carries `Req` scheduling metadata
— and it is NOT a retirement candidate. But its correctness today depends on
the movers filling the same `req_pool_idx` rows in the new pool. **Retire the
mover alone and a carried request's `req_pool_idx` points at unwritten memory.**

Therefore the fence (W1) and the retirement (W5) must land TOGETHER, and the
cutover must leave the new phase's device tier in a state where a lookup MISSES
rather than returning stale rows. That last point is the correctness core of
the whole change and is where a red-first test must bite hardest.

### The correctness core, and why the obvious action is already known to crash

Dropping the tree at cutover -- exactly what "make the lookup miss" needs --
**has been tried and it took the instance down on all three ranks**
(2026-08-23, recorded verbatim at `phase_flip_runtime.py:4590-4620`):

    cache_finished_req -> dec_lock_ref
    -> full_component.py:239  `if cur.id in skip_lock_node_ids`
    AttributeError: 'NoneType' object has no attribute 'id'

with the cause stated in the same comment:

> "PARKED IS NOT UNREFERENCED. The cutover carries RESIDENT requests across,
> and each holds a `last_node` with a lock ref. `reset()` rebuilds the root,
> orphaning those nodes, so the parent walk in `dec_lock_ref` no longer
> terminates at the live root and runs off the top into None."

#825 withdrew the ACTION and kept only detection (`SGLANG_TREE_RECONCILE`,
off). Its own note says the fix "needs to be built against the lock refs --
evicting only the unlocked portion, or reconciling at a point with no resident
reqs -- and that is a design, not a flag flip."

### RESOLVED: the no-KV design removes the precondition of that crash

Both blockers -- the stale `req_pool_idx` and the orphaned `last_node` -- are
the SAME fact wearing two hats: **a resident request carried across the
cutover**. Remove the carry and both disappear, and the user's "no carry" rule
is what removes it.

The mechanism already exists. `retract_all` (`schedule_batch.py:1812`) walks
`release_req` (:1783) over every request, and `release_req` performs exactly
the two releases needed:

    release_kv_cache(req, tree_cache, is_insert=False)   # rows AND lock ref
    req.reset_for_retract()

It takes precisely the objects the flip already holds (`req_to_token_pool`,
`token_to_kv_pool_allocator`, `tree_cache`) and returns the retracted list.

**So the seam order is:**

1. FENCE — `maybe_flip_writeback` (persist to the canonical store) with
   `hicache_demotion` on for the evict-before-persist edge.
2. RETRACT ALL — every resident request releases its rows and its tree lock
   ref. No carried `req_pool_idx`; no locked node.
3. TREE RESET — now safe, because #825's precondition (resident lock refs) is
   gone. The device tier is invalidated, so a lookup MISSES.
4. CUTOVER + weights refill.
5. RE-ADMIT — the retracted requests prefill against their cached prefix,
   served by HiCache/Radix read-through.

Which is the user's sequence verbatim: *phase over, layout stops, KV in
HiCache, layers move, KV from HiCache, layout starts.*

**The carry only ever existed to avoid a re-prefill.** With read-through, that
re-prefill is a cache hit, which is why "no carry" is the correct rule rather
than a simplification — and it is also precisely the "honest warm-up cost as
served-request latency" the user named as the validation metric. That cost is
REAL and must be measured, not assumed small: it is the price this design pays
in exchange for deleting the mover, the staging reserve and the crash class.

### Caveat 1 — DISCHARGED, and it makes the ORDER load-bearing

`release_kv_cache` (`mem_cache/common.py:1749`) routes to
`tree_cache.cache_finished_req(req, is_insert=False)`, and the live
`dec_lock_ref` (`mem_cache/hi_mamba_radix_cache.py:1610`) walks the chain:

    while node != self.root_node:
        ...
        node = node.parent

So the lock ref IS released along the whole parent chain -- **provided the
walk still terminates at the live root.** That is precisely the loop #825
crashed in, once `reset()` had rebuilt the root and orphaned the nodes.

**RETRACT STRICTLY BEFORE RESET.** The order is not stylistic; reversing it
reproduces the 2026-08-23 three-rank crash exactly. A red-first test must pin
the order, not merely the outcome.

(Two paths bypass `cache_finished_req` and must be checked by the build, both
already named in that function: `req_pool_idx is None` under MambaRadixCache,
and `kv_spill_state == "host"`, which routes to the kv-session-offload
manager's own release.)

### Caveat 2 — NOT a clean pass: retraction inflates the policy's own input

Retracted requests go back to the waiting queue, so their full context
reappears as `pending_prefill_tokens` -- the quantity the flip policy compares
against N. But N is priced from X and P, the UNCACHED prefill throughputs
(`DEFAULT_TP_PREFILL_TOK_S` / `DEFAULT_PP_PREFILL_TOK_S`). Retracted tokens
are CACHED by construction: the fence just persisted them, and read-through
serves them. They re-prefill far faster than the bar assumes.

**So every cutover would hand the policy a large pending-prefill figure whose
real cost is a small fraction of what the bar prices it at, and it would do so
in BOTH directions.** That is a thrash pathway, and it is created by this
design rather than inherited: today the carry keeps those tokens out of the
pending count entirely.

**CONFIRMED FROM THE COUNTER, AND IT HAS A MEASURED PRECEDENT.**
`Scheduler._pending_prefill_tokens` (`scheduler.py:10508`) computes

    pending = sum(len(req.origin_input_ids) for req in queued)

the FULL prompt, not the uncached extend. A retracted request therefore
contributes its entire context to the figure no matter how much of it is
cached. There is no prefix-residency term anywhere in that function.

And the identical failure has already been measured once, from a different
cause, in a comment inside that very function (#731, 2026-08-17): a cutover
left a request both resident and queued, so one prompt was counted twice --

> "51,369 -> 102,307 tokens across one cutover, within rounding of exactly
> 2x. The inflated backlog drove the flip policy past its threshold -- six
> cutovers, nothing served."

So an inflated pending figure across a cutover driving the policy into thrash
is a PRECEDENTED failure of this exact code path, not a speculative one. #731
fixed its instance (the carry consumes the queue entry) and deliberately did
NOT add a blanket dedup, on the stated grounds that hiding a real
double-booking would make the class silent again. Retraction re-creates the
same shape by a route #731's fix does not cover, because nothing is
double-counted here -- the tokens are counted ONCE, at a price that is wrong.

This is unresolved and must not be hand-waved. Candidate directions, none yet
evidenced:

* price the pending figure by CACHE RESIDENCY -- count a cached token at its
  read-through cost, not at uncached prefill cost. This is the honest fix and
  it is the same class as #856(b): the decision is only as good as the
  quantity it compares.
* exclude just-retracted tokens from the pending figure for one dwell window.
* lean on the existing guards (`min_dwell_s`, drain mode, the staging rate
  limit) and PROVE by measurement that they bound it -- acceptable only with a
  window that shows it, never by assertion.

Note the interaction with #856(b): the round-trip correction RAISES N
(8.50 -> 18.06 s, N 18614 -> ~39500), which makes this thrash pathway harder
to trigger. That is a mitigating accident, not a fix, and must not be cited as
one.

## Fence coverage — what is and is not already persisted

Under `--hicache-write-policy write_through` most live rows are already staged:
`maybe_cache_unfinished_req` (`mem_cache/common.py:116`) runs synchronously in
`process_batch_result_prefill`
(`managers/scheduler_components/batch_result_processor.py:291,380`) after every
round that produces tokens, and the flip's quiescence predicate
(`phase_flip_runtime.py:608`, `build_flip_quiescence_fn`) requires an empty
`result_queue` and no half-written chunk before a flip may commit. So at the
fence instant the un-hashed tail should be near-empty — **but that is reasoned,
not measured**: no census exists at the quiescent-flip instant. Build item.

Named exceptions, neither closed by the ordinary cadence:

* **`skip_radix_cache_insert=True`** (`mem_cache/common.py:117-132`, warmup /
  bootstrap probes) — rows are deliberately never inserted and are freed at
  completion. Out of scope BY DESIGN; not a flip blocker.
* **`req.kv_spill_state == "host"`** (`mem_cache/common.py:134-146`) — rows past
  `kv_spill_boundary` are already off-device via the SEPARATE kv-session-offload
  mechanism, not HiCache. Orthogonal; must not be confused with read-through.

## Edge case 1 — rows radix-evicted mid-phase before persistence

Already built and already counted: `mem_cache/hicache_demotion.py` writes a
prefix to the canonical disk tier at BOTH eviction seams, non-blocking, bounded,
with `DemotionStats` (`:56-65`) carrying `demoted`, `dropped_backpressure`,
`skipped_not_persistable`, `skipped_no_storage`, `failed`. Its docstring names
the exact gap: under write_through, eviction is where prefixes are lost.

**It is OFF by default.** The fix is to turn it on for phase-flip deployments
(and/or drain it as part of the fence) — not to build anything. Anything beyond
its bound is honest recompute-on-miss, already named and counted rather than
silent. No fallback to the mover.

## Edge case 2 — phase flip with hierarchical cache OFF

Under this design the flip STRUCTURALLY requires HiCache. That must be a
**validate-early launch refusal**, following the exact #806 precedent
(`c0a6347611`, *"Refuse --enable-phase-flip x --disable-radix-cache at launch,
not 15 times at runtime"*), which moved the refusal into
`ServerArgs.__post_init__` after `materialize_declarations(self)`
(`server_args.py:7150-7152`). The new check is the mirror image and belongs
beside the existing `enable_hierarchical_cache` conditionals in the same
`__post_init__`. It must NOT silently revive a mover.

## Retirement inventory

REPLACE / RETIRE verdicts are scoped **to the flip path**. Nothing here is a
bulk deletion, and shared machinery with a named other consumer is KEPT.

| verdict | item | file:line | reason |
|---|---|---|---|
| RETIRE (flip path) | KV wave-mover call sites | `phase_flip_runtime.py` `_flip_waves` :6694, `_labelled_movers`, the wave loop | the flip carries no KV; `read/exchange/write` legs collapse to the fence |
| KEEP | `kv_reshard.py` exchange primitives + #297 domain | `mem_cache/kv_reshard.py` | different consumer (DCP rank-count reshard); only the flip's USE is retired |
| RETIRE (flip path) | `GdnFlipMover.move()` full-state exchange | `managers/gdn_flip_mover.py:504` | packs EVERY live slot's full conv+temporal state on every flip; replaced by anchor resume + fence delta |
| KEEP | `gdn_flip_preconditions()` | `managers/gdn_flip_mover.py` | structural/geometry check; **no other consumer found** — search set: `gdn_flip_mover.py` itself + `phase_flip_runtime.py:3191-3193` import sites only, NOT the wider tree. Verify before deleting. |
| RETIRE (flip path) | `_staging_bytes`'s `wave_peak` term and its funding/refusal chain | `phase_flip_runtime.py:7073-7273` | this is the term behind the 2339.11 MiB `tp_to_pp` staging reserve, the 25 staging-rate-limit refusals and the 17 FLIP ABANDONED in W25 |
| KEEP | `_arena_tail_bytes`, `backing_slack` | same function body | price the WEIGHTS refill commit, independent of `wave_peak` (verified from the function body) |
| REPLACE | fence | `mem_cache/hicache_flip_writeback.py` | already IS the fence (stage + bounded ack drain, `deadline_s` default 2.0 s); extend, do not rebuild |
| REPLACE | evict-before-persist coverage | `mem_cache/hicache_demotion.py` | exists with counters; turn ON |
| REPLACE | read-path switch | `mem_cache/hicache_phase_binding.py:314` (#719 rebind), `phase_flip_rebind_hicache` | flag currently `False`; necessary but NOT sufficient |
| KEEP | #718/#760 disarm | `mem_cache/hicache_phase_guard.py` | unconditional safety; its predicate starts gating correctly per-phase once rebind is armed |
| KEEP | `phase_flip_resident_carry.py` | :64-76 | carries `Req` metadata, not KV bytes — see BLOCKER above |
| KEEP | `kv_row_ownership.py` / #822 authority | — | ownership laws are independent of where the bytes live |
| KEEP | `mamba_ckpt_utils.py` anchor grid | — | exactly what anchor-based resume needs, unchanged |

### Deletable-later (follow-up, NOT part of this change)

A pass over `phase_flip_spill.py`, `corridor_guard.py` and
`funding_authority.py` for KV-wave-specific provider entries once the
`wave_peak` term is gone. **Not re-verified**; recorded as a follow-up rather
than asserted.

## Validation metric — changed

Not "rows carried". The two numbers that matter:

1. **Cutover-blocking time = fence + weights refill.** Extend the existing
   `PHASE-FLIP DONE` stats dict and log line
   (`phase_flip_runtime.py:10201-10251`) with `writeback_fence_ms` from
   `FlipWritebackReport.elapsed_s`, reusing the existing
   `seam_census.mark("flip_writeback")` (`phase_flip_runtime.py:9357`) rather
   than adding a second clock.
2. **Honest warm-up cost as served-request latency after cutover.** **No
   instrument exists** — searched `phase_flip_runtime.py` for `warm.*latency`,
   `post.cutover.*latency`, `first.*decode.*after.*flip`, no hits. This must be
   built, tagged by "rounds since last cutover".

## Tests that pin the OLD contract

Under the new contract a flip that still moves KV must FAIL. The following are
candidates from a filename-scoped sweep — **this is a file list, not a
per-assertion audit**, and is explicitly incomplete:

`test_phase_flip_corridor_gate_631.py`, `test_phase_flip_staging_reserve_631.py`,
`test_phase_flip_arena_tail_631.py`, `test_seam_arena_tail_additive_656.py`,
`test_phase_flip_mover_streaming_631.py` (the `wave_peak`-funded cluster);
`test_gdn_flip_mover.py`, `test_gdn_flip_tree_slots_767.py`,
`test_gdn_payload_trailer_802.py`, `test_mamba_slot_union_801.py`,
`test_mamba_slot_union_device_802.py` (the GDN cluster);
`test_kv_reshard_headroom_363.py:466,517` (the only literal wave/live-slot
assertions found by the narrow sweep).

## Open gaps, named rather than assumed

* **"#706 rows on the full plan" (#735 open item) could not be located.**
  `git log --all --grep='#735'` and `--grep='full plan'` both resolve to a
  DIFFERENT topic: non-contiguous PP layer placement (31/17/16),
  `NOTE_735_step1_contiguous.md`, `NOTE_735_flip_world_step1.md`. No `#706`
  co-occurrence in `docs/dev/*.md`. Either a mis-citation or absent from this
  tree.
* fp8 `kv_cache_dtype` interaction with the canonical page format — not verified.
* Worst-case un-hashed resident tail at the exact quiescent-flip instant — not
  measured.
* GDN worst-case delta bytes (tokens-since-anchor x per-token state) — no
  constant or counter located.
* Host-RAM footprint under #810 staging-ring rules and host-tier read bandwidth
  under lazy warm-up — not independently derived.
