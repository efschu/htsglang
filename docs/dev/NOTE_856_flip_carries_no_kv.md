# NOTE 856 — the flip carries no KV: design, blockers, retirement inventory

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
