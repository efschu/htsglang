# HANDOFF 670 — #656 / #631 Route A, successor 27

Predecessor: HANDOFF_669 (successor 26). Read that for the 2.1b design
and its ceiling. This file reports what happened when section 2.1 was
first actually RUN, which is not what the note predicted.

---

## 0. THE ONE-LINE VERSION

**Section 2.1 had never executed a single instruction on this rig, and
when it finally did it killed the instance.** Four defects stood between
the shipped code and one working flip; all four are fixed, tested and
pushed. The row-block A/B is the first measurement of the feature that
exists.

---

## 1. ERRORS FIRST — WHAT THE INHERITED CODE ACTUALLY DID

HANDOFF_669 shipped `_stream_wave` "dark, pending its A/B" and priced
section 2.1 as the route to the >=600000 floor. The honest status was
worse than dark: **unrunnable, in four independent ways.** None is
visible from the unit tests, and I want the reasons recorded because
they are all the same shape — a probe that cannot observe the thing it
claims to check.

### 1a. No commit chunk on any boot this rig takes (`0de295bb29` prereq)

`commit_span`/`decommit_span` raise unless the arena was built with a
commit chunk (`KvVmmArena._require_chunk`): an unchunked arena maps one
monolithic extent per buffer, and `cuMemUnmap` only takes whole
mappings. `SGLANG_FLIP_SEAM_CHUNK_MIB` defaults to **0**. So on a default
boot the streamed seam raises — inside the flip's no-return region.

### 1b. The gate answered from `hasattr`

`is_span_swappable()` checked that the METHOD existed. It always does.
It now asks the pool whether its ARENA can do span ops. A false no costs
a slower seam; a false yes costs the instance.

### 1c. The chunk forced handle retention, which defeats exclusive backing

Setting the chunk also set `retain_handles=True`. Retention parks
released handles in `KvVmmArena._retained` — **per arena**, and the
flip's two layouts are two arenas. The PP arena parks precisely the
pages the TP arena then asks the driver for, and `_take_retained` on the
TP side cannot see another arena's park. Both layouts stay resident and
exclusive backing — the entire reason this pool is VA-backed — is gone.
`seam_chunk_and_retention()` separates them; retention defaults off and
stays reachable via `SGLANG_FLIP_SEAM_RETAIN_HANDLES=1` for a
single-arena user (the #330 dial does recycle its own handles).

### 1d. `HybridLinearKVPool` never forwarded the span surface (`42fa5ad423`)

**This is the one metal found.** The object the flip holds is not the
pool that owns the arena; it is a wrapper that forwards `release_backing`
and `restore_backing` and forwarded NEITHER span variant. So even with
1a–1c fixed, `is_span_swappable` looked straight past a capability the
underlying pool had and took the whole-wave branch on every flip without
saying so.

Evidence, boot 2: the tune file was read nine times, the block count
reached 32, and the census recorded **zero** `backing_*_span` marks with
an unchanged staging reservation at every arm. A capability probe a
wrapper can drop is a capability that turns itself off.

### 1e. And then it crashed: `commit_span` mapped over its neighbour (`0de295bb29`)

First boot on which the streamed path engaged, 18:07:40, all three ranks:

    _stream_wave -> restore_wave_span -> commit_span
    RuntimeError: cuMemMap failed: CUDA_ERROR_INVALID_VALUE
    SIGQUIT received. It usually means one child failed.

`commit_span` rounded its range outward to the **commit chunk** (16 MiB)
while `commit_range` has always rounded to the allocation
**granularity** (2 MiB). Buffer VA extents are laid out
granularity-aligned, so a chunk-rounded `hi` overshoots the end of its
own buffer by up to chunk-1 bytes and asks the driver to map over the
NEXT buffer's live mapping.

**The chunk is a handle SIZE. It was never an alignment.**

It survived "built and tested" because the legacy whole-pool path never
produces a span ending anywhere but at a buffer's own end, and because
the span substrate's tests never ran against a real arena with a
neighbour to collide with. Both were true; neither was sufficient.

### 1f. A residual that GREW with the block count

`decommit_span` rounds inward, so for an interior boundary block `b`'s
`hi` rounds below it and block `b+1`'s `lo` rounds above it: the chunk
straddling it was released by neither — `(B-1)` chunks per buffer left
mapped, working directly against the 1/B the loop exists to win.
`_stream_wave` now releases `[0, hi)` cumulatively, which is sound
because `_execute` drains the retained leg and the exchange before the
seam opens, so no source row is live anywhere in that loop.

### 1g. A cost the cumulative release carries, measured small but not free

Releasing `[0, hi)` per block makes each call walk that buffer's whole
extent list, so the seam is `O(B x extents)` rather than `O(extents)`.
On this rig at B=16, chunk 16 MiB, pool 500000 that is roughly 2.6M list
steps per flip and shows up as the +4% flip latency -- acceptable, and
measured rather than assumed. It scales with `B x pool / chunk` though,
so a much larger pool or a much smaller chunk could make it matter. The
O(1) form is available if it ever does: release only
`[prev_hi - granularity, hi)`, which covers the one straddling granule
and nothing already gone. Not done here because it would be an
unmeasured optimisation of a measured-acceptable cost.

---

## 2. THE ACCOUNTING HAD TO MOVE WITH THE LOOP

`_backing_slack_bytes` charged a whole layer span whatever the block
count. Left alone, the shrink would have been real and **never cashed**:
the gate is what refuses a flip, so the pool stays capped exactly where
it was and the change measures as inert. Per wave the peak is now

    max over b in [0, B) of ((b+1)*com - b*rel) / B

which collapses to `com` at B=1, keeping the block count a one-variable
A/B. `_effective_row_blocks` mirrors `_execute`'s branch conditions
rather than reading the knob, because pricing blocks the loop is not
running under-reserves and the gate is the last check before the
no-return region. A chunk-granularity floor stops the model claiming a
shrink the driver cannot deliver.

This is the third time in this chain that order/count/accounting had to
move as one unit (HANDOFF_669 section 2.2 was the second). Treat it as a
standing invariant of this file, not a coincidence.

---

## 3. A CORRECTION TO MY OWN FIRST READ — THE ORDER IS NOT BROKEN

I spent a sweep concluding that `ordered_layer_waves` was leaving 36% on
the table, because my sweep reported `max over all ranks`. That is the
wrong statistic and it produced a wrong conclusion.

`ordered_layer_waves` assigns the transient **deliberately** to the
largest-share rank — the 5090, the one card with GiB of corridor slack —
and minimises the BINDING ranks in pass 1. Split per rank at pool
600000:

| B | rank 0 (5090, absorber) | rank 1 (3080) | rank 2 (3080) |
|---|---|---|---|
| 1 | 1831.1 | 585.9 | 585.9 |
| 4 | 1446.5 | 311.3 | 366.2 |
| 16 | 1350.4 | 242.6 | 311.3 |
| 32 | 1334.4 | 231.2 | 302.1 |

The binding cards pay 586 MiB at B=1 and **~300 MiB at B=16**, a 2×
cut where it matters. `s27_seam_ceiling_sweep.py` now reports per rank
and refuses to collapse them, with the reason in the code.

**Desk model validated on metal to 0.05%.** Pool 500000, B=1:
predicted 1525.9 / 488.25 / 488.25 MiB, measured **1526.11 / 488.37 /
488.50**. The same model reproduces HANDOFF_669's independent desk
ceiling (473,157) at 473,085 — 0.02%.

---

## 4. WHAT >=600000 ACTUALLY NEEDS

With a 16 MiB chunk the floor binds at B=16 and further blocking buys
nothing. At pool 600000 the binding card then needs

    payload leg          687 MiB   (HANDOFF_669's measurement)
    backing transient    325 MiB   (was 586 at B=1)
    -------------------------------
    total               1012 MiB   against a 753 MiB budget

So row blocking closes roughly half the gap and **does not on its own
reach the floor.** The remaining ~259 MiB has two candidate routes, and
the second is the one the user's spec actually names:

1. **Row-block the EXCHANGE**, which is where the 687 MiB payload leg
   lives. Deferred by successor 26 as the riskier half because it needs
   a GLOBAL round count — derive it from the replicated plan, never
   rank-local, because a rank-local count deadlocks the group and looks
   exactly like a hang.
2. **Spill rung 2 (draft weights) at the seam — START HERE NEXT SHIFT.**
   Spec item 6 names spill depth as the mechanism for full KV and
   explicitly accepts a longer flip. `phase_flip_spill.py` has
   `DEPTH_DRAFT_WEIGHTS = 2` with a written `DraftWeightSpill` class
   (spill / restore / `payload_mib`), but
   `IMPLEMENTED_DEPTH = DEPTH_ALLOCATOR_CACHE = 1`, so rungs 2 and 3
   parse and are then refused.

   **Its own docstring prices the payload at ~2 GB per rank** ("~2 GB of
   scattered per-tensor storages become one block"). The shortfall at
   600000 is 259 MiB. If that number survives contact with
   `payload_mib`, rung 2 does not merely close the gap, it closes it
   roughly eight times over — and it is the mechanism the user asked
   for rather than a substitute for it.

   The fit is better than arithmetic. Under STRICT PURITY the drafter is
   used only for MTP decode, and decode happens only in the TP layout,
   so the drafter is genuinely IDLE for the whole PP phase. The spill
   window and the unused window are the same window; this is not a
   trade of latency for memory, it is an asset that was resident for no
   reason.

   Two cautions. This is the same shape of inheritance as the dark seam
   — written, unreferenced, untested — so **measure `payload_mib` before
   building on it**, and read section 1 of this file first. And its
   docstring calls itself "RUNG 1" while the ladder has it at 2; the
   numbering drifted when the allocator-cache rung was inserted below
   it, so trust `DEPTH_DRAFT_WEIGHTS`, not the prose.

The seam's staging need is a SEAM-INSTANT need, not a resident one,
which is exactly what a seam-time spill is for. That makes route 2 the
better-matched lever as well as the user-specified one.

---

## 5. ALSO DONE THIS SHIFT (user directives, 2026-08-10)

**First-chunk dynamic chunking (`a9aae4dd1e`).** The gate was
`self.chunked_req is not None`, so only an in-flight partial prefill got
a predicted size. A prompt short enough to finish in one chunk never
becomes a `chunked_req`, so the entire class of requests the predictor
could serve end to end was excluded, and for long prompts the chunk that
sets the pipeline's opening bubble was the one taken blind. The
predictor only ever needed a `history_len`, which is 0 for a prefill
that has not started. Moved into `Scheduler.dynamic_chunked_prefill_size`
so it can be pinned directly rather than through a batch path where a
branch above can return early and make the assertion vacuous. Refusals
kept and tested (feature off; predictor not ready) — "always dynamic"
would have passed the positive tests and broken every boot.

**The graph baseline for item 8, read off this boot so nobody spends a
boot finding it.** Captured today: draft DECODE, draft EXTEND, draft
VERIFY. Disabled: PREFILL (`cuda_graph_config` resolves
`prefill.backend='disabled'`). Capture cost 0.12-0.30 GB per rank per
kind.

That inverts the obvious reading of spec item 8. The user's instruction
is to MEASURE draft graphs and leave them out if NEXTN gains nothing --
and they are currently ON, so the item-8 A/B is a REMOVAL experiment,
not an addition one, and the null hypothesis is that ~0.3 GB per rank of
corridor is being spent for nothing. That memory is the same corridor
the seam is fighting for, so a negative result there is worth as much to
item 6 as it is to item 8.

**Threshold-decode arm (spec item 10)** is specified and the flag is
confirmed present and parsed (`phase_purity.parse_purity`,
`threshold:<n>`); it is a boot-time server arg so it needs its own boot.
NOT YET MEASURED.

---

## 6. STATE, AND WHAT I WOULD DO NEXT

Commits, all pushed on `feat/route-a-631`:

    b10319b8d9  section 2.1 prerequisites + block-aware accounting
    a9aae4dd1e  first-chunk dynamic chunking
    42fa5ad423  HybridLinearKVPool span forwarding
    0de295bb29  commit_span granularity alignment (the crash)

Flip family **738 passed / 1 failed**; the single red is the inherited
pre-existing `_staging_bytes` over-reservation in
`test_phase_flip_mover_streaming_631`, untouched here (baseline 723/1).
ruff clean on every touched file.

Next, in order:

1. **DONE: the row-block A/B.** Clean same-boot, same-direction arms at
   pool 500000 (bench 2j). Binding 3080s: 488.67 -> 305.55 -> 276.51 MiB
   at B = 1, 4, 16. **B=32 returns exactly the B=16 numbers** because the
   16 MiB chunk floor binds there, and costs 8% more latency for it, so
   **B=16 is now the default** -- the last count that buys anything, the
   largest that costs nothing. +4% flip latency against the floor arm.
   Inert until `SGLANG_FLIP_SEAM_CHUNK_MIB` is set, which is still 0:
   do not move that default until a LOADED corridor run exists at B=16,
   because every number here was taken at 90 live slots and prices the
   seam's constant, not its behaviour under a full pool. That loaded run
   is the single cheapest piece of evidence still missing.
2. **Price spill rung 2 before building on it** — `payload_mib` per rank
   is the whole question, and section 1 is the reason not to trust the
   written-but-dark code until it has run once.
3. **Graph A/Bs (spec item 8)**, still untouched through eleven
   successors. Now carrying two extra arms: the fixed dynamic-chunking
   arm with engagement proof, and the threshold-purity arm.
4. Final all-axes acceptance at whatever pool the seam supports.

### Traps that cost me time, so they do not cost the next shift

* **`seam_scaling_reboot.py` needs a LIVE server** to replay from. When
  the instance is dead you need `--from-capture <cmdline> <env>`, and no
  raw capture is saved — only the human-readable `replay-*.txt` under
  `/spinning/evidence-631/boot-captures/`. Its `baseline argv:` and
  `baseline env:` sections reconstruct into the two files the flag
  wants; `/spinning/evidence-631/s27/{cmdline,env}.txt` are ready-made.
* **A backgrounded reboot can lose the log.** One boot's server inherited
  a task-output fd instead of `/spinning/serving-30030.boot.log`, and the
  A/B then measured a log nothing was writing to — reading as "the
  feature did nothing". Check the log is GROWING before believing a null
  result. `setsid ... < /dev/null` fixed it.
* **`pkill -f` self-matches** and killed my own shell (exit 144). It is
  forbidden by the brief anyway. Do not reach for it.
