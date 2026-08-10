# HANDOFF 668 — successor 25, task #656 / #631 Route A

**In progress. Errors and corrections first, as the chain requires.**

Base: `bf0a38cb1e` (HANDOFF_667, successor 24). Branch `feat/route-a-631`.

---

## 1. CORRECTIONS TO WHAT I INHERITED

### 1.1 Idle free memory is NOT a capacity baseline — it is age-dependent

Measured, not argued. At pool 380000 the live (3.5 h old) process showed
free `3523 / 6712 / 3535` MiB on nvidia-smi index 0 / 1 / 2. After
rebooting to pool **410000** — a strictly LARGER pool, which must consume
MORE — free went **UP** to `3749 / 7058 / 3741`.

The pool really was 410000 (`get_server_info: max_total_num_tokens =
410000`, `pp_stage_ratio [14,10,8]` unchanged), so this is not a boot that
silently ignored the flag. The rise is accumulated allocator cache in the
older process: roughly **520 / 756 / 440 MiB** that torch had taken and
never returned.

Consequence for anyone doing capacity arithmetic here: **a fresh boot's
idle free is optimistic by several hundred MiB per card.** Only the
corridor MINIMUM under a standardised load compares between two pools.
Successor 24's baseline-free term (3852 MiB, inferred as measured-minimum
plus the staging peak added back) is the one number in the ledger a real
boot could contradict, and this is exactly the kind of effect that would
contradict it.

### 1.2 The ~630k figure assumed the staging CONSTANT vanishes

HANDOFF_667 section 4 prices restore-first at `P <= ~630,000`. Re-deriving
it: with the corridor equation

    3852 - (P - 380000)*9.766/1000 - S(P) >= 1024

and `S(P) = c*P/1000 + const`, W=16 restore-first gives `c = 0.610`, so

    P = (6539.1 - const) / 0.010376

`const = 0` gives 630,200 — the handoff's figure. But successor 24's own
measured constant is **357 MiB**, which gives **595,700 — BELOW the user's
600000 floor.** The 630k headline is only reachable if the streaming also
removes the constant, not merely the slope. It is therefore not enough to
reorder the seam; the payload legs have to be row-chunked as well, and the
wave-boundary slack has to go with them.

### 1.3 Prefix-only backing granularity CANNOT stream the seam — the
### direction of travel is the obstacle, and it is not a tuning matter

This is the finding that decides the design, and it took a false start to
see. The owner API (`finalize(rows, buffer_indices)` /
`shrink(rows, buffer_indices)`) does expose a row-count axis, so
"sub-layer granularity already exists" looks true. It is true only for
PREFIXES — `commit_range` grows a contiguous watermark from 0 and
`decommit_range` drops the tail above a keep point.

That is not sufficient, because reading consumes from one end and writing
fills the other:

* process rows ASCENDING — the destination grows as a prefix `[0, t)`,
  fine, but the source still owes rows `[t, N)`, a SUFFIX, so it cannot
  release anything until the very end. Peak `N + N`.
* process rows DESCENDING — the source shrinks as a prefix `[0, t)`,
  fine, but the destination is being written at its TOP, so it needs
  `[0, N)` backed from the first write. Peak `N + N`.

Both orders peak at twice the layout, which is worse than today. The two
lists are index-aligned and both ascending, so the orders are LOCKED
together — one cannot pick ascending on one side and descending on the
other. **A suffix-capable commit is therefore required; it is not
avoidable by scheduling.** Anyone who reads `shrink(rows, ...)` and
concludes the mechanism is already there will lose the same hours.

### 1.4 With layer granularity alone, the target is out of reach

If the seam stays layer-granular, the irreducible transient is one
DESTINATION layer span, and the binding direction is `tp_to_pp`, where a
PP layer spans the FULL pool: `P * 2048` bytes = **1.953 MiB per 1000 pool
tokens**. Whichever rank owns the first layer processed pays it with
nothing yet released, and no ordering removes it (every rank owns layers;
the circularity is that a peer cannot release a source layer until its
owner has written it, and the owner cannot write until its destination
layer is backed). That prices out at roughly **553,000** — short of the
floor. Hence the row-granular design below rather than a cheaper reorder.

---

## 2. THE DESIGN — the row-streamed seam

Three changes, additive and opt-in, so the existing 680-test path is
untouched by construction:

1. **Arena, `KvVmmArena`**: add `commit_span(offset, lo, hi)` /
   `decommit_span(offset, lo, hi)` operating on explicit chunk-aligned
   extent ranges instead of a contiguous-from-zero watermark. Requires
   uniform chunk extents (`commit_chunk_bytes`); without a chunk each
   buffer holds one monolithic extent and interval ops degenerate.
   NOTE: the existing seam-chunk knob `SGLANG_FLIP_SEAM_CHUNK_MIB` also
   switches on `retain_handles`, which PARKS unmapped pages per-arena
   (owned, not driver-free). Parking defeats exclusive backing — both
   layouts would hold their pages continuously. The two must be
   decoupled: chunked extents WITHOUT retention.
2. **Pool**: `release_backing` / `restore_backing` gain a row-range form
   that maps rows to byte spans (layout is row-major, `row_bytes` per row,
   so the map is linear).
3. **Seam**: descending row-block pipeline per wave — the destination's
   backing grows as a SUFFIX just ahead of the writes, the source's shrinks
   as a prefix just behind the reads, and the exchange is row-blocked so
   the payload legs stop scaling with the live set.

Predicted result: the transient becomes `O(block) + extent quantisation`,
a constant in both the live set and the pool. Corridor-limited ceiling
then falls out of resident growth alone.

**UNVERIFIED UNTIL MEASURED.** Everything in section 2 is derived. The
numbers that decide it are the corridor minima from the 410000 run and
from each ladder step, not this arithmetic.

### 2.1 The streaming schedule, and why it balances EXACTLY

This is the part worth reading carefully; it is short, and every previous
attempt to reason about the seam informally got the direction wrong.

Stream over the DESTINATION pool's row space, DESCENDING, in blocks. Let
`p` be the fraction of that row space already processed, counting from
the top.

* the destination's backing is a SUFFIX that grows downward as `p` rises:
  `p * S_dst` bytes are backed;
* the source's backing is a PREFIX that shrinks as `p` rises: `(1-p) *
  S_src` bytes are still kept.

The two cancel because **a rank's PP residency and its TP residency are
EQUAL** — `S_dst == S_src == S`. That identity is not an assumption; it
is forced by the geometry and was already established in HANDOFF_667
section 4: rank `r` holds `|stage_r|` layers over the full pool in PP, and
all 16 layers over its `share_r` token slice in TP, and
`16 * share_r == |stage_r|`. So

    total backed  =  p*S + (1-p)*S  =  S,  for every p.

**Constant, not a slope, and not merely bounded — exactly the resting
layout.** The transient left over is one block of destination backing plus
one block of payload, both chosen by the block size and neither scaling
with the pool or the live set.

Why DESCENDING and not ascending: the source can only give back a prefix
cheaply once the rows above it are read, and reading descends; the
destination can only be written where it is already backed, and its
backing grows downward to meet the writes. Ascending inverts both and
peaks at `2S` (section 1.3).

Why the row spaces line up: the destination row of a slot is monotone in
the slot id in BOTH directions (`pp` row of slot `L` is `L` itself; `tp`
compact rows of an ascending slot list are ascending), so one descending
walk over slots drives both sides monotonically. Rows that are never
written still get backed on the way past, which is required anyway —
the destination must end fully backed because it becomes the resting
layout.

### 2.1b A CHEAPER INTERMEDIATE that may clear the floor on its own

Derived after 2.1 and worth trying FIRST, because it avoids the one
genuinely dangerous change (row-blocking the collective) and is a
contained edit to the wave loop.

Three moves:

1. **Restore-first per wave** — restore the destination wave's backing
   before releasing the source wave's, instead of after.
2. **Lift the wave count to `n_layers` (16).** `default_wave_count` is
   capped at the SMALLEST stage (4 here) only because a wave's releases
   must pay for its commits under release-first. Restore-first budgets the
   overlap explicitly, so the cap goes away. The payload slope is
   `~18 MiB/1000 live slots / W`, so W=4 -> 4.5 and W=16 -> **1.13**.
3. **Choose the wave ORDER so the transient lands on the card with the
   most headroom.** This is the part that is not obvious. Per wave, rank
   `r` restores its destination layers (in `tp_to_pp` a PP layer spans the
   FULL pool, 1.953 MiB/1000 pool tokens) and releases the wave's TP layer
   (`share_r` of that). They balance over the whole flip but not wave by
   wave, so the peak is

       max_j [ 1.953 * M_r(j)  -  1.953 * share_r * j ]

   where `M_r(j)` counts rank `r`'s destination layers among the first
   `j+1`. Whoever owns the FIRST layer processed pays a full layer with
   nothing yet released — that payment is unavoidable, but it is
   ASSIGNABLE. Put rank 0's layer first (the 5090, the card with by far
   the most headroom) and delay each 3080's first owned layer to
   `j0 >= 1/share_r` — with `[7,5,4]` that is `j0 >= 4` for rank 1 and
   `j0 >= 5` for rank 2 — and both BINDING cards pay approximately
   ZERO transient, while the 5090 absorbs ~1.95 MiB/1000 pool tokens.

Predicted ceiling on successor 24's inferred baseline: about **594,600**
— just under the floor. On the corridor minima actually measured here it
may clear 600000; that is a question for a boot, not for more arithmetic.

**Why this is low-risk despite touching the seam.** Restore-first does
not change WHICH bytes are read or written, nor in what order — only when
physical pages are mapped. Byte identity is therefore preserved by
construction, and the existing byte-identity tests
(`TestByteIdentity`, `TestSeamWavesAreByteIdentical`,
`TestSharedArenaReadsPrecedeWrites`) are the net that proves it. The
tests that MUST go red and be rewritten to the new contract are the
order pins: `SeamOrderingTest` (three methods asserting
release-then-reclaim-then-restore) and
`TestPreWriteSeamOrdering::test_hook_fires_between_last_read_and_first_write`.
Aliased pools must keep the OLD order — `_pools_alias()` already gates
that, and it must stay gated.

### 2.2 What is BUILT and what is NOT

Built, tested, committed:

* `KvVmmArena.commit_span` / `decommit_span` — arbitrary chunk-aligned
  extent ranges, asymmetric rounding (commit outward, decommit inward),
  contiguous-from-zero watermark refresh so the legacy prefix path stays
  correct while a span op has left a hole. 10 tests, red first.
* `KvVmmBufferOwner.back_token_span` / `release_token_span` — the
  row-to-byte map, with the asymmetry pinned so it cannot be "tidied"
  into one helper. 5 tests.
* `HostKvPool.release_backing_span` / `restore_backing_span` — the
  layer-subset entry point.

NOT built — this is the remaining work and it is the risky half:

* the streamed `_execute` loop itself (section 2.1);
* row-blocking the EXCHANGE, which needs a GLOBAL round count so all
  three ranks call the collective the same number of times. Derive it
  from the replicated plan (`ceil(max_r |slots owned by r| / block)`),
  never from a rank-local row count — a rank-local count deadlocks the
  group, and that failure mode looks exactly like a hang;
* `_staging_bytes` re-derived for the streamed peak;
* the chunked-extent requirement. `SGLANG_FLIP_SEAM_CHUNK_MIB` currently
  also switches on `retain_handles`, which PARKS unmapped pages per-arena
  (owned, not driver-free). Parking defeats exclusive backing outright —
  both layouts would hold their pages continuously — so the two MUST be
  decoupled before the streamed seam can use chunked extents. This knob
  is a trap in its present form.

---

## 2.3 THE TENSION THAT EXPLAINS WHY THE SEAM WORK IS NOT OPTIONAL

Stated in the user's own terms, because it is the clearest argument for
doing section 2.1b and it is not obvious from any single measurement.

The corridor law has two halves: never breach 1024 MiB free per card, AND
be well filled (free NEAR 1024, not multiple GiB above). **With the
current seam those two halves cannot both be satisfied.**

* Size the pool so the corridor is well filled at TYPICAL occupancy and
  the pool goes past ~438,000. A flip at FULL occupancy is then
  unaffordable, and under strict purity an unaffordable flip does not
  degrade, it WEDGES (HANDOFF_667 section 1.1).
* Size the pool for anti-wedge safety at ~430,000 and the corridor sits
  loose whenever occupancy is normal — the acceptance run below holds its
  minimum around 2650 MiB on the binding card, some 1600 MiB above the
  floor.

The cause is that staging scales with the LIVE SET while the pool must be
sized for the worst case, so the headroom reserved for a full-occupancy
flip is idle at every other moment. Removing the occupancy-dependent term
is exactly what sections 2.1 and 2.1b do, and it is what collapses the two
halves back into one operating point.

**So "the corridor is loose" is not slack left on the table by
carelessness — it is the price of the current seam, and it is the same
price as the ~600000 shortfall.** One fix buys both.

## 3. WHAT THE CAPACITY LADDER SETTLED

Full tables in PROD_BRINGUP_BENCH section 2g. The three results that
matter:

1. **The ~432,000 anti-wedge ceiling is CONFIRMED, by a third independent
   method.** My own measurement — baseline free on the binding card
   excluding staging, taken as `min_free + staging_reserved` at pool
   500000 — gives **437,862**, against successor 24's 432,861 and the qwen
   log regression's 437,235. Within 1.2%. The inferred-baseline term that
   successor 24 flagged as his weak link has now been measured and did not
   move the answer.
2. **A 500000 boot holding the corridor is NOT a refutation of that
   ceiling**, and reading it as one would have been this chain's sixth
   false closure. I nearly made it. The ceiling is defined at FULL
   occupancy; that run's live set peaked at 131,288 slots, **26% of the
   pool**. The deciding term was never exercised. What the run did settle
   is the staging model, in the ledger's favour: 1047.6 MiB measured at
   131,288 live slots against 950 predicted.
3. **The honest maximum for the current seam is ~435,000**, because one
   long request is enough to reach the unsafe region on its own: at pool
   500000 a 393,216-token request prices staging at 2132 MiB against 2752
   MiB of baseline — a breach — while at 430000 the same request lands at
   1304 MiB free.

The user's **>= 600000 (spec item 6) is NOT met and is not reachable by
stepping the pool.** It is blocked by the staging SLOPE, not by the
intercept, which is why the remaining work is the seam and not a bigger
number in the launch command.

## 4. STATE AT HANDOFF

* Branch `feat/route-a-631`, pushed to the fork.
* Flip family: **700 passed, 1 failed** — the pre-existing
  constant-dominated `_staging_bytes` red, newly VISIBLE because I fixed
  the runner's collection gap. It is documented in the commit that
  exposed it, not silenced. Baseline before my changes was 680/680 with
  that file never collected.
* Serving at pool **430000**, the acceptance configuration.
* GPU arbitration holder is mine; heartbeat running.

### 4.1 The arbitration collision, and the rule that came out of it

At 15:23:22Z successor 24 — believing this strand had finished — rebooted
serving from my live 500000 experiment down to 410000, killing the run
mid-flight. It checked for load PROCESSES and for traffic in the serving
log, and concluded the rig was quiet. It never listed
`/spinning/gpu-arb/heartbeat.*`, where my heartbeat was 31 seconds old.

**STANDING RULE, now in the holder: before ANY serving reboot, list
`/spinning/gpu-arb/heartbeat.*` by mtime and treat anything fresher than
120 s as a live peer.** Process checks and log greps do not cover it.
Successor 24's own account is preserved at
`/spinning/gpu-arb/holder.s24-collision-note`.

## 5. WHAT I DID NOT GET TO, in the order I would take it

1. **Section 2.1b, the restore-first + W=16 + ordered-waves change.** The
   design is complete and quantified, including which tests must go red.
   Try it BEFORE 2.1 — it avoids row-blocking the collective, which is
   the one change that can deadlock the group.
2. **Section 2.1, the fully streamed seam.** The arena and pool
   substrate for it is built and tested; what remains is the `_execute`
   loop, the global round count, and re-deriving `_staging_bytes`.
3. **Decouple `SGLANG_FLIP_SEAM_CHUNK_MIB` from `retain_handles`.**
   Chunked extents are required by both routes; handle retention parks
   pages per-arena and defeats exclusive backing. Today they are one knob,
   so the knob is a trap.
4. **The graph A/Bs of spec item 8** — NEXTN draft graphs, DFLASH x
   graphs, PP-prefill graphs, and the prefill chunk ladder. Untouched by
   me. Decode graphs and strict purity ARE verified on metal (section 6).
5. **The recoverable abandon** (HANDOFF_667 section 1.1c) and the
   **wave-count floor guard** (1.2). Both still unbuilt.
