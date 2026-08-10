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

## 3. STATE

* Serving rebooted to pool **410000** on the inherited commit, health 200,
  geometry `pp_stage_ratio 14,10,8` unchanged, single variable moved.
* Flip-family suite at base: **680 passed, 0 failed**.
* GPU arbitration holder is mine; heartbeat running.
