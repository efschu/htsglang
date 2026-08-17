# TICKET 746 — the parked extent should be exact, not last-seen


> Register task **#746**. Filed briefly as "TICKET 745", which collided with
> register-#745 (GDN checkpoints) — renamed. Nothing here relates to GDN.
Follow-up to **#744** (`5085766fa9`), filed by its author against its own
known limitation. Small, desk-sized, not urgent — #744 is safe without it.

## What #744 ships

`_flip_pending()` answers `(rows, max_row_id)` for the requests a phase flip
has parked, and both the trigger (`_nothing_resident`) and the safety net
(`_max_live_row`, read by `_shrink_to`) consult it. The value comes from a
sticky attribute on the flip's own live-slot enumeration: the **last
enumeration that saw requests**, remembered in `build_flip_live_slots_fn._live`
and consulted only while a flip is armed.

## Why that is not exact

The correct answer is "the rows this flip will pack", which is fixed at ARM.
What is returned is "the rows the most recent non-empty enumeration saw", which
is the same thing in the common case and diverges in two:

1. **A flip arms before any enumeration has seen requests.** There is no sticky
   value, `_flip_pending()` answers `(-1, -1)` = UNKNOWN, and both consumers
   block. Safe, but it blocks the rung for the whole flip on no evidence.
2. **The resident set changed between the last enumeration and the arm.** The
   remembered extent is then stale — too large (blocks more than needed, safe)
   or too small. Too small is the one worth naming: it would under-report the
   parked rows. Line 2 (the armed gate) covers this case today, which is
   exactly why #744 shipped two independent lines rather than one.

Neither case is a correctness hole while the armed gate stands. Both are
reasons the gate cannot be removed later without doing this first.

## The fix

Snapshot the extent at arm rather than inferring it. The flip already measures
what it needs at that moment — `FLIP EXTENT PROBE` logs
`seqlen=... kv_allocated_len=...` per request at arm time, and
`_probe_allocated_extent` runs there. Capture `(rows, max_row_id)` into the
controller when the flip arms, clear it when the flip commits or is abandoned,
and have `_flip_pending()` read that instead of the sticky enumeration value.

Acceptance:

- a flip that arms with no prior enumeration reports the real extent instead of
  UNKNOWN (case 1 above turns from "blocks" into "blocks exactly what it
  should");
- the snapshot is cleared on BOTH exits — commit and abandon. A snapshot that
  outlives its flip pins the rung permanently, which is the M5 failure mode
  #744's mutation matrix already refuses;
- `test_evict_rung_flip_park_744.py` stays green unchanged, including
  `test_THE_RUNG_IS_NOT_DEAD_OUTSIDE_FLIPS`.

## Do not

Do not use this as a reason to drop the armed gate. The two lines are
deliberately independent because the failure mode is a silent eviction followed
by a delayed fault — nothing between the two says anything is wrong, so a
single mechanism has no second chance.

## Built (fix/746-arm-time-snapshot)

`arm()` measures the extent through the flip's own live-slot enumeration at
the arm instant (`_snapshot_parked_extent`) and stores it on the controller;
`parked_extent()` serves it gated on `_pending`, and every exit -- the commit
and all four abandon paths -- clears it. The rung's `_flip_pending` probe
reads that snapshot; the sticky last-enumeration channel is removed at writer
and reader both. #748's exclusion ceiling and its UNKNOWN-while-armed
wholesale refusal stand untouched; UNKNOWN is now confined to a flip whose
arm-time measurement itself failed.

Acceptance status:

- arm-before-enumeration reports the real extent: BUILT, driven through the
  real `arm()` in `test_flip_arm_snapshot_746.py` (idle arm = exact
  `(0, -1)`, populated arm = exact rows/top);
- cleared on BOTH exits: BUILT, behaviorally for the three hermetically
  callable abandons plus an AST pin that every `_pending = None` site clears
  the snapshot, plus the accessor's own `_pending` gate as second defence;
- `test_evict_rung_flip_park_744.py` green unchanged, including
  `test_THE_RUNG_IS_NOT_DEAD_OUTSIDE_FLIPS`.

Mutation matrix (production code mutated, suite must go red): N1 arm does
not snapshot -> 2 failed; N2 accessor ungated -> 1; N3 commit does not clear
-> 1; N4 abandons do not clear -> 6; N5 snapshot survives flip (M5 analog,
N2+N3+N4) -> 8; N6 measurement failure read as empty -> 2; N7 rung reads
missing snapshot as empty -> 1. All killed.

Not desk-provable: a metal flip arming with the snapshot while the rung
funds the seam from rows above the parked ceiling (the #748 unstrangle,
now exact). Rides the review boot / a later window arm.
