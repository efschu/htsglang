# TICKET 745 — the parked extent should be exact, not last-seen

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
