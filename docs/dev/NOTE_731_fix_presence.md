# NOTE 731 — the fix is PRESENT on the harvest lineage; closing with evidence

The register said "Fix-Bau laeuft" with no landing commit. It landed:
**`fdcf837206`** ("[#731] A request must exist in exactly one place after a
cutover"), authored by F4-r5 2026-08-17 15:42, and it is an ancestor of BOTH
`c546eed923` and F4-r5's current harvest tip `da818719fe` (verified with
`git merge-base --is-ancestor`). My later interaction pin `fb861cd07f` is on the
tip as well.

No build was needed. This is the second time this week the prior-art gate
turned a build ticket into a determination.

## All three converged parts, at file:line

| part | site | evidence |
| --- | --- | --- |
| 1. carry CONSUMES the queue entry | `phase_flip_resident_carry.py:702` `_consume_carried_from_waiting_queue`, called at `:689` | present |
| 2. counter de-duplicated at the intersection | `scheduler.py:8767` "#731: THE TERMS BELOW MUST NOT RE-BILL WHAT THE QUEUE ALREADY DID", `:8821` "the waiting-queue term already billed it" | present |
| 3. `duplicate_resident_reqs` universe + waiting_queue | `phase_flip_resident_carry.py:380` emits a `queued:` prefix so the resident-vs-resident and resident-vs-queued shapes stay distinguishable (`:353`) | present |

Part 2 is de-duplicated **at the intersection**, not by a blanket per-rid rule —
deliberately, so a request genuinely holding budget in two places stays visible
rather than being silenced the way this defect was.

## The specified hermetic red-first exists, with the specimen numbers

`test_carry_queue_duplication_731.py:184`
`test_a_request_in_both_sets_is_counted_once`, docstring:

> RED before the fix: this returned 2x the prompt (51369 -> 102307).

`51369` is used verbatim as the fixture prompt length. Its complements are
there too, which is what stops the dedup from becoming a blanket suppressor:
`test_distinct_requests_still_add_up`, `test_queued_only_is_counted`,
`test_resident_only_with_unprefilled_tail_is_counted`,
`test_an_already_queued_arrival_is_not_counted_again`.

**Run: 23 passed** (`test_carry_queue_duplication_731.py` plus the
`#731 x #744` interaction pin `test_carry_parked_extent_interaction_731_744.py`),
hermetic, `CUDA_VISIBLE_DEVICES=99`.

## The combined arithmetic with #677 — the churn's full shape

#731 inflates the numerator (apparent backlog = 2x true). #677 deflates the
denominator (break-even 7,004 instead of 49,248). They multiply:

|  | arms when true backlog exceeds |
| --- | --- |
| both defects live | **3 502 tok** |
| both fixed | **49 248 tok** |
| combined eagerness | **14.1x** |

Against the soak driver's own ceiling — 4 sessions capped at ~48 000 chars
(~12 k tokens) each, so ~48 000 tokens if all four are simultaneously full:

* **before:** the trigger sits at **7.3 % of the ceiling** — essentially always
  satisfied, which is the churn;
* **after:** the trigger is **103 % of the ceiling** — it exceeds what this
  workload can produce at all.

**Prediction for the harvest boot, and the caveat that comes with it:** with
both fixes the ECONOMIC window should arm **never** on the soak workload. Any
flip that still occurs is therefore attributable to the IDLE-LOCK escape path
(#759), not to the economics — which makes the two paths cleanly separable in
the log for the first time. If flips are still wanted on this workload for
other reasons, the economic path will not supply them at this cost; that is a
consequence of pricing the flip honestly, not a regression.
