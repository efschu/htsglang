# #739 residual: the admission-wedge alarm is benignly reachable under a big prefill + flip

Filed by strand 2 from the W22 window (2026-08-24, pin `e5a37866d7`, log
`/spinning/evidence-665-f1/boot_w22_0824_0656.log`). NOT a run blocker, NOT a
wedge. A precision defect in a detector, filed so the next reader does not have
to re-decide it from a boot log.

## What happened

The W22 acceptance block said `ADMISSION-WEDGE -> 0`. The window produced 7.
Reported as VIOLATED-as-written, then characterised: they are false positives.

    07:01:12 PP2  age 30.7s   |  07:01:12 PP1  age 30.7s
    07:03:02 PP2  age 25.2s   |  07:03:02 PP1  age 25.2s
    07:03:04 PP0  age 27.1s

## Why they are not a wedge

THE DISCRIMINATOR IS THE AGE TREND, not the alarm's presence.

* Here the age FALLS between occurrences: 30.7s -> 25.2s. A falling age means
  the first-token progress clock was RESET in between, i.e. work completed.
* In the wedge specimen (`wedge_1122_112408`) the age climbs monotonically --
  127.4 -> 218.2s -- and never recovers.
* Corroboration from the same window: 33 cutovers completed, and the load
  driver got HTTP 200 on 123 of 123 requests with health 200 throughout.

The alarm text is honest about what it measures ("NO first token for 30.7s
(>= 20.0s) and no prefill chunk either"). The defect is that the CONDITION is
reachable without a wedge: a 24000-character prompt chunked at 4096 tokens,
overlapping a phase flip whose seam alone measured 7.46s, can legitimately go
25-30s between first tokens.

## The actual residual, and why #739 should have caught it

#739 added a second signal precisely so a visible mega-prefill would not alarm
("prefill is progress too -- stop alarming on a visible mega-prefill",
f9484c3032). These alarms fired ANYWAY. So one of:

1. the pending-token delta is not observed across a flip -- the layout change
   resets or re-scopes the counter the delta is computed from, so a prefill
   that spans a cutover looks like no progress on both sides of it; or
2. the delta check has a gap on the PP path specifically (the alarms are on
   PP0/PP1/PP2 in a `--pp-size 3` layout, and the specimen family is PP too); or
3. the chunk boundary: "no prefill chunk either" is evaluated at an instant
   where a chunk had finished and the next had not started.

Deciding between these is desk work: the log carries the pending-token counts
around each alarm (`pending prefill 21406 tok` lines at 07:07:30 and neighbours)
and the flip timestamps, so the correlation between alarm and cutover is
measurable without a boot.

## What must NOT be done

Do not tune the 20.0s threshold. It was not tuned during W22 on purpose. The
alarm is correct that no first token appeared; raising the number would trade a
false positive for a blind spot in the one detector that names this failure
class, and the specimen it exists for showed 127s+ ages -- far above any
threshold worth setting. The fix belongs in the SECOND signal (#739's delta),
which is what distinguishes "slow because it is working" from "stopped".

## Acceptance for the fix

A hermetic falsifier with the W22 shape: a mega-prefill spanning a simulated
cutover, first-token gap 25-30s, pending tokens strictly decreasing across the
window => the verdict must be silent. Twinned with the specimen shape:
monotonically rising age, pending tokens flat => the verdict must alarm. Both
directions, per the standing rule that a detector is only a finding once it has
been shown to measure what it claims.
