# ANALYSE #536 — fast-lane starvation: root re-verified here, remedy already built dark

Verdict: **the root is confirmed on THIS lineage, the suggested fix direction is
falsified by the code's own docstring, and the decided remedy is already built
with tests on `fix/536-fast-lane-reserve` — dark, unwired, and not on this tip.
Determination, not a rebuild.**

## 1. Root, verified on the control tip (not assumed)

The brief's instinct was right and is now checked at code: the fast lane waits
for an **entire prefill drain**, not a chunk boundary.

**Ordering is already correct**, so it is not the cause.
`_sort_by_priority_and_fcfs` (`schedule_policy.py:382`) sorts the fast tier
first, and #552's aging promotes an aged heavy request only to
`fast_lane_priority - 1` — deliberately *below* the fast tier. The docstring
says why, and it falsifies the proposed remedy in advance:

> "a heavy req cannot preempt, so promoting it ABOVE fast would only wedge the
> admission loop and starve the fast lane"

**Preemption exists but cannot reach the held memory.**
`SchedulePolicy.preempt_to_schedule` (`schedule_policy.py:1614`) iterates
`self.running_batch.reqs` — running requests. A co-tenant heavy prefill under
chunked prefill is being filled a chunk per round; its pool draw is not a
running decode this loop can evict. So the fast request is first in the queue,
the preemption path finds nothing it may take, and it waits until the prefill
finishes. That is the 34.5 s: a drain, not a boundary.

The mechanism in one line, unchanged from the accepted #536 note: **priority
orders the queue; it cannot release memory another request holds.**

## 2. Why the proposed fix direction must not be built

"Preemption point or priority admission at a chunk boundary" is the option the
code and the accepted premise correction both rule out:

* promoting anything above the fast tier "would only wedge the admission loop
  and starve the fast lane" (`schedule_policy.py:394-398`);
* the fast lane is *already* first, so priority admission has nothing left to
  give it;
* the thing it lacks is memory, and no ordering change produces memory.

Hence a **reserve**, not chunk preemption — which is the direction already
decided and already built.

## 3. The remedy exists, dark, on another branch

`98125f1436` on `fix/536-fast-lane-reserve`:
`managers/fast_lane_reserve.py` (pure, hermetic) +
`test_fast_lane_reserve_536.py` + `docs/dev/WINDOW_TICKET_536.md`.
`solve_fast_lane_reserve` prices the reserve with provenance (the operator's
declared fast-lane prompt ceiling, else one admission chunk — never a magic
number); `admissible_tokens` is the rule: a heavy request sees the pool MINUS
the reserve. It is **default-off and unreferenced by design**, because the
operator decision gated it on a lane-ON/OFF live observation that does not
exist yet.

It is **not** in this branch's ancestry.

## 4. Why the specimen still occurs — defect, not misconfiguration

Not a mis-set flag. On this lineage the remedy is not merely off, it is *absent*
— never wired into the admission path and not present in the tree. The
2026-08-04 specimen predates the module entirely. A fast-lane request behind a
co-tenant prefill today takes the same path it took then.

What *is* here is the **reverse-direction protection**, already built and
flag-gated: `preempt_to_schedule` will not preempt heavy requests below
`--fast-lane-reserved-heavy-slots` (`schedule_policy.py:1650-1663`), so an
armed fast lane cannot starve the prefill in the other direction. Item 3's
"pin the reverse direction" is therefore already satisfied in code.

## 5. Interactions (item 4)

**#617 dynamic chunks — engaged, as the reserve's floor.** The reserve's
fallback price is *one admission chunk*, "because a chunk is the largest slice
the admission path takes in one pass and anything smaller cannot admit even
one". So a dynamic chunk size moves the reserve's floor with it; the two are
coupled by design and the coupling is priced, not incidental.

**#689 batch formation — different quantity, no contention.** Formation decides
*order and shape*; the reserve decides *how much of the pool a heavy request may
draw*. The same separation that keeps #552 and #536 apart: #552 acts on ORDER,
#536 on MEMORY, and the reserve module deliberately contains no ordering
vocabulary so the two cannot start contending for one quantity by accident.

**#699 wedge detector — the brief's premise is STALE for this lineage, and the
interaction is the opposite of the one feared.** The detector is **wired here**
(`2753c764ba`, in this branch's ancestry: `invariant_checker.py` +80,
`scheduler.py` +30, `batch_result_processor.py` +11, 266-line test). But it
cannot misclassify a bounded fast-lane wait, because it returns early:

```python
if r > 0:
    return False, f"{r} request(s) running: the box is serving, not wedged"
```
(`invariant_checker.py:568`)

The #536 specimen has a co-tenant prefill running, so `r > 0` and the verdict is
always "not wedged". The two classes are **disjoint**: #699 is *queued > 0,
running == 0, no first token for >= 20 s*; #536 is starvation **while serving**.

That is reassuring for the fix and is also a **coverage gap worth naming**:
nothing in the tree alarms on #536-class starvation today. The 20 s threshold
was chosen from the wedge band (11.87–62.65 s, `invariant_checker.py:525-530`)
and the 34.5 s specimen sits inside that band while being invisible to it.

## 6. Window item for F4-r4

Unchanged from `WINDOW_TICKET_536.md` and still the gate on arming anything:
**a lane-ON/OFF observation under real session load** — Session-Load, never
manual fill batteries — measuring `mt_first_token` for a fast-lane request
against a co-tenant prefill, with the reserve off and on. Without it the reserve
stays dark, and this determination changes nothing about that.
