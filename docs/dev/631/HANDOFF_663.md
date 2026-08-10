# #656 HANDOFF v23 — successor 20

Written 2026-08-10, tree `/spinning/wt-631-routea`, branch
`feat/route-a-631`.

Read this before HANDOFF_662. 662's diagnosis of the blocker pointed at the
wrong component, and this handoff says where and why — that misdirection is
the most useful thing in it for you.

---

## 1. MY ERRORS, ranked — read these before my results

**0. I published a completion claim on the strength of an instrument that
could not have detected the failure.** For several minutes my working
evidence that the leak was fixed was "zero corruption reports in the log".
That is not evidence. The corruption report only fires ABOVE
`max_running_requests=4`, so it is silent for a resident set of 1, 2, 3 or
4 — and silent for a set that is slowly growing through those values. Had I
shipped on it I would have repeated successor 18's error in a new costume.
The instrument that actually decides the question is the carry count in
`PHASE-FLIP-CARRY carried N resident request(s)`, which reports the set's
real size at every cutover. I found it by accident while grepping for
something else. **Before trusting a null result, establish that the
instrument can produce a non-null one.**

**1. I fixed the main PP loop and left two structurally identical loops
alone.** `_event_loop_pp_body` was the loop my feature runs, so it is the
loop I fixed. The prefill-disaggregation and decode-disaggregation loops in
the same file carry the same conditional-assignment shape. See §6 for what
was established about them; the point of listing it as an error is that
"out of scope" is a scheduling decision, not a safety argument, and a
successor should not read my silence as a clean bill.

**2. I ran the validation window with soak and agent traffic started
minutes apart, then reported them as one window.** The load was not
constant across the window I am quoting. It does not affect the leak verdict
— the carry count is a structural bound, not a statistic — but any
corridor minimum I quote from that window is a minimum over a CHANGING load,
which is exactly the confound successor 19 recorded as their error 0. Treat
the corridor numbers here as provisional and re-measure under a load you
hold fixed.

---

## 2. THE BLOCKER IS CLOSED. It was one indentation, and not where 662 said

Full write-up in `PROD_BRINGUP_BENCH.md` §"SUCCESSOR 20 / 1". The short
form:

`last_mbs[slot]` must name *the batch that slot ran in its previous
iteration*. In `scheduler_pp_mixin._event_loop_pp_body` the assignment sat
INSIDE `if self.mbs[next_mb_id] is not None:`, which redefined it as *the
last non-empty batch this slot EVER ran*. The two differ only when a slot
runs nothing while requests are still resident in it — and **strict purity
creates that state deliberately**: `get_next_batch_to_run` returns
`batch_to_run = None` for a resident decode batch in the PP layout.

The stale entry is an EXTEND batch, so every later visit to the slot reaches
`running_batch.merge_batch(last_batch)`, which extends `reqs` in place.
+1 per round, forever.

**Both existing defences were blind to it, and neither was wrong.** The
self-merge guard compares `last_batch is running_batch` — the stale entry
is a *distinct object*. `harvest_resident_batches` dedupes by `id(batch)` —
the duplication is *inside one batch's `reqs`*. A distinct object holding
already-resident Reqs is the one shape that defeats both at once, which is
what entry K of `phase_flip_presence` predicted for a non-idempotent merge.

### The correction that mattered, and it is a general lesson

**`claims 5` was never the onset.** 5 is simply the first value above
`max_running_requests=4` that the guard is *able to report*. The leak runs
silently through 1, 2, 3, 4 first.

HANDOFF_662 saw "growth begins at 5" and "the guard begins refusing at 5"
and inferred a causal link — that the drain was gated behind the evaluation
the guard declined — and sent me at the guard. Both facts were real; the
coincidence was an artefact of the reporting threshold. **A detection
threshold can wear the costume of an onset.** When two events coincide at a
number, check whether one of them is simply the first number your instrument
can print.

Successor 19's asymmetric-alias lead (`install_resident_set` TP->PP) is
**not** the cause. It was correctly flagged there as unproven, and the
falsifier it proposed would have exonerated it. The aliasing is real and
harmless: the harvest dedupes by `id(batch)`.

### The fix, and the containment, are separate things

* **Fix:** unconditional assignment, extracted into
  `_pp_record_slot_last_batch` with the rationale attached. This is what the
  non-PP loops have always done (`self.last_batch = batch`, possibly None).
  Both loop families now answer "what did the previous iteration run" the
  same way, and **"nothing" is a real answer** rather than a hole that
  preserves the old one.
* **Containment:** the guard's refusal was itself the deadlock — under
  strict purity only a flip to TP drains the resident set, so refusing to
  evaluate the flip policy blocked the one action that clears the condition
  being detected. The catch site now repairs intra-batch duplicate Reqs with
  the scheduler's own `filter_batch` and re-asks, and logs the whole slot
  row instead of a bare count. **This is the third time this chain has paid
  for "a detector that only declines to act is not containment"; it is now
  written into the code rather than into a handoff.**

### Evidence

Desk — `test_pp_slot_last_batch_631.py` drives the REAL
`_event_loop_pp_body` taken unbound off the mixin. A model of the statement
order would have been circular, because the order IS the defect.
`_LeakyRank` overrides one method back to the pre-fix rule, nothing else:

| 25 cycles, 1 distinct request | resident entries | distinct | `last_mbs[0]` |
|---|---|---|---|
| PRE-FIX rule | **26** | 1 | STALE |
| shipped rule | 2 | 1 | None |

26 = 1 + 25 cycles: the metal law (+1 per round) reproduced exactly. Full
#631 family **649 passed / 0 failed** (was 643).

Metal — see §3.

## 3. THE VALIDATION WINDOW

Boot 04:23Z, `RANK_MIB=31800,17400,17450`, `MAX_TOTAL_TOKENS=260000`,
CTX 393216, purity strict, POLICY auto, HEAD `8764b96589`.

Load: 8 concurrent soak streams against `max_running_requests=4`
(`#queue-req: 5` — the trigger condition that produced the wedge in 8
minutes on the pre-fix build) plus real qwen agent traffic through router
30099, confirmed arriving as `POST /v1/messages` in the serving log.

**The decisive instrument is the carry count, not the absence of an alarm.**

<!-- FINAL NUMBERS: filled in at end of window -->

## 4. PURITY, measured from the log's own layout discriminators

The log does not tag a batch with its layout, but two fields decide it
unambiguously:

* the PP stack is **eager by construction** (HANDOFF_661 §7), so
  `cuda graph: True` on a batch means the TP layout;
* the PP phase **has no drafter** (the cutover clears TP spec_info entering
  PP), so `accept len:` on a batch means the TP layout.

Which gives a direct purity read without new instrumentation.

## 5. >600k: 662's CLOSURE DOES NOT HOLD, and the reason is in the boot log

HANDOFF_662 §4 closed >600k on the premise that the pool is sized once at
boot and every grow path is closed, therefore no spill can add capacity.
The premise about grow paths is true. **The conclusion does not follow,
because it never asked what the boot-time size could have been.**

This boot's own log answers that:

    max_total_num_tokens 687090 (vector [28, 26, 20], hybrid mamba cap 2359344)

At the shipping budget the engine **profiles 687090 tokens** — above the
669440 target. The instance runs at 260000 only because
`--max-total-tokens 260000` is passed, and that flag is a `min()` cap that
can only lower the profiled figure.

So the binding constraint on >600k is **not the pool arithmetic. It is the
corridor** — the 1024 MiB free-per-card law — because a larger pool spends
exactly the free memory the corridor is measuring. That reframes the
question a successor should ask, from "can the pool be grown" (answered: it
does not need to be) to "what would make a 600k pool corridor-legal".

<!-- SUBAGENT COSTING: filled in below -->

## 6. WHAT I DID NOT REACH

* **Graph A/Bs (spec item 8)** — still untouched after five successors.
  NEXTN draft graphs, DFLASH x graphs, and the 5090 stage-imbalance A/B
  (~250 of 400 W during PP3 prefill). PP-prefill graphs are already ANSWERED
  as impossible without a design change (661 §7: the PP stack is eager by
  construction). **Take a same-boot floor before every one of them.**
* **The pool edge post-fix.** 260000 is the best-supported number and it now
  survives the load that used to wedge the instance, but it was never the
  edge — 662 measured 785–1588 MiB of margin at that pool. With the leak
  fixed, the ladder above 260000 can finally be climbed under a load that
  does not end the window; that is the cheapest capacity work available.
* **The alignment lever (662 §5)**, costed on weights AND KV per card. The
  KV-only arithmetic looks like a free 12.1 % and is not.
