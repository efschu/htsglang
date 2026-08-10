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

**0b. I reported a log number as a finding without checking whether it was
an echo of my own configuration.** I read `max_total_num_tokens 687090` out
of the boot log, saw it was above the 669440 target, and announced that
HANDOFF_662's >600k closure was contradicted. 687090 is
`--max-total-tokens 260000` floored to a rank-unit boundary and multiplied
back up by the split factor; the same log sentence prints
`local capacity 260000`. The refutation was inside the evidence I quoted,
and I did not read to the end of the line. The conclusion happened to
survive — a genuine pre-cap profile one line earlier says ~953786 — but I
reached it by luck, not by method. **A number computed downstream of a flag
you set yourself can never tell you what the rig would do without that
flag.** See §5.

**0c. I had no liveness check on my own agent traffic, and agent traffic is
a GREEN-RUN REQUIREMENT.** I rebooted at 05:01Z under two running agents and
killed both (`502 router could not reach the local endpoint`) — the exact
error I had quoted from successor 19 an hour earlier and then committed
anyway. One announced its death; the other simply stopped. Worse, my check
for "is agent traffic flowing" was reading the request COUNT, which cannot
distinguish "running" from "died five minutes ago" — two samples taken close
together look identical either way. **A count is not a liveness signal; only
its DERIVATIVE is.** The green criterion requires real agent traffic, so an
unnoticed agent death silently converts a green run into a soak-only run
that still looks green. Fixed by monitoring the count's rate of change and
alarming when it is flat for two consecutive intervals. **A successor should
assume that anything requiring an external producer needs a stall alarm, not
a presence check.**

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

Judged evidence at 06:00Z, 55 minutes into the T=190000 run:

| axis | value |
|---|---|
| both layouts visited | **357 pp_to_tp and 357 tp_to_pp** re-dispatches, exactly balanced |
| flips (cutovers complete) | 711 |
| prefill only in PP | **11344 prefill batches, 0 with a CUDA graph** — the PP stack is eager by construction, so a graphed prefill would mean TP |
| decode only in TP | **768 decode batches, 768 carrying `accept len`** — the PP phase has no drafter, so `accept len` can only be TP |
| purity gate active | 119 refusals of `purity: prefill cannot run in tp` |
| graph coverage (decode) | 762 / 768 = **99.2 %** |
| accept length | 2.15 |
| resident-set corruption | **0** |
| repairs performed | **0** |
| tracebacks / crashes | **0** |
| health | 200 throughout |
| agent requests via router 30099 | 197 and rising |
| pool | 190000 |
| corridor time-series MIN | 1369 / 1646 / **1037** over 24205 samples (floor 1024) |

**The leak axis is clean across 711 flips under sustained queueing
pressure.** That is the blocker, and it is closed.

**The corridor is the open axis.** 1037 MiB leaves 13 MiB of margin, and it
was reached while the over-driven soak was draining; since the load became
agents-only it has been stable at ~1165. The number is therefore a floor on
a load that was heavier than the acceptance point, not a verdict on the
acceptance point itself.


## 4. PURITY, measured from the log's own layout discriminators

The log does not tag a batch with its layout, but two fields decide it
unambiguously:

* the PP stack is **eager by construction** (HANDOFF_661 §7), so
  `cuda graph: True` on a batch means the TP layout;
* the PP phase **has no drafter** (the cutover clears TP spec_info entering
  PP), so `accept len:` on a batch means the TP layout.

Which gives a direct purity read without new instrumentation.

## 5. >600k: 662's closure does not hold — but my first evidence for that was garbage

HANDOFF_662 §4 closed >600k on the premise that the pool is sized once at
boot and every runtime grow path is closed, therefore no spill can add
capacity. The claim about grow paths is true. **The conclusion does not
follow, because it never asked what the boot-time size could have been.**

### First, the number I got wrong, because the shape of the mistake matters

I read this line out of the boot log and reported it as a profiled capacity
above the 669440 target:

    max_total_num_tokens 687090 (vector [28, 26, 20], hybrid mamba cap 2359344)

**It is not a profile. It is my own `--max-total-tokens 260000` rounded and
re-expanded.** `_apply_token_constraints`
(`model_runner_kv_cache_mixin.py:4422-4428`) applies the user cap FIRST,
then divides by each rank's ratio and multiplies back by the split factor:
`260000//28 = 9285`, `//26 = 10000`, `//20 = 13000`; `min = 9285`;
`9285 x 74 = 687090`, and line 4541 clamps it straight back to 260000. The
same log sentence prints `local capacity 260000` — the refutation was
inside the evidence I quoted.

**The lesson is not "check your arithmetic".** It is that a number computed
downstream of a flag I set myself can never be evidence about what the rig
could do without that flag, and 687090 was numerically close enough to the
target to feel like a discovery. **Before a log number becomes a finding,
establish that it is not an echo of your own configuration.**

### The real ceiling, and it is higher than the number I wrongly cited

One line earlier is a genuine PRE-cap profile, taken from a raw configurator
call that never sees the user limit:

    Uneven DCP: ... raise max_total_num_tokens from 953786 to ~1198144
    (per-rank profiled capacity [609580, 335126, 299542];
     active vector [28, 26, 20] leaves ranks idle)

`min(609580//28, 335126//26, 299542//20) x 74 = 953786`, verified. So at the
shipping budget the pool arithmetic would hand out **~953,786 tokens**, and
662's premise is wrong with a wider margin than I claimed. Caveat, stated
because it is load-bearing: this is a pre-graph-capture profile and
post-capture resizing is inactive in this boot, so it is *what the sizer
would have handed out*, not a number anyone has booted.

### So the binding constraint is the corridor, and it is quantified

The per-token cost is confirmed exactly against this boot:
`260000 x 32 layers x 256 B = 1.983 GiB` matches PP0's reported K size, and
the 2:1:1 split confirms the layer shares. 662 §5's expression holds:

    per-rank KV bytes = T x 32 KiB x max(layer_share_r, token_share_r)

with shares (0.500, 0.351, 0.270), sum 1.1216. Cost of going from 260000 to
669440, against this window's measured corridor minima:

| rank | share | delta KV | measured min @260k | projected min @669440 |
|---|---|---|---|---|
| 0 (5090) | 0.500 | +6397 MiB | 2167 | **-4231** |
| 1 | 0.351 | +4496 MiB | 2612 | **-1884** |
| 2 | 0.270 | +3458 MiB | 1809 | **-1649** |

**669440 is short by 5-7 GiB per card.** Anchored on the measured minima,
the largest corridor-legal pool is **T ~ 333,000, rank-0-bound** — which
also says 260000 is *not* the edge, and is the cheapest capacity work
available to a successor.

### The spec's own spill mechanic cannot close it, and the reason is sharper than "no grow path"

The arena is `max(pp_bytes, tp_bytes)` **fixed for process life**
(`phase_flip_boot.py:544`), and it cannot be shrunk even in principle
because **the TP decode CUDA graphs bake its absolute addresses**
(`phase_flip_boot.py:569`). But suppose it could be. It still gains nothing,
for a reason that applies to both quantities at once:

* the **pool ceiling** is a worst-PHASE quantity, and each rank's idle tail
  exists only in that rank's non-binding phase, so `min over phases` of the
  bytes freed is **0** on every rank;
* the **corridor minimum** is a worst-TIME quantity, and the tail is idle in
  precisely the phase that does not set the minimum, so `min over time` is
  **0** as well.

**I had conceded that the tail was at least real for the FREE column. That
concession was wrong** — it is real instantaneously and worthless to a
continuous-minimum law. To make the tail spendable you would have to change
what the `max()` is taken over, not when it is released: equalize the two
layouts' per-rank weight footprints so `pp_bytes_r ~ tp_bytes_r`. Even then
it is ~1.2-1.8 GiB per card against a ~6.4 GiB shortfall on rank 0.

And `phase_flip_spill.py` is not a partial implementation of spec item 6 at
all: `get_spill_ladder` has no caller, `phase_flip_spill_depth` is not a
`ServerArgs` field, and the ladder is built over the **draft** model's
parameters — which `phase_flip_boot.py:562` says are not arena-backed and
stay resident across both phases. It targets a different asset entirely.

**Honest closure: >600k is unreachable on this rig at this budget, and the
blocker is the corridor, not the pool sizer.** The gap is ~6.4 GiB on the
5090 alone; the spec's spill mechanic is worth 0 against it. Closing it
needs a fourth card or a smaller per-token cell, and that is a decision for
the user, not a defect for a successor to fix.

## 5b. THE 807 MiB ODDITY IS RESOLVED: it is real and it does not bind

662 flagged, and the bench repeated, that `RANK_MIB=31800` on a 32607 MiB
card leaves 807 MiB — under the 1024 floor **by construction** — and warned
that every passing config held only because the engine did not consume its
whole budget. It has been carried as unresolved by every document since.

Measured at 30 minutes of the T=190000 run:

| card | total | RANK_MIB | used | NVML free | overshoot |
|---|---|---|---|---|---|
| gpu0 3080 (rank1) | 20480 | 17400 | 18267 | 1789 | **+867** |
| gpu1 5090 (rank0) | 32607 | 31800 | 29279 | 2810 | -2521 |
| gpu2 3080 (rank2) | 20480 | 17450 | 18739 | **1317** | **+1289** |

**The oddity is real and it never binds.** Rank 0 sits 2521 MiB UNDER its
budget, so the 5090 is the most comfortable card in the rig. What binds is
the opposite failure on the other two: **the 3080s exceed their budgets by
867 and 1289 MiB.** `RANK_MIB` is advisory in both directions, and only 662's
half of that was known.

Sharpest form: **rank2 is the tightest card while holding the SMALLEST token
share (20 of 74)**, so its pressure is weights plus per-rank overhead, not
KV. Tuning the token vector or the pool attacks the wrong term for the only
card that actually decides the corridor.

Two consequences a successor should not have to rediscover:

* **The carve-out is exact.** `total-used` minus NVML `free` is **424** on
  both 3080s. Sizing against `total-used` reads 1741 MiB of headroom where
  1317 exists — one whole safety margin of error.
* **The untried cell is LOW budget x LOW pool, on the 3080s.** 662 lowered
  `RANK_MIB` against a 460000 pool, got a worse result, and concluded
  "change the pool, not the budget". That conclusion is sound for the cell it
  ran and is not evidence about this one, which nobody has run.

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

## 7. HOUSEKEEPING A SUCCESSOR INHERITS

**A retired claim is still live in the text.** 659 §1 costed the cutover at
"1.4-3.0 GiB per card" and concluded ">600k and auto-flip are in direct
tension". 660 §1 WITHDREW it — the seam peak was a PHASE HOLD, not a cutover
cost. But 658 §4e still cites "a 1.4-3.0 GiB seam" to condemn the fairness
windows, and 661 §3 then had to reverse that separately ("a policy knob was
blamed for a sizing defect"). **The number outlived two of its own
retractions.** When you retire a figure, grep the corpus for it.

**The PD loops carry defect R's exact shape.** Confirmed by audit:
the prefill-disaggregation loop reads `self.last_batch = self.last_mbs[mb_id]`
(~:348) with the conditional assignment at ~:434-440, and the
decode-disaggregation loop reads at ~:494 with the conditional at ~:616-623.
**Whether either is HARMFUL is NOT established** — that needs (a) their
batch-selection returning None while requests stay resident, and (b) a
mutating consume of `last_batch`. Both were left unverified when my audit
died. Do not read my silence as a clean bill.

**Corpse-table entries H, J.2 and I remain open**, as does the unnamed J.1
AUDIT CANDIDATE (docstring ~:525-532) — the false assumption that
`scheduler.running_batch` names the rank's resident set under
`event_loop_pp`. Defect R is arguably its third instance, since the whole
bug is a slot-scoped handle being treated as durable. **That audit pass is
now overdue by four handoffs.**

## 8. THE VMM SPILL ROUTE (user directive, 06:00Z) — the premise is already satisfied

The user overrode the >600k closure and ordered a VA-stable physical-release
route: spill the INACTIVE layout's weight shards to host RAM each phase,
restore at the flip, keep the virtual address stable so captured decode
graphs stay valid, and re-size the pool against the freed bytes. Estimated
in the order as ~9 GiB per card.

**My objection (a) is withdrawn.** I had argued the arena cannot shrink
because the TP decode graphs bake its addresses. A VA-stable release
dissolves that, and this tree PROVES it dissolves it, because the mechanism
is already here and already load-bearing for a different asset:

* `kv_vmm_backing.py:208/350/358` — `cuMemAddressReserve`, `cuMemMap` /
  `cuMemSetAccess`, `cuMemUnmap` / `cuMemRelease`.
* `phase_flip_runtime.py:1434-1491` — the per-flip KV backing swap, whose
  own comment is the refutation of my objection: *"The VA reservations are
  untouched, so every address the TP stack's decode graphs baked in stays
  valid across any number of flips."*

So VMM works on this driver and VA-stable release is proven in production
here. The route is technically open.

**It has nothing left to reclaim, and that is a code fact, not an argument.**
The inactive layout does not occupy device memory at all:

* the arena is ONE per rank, sized `max(layout_pp.total_bytes,
  layout_tp.total_bytes)` — `phase_flip_boot.py:544-547`,
  `weights_arena.py:233-235`;
* `snapshot_and_free` (`phase_flip_boot.py:247-270`) frees every device
  original and rebinds it to a 0-element placeholder, leaving the layout
  only as a **pinned host image**. Its docstring gives the reason in the
  rig's own numbers: three copies resident would be
  *"14.7 + 12.4 + 14.7 GB > 31.8"* and would not fit the 5090;
* a flip is therefore a host->device `copy_` into the fixed arena VA
  (`weights_arena.py:378-381`), checksum-verified on device.

**The ordered spill is already implemented and already fully banked.** What
remains is only the arena's idle tail, `arena - current_phase_bytes`:

| rank | releasable in PP phase | releasable in TP phase | **worst case over time** |
|---|---|---|---|
| 0 (5090) | 0 | 1773 MiB | **0** |
| 1 (3080) | 1234 MiB | 0 | **0** |
| 2 (3080) | 0 | 1191 MiB | **0** |

The tail exists on each rank only in that rank's NON-binding phase. A
corridor floor is a continuous law governed by the worst instant, so the
worst-case-over-time credit is **0 MiB on every rank**. Building a VMM path
under the weights arena — which would mean moving it off the caching
allocator (`weights_arena.py:235` is a bare `torch.empty`; the call site is
in no `MemPool` context) — buys zero corridor.

**What this does NOT say.** It does not say >600k is impossible; it says
this particular funding source is empty because it was already spent at
boot. The real gap and the real lever are unchanged and are in §5b: the two
3080s exceed `RANK_MIB` by 867 and 1289 MiB while the 5090 sits 2521 MiB
UNDER its own, so the binding term is per-rank budget overshoot, not
inactive-layout residency. **The untried experiment is low `RANK_MIB` x low
pool on the 3080s.**

**One hook worth recording for whoever funds the pool differently.**
`model_runner_kv_cache_mixin.py:755-783` already adds a signed byte
correction (`correction_gb`) to `rest_memory` immediately before
`_profile_available_bytes` returns — the exact place a "N bytes will be
freed later" credit belongs, additive, and off by default so the default
path stays byte-identical. If a future asset genuinely frees worst-case
bytes, that is where the credit goes; no re-architecting required.
