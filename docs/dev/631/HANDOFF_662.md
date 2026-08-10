# #656 HANDOFF v22 — successor 19

Written 2026-08-10, tree `/spinning/wt-631-routea`, branch
`feat/route-a-631`.

Read this before HANDOFF_661. 661's ship number (460000) is retired here,
and the agent-traffic blocker it left open is closed — the router was never
the defect.

---

## 1. MY ERRORS, ranked — read these before my results

**0. I ran a confounded A/B and then acted on it.** I measured the corridor
at pool 460000 under one set of live agent agents, then at 380000 under a
*different* set, saw the second reading was lower, and concluded "the pool
does not control the corridor — `RANK_MIB` does". The two windows had
different loads, so the comparison established nothing. Acting on it cost
two boots and ~25 minutes: I lowered `RANK_MIB` to 30500/16550/16500 and
made the rig **worse** (852 MiB free on the 5090 *at idle*, already under
the floor). The same-load-floor law exists precisely for this, and I know
it — I skipped it because the second reading felt like a trend. **Two
measurements under two different loads are not an A/B, they are two
anecdotes.** The correct move was to hold the load fixed and change one
variable, which is what finally worked (340000 on the proven `RANK_MIB`).

**1. I dispatched agent traffic and then rebooted the server underneath
it.** Three qwen agents died mid-task with `502 router could not reach the
local endpoint`, losing their partial work. Agent traffic is a *dependency*
on a stable instance, not a background decoration. Dispatch it only against
an instance you are not about to restart, and if you must reboot, expect to
re-issue every agent.

**2. I let a subagent run Python against the live tree during a corridor
measurement.** The Explore agent executed `derive_pp_layer_split` to verify
a layer split; importing that module can initialise CUDA and place a
~300–500 MiB context on each card, which is exactly the magnitude of the
drop I was trying to explain. I did check (`nvidia-smi` showed only the
three scheduler PIDs, so it was clean) — but I checked *after* the fact.
Every read-only briefing from #3 onward carries "do NOT run python that
imports torch or touches a GPU". It should have been in briefing #1. A
measurement rig has to be defended in the briefing, not audited afterwards.

---

## 2. THE AGENT-TRAFFIC LANE: closed, and the router was innocent

This was the gate blocking green since HANDOFF_661 §9. Full evidence in
`PROD_BRINGUP_BENCH.md` §"SUCCESSOR 19 / 1".

A direct probe through the router works and lands on this instance:

    router  POST /v1/messages -> local (model=Qwen3.6-27B)
    serving [01:08:13] "POST /v1/messages HTTP/1.1" 200 OK

**The defect is the dispatch, not the plumbing.** The `qwen` /
`local-model` / `local-model-think` agent definitions carry
`model: Qwen3.6-27B`, and the router routes on an exact match of that
string. **Passing an explicit `model:` argument to the Agent tool overrides
the frontmatter**, so the request leaves as `claude-*` and goes upstream to
Anthropic. Control pair, same task, one minute apart:

| dispatch | router verdict |
|---|---|
| `subagent_type: qwen`, no model argument | `-> local (model=Qwen3.6-27B)` |
| `subagent_type: qwen`, `model: haiku` | `-> upstream (model=claude-haiku-…)` |

**The trap is that a standing memory law causes it.** `agent-modellwahl`
says "model IMMER explizit". Applied to a local-model agent that rule
silently sends the work to Anthropic and produces exactly successor 18's
symptom: agents that run correctly while the serving log stays empty.
**For qwen/local-model agents, pass no `model` argument.**

## 3. 460000 IS RETIRED. The best-supported pool is 260000

Successor 18 established 460000 on a synthetic soak with **no agent
traffic**, and the user's green criterion *requires* agent traffic. Adding
it breaks the corridor on all three cards — and the corridor minimum keeps
falling as the agent CONTEXT grows, because the allocator's high-water mark
follows the largest request shape it has seen:

| pool | load | gpu0 | gpu1 (5090) | gpu2 | verdict |
|---|---|---|---|---|---|
| 460000 | soak only (s18) | 1397 | 1354 | 1451 | 0 breaches |
| 460000 | soak + agents | 591 | 270 | 621 | BREACH x3 |
| 340000 | soak + light agents | 1191 | 1166 | 1089 | 0 breaches, margin 65 MiB |
| 340000 | soak + heavy agents | 1029 | **886** | **869** | BREACH gpu1, gpu2 |
| 260000 | soak + heavy agents | 2167 | 2612 | 1809 | 0 breaches / 3150 samples |

Floor is 1024 MiB. **A capacity number is only real under the load the
acceptance criterion names**, and this is the second time in this chain that
a rung was climbed under a load lighter than the bar (s18 caught the same
error going from bs=1 to bs=4; this is the bs=4→agent-traffic instance).

**A plateau is only a plateau for the workload mix in flight.** At 340000 the
corridor read *identically* 1191/1166/1089 for two consecutive 2-minute
buckets — a textbook steady state — then breached within two minutes of two
large-context agents joining. Provoking the worst shapes at minute 10 cost
ten minutes; discovering them at minute 40 would have cost the window.

**Change the pool, not the budget.** Lowering `RANK_MIB` to
30500/16550/16500 made the 5090 *worse* (852 MiB free at **idle**) because
the KV pool is a hard requirement while `RANK_MIB` is advisory — a too-small
budget against a too-large pool simply overshoots.

**A structural oddity for a successor:** `RANK_MIB=31800` on a 32607 MiB card
leaves 807 MiB, **below the 1024 floor by construction**. Every config that
held did so only because the engine did not consume its whole budget. The
budget and the corridor law have never been reconciled.

Two properties worth carrying:

* **Free memory swings by GiB across a flip** (the KV backing
  release/restore leg), so a corridor minimum is meaningless without the
  phase it was taken in.
* **A flat corridor reading can mean an idle server.** The first 90 s of the
  460000 window read a rock-steady 1393/1352/1447 — that was the soak
  spinning up, not the corridor passing. Flatness is the tell.

260000 still leaves 785–1588 MiB above the floor, so it is not the edge —
but no pool has yet carried a full 60-minute window, so the edge stays
unmeasured until the wedge in §5b is fixed.

## 4. FULL KV >600k: the honest closure the spec asks for

The user's spec item 6 orders: at the phase change, spill everything cold of
the inactive layout — **including the PP weight shards** — to system RAM.
Successor 18 closed >600k costing only the draft assets, so this was the
right thing to re-open.

**The ordered mechanic is already implemented and already banked.**
`weights_arena.py:2-8`, `phase_flip_boot.py:477-547`: both layouts'
parameters live one at a time in ONE arena per rank sized
`max(pp_bytes, tp_bytes)`, and a flip rewrites its contents from the other
layout's **pinned host image**. The inactive layout's weights are already in
system RAM — at a cost of 59.75 GiB of pinned host memory already being
paid. Measured arena sizes:

| rank | PP layout | TP layout | arena = max | idle tail | tail freeable in |
|---|---|---|---|---|---|
| 0 (5090) | 14936 MiB | 13163 MiB | 14936 MiB | 1773 MiB | TP phase (binding) |
| 1 (3080) | 6690 MiB | 7924 MiB | 7924 MiB | 1234 MiB | PP phase (wrong side) |
| 2 (3080) | 9115 MiB | 7924 MiB | 9115 MiB | 1191 MiB | TP phase (binding) |

So the residue the spec's mechanic could still reclaim is the arena's **idle
tail**, not the shards — and unlike the draft-asset rung, two of three ranks
free it in the phase that binds.

**It still cannot fund >600k, for a structural reason rather than an
arithmetic one: the KV pool is sized ONCE at boot from a single
`mem_get_info` reading, and every runtime grow path is closed.**
`_profile_available_bytes` (`model_runner_kv_cache_mixin.py:608-700`) never
re-measures; `--max-total-tokens` is a `min()` cap that can only lower
(`:4406-4427`); `post_capture_resize_kv_pool` is off by default, shrink-only,
and excluded for the TP stack; `runtime_set_backing_tokens` is bounded by the
boot VA reservation; and `phase_flip_boot.py:636-650` refuses boot unless
`tp_capacity >= pp_capacity` because one allocator serves both layouts for
process life.

**Therefore no runtime spill of any asset can add KV capacity — freed bytes
become idle card slack.** This is the third handoff to arrive here.
**The framing itself is the defect: "can we spill at runtime to grow the
pool" is unanswerable, and the answerable question is "what pool can the rig
BOOT with and still hold the corridor".** Under the load the acceptance
criterion names, that answer is 260000 — so >600k is further away than
successor 18's negative suggested, not closer.

`phase_flip_spill.py` is written but **not wired**: `get_spill_ladder` has no
caller outside its own module and `phase_flip_spill_depth` is not a
`ServerArgs` field. Its docstring describes `_cutover` step-7b call sites
that do not exist.

## 5. THE ONE UNTRIED LEVER: a 12.1 % capacity tax from a vector mismatch

Per-rank KV bytes obey one expression, validated exactly against **three**
pool sizes (540000, 460000, 380000, 340000 — all four predicted to the
reported decimal):

    per-rank KV bytes = T x 32 KiB x max(layer_share_i, token_share_i)

The `max` is there because one arena backs both layouts. The ship config runs
`pp_layer_ratio [32,16,16]` (0.500/0.250/0.250) against token vector
`28,26,20` (0.378/0.351/0.270), so the rig pays

    max = (0.500, 0.351, 0.270),  sum = 1.121

— **112.1 % of one KV cache to store one KV cache.** Cross-check: 460000 x
32 KiB x 1.121 = 15.74 GiB, and the three measured arenas sum to
7.02 + 4.93 + 3.79 = 15.74 GiB exactly.

Aligning the two vectors drives the sum toward 1.000. **This is a boot-flag
change, not a feature** — the cheapest capacity lever in the whole chain and
the only one still untried.

**Two warnings before someone spends a shift on it.**

1. **It is in direct tension with PP compute balance.** `pp_layer_ratio` sets
   both the PP stage's compute share and its KV share. HANDOFF_661 §8 records
   the 5090 drawing ~250 of 400 W in PP prefill, which argues for giving it
   *more* layers; capacity argues for *fewer*, because its layer share (0.500)
   exceeds its token share (0.378) and is therefore the binding term. **The
   5090 cannot be both the compute-heavy PP stage and the capacity-efficient
   one.**
2. **The layer split also moves the weight arena, and that partly cancels the
   gain.** Moving rank0 32→24 layers frees ~1.77 GiB of arena there but adds
   ~1.31 GiB on rank1, and the 3080s are the cards with no slack. I costed
   `[24,23,17]` and it *breaks* both 3080s while giving the 5090 3.5 GiB it
   does not need. **Any candidate split must be costed on weights AND KV
   together, per card** — the KV-only arithmetic looks like a free win and
   is not.

## 5b. THE WEDGE: a self-reinforcing deadlock, and it is the real blocker

**This is the most important thing in this handoff.** The green run died at
02:05Z, and the mechanism is a defect that the strand's sibling task
(#622/#649, "why serving keeps wedging") has been hunting.

Log evidence, `/spinning/evidence-631/WEDGE_20260810T0201Z_residentcarry.log`
(also `/spinning/serving-30030.boot.log.s19-260k-WEDGE-0205`):

    02:01:10 PP0  Prefill batch ... #running-req: 1, #queue-req: 6,
                  #pending-token: 177339
    02:01:10 PP0  PHASE-POLICY refusing to evaluate the flip policy this
                  round: the resident set is corrupted (PHASE-FLIP-CARRY
                  running_mbs[0] claims 5 resident request(s), above
                  max_running_requests=4 ... This is defect M

The claimed count then walks **5 → 13338 → 26671 → 40005 → 53338 → …→
168854**, in exactly uniform increments: **+1 per scheduler round**, ~700/s.

### It is not successor 18's bug

s18's self-merge doubled the batch (powers of two). This grows **linearly**,
and the s18 guard is present and firing in this very build (`SELF-MERGE
REFUSED` 132708 times in the same window). **The doubling was fixed; a
linear leak remains underneath it.**

### It is a miscount, and then a deadlock

* **Miscount:** on the same round the scheduler reports `#running-req: 1`
  while `running_mbs[0]` claims 5. 168854 concurrent requests is impossible
  with `max_running_requests=4`. And the growth rate (~700/s) is orders of
  magnitude above the arrival rate (6 requests in 7 minutes), so
  **`running_mbs[0]` is accumulating a duplicate of the same request once
  per scheduler round.**
* **Deadlock:** the guard (`phase_flip_resident_carry.py:236-245`) refuses to
  evaluate the flip policy while the set looks corrupt. But under **strict
  purity decode is forbidden in the PP layout**, so *only a flip to TP can
  drain the resident set*. **The guard blocks the one action that would
  clear the condition it is detecting**, and the instance can never recover.
  1115 flips happened before the wedge; zero after.

### Why it fires now and not for successor 18

The trigger is **queueing pressure** — `#queue-req: 6` against
`max_running_requests=4`. Successor 18's synthetic soak never exceeded the
design point. Real agent traffic does, easily, because agent turns arrive in
bursts. **So closing the agent-traffic gate is what exposed this**: the
blocker has moved from "traffic does not arrive" to "traffic wedges the
scheduler", and the green criterion cannot be met until this is fixed.

### For the successor, the shape of the fix

Two separable defects, and they should not be conflated:

1. **The leak** — find what appends to `running_mbs[0]` once per round
   without removing the finished/duplicate entry. Start at
   `harvest_resident_batches` (`phase_flip_resident_carry.py:209-260`) and
   the `running_mbs`/`last_mbs` stores in `scheduler_pp_mixin.py:198/250`,
   the same pair that hosted the s18 aliasing bug.
2. **The deadlock** — a detector that only refuses is not containment; this
   is the *second* time that exact lesson has been paid for in this chain
   (HANDOFF_661 §2 records the first). Under strict purity the refusal must
   not be able to outlive the condition: either repair the set, or permit the
   flip that drains it, but never spin forever.

**Do not "fix" this by raising `max_running_requests`.** The count reaches
168854; any ceiling is crossed within seconds. And do not relax strict
purity — that hides the deadlock without touching the leak, and purity is a
hard user requirement (spec item 10).

### The corpse table already predicted this leak — read entry K first

A survey of the 800-line corpse table in `phase_flip_presence.py` (lines
14-812) puts this defect squarely in its most populated theme. Three entries
bear directly on it:

* **K — THE RESIDENT CARRY.** Its stated design law is that *"the hazard is
  DUPLICATION, not loss: `merge_batch` extends in place, so dedupe by batch
  identity and refuse loudly if one Req is reachable through two distinct
  batches."* That is precisely the failure now observed — one request
  entering `running_mbs[0]` once per round. **Start here.**

  **And K names the mechanism outright:** *"`merge_batch` extends in place,
  so it is not idempotent; a second merge enters the same Req twice."* A
  non-idempotent merge re-executed once per scheduler round produces exactly
  the observed law — **+1 per round, ~700/s, unbounded**. The question for
  the successor is therefore not "what is appending?" but **"what is
  re-merging the same batch every round, and why did the identity dedupe not
  catch it?"** Note that s18's identity guard catches `X.merge_batch(X)` —
  the *same object* — so a re-merge of a *distinct object holding the same
  Req* passes straight through it. That is the gap between the two bugs, and
  it is why fixing the doubling did not fix this.

  This reading is not one agent's opinion: **two independently briefed
  surveys of the 800-line docstring converged on it**, and both flag the same
  entries as still open.
* **J.1 — SLOT SCOPE**, and its unnamed **AUDIT CANDIDATE** successor
  (docstring lines 525-532), which is marked **OPEN**: the false assumption
  that `scheduler.running_batch` names the rank's resident set is available
  to any code running under `event_loop_pp`, two instances were already
  confirmed and fixed (J.1 and the GDN mover), and the docstring says the
  general case is *"worth an audit pass of its own; not part of #631"*.
  **That audit pass is now overdue and this wedge is plausibly its third
  instance.**
* **L** — the matching lesson that a non-empty `last_batch` means "requests
  are resident", not "work is in flight".

**Naming caution:** the runtime error text calls this "defect M"
(`phase_flip_resident_carry.py:242`), but the docstring's own entry **M** is
a different defect entirely (the PP chain ring read off the live `ps`). The
label in the log does not index the corpse table — do not follow it to the
wrong entry.

Other entries the survey found still **OPEN**, for a successor's awareness:
**H** (a pre-entry abandonment leaves a live flag; a withdrawal is not
publishable), **J.2** (the row extent over-counts by one; deliberately not
yet cut), and **I** (quiescence is rank-local while the obligation is
pairwise).

## 6. State at handover

* HEAD as committed this shift; suite status recorded in the commit message.
* Serving UP on 30030, boot `06ed5afc25`, `RANK_MIB=31800,17400,17450`,
  `MAX_TOTAL_TOKENS=260000`, CTX 393216, purity strict, both fairness
  windows on, POLICY=auto.
* Boot replay: `/tmp/s18_boot.sh` (pass `RANK_MIB` and `MAX_TOTAL_TOKENS`).
  Note its defaults differ from `route_a_631_prod_boot.sh`, whose own
  defaults are NOT the proven config.
* **Boot provenance caveat:** the banner records the last *commit* plus
  `dirty_files=N`. Successor 18's run showed `commit=1c7d3c3e0b` while
  actually running the uncommitted self-merge fix — the banner alone cannot
  name the patch level, `dirty_files` is load-bearing, and the way to
  confirm a fix is live is to grep the log for its own log line
  (`SELF-MERGE REFUSED` fired 494106 times, which is how I confirmed it).

## 7. Next steps, in order

1. **Fix the wedge (§5b). Nothing else matters until this is done** — the
   green criterion requires agent traffic, and agent traffic wedges the
   scheduler within ~4 minutes. This is the whole blocker now.
2. **Then re-run the green run.** No green run of mine completed: three were
   aborted by me for corridor breaches (460000, 380000, 340000) and the
   fourth, at T=260000, was killed by the wedge at 02:05Z after 8 minutes.
   T=260000 held the corridor cleanly while it lived — min 2167/2612/1809,
   **0 breaches in 3150 samples** with heavy agent traffic — so 260000 is
   the best-supported pool, but it has NOT carried a 60-minute window.
3. **Re-establish the capacity edge under agent traffic** once the instance
   can survive one. 260000 left 785–1588 MiB of margin, so the edge is
   higher; it was not chased because the wedge ended the window.
3. **The alignment lever (§5)**, costed on weights *and* KV per card.
4. **Graph A/Bs (spec item 8), still unmeasured** — I did not reach them.
   NEXTN draft graphs (`SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH`), DFLASH x
   graphs. PP-prefill graphs are ANSWERED as impossible without a design
   change (HANDOFF_661 §7: the PP stack is eager by construction).
5. **Put the full-KV physics to the user** (§4). The requirement is written
   against 669440, which our own bench calls a TP-only modelled ceiling.
