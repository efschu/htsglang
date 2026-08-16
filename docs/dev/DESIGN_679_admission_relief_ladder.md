# DESIGN 679 — The admission relief ladder, and what must be tried before a park

Status: design note, desk-only. Written for #677 (Q3) to consume directly.
Companion to the #679 fix landed in `c4b88e1923`.

## 0. Why this note exists

#679 landed two layers: a **park guard** at admission (a chunked prefill is
scheduled only for tokens the pool can actually fund, otherwise it parks and is
retried) and a **retry net** at `alloc_token_slots` (one bounded relief, one
retry, then the original error). The net's provider registry is **empty**, and
the close-out said why:

> the rank-local relief that could pay here is already spent by then, and the
> ones that could genuinely free tokens are collective and belong on the
> admission path.

This note makes that concrete. It answers: which collective reliefs exist
today, what each frees at the admission decision point, how each is invoked
without splitting the group, and in what order they should be tried **before**
a park.

A park is not free. It is a request that made no progress this round. The
ladder below is what should be spent before accepting that.

## 1. The constraint that shapes everything: split-batch divergence

Every relief here mutates the batch or the pool. Under uneven DCP the ranks'
`available_size()` differ by construction (weighted ownership), so **any branch
taken on a rank-local size can split the group**: the binding rank relieves,
its peers do not, the batches diverge, and the ranks stop agreeing on which
collectives run. That is a hang, not a slowdown.

This tree has already paid for that lesson twice on this path, and both scars
are load-bearing precedent:

- **#603** — the decode-OOM *decision* moved from `check_decode_mem`'s local
  comparison to the min-reduced value. The eviction side effect stayed local;
  only the comparison moved.
- **#583 desync site 2** — the decision was uniform but the *loop bound* was
  not, so ranks entered retraction together and then popped **different numbers
  of victims**. A rank that popped to empty skipped `run_batch` and went round
  to `recv_requests` while its peers entered the decode collective. Observed
  2026-08-05 21:10.

So the rule this ladder is built on, stated once:

> **Every quantity that feeds a branch OR a loop bound must be the reduced
> one.** Not just the branch. The reduce itself happens once per iteration,
> pre-branch, unconditionally (`_update_uniform_pool_budget`), so the decision
> point takes **no collective at all** — it reads a value that is already
> agreed.

### 1.1 The iteration order, verified — and it is favourable

```
scheduler.py:4777   _update_uniform_pool_budget()     the reduce
                    "Unconditional and pre-branch ... every rank reaches this
                     line exactly once per iteration"          (its own comment)
scheduler.py:5089   get_new_batch_prefill(...)        ADMISSION — this ladder
scheduler.py:5137   update_running_batch(...)         where retract_decode lives
```

Two consequences, both load-bearing:

1. **Rungs 1–3 need no new collective at admission.** The reduced value is
   already published by the time `get_new_batch_prefill` runs. A ladder there
   reads an agreed number and takes no reduction of its own — the same
   no-collective-at-the-decision-point property the decode path has.
2. **Retraction is genuinely downstream.** `retract_decode` sits at 5950 inside
   `update_running_batch` (5818), which runs *after* admission at 5089. That is
   not an accident of style; it is why #679's crash had nothing to fall back on,
   and it is exactly what rung 3 has to work around.

And the subtlety worth copying rather than rediscovering: the decision uses the
**pre-evict** reduced value on purpose. In the regime where the decision is
close, eviction frees ≈ nothing, so the pre-evict number is both exact and
uniform. Re-reducing after eviction would buy nothing and cost a collective.

## 2. The reliefs that exist today

Four, and only four, can free KV at an admission decision. Ranked here by what
they cost, not by what they free.

### 2.1 Radix eviction — `evict_from_tree_cache`

- **Frees:** tokens held only by the prefix cache, i.e. cached prefixes no
  live request is locking.
- **Costs:** future prefix-cache hits. No request loses progress.
- **Group uniformity:** already solved. The side effect is rank-local and
  always run; `uniform_avail_for_evict` gives the group-published floor for the
  *decision*. This is the pattern, not an exception to it.
- **At the admission point:** already spent. `alloc_token_slots` calls it
  before allocating, and the #679 crash walked straight through it with
  `evictable 0`. **It is the cheapest rung and it is already at the bottom of
  the ladder — treat it as the baseline, not as a rung to add.**

### 2.2 `#287` ladder — `throttle_before_retract` / `admission_cap`

- **Frees:** nothing, immediately. It lowers *inflow*.
- **Costs:** throughput, for as long as the cap is down.
- **Group uniformity:** the ladder's own contract — replicated inputs, a
  consensus boundary every `consensus_interval` (default **8**) rounds, and a
  MIN-reduction carrying `[v, -v]` pairs so a desync raises `KvLadderError`
  rather than hanging.
- **At the admission point:** **too slow to be a rung**, and this is the key
  finding for #677. A cadence of 8 rounds cannot answer a chunked-prefill burst
  that exhausts the pool in fewer. Its actuators reshape *future* admission.
  It also requires `--kv-pressure-ladder` to exist at all, and is
  planned-only for `kv_spill` / `weightless_rank`.
  **Use it as the slow outer loop, never as the fast inner one.** The decode
  path already treats it exactly this way: `throttle_before_retract` runs
  *before* retraction to stop the freed slots being handed straight back on the
  next prefill pass — it prevents the *repeat*, it does not fix the *instance*.

### 2.3 KV session offload — `kvso.try_spill(batch, need=…)`

- **Frees:** the victim's **tail overhang only** — `chunk_ceil(need,
  block_size)` tokens, block-aligned, over-eviction margin ≤ block−1. The head
  `[0, boundary)` stays device-resident and keeps its tree lock and protected
  prefix. A whole-session spill is just the `boundary == protected` special
  case.
- **Costs:** host bandwidth, and the victim decodes from host afterwards. It
  does **not** lose progress — that is the whole point versus retraction.
- **Group uniformity:** decided from `dcp_min_avail()` (the same single
  pre-branch reduce), and `need` is sized from that reduced value, so every
  rank spills the same shortfall. **One victim into one free host region per
  call.** Returns `False` when no region is free — never an inconsistent
  partial spill.
- **At the admission point:** **the best rung available.** It frees a bounded,
  *chosen* amount, it is already reduced-value-driven, and it costs no
  request's progress.
- **Its bound is the host region supply.** When `try_spill` returns `False` the
  budget is exhausted and the ladder must fall through — that is a real,
  reachable state (the decode path documents it as the trigger for stock
  retraction), not a theoretical one.

### 2.4 `retract_decode(server_args)`

- **Frees:** whole requests' KV, released **without** inserting into the tree —
  "we need the space instantly". The most tokens per call, by far.
- **Costs:** the victim's entire decode progress; it goes back to the waiting
  queue and re-prefills. Under FCFS this is the loudest thing the scheduler can
  do to a user.
- **Group uniformity:** requires **two** reduced values, not one — the entry
  decision *and* `batch.uniform_avail_floor`, which bounds the retraction loop
  and the last-survivor test. #583 is exactly the case where only the first was
  reduced.
- **At the admission point:** **last resort, and it is the rung the prefill
  path cannot currently reach at all** — `retract_decode` is called only from
  `update_running_batch`, which runs *after* `get_new_batch_prefill` in the
  same iteration. That ordering is the whole reason #679's crash had nothing to
  fall back on.

## 3. The ladder, in order, before a park

The decode-OOM branch already implements this shape. The admission path should
**mirror it rather than invent a second one**:

```
0. evict_from_tree_cache(need)          rank-local side effect, always
                                        (baseline — already in alloc_token_slots)
   decide on uniform_min_avail()        reduced, pre-evict, no collective here
   ── if it fits, admit. Done. ─────────────────────────────────────────────
1. kvso.try_spill(batch, need=need)     bounded, chosen, costs no progress
   ── if True, re-check and admit. ─────────────────────────────────────────
2. throttle_before_retract(limiter, bs) lower inflow so this does not repeat
3. retract_decode(server_args)          with batch.uniform_avail_floor set
   ── if it frees enough, admit. ───────────────────────────────────────────
4. PARK                                 #679: schedule nothing, keep the req,
                                        retry next round
```

Rungs 1 and 3 are the only ones that free tokens *now*. Rung 2 frees nothing
and belongs between them for the reason the decode path gives: it stops the
slots rung 3 just freed being handed straight back to the waiting queue on the
next prefill pass, turning a repeated discard into a single one.

**Parking stays the floor of the ladder, not its replacement.** Rung 4 is
reached when the host region supply is exhausted *and* retraction has nothing
left to take — the state where the only honest answer is "not this round".

### 3.1 What this changes about the empty retry net

Nothing in this ladder belongs at `alloc_token_slots`. Every rung is either
already spent there (rung 0) or collective-adjacent (1–3), and by the time
execution reaches that site the group has committed to a batch. The registry
stays a rank-local seam. **If this ladder is built, the `NO relief provider is
registered` line should become unreachable in practice — and if it still
appears, that is the admission-defect indicator it was written to be.**

## 4. Composition contract with #677 (Q3)

#677 owns a **chunk-admission phase gate with hysteresis drain**. The division
of labour:

| | decides |
|---|---|
| **#677 drain state machine** | **WHEN** admission is attempted — which phase/drain state permits a chunk to be considered at all |
| **#679 park guard + this ladder** | **WHAT** happens once it is attempted — what relief is spent, and whether a chunk is finally scheduled or parked |

Both read pool headroom. That is the collision, and it needs three rules.

**Rule 1 — the park guard is the final authority; the drain gate is the prior
one.** A phase gate cannot make memory exist. If the drain state machine says
"admit now" and the pool cannot fund a page, the chunk **must still park**. The
gate may only ever *narrow* what the park guard would allow, never widen it.
Ordering is therefore fixed: `drain gate (when) → relief ladder (what can be
freed) → park guard (final yes/no)`.

**Rule 2 — any headroom quantity #677 branches on must be the group-published
floor.** `uniform_min_avail()` / `uniform_avail_for_evict()`, never
`available_size()` or `evictable_size()` directly. This is not style. Under
uneven DCP a rank-local reading in the drain gate re-introduces the exact
divergence class of #603/#583, and it will do so *in the gate*, which is
upstream of every safeguard #679 added. The park guard reading the reduced
floor does not protect a gate that reads a local one.

**Rule 3 — parking must be reachable from every path that reaches
`alloc_for_extend`.** A drain state that bypasses the park guard re-opens the
crash. Concretely: if #677 introduces a new admission path (a drain-flush fast
lane, a phase-boundary catch-up), that path needs the same
`chunk_tokens_the_pool_can_fund` check, not a copy of its logic.

### 4.1 The interaction worth testing explicitly

Hysteresis and parking can **beat against each other**. A drain state machine
that admits on a rising edge, plus a park guard that refuses on low headroom,
can produce: admit → park → drain thinks work was taken → hysteresis holds →
nothing progresses. The park is invisible to a gate that counts *admissions*
rather than *scheduled tokens*.

So: **#677's drain accounting must count scheduled tokens, not admission
attempts.** A parked chunk scheduled zero tokens and must not advance the drain
state. If it does, the two mechanisms will deadlock each other at exactly the
pressure where both are needed. This is the single most likely composition
failure and it is cheap to test hermetically on both sides.

## 5. Open items this note does not close

- **The ladder is not built.** This is a design note; #679 shipped the park and
  the (empty) net. Rungs 1–3 at admission are unimplemented.
- **Rung 3 ordering is a real refactor.** `retract_decode` (5950, inside
  `update_running_batch` at 5818) runs after `get_new_batch_prefill` (5089).
  Calling it from the admission path means either moving the call or giving
  admission a bounded retraction entry point — and it must carry
  `uniform_avail_floor` with it or it reintroduces #583. Note the asymmetry
  that makes this tractable: the *decision* needs no new collective (§1.1), so
  the refactor is about reaching the actuator, not about agreeing to use it.
- **The 45 s window amplifies the arrival rate, not the fix.**
  `SGLANG_PHASE_POLICY_PP_WINDOW_S=45` admits ~3× the concurrent prefills of
  the 15 s regime, so rungs 1–3 will be reached more often. Nothing above
  depends on the window length; the ladder is a function of what the pool can
  fund at the moment of scheduling, which is exactly what a longer window
  drives to zero.
- **Rung 1's bound is unmeasured on this rig.** How many host regions exist,
  and therefore how many spills the ladder gets before falling through to
  retraction, has not been measured under the 5-lane load that produced the
  #679 crash.

## 6. Boot validation — what one loaded window must show to turn it ON

Written after the ladder landed (`82ba7e2c10`). The ladder is **off by default**;
this is the evidence required to change that.

### 6.1 The window

- Boot the ladder arm **without rung 3**: `SGLANG_ADMISSION_RELIEF_LADDER=1`,
  `SGLANG_ADMISSION_RELIEF_RETRACT` unset. Rungs 1–2 cost bandwidth and latency;
  rung 3 destroys decode progress. Judge the cheap arm first.
- **Five lanes**, the load that produced #679 and #680.
- **≥ 30 minutes.** Both crashes arrived ~19 minutes in; a shorter window that
  shows nothing has shown nothing.
- A **ladder-off control** at comparable pressure. The `3374029942` boot is one.

### 6.2 The null result, stated first

If `KV-ADMISSION-LADDER` never appears, the window **proves nothing** — the
pressure regime was not reached. That is a null result, not a pass, and it must
not be recorded as one. Every criterion below is conditional on engagement.

### 6.3 Accept

1. **Engagement.** `KV-ADMISSION-LADDER` present, with its before/after
   availability line.
2. **Parks fall per engagement.** See §6.5 — the ratio, not the count.
3. **No hard OOM.** Zero `Out of memory. Try to lower your batch size`.
4. **No `NO relief provider is registered`.** That line means admission let
   unfunded work reach the allocator, which the ladder was supposed to make
   less likely, not more.
5. **Both flip directions still commit.** The ladder must not starve the seam
   by spilling or retracting what the flip needed.
6. **No new wedge.** No `WITHHOLDING presence` spin, no rank divergence. The
   split-batch class is an immediate reject, not a regression to weigh.

### 6.4 Interaction with the #680 warning — do not read them independently

`draft_token arrived as torch.int32 ... #680` fires when the **trivial verify
path** is taken, and that path is entered when **drafting is disabled at high
batch size** (or on a flip bootstrap).

The ladder pushes on exactly that input. Rung 2 lowers inflow and rung 3
retracts victims; both **shrink the batch**, which can re-enable drafting and
make the trivial path *less* frequent.

Two consequences, and both are easy to get backwards:

- **The #680 warning going quiet is NOT evidence that the #680 fix is inert.**
  It may be evidence the ladder is working. Do not conclude "the dtype fix never
  fires, so it was unnecessary" from a ladder-on window.
- **The #680 warning becoming MORE frequent is a ladder warning sign.** It would
  mean the ladder is driving the instance into the degenerate-batch regime —
  retracting or throttling so hard that batches collapse into the trivial path.
  That is a rung-3 over-actuation signal, not a speculation bug.

So the #680 line is read here as a **batch-shape probe**, not as a dtype signal.

### 6.5 Interaction with the park counters — the ratio is the metric

The ladder runs immediately before the park decision, so **every ladder
engagement is a park that was about to happen**:

```
engagements   = KV-ADMISSION-LADDER lines
parks_after   = "chunked prefill PARKED" lines
rungs_paid    = engagements - parks_after        (relief that prevented a park)
```

- **Judge `parks_after / engagements`, never `parks_after` alone.** More parks
  can simply mean more pressure. A ladder that engaged 100 times and still
  parked 100 times freed nothing.
- A ratio near **1.0** with engagement present is the **host-region bound**
  showing itself — rung 1 exhausted and rung 3 disabled in this arm. That is
  the quantity §5 says has never been measured, and this window measures it.
  It is a finding, not a failure: it says the cheap arm cannot pay here and
  rung 3 is required.
- A ratio near **0** means rungs 1–2 carry the load and rung 3 may never be
  needed on this rig.

### 6.6 Rung 3, only as a second window

Enable `SGLANG_ADMISSION_RELIEF_RETRACT=1` only after 6.3 passes. Additional
requirement, because rung 3 is the only rung that can lose work:

- **Every retracted request must come back.** `KV cache pool is full. Retract
  requests.` lines must be matched by those requests completing afterwards. The
  shared actuator requeues them (`_add_request_to_queue(..., is_retracted=True)`)
  and a test pins that, but a leak here costs a user their request, so it is
  confirmed on metal rather than assumed.
- Watch the #680 line per §6.4: rung 3 is the rung most able to collapse batches.

### 6.7 Reject

Any of: a hard OOM; a rank divergence or wedge; retracted requests that do not
return; parks not falling per engagement **while rungs report they paid** (which
would mean the accounting lies); or a decode-latency regression beyond what the
spill's host bandwidth explains.
