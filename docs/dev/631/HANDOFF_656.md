# #656 HANDOFF — automatic PP/TP phase-flip controller

Written 2026-08-09. Tree `/spinning/wt-631-routea`, branch `feat/route-a-631`.

Read this, then the DESIGN LAW header of
`python/sglang/srt/managers/phase_flip_presence.py`. That header is the
design document and carries the corpse table A–H: every falsified design,
what was *observed* versus what was *inferred*, and a can-fail test for
each. Do not re-walk any of them.

---

## 1. State right now

- **HEAD**: `cc7d1a0edf`
- **Tests**: `bash scripts/run_631_flip_family.sh` → 349 passed.
  `test/registered/unit/managers/test_pp_chain_receiver.py` → 8 passed
  (not in the family script; run it too).
- **Production**: serving on port 30030, `POLICY=manual`, healthy.
- **The flip does not yet complete a cycle.** One defect remains (G).

### The last commits, and what each carries

| Commit | Carries |
|---|---|
| `cc7d1a0edf` | **H**: publishable withdrawal (`WITHDRAWN`/`ENTERING` markers), two-phase entry, tie-break |
| `8b9704e21a` | Metal verification: deadlock class gone; defects **G** and **H** exposed |
| `d547568117` | Quiescent-announce + bounded spin at the hook; mislabeled-collective fix |
| `2c838b771f` | Metal: round-scoping necessary but **not** sufficient; two corrections to earlier readings |
| `b51480f177` | **Round-scoped entry evidence** (evidence scope = guarantee scope) |
| `dc4a8549bf` | Boot 18 **reproduced** with all three stacks |
| `526e53cffc` | **Corpse F**: the non-blocking pump is measured dead, and always was |
| `8b132eded4` | First boot-18 fix, built then falsified by its own transport premise |

---

## 2. THE MEASURED TRANSPORT FACTS

These are executable pins in `test_pp_chain_receiver.py`, not opinions.
**Every design that ignored one of them has died.**

1. A posted `irecv` **never** completes by polling `is_completed()` — 4 s,
   8 B and 512 KiB. A non-blocking drain built on it absorbs nothing.
2. An `isend` **never** completes by polling `is_completed()` either —
   *not even after the peer has fully consumed the message*. This is why
   `pp_pump_send_req_work` is dead code (corpse F).
3. **Only `wait()` progresses a transfer on this build.**
4. Two-sided wire fact: an unconsumed forward returns in 0.00 s when the
   receiver has *posted* an irecv and merely not completed it; when the
   receiver has posted **no** irecv, the sender's `wait()` **blocks**.
5. Positive evidence, the one behaviour to build on: the **recv side's
   `wait()` drives the transfer**. Arms propagated correctly across boots
   14–18 by exactly this route.

---

## 3. WHAT REMAINS: defect G, and the approved fix

### The defect (measured 2026-08-09 00:06–00:09Z)

A spinning rank stops issuing the per-pass chain forward. Downstream
stages reach the hook **only** by returning from their blocking chain
recv, which that forward is what satisfies. So the first rank to become
quiescent — rank 0, the intake rank, always — prevents every rank behind
it from becoming ready. The retry is **bounded but not convergent**: the
same rank always drains first, so the starvation reproduces identically
every epoch.

Note it starves **both** inbound directions: rank 0 spinning also starves
rank 2's output-return sends, so rank 2 can never quiesce either. Cover
both channels or the fix does nothing.

### The approved design: the ARMED SERVICE LOOP with a SEND-COUNTER

While **armed**, a rank replaces its blocking pass loop with a
non-blocking service loop that:

1. **Greedily consumes every inbound channel** — the request chain *and*
   the output-return path — never skipping an available message.
2. **Never blocks on any outbound.** A quiescent rank owes none; a
   draining armed rank still does its real sends inside normal passes.
   The service loop replaces only the *waits*.
3. **Reaches the hook by poll**, so no rank depends on upstream traffic to
   arrive at the entry.

Spin-at-hook then degenerates to this same loop with `ready=true` — one
mechanism, not two.

**How to consume without `is_completed()` (this is the load-bearing part).**
Use a monotone send-counter alongside the presence markers:

- Each sender publishes a per-message monotone counter in `/dev/shm`,
  single-writer, same discipline as the gate's markers.
- **THE ORDERING CONSTRAINT: publish the counter strictly AFTER the isend
  is posted.** The only possible skew is then *counter-lags-send*, which
  is the safe direction — a message may be consumed late, never
  phantom-received. Publishing first would let a receiver block on a send
  that does not exist.
- A receiver compares the published count against its own consumed count.
  When the sender is ahead, a message is **provably in flight**, so the
  *blocking* `recv()` is safe to call: it is bounded by transfer time, not
  by peer scheduling. This is fact 5 above, used deliberately.
- Framing comes from `PpChainReceiver`
  (`python/sglang/srt/managers/pp_chain_receiver.py`), currently parked
  behind `SGLANG_PP_CHAIN_RECEIVER=1` (default off). It is a correct
  two-step state machine — `point_to_point_pyobj` sends a size then a
  payload, and once the size is consumed the payload **must** be consumed
  or every later message is misframed. This build finally earns its keep;
  use its `recv()`, not its `poll()`.

**Why this is not the bounded-recv corpse.** That corpse's failure driver
was completing iterations *without* consuming while the upstream kept
sending: rates decoupled, unmatched sends accumulated, senders blocked.
The service loop consumes **greedily** (never skips an available message)
and exists **only in the armed state**, where admissions are held — so the
accumulation driver is absent by construction. Write this distinction into
the corpse table next to the original entry; it will look like a
resurrection to the next reader.

### Flip-commit hygiene (required)

At entry, quiescent + fully-serviced implies **all inbound channels are
empty**. Assert it loudly. A non-empty channel at entry means a framing or
quiescence bug, and a stale half-consumed message crossing the
re-formation would misframe the post-flip stream. Cheap assert, catches the
nastiest silent failure this change can introduce — including a sender
that crashed between posting and publishing its counter.

### Can-fail set for G

1. **The metal specimen**: at-boot idle `POLICY=auto` must **COMMIT** a
   flip, not abandon-recur.
2. **Recurrence falsifier**: the epoch-recurring abandonment pattern from
   the 00:06Z run must be impossible — N epochs without convergence at
   idle = red.
3. **Channels-empty-at-entry** can-fail.
4. All existing pins stay green (349 + 8).

---

## 4. THE ACCEPTANCE PROGRAM (only after G lands and a flip commits)

Run on the same boot pattern, `POLICY=auto`, **zero manual flips**:

- Mixed load: `scripts/route_a_631_policy_acceptance.py` — concurrent long
  prefills + decodes, bs ≤ 4.
- Report from **one unmanned log**: flip cadence; **PP-class prefill
  throughput**; **TP decode with accept length**; the **idle-return-to-PP**
  leg; **abort count**.
- If pass: **regression gate A-vs-A vs `9a929352c9`**.
- Then: **ship + reboot production from the named commit**, update
  `/spinning/gpu-arb/holder` and the policy section of
  `docs/dev/631/PROD_BRINGUP_BENCH.md`.

**The honesty bar**: a pass is a completed **PP → TP → PP cycle under load
in one unmanned log**. Not a report. Not "the wedge changed shape".

---

## 5. Evidence on disk

`/spinning/evidence-631/` — three full three-rank specimens, each with
Python + `--native` + `--locals` stacks, presence markers, process table
and a bounded log slice:

- `wedge_20260808T231450Z_INSIDE_REDUCTION` — the epoch-scoped wedge, and
  **the first capture that ever recorded rank 2's stack**.
- `wedge_20260808T233910Z_KVPRESSURE_DIVERGENCE` — the round-scoped build
  still wedged; despite the name this is the **flip's own** reduction (see
  §6).
- `wedge_20260808T230757Z_dryrun` — healthy-idle baseline for comparison.

Capture harness: `scripts/route_a_631_wedge_capture.sh`. Automatic
triggers (presence abandonment in the log, or repeated `health_generate`
failure); `--once <tag>` for a manual capture. **Prove it against a
healthy server before relying on it.** Boot 18's rank-2 stack was lost
because capture was manual and the log was truncated — the boot script now
rotates the serving log instead.

---

## 6. Traps that already cost time

- **The mislabeled collective** (fixed in `d547568117`, but know it):
  `default_collective_min` used to hardcode its own module's label, so the
  flip's reduction timed out as `kv_pressure_ladder.consensus` and sent a
  live investigation into the wrong subsystem for an hour. It now takes a
  `label`; the flip passes `phase_flip.consensus`. **There is no second
  subsystem bug.**
- **Do not trust a predecessor's stack attribution.** Boot 18's rank-2
  state was never recorded, and reasoning built on that guess survived
  several designs. The corpse table now separates observed from inferred;
  keep it that way.
- **A pin that hangs tells you nothing.** One can-fail test spun forever
  under mutation on a frozen fake clock. Make mutated code *terminate and
  fail*, not hang.
- **`is_completed()` looks like it works.** It does not. See §2.

---

## 7. Standing warnings

- **`POLICY=auto` must not be booted until G lands.** It wedges at boot
  with zero traffic. `POLICY=manual` is the safe configuration and is what
  serves.
- **Production stays manual and serving.** Whoever stops it owns bringing
  it back.
- The **watchdog is decommissioned** by user order — do not re-arm it.
- **Port 30099 (the local router) is never touched** from an agent session.
- No broad `pkill` — match your own PIDs (a self-kill already happened
  once). Commits: author `efschu` only, no trailers, English throughout,
  test results in the commit message.
- GPU arbitration via `/spinning/gpu-arb/holder`: read it, update it on
  every boot.

---
---

# HANDOFF v2 — written 2026-08-09 by successor #3

**Everything above is still true as history. This section supersedes §3
(defect G) and §4 (the acceptance program's readiness).** Read the corpse
table in `python/sglang/srt/managers/phase_flip_presence.py` alongside it;
entries G (resolved), I and J are the current state of the art.

## 1. State

- **HEAD**: `264f6142da` on `feat/route-a-631`.
- **Tests**: `bash scripts/run_631_flip_family.sh` → **379 passed**. The
  script is now the whole family: it also carries
  `test_pp_chain_receiver.py` and the new `test_phase_flip_counters.py`,
  so there is no longer a second suite to remember.
- **Production**: serving on 30030, `POLICY=manual`, healthy
  (`GET /health_generate` 200).
- **`POLICY=auto` is UNATTENDED-FORBIDDEN.** It now *commits* the flip and
  then dies ~1 pass later. See §4.

## 2. What landed

**Defect G is fixed and proven on metal.** The armed service loop with
monotone `/dev/shm` send-counters (publish strictly after the isend post)
went in exactly as specified. All three ranks now reach the flip entry
with no traffic driving them there — "group present for epoch 0 after
0.00s", all three ENTERING markers, rank 0's stack inside the reduction.
Every predecessor boot had a rank blocked upstream of the gate.

**The first policy-driven flip in this feature's history committed**, at
2026-08-09 01:37:12Z, and has reproduced on every `POLICY=auto` boot
since. Log: `/spinning/evidence-631/FIRST_AUTO_FLIP_COMMIT_20260809T0137Z.log`.

**G was not the last thing in the way, and the real unblocker was
elsewhere.** `build_flip_quiescence_fn` called
`Scheduler._pp_microbatches_drained` — the *fully-idle* predicate, which
also requires an empty `running_mbs`, the resident decode set. The policy
arms `pp_to_tp` precisely *because* requests are decoding, so the arming
condition and the quiescence condition could never hold together and every
automatic flip abandoned at the park deadline, for ever. It also
contradicted the function's own docstring and `build_flip_live_slots_fn`.
Quiescence now reads `mbs` (pipeline in flight) only.

**Permanent diagnostics** (keep them — they paid for themselves four
times): the withhold reason, the quiescence reason, the flip extent probe,
and the **pool census bracket**. Each of the diagnoses below was cheap
only because of them.

## 3. Defect J, in three parts

**J.1 — SLOT SCOPE. Proven, fixed.** `scheduler.running_batch` and
`last_batch` are rebound to `running_mbs[mb_id]` / `last_mbs[mb_id]` at the
top of every slot iteration under `event_loop_pp`. They name ONE
microbatch slot, not the rank's resident set, and the flip's hook fires at
the end of an arbitrary slot. Evidence:

    at-arm       cur_slot_reqs=1 resident_reqs=1 resident_slots=[1]
    pre-cutover  cur_slot_reqs=0 resident_reqs=1 resident_slots=[1]

Rows not enumerated are not MOVED, so this was silent context corruption,
not an accounting bug. `_live_reqs` now enumerates every resident slot.

**J.2 — ROW EXTENT. Measured, deliberately NOT cut.** With J.1 fixed the
extent probe fires for the first time (page_size=1):

    seqlen=82  kv_allocated_len=81  kv_committed_len=81
    cache_protected_len=80  delta_vs_seqlen=-1

`seqlen` **over**-counts, so the enumeration currently moves a row the
allocator does not own. The right basis is `kv_allocated_len`
(page-aligned) — it is the invariant checker's own charge basis, so
enumeration and invariant would then share one source of truth and the
seqlen arithmetic disappears entirely.

**OWED BEFORE CUTTING IT** (operator instruction, not optional): take the
measurement on a flip with the **spec path LIVE**. In the reading above
`kv_committed_len == kv_allocated_len`, i.e. the #486 reserve was at rest.
That reserve is `W + L` — `W = get_alloc_len_per_decode()` = max(topk *
num_steps, num_draft_tokens) = 4 on this rig's NEXTN config, plus commit
lag L — so the general form of the over-count is still unknown and one
config's sign is not enough to re-cut an enumeration whose errors are
silent.

**J.3 — THE CUTOVER DOES NOT CARRY THE RESIDENT SET. Root cause, and the
blocker.** Survival oracle, 2026-08-09 02:36:05-07Z:

    POOL CENSUS at-arm        cur_slot_reqs=1 resident_reqs=1 slots=[0]
    POOL CENSUS pre-cutover   cur_slot_reqs=1 resident_reqs=1 slots=[0]
    POOL CENSUS post-cutover  cur_slot_reqs=0 resident_reqs=0 slots=[]
    cutover complete x3, then
    AssertionError: x_lru should not be locked when idle,
        x_lru.full_lock_ref=1, x_lru.id=5
    -> Mamba Radix tree sanity check failed -> SIGQUIT

The request is present and enumerated right up to the cutover and gone
immediately after. **The KV move is fine** (balanced cells, and J.1's
enumeration now covers the request); the CUTOVER drops the requests when
it swaps stacks and scheduler topology. The stranded KV page and the
stranded mamba lock are TWIN SYMPTOMS OF ONE OMISSION — fixing either
alone is treating a shadow.

**Consequence, stated plainly: a flip under load is not merely unproven,
it is currently IMPOSSIBLE.** Any request resident at the cutover is
destroyed. Every flip observed so far committed only because nothing had
to survive it.

## 4. Method notes that will save the next reader a boot

- **The falsifier of record for J** is the POOL CENSUS bracket (`at-arm` /
  `pre-cutover` / `post-cutover`, reproducing the invariant checker's
  `expected - free - cached` arithmetic) plus the survival oracle.
- **The unmasking trick**: `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`
  demotes the idle leak check from raise to warn. Without it the
  accounting crash fires first and hides everything downstream of it —
  that demotion is what exposed the mamba lock and J.3.
- **Two of my own hypotheses died here, both plausible, both wrong.**
  First: "seqlen under-counts the spec reserve, so a row is missed" —
  falsified by the census (the page was unaccounted *before* the flip
  touched anything, and a no-flip control boot stayed clean). Second: "the
  request finishes while armed and its overallocated row is never freed" —
  falsified by reporting both slot scopes (it had not finished; the hook
  was sampling an empty slot). **Do not re-derive either from the symptom;
  they fit it perfectly and are still wrong.**
- A wrong guess in KV-row accounting does not fail loudly. It moves the
  wrong bytes or leaves the right ones behind, and the request's context
  is then quietly corrupt. Measure first, every time.

## 5. THE NEXT BUILD, named precisely

**Carry the resident request set across the stack/topology swap** —
`build_production_flip_cutover` plus the scheduler topology snapshot in
`phase_flip_runtime`. What must cross is not just KV cells:

1. the **request objects** themselves (the resident set across all
   microbatch slots — see J.1 for why `running_batch` is the wrong
   handle);
2. the **scheduler bookkeeping** for them (`req_to_token` rows,
   `kv_allocated_len` / `kv_committed_len` / `cache_protected_len`, the
   radix locks and protected refs);
3. the **mamba/GDN state AND ITS LOCKS** — the stranded `full_lock_ref=1`
   is the direct evidence that these are not currently handled.

**Standing architecture context to read BEFORE designing this** (do not
skip; the resident-carry design has to compose with it):

- **#635/#636**: KV handover between the PP layout (`dcp_size=1`) and the
  TP layout (`dcp_size=3`) is **not a simple transfer**. It hangs on four
  silent preconditions. The resident-carry must compose with that reshard,
  not sit beside it.
- **#212**: store routes **truncate GDN state**. Mamba/GDN state must be
  moved **deliberately and explicitly**; anything that relies on a store
  path to carry it will silently lose it. This is very likely why the
  mamba lock strands today.

Then, in this order: the **can-fail set written from what that build
actually produces** (not from a mechanism guessed in advance — that is why
the pins were not written yet), then the **acceptance program verbatim**
from §4 of the original handoff, with the honesty bar unchanged: a
completed PP→TP→PP cycle **under load** in one unmanned log, not a report.

## 6. Standing warnings (unchanged, and one added)

- `POLICY=auto` must not be left unattended: it commits the flip, then
  dies. `POLICY=manual` is the safe configuration and is what serves.
- Production stays serving; whoever stops it owns bringing it back.
- Watchdog stays decommissioned. Port 30099 is never touched.
- Commits: author `efschu` only, no trailers, English throughout, test
  results in the message.
- **New**: `scheduler.running_batch` under `event_loop_pp` is a per-slot
  handle. Anything reading it from a per-slot hook and treating it as the
  rank's resident set is making J.1's mistake. Flagged as an audit
  candidate beyond #631; not chased here.

---
---

# HANDOFF v3 — written 2026-08-09 by successor #4

**Supersedes v2 §5 ("THE NEXT BUILD"): that build is DONE and proven on
metal.** The corpse table in `phase_flip_presence.py` now carries entries
K, L and M; read those with this.

## 1. The headline

**A flip under load works, in both directions.** v2 closed with "a flip
under load is not merely unproven, it is currently IMPOSSIBLE -- any
request resident at the cutover is destroyed". That is no longer true.
Measured, one request decoding throughout, `SPEC=off`:

    PHASE-FLIP-CARRY carried 1 resident request(s) ... into the tp phase
    PHASE-FLIP DONE pp_to_tp (epoch 1)    x3 ranks
    PHASE-FLIP-CARRY carried 1 resident request(s) ... into the pp phase
    PHASE-FLIP DONE tp_to_pp (epoch 2)    x3 ranks
    oracle: committed legs {'pp_to_tp': 3, 'tp_to_pp': 3}
    oracle: correct prefix 197 numbers (no break: every token was the
            next integer)
    oracle: VERDICT PASS

A `PP -> TP -> PP` round trip under a live request, both cutovers
committed on every rank, and the answer exactly right.

## 2. Three defects stood between v2 and that result

Each was found by a leg that could not commit, never by reading code.
Full reasoning is in the corpse table; the short form:

- **K (= J.3's fix): the cutover dropped the resident decode set.** Step 6
  calls `init_pp_loop_state()`, which rebinds `running_mbs` -- and under
  `event_loop_pp` that array IS the resident set. The stranded KV page and
  the stranded mamba lock were two symptoms of one omission. The carry
  lives INSIDE `init_pp_loop_state`, not at the cutover, because
  `event_loop_pp` calls it again at its own entry and would otherwise wipe
  a carry installed for it. New module:
  `managers/phase_flip_resident_carry.py`.
- **L: `tp_to_pp` could never reach quiescence under load.** `last_batch`
  non-empty means "requests are resident", not "work is in flight", under
  `event_loop_normal`. Same category error as the
  `_pp_microbatches_drained` one v2 fixed. General form, now stated twice
  and worth remembering: **what must be quiet is the MACHINERY, never the
  WORKLOAD.**
- **M: the PP chain's ring was read off the live `ps`.** The cutover
  rewrites `ps` per phase, so in TP (`pp_size=1`) upstream degenerated to
  SELF, and the hygiene check compared a rank's own send counter against
  its own consume counter. Rank 0 withheld presence for 8889 rounds over
  24 messages nobody had failed to consume. General form: **any quantity
  derived from `ps` is phase-scoped now.**

**Plus a second, SILENT occurrence of J.1, found by audit** (v2 flagged
the class as an audit candidate; this is a confirmed second instance):
`gdn_flip_mover` enumerated `scheduler.running_batch` -- one microbatch
slot -- so a request resident elsewhere kept correct KV and had its
conv/ssm state left behind. #212's truncated-GDN shape, and nothing
raises. Now `resident_mamba_slots()` over the same `_live_reqs`
authority. **The class is NOT closed**: anything reading
`running_batch`/`last_batch` from a per-slot hook is still suspect.

## 3. THE WALL: speculation and the carried request

**`SPEC=on` + a carried request KILLS THE INSTANCE.** This is the one
thing between here and the acceptance program as written, and it is a
separate build rather than a variant of the carry.

Measured 2026-08-09 03:32:14Z, all three ranks, one pass after a
committed `pp_to_tp` that carried one request:

    File ".../speculative/eagle_worker_v2.py", line 2067, in
        forward_batch_generation
    File ".../speculative/eagle_worker_v2.py", line 1021, in draft
    File ".../speculative/eagle_draft_cuda_graph_runner.py", line 623,
        in execute
    File ".../model_executor/runner_utils/buffers.py", line 47, in
        foreach_copy
    RuntimeError: output with shape [1, 1] doesn't match the broadcast
        shape [0, 1]
    -> SIGQUIT

**The source is EMPTY where the destination expects one row**: the
carried request's draft input has no rows. The cause is structural and is
not a bug in the carry: **speculation belongs to the TP decode phase, so a
request that prefills in the PP phase has no draft state at all** -- the
PP phase carries no draft worker by design (`build_flip_draft_worker`
returns None there, and the cutover documents the PP phase as
"bit-for-bit the state an instance without speculation has"). In a normal
spec instance a request gets its draft KV from the `draft_extend` that
follows its target extend. A carried request never had one.

This is not an edge case -- **it is #631's central path**: every request
is supposed to prefill in PP and decode in TP.

Options, none of them tried yet, in increasing order of principle:

1. **One non-spec target decode step after the cutover** for carried
   requests, to produce a hidden state, then `draft_extend` them in. The
   draft's prefix KV stays empty, so acceptance for those requests starts
   low and recovers; correctness is unaffected because the target
   verifies every proposed token.
2. **Carry the target's last hidden state per resident request across the
   flip** (one vector per request) and run `draft_extend` at the cutover.
   Cheaper, and it is the same mechanism sglang already uses at the
   prefill->decode transition.
3. Full draft-prefix reconstruction. Requires target hidden states for
   the whole prefix, which the PP phase discards. Expensive; probably not
   worth it, since (1) and (2) are exact and only cost acceptance rate.

**The hazard to watch for in ANY of them**: sglang assumes the draft's
sequence length equals the target's. A carried request whose draft
context legitimately starts mid-sequence violates that, and getting it
wrong is the silent kind -- the draft reads uninitialised rows and
nothing raises, because the target still verifies. Design the draft's own
seq_len bookkeeping deliberately, and pin it.

## 4. The survival oracle is now a standing harness

`scripts/route_a_631_survival_oracle.py`. Two properties were bought with
misleading results and must not be simplified away:

- **Commit evidence comes from the serving log, not the HTTP response.**
  `/phase_flip` returns 200 for ARMED; a leg that parks and abandons
  returns 200 too. A refused leg once read here as a green round trip.
- **The verdict is anchored to the FLIP POINT**, not to total length. The
  first probe was a raw completion and the model editorialised about how
  far to count -- the no-flip control drifted EARLIER (32 numbers) than
  the flip run (43). That control is what proved the drift was the model
  and not the cutover, and it is why the probe is now chat with thinking
  off.

## 5. Method notes

- **Every defect this session was found by a leg that could not commit,
  and the permanent diagnostics are what named each one in one boot**:
  the withhold reason gave M verbatim ("tensor-dict wire has 24
  unconsumed message(s) from rank 0" -- from rank 0, about itself), the
  quiescence reason gave L, the pool census bracket gave K. Keep them.
- **A green oracle that never flipped is the failure mode to fear.**
  Check `committed legs` before believing any verdict.
- `SPEC=off` on the boot script isolates the carry from the draft
  question. Use it for any carry work.

## 6. THE ACCEPTANCE PROGRAM — where it actually stands

Run 2026-08-09 03:45-03:50Z, `POLICY=auto SPEC=off`, N=4096, min dwell
15 s, idle dwell 5 s, `scripts/route_a_631_policy_acceptance.py`
(1 prefill worker on 8k/32k rungs, 3 decode workers, 180 s mixed then
60 s idle). The script issues NO flip call of any kind.

    phase at start: tp
    observed phase timeline (10 transitions):
      t=  2.0s tp   18.0s pp   44.0s tp   70.0s pp   76.0s tp
        108.1s pp  110.1s tp  134.1s pp  162.1s tp  208.8s pp
    phase after idle: pp
    post-idle 32768-tok probe from rest: 7216.5 ms (4540.7 tok/s)
    requests: 27 total, 27 ok, 0 ABORTED

Against the operator's bar, item by item:

| Required | Status |
|---|---|
| one automatic PP->TP->PP cycle under mixed load | **YES** -- five of them |
| POLICY=auto, ZERO manual flips | **YES** |
| flip cadence | **YES** -- 12 arms, every interval >= the 15 s min dwell (23, 31, 16, 15, 15, 19, 17, 37, 16, 15, 15 s); no thrash |
| PP-class prefill throughput | **YES** -- 4540.7 tok/s on the 32k rung from rest, in PP |
| idle-return-to-PP leg | **YES** -- returns to the resting layout and stays |
| abort count | **YES** -- 0 of 27 |
| TP decode **with accept length** | **NO** -- `SPEC=off`. Blocked by section 3's wall. |

**So the honesty bar is met for everything except accept length**, and
that one item is blocked by a named, reproduced defect rather than by
anything unmeasured. Do not report this as a full acceptance.

**The 2 s gap at t=108.1->110.1 is a sampling artifact, not thrash**: the
timeline is sampled from the log while the arm-to-commit latency varies,
and the ARM timestamps are all >= 15 s apart.

### Defect N, which this run exists because of

The FIRST attempt at this run (03:36-03:39Z) produced 2 transitions and
spent its entire 186 s mixed phase in the TP layout -- prefill-heavy load
served by the decode layout. **The operator spotted it from the outside**
("der Last zufolge läuft gerade das tp layout und nicht das pp layout")
before the run had even finished.

The cause was the fourth instance of this feature's recurring shape:
`pending_prefill_tokens` summed the WAITING QUEUE while the comment at
its use site said "admitted but not yet computed". A long prompt under
chunked prefill sits on `scheduler.chunked_req`, not in the queue, so the
policy read 0 pending prefill for the whole duration of exactly the work
the PP layout exists for, and the TP->PP rule could not fire. The metric
now adds the unfilled remainder (`len(origin_input_ids) -
extend_range.end`, the scheduler's own chunked-prefill basis).

**A declining policy was also silent** -- only arming decisions were
logged -- so "wrong layout under load" and "the hook never runs" looked
identical, and the second was my first hypothesis and was wrong. The
throttled hold reason now carries the standing reason with its two
inputs; it is what made the fix verifiable in one boot.

## 7. J.2 IS STILL OPEN, and the wall is why

v2 left J.2 with an explicit instruction: take the row-extent measurement
with the **spec reserve LIVE** (`get_alloc_reserve_per_decode` = W + L,
W = 4 on this rig's NEXTN config), and only then cut
`build_flip_live_slots_fn`'s enumeration from `seqlen` to
`kv_allocated_len`. **That measurement was NOT taken and the cut was NOT
made.**

The reason is structural rather than an omission, and it is worth
stating so the next reader does not go looking for the boot that was
skipped: the probe only reports when a flip ENUMERATES A RESIDENT
REQUEST, and with speculation armed a flip with a resident request now
deliberately WAITS (section 3). So `SPEC=on` never produces the reading,
and `SPEC=off` produces it with the reserve at rest -- which is exactly
the reading v2 already had and already ruled insufficient.

**J.2 is therefore blocked behind the same wall as the acceptance's
accept-length item.** Clearing the draft-state question unblocks both at
once, which is the strongest reason to do that build next rather than
anything else in this file.

The enumeration meanwhile still moves `req_to_token[idx, :seqlen]`, one
row beyond what the allocator owns on the one config measured. That is a
known over-read of a stale row, not an under-read: it moves a row nobody
owns rather than missing a live one, which is the safe direction of the
two and is why shipping it while the measurement is owed is defensible.
Do not "fix" it on the strength of the SPEC=off reading alone -- that is
precisely the one-config re-cut v2 refused, and the errors are silent.

## 8. THE J.1 AUDIT, carried out and still open

v2 flagged "anything reading `running_batch`/`last_batch` from a per-slot
hook" as an audit candidate and deliberately did not chase it. It has now
been swept across `python/sglang/srt/` (delegated to the local Qwen
worker; the sweep is mechanical, the judgements below are mine).

**The class is real and it is not confined to #631: there are now THREE
confirmed instances**, and the third was in this feature's own policy.

### Fixed here

- **`gdn_flip_mover`** (second instance, silent): moved the conv/ssm state
  of one microbatch slot. Fixed as `resident_mamba_slots()`.
- **`maybe_arm_phase_policy`'s `running_bs`** (THIRD instance, and
  load-bearing): the policy hook runs inside `recv_requests()`, i.e. once
  per microbatch slot right after that slot's rebind. The PP->TP rule is
  `pending <= N AND running_bs > 0`, so a request decoding in slot 1
  while the hook fired for an empty slot 0 read `running_bs = 0` and the
  flip was **not armed** -- the decision depended on WHICH SLOT the hook
  sampled rather than on the load. Now counted from
  `harvest_resident_batches`. Note this is the arming side of exactly the
  shape defects L and the `_pp_microbatches_drained` bug had on the
  quiescence side.

### NOT fixed, named here, in severity order

None of these are #631's, and per scope they are named rather than
chased. All are confirmed by reading, none by measurement.

1. **`Scheduler.pause_generation` (scheduler.py ~7184-7251) is the worst
   one.** It reads `self.running_batch`/`self.last_batch` with **no
   `pp_size` branch at all** and, for a `"retract"` request, retracts
   `self.running_batch.reqs` wholesale. Its immediate sibling
   `_abort_request_now` shows the correct pattern right next to it
   (`if self.ps.pp_size == 1: [running_batch, last_batch] else:
   [*running_mbs, *self.mbs]`). Under PP a retract therefore appears to
   succeed while silently leaving every request resident in the other
   slots. Fix by copying the sibling's branch.
2. **`pool_stats_observer.active_pool_idxs`** feeds the invariant
   checker's `session_held_*` accounting and is reachable from the
   watchdog dump at ANY moment, not only at a quiescent boundary. A
   per-slot read there means a wrong `session_held`, i.e. a **false leak
   report** -- which is exactly the signal this feature spent three boots
   learning to trust.
3. **`load_inquirer.get_loads`** reports `len(running_batch.reqs)` as the
   instance's running-request count for `/v1/loads` and DP load
   balancing: under PP it publishes one slot's count as the rank's.
4. **`kv_pressure_runtime`'s `spill_fn`** offers only the current slot's
   batch to `try_spill`, so requests resident in other slots are never
   offered for spill under pressure.
5. **`kv_session_offload.release_finished_spilled_req`** and
   **`kv_session_spill_destination._release_parked_req`** clear
   `batch_is_full` on whichever slot is bound, not on the slot whose
   admission the freed KV actually unsticks.
6. **`is_fully_idle`**'s `last_batch` conjunct: `_pp_microbatches_drained`
   checks `running_mbs` and `mbs` across all slots but NOT `last_mbs`, so
   that one conjunct still reads a single slot.
7. **`build_gdn_flip_guard`** still carries the old pattern but **has no
   call site anywhere** -- it is the 5.3 placeholder the real mover
   superseded. Delete it rather than fix it.

**The general form, for the fourth time in this file:** under
`event_loop_pp`, `running_batch`/`last_batch` are per-slot ALIASES, and
every consumer that wants "this rank's requests" must harvest the slot
array instead. `phase_flip_resident_carry.harvest_resident_batches` is
the one authority; use it.

## 9. WHERE IT STANDS, and the two walls left

### What works, measured

| | |
|---|---|
| autoswitch, both directions, zero manual flips | YES -- 12 transitions in one acceptance run, 5 automatic PP->TP->PP cycles |
| a request decoding ACROSS a cutover | YES -- 197 consecutive correct integers, both legs committed on all 3 ranks |
| prefill in the PP layout | **4032-4553 tok/s** against **1525 tok/s** in TP (~3x) |
| decode in the TP layout with MTP | **113 tok/s**, accept len 3.7-4.0, graphs on, against **43 tok/s** without MTP (~2.6x) |
| agentic multi-turn (5 turns, growing context) | PASS -- real prefill in PP every turn, decode in TP every turn |
| aborts | 0 of 27, 0 of 30 |

Those are the two gains the phase flip exists to collect, and both are now
collected on this rig.

### Wall 1: SPEC=on + POLICY=auto runs out of memory under load

Measured 2026-08-09 05:09:41Z, agentic multi-turn load:

    torch.OutOfMemoryError: Tried to allocate 1024.00 MiB
    GPU 0: 883.69 MiB free      GPU 1: 650.38 MiB free
    -> SIGQUIT

**This also breaches the operator's standing corridor rule** (exactly
1024 MiB free per card, continuously). So the rank budget
`RANK_MIB=22200,14700,14700` is too aggressive for THIS configuration:
speculation's draft stack plus a flip that must carry a resident set
needs headroom the budget does not leave. `SPEC=on POLICY=manual` at the
same budget serves fine for hours, because nothing flips under load.

The remedy is a budget, not a redesign: lower `RANK_MIB` for the
auto+speculation configuration and re-measure the corridor at 100 ms
sampling through boot, flip and decode. Do not raise it back without that
sampling -- that is what the corridor rule is for.

### Wall 2: the carried request still has no draft state

Section 3 stands. It is currently mitigated rather than fixed: with
speculation armed, a `pp_to_tp` flip WAITS until nothing is resident
instead of crashing the draft graph runner. That mitigation is what makes
`SPEC=on` safe at all; it also means the flip does not help a request
that is already decoding, which is the case the draft-state build must
close.

### Therefore

**Production rests on `SPEC=on POLICY=manual`**, which is the
configuration that has served without incident throughout this work, and
is what the instance was left on. `POLICY=auto` is proven and fast at
`SPEC=off`; at `SPEC=on` it needs Wall 1 cleared first.

### One assumption of mine that metal falsified, kept as a warning

The carry originally REFUSED when one request was reachable through two
batches ("a request belongs to exactly one microbatch slot"). Under a
real agentic load that refusal fired and took the instance down
(04:55:43Z). The duplication is real; it is now resolved by filtering
before the merge. The check was right to be loud and wrong to be fatal --
and the assumption behind it was simply untrue.

---

# HANDOFF v4 — the SPEC=on + POLICY=auto attempt

## 1. Wall 1 (OOM) is CLEARED, by derivation rather than iteration

The failing allocation and the free memory at failure were read per rank
from the crash log, and the new budget derived as

    reduction = failed_alloc + corridor_target - free_at_failure

| rank | card | free at OOM | reduction | 22200,14700,14700 -> |
|---|---|---|---|---|
| 0 | 5090 | 883.69 MiB | 1164 | **21000** |
| 1 | 3080 | 650.38 MiB | 1398 | **13300** |
| 2 | 3080 | 490.38 MiB | 1558 | **13100** |

At that budget the OOM does not recur, and the corridor holds:

    #631 VRAM CORRIDOR -- 2650 samples over 268.0s at 100 ms, floor 1024
    gpu 0 (3080) MIN free 2642.4     breaches 0
    gpu 1 (5090) MIN free 4897.7     breaches 0
    gpu 2 (3080) MIN free 2714.4     breaches 0
    CORRIDOR HELD: True

**But the budget is now too LOOSE, and that is also a corridor violation
in the operator's sense**: the rule is free NEAR 1024, not far above it.
Headroom above the floor is +1618 / +3874 / +1690 MiB, i.e. ~7 GiB of
this rig is being left unused. The derivation was deliberately
conservative (it subtracted the whole failed allocation AND the whole
corridor); the honest next step is to give most of that back once a run
completes without faulting, and re-sample. **Do not raise it without the
100 ms sampling** -- `scripts/route_a_631_corridor.py` is the harness,
and it reports the per-card time-series minimum from the NVML FREE field
(never total-used: a ~424-518 MiB carve-out is invisible to that
subtraction).

## 2. Wall 2 answered BY MEASUREMENT: waiting is NOT acceptable

The coordinator asked whether the "speculating flips WAIT" mitigation is
good enough or whether the draft-state carry must be built. Measured
2026-08-09 05:21Z, SPEC=on POLICY=auto under agentic load:

    05:21:01  armed (pp_to_tp) but NOT QUIESCENT: 1 resident request(s)
              would enter a SPECULATING TP phase with no draft state
    05:21:08  (same, all three ranks)
    05:21:16  FLIP ABANDONED: pp_to_tp was armed for 30.0s without the
              group reaching a quiescent boundary (deadline 30s)
    05:21:16  Scheduler hit an exception
              -> CUDA error: an illegal memory access was encountered

**Two findings, and the second is the serious one.**

**(a) The mitigation cannot work under sustained decode.** The guard's
readiness condition is "nothing resident", and under a continuous agentic
load something is always resident. So the flip never becomes ready, parks
for the full deadline, and abandons -- every time. Flip cadence under
sustained decode with speculation is therefore ZERO, and the feature
delivers nothing in exactly the configuration production runs. **The
draft-state carry is required, not optional.**

**(b) ABANDONING A FLIP IS NOT SAFE UNDER LOAD, and the corpse table says
it should be.** The design's standing claim for the pre-entry abandon
path is "NOTHING was entered and nothing was moved". Here the abandonment
is immediately followed by a device-side illegal memory access. The
barlink BAR1 status poll in the traceback is where it SURFACES, not where
it originates -- a sticky CUDA fault reports at the next synchronising
call. This is a new defect, it is NOT the OOM, and it is NOT the draft
state; it is the abandon path under load with speculation armed.

Specimen: `/spinning/evidence-631/oom_and_abandon_20260809T0521Z/`
(`serving.log` = the faulting boot, `oom_prior_boot.log` = the OOM boot
the budget derivation came from).

## 3. What the next session should do, in order

1. **Diagnose (b) first.** It is a memory-safety fault on the abandon
   path and it can corrupt rather than merely stop. Reproduce with
   SPEC=on POLICY=auto plus sustained decode -- it reproduced within one
   load run -- and capture with the wedge harness. Until it is understood,
   POLICY=auto must not run with speculation.
2. **Then build the draft-state carry** (v3 section 3 lists three options
   and the shared hazard). Measurement now says the alternative -- waiting
   -- is worth nothing under production load.
3. Only then re-run the acceptance at SPEC=on POLICY=auto, tighten the
   budget back toward the corridor, and run the A-vs-A gate vs
   `9a929352c9` with a same-boot floor.

## 4. Production

Left on **`SPEC=on POLICY=manual`**, verified serving (`health` 200,
`health_generate` 200, a real chat completion returned). That is the
configuration that has served without incident throughout, and it is the
only one currently proven with speculation.

## 5. The draft-state carry is now PROVEN necessary, both alternatives measured

Two mitigations were tried against the "a carried request has no draft
state" wall. Both are measured, both are worthless, and between them they
close the question the coordinator asked.

**(a) ARM AND WAIT** -- readiness holds the flip until nothing is
resident. Under sustained agentic decode something always is, so
(2026-08-09 05:21Z) the flip parked the full 30 s deadline, abandoned, and
the ABANDON PATH FAULTED: `CUDA error: an illegal memory access was
encountered`, immediately after `FLIP ABANDONED`. Cadence zero, plus a
memory-safety fault.

**(b) DO NOT ARM** -- the policy declines instead of arming a flip that
cannot become ready (decided rank-locally, which is safe because only the
request-origin rank evaluates the policy and the arm is broadcast from
it; a refusal inside `PhaseFlipRuntime.arm` would risk diverging epochs,
corpse H). Measured 2026-08-09 05:33Z, agentic multi-turn:

    turn   fresh  pfx phase  prefill tok/s  dec phase  decode tok/s
    1    53054    pp                3947.4  pp                 16.7
    2       28    pp                 136.9  pp                 16.9
    3       30    pp                 142.6  pp                 17.0
    4       25    pp                 128.3  pp                 16.9
    automatic flips: 0        decode ran in TP: False

**No crash -- the instance stayed up for the whole run, which is why (b)
is kept as the safe default -- but the instance is now PINNED IN PP.**
Decode runs at **16.8 tok/s against the 113 tok/s it does in TP with
MTP**, because the PP layout carries neither the decode graphs nor the
draft worker. Prefill is right (3947 tok/s) and decode is 6.7x wrong.

So: waiting costs a fault, declining costs the decode layout, and there
is no third mitigation. **Build the draft-state carry** (section 3 lists
the options and the shared hazard; heed #108/#635 -- the draft KV is its
own coordinate system and must not be mixed with the target pool's).

Until then the useful configurations are:
* `SPEC=off POLICY=auto` -- switches, pays, 0 aborts (the 3x/2.6x ledger);
* `SPEC=on POLICY=manual` -- speculating decode at 113 tok/s, no automatic
  layout changes.

---
---

# HANDOFF v5 — written 2026-08-09 by successor #5

**Supersedes v4 §2(b) and §3.1.** The abandon-path "illegal memory access"
is diagnosed, the corrupting kernel is named, and the corruption hole is
closed and shipped (`118cdf2cbb`). Read the corpse table entries P and Q
in `phase_flip_presence.py`'s neighbours (they are recorded in
`test_phase_policy.py` and `scheduler_pp_mixin.py` respectively).

## 1. The IMA is NOT the flip moving KV, and the log order proves it

v4 wrote "abandonment is immediately followed by a device-side illegal
memory access". The order in the specimen is the other way round:

    05:21:16  FLIP ABANDONED (30 s deadline), all three ranks
    05:21:16  PP2  ValueError from fused_recurrent_gated_delta_rule_packed_decode
                   `ssm_state_indices` must have shape [B]
                   (got (1,); expected (2048,))
    05:21:16  SIGQUIT
    05:21:17  ONLY NOW the IMA, x156, ALL inside barlink's BAR1 status poll
    05:21:17  PP0/PP1 die of gloo "Connection closed by peer" -- teardown

There is **no OOM in that boot at all** (0 hits for OutOfMemory / "Tried to
allocate"). The first fault is a shape mismatch: a decode batch of ONE
request carrying **2048** rows of hidden state, and 2048 is exactly that
boot's `chunked_prefill_size`.

**THE KERNEL THAT ACTUALLY WROTE.** One call *before* the guarded
`packed_decode`, `gdn_backend.forward_decode` hands the same mismatched
pair to `causal_conv1d_update`. It launches one program per row of `x`,
each reading `conv_state_indices[row]`: with 2048 rows and a 1-element
index tensor, 2047 reads run off the end and the garbage becomes a
conv-state line number — an out-of-bounds READ and then an out-of-bounds
**WRITE** into another request's state. The assert for exactly this
existed, behind `validate_data`, which defaults to False.

**Fixed unconditionally on both paths.** Only the SHORT direction is
refused; a longer index tensor is harmless and refusing it would break
correct callers. Pinned by a mutation-proven can-fail
(`test/registered/unit/layers/test_causal_conv1d_bounds_631.py`).

**Standing trap, it cost the first investigation an hour:** the barlink
BAR1 poll is where a sticky CUDA fault SURFACES, never where it happens.

## 2. Defect Q: the armed window has no pass clock (measured, real, and
   NOT the cause)

`mb_id` is a purely rank-local loop index; there is no shared pass
counter. What holds the PP stages in phase is the BLOCKING chain receive.
The armed intake rule returns `[]` immediately on every rank, so nothing
blocks and every rank free-runs for the whole armed window.

Measured on metal, one 30 s armed window that abandoned under load:

    rank 0: 266486 slot iterations   rank 1: 204194   rank 2: 226334
    SPREAD 62292        (a second abandon: 62241, same shape)

Control, same boot pattern: **51 committed flips, every pass-clock window
SPREAD 0** — a commit re-forms loop state via `init_pp_loop_state`, an
abandon does not.

**SAY THIS PLAINLY, because it is the honest limit of this session:** the
drift is real and reproducible, and it is **NOT shown to cause the shape
mismatch**. Two abandons under load — one with a 40k-token prefill fired
mid-window, reproducing the specimen's backlog — produced those spreads
and **no fault**, and all three ranks disarmed at `mb_id=0`, so the
microbatch phase survived. **The origin of the 2048-vs-1 pairing is still
open.** The corruption it caused can no longer happen silently; the
mismatch itself can.

Instrument: `CHAN_PASS` gauge + the `PASS-CLOCK` log line, reported from
one rank for the whole group. Reader: `scripts/route_a_631_pass_clock_report.sh`.

## 3. Corpse P, so it is not re-derived

"The armed rank drops requests off the chain" is the obvious reading of
the armed branch and is wrong twice over: `_pull_raw_reqs` returns `[]` on
every rank while armed (rank 0 leaves its requests in the zmq socket,
which IS the buffer), and `PpChainReceiver.recv()` drains its inbox before
touching the wire. Nothing is lost. Full record in `test_phase_policy.py`.

## 4. Where to go next, in order

1. **Find the origin of the batch/metadata mismatch.** The guard now turns
   it into a loud ValueError instead of a silent write, so the next
   occurrence is safe to catch and names itself. Re-run the SPEC=on wait
   variant under sustained agentic load until it fires. Do NOT assume
   defect Q is the cause — it is measured and it did not reproduce it.
2. **The draft-state carry** (v3 §3, still the blocker for SPEC=on+auto).
   v4's measurement stands: waiting is worth nothing under sustained
   decode.
3. Then acceptance at SPEC=on POLICY=auto, the budget give-back toward the
   1024 MiB corridor, and the A-vs-A gate vs `9a929352c9`.

## 5. Tests and state

- `scripts/run_631_flip_family.sh` -> **425 passed** (was 417).
- Metal, this tree, SPEC=on POLICY=manual: boot healthy, PP decode,
  `pp_to_tp` cutover committed on all three ranks, TP decode with MTP at
  **accept len 3.27**, correct answer, **0 guard false positives, 0 faults**.
- Evidence: `/spinning/evidence-631/abandon_q_20260809T0556Z/`
  (`control_specoff_commits.log` = 51 commits, all SPREAD 0;
  `abandon_specon_drift.log` = the two abandons with SPREAD 62292/62241).

## 6. ROOT CAUSE FOUND (added same day, after v5 was written)

v5 left "the origin of the 2048-vs-1 pairing is STILL OPEN". It is now
closed, reproduced and evidenced. Specimen
`/spinning/evidence-631/pp_proxy_mispair_20260809T0626Z`.

**Reproducer**: park deadline 5 s (`SGLANG_PHASE_FLIP_PARK_DEADLINE_S=5`),
a manual `pp_to_tp` arm every 7 s, and the mixed acceptance load. At
SPEC=on a resident decode makes every `pp_to_tp` park and abandon, so this
buys ~12 abandons a minute instead of 2.

    06:26:34 PP0] FLIP ABANDONED
    06:26:34 PP2] FLIP ABANDONED
    06:26:34 PP1] FLIP ABANDONED          <- LAST
    06:26:35 PP1] #631 PP proxy/batch mismatch: received hidden_states
                  with 1 row(s) for a batch of 24 token(s) (bs=1)

**The mechanism.** The abandon is RANK-LOCAL — every rank times out on its
own clock, so the ranks disarm at different instants. A rank that has
already disarmed resumes launching and sends its proxy hidden states. Its
downstream is still armed and still withholding, so that rank's
`cur_batch` is None — and the proxy recv is guarded by **this rank's**
`cur_batch`, never by whether the upstream sent. The message is not taken.
It strands in `_pp_tensor_dict_inbox`, and since the pairing is **purely
positional** (`PPProxyTensors` has no mb_id, no sequence number, no
length; the receive demultiplexes only on `__msg_type__`), every later
proxy recv on that rank is off by one, silently, for the rest of the
loop's life.

**The rank that disarms LAST is the rank that strands.** PP1 above is both.

**Why a commit never showed it**: the cutover re-enters `event_loop_pp`
and `init_pp_loop_state` resets every buffer, `_pp_tensor_dict_inbox`
included. The three abandon functions in `phase_flip_runtime` clear
`_pending`, `_armed_at`, `_last_hold_reason` and touch no channel, no
buffer and no loop state. That is the whole asymmetry — and it is
mechanical, NOT defect Q's pass-count drift, which remains open on its own
merits (an armed rank running 62k iterations behind a peer is unhealthy
regardless).

**It also corrects v5**: the fault needs no 2048-row chunk. Here the
direction is the opposite — a 1-row decode hidden state meeting a 24-token
extend batch. Any two microbatches of different width will do.

**THE FIX, named but NOT built** (next step): a rank must not resume
launching while a peer is still armed. It belongs at the RESUME, not at
the abandon — the abandon cannot be made instantaneous across ranks, but
the withhold can be held until the group has disarmed, reusing the
presence markers that already exist in /dev/shm. **A blind reset of loop
state on abandon is NOT the fix**: unlike a commit, an abandon has no
quiescence guarantee, so it would discard in-flight microbatches.

**Instrument shipped** (`36f9c8be90`): one shape check at
`model_runner.forward`, the single funnel every PP stage's forward passes
through. The model asserted the proxy's presence but never its shape.
0 false positives across a full boot with cuda-graph capture; suite 425.

## 7. THE RESUME GATE: built, unit-proven, METAL-FALSIFIED, parked

Branch `feat/route-a-631-resume-gate` (do NOT merge). The feature branch
stays at `2e7131de7d`, which is verified healthy and is what serves.

**What it does.** Holds the withhold past the disarm until every rank has
published a disarm marker (/dev/shm, presence-gate discipline), closing
the §6 skew. Built deliberately as a PREDICATE re-evaluated per pass, not
a rendezvous -- a gated rank keeps cycling and servicing its channels --
and bounded at 1 s with a loud expiry so a dead peer cannot hold the
survivors out of service.

**Unit evidence**: suite **433 passed** (425 + 8 gate tests); the falsifier
is mutation-proven (forcing the gate open fails 5 tests); a boot came up
clean with 0 false positives.

**METAL FALSIFIED IT.** Under the reproducer the instance WEDGED at
06:47:37Z. py-spy, specimen
`/spinning/evidence-631/gate_wedge_20260809T0647Z`:

    rank 0  _pp_commit_comm_work <- _pp_forward_and_process_input_requests
            (the BLOCKING top-of-pass chain flush, unarmed branch)
    rank 1  _pp_recv_proxy_tensors -> _pp_recv_typed_dict
    rank 2  _pp_recv_proxy_tensors -> _pp_recv_typed_dict

That is the corpse table's founding deadlock class verbatim: rank 0 blocked
on the request chain while its peers sit in the HIDDEN-STATES exchange.

**THE LESSON, and it is a general one about this feature.** The gate blocks
on nothing, and I still recreated the deadlock class. **The design law is
about the SKEW, not only about who calls wait().** Any mechanism that
changes WHEN a rank launches relative to its peers can drive adjacent ranks
into different blocking channels, whether or not the mechanism itself
waits. Withholding is therefore the wrong lever.

The same run also produced the first PASS-CLOCK line where the microbatch
phase actually diverged (previously always `mb_id=0`):

    rank 1 ran 5932 slot iterations, rank 2 ran 10957,
    armed at mb_id=0, DISARMED AT mb_id=2

**NAMED DIRECTION FOR THE NEXT ATTEMPT**: fix the RECEIVE side, not the
launch side. Consume a proxy when the UPSTREAM says it sent one -- the
`CHAN_DICT` counters already publish exactly that -- instead of when this
rank happens to have a batch. That removes the strand without moving any
rank's launch timing, so it cannot manufacture skew. It needs a decision
about what a rank does with a proxy it has no batch for (discarding loses a
microbatch), which is the real design question and is not yet answered.

**Defect Q is now PARKED, not closed** -- and the run above is the first
evidence that its drift can reach the microbatch phase.

## 8. THE RECEIVE-SIDE FIX: variant B chosen, A rejected, NOT YET BUILT

Both shapes were weighed against the corpse table rather than by
preference. **B is chosen.** Neither is built yet -- this section is the
design decision, not a claim of work done.

### A (rejected): consume on the upstream's CHAN_DICT counter, buffer
proxies this rank has no batch for

**Rejection reason.** It rebuilds sequencing discipline IMPLICITLY: the
buffer must be a strict FIFO whose head is consumed exactly once per
upstream send, an invariant maintained at two ends that can disagree --
which is the precise shape of the bug being fixed, re-created one layer up.
It also decides WHEN a buffered proxy is released ("once this rank's own
batch materializes"), i.e. it moves per-rank timing, and §7 is what that
costs: the refined law says any mechanism that changes when a rank
proceeds relative to its peers can drive adjacent ranks into different
blocking channels, whether or not the mechanism itself waits. It is
additionally adjacent to the bounded-recv corpse (unconsumed messages
accumulating while an upstream keeps sending).

### B (chosen): make the pairing NON-POSITIONAL -- stamp the proxy

Stamp each proxy with `mb_id` plus a monotone per-channel seqno and the
row count. The consumer matches on the stamp; a leftover from an abandoned
window matches nothing and is dropped LOUDLY with its identity logged.

**Why it satisfies the refined law**: no launch timing moves and no new
synchronization point exists. A strand becomes harmless BY CONSTRUCTION
rather than fenced, so the off-by-one class dies permanently.

**It is also the corpse table's own recurring lesson.** "The gate's
guarantee is per-round; its evidence is per-epoch" and "any quantity
derived from `ps` is phase-scoped now" are the same error: an identity
INFERRED from context instead of carried. Positional pairing is exactly
that -- evidence whose scope does not cover its guarantee.

### The implementation vehicle, already located

`PPProxyTensors` (forward_batch_info.py:1572) is a plain
`tensors: Dict[str, torch.Tensor]`, and the send site transmits that dict
directly (`_pp_send_dict_to_next_stage(result.pp_hidden_states_proxy_tensors
.tensors, ...)`). **So the stamp can ride as an extra tensor entry in the
dict -- no header to invent and no transport change.** Construction sites
are few (scheduler_pp_mixin.py:1305, 1733, 1903, 1954).

**THE ONE RISK TO CHECK FIRST, before writing anything**: any consumer that
iterates the dict blindly rather than reading known keys (a model forward
doing `for k, v in tensors.items()`, a cuda-graph buffer copy, or the
`__getitem__` slice path at :1587 which maps over ALL entries and would
try to slice a scalar stamp). A grep of the model/executor paths did not
show proxy-dict iteration, but that check is not complete and must be
finished before the stamp is added. If a blind consumer exists, keep the
stamp in a sibling field rather than inside `tensors`.

`self.mb_metadata: List[Optional[PPBatchMetadata]]` is RANK-LOCAL and never
crosses the wire, so it is not the vehicle -- it cannot carry the identity.

### Acceptance for whichever lands

The on-demand reproducer: 5 s park deadline, repeated `pp_to_tp` arms with
resident decode (so every arm abandons), mixed acceptance load. Require N
abandon cycles with no strand, ranks resuming aligned, and the
`model_runner.forward` shape check never firing. **Keep that check as a
standing tripwire regardless** -- it is what turned this class from silent
corruption into a named defect.
