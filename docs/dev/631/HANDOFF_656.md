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

---
---

# HANDOFF v6 — written 2026-08-09 by successor #6

**The stamp's DETECTION is metal-proven. Its DISPOSAL is metal-falsified.**
Variant B as shipped in `d733266b5d` wedges the instance. The two halves
must be judged separately, because one of them is right.

## 1. What was proven first, off metal

Three unit pins the variant B commit named as required and did not have
(`test_pp_proxy_stamp_631.py`, registered in the family script; suite
**439 passed**, was 425). Writing them found a real defect in the shipped
fix: **the stamp was left in the dict and travelled on into model
compute.** The commit had argued this was safe by the `__msg_type__`
precedent, but that precedent shows a non-tensor entry SURVIVES THE WIRE
-- it does not show one is safe to COMPUTE ON.
`PPProxyTensors.__getitem__`'s slice path maps `v[key]` over every entry,
and a tuple slices to a shorter tuple instead of raising, so the failure
mode would have been silent nonsense. The stamp is now popped at the
delivery boundary (`5474f60622`).

A full consumer sweep settled the transport question that was left open as
"the one risk to check first": `_split_tensor_dict`
(`distributed/parallel_state.py:227-250`) has exactly ONE branch point,
`isinstance(value, torch.Tensor)`, and everything else is carried verbatim
through a generic `pickle.dumps`/`pickle.loads` of the metadata list.
A tuple is therefore transported identically to the `__msg_type__` string
-- **not** str-specific handling that happens to work. Every compute-path
consumer reads known keys (`hidden_states`, `residual`, `topk_indices`)
or iterates a fixed, locally-allocated buffer key set
(`runner_utils/buffers.py:134-144`), never the wire dict's own keys. The
single blind consumer of a received proxy is the opt-in
`debug_utils/dumper.py:1775`, which is already broken for `__msg_type__`
today; the strip closes it for the stamp.

## 2. THE DETECTION WORKS. This is the metal line, and it is exact

Boot of `5474f60622`, SPEC=on POLICY=manual, park deadline 5 s, the
abandon-under-load reproducer (now a script, see §5):

    07:19:23 PP1] PHASE-FLIP PROXY LEFTOVER DROPPED: stamp mb_id=2
                  seq=2811 rows=1 arrived while this rank is on mb_id=1

**`rows=1` is the specimen's own signature** -- a one-row decode hidden
state arriving for a different microbatch, which is precisely the pair
that used to reach `causal_conv1d_update`. PP1 is again the rank that
strands. The message was named and refused instead of computed on:
**0 proxy/batch mispairs**, 0 conv1d guard hits, 0 illegal memory
accesses, across 3 abandons.

**The same-boot CONTROL is clean too**: same load, zero arms, therefore
zero abandons -> **0 drops**, 14 requests, 2240 tokens, 0 aborts, health
200. So the stamp does not fire on healthy traffic; it fires on exactly
the condition it was built for.

## 3. THE DISPOSAL WEDGES THE INSTANCE — corpse R

Six seconds after the drop the instance stopped. Specimen
`/spinning/evidence-631/wedge_20260809T072021Z_stamp_drop_wedge_20260809T0719Z`
(all three stacks, presence markers, timeline).

    PP1  _pp_recv_proxy_tensors (scheduler_pp_mixin.py:1790)  <- the
         SECOND recv, the one the drain loop makes after dropping
    PP2  _pp_recv_proxy_tensors (scheduler_pp_mixin.py:1790)
    PP0  _pp_recv_dict_from_prev_stage (scheduler_pp_mixin.py:1987)
         <- the OUTPUT channel, waiting on PP2

A closed cycle: PP0 waits for PP2's output, PP2 waits for PP1's proxy,
PP1 waits for a proxy from PP0 that **was never sent and never will be**.

**THE MECHANISM, and it is the bounded-recv corpse read backwards.** That
corpse said: complete a pass without consuming and unmatched SENDS pile
up until the senders block. Its dual is what happened here: consume TWICE
in one pass and the surplus RECV blocks for ever, because the wire owes
exactly one message per pass and no more. Dropping a message and then
taking "the next one" silently assumes a next one exists. On this wire it
does not.

**So the drain loop is the defect, not the stamp.** `for _ in range(8)`
looked like a bound on a spin; it is actually a licence to make up to
eight blocking calls where the contract permits one.

## 4. THE NAMED DIRECTION: void the SLOT, never the message

The invariant the receive side may not break is **exactly one message
consumed per pass**. The stamp does not need to break it to do its job:
it only has to change what the rank DOES with the message it was always
going to take.

    take one message per pass, as before;
    if the stamp names this slot   -> compute on it, as before;
    if it does not                 -> this slot is VOID: do not compute,
                                      and pass the void downstream as
                                      this pass's proxy send.

No extra recv, no missing send, 1:1 preserved end to end, and no rank's
launch timing moves -- which is what the refined law demands. The void
propagates one hop per pass exactly like real hidden states do, so the
pipeline drains it instead of deadlocking on it.

Open sub-questions, honestly: what the void does to the requests in that
microbatch (retract vs re-queue), and whether `_pp_make_skip_output_result`
-- which already exists for the output channel's skip -- is the right
shape to reuse for the last rank's return leg.

## 5. THE SIBLING WIRE HAS THE SAME HOLE, still unstamped

`_pp_recv_dict_from_prev_stage` (the last rank's `next_token_ids` return
to rank 0) is guarded at `scheduler_pp_mixin.py:1979` by
`mbs[next_mb_id] is None` -- i.e. by **this rank's own batch**, which is
the identical guard shape that produced the proxy strand. It carries no
stamp. Rank 0 is the rank that arms first and disarms last in these runs,
so it is the natural victim. A mispair there is worse than a shape error:
`next_token_ids` for the wrong microbatch is plausible-looking wrong
output. Stamp it with the same mechanism once the disposal is settled.

## 6. Reproducer and control are now a script, not a recipe

`scripts/route_a_631_proxy_strand_repro.py`. `--cycles N` runs N
arm/abandon cycles against sustained decode; `--cycles 0` is the CONTROL
(same load, no arms). It prints a PASS/FAIL verdict over: abandons
occurred, zero mispairs, no drain gave up, no IMA, server healthy, zero
aborts. Both runs above came from it. Results:
`/tmp/route-a-631/control_stamp.json` (PASS),
`/tmp/route-a-631/repro_stamp.log` (the wedged run).

**Note the verdict design**: leftover drops are reported but are NOT a
pass condition in either direction. A drop is the fix working; zero drops
in a run with abandons means the strand did not occur on that schedule.
The pass condition is the absence of the MISPAIR.

---

# HANDOFF v6, part 2 — corpse S, and where this shift actually got to

Written the same day, after v6 above. Read v6 first: it is the corpse-R
half. **The tree HEAD is `d82d88d1a2` and it SERVES** (health 200, correct
completion, control PASS at 07:41:20Z).

## 1. Ledger of this shift

| | verdict | evidence |
|---|---|---|
| the stamp's DETECTION | **PROVEN on metal** | 07:19:23Z, PP1, `stamp mb_id=2 seq=2811 rows=1` -- the mispair specimen's own signature, named and refused |
| the stamp reaching model compute | **found and fixed** | `test_the_stamp_is_stripped_before_the_model_sees_it` failed against `d733266b5d` |
| tuple-over-the-wire safety | **settled, not assumed** | `_split_tensor_dict` has ONE branch, `isinstance(value, torch.Tensor)`; the rest is generic pickle |
| DISPOSAL: drop and retry | **corpse R** | wedged in 6 s; surplus recv against a one-per-pass debt |
| PREVENTION: armed drain | **corpse S** | ate an OUTPUT message in one arm; PP1 then waited for it for ever |
| the box | **serving** | `d82d88d1a2`, SPEC=on POLICY=manual, control PASS |

Two designs died on metal in one shift, and both died of the same family
of error: **reasoning about this wire from its topology instead of from
the code that reads it.**

## 2. THE FACT THAT BOTH CORPSES TURN ON, stated once and plainly

**The upstream wire is MULTIPLEXED and it owes exactly one message per
pass.** Two consequences, and each killed a design:

* **You may not take more than you are owed.** Corpse R dropped a message
  and took another. The wire had nothing more; PP1 and PP2 blocked in the
  proxy recv, PP0 in the output recv, a closed cycle.
* **You may not discard what you cannot identify.** Corpse S discarded by
  position on the wire rather than by kind. The proxy forward and the
  output return share that wire and are separated only by `__msg_type__`
  AFTER coming off it. A proxy for a pass an armed rank never ran is void;
  an OUTPUT belongs to work launched BEFORE the arm and is still owed.

`_pp_recv_typed_dict` and its inbox exist because of exactly this. Both
designs treated the demultiplexer as an obstacle to route around. It is
the statement of the fact they got wrong.

## 3. What the next attempt should and should not do

**Do not** re-enable `pp_flip_drain_tensor_dicts` as written. It is kept,
uncalled, with the specimen inside it.

**A correct drain must demultiplex first**: stash `output` in the inbox
where its consumer already looks, and only then decide about `proxy`. That
decision is still not free -- a microbatch launched before the arm is in
flight and its proxy is owed too, so "an armed rank runs no passes" does
not make every proxy void either. The honest form of the remaining
question is: **which proxies is an armed rank still owed?** Neither corpse
answered it, and neither did I.

**The detection is worth keeping regardless.** It converted this class
from silent corruption into a named identity twice in one shift, and it
costs nothing on the healthy path (three control runs, zero drops, decode
13.89 vs 14.06 tok/s across builds).

**The residual limit stands**: the match is on `mb_id` alone, cyclic
modulo `pp_loop_size` = 3 here (`pp_async_batch_depth=0`, confirmed in the
boot log), so a leftover a whole cycle stale is accepted. The seqno is
stamped and unused, and the reason it cannot simply be used is worth
recording: upstream's seqno and this rank's consumed count stay
CONSECUTIVE across a strand, so equality holds and the stale message is
accepted anyway. Distinguishing them needs a shared pass clock, which is
defect Q, which is open.

## 4. Untouched by this shift, in the order the program asked for them

1. **Draft-state carry** -- not started. v4's measurement stands: waiting
   is worth nothing under sustained decode, declining pins the instance in
   PP at 16.8 vs 113 tok/s.
2. **Full acceptance** (auto + graphs + spec + max KV, corridor near
   1024 MiB) -- not started. The budget is still ~7 GiB loose: measured
   free at idle on this boot was 5503/5792/4171 MiB against a 1024 floor.
3. **A-vs-A gate vs `9a929352c9`** -- not started.
4. **The sibling OUTPUT wire is still unstamped** and has the identical
   guard shape (`mbs[next_mb_id]` at :1940). Corpse S makes it more
   interesting, not less: that wire is now known to carry work an armed
   rank still owes.

## 5. Tools this shift leaves behind

* `scripts/route_a_631_proxy_strand_repro.py` -- the reproducer AND its
  control, with a PASS/FAIL verdict. `--cycles N` for arm/abandon cycles,
  `--cycles 0` for the control. It found both corpses.
* `test/registered/unit/managers/test_pp_proxy_stamp_631.py` -- 17 pins,
  registered in the family script (442 total). The refusal pin asserts
  `recv_calls == 1`, so corpse R cannot be reintroduced silently.
* Specimens: `/spinning/evidence-631/stamp_drop_wedge_20260809T0719Z`
  (corpse R) and
  `/spinning/evidence-631/wedge_20260809T073423Z_armed_drain_ate_output_20260809T0733Z`
  (corpse S).

## 6. Delegation, for the record

One subagent, for the mechanical consumer/transport sweep in §1 of v6.
Everything else was interactive because it was metal work on a single
shared instance, which does not parallelise: the three cards are one
resource and every design question this shift asked was answered by a boot
rather than by reading.

---

# HANDOFF v7 — written 2026-08-09 by successor #7

**The predecessor's open question was the wrong question, and answering the
right one closed defect Q twice.** Read v6 part 2 first for the corpses;
this section replaces its framing.

## 1. THE RE-DIAGNOSIS, and how it was reached

v6 asked: *which proxies is an armed rank still owed?* It assumed a
message was stranded on the wire. **Nothing is stranded.** A parked rank
neither sends nor receives a proxy (the park makes
`get_next_batch_to_run` return `batch_to_run=None`, so `cur_batch` is
falsy and both the send and the receive are skipped), so the
one-message-per-pass contract holds through an armed window and the
counters stay balanced.

The answer was already in the log, in an instrument my predecessor built
and then read as a curiosity. The corpse-R boot, 07:19:23Z:

    rank 0  44477 armed iterations, armed at mb_id=2, DISARMED AT 2
    rank 1  33690 armed iterations, armed at mb_id=2, DISARMED AT 0
    rank 2  38069 armed iterations, armed at mb_id=2, DISARMED AT 2

**Every rank arms on the same slot and leaves on a different one.** From
that instant stage k computes its slot-s batch while stage k+1 applies the
result to its slot-s' batch -- permanently, not once. The stamp fires on
the first such message, which reads exactly like one stale leftover. That
misreading is what killed both disposals: corpse R dropped it and made a
second blocking receive against a debt of one (wedge in 6 s), and corpse S
drained a wire with no surplus on it and so could only eat something that
WAS owed.

**Why the index drifts.** In steady state the pass loop is paced by the
request chain -- one blocking receive per slot iteration -- so rank k's
i-th iteration is rank k-1's i-th and the indices cannot diverge. An armed
rank admits nothing and launches nothing, so its iterations become
free-running spin (~8 kHz, tens of thousands per window) and each rank
abandons on its own park deadline wherever it happened to stop.

## 2. FIX ONE: hold the slot (`_pp_flip_hold_slot`), commit `0600d8ae34`

Once the armed window has run the pipeline dry (every `mbs` slot `None`,
no half-written chunk), the loop HOLDS its slot instead of walking it.
Same spin, same channel servicing, same gate polling, one index. The hold
is reached on the same slot on every rank because the arm rides the 1:1
ordered request chain and every rank then needs exactly `pp_loop_size`
parked iterations to null every slot. No launch timing moves and no rank
waits on a peer, so the refined design law is untouched. With the flip
disabled the `while` loop is the previous `for` loop line for line.

`CHAN_SLOT` makes the falling edge return a VERDICT rather than a number:
the group's resume slots, AGREED or DIVERGED, on one line from one rank.

**METAL, 12 arm/abandon cycles under sustained decode with SPEC on:** 36
abandonments, 36 verdicts, **all `RESUME SLOTS [2, 2, 2] -- AGREED`**,
with the spin spread as large as ever (8110-8694 iterations). 0 refusals,
0 mispairs, 0 illegal accesses, 0 aborts, health 200. Control (same load,
no arms): 0 drops, decode 13.89 tok/s against 14.06 on the previous build.
Evidence `/spinning/evidence-631/slot_hold_Q_CLOSED_20260809T0807Z`.

## 3. FIX TWO: the cadence counter — the SIBLING, and the old TP wedge

The reproducer only ever abandons, so it never exercises what is past the
cutover. A committed flip on an idle server then killed the box:

    barlink collective 'phase_flip.consensus' made no progress for 120s

PP0 in `event_loop_normal -> get_next_batch_to_run ->
_phase_flip_on_round`, inside the reduction; peers never arrived, then
SIGQUIT. `PhaseFlipRuntime._round` is a RANK-LOCAL counter and
`_round % _interval` gates ENTRY TO A BLOCKING COLLECTIVE. The armed
window called `on_round` 37371 / 28677 / 32344 times in ONE 5 s window, so
the ranks emerged incongruent mod 8, their periodic entries never coincide
again, and the first to enter waits for peers that are waiting on the
broadcast it owes them.

**This is pre-existing** -- the counter behaved this way before the slot
hold -- and is very likely the old `attempt12-FLIP-OK-tp-serving-wedge`.

**Fix:** count only the rounds the cadence actually gates, i.e. calls with
`require_armed_and_parked=False` (the lockstep TP path). **The first cut
was wrong** ("count unarmed rounds only") and the existing KV-move
correctness pin caught it: an armed rank on the periodic path must still
reach the reduction, or a flip armed in the TP phase can never commit.
Keep that pin honest -- it is the only thing that separates the two rules.

## 4. THE GENERAL FORM, now confirmed twice

**A QUANTITY IS ONLY IN LOCKSTEP WHILE SOMETHING KEEPS IT THERE.** Neither
`mb_id` nor `_round` was ever synchronised by design; both were
synchronised by traffic, as a side effect. The armed window removes the
traffic, the invariant evaporates silently, and every rank-local check
still passes. Ask of any cross-rank index in this feature: what,
mechanically, holds it equal, and is that thing still running in the state
I am designing? Candidates not yet audited: anything else keyed off a
per-iteration counter across ranks.

## 5. What is still open, in the order the program asks for it

1. **The round trip under load.** `scripts/route_a_631_roundtrip_probe.sh`
   runs abandons -> commit -> serve in TP -> a soak longer than the 120 s
   collective bound -> the return leg. Run it and believe only its
   summary.
2. **Draft-state carry** -- untouched. Still the wall for SPEC=on under
   load: the quiescence predicate refuses to carry a resident request into
   a speculating TP phase, so every armed window under load abandons by
   design.
3. **Full acceptance** (auto + graphs + spec + max KV, corridor near
   1024 MiB). Budget still ~7 GiB loose.
4. **A-vs-A vs `9a929352c9`**, then ship.
5. **The OUTPUT wire is still unstamped.** With the slot fixed this is now
   defence in depth rather than a live defect, but the guard shape is the
   same one that produced the proxy strand.
6. `pp_flip_drain_tensor_dicts` is still present and uncalled. It is
   corpse S, and the re-diagnosis makes it doubly wrong: it disposes of
   something that was never there. Delete it or keep it only as a specimen.

## 6. THE ROUND TRIP IS PROVEN — added after v7 was written

`scripts/route_a_631_roundtrip_probe.sh`, 2026-08-09 08:24:14-08:28:42Z,
boot SPEC=on POLICY=manual, park deadline 5 s. Evidence
`/spinning/evidence-631/roundtrip_PASS_20260809T0828Z`.

    STEP 1  6 arm/abandon cycles under load        repro VERDICT PASS
    STEP 2  commit on an idle server               3 cutovers
    STEP 3  serve in the TP phase                  "391" (17 x 23)
    STEP 4  idle soak 150 s                        10/10 health 200
    STEP 5  serve, then the return leg             6 cutovers total
            serve in the PP phase again            "13" (8 + 5)

    health 200 | slot AGREED 18 | slot DIVERGED 0 | proxy refusals 0
    collective timeouts 0 | SIGQUIT/Fatal 0

**Step 4 is the one that matters for the cadence fix.** The previous boot
died at cutover + 120 s exactly; here the instance was healthy 129 s and
150 s after its cutover, with zero collective timeouts. **Step 5 is the
first PP -> TP -> PP round trip on this strand with serving verified in
BOTH phases.**

What this does NOT show, stated so nobody reads it as more than it is: the
commit was taken on an IDLE server. A flip that commits while requests are
resident is still gated by the draft-state carry (item 2 above) -- with
SPEC on, the quiescence predicate deliberately refuses to carry a resident
request into a speculating TP phase, so every armed window under load
still abandons by design. That is the next build, and it is unchanged by
this shift.

---
---

# HANDOFF v8 — written 2026-08-09 by successor #8

**The draft-state carry is BUILT and the wall is down.** A flip now COMMITS
under load with `SPEC=on`, repeatedly, in both directions, and the instance
survives. That was item 1 of the program and it had blocked every armed
window under load by design.

One defect remains, it is precisely located, and it is **not** the draft
carry: a request crossing a `pp_to_tp` cutover loses **exactly one output
token** on the way to the client, while its own scheduler state is provably
consistent.

## 1. State right now

- **HEAD** `6045bb77cf`, branch `feat/route-a-631`. Box SERVING on it,
  `POLICY=manual SPEC=on RANK_MIB=21000,13300,13100`, PP phase, health 200.
- **Suite** `bash scripts/run_631_flip_family.sh` -> **492 passed** (was
  460). New file `test_phase_flip_draft_bootstrap_631.py`, 32 pins,
  6 of them can-fail proofs.
- Commits this shift, in order: `9430d2e4e2` (the bootstrap),
  `18adc81cb4` (hybrid pool layer ids), `9e9f0affde` + `4147972205` (the
  verify-width defect), `1fa147e525` (the scrub falsifier switch),
  `6045bb77cf` (pending-token root + the clock report).

## 2. What is proven on metal

`scripts/route_a_631_draft_carry_probe.py`, three counting requests
decoding, flip armed 4 s in, run 09:28:54-09:29:27Z:

    cycle 0  pp_to_tp under load   COMMITTED, bootstrap on all 3 ranks
    cycle 1  tp_to_pp under load   COMMITTED, 3/3 answers CLEAN
    bootstrapped 3 | retuned 6 | cutovers 33 | abandoned 0
    faults 0 | collective timeouts 0 | health 200

**The return leg is content-correct under load.** Both legs commit with
requests resident and speculating. Nothing parks, nothing abandons,
nothing faults.

## 3. The mechanism, and why this cut

The draft KV pool is indexed by the TARGET's slot ids -- `req_to_token`
and the allocator are SHARED, only the KV buffers are separate -- so a
carried request's draft rows are allocated but hold the previous
occupant's bytes, and the draft decode reads `[0, seq_lens)`
unconditionally. They are scrubbed to zero. And the draft chain needs a
hidden state the PP phase never captured, so the first post-flip round
runs the in-tree 1-node trivial verify (always accepts its root,
functionally a plain decode, captures FULL) and the ordinary
`_draft_extend_for_decode` seeds the real chain from it. It is the
kv-session-offload BOOTSTRAP TICK at a different boundary.

Rejected: carrying the last target hidden (v3 §3 option 2). It needs
`capture_hidden_mode = LAST` on PP decode batches, which perturbs the
default path and is what makes `recapture_if_needed` re-record every plain
decode graph. One eager round per flip is cheaper.

Given up, deliberately: the scrubbed prefix reads as zeros, so a carried
request's ACCEPTANCE starts low and recovers behind the cutover point. Its
ANSWER is unaffected -- the target verifies every proposed token.

## 4. Three defects found on metal, each by a loud failure

1. **The draft pool is HYBRID** (`18adc81cb4`). The scrub derived a layer
   range from `start_layer + layer_num`, the MHA shape. This model's draft
   pool is a `HybridLinearKVPool` with no `layer_num` at all -- it carries
   `full_attention_layer_id_mapping`, and `get_key_buffer` refuses any id
   outside it. It refused before a byte moved.
2. **The verify WIDTH is not the configured width** (`4147972205`, and it
   is the general one). `_draft_extend_for_decode` strides every quantity
   it builds over `speculative_num_draft_tokens`. What they stride over is
   the number of token rows the verify JUST PRODUCED. Equal in every
   ordinary round; the bootstrap is the first caller for which they
   differ. Ruling out the graph is what found it -- the same failure
   reappeared on the eager path. The width is now a parameter, and a
   narrowed round takes the eager path END TO END (one decision, because
   `prepare_for_draft_extend` uses the graph runner's verdict to decide
   whether to build attention metadata at all).
3. **`ScheduleBatch.spec_algorithm` is a per-batch FIELD** (`9430d2e4e2`),
   copied at construction, so a batch built in PP carries NONE across the
   cutover while `prepare_for_decode` branches on it. Retuned in both
   directions. The TP->PP mirror has the same hole and was never hit
   because every flip that had ever committed committed on an idle server.

## 5. THE REMAINING DEFECT, located but not fixed

A request crossing `pp_to_tp` under load loses **one output token**.
Raw text, 09:28:54Z (`19` `2` `21`: the `0` of "20" is gone, everything
after it correct):

    ...16 17 18 19 2 21 22 23 24 25 26 27 28 29 30 31 32...

**It is not the model.** The no-flip CONTROL is clean: 3/3 counts 2..120
unbroken, same load, same checker (`--no-flip`). Run it before believing
anything about content -- this strand has already had a control that
drifted EARLIER than the flip run.

**It is not the scrub.** A run with `SGLANG_PHASE_FLIP_DRAFT_SCRUB=0` was
corrupted identically. That switch exists for exactly this separation:
content cannot depend on the scrub, because the target verifies every
proposed token.

**It is not a state off-by-one.** This is the finding that redirects the
next attempt, and it comes from the clock report the cutover now logs:

    2ede3499: seq_lens=79 seqlen=80 out[-2:]=[17, 15] input_id=15
    f200dd30: seq_lens=78 seqlen=79 out[-2:]=[220, 17] input_id=17
    731abd1a: seq_lens=78 seqlen=79 out[-2:]=[220, 17] input_id=17

Every request has `seqlen == seq_lens + 1` and `input_id ==
output_ids[-1]`. The two clocks AGREE. The token is in `output_ids`; it
does not reach the client.

**So look at the OUTPUT path across the cutover, not at the KV or the
draft state.** `scheduler.output_streamer` is rebuilt at cutover step 4b
(`_dc2.replace(..., ps=...)`), and the per-request send offsets and the
detokenizer's incremental state are what decide which appended tokens are
actually emitted. Note what disappears: a digit token INSIDE a multi-digit
number, which is precisely what incremental detokenization buffers.

Two more datapoints for whoever picks this up:
- The `tp_to_pp` leg is CLEAN. Only the leg that leaves the PP loop loses
  a token, which puts the PP loop's own result/stream handling at the
  cutover in the frame.
- Changing the root token source (output tail -> `batch.input_ids`)
  changed WHICH of the two tokens of "20" was lost, and the clock report
  then showed the two sources are equal for these requests -- so that
  change was a no-op and the loss is upstream of the root choice. Do not
  re-walk it.

## 6. The program from here

1. **The lost token.** Section 5 says where to look and what has already
   been ruled out. The probe and its control are one script.
2. Then acceptance at `POLICY=auto` with graphs + spec + max KV in one
   unmanned log, and the budget tightened toward 1024 MiB (still ~7 GiB
   loose; this shift ran 21000,13300,13100 to stay clear of the spec OOM
   and did not touch the corridor).
3. A-vs-A vs `9a929352c9`, then ship.

Accept length after a flip into TP is still UNMEASURED -- the probe greps
for it and found nothing, so the log key is wrong or the metric is not
emitted per request. Fix the probe's `accept_len` extraction (use
`meta_info`, not the log; `spec_ema_accept_len` is NOT the accept length)
before claiming the TP phase speculates after a carry.

## 7. Tools this shift leaves behind

- `scripts/route_a_631_draft_carry_probe.py` -- counting task under load,
  flip armed mid-decode, `--no-flip` control, raw text window at a break.
  A count is a determined sequence, so a corruption is LOCATED instead of
  argued about; that is what made section 5 possible.
- `SGLANG_PHASE_FLIP_DRAFT_SCRUB=0` -- separates the scrub from the seed
  in one boot.
- The cutover clock report -- `seq_lens`, `seqlen`, output tail and
  `input_id` per request, printed where the flip happens.

## 8. ADDENDUM (same shift): the output-wire suspect is ELIMINATED

The operator's steer was the output-return path -- the one wire still
unstamped, plus `init_pp_loop_state` clearing output holders with no
drain. **It is measured NOT the cause**, and this is the shift's most
useful negative result because it was the leading hypothesis.

What was done (`3b11027c6a`): the quiescence predicate gained the one
in-flight output state it never named -- `pp_outputs`, the one-slot buffer
holding an output already RECEIVED off the ring and awaiting the next
pass's processing (every other clause names a wire or a queue) -- and the
cutover now reports what `init_pp_loop_state` is about to destroy instead
of assuming quiescence emptied it.

**The report says the path is empty. 12 cutovers, 12 "output path empty at
cutover", zero "CUTOVER DISCARDS IN-FLIGHT OUTPUT".** `pp_outputs`,
`last_rank_comm_queue`, `send_output_work` and the tensor-dict inbox were
all empty every time. The token is not lost in the PP output wire or its
buffers. The gate stays as defence in depth, honestly labelled: it has
never fired.

The corruption is unchanged with it in (probe `--cycles 4`, 09:45Z):
`pp_to_tp` drops a token (2 of 3, then 2 of 3), `tp_to_pp` shows the
DUPLICATE face ('1118' where '118' was due, 3 of 3 in one cycle and 0 of 3
in the next). Suite 496 passed.

### What is now known, and it is a lot

- The token is missing from the client's **`output_ids` array**, not only
  the text -- detokenizer and tokenizer-manager are out.
- **Rank 0 is the only emitter in both phases.** The detokenizer socket is
  built once at boot (`scheduler.py:934-941`, called only from `:587`), so
  the cutover's `ps` replacement cannot move it. In PP, rank 0 never
  COMPUTES a token: the last stage samples and ids return around a ring
  2->0->1->2, unstamped and paired positionally.
- Every send cursor (`send_token_offset`, `send_decode_id_offset`,
  `finished_len`, `finished_output`) lives on the **Req**, is carried, and
  no phase-flip module touches any of them.
- Both faces occur: a DROP is that cursor running ahead of what was sent,
  a DUPLICATE is it lagging. `pp_to_tp` drops, `tp_to_pp` duplicates.

### The next instrument, named precisely

Everything above is consistent with exactly one remaining shape: **rank
0's own `req.output_ids` disagreeing with the tokens the model actually
generated**, by one, in either direction. Rank 0 emits from its own copy
and its cursor is self-consistent with that copy, so a copy that is short
produces a drop and a copy that double-appends produces a duplicate --
with no wire, no cursor and no detokenizer at fault.

So log, per resident request, at the FIRST pass after the cutover and on
ALL THREE ranks: `len(req.output_ids)`, `req.send_token_offset`, and the
last two ids. The existing cutover clock report
(`bootstrap_clock_report`) already prints exactly these at the cutover and
showed the ranks AGREEING there -- so the divergence, if it exists, opens
in the pass right after. Compare the three ranks; the one that differs
names the mechanism.

Do not re-walk: the model (control clean 3/3), the scrub
(`SGLANG_PHASE_FLIP_DRAFT_SCRUB=0` corrupts identically), the detokenizer
(ids are short too), the emitter identity (frozen at boot), the send
cursors (Req-owned, untouched), and now the output wire and its buffers
(empty at all 12 cutovers).

---

# HANDOFF v9 — written 2026-08-09 by successor #9

**The one-token loss had a cause, it is fixed, and the fix is on metal.**
`c75300cc8c`. What remains is a DIFFERENT defect with a different
signature, and this shift also proved the flip CONTENT-CORRECT in both
directions under load with speculation off — the first such result on
this strand.

## 1. State right now

- **HEAD** `c75300cc8c`, branch `feat/route-a-631`. Suite
  `bash scripts/run_631_flip_family.sh` -> **518 passed** (was 496).
- Box SERVING on it, `POLICY=manual`, **`SPEC=off`**,
  `RANK_MIB=21000,13300,13100`, health 200. (The last boot of the shift
  was the SPEC=off partition run; reboot with `SPEC=on` to resume the
  hunt.)
- Two new files: `managers/phase_flip_output_trace.py` (the instrument)
  and two test files, 22 pins, 3 of them can-fail proofs.

## 2. THE FIX: the verify's output stride is the width that RAN

`verify` stamped its `GenerationBatchResult` with the INSTANCE's
configured draft width. **Three** consumers divide that result's
row-strided tensors by the field: `_resolve_spec_v2_tokens` (each
request's accepted run at `[i*stride, i*stride+accept_len)`), the
adaptive controller's rung attribution, and the `return_hidden_states`
lane.

The two widths are equal on every ordinary round. The phase-flip
BOOTSTRAP round is the first caller for which they differ: a 1-node
trivial verify on an instance configured for 4, so `predict` has `bs`
rows while the consumer strode over 4. **Request 0 sliced its single
token correctly and every later request sliced PAST THE END and got an
EMPTY LIST** — `output_ids.extend([])` appends nothing. The KV had
advanced, so the answer resumed exactly one token short.

Fixed at the SOURCE (`verify_input.draft_token_num`), not per consumer,
so all three agree by construction. **Byte-identical on the default
path**: the normal draft path sets `draft_token_num` from the same
configured value, so only a narrowed round sees any change. It is the
same general form as the `_draft_extend_for_decode` width defect fixed
in `4147972205`, one consumer further along — and that is now a family
of two. **A third occurrence should be assumed, not hoped against:**
anything that divides a verify result by a width is suspect.

Metal, the same round before and after:

    10:13:17Z  round kind=decode -- 3367da51 have=49 +[220]
                                  | bcf3eb14 have=49 +[]
                                  | 6772dc40 have=49 +[]
    10:30:49Z  round kind=decode -- d19424a5 have=49 +[220]
                                  | 58679a46 have=49 +[220]
                                  | c396fa82 have=48 +[15]

The first break moved from position 18 to position 28.

## 3. THE INSTRUMENT, and the two hypotheses it killed

`managers/phase_flip_output_trace.py`. Per-rank output clocks for a ring
of passes before a cutover and a countdown of passes after it; the
tokens each round PRODUCED next to what it APPENDED (with `accept_lens`
and the stride the slicing used); and an emit-slice continuity check —
a GAP is the drop face, an OVERLAP the duplicate face.

Costs the default path nothing: every entry point is reached only from
the phase-flip round hook or the cutover, and the ring is maintained
only while a flip is PENDING. `SGLANG_PHASE_FLIP_OUTPUT_TRACE=0`
silences it; `SGLANG_PHASE_FLIP_OUTPUT_TRACE_POST` widens the window
(the first defect sat in the first post-cutover round, the second did
not — widen before concluding "nothing happened").

**It killed the two hypotheses v8 named, in one boot, both negative:**

1. **The three ranks' `output_ids` agree EXACTLY** at and after every
   cutover — same lengths, same offsets, same tails, every pass. v8's
   "rank 0's own copy disagrees by one" is MEASURED FALSE.
2. **The emitted slices are continuous** — no gap, no overlap, the
   continuity check has never fired. The client's array IS this rank's
   list.

Both were reached from the code before the boot, too, and the second is
worth keeping in mind: `_stream_output_generation` builds its
accumulator, fills it and flushes it inside ONE call, so there is no
cross-pass buffer for a cutover to throw away. The streamer-replacement
theory dies by reading, without a boot.

## 4. NEW MILESTONE: SPEC=off is CONTENT-CLEAN, both directions

`SPEC=off`, `--cycles 2 --concurrency 3`, run 11:0xZ:

    cycle 0  pp_to_tp under load   3/3 counts 2..120 unbroken
    cycle 1  tp_to_pp under load   3/3 counts 2..120 unbroken
    cutovers 21 | abandoned 0 | faults 0 | collective timeouts 0
    health 200

**Six determined answers, byte-complete, across flips in both
directions under load.** This is the partition that matters: the carried
KV rows AND the carried mamba/GDN linear state cross a cutover
CORRECTLY. Everything still broken lives in the SPECULATION path after a
carry, and nothing else.

It is also a shippable configuration, and the acceptance program should
record it as one rather than treating spec as all-or-nothing.

## 5. THE REMAINING DEFECT, characterised but not fixed

With `SPEC=on`, a request carried across `pp_to_tp` still corrupts, and
it is NOT the width defect. Metal, 10:41:08Z, all three ranks identical
(rid order is batch order):

    [accept_lens=[4, 4, 4] stride=4] cc0e0bcd have=74 +[17, 24, 220, 18]
                                   | 3d6a2083 have=74 +[17, 24, 220, 18]
                                   | 5c978fd3 have=74 +[17, 24, 220, 18]
    [accept_lens=[4, 1, 1] stride=4] cc0e0bcd have=78 +[15, 220, 18, 16]
                                   | 3d6a2083 have=78 +[220]
                                   | 5c978fd3 have=78 +[220]

Read it carefully, because every word is a constraint:

- The requests are **byte-identical in content and position** — same
  prompt, same `have`, the same appended run every round since the
  cutover — and the probe decodes at **temperature 0**. Under greedy
  decoding three identical rows MUST sample the same token.
- Request 0 samples `15` ('0', completing "30"). Requests 1 and 2 sample
  `220` (' '). With `accept_len=1` that token is the target's own bonus
  at the ROOT position, so this is the TARGET disagreeing with itself
  across batch rows, not a draft being rejected.
- They then continue self-consistently ("…29 3 30 31 32"), so the wrong
  token went into their KV too. The corruption is an INSERTION, not a
  loss — the opposite face from the defect this shift fixed.
- **The batch's index-0 request is correct in BOTH defects.** The fixed
  one was index-proportional by construction (`i * stride`). That this
  one also spares index 0 is the strongest single clue available and
  should drive the next attempt.
- It appears at the FIRST round whose `accept_lens` split — round 9
  after the cutover, ~28 tokens in — never before it. Seven identical
  `[4,4,4]` rounds precede it.
- The no-flip CONTROL is clean 3/3 with the same three concurrent
  requests, so it is not the model and not batching.

### What to do next, in order

1. **Find the third member of the width family.** Grep every consumer
   that strides a verify-shaped tensor by a width or by a batch index:
   `eagle_prepare_for_verify`'s `out_cache_loc` / `req_to_token` writes,
   `_draft_extend_for_decode`'s `select_index`, the accept-path
   compaction (`_finalize_accept_tree_path`), and the draft KV slot
   allocation. The bootstrap round runs ONE token wide where every other
   round runs four; anything that recorded a per-request offset during
   that round is wrong by `i * 3` for row `i` — and index 0 by zero.
   That is exactly the observed shape, and it explains why the damage
   surfaces LATER (the bad offset is only read once the sequence reaches
   those rows).
2. Instrument the root sample directly: log, for one post-cutover round,
   each row's root token, its `seq_lens`, and its `out_cache_loc` /
   `req_to_token` tail. The target disagreeing with itself across rows
   is an indexing fact and will be visible there.
3. Do NOT re-walk the carried KV or the mamba/GDN state: SPEC=off moved
   both correctly, 6/6, section 4.

## 6. DO-NOT-RE-WALK (updated; the v8 list still stands)

Added by this shift, each measured:

- The three ranks' `output_ids` do not diverge (section 3).
- The emit slices are continuous, and the streamer cannot lose a
  buffered slice at a cutover (section 3).
- The carried KV and the carried mamba/GDN linear state are correct
  across cutovers in both directions (section 4).
- The verify output stride (section 2) — FIXED, with a can-fail pin that
  reproduces the empty slice in unit form.

Unchanged from v8: the model, the draft KV scrub, the detokenizer, the
emitter identity, the Req-owned send cursors, the PP output wire and its
four buffers.

## 7. Tools this shift leaves behind

- `managers/phase_flip_output_trace.py` and its two env switches.
- `test/registered/unit/managers/test_spec_verify_width_631.py` — the
  defect in unit form: three rows, stride 4, requests 1 and 2 slice
  empty. Use it as the template for the third family member.
- The method that worked, stated plainly: the round trace prints what a
  round PRODUCED next to what it APPENDED. "Appended nothing" and
  "produced nothing" are different defects, and no amount of reasoning
  about `len(output_ids)` separates them.

## 8. Two more eliminations, added after v9 was written

- **The draft KV scrub is INNOCENT for the remaining defect.** Re-run
  with `SGLANG_PHASE_FLIP_DRAFT_SCRUB=0` (confirmed live: three "SCRUB
  DISABLED" lines, one per rank): the corruption is IDENTICAL, same
  position 28, same shape. Predecessors ran this switch against the
  WIDTH defect; this is a fresh measurement against the one that is
  left, and it closes the same door again.
- **`--speculative-num-steps 0 --speculative-num-draft-tokens 1` FAULTS
  the instance** when a flip bootstrap runs (3 faults, health 0,
  11:01Z). It was tried as a cheap way to make the bootstrap round's
  width equal the configured width and so test whether the 1->4
  TRANSITION is what breaks. It is not a usable probe on this build and
  the question stays open. Do not re-walk it as written; if the
  transition hypothesis is worth testing, test it with an instrument on
  the allocation (section 5, step 2) rather than by reconfiguring the
  drafter.

Serving left on `SPEC=on POLICY=manual RANK_MIB=21000,13300,13100`,
health 200, on this commit — the configuration the next shift wants for
the hunt. `SPEC=off` is the configuration that is content-clean today.

---

# HANDOFF v10 — written 2026-08-09 by successor #9 (second shift)

**`pp_to_tp` is CONTENT-CLEAN under load with speculation on.** 6 of 6
determined answers byte-complete across two independent runs, where
every run before this shift corrupted 2 of 3. HEAD `a970a8fed9`, suite
**522 passed**.

## 1. The fourth width defect, and why it was the answer-corrupting one

`commit_mamba_states_after_verify` recovers each request's accepted TREE
STEP as `accept_index[req, accept_lens-1] - arange(0, bs*W, W)[req]`.
`accept_index` holds GLOBAL node ids minted in the width the verify RAN;
handed the CONFIGURED width it becomes `i - 4i` on the 1-wide bootstrap
round — negative step ids for every row but row 0.

**The damage is a RECURRENT state written from the wrong step**, which is
why six shifts could not find it by looking at token bookkeeping:
nothing raises, and the KV and append clocks stay in exact agreement
(`kv == seen - 1` on every row, every round — measured). The row decodes
on from a subtly wrong linear-attention state and DRIFTS, surfacing as a
wrong token ~28 tokens later. Every collected constraint falls out of
that one mechanism: row 0 spared, SPEC=off clean, no-flip control clean,
scrub innocent.

Converted with it: `clear_unaccepted_c128`, `compute_spec_v2_logprobs`
(step count now taken from `accept_index`'s own width, so the arange
matches the tensor it indexes), and `_draft_extend_for_decode`'s
`verify_width`, now passed UNCONDITIONALLY.

**THE FAMILY IS NOW FOUR** (`4147972205`, `c75300cc8c`, and two here).
Treat "a width or a stride taken from `self.speculative_num_*` rather
than from the tensor or input at hand" as a defect until proven
otherwise. The pattern that survives is `bonus_tokens`' kernel, which
takes its stride from `accept_index.shape[1]` — the tensor's own width.
Copy that.

## 2. Method note: what actually found it

Two instrument readings, in this order, and neither was a guess:

1. **The flat row tensor.** `flat=[24,220,18,15, 220,0,0,0, 220,0,0,0]`
   with `accept_lens=[4,1,1]`: the expected token sat at NO other index.
   That killed "the slicing is misaligned" and pushed the defect
   upstream of `predict` in one line.
2. **`kv` against `seen` per row.** Exact agreement everywhere ruled out
   a commit/append desync and left the STATE itself — which is what made
   a recurrent-state defect the only candidate standing.

Then a delegated sweep enumerated every verify-row-to-request mapping
and found the one consumer that both fires on a hybrid GDN model and
corrupts silently. Instrument, read, narrow, then sweep — not theory.

## 3. WHAT IS LEFT: the tp_to_pp DUPLICATE face

`pp_to_tp` no longer corrupts. `tp_to_pp` still does, and it is the
DUPLICATE face — one token appended twice, 1-2 of 3 requests:

    got 1118, expected 118      (an extra '1')
    got 11117, expected 117     (an extra '1')

At the cutover all three requests sit at `tail=[220, 16, 16]` (" 11")
and the post-cutover passes show `+0` for several passes, so the extra
token lands as the PP phase resumes.

**The named hypothesis, and it is well-founded but METAL-UNPROVEN.** The
two phases hold DIFFERENT pending-token conventions:

- plain decode (PP): the last appended token is NOT yet in the KV; it is
  the next round's input. Measured this shift: `kv == seen` exactly on
  every PP round.
- speculative decode (TP): every accepted token IS committed during
  verify, and the next round re-roots at `bonus_tokens`.

A carry that hands PP an `input_ids` whose token the KV already holds
would have the PP round regenerate it — one token, appended twice,
exactly the observed face. The cutover clock report checks
`seq_lens == seqlen - 1`, but it only runs on the `tp_phase` leg, so
this direction has never been checked at all.

**Next step, precisely.** Print the same clock report on the tp_to_pp
leg (it is `bootstrap_clock_report`, and it takes any batch), plus
`kv_committed_len` per request, at the cutover and for the first three
PP passes. If `kv == seen` at the cutover while the carry assumes
`kv == seen - 1`, that is the defect and the fix is to reconcile the
convention on the leg, not to touch either phase's own bookkeeping.

## 4. State

- HEAD `a970a8fed9`, suite 522 passed, working tree clean.
- SERVING on that commit, `POLICY=manual SPEC=on
  RANK_MIB=21000,13300,13100`, health 200.
- `SPEC=off` remains content-clean in both directions (v9 section 4).
- Acceptance program (v9 section 6 / the anchored capacity spec) is
  untouched and still waits on the tp_to_pp leg.

## 5. ADDENDUM (same shift): the convention-mismatch hypothesis is METAL-FALSIFIED

The carry clocks now print on BOTH legs (`[#631 OUTTRACE] carry clocks`),
which is the check the `tp_to_pp` direction had never had. Measured
11:37Z, same boot, both cutovers:

    carry clocks dir=pp_to_tp -- 8338d8f4 seen=79  kv=78  tail=[220, 17]
                              | 04455e63 seen=79  kv=78
                              | 7fa6b9bb seen=79  kv=78
    carry clocks dir=tp_to_pp -- 86823e3b seen=395 kv=394 tail=[220, 16]
                              | 12a8af99 seen=393 kv=392
                              | 68d1d9a1 seen=392 kv=391

**`kv == seen - 1` on BOTH legs, every request.** The two phases do NOT
disagree about the pending token at the cutover: the last appended token
is out of the KV in both directions, exactly as PP expects. So the carry
does NOT hand PP a token the KV already holds, and **normalizing the
handed-over `(input_id, seqlen)` pair would be a fix for a defect that
is not there.** v10 section 3's hypothesis is dead; do not spend a boot
on it.

(The `kv == seen` reading in v10 came from PP-phase ROUND lines, sampled
at a different point in the pass than the cutover — after that round's
commit and before its append. It described the sampling point, not a
convention. A clock is only comparable to another clock read at the same
point in the pass; that is the lesson worth carrying.)

### What the same run adds about tp_to_pp

- `pp_to_tp` clean again, 3/3 — **9/9 across three independent runs**
  since `a970a8fed9`. The mamba-step fix holds; no regression.
- `tp_to_pp` still corrupts, and one specimen is WORSE than a duplicate:
  alongside the familiar `1119` for `119`, one request degenerated into
  repetition — `ends [100, 100, 100, 100, 100, 1]`. A repeated n-gram is
  a STATE signature, not an off-by-one in token bookkeeping.
- That points back at the recurrent state on the OTHER leg. Leaving a
  speculating TP phase, a request's mamba state was last committed at an
  accepted STEP inside a verify tree; entering plain PP decode it must
  correspond to exactly the committed prefix. The GDN mover moves the
  state, but nothing has ever checked that the state it moves is at the
  right step for a phase that no longer speculates.
- **Named next instrument** (unproven, but this is where the evidence
  points): log, on the tp_to_pp leg, each request's committed mamba step
  against `accept_lens - 1` of the LAST verify before the cutover, and
  compare the moved state's step on both sides of the mover. The
  `commit_mamba_states_after_verify` call is now width-correct; the open
  question is whether the last spec round's committed step is the one a
  plain-decode phase should resume from.
- Do NOT re-walk: the pending-token convention (this section), the
  output wire, the send cursors, the detokenizer, cross-rank divergence.

# HANDOFF v11 — written 2026-08-09 by successor #10

**`tp_to_pp` with SPEC=on is CONTENT-CLEAN.** 36 of 36 determined answers
byte-complete across 6 `tp_to_pp` and 6 `pp_to_tp` crossings under load,
two independent runs, plus a third instrument that reads the id stream
directly. HEAD `68817ac687`, suite **530 passed**. The last correctness
defect on this strand is closed.

## 1. THE DEFECT: a relay the source phase stopped maintaining

A plain-decode round does not carry its input token on the batch. The
last speculative round ends with `batch.input_ids = None` ("rebuilt next
iter from draft_token", `scheduler.py`), and the next round's
`resolve_forward_inputs` gathers the token out of
`future_map.output_tokens_buf` — a pool-indexed relay whose contract is
that a round which READS a row was preceded by a round that WROTE it.

The TP speculative phase breaks that contract while it runs, and it is
entitled to: the NON-OVERLAP V2 path (`scheduler.py`, `elif not
batch.spec_algorithm.is_none()`) drives the worker synchronously and
installs `next_draft_input` directly, so it never calls
`_relay_forward_payload` and never stashes. Inside the phase this costs
nothing — the next round rebuilds `input_ids` from the draft tokens
instead of gathering.

**At the cutover the READER changes.** The first PP decode gathers, and
what it gathers is the token the PREVIOUS PP phase stashed — stale by the
entire TP phase, hundreds of tokens. The model appends a foreign token to
a correct prefix and answers from it.

## 2. Why it looked intermittent, and why only one leg

Measured on every `tp_to_pp` cutover of boot 11:57:26Z: **3 of 3 requests
would have gathered a stale token, every time** (`relay=198 / 279 / 628 /
17 / 15 / 220` against `last=16`). The MISMATCH rate is 100 %; the
visible corruption rate was ~25 % (3 of 12 requests over 4 crossings).

Both numbers are correct, and the gap between them is the reason six
shifts read this as intermittent: on a hard-determined counting task the
model usually recovers from one foreign token, so most crossings absorb
the defect silently. **A defect's firing rate and its visible rate are
different quantities, and instrumenting the mechanism gives the first
one.** Rate-based reasoning over answers alone would have kept mis-sizing
this forever.

The direction dependence is equally mechanical: the PP phase stashes on
EVERY pass (`scheduler_pp_mixin`), so a batch entering the TP phase finds
the relay fresh. `pp_to_tp` was never exposed. That asymmetry is the
whole of it — no state, no step, no width.

## 3. THE FIX, and the rule it comes from

`reseed_decode_input_relay` (`phase_flip_resident_carry.py`), called on
the `tp_to_pp` leg only (`phase_flip_runtime.py`, the `else` of the
draft-bootstrap arm). It re-derives each carried request's last token
from `req.output_ids[-1]`, stashes it with the same primitive the
batch-rebuild path already uses, and clears the leftover speculative
`input_ids` so the gather is the only source.

    A VALUE HANDED ACROSS THE CUTOVER MUST BE RE-DERIVED FROM THE TRUTH
    AT THE HANDOVER, NEVER INHERITED FROM A BUFFER THE SOURCE PHASE
    STOPPED MAINTAINING.

This is the fifth law of the strand, and it is the general form of the
width family's rule (take the width from the tensor that ran) applied to
values instead of shapes: ask the thing that KNOWS, not the thing that
usually knows.

## 4. What was falsified on the way, so it is not re-walked

- **The committed mamba STEP is innocent.** v10 section 5 named it as
  the next instrument. It was not built, because the metal answered a
  cheaper question first: at the cutover every request sat at
  `tail=[220,16,16]` (" 11") and the FIRST PP round handed one request a
  SPACE where the digit belonged, then the answer continued coherently.
  One wrong token with a correct continuation is a wrong INPUT, not a
  drifting recurrent state — a drifting state does not self-heal.
- **Not a lost token either.** The new `route_a_631_tail_loss_probe`
  renders the returned `output_ids` back into digits without a tokenizer
  and compares that stream to the text. Pre-fix both streams broke at the
  same place, which killed the send-cursor and detokenizer readings for
  this face too.
- The v10 section 5 falsification of the pending-token convention still
  stands and is untouched by this.

## 5. Method note

The finding came from reading the pass trace AT the cutover instead of
reasoning about the carry: `at-cutover` gave the tails, the next `round`
line gave what each request was handed, and the two together located the
defect in one reading. The delegated code trace then supplied the exact
wire (`resolve_forward_inputs` ← `output_tokens_buf`) and the fact that
the non-overlap spec path never writes it. Instrument, read, then sweep —
the same order that found the width family.

## 6. State

- HEAD `68817ac687`, suite 530 passed, working tree clean.
- SERVING on that tree, `POLICY=manual SPEC=on
  RANK_MIB=21000,13300,13100`, health 200, boot 11:57:26Z.
- Probes: `route_a_631_draft_carry_probe.py` (answers),
  `route_a_631_tail_loss_probe.py` (ids vs text, NEW),
  `route_a_631_roundtrip_probe.sh` (survival).

## 7. NAMED RISK, not yet a defect

The `tp_to_pp` leg leaves `batch.spec_info` in place. The comment at the
retune site says "its spec_info is simply not read there", and that is
false as written: `filter_batch` and `merge_batch` call into it
unconditionally when it is truthy, and `capture_hidden_mode` is derived
from it in `forward_batch_info`. Nothing observed has been traced to
this, and the answers are clean, so it is recorded rather than fixed —
but a PP phase capturing hidden states it has no drafter to consume is
wasted work at best, and the first post-flip `merge_batch` calls
`EagleDraftInput.merge_batch(None)`. Worth a can-fail before the
acceptance soak is trusted for long dwells.

## 8. ADDENDUM (same shift): where the acceptance program actually stands

The correctness work is finished; this is the state the capacity work is
handed over in, with the unproven parts labelled as such.

**Proven this shift, on metal:**

- The survival soak passes on the fix: `route_a_631_roundtrip_probe.sh`
  reports health 200, 6 cutovers, slot AGREED 3 / DIVERGED 0, 0 proxy
  refusals, 0 collective timeouts, 0 SIGQUIT.
- CUDA graphs were ON for every correctness run above
  (`disable_cuda_graph=False`, capture completed), so the 36/36 is a
  full-perf result, not an eager one.
- `POLICY=auto` boots and ARMS on its own: `PHASE-POLICY armed: N=7004
  tok (break-even 3.2s / (1/1681 - 1/7245.5)), min dwell 3s, idle dwell
  3s, resting layout pp` and repeated `PHASE-POLICY arming tp_to_pp:
  idle 3.0s >= 3s, returning to the prefill resting layout`.
- Two more clean `tp_to_pp` crossings under the corridor run, and two
  more after the final reboot: **8 clean `tp_to_pp` crossings** total on
  the fix, 0 corrupt.

**THE 5090 CANNOT BE FILLED THROUGH `--rank-gpu-memory-mib`, and this is
a wall, not a tuning miss.** `RANK_MIB=23790` for rank 0 is refused at
boot:

    ValueError: The per-rank budget of 23790 MiB for rank 0 on GPU 0 is
    not physically available: the rank holds 7.90 GiB and 13.27 GiB of
    the device is free to it (31.34 GiB total)

21.17 GiB is all the check can see, because the two-stack weights arena
is resident at the moment the check runs, while the corridor sampler
later measures ~2.9 GiB free on that card at runtime. The headroom is
real and the check cannot reach it. Filling it needs the arena's
residency during the check to change, which is exactly the #297/#635
machinery the capacity spec authorises. `21500` boots; `22070` (the
value a corridor-target would ask for) would not.

**A MEASUREMENT TRAP, recorded because it nearly became a wrong fix.**
At `RANK_MIB=21500,14090,13840` the corridor BREACHED: min free
570 / 2598 / 586 MiB, 84 breaches on each 3080. Backing the 3080s off to
13780 / 13540 gave min free 1936 / 3974 / 2008 with ZERO breaches -- but
so did the ORIGINAL 13300 / 13100 (1914 / 3912 / 1866). The budget delta
did not move the minimum at all.

The difference between those runs was the LOAD, not the budget: the
breaching run had `route_a_631_acceptance.py full` on it (long prefills),
the others had the counting probe (decode-dominated). A slope derived
across two load shapes is not a slope. **The corridor minimum must be
sized against the ACCEPTANCE load, and every corridor number in this
strand must name the load it was taken under.** None of the pre-existing
corridor figures do.

**Still METAL-UNPROVEN, in the order the spec asks for them:**

1. accept-len evidence in the unmanned log. `route_a_631_acceptance.py`
   returned `spec_accept_length: null` / `spec_accept_rate: null` on this
   boot; whether that is a missing `meta_info` field or a request that
   never ran in the TP phase was not determined.
2. the one unmanned log carrying flips + graphs + spec + max KV together,
   with per-phase KV pool sizes.
3. the bs=1 spill leg (#364 / #104 / kvso / #287) -- untouched.
4. the bs=1 YaRN leg beyond 262144 -- untouched.
5. the A-vs-A gate against `9a929352c9` -- untouched.

**State at end of shift:** SERVING healthy on the fix tree,
`POLICY=auto SPEC=on RANK_MIB=21500,13780,13540`,
`PHASE_POLICY_TP_TOK_S=1681.0`, corridor HELD under the flip probe load
and UNVERIFIED under the acceptance load.

---

# HANDOFF v12 — written 2026-08-09 by successor #11

Three defects closed with can-fail proofs, one wall broken open, and ONE
NEW STRUCTURAL FINDING that reframes the corridor problem the last two
shifts were fighting. Read section 3 first if you read nothing else.

## 1. THE 5090 WALL WAS NOT PHYSICAL — it was a per-stack baseline

The refusal
`the rank holds 7.90 GiB and 13.27 GiB of the device is free to it` was
read for two shifts as "the card cannot be filled". The arithmetic
explains itself once written out. With one rank on a card,

    reachable = used_by_me + device_free
              = (pre_load_free - now_free) + now_free
              = pre_load_free

exactly. The check was never about the card; it was "does the budget fit
in whatever was free when THIS STACK began loading". A phase-flip
instance builds THREE stacks in one process, each re-entering
`init_torch_distributed` and taking its own pre-load reading with the
previous stack resident. Measured on the 5090, boot 12:27:

    stack 1 (PP weights)  Load weight begin. avail mem=30.46 GB
    stack 2 (TP weights)  Load weight begin. avail mem=23.40 GB
    stack 3 (MTP draft)   Load weight begin. avail mem= 8.78 GB

So the check got STRICTER as the budget grew: a bigger budget makes stack
1 allocate more, which lowers stack 2's baseline, which refuses the
budget that caused it. A feedback loop wearing a physical limit's
clothes.

FIX: read what THIS PROCESS holds from NVML (registry identity map, never
the CUDA ordinal) plus NVML's free column. Both terms are stack-invariant
and they count what torch cannot see — CUDA context, VMM arena handles,
raw workspaces. Delta arithmetic kept as fallback where NVML cannot
attribute the pid (MPS, pid namespaces); the message names its basis.

METAL: `BUDGET-REACH[nvml] rank 0: budget 23000 MiB, this process holds
15.62 GiB, card free 15.71 GiB of 31.84 GiB total -> reachable 31.33 GiB,
shortfall 0.00 GiB`. 23000 boots where >21500 was refused.
`max_total_num_tokens` 672142 -> 739186.

## 2. THE NULL ACCEPT-LEN WAS A DEAD WIRE, NOT A MISSING MEASUREMENT

The probe was right all along: it uses `/generate` and reads
`meta_info["spec_accept_length"]` correctly. The server never emitted it,
while the scheduler LOG printed `accept len: 3.20, accept rate: 0.73,
cuda graph: True` on the very same boot — because `metrics_reporter`
reads `scheduler.spec_algorithm` LIVE, and the streamer holds a VALUE
COPY.

`SchedulerOutputStreamer` is a slots dataclass carrying `spec_algorithm`,
and `_GenerationStreamAccumulator` gates every per-request spec counter
on that copy. A phase-flip instance PARKS speculation at boot (it rests in
PP), so `init_output_streamer` copies NONE; the cutover refreshed only
`ps`, so the copy stayed NONE for the life of the process. Counters were
accumulated on the Req and dropped on the way out.

FIX: refresh `spec_algorithm` with the phase being entered, and PIN it in
`verify_flip_cutover` — a stale copy here is otherwise SILENT (answers
stay correct, only the evidence vanishes), which is precisely what the
self-check exists to make loud.

CAN-FAIL PROVEN: with the production line reverted, both new tests go
red; restored, green.

**HONEST LIMIT — do not claim this closed.** `meta_info` was still empty
on metal after the fix. The remaining gate is in `tokenizer_manager`:
shipping requires `spec_verify_ct[i] > 0` AT FINISH TIME, and the
decision is made by the phase active when the request COMPLETES. Under
auto policy requests routinely start in TP and finish in PP (observed
directly: `phase_before tp phase_after pp`), and those can never ship
their counters. My fix is necessary but not sufficient. THE ACCEPTANCE
LOG'S ACCEPT-LEN EVIDENCE THEREFORE COMES FROM THE SCHEDULER LOG, which
carries it reliably (105 and 144 lines in the two runs below).

## 3. THE FINDING THAT MATTERS: the corridor is not the KV budget's to give

The last shift recorded that a budget delta "did not move the minimum at
all" and attributed it to differing load shapes. That was half the story.
I ran the SAME load twice, changing only RANK_MIB:

    load: 1 long-prefill worker over 8192/32768, 3 decode workers,
          bs<=4, 300s mixed + 60s idle, POLICY=auto, SPEC=on, graphs on

    card            budget cut    MIN free before -> after
    5090            -910 MiB      114 -> 804   (+690)
    3080b (rank 2)  -990 MiB       42 -> 292   (+250)
    3080a (rank 1)  -980 MiB       50 ->  54   (+4)

On the 3080a a full gigabyte of KV budget bought FOUR MiB of corridor.
The recovery is not proportional and on one card it is essentially zero.

MECHANISM, and it is the same one that caused the crash in section 4:
torch's caching allocator expands into whatever the KV pool did not take
and, by design, never returns those pages to the driver. Free memory on a
card doing heavy prefill therefore converges toward zero REGARDLESS of
the budget — lowering RANK_MIB just hands the freed pages to torch
instead of to the corridor.

CONSEQUENCE FOR THE NEXT SHIFT: **the corridor floor cannot be held by
tuning `--rank-gpu-memory-mib`.** Iterating that knob is the trap two
shifts have now fallen into. The knob that actually governs the minimum is
torch's allocator growth. Candidates, untested, in the order I would try
them:
  * `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — lets torch
    return segments to the driver; must be checked against CUDA graph
    capture and the VMM arena before it is believed;
  * `garbage_collection_threshold:<f>` paired with
    `set_per_process_memory_fraction`, so a reserve ceiling exists to
    collect against;
  * an explicit `torch.cuda.empty_cache()` at the phase-flip seam, which
    is a place the code already owns and where a re-warm is affordable.
Whichever is chosen, measure it with `route_a_631_corridor.py` under the
load above — the numbers in this section are the baseline to beat.

## 4. A CRASH FOUND, AND ITS FALSE COMMENT CORRECTED

The inherited HEAD DIED under the acceptance load at 12:47:45, ~20 min in
(evidence: `docs/dev/631/evidence/s11_flip_arena_oom_crash_20260809T1247Z.log`):

    RuntimeError: cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY
      _swap -> restore_backing -> finalize -> _back_spans -> commit_range

then SIGQUIT, instance lost. `_build_kv_backing_swap` asserted in a
comment that the restore "cannot fail for want of memory" because boot
sized the budget for max(PP, TP). That holds only against a static
allocator, and section 3 is why it is false under load.

FIX: `_mem_create_reclaiming` — on OUT_OF_MEMORY, release torch's cached
blocks and retry ONCE. Zero cost on the happy path, a genuinely full card
still raises, a non-OOM error is never retried. The false comment is
corrected in place.

**METAL-UNPROVEN, labelled as such:** the reclaim fired ZERO times across
both acceptance runs. It is unit-proven insurance
(`test_kv_arena_reclaim_631.py`, 4 cases including the ordering pin), not
a metal-confirmed recovery. Do not report it as a proven crash fix.

## 5. State at end of shift

HEAD `266a09c85b`, suite `scripts/run_631_flip_family.sh` **532 passed**
(530 inherited + 2 new), working tree clean apart from this handoff.

SERVING HEALTHY on that tree, `RANK_MIB=22090,12800,12550 POLICY=auto
SPEC=on PHASE_POLICY_TP_TOK_S=1681.0`, graphs on, health 200,
`max_total_num_tokens=590224`.

Two full acceptance runs, both with ZERO aborted requests:

    run 1  RANK_MIB 23000,13780,13540   max_total 739186
           77/77 ok, 0 aborted, 29 policy transitions, 90 cutovers,
           105 accept-len log lines, corridor MIN 50/114/42  BREACH
    run 2  RANK_MIB 22090,12800,12550   max_total 590224
           91/91 ok, 0 aborted, 19 policy transitions, 66 cutovers,
           144 accept-len log lines, corridor MIN 54/804/292 BREACH

Both runs returned to the PP resting layout on their own and the
post-idle 32768-token probe prefilled in PP at 4467.9 and 4480.2 tok/s —
the PP-class rate, i.e. the flip is not in the latency path from rest.

Run 2 is the boot left standing: it is the closer of the two to the
corridor law on every card. NEITHER SATISFIES IT. The corridor is the
open item, and section 3 says why the obvious knob will not close it.

## 6. Free capacity nobody has taken yet

The server prints its own restart hint and it has been ignored:

    Uneven DCP: restart with SGLANG_UNEVEN_TOKEN_VECTOR=30,19,15 to raise
    max_total_num_tokens from 739186 to ~871552 (per-rank profiled
    capacity [408556, 259734, 213910]; active vector [28, 26, 20] leaves
    ranks idle).

+18% KV for free. `route_a_631_prod_boot.sh` pins `28,26,20` as its
default. I did NOT take it this shift, deliberately: it raises the pool on
the ranks that are currently UNDER-using their budget, so it moves VRAM
and must not be bundled into a corridor measurement that is isolating the
budget. Take it as its own change, with its own corridor run.

## 7. Still METAL-UNPROVEN, in the spec's order

1. the corridor at ~1024 per card under the acceptance load — see §3;
2. `meta_info` accept-len end to end — see §2;
3. the bs=1 spill leg (#364 / #104 / kvso / #287) — untouched;
4. the bs=1 YaRN leg beyond 262144 — untouched;
5. the A-vs-A gate against `9a929352c9` — untouched;
6. the arena reclaim on metal — see §4.

---

# HANDOFF v12 ADDENDUM (same shift): THE CORRIDOR HOLDS

Written after v12 above. Sections 3 and 5 of v12 said the corridor was
the open item and that the budget knob would not close it. Both stand —
and the corridor is now CLOSED, by the knob section 3 named.

## The result

    #631 VRAM CORRIDOR -- 3901 samples over 390.3s at 100 ms, floor 1024
    gpu  name                  total   MIN free    mean   breaches
    0    RTX 3080              20480     1034.4  2620.1          0
    1    RTX 5090              32607     2823.7  3645.3          0
    2    RTX 3080              20480     1228.4  2219.2          0
    per-card MINIMUM free: 1034, 2824, 1228 MiB   (floor 1024)
    CORRIDOR HELD: True

Under the acceptance load itself — POLICY=auto, SPEC=on, CUDA graphs on,
1 long-prefill worker over 8192/32768, 3 decode workers, bs<=4, 300 s
mixed + 60 s idle. 87/87 requests ok, 0 ABORTED, 23 policy-driven phase
transitions, 129 accept-len log lines. Returned to the PP resting layout
on its own; the post-idle 32768-token probe prefilled in PP at
4481.0 tok/s, so the flip is not in the latency path from rest.

Config: `RANK_MIB=22700,11920,11970`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, both now the defaults
in `scripts/route_a_631_prod_boot.sh`.

## How it was reached, and the ONE lever that mattered

Four runs of the SAME load, changing one variable at a time:

    run  RANK_MIB              alloc      MIN free per card    held
    1    23000,13780,13540     default      50 /  114 /   42   no
    2    22090,12800,12550     default      54 /  804 /  292   no
    3    22090,12800,12550     expandable  146 / 1672 /  448   no
    4    22700,11920,11970     expandable 1034 / 2824 / 1228   YES

Run 1 -> 2 is the budget lever alone: ~1 GiB off each card bought +690,
+250 and +4 MiB of floor. Run 2 -> 3 is the allocator lever alone, same
budgets: +92, +868, +156 MiB, and the 5090 went from 1400 breaches to
zero. The allocator is what governs the minimum, because torch's caching
allocator expands into whatever the KV pool does not take and the default
segment allocator never returns those pages to the driver.

Only AFTER the allocator was fixed did the budget lever become
predictive: run 3 -> 4 moved each card by very nearly the distance it was
short, which is what let the final numbers land on the floor rather than
near it.

## What this cost, and the two ways to win it back

`max_total_num_tokens` is 459392 at the corridor-compliant budget, down
from 739186 at the breaching one. That is the honest price of the floor.
Two named, unspent credits:

1. The 5090 finishes the run with ~1800 MiB above the floor. rank 0 can
   take most of that. On its own it will NOT raise the global pool while
   rank 1 is the min-reducing rank.
2. `SGLANG_UNEVEN_TOKEN_VECTOR=30,19,15`, which the server prints as its
   own restart hint (+18% at the old budget). This is the change that
   relieves the min-reduction, so it is the one that makes credit 1
   spendable. Take them TOGETHER, in one boot, with its own corridor run.

## Unchanged from v12 section 7

Still metal-unproven: `meta_info` accept-len end to end (the scheduler log
carries it; the wire does not), the arena reclaim on metal (fired 0 times
across four runs), the bs=1 spill leg, the bs=1 YaRN leg beyond 262144,
and the A-vs-A gate against `9a929352c9`.

---

# HANDOFF v13 — written 2026-08-09 by successor #11 (second shift)

Ordered work from the operator: spend the two capacity credits together,
close the second accept-len gate, then the bs1-spill and YaRN legs. One
is closed, one is a MEASURED WALL, and the legs are untouched — context
exhausted. Everything below is labelled by what it is.

## 1. THE SECOND ACCEPT-LEN GATE IS CLOSED (code + tests; metal unproven)

v12 §2 left `meta_info` empty even after the streamer's stale
`spec_algorithm` copy was fixed, and named the remaining gate. It was in
`_GenerationStreamAccumulator`: the spec-counter appends were gated on
the LIVE phase. A phase-flip instance speculates only in TP, so a request
that verified in TP and FINISHED after the return flip had its counters
dropped — the numbers were on the Req and the phase active at completion
decided they would not be shipped. Under POLICY=auto that is a large
share of all traffic (observed directly: `phase_before tp phase_after
pp`).

Fixed by keying the predicate on the SERVER CONFIG
(`server_args.speculative_algorithm is not None`), threaded into the
accumulator as `spec_configured`. Constant for the life of the process,
so the lists stay BATCH-ALIGNED — which matters because
`tokenizer_manager` indexes them by request position, and the ragged
appends are exactly what made its defensive `len(...) > i` guard
necessary. A non-speculating server is byte-identical (predicate false).

Caught during implementation and worth recording: the first version read
`self.server_args` inside the accumulator, which has no such field — it
would have raised AttributeError on the first streamed batch. The
accumulator takes a plain bool instead.

TESTS: `test/registered/unit/managers/test_spec_counter_wire_631.py`, 4
cases — the PP-finish case, the batch-alignment pin, the can-fail that
the old predicate drops them, and the non-speculating byte-identity.
Suite 540 passed.

**METAL-PROVEN**, on the restored boot, first request:

    spec_accept_rate 0.6625, spec_accept_length 3.0,
    spec_num_correct_drafts 159, spec_num_proposed_drafts 240,
    spec_verify_ct 80, spec_correct_drafts_histogram [11, 18, 12, 39]

So the acceptance log now carries accept-len on the `meta_info` wire for
every request class, not only in the scheduler log. Both halves of the
defect are closed: the streamer's stale value copy (v12 section 2) and
the phase-at-finish gate above.

## 2. THE CREDITS ARE MOSTLY NOT SPENDABLE — a measured wall

The server prints a restart hint
(`SGLANG_UNEVEN_TOKEN_VECTOR=... to raise max_total_num_tokens from X to
~Y`) and both this shift and the operator read it as free capacity. IT IS
NOT. The hint maximizes CAPACITY UTILISATION and knows nothing about the
corridor. Three attempts, all crashed:

    run  vector      RANK_MIB              max_total  MIN free        result
    4    28,26,20    22700,11920,11970       459392   1034/2824/1228  HELD
    5    34,17,13    24400,11920,11970       607680   1952/  16/1994  OOM crash
    6    34,17,13    23050,11920,11970       607680   1950/  34/1852  OOM crash
    7    28,20,16    22700,11920,11970       500800    922/ 816/ 904  breach+crash

WHY, and this is the part that was not understood before: the vector
does not merely re-label ratios, it sets each rank's ALLOCATED pool as
`unit x ratio`. Moving rank 0 from 28/64 to 34/64 grew its pool from
173824 to 322830 tokens — roughly +4.6 GiB on the 5090, against the
2824 MiB of slack that existed. Backing the BUDGET off 1350 MiB barely
moved the minimum (16 -> 34 MiB), because the budget only caps capacity
while the vector decides the allocation. That is the same
budget-is-not-the-lever lesson as v12 §3, in a second costume.

Run 5's crash is `torch.OutOfMemoryError` in the allocator, NOT the arena
path — the reclaim correctly did not fire (0 events), so it is still
metal-unproven.

MEASURED CEILING. Using the run 4 numbers and the budget sweep's
32 tokens/MiB on rank 0, the per-rank pool ceiling that keeps every card
at the floor is [231436, 161415, 125203] tokens, i.e. an achievable total
around 510k, vector ~29,20,15 — about +11%, not the +32% the hint
advertised. Run 7 tested the conservative 28,20,16 (predicted
1409/1187/1196) and measured 922/816/904: the model is optimistic by
~400 MiB per card, uniformly. So the REAL ceiling is below +9%, and the
honest reading is that **the current layout is already within ~10% of
what this rig can hold at the corridor floor.** Do not spend another
shift chasing the hint; if the pool must grow materially it needs the
PP3-phase KV layout rebuild (#297/#635) the spec authorises, not vector
tuning.

## 3. State at end of shift

HEAD `76e93a62a6` plus the §1 fix (committed on top). Suite **540
passed**. SERVING HEALTHY on the PROVEN configuration — `RANK_MIB=
22700,11920,11970`, vector 28,26,20 (the prod_boot defaults),
`expandable_segments:True`, POLICY=auto, SPEC=on, graphs on,
`max_total_num_tokens=459392`, health 200. That is the run-4 config whose
corridor HELD at 1034/2824/1228 with 87/87 requests and 0 aborts.

Prod boot defaults were NOT changed by this shift's experiments; the
crashed configurations exist only as evidence logs
(`docs/dev/631/evidence/s11_run5..7_*`).

## 4. Untouched, and why

The bs1-spill leg and the YaRN leg were not started: the capacity
experiments consumed the shift, and each crash cost a boot plus a 390 s
measurement. They are unblocked — nothing found here stands in their
way — and should be taken FIRST next shift, before any further capacity
tuning, since §2 says the remaining capacity upside is small while those
two legs are entirely unmeasured.

Also still open: the A-vs-A gate vs `9a929352c9`, `PROD_BRINGUP_BENCH.md`,
and the one unmanned log carrying everything at once.
