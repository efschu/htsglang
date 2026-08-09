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
