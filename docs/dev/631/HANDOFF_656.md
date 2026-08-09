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
