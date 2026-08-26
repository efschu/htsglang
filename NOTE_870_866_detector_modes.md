# NOTE #870 + #866 — a detector whose signature a new operating mode made ambiguous

Runner: R8. Tree: `/spinning/wt-870-detector`, branch `probe/870-detector-modes`,
base **d7be60817c** — the same commit the live #857 instance is running from
(`/proc/851289/cwd` -> `/spinning/wt-861c`, `PYTHONPATH=/spinning/wt-861c/python`),
so the code read here is the code that fired.

**Every test run in this note used `CUDA_VISIBLE_DEVICES=""`**, verified on the
process. **The #857 instance was never touched**: no restart, no load driver, no
request, no card. Everything measured from it is a READ of
`/spinning/evidence-665-f1/boot_w40_857strict_0825_2342.log` and of `/proc`.
`nvidia-smi --query-compute-apps` showed only PIDs 851455/851456/851457
throughout.

---

## RESULT IN ONE BLOCK

    class            a detector's signature was made AMBIGUOUS by a legitimate
                     new operating mode -- second instance, after #739
    root             TWO defects, and the second survives fixing the first:
                     (1) `idle_locked_seen` was never passed by the only
                         production caller -- unwired since it was written
                     (2) even when true it decorated the MESSAGE, never the
                         verdict
    #866             DOWNSTREAM of the same root, no separate fix needed:
                     recovery returns None before allocating anything unless
                     `alarm` is true (invariant_checker.py:955)
    fix              a third suppression term in #739's idiom: NAMED hold AND
                     ARMED flip AND bounded age. Any one missing -> alarm.
                     Over the bound -> still alarms, but as a STUCK FLIP.
    tests            12 new, hermetic, 0.1 s; wedge family 151 passed +
                     92 subtests, 0 failed. The 17 true hits untouched.
    NOT measured     the fix has never run on a card. No boot, no restart.

---

## 0. PRIOR ART — and it changed the fix twice

### 0.1 This was already filed, and I nearly re-derived it

`docs/dev/RESIDUAL_739_admission_wedge_precision.md` (filed 2026-08-24 from
window W22, pin `e5a37866d7`) records **7 false alarms of this exact class**,
and calls it in as many words: *"NOT a run blocker, NOT a wedge. A precision
defect in a detector."* The operator called this the second instance of the
class; the register is right, and the residual is the first WRITE-UP of the
second instance. Nothing had landed for it since.

**It also proposes a discriminator I did not use, and it is a good one:** *"THE
DISCRIMINATOR IS THE AGE TREND, not the alarm's presence."* A falling age means
the progress clock was reset, i.e. work completed; a real wedge climbs
monotonically (`127.4 -> 218.2s` and never recovers).

I used the hold REASON instead, for two reasons worth stating rather than
assuming: the trend needs history the detector does not keep (it is a
single-shot verdict over three numbers), and a reason is CAUSAL where a trend
is statistical — the reason says why the wait is legitimate, which is what the
operator's specification asks the alarm to read.

**But the residual's discriminator is a free second opinion, so I ran it.**
PP0's age sequence across the #857 boot:

    29.6  39.0  182.6  176.1  186.1  22.3  45.3  51.6  50.8  50.3  49.3
    48.3  28.1  46.2  50.7  50.1  49.2  48.8  47.8  47.9  47.5  46.5

Non-monotonic, with outright RESETS (186.1 -> 22.3, 48.3 -> 28.1) and long
falling runs. By the residual's own test these are false positives. **Two
independent methods — one causal, one statistical — agree.** That is worth more
than either alone, and it is the reason this note claims the diagnosis rather
than proposing it.

### 0.2 The general mechanism already exists, and is wired to nothing

`python/sglang/srt/managers/progress_liveness.py` defines exactly this concept,
generically:

    :82   #: Deliberate pause: a phase flip, a maintenance hold, a GPU-arb claim.
    :84   inhibited: bool = False
    :85   inhibit_reason: str = ""
    :195  if any(s.inhibited for s in win): ... "a flip or maintenance hold
          stops the counters legitimately."

**`grep -rn progress_liveness python/sglang/srt/` outside its own file returns
nothing.** No caller, anywhere. So the general answer to "the system is
legitimately paused" was designed, named, documented — and never connected to
the detector that needed it. The 41 alarms are the cost of that disconnection.

Consequence for this ticket, stated plainly: **converging the two is the
structural fix and it is NOT done here.** It replaces this detector's whole
input path with a windowed sampler, on a shipped line, with no card to validate
it on. What I did instead is name the convergence at the call site
(`invariant_checker.py`, the `#870` comment) and choose parameter names that
make it a rename rather than a redesign.

### 0.3 #739's mechanism CANNOT simply be extended — checked, because I was told to

The instruction was: if #739's mechanism only needs extending rather than
duplicating, that is the fix. It does not extend. `admission_wedge_verdict`
hardcodes two named clocks as two keyword parameters with two inlined `if`
branches; there is no list, no registry, no plugin point. Adding a third
legitimate state means a third parameter, a third `getattr`, a third branch —
by hand, in the same style.

So the honest position: **this is a third duplication of a pattern that should
be a registry, chosen deliberately over the two alternatives** (build the
registry now, or wire `progress_liveness`), because both alternatives rewrite a
shipped detector's input path and neither can be validated without a card
tonight. The debt is recorded in §5, not hidden.

---

## 1. THE MEASUREMENT — what actually fired, with a two-sided tally

    grep -c "ADMISSION-WEDGE"  ->  53

    53 = 41 detector alarms + 12 recovery lines
    41 alarms over 23 distinct timestamps
    per rank: PP0 23 + PP1 15 + PP2 15 = 53   (tally: 53 == 53)
    12 recovery = 2 episodes x 3 ranks x 2 lines (post + NOT APPLICABLE)

The operator measured 44 at 00:08Z; it was **53 by 00:15:47Z and still
growing** — the alarm was ongoing, not a startup transient.

**The corroboration channel was silent in all 41:** every alarm ends
`"no phase-policy corroboration seen -- the wedge class is broader than that
path"`. And in the same second as the first one, the phase policy logged:

    23:46:26  PHASE-POLICY arming pp_to_tp: IDLE-LOCKED: no batch of either
              work class can be built in the pp layout (8 req resident,
              0 tok prefill pending)

**Eight requests. The same eight the detector called a wedge.** The reason
existed, was computed, was logged, and did not reach the detector.

---

## 2. THE ROOT — two defects, and the second survives fixing the first

**(1) `idle_locked_seen` was UNWIRED.** It is a parameter of
`admission_wedge_verdict` (`invariant_checker.py:554`), documented at `:588`,
and covered by tests (`test_admission_wedge_detector_699.py:79,84`). The only
production caller, `check_admission_wedge_once`, passed three arguments and
never it. **It could not be True on a live boot, whatever the phase policy
did** — which is exactly why every one of the 41 lines says the corroboration
was missing.

**(2) It was advisory only.** Even when True, the parameter appended a phrase
to the message. The verdict never read it. So wiring alone would have produced
41 better-worded false alarms.

This is the shape #421's unwired-feature class exists for: a signal that is
defined, documented, and tested, and that no production path feeds.

---

## 3. THE FIX

**Publish** (`scheduler.py`, at the one line that already classifies an arm as
idle-locked): stamp `last_layout_hold_time` and `last_layout_hold_reason` when
the arming reason starts with `IDLE_LOCKED`. Published there rather than in a
new subsystem because a second place deciding the same thing is a second thing
to keep in sync — the argument `is_armed` itself makes against mirrored flags
(`phase_flip_runtime.py:6350`). Deliberately NOT cleared on the other branch:
the stamp is an EVENT TIME, and clearing it would make "how long has this hold
run" unanswerable across the re-arms a long hold is made of.

**Read** (`invariant_checker.py`): `flip_armed` from
`phase_flip_runtime.is_armed()` — "#631: the one authority for arming,
deliberately not a mirrored flag" — plus the two stamps, all via `getattr` with
a `None` fallback, so a scheduler stand-in carrying none of them produces the
pre-#870 verdict unchanged.

**Decide**: suppress only when **a NAMED hold AND an ARMED flip AND
`hold_age < ADMISSION_WEDGE_HOLD_GRACE_SECONDS`** (= 3x the alarm threshold).

    hold reason, no armed flip   -> a hold nothing is ending IS a wedge -> alarm
    armed flip, no named reason  -> an arm this detector cannot attribute
                                    must not be trusted to explain silence -> alarm
    signals absent               -> pre-#870 verdict exactly -> alarm
    over the grace               -> ALARM, but as a STUCK FLIP:
                                    "Look at the flip, not at admission"

That last line is the point. The grace is **not a safety margin, it is the
boundary between two different defects.** Below it the wait is specified
behaviour; above it the flip has stopped making progress — still a real fault,
still alarmed, but named correctly instead of sending the next hunt at
admission the way this one was sent.

---

## 4. #866 — one root kills both, and the link is structural

The recovery actuator posted corridor-relief requests and the gate answered
`exit 'headroom-sufficient'`: it looked for a VRAM shortage
(`corridor_admission.py:790`, `free - want >= guard.floor_bytes`) and correctly
found none, because **a layout hold is not a shortage.**

It ran at all only because the alarm had stood past the recovery threshold, and
`AdmissionWedgeRecovery.step` returns `None` before allocating anything
whenever `alarm` is false (`invariant_checker.py:955`). So a detector that no
longer alarms on a legitimate hold cannot post a relief request for one either.
**#866 needs no separate fix.** Pinned end to end
(`OneRootKillsBothTicketsIncluding866`), not argued.

---

## 5. THE SIBLING SWEEP — who knows the hold is legitimate?

| detector | file:line | knows about the hold? |
|---|---|---|
| #699/#739 admission wedge | `invariant_checker.py:549` | **NO** — fixed here |
| #866/#800 wedge recovery + escalation | `wedge_recovery.py:214`, `corridor_admission.py:790` | **NO** — measures VRAM; fixed by inheritance |
| #604 turnkey crash watchdog | `turnkey/watchdog.py:243` | **NO, by inheritance** — consumes the #799 published verdict |
| #821/#824 PP blocked-receive | `invariant_checker.py:1178` | **NO** — bare 300 s compare |
| #838 layout conformance (economy) | `layout_conformance.py:432` | **YES** — the only one that checks |
| #677 `layout_hold_verdict` | `phase_policy.py:176` | it IS the hold, bounded in ROUNDS not seconds |

**There is no shared helper.** `layout_conformance`, `phase_policy` and
`phase_purity` each hand-write their own notion of "is this hold OK", and the
admission-wedge family calls none of them. That, plus §0.2's unwired
`progress_liveness`, is the class's real habitat.

### 5.1 An exposure I checked before reporting, and it was NOT armed

#604 consumes the published wedge verdict and, at `wedge_confirmations = 3`
with a 20 s poll, asks systemd to **restart the whole serving unit**. On this
boot the alarm stood continuously for 182.6 s — far past three confirmations.

**But it was not armed here, and I checked rather than asserting it:** the log
contains zero `wedge_status`/`ADMISSION-WEDGE-UNRECOVERED` lines, no wedge
status file exists, and no turnkey watchdog process is running. This was a
manual acceptance boot, not the autoboot path.

So the correct statement is: **the false alarm is one armed watchdog away from
restarting a healthy instance, and on this boot that watchdog was not running.**
Not "it nearly restarted the box" — which is what the numbers alone would have
supported, and which would have been wrong.

---

## 6. WURZEL-VOR-WIRKUNG

**(1) CLASS.** A detector whose signature was made AMBIGUOUS by a legitimate
new operating mode. The signature `queued>0, running 0, no first token` meant
exactly one thing when it was written and now means two, and the detector reads
the numbers without the reason. Second instance after #739 (whose blind spot
was "prefill counts as progress"); first written up in RESIDUAL_739.

**(2) SIBLINGS.** §5 — six detectors, one of which (#838) already does it
right, four of which do not, and no shared helper. Swept from source, not
guessed. The unwired `progress_liveness.inhibited` (§0.2) is where the shared
helper was supposed to be.

**(3) CHECK.** The detector now reads the hold REASON, not only the clock, and
the reason must be corroborated by an armed flip and bounded in time.
Can-fail proven in BOTH directions, as specified: without a readable hold the
old alarm still fires (3 separate tests, one per missing signal), and a real
wedge is still caught. A hold that outlives its window still alarms — with a
different, accurate name.

---

## 7. WHAT I DID NOT MEASURE

* **The fix has never run on a card.** It is proved hermetically to suppress
  the #857 signature and to keep firing in five separate degraded-signal cases.
  Whether the live boot goes quiet is one card run, and it is owed.
* **`last_layout_hold_time` is stamped on the ARMING path only.** If a hold can
  exist without an idle-locked arm, that shape is unstamped and will still
  alarm — the conservative direction, and untested because I have no specimen
  of it.
* **The convergence with `progress_liveness`** (§0.2) is designed for, not done.
* **#821's 300 s SIGQUIT exposure** (§5) is reported, not fixed: a hold bounded
  in ROUNDS has no guaranteed second-bound under 300 s. No specimen observed.
* **Nothing outside the wedge family was run.** No claim about the rest of the
  suite.
