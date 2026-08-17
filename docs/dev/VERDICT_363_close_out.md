# VERDICT — #363 Dynamischer Regime-Regler, close-out (2026-08-17)

Verdict: **DELIVERED as an instrument and an actuator chain; DORMANT BY FLAG on
today's config; one axis SUPERSEDED. Closing.**

Read with `DESIGN_363` §21 (the remainder determination), which currently lives
on branch `feat/363-remainder` and is **not** on `train/0817-desk` — the train
carries `STUB_363_584_verdicts.md` instead. **#706-Lane: `feat/363-remainder`
should be in the candidate list, or §21 is lost from the train.**

## 1. What was verified at code, on today's HEAD

`train/0817-desk`. Anchors re-checked by symbol; **no drift**:

| §21 claim | today |
| --- | --- |
| `MsStageDecider` live, `decide` at `regime_ms_clock.py:718` | holds, same line |
| pack/unpack group stat, MAX total / MIN wait | holds, `regime_ms_clock.py:188` / `:216`, contract stated verbatim at `:222-233` |
| the clock consumes a GROUP statistic, never the rank-local number | holds |

## 2. The open determination: does the decider have an ACTUATOR?

**Yes, and the chain is whole.** This was the question worth asking, because
#578 found the neighbouring seam looking wired and producing nothing
(`build_regime_stage_table` calling `planner_candidates` without `solve_fn`;
`AUDIT_421_UNWIRED.md` B.9). That was the FEED half. This is the other half:

```
MsStageDecider.decide                       regime_ms_clock.py:718
  -> RegimeObserver._intra_phase_decide     regime_runtime.py
  -> overrides `target`  ("target = proposed")
  -> if self._mode == MODE_ACT and target is not None
  -> _act_interlocks (veto path)
  -> self._commit_fn(target, ...)
  -> RegimeActuator.apply                   regime_act.py:152
       _vram_apply      -> kv_capacity_runtime.apply_budget_request   (#330 dial)
       _reshard_arm     -> kv_reshard_runtime.arm                     (#297)
       _phase_flip_arm  -> scheduler.arm_phase_flip                   (#631)
```

Every axis binds to a real runtime or is left `None` — never stubbed — so an
unwired axis produces a refusal that names the missing flag
(`--kv-reshard-vectors`, `--enable-vram-dial`) instead of half-moving in
silence. `build_regime_actuator` logs `wired axes ... or NONE (every proposal
will be refused)`.

**So this is NOT the #578 shape.** The distinction matters and is the reason
the question is asked every time: #578 was a broken path; #363 is a whole path
that is switched off.

### What "switched off" means, precisely

On `deploy/turnkey/stack.rig3.toml` this rig passes `--enable-phase-flip` but
neither `--regime-stage-clock` nor `--regime-controller`. Therefore:

* no `--regime-stage-clock` -> `stage_clock = None` -> **the decider is never
  constructed**; no proposal is ever made;
* no `--regime-controller act` -> no `commit_fn`; observe holds no actuator
  path by construction, and that property has its own pins;
* `act` is additionally refused at parse time without the 4-item evidence file
  (`regime_stages.EntryGate`).

Arming it is a serving-path decision — review and boot — and is deliberately
**not** done here.

## 3. Superseded

The `#297` KV delta-move at the phase boundary is superseded on both axes per
§21.2: the weight axis is dead by pricing (#704a established `PhaseFlipStacks.
refill` as the actuator that exists), and the KV axis is carried by the #656
flip controller. #363 does not need to deliver it.

## 4. Residue, named

1. **The controller has never run in `act` on metal.** §11.7's card gates are
   the remaining work, and they are a window, not a desk task.
2. `--regime-stage-clock` is desk code by its own admission
   (`regime_ms_clock.py:141-144`, `docs/dev/363/TICKET_363_STAGE_CLOCK.md`).
3. The bootstrap deadlock #363 defect 7 fixed (canon needs observe rows that
   only act mode used to produce) is fixed in code and unexercised on this rig,
   because the flag is off.

## 5. What this cut adds

`test/registered/unit/managers/test_regime_decide_to_act_chain_363.py` — a
ratchet on the chain above, link by link, plus the "unwired axis is `None`, not
a stub" property and its falsifier. It does **not** claim the controller runs.
It exists so that if a future edit breaks a link, #363 does not quietly return
to the state #578 found, with the next reader re-deriving all of this to notice.

Can-fail proven: removing `target = proposed`, and stubbing the #330 dial call,
each turn it red.
