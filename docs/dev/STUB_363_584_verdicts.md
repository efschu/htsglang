# STUB — parked verdict material for #363 and #584

PARKED mid-work under the bugs-first order (2026-08-17). Not a verdict. No
conclusion here is finished; the absorption mapping that both verdicts turn
on is incomplete. Resume from here rather than from scratch.

Both qwen evidence sweeps were stopped by the same order before either
returned findings, so everything below is my own evidence (git history,
FEATURE_CATALOG, the ticket files), not theirs.

## #363 — dynamic regime controller

**Do not assume SUPERSEDED.** The item's own ACT window verdict R3
(`a73a0d8da8`, 2026-08-14) POSTDATES the #656/#657/#658 commits (all
2026-08-11), so it was already assessed with those successors in tree and
still did not close.

Its refusal ladder ran to the end (quoting `a73a0d8da8`): "the planner could
not solve" (R13) -> "solved but carries no measurement" (R14) -> "HAS a
measurement and the measurement says no" (R15).

The controller's own staged rollout (`DESIGN_363_regime_controller.md`
section 6) is the right frame for the verdict:
- v1 observe-only — SHIPPED (`managers/regime_classifier.py`,
  `regime_runtime.py`, `--regime-controller observe`).
- v2 reachable half, VRAM grow + KV token vector — SHIPPED as `act`
  (`RegimeActuator`, `managers/regime_runtime.py:721-727` from
  `managers/scheduler.py:3401`, issuing a #330 budget GROW and a #297 reshard
  via `managers/regime_act.py:182-189`).
- v3 weight mover — NOT BUILT. Per `FEATURE_CATALOG.md:439`: no pointer flip,
  no diff spill, no pre-capture; gated on #286.

**The successor scope floated by the coordinator (per-rank ms/round-driven
ladder) already exists and already answered.** `regime_ms_clock.py` is in
tree, and `docs/dev/363/TICKET_363_ACT_VERDICT_R3.md:357` records criterion
A3 as **ANSWERED, NEGATIVE**: gain -3.16 % inside a 4.43 % band. Criterion A1
(`stage_clock_proposals > 0 and actuations > 0`) is FAIL at 0 and 0, and R3's
own point is that this is now a MEASURED refusal, not broken machinery.

So the likely honest verdict is neither SUPERSEDED nor "build the ladder":
the ladder is built, wired, and refused on the merits because this rig has no
stage whose gain clears its own noise band. Remainder candidates, unranked:
(a) v3 weight mover, gated on #286; (b) a signal-availability question — find
or construct an operating point with a real signal, or reduce the 4.43 % band.

**Absorption mapping — INCOMPLETE, this is the gap.** Working hypothesis only,
needs the file:line pass that did not happen:
- #297 reshard actuator and #658 VRAM budget marriage look like the two v2
  EDGE ACTUATORS #363 consumes (section 4.4), i.e. dependencies rather than
  supersessions.
- #657 pressure ladder / CorridorGuard is #287 lineage and keys on MEMORY
  PRESSURE, a different control input from #363's load shape.
- #656 is the one with genuine overlap: it auto-switches, and
  `phase_policy.break_even_tokens` (`managers/phase_policy.py:423`) keys on a
  token count, which IS a load-shape signal. But it moves ONE axis (phase
  layout) on a break-even rule, where #363 selects a whole pre-solved
  configuration on a signal-vs-band rule. Whether that is absorption or a
  neighbouring mechanism is exactly the undone work.

## #584 — planner sole authority for VRAM

Three pillars: one authority / auto-measurement / dynamic recalculation.

- **Auto-measurement: looks DELIVERED.** `c9ffa2a342` records the first
  end-to-end `--pp-solve-cut` run on this rig, writing measured rates to disk
  keyed by GPU UUID (5090 203.57 TFLOPS / 1661.7 GB/s; the two 3080s 51.14 and
  50.81 TFLOPS), correctly throttled to this rig's deliberate power caps. It
  also caught the borrowed rates as STALE: bandwidth reproduced exactly, GEMM
  was 12-22 % off after the 2026-08-05 power-target change.
- **One authority: mechanically ENFORCED, which is stronger than nominal.**
  `e0dce86ea1` found a derivation reading module CONSTANTS instead of the
  operator's measured knobs, producing a ladder that "looked solved and was
  not". `derived_provenance` now marks a value SOLVED only if inputs were
  supplied AND the result differs from what defaults alone produce; wired
  through `weakest`, a DEFAULTED input drags the whole boot verdict down.
- **Dynamic recalculation: UNRESOLVED — the open pillar, and the one to do
  first on resume.** Not established whether any budget is recomputed after
  boot. `38c1161fd4` ("Ask the arena what it can hold, instead of trusting
  what we remembered") is the most promising lead for a runtime re-query.
- #584's governing window answer is likewise a REFUSAL, driven by memory
  contiguity/packing, not by a missing authority: the aggregate FITS by
  8716 MiB and the solve still fails.

## Loose end found on the way

`FEATURE_CATALOG.md:426` cites a 4.2 % A-vs-A floor. R3 retracted that figure
and records 4.43 % as the honest floor (the earlier 0.03 % was a staleness
artefact). The catalog's conclusion survives either number, since its 2.0 %
tolerance still only breaks ties — so this is a citation fix, not a behaviour
question. Not fixed here; parked with everything else.
