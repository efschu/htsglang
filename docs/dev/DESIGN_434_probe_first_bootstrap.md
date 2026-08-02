# #434 cut 2 — probe-first bootstrap on unknown hardware (design note, not built)

Slice 1 of #434 covered three of the four cuts (default-behavior honesty, the
constant audit, the generality proof suite). This note specifies the fourth,
which is a build and is deliberately not in that branch, plus the standing
coupling to #363 that it enables.

The directive it serves: *the optimum per task must always be selected
automatically by the planner — for every hardware combination, model and quant
format; nothing tailored to this rig.* Slice 1 established that the solver's
answer follows the profile. This cut is about the sentence before that one:
**where the profile comes from on a machine nobody has measured.**

## 1. The gap, stated exactly

`--rank-tp-ratio auto-performance` already probes and caches. `get_hardware_profile`
(`python/sglang/srt/uneven_perf.py:1748`) reads a cache keyed by
`(sorted GPU UUIDs, driver, PROFILE_VERSION)` — `profile_cache_path`, same file
— and on a miss runs the stage-0 micro-probe in an isolated subprocess and
writes the result. That is already fingerprint-scoped caching, and it already
works on hardware nobody has seen.

The gap is on the other side of the flag:

1. **The capacity-first default never probes.** Plain `--rank-tp-ratio auto`
   derives budgets from NVML totals and stops. After slice 1 it at least
   *names* the optimizer (`_CAPACITY_FIRST_DEFAULT_NOTICE`,
   `python/sglang/srt/server_args.py`), but a first-time user on unknown
   hardware still has to know to re-launch with a different flag to get an
   optimized plan.
2. **Other consumers each bootstrap separately.** `memtier` bootstraps from
   live facts with every cost `ABSENT` naming its probe
   (`python/sglang/srt/memtier/fingerprint.py`, `memtier/adapters.py`); the
   `#213` card probe (`sglang.srt.rigmon.card_probe`) writes its own artifact;
   the planner's cost model reads whichever of these it finds
   (`planner/cost_model.py:901`, `_cached_card_probe`). Three artifacts, three
   freshness rules, one machine.
3. **A miss is silent in one direction and loud in the other.** The
   auto-performance path refuses by name when it cannot resolve the CUDA↔NVML
   bridge (#397, no emulation fallback). The capacity-first path has nothing to
   refuse — it simply plans without measurement, which is correct for what it
   claims to be and wrong as the thing most boots actually run.

## 2. Shape of the build

**One bootstrap, keyed by the #407 fingerprint, feeding every consumer.**

```
boot
 └─ hardware_profile_for_machine()            <- new, the single entry point
      ├─ fingerprint the box (memtier/fingerprint.py: hardware_key / model_key)
      ├─ EXACT match in the local store  -> use it whole
      ├─ MODEL match                     -> use card templates ONLY
      │                                     (no host, filesystem or remote row
      │                                      — licensed_document() already
      │                                      enforces exactly this)
      ├─ no match, probing allowed       -> short probe, store under EXACT
      └─ no match, probing refused       -> every cost ABSENT, naming its probe
```

The scope rule is not new and must not be re-invented here: `MatchScope.EXACT`
licenses every tier, `MatchScope.MODEL` licenses device-model templates and
nothing else, `MatchScope.NONE` licenses nothing
(`memtier/fingerprint.py:237-345`, `licensed_document`). A GEMM lane rate is a
property of a card model; a host-RAM bandwidth is a property of one box. The
same distinction decides what a bootstrap may reuse from a neighbour machine.

**Probe budget.** The stage-0 card probe measures in seconds, not minutes
(#213). A probe of that order is affordable once per fingerprint per
`PROFILE_VERSION`, amortized over every subsequent boot of that machine, and
that is the argument for making it the default rather than an opt-in. Two rules
keep it honest:

* the probe is **skippable and its skip is loud** — `SGLANG_PERF_PROBE_*`
  already carries this vocabulary, and an unmeasured lane must stay ABSENT
  rather than inheriting a neighbour's number;
* the probe's own timing is **recorded in the artifact**, so the cost of the
  default is a number in the log and not a claim in a design note. Slice 1
  deliberately quotes no probe duration for this reason: the figure in the
  ledger was measured on the development rig and is exactly the kind of
  transferred constant this task exists to remove.

**GEMV before exponents.** #231's precedent stands: where a probe can measure
the quantity directly, the probe replaces the fitted coefficient. The bootstrap
should therefore prefer measuring a decode-shaped GEMV over inheriting the
residual exponent, and the audit
(`docs/dev/AUDIT_434_planner_constants.md`) records which coefficients still
have no probe behind them.

## 3. What changes at the flag surface

Three options, in increasing order of behavior change. The recommendation is B.

| | behavior | risk |
|---|---|---|
| A | keep `auto` byte-proportional; probe lazily in the background and only *report* what `auto-performance` would have solved | zero plan change; the operator still has to act |
| **B** | **`auto` bootstraps a profile and prints the solved companion vector as a PIN HINT; the installed plan stays capacity-first unless the operator opts in** | boot time grows by one probe on a cold fingerprint; no plan changes without an explicit flag |
| C | `auto` installs the solved vector | changes every existing boot's plan; needs its own campaign and a way back |

B is the honest middle: it removes the "you had to know the flag existed"
failure without changing what any current command does. C is a defensible
end-state but is a separate decision with its own evidence bar, because the
weight split is not runtime-movable (see §4) and a wrong default costs a
restart.

## 4. The standing coupling to #363

`docs/dev/ANALYSE_363_dynamic_regime_controller.md` establishes the actuator
table: the KV token vector, the per-card VRAM budget, the spec algorithm/k and
lane placement all move at runtime; **the weight (MLP/GEMM) shard cut does
not**. The controller's action space is the movable rows.

That makes the coupling between the two halves precise:

* **This cut owns the cold half.** The profile a machine is planned from, and
  the fingerprint it is cached under. Everything the #363 controller decides at
  runtime is a choice *within* a plan the bootstrap made possible.
* **#363 owns the warm half.** It watches per-rank ms/round and flips between
  discrete, pre-solved stages. Its stages have to come from the same solver and
  the same profile, or the runtime will be optimizing against a model the boot
  did not use.
* **The seam is the artifact, not a function call.** The controller should read
  the same fingerprint-keyed profile, and it should be able to write back: a
  live ms/round measurement is a better datum than a boot-time probe estimate,
  and feeding it into the store closes the loop that #231 opened for the GEMV
  rate. That write-back is what turns "the planner is general" into "the
  planner gets better on every machine it runs on".
* **What does NOT couple:** the weight vector. A regime flip cannot move it, so
  the phase-optimal recipe (#354/#357) stays a per-boot decision until a
  runtime weight mover exists. Naming this here so a future reader does not
  wire the controller to a lever it does not have.

## 5. Follow-up tasks this note proposes

1. `hardware_profile_for_machine()` — the single fingerprint-scoped bootstrap
   entry point, with the EXACT/MODEL/NONE scope rule reused, not re-derived.
2. Option B at the flag surface: `auto` bootstraps and prints the companion
   solve as a PIN HINT; no installed-plan change.
3. Unify the three artifacts (auto-performance profile, `#213` card probe,
   memtier registry) behind that entry point, with one freshness rule.
4. Probe-duration recording, so the cost of a probe-first default is a measured
   number per machine rather than a quoted one.
5. #363 write-back: live per-rank ms/round measurements land in the same store
   the boot planner reads.
