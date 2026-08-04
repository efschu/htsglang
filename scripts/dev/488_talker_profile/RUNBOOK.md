# #488 precursor — turnkey runbook

Status: **READY**. Not run. The one in-tenant call is timed by the coordinator,
not by this agent — the user is testing live.

## What this answers, in one line

Of the **5511 ms** a turn spends between TTS start and first audio (91 % of a
6076 ms turn), how much is kernel work and how much is launch/Python overhead?
That number decides whether lever (1) — the raw predictor loop plus two CUDA
graphs — is worth 1.5-2.5 days, or whether the whole redirect in
`ANALYSE_488_talker_lane_layout.md` is wrong.

## Preconditions, all checked by the script itself

| precondition | value | enforced at |
|---|---|---|
| free VRAM on the card, after our transient | ≥ 400 MiB | `check_headroom`, refuses with the reason |
| calibration transient | 96 MiB (two 4096² fp16 operands + output) | `_CALIB_MIB` |
| per-arm wall deadline | 60 s | `_ARM_DEADLINE_S` |
| whole-run deadline | 360 s | `_RUN_DEADLINE_S` |
| instrument may testify at all | calibration arms separate ≥ 0.40 **and** the GPU-bound arm stays under a 0.35 gap | `check_discrimination` |

Measured on 2026-08-04: 5090 total 32607 MiB, rank 0 (pid 3953294) 22436 MiB,
translator tenant (pid 3954713) 5910 MiB, **3605 MiB free**. The 96 MiB
transient leaves ~3509 MiB — comfortable. That is a fact about that moment, so
the script re-reads it every run and refuses rather than assuming.

## The call

In the tenant process (pid 3954713), against the module it already holds. No
second load, no new process, no server restart, nothing written to conversation
state, no audio synthesised:

```python
import sys, json
sys.path.insert(0, "/spinning/wt-488-talker-lane/scripts/dev/488_talker_profile")
from profile_talker_steps import profile_loaded_model

report = profile_loaded_model(tts._model)        # the InProcessQwen3Tts wrapper
open("/spinning/gpu-battery-results/488_precursor.json", "w").write(
    json.dumps(report, indent=2)
)
```

`tts` is the `InProcessQwen3Tts` instance the launcher built
(`srt/translator/inprocess_tts.py`). The wrapper is reached through
`tts._model`; the script raises by name if it is handed anything else.

**Best moment:** immediately after agent 8's restart bundle brings the tenant
back and before the user resumes. The talker is warm, idle, and nobody is
mid-turn. Cost then is ~6 min of an idle talker rather than ~6 min of a busy one.

## Identifying the process during a VRAM triage

The standalone path renames itself **`sglang::488-talker-profile-GUEST`** in
both `/proc/<pid>/comm` (truncated to `sglang::488-tal`) and
`/proc/<pid>/cmdline`, so `ps`, `top`, `py-spy` and `nvidia-smi` all attribute
it on sight. "GUEST" is deliberate: it answers the second triage question —
this process is a visitor on a card it does not own, and it is expected to
disappear on its own.

The in-process path needs no marker: it runs inside the tenant's own pid.

## Abort path

* **Self-limiting.** Every arm is bounded by iteration count *and* wall clock;
  the whole run is bounded at 360 s. There is no unbounded wait anywhere, so
  the worst case is that it finishes late, not that it hangs.
* **Interrupting it is safe.** The only state it creates is the 96 MiB
  calibration transient, freed per arm; it never mutates the model, the cache
  or the conversation. A `KeyboardInterrupt` mid-arm loses the measurement and
  nothing else.
* **If the user starts a turn while it runs:** let it finish (≤ 6 min) or
  interrupt. Do NOT kill the tenant — the profile is a guest in that process.
* **If it refuses:** that is a result, not a failure. Two refusals are
  possible and they mean different things:
  * `REFUSED -- only N MiB free ...` → the card filled up; retry later.
  * `REFUSED -- the known GPU-BOUND arm shows a gap fraction of ...` → the card
    is contended and **no verdict may be drawn**; retry when the talker is idle.
    This one exists specifically so a busy box cannot manufacture the
    overhead-bound answer we expect.

## Expected artefacts

One JSON document. The fields that decide anything:

```
headroom        : free_mib, calibration_transient_mib, corridor_floor_mib
discrimination  : ok, separation, the two calibration gap fractions
arms            : calib_gpu_bound, calib_launch_bound, trunk_step,
                  predictor_step, predictor_generate, frame (DERIVED)
                  -- each with wall_per_iter_ms, kernel_per_iter_ms, gap_ms,
                     gap_fraction, kernels_per_iter, sync_count
rtf             : measured_rtf, kernel_only_rtf, recoverable_factor,
                  plus ANALYSE_488's bandwidth floors for comparison
verdict         : one of three sentences
```

The three verdicts, verbatim from the script and each unit-tested:

* `OVERHEAD-BOUND confirmed` (gap ≥ 70 %) — lever (1) is the right cut, TP is not.
* `PREMISE FALSIFIED` (gap ≤ 30 %) — kernels really do account for the wall
  clock; ANALYSE_488's redirect needs revisiting and the TP row
  (`planner/rejected.py`, `tts_talker_tensor_parallel`, NOT_DEFAULT and
  therefore unlockable) gets reopened.
* `MIXED` — report the bands, do not pick a side.

## What this deliberately does NOT measure

The text-encode / token-gen / vocoder split of a real turn. Agent 8 is building
that instrument on the tenant side; duplicating it here would give two
instruments disagreeing about the same 5511 ms. This one answers a different
question — *within* the token-gen phase, kernels versus overhead — and the two
compose: his split says which phase to attack, this one says with which lever.
