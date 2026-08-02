# #433 -- letting the planner SOLVE the INT8-W8A8 prefill vector

Desk-only analysis (`CUDA_VISIBLE_DEVICES=99`, no `/spinning/gpu-arb` hold,
no GPU touched). Repo tip `d85d964423`. All commands and line numbers below
were actually executed / read during this task; none of this is inferred
from memory.

## 1. What #424 actually did (RUNSHEET.md), and what it did NOT do

`/spinning/gpu-battery-results/2026-08-02_424_phase_record_bench/RUNSHEET.md:66,68`:

```
| int8_prefill | --rank-tp-ratio auto --rank-mlp-ratio 10,1,1 | --rank-kv-ratio 2,11,10 | 4500,4200,4200 |
| fp8_prefill  | --rank-tp-ratio auto --rank-mlp-ratio 10,1,1 | --rank-kv-ratio 2,11,10 | 4500,4200,4200 |
```

Both formats were pinned to the **same** `--rank-mlp-ratio 10,1,1`, with
plain `--rank-tp-ratio auto` -- **not** `--rank-tp-ratio auto-performance
--rank-perf-tune phase-prefill`. The optimizer was never invoked for the
prefill arms; #424's own runsheet explains why (`RUNSHEET.md:92-97`):
FP8's genuine solve (`16,1,1`, from #354) breached the 400 MiB VRAM corridor
floor in practice (87 MiB free on the 5090), so both formats were pinned to
`10,1,1 + --rank-kv-ratio 2,11,10 + reserve 4500,4200,4200`, the pair task
#320 had previously booted and measured safe. `10,1,1` is real (it is one of
INT8's own candidates, see below) but it was **reused from a different
task's corridor-safety decision, at a context length (131072) and reserve
#424 introduced**, not re-solved for this operating point.

So the premise "10,1,1 was FP8's lopsided shape, never INT8's solved
optimum" is half right: `10,1,1` was **not FP8's** shape (FP8 solves to
`16,1,1`, a different vector -- see the fixture below), but it also was
**not INT8's optimum at the operating point #424 actually measured**. It was
a manual pin, and pinning skips the optimizer entirely.

## 2. Was an INT8 phase-prefill solve ever recorded? Yes -- #360

`/spinning/gpu-battery-results/2026-07-31_360_int8_quality/plan_c_int8_pref.txt`
is a **real boot log** (not a desk solve) from
`/spinning/htsglang-gpu/.venv/bin/python -m sglang.launch_server --model-path
/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8 --tp-size 3
--rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance --rank-perf-tune
phase-prefill --rank-auto-reserve-mib 5500,2700,2700 ...` (full command in
`pyspy_c_int8_pref.txt` of the same directory). It genuinely invoked the
optimizer and got:

```
CHOSEN MLP vector: 16,2,3 (materialized units [104, 13, 19]; predicted ctx
~406279 >= floor 406279; predicted per-rank capacity [83775, 147826, 174678];
predicted decode step +16.8% vs the VRAM-auto split)
```

-- a genuinely different, less-concentrated split than `10,1,1`
(units [113, 12, 11]), at context-length 32768, reserve `5500,2700,2700`.
This is legitimate: the "CHOSEN MLP vector" line only prints when the
optimizer's own admissibility gates accept a candidate
(`python/sglang/srt/uneven_perf.py:5820-5831`); it is not a forced pin.

## 3. #354's invocation pattern (located)

`/spinning/gpu-battery-results/2026-07-31_354_phase_optimal/run_arm.sh` +
`solve_vector.py` in the same directory: #354 predates `phase-prefill` /
`phase-decode` as enum values (those were added by #357, see
`uneven_perf.py:4833` docstring: *"`--rank-perf-tune` targets that plan the
PHASE-OPTIMAL recipe (#357)"*). #354 booted with plain `--rank-perf-tune
enc`, which REJECTS every concentrated candidate that trips the decode-knee
guard. Its `plan_int8_auto.txt` / `plan_fp8_auto.txt` show the optimizer's
own refusal line naming the **best forfeited candidate**:

```
tune=enc: enc has no effective lever at this operating point. 8 concentration
candidates evaluated, none accepted (unbootable 5, knee 3). The best of them,
10,1,1, would have predicted +9.1% prefill and is rejected by: unbootable.
Keeping the plain VRAM-auto split.
```

The `_prefopt` arms then **manually pinned** that forfeited vector
(`--rank-mlp-ratio 10,1,1` / `16,1,1`) at a raised reserve
(`4500,2700,2700`) to measure it anyway -- exactly the same "read the
refusal line, then pin" pattern #424 later reused from a different source
(#320's corridor-safe pair). `10,1,1` for INT8 and `16,1,1` for FP8 are
**independently solved values** (different lane ratios: INT8 3.68:1:0.90 vs
FP8 9.73:1:1.01, `lanes.txt` of the #354 directory), not one format's shape
copied onto the other.

## 4. Desk re-solve at repo tip `d85d964423` (this task)

**Live NVML/CUDA resolution genuinely cannot run here**, and this is by
design, not an oversight. Reproduced directly:

```
$ CUDA_VISIBLE_DEVICES=99 PYTHONPATH=.../python python solve_int8.py
...
ValueError: --rank-gpu-id names CUDA device(s) [0, 1, 2], which this host
cannot resolve to a physical card. ...
uuid=... nvml=0 cuda=- ... RTX 3080 ...
uuid=... nvml=1 cuda=- ... RTX 5090 ...
uuid=... nvml=2 cuda=- ... RTX 3080 ...
```

NVML sees the cards; CUDA enumerates none (`CUDA_VISIBLE_DEVICES=99`), and
`_resolve_rank_gpu_cards` (`server_args.py:596`) requires the CUDA<->NVML
bridge to place `--rank-gpu-id`. Per `docs/rig-runbook.md:2632`
("**One bridge, and no fallback (#397).**"), the emulation fallback that
used to paper over an unresolved CUDA order was deliberately removed:
*"`--rank-auto-reserve-mib` and the hardware micro-probe refuse outright"*
when the CUDA order can't be resolved. So the live boot-time solve path is
correctly, deliberately, un-runnable on this desk -- not a gap to route
around.

**The stored-profile path does exist**, and it is exactly how the repo's
own regression tests validate this machinery:
`test/registered/unit/planner/test_phase_optimal_targets.py` mocks
`uneven_perf.get_hardware_profile` with a static v3 hardware profile
(measured GEMM lanes, no live NVML) and calls
`uneven_perf.apply_auto_performance()` directly. Ran it at repo tip
(`12acb5dbce`, one commit ahead of `d85d964423`, diff is stash-rescue docs
only -- verified with `git diff --stat d85d964423 HEAD`, no `uneven_perf.py`
change):

```
$ HTSGLANG_TEST_MODEL_DIR=/spinning/llm_stuff/club-3090/models-cache \
  CUDA_VISIBLE_DEVICES=99 python -m pytest -q \
  test/registered/unit/planner/test_phase_optimal_targets.py
14 passed, 17 subtests passed in 5.13s
```

All 14 tests pass, confirming the shipped phase-prefill/phase-decode
machinery is intact and unchanged at tip. Reused the same mock to sweep
reserves and one extra operating point (script:
`/tmp/433solve/desk_solve.py`, `/tmp/433solve/desk_solve_424point.py`):

| operating point | tune | CHOSEN vector | predicted prefill gain |
|---|---|---|---|
| ctx 32768, reserve 4500,2700,2700 (#354's own reserve) | phase-prefill | **10,1,1** | +9.5% |
| ctx 32768, reserve 5500,2700,2700 (#360's reserve) | phase-prefill | **10,1,1** | +9.8% |
| ctx 131072, reserve 4500,4200,4200 (**#424's actual point**) | phase-prefill | **8,1,1** | +8.5% (10,1,1 is 2nd-best, +8.2%) |
| any of the above | phase-decode | plain VRAM-auto split (`rank_mlp_ratio=None`) | -- names the companion prefill vector in the log |
| FP8, ctx 32768, either reserve | phase-prefill | 16,1,1 | +22.9% / +23.5% |

Two things this table nails down:

1. **`10,1,1` genuinely is INT8's own solved phase-prefill optimum at
   #354/#360's context length (32768)** -- confirmed independently of the
   #354 forfeited-candidate log, by the current-tip optimizer running
   through the "phase-prefill" code path (advisory knee guard, binding
   fundability) rather than the old "enc" path. It is not FP8's shape
   (FP8 solves to a different vector, `16,1,1`, from the same lane fixture).
2. **At #424's own actual operating point** (context 131072, the reserve
   #424 introduced), the genuine optimizer's *best* candidate is **not**
   `10,1,1` -- it's `8,1,1`, by a narrow margin (+8.5% vs +8.2%, close to
   noise but a different vector). #424 pinned `10,1,1` without ever asking
   the optimizer this question at its own operating point.
3. `phase-decode` reproduces the expected #265/#354 finding at every point
   tested: plain VRAM-auto split, no MLP override.

**Caveat, stated plainly**: the frozen #354-window lane fixture used above
(`int8_native` 676.69 / 183.78 / 164.77 TFLOPS) does not reproduce #360's
live-boot `CHOSEN 16,2,3` -- #360's own log shows freshly-probed lanes of
690.3 / 188.8 / 189.3 TFLOPS (`plan_c_int8_pref.txt:4-6`), meaningfully
different from the frozen fixture (rank 2 in particular: 189.3 live vs
164.77 fixture, ~15%). GEMM-lane measurement is not perfectly stable
run-to-run on this rig, and the candidate ladder is sensitive to it. This
desk analysis reproduces the *mechanism* and the *general answer* ("a real,
context-dependent, format-specific lopsided vector exists and is close to
but not identical to `10,1,1`"), not a single frozen number good for every
boot.

## 5. Verdict: (b) -- a real lopsided vector exists; do not treat "no gain"
as settled, and do not hand-pin a stale number

The planner does **not** collapse to "plain/VRAM-auto" for INT8 phase-prefill
anywhere tested here -- every operating point above except `phase-decode`
solves to a real, admissible, non-trivial concentration (+8.2% to +9.8%
predicted, roughly half of FP8's +22.9%/+23.5%, consistent with INT8's
flatter 3.68:1:0.90 lane ratio vs FP8's 9.73:1:1.01). The one-layout
recommendation ("`10,1,1` loses on INT8, ship the decode layout only") is
**not confirmed** -- it was measured against a stale, borrowed pin at a
context length and reserve the vector was never solved for.

**Arm spec for the next GPU window** -- ask the optimizer, do not re-pin a
number from this note:

```
--tp-size 3 --rank-gpu-id <resolved via `python -m sglang.srt.registry.nvml --map`> \
--rank-tp-ratio auto-performance --rank-perf-tune phase-prefill \
--rank-auto-reserve-mib <R0>,<R1>,<R2> \
--kv-cache-dtype fp8_e4m3 --context-length <target> --trust-remote-code
```

at the actual context length and reserve the serving profile will run at,
and read the `CHOSEN MLP vector` line off that boot's own log --
`python/sglang/srt/uneven_perf.py:5824` prints it, plus a `PIN HINT` line
(`uneven_perf.py:5837-5841`) giving the exact `--rank-tp-ratio auto
--rank-mlp-ratio <chosen>` to skip the probe on repeat boots of the *same*
operating point only. For a companion decode boot, the same command with
`--rank-perf-tune phase-decode` names the paired prefill vector without
installing it (`uneven_perf.py:5805-5817`).

**Corridor caveat carried over from #424** (`RESULTS.md` section 3): the
`10,1,1` pin left only 286-316 MiB free on the 5090 at reserve
`4500,4200,4200` / `4500,2700,2700` -- inside the 400 MiB corridor floor.
The lower-concentration candidates this desk solve found admissible at the
same operating point (`8,1,1`: 7985 MiB residual free on rank 0 at ctx
131072/reserve 4500,4200,4200; `6,1,1`: 7270 MiB) clear the corridor with
large margin for a small predicted-gain cost (+8.5% / +7.3% vs `10,1,1`'s
+8.2% at that same point) -- worth checking on the real boot's own
`residual free` line before choosing which candidate to run, rather than
assuming the top of the ladder is the one to take.

## Coupling fix built, GPU confirmation pending (#435)

The #433 GPU arm ran and answered its own question, and in doing so exposed a
defect the desk analysis above could not see: **the phase-prefill solve picks
an MLP vector but leaves the coupled KV token vector on the VRAM-budget
split.** Full evidence in `/root/addendum_433.md`; the short form is that the
arm booted `--rank-mlp-ratio 8,1,1` (solved live, confirming this note's
prediction to 0.1 pp) with token vector `[31,17,16]` while the optimizer's own
matched vector was `[12,26,26]`. Rank 0 owned 48 % of the tokens holding 13 %
of the capacity, and `max_total_num_tokens` came out at **125 504** against
the **358 693** the admissibility gate had just accepted and the **431 360**
of the decode arm. The throughput comparison in the addendum is still valid as
measured (its probes fit inside 125 504 tokens), but that arm was not running
the configuration the optimizer thought it had solved.

**Built (desk, no GPU):** `_phase_solve_owns_kv_ratio` in `uneven_perf.py`
gates a new install at the end of `apply_auto_performance`: under
`--rank-perf-tune phase-*`, non-solo placement, and the default
`--rank-kv-ratio coupled`, the chosen candidate's own
`predict_capacity()["token_vector"]` is parked in `rank_kv_capacity_seed`,
which `resolve_cp_token_ratios` consumes above the budget estimate and below
`SGLANG_UNEVEN_TOKEN_VECTOR` and an explicit `--rank-kv-ratio`. The plan log
gains a `coupled KV ratio MATCHED to the solve:` line naming the vector.
Hermetic regression on a synthetic foreign rig:
`test/registered/unit/planner/test_phase_kv_coupling.py`.

Two limits of the fix, both deliberate and both things the confirmation arm
has to read off the boot rather than assume:

1. **The seed is the PREDICTED match, not the measured one.** #433's own
   numbers separate the two: `[12,26,26]` (from predicted capacity) funds
   355 958 tokens, `[8,28,28]` (from the capacity the boot actually measured)
   funds 464 342. Only the second clears the decode arm's 431 360. Converging
   onto the measured optimum is what `--rank-kv-ratio capacity` already does
   (phase-2 install after profiling), and the fix leaves that mode untouched.
2. **The fundability gate still prices the base-plan token vector.** Its
   reject term is capacity the vector does NOT use, which a matched vector has
   none of; re-basing it collapses every residual to the bare reserve and the
   gate stops discriminating -- measured on the #264/#354 fixture,
   `--rank-kv-ratio capacity` (which already prices per-candidate) accepts
   `16,1,1` at reserve `3000,2700,2700`, the configuration #264 booted into an
   OOM. Not widened here. The consequence for this arm: post-fix the
   non-binding ranks no longer keep the unused-capacity slack the gate credits
   them with, so **the corridor risk moves off rank 0 and onto ranks 1 and 2**
   -- #433 measured 7045 / 7109 MiB idle there, and matching spends it.

   **Closed by #437.** Limit 2 above is no longer the state of the tree: the
   gate now follows the vector the boot runs. A FIXED token vector (the plain
   coupled default, an explicit `--rank-kv-ratio` pin) keeps the relative
   base-plan pricing unchanged -- there the slack really does move with the
   candidate. A MATCHED vector (`capacity`/`speed`, and the phase arms via
   `_phase_solve_owns_kv_ratio`) is priced ABSOLUTELY instead: every rank's
   predicted residual free VRAM must cover the derived reserve demand, on ALL
   cards rather than only on the binding one, which is exactly the risk shift
   this note predicted. The plan log states which basis it used, and when the
   VRAM-auto split itself does not clear the demand it says so instead of
   quietly installing nothing. #330's 400 MiB corridor is priced next to the
   demand and REPORTED (`CORRIDOR-TIGHT`) rather than enforced -- #354's
   `16,1,1` booted and served at 87 MiB free, so a candidate between the
   demand and the corridor is a decision, not an error, and the operator is
   now told before the boot rather than after.

   Practical consequence for the confirmation arm below: at
   `4500,2700,2700` the phase-prefill solve now refuses every candidate on
   ranks 1 and 2 and says so, which is the same remedy this note's corridor
   gate already prescribed ("the retry raises `--rank-auto-reserve-mib` on
   those two cards"). Pin the reserve for all three cards above the derived
   demand before the arm runs.

### Confirmation arm spec

Same rig, same INT8-W8A8 checkpoint, same context and probes and transport as
`bench845_update_table.md`'s `int8_decode` row, so the comparison stays
one-variable. Resolve `--rank-gpu-id` at runtime
(`python -m sglang.srt.registry.nvml --map`), never from a fixed index.

```
--tp-size 3 --rank-gpu-id <resolved> \
--rank-tp-ratio auto-performance --rank-perf-tune phase-prefill \
--rank-auto-reserve-mib auto --context-length 131072 \
--kv-cache-dtype fp8_e4m3 --trust-remote-code
(no --rank-mlp-ratio, no --rank-kv-ratio)  + barlink BAR1
```

**Sub-arm A -- the fix does what it says.** Exactly the command above.
Read from the boot's own logs:

* the plan log must carry `coupled KV ratio MATCHED to the solve: ... -> v`;
* the `Uneven-DCP token sizing` line must report that same `v`, not
  `[31,17,16]`;
* `max_total_num_tokens` must land within 10 % of the plan's own
  `predicted ctx ~N` for the chosen vector, and must be **>= 300 000**
  (the 125 504 baseline is the falsifier: anything near it means the seed did
  not reach the boot).

A is a pass/fail on the coupling, not on throughput, and it does not by itself
decide anything about the INT8 layout.

**Sub-arm B -- the arm that can flip the recommendation.** Same command plus
`--rank-kv-ratio capacity`, so the measured phase-2 install converges the
vector onto the profiled per-rank capacity. Pass requires BOTH:

* `max_total_num_tokens >= 431 360` (the decode arm's pool), and
* prefill at s=1 and s=8 at **within-floor parity** with the decode arm --
  i.e. the delta is inside the LOOSER of the two boots' same-boot A-vs-A
  floors, which must be re-measured in this boot (3 identical draws each;
  #433's own boot floored at 3.5 % prefill / 27.5 % decode, and #424's at
  3.0 % / 12.9 %).

If both hold, the INT8 recommendation flips from "ship the decode layout" to
**prefill-layout-with-match**: the same pool as the decode arm plus the
prefill vector, with no measurable prefill cost. The #845 addendum decision
follows sub-arm B's outcome, not sub-arm A's.

**Corridor gate, both sub-arms:** sample min-free on ALL THREE cards for the
whole window (not only rank 0, per limit 2 above). The 400 MiB absolute floor
applies per card. If ranks 1/2 come in under it, the arm is invalid regardless
of the pool number, and the retry raises `--rank-auto-reserve-mib` on those
two cards rather than re-pinning a vector.

**Not in this arm:** any hand-pinned `--rank-mlp-ratio` or `--rank-kv-ratio`
vector. The whole point is that the optimizer now pairs them itself; pinning
either half re-creates the #354/#424 manual pairing this fix retires.

## Commands run (for reproduction)

```
# regression test, current tip, no GPU:
cd /spinning/wt-432-stash-rescue
CUDA_VISIBLE_DEVICES=99 PYTHONPATH=/spinning/wt-432-stash-rescue/python \
  HTSGLANG_TEST_MODEL_DIR=/spinning/llm_stuff/club-3090/models-cache \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest -q \
  test/registered/unit/planner/test_phase_optimal_targets.py

# desk sweep scripts (this task):
/tmp/433solve/desk_solve.py           # reserves 4500/5500,2700,2700 @ ctx 32768
/tmp/433solve/desk_solve_424point.py  # #424's own point: ctx 131072, reserve 4500,4200,4200
```

No GPU was touched. No `/spinning/gpu-arb` hold was taken. `git stash` was
not used anywhere in this task.
