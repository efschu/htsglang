# RUNSHEET #363 — card gates 1+2 (and what gate 3 needs next)

Everything the entry gate of `DESIGN_363_regime_controller.md` §11.7 needs
from a card, written so the window is spent measuring rather than deciding.

**Gates 1 and 2 ride along on ONE observe boot.** Observe actuates nothing, so
it is safe on any window whose primary job tolerates a second server — and its
trace is by construction an open-loop record, which is exactly what gate 2's
counterfactual needs. Gate 3 needs its own pair of boots; gate 4 needs three
arms and cannot start until 1–3 are recorded, because it needs `act`, which
needs the evidence file.

Order is forced: **1+2 → 3 → 4.**

---

## 0. Before the window

Card-less, already done — every tool below has a smoke that runs without a
GPU, and all three pass:

```bash
python scripts/regime_gates/workload.py  --dry-run
python scripts/regime_gates/readout.py   --smoke
python scripts/regime_gates/f2_replay.py --smoke
```

Check the model root and that no other server holds the port:

```bash
MODEL_ROOT=/spinning/llm_stuff/club-3090/models-cache
ls -d $MODEL_ROOT/Qwen3.6-27B-INT8-W8A8
ss -ltnp | grep 30000 || echo "port free"
```

---

## 1. The observe boot

The standard INT8 production recipe plus three flags. Nothing else changes,
which is the point: the trace has to be of the server people actually run.

```bash
MODEL_ROOT=/spinning/llm_stuff/club-3090/models-cache
OUT=/spinning/gpu-battery-results/$(date +%F)_363_gates
mkdir -p $OUT

SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
python3 -m sglang.launch_server \
  --model-path $MODEL_ROOT/Qwen3.6-27B-INT8-W8A8 \
  --served-model-name Qwen3.6-27B \
  --tp-size 3 \
  --rank-gpu-id 0,1,2 \
  --rank-tp-ratio auto-performance \
  --rank-auto-reserve-mib 5500,3800,3800 \
  --max-model-len -1 \
  --port 30000 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --speculative-algorithm NEXTN \
  --speculative-num-draft-tokens 4 \
  --enable-metrics --enable-metrics-for-all-schedulers \
  --regime-controller observe \
  --regime-trace $OUT/regime-rank.jsonl \
  2>&1 | tee $OUT/boot.log
```

Three flags and why each is not optional:

| flag | why |
|---|---|
| `--regime-controller observe` | the mode under test. It actuates nothing (§11.2), so the boot is as safe as an unflagged one |
| `--regime-trace PATH` | the artifact BOTH gates consume. Without it the desync count exists only in a log line and the replay has no input |
| `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` | the per-rank device timing (#252) the tier-L spread is built from. Without it `rank_ms_spread_pct` is absent for the whole run and the one-boundary-stale veto has never been exercised |

`--rank-auto-reserve-mib 5500,3800,3800` is the reserve #360 validated against
real long prompts (runbook §4). The workload below sends 12 k-token prompts,
so the smaller runbook reserves will OOM in the GDN prefill scratch.

**Per-rank trace files.** The desync count is a property of the GROUP, and
`readout.py` refuses a multi-rank verdict built from one rank's file. If the
scheduler processes share a filesystem path they will interleave into one
file; take one file per rank if the launcher gives each rank its own working
directory, otherwise pass `--ranks 1` and record in the note that the trace is
rank-0's. The refusal text says which situation you are in.

Confirm the observer armed, before spending the window on a workload:

```bash
grep REGIME-OBSERVE $OUT/boot.log | head -20
```

Expect: `armed (OBSERVE-ONLY: ...)`, the stage-table line (`N stage(s), M
reachable, K flip target(s)`), and one `stage ...` line per stage with its
reachability reason. On a rig without `--kv-reshard-vectors` the honest answer
is `1 stage, 0 flip targets` — that is a valid gate-1 run, and it is also the
finding that gate 4 will need a reshard-declared boot.

---

## 2. The workload

Walks the four shapes the classifier names, twice, so the trace contains a
regime RETURN (which is what hysteresis and dwell are judged on).

```bash
python scripts/regime_gates/workload.py \
  --base http://127.0.0.1:30000 --model Qwen3.6-27B --repeats 2 \
  2>&1 | tee $OUT/workload.log
```

| phase | shape | expected regime |
|---|---|---|
| `prefill_burst` | 4 × 12 k-token prompts admitted at once | `PREFILL_HEAVY` |
| `decode_drain` | 4 × 512-token generations, empty queue | `DECODE_HEAVY` |
| `idle` | 45 s of nothing | `MIXED` — an idle window is no measurement, not 0 % prefill |
| `mixed` | short arrivals during long generations | `MIXED` |

Runtime: roughly 8–12 min for two cycles at these sizes. A phase whose regime
never appears in the trace is a finding about the thresholds (§3.4 constants
are still provisional), not a reason to re-tune mid-window — record it and
carry it into gate 3.

**Shut the server down cleanly** (`SIGINT`, not `SIGKILL`). The summary line
is written on close, and `readout.py` refuses a trace without one: "zero
desyncs" and "zero desyncs so far" are different claims.

---

## 3. Gate 1 — desyncs

```bash
python scripts/regime_gates/readout.py \
  --trace $OUT/regime-rank.jsonl --ranks 1 \
  --evidence $OUT/regime-gate-evidence.json \
  --note "INT8-W8A8 TP=3, workload.py 2 cycles"
```

Passes only when all of these hold, and each refusal names itself:

* a `summary` line exists (the run ended cleanly);
* there are verdicts (a consensus boundary was reached);
* at least two distinct regimes appear — a run that produced one shape never
  asked the ranks to agree about a change;
* the run was not `uncoordinated` (a multi-rank group with no consensus
  channel never checked agreement, so a zero there means nothing);
* `desyncs == 0`.

On a pass it writes the `desyncs_zero` entry with a `source` built from the
run's own facts. The gate refuses an unattributed pass, so the source is
generated rather than typed.

**If it fails on desyncs:** that is the most valuable outcome of the window.
It BLOCKS act, and the same disagreement under an actuator is the
#94/#194/#259 hang. Keep the trace, do not retry to get a green.

---

## 4. Gate 2 — F2 live replay

```bash
python scripts/regime_gates/f2_replay.py \
  --trace $OUT/regime-rank.jsonl \
  --booted-pool  $(curl -s http://127.0.0.1:30000/get_server_info |
                   python3 -c 'import json,sys;print(json.load(sys.stdin)["max_total_num_tokens"])') \
  --target-pool 96256 \
  --evidence $OUT/regime-gate-evidence.json \
  --note "counterfactual against the #354 FP8 prefill arm pool"
```

Take `--booted-pool` from the live server **before shutting it down** (the
command above assumes it is still up; otherwise read it out of `boot.log`).
`--target-pool` is the pool of the stage a controller would have selected —
use the real one from the boot stage table if the table has a flip target, and
otherwise the #354 prefill arm's 96 256, which is the tightest case on this
rig.

Three verdicts the tool can return:

1. **NON-DETERMINISTIC** — the open-loop replay does not reproduce the
   recorded sequence, even though nothing actuated. The classifier depends on
   an input the trace does not carry; F2 cannot be evaluated until that input
   is recorded. Report the diff, do not paper over it.
2. **SELF-CONDITIONING** — the counterfactual produces transitions the trace
   does not. Those would have been the controller's own doing. Blocks act.
3. **Pass**, with `interlock_was_load_bearing` telling you *why* it passed:
   `true` means the admissibility interlock refused the trap the workload
   really did approach (strong evidence); `false` means the workload never
   got near it (weak evidence — note it, and prefer a longer/heavier
   `--burst` next time).

---

## 5. Read out and park

```bash
cat $OUT/regime-gate-evidence.json
```

Two of four items. `--regime-controller act` still refuses, and its refusal
now shows two `[ok]` lines and two `[MISSING]` — which is the expected state
after this window and confirms the evidence file is being read.

```bash
# Expect a refusal naming f3_bands_measured and f4_card_comparison only.
python3 -c '
import sys; sys.path.insert(0,"python")
from sglang.srt.managers.regime_stages import load_gate_evidence
print(load_gate_evidence("'$OUT'/regime-gate-evidence.json").refusal())'
```

Archive `$OUT` and record the path in the task ledger.

---

## 6. Gate 3, if time remains in the same rotation

Gate 3 is the per-signal A-vs-A band (#360): **two identical boots**, same
recipe, same workload, same seed, and the band is what the arm measures
against itself.

```bash
for run in a b; do
  OUT3=/spinning/gpu-battery-results/$(date +%F)_363_gate3_$run
  mkdir -p $OUT3
  # ... the section-1 boot, with --regime-trace $OUT3/regime-rank.jsonl ...
  python scripts/regime_gates/workload.py --repeats 2 | tee $OUT3/workload.log
  # ... clean shutdown ...
done
```

What it must produce, per classifier input signal (`prefill_share`,
`decode_share`, `occupancy`, `queued_prompt_tokens`, `rank_ms_spread_pct`):

```
band(s) = max over aligned windows of |s_runA(w) - s_runA'(w)|
```

and then, for every threshold in §3.4, the check that it sits outside its own
band by at least the band again (2×, because a threshold one band away still
spends half its time on the wrong side of itself).

`regime_classifier.signal_band()` and `clears_band()` are the same functions
the controller compares against — use them rather than recomputing, so the
experiment and the runtime cannot drift.

**No band-computing script is written yet.** It needs the two traces to exist
before its shape can be settled honestly (window alignment across two boots is
the part that will be fiddly, and guessing at it at the desk would produce a
tool that fits an imagined trace). Writing it is the first desk task after
this window returns.

Provisional constants gate 3 must replace or confirm: `window_rounds` 64,
`enter_prefill` 0.35 / `exit_prefill` 0.15, `enter_decode` 0.90 /
`exit_decode` 0.70, `spread_veto_pct` 25, `PRESTAGE_SINGLE_PROMPT_TOKENS`
8192. The occupancy marks are #287's and are deliberately NOT re-derived.

---

## 7. Gate 4, for the window after

Three arms (`off` / `observe` / `act`) over one workload, judged on ms/verify
**and** ms/prefill against the `off` arm's own band, at equal or better
`max_total_num_tokens`. It needs `act`, so it needs gates 1–3 in the evidence
file first, and it needs a boot with `--kv-reshard-vectors` covering a real
flip target — otherwise the act arm has nowhere to go and the comparison
measures nothing.

---

## Appendix — what a desk smoke already proved, and what it cannot

The three tools pass their smokes card-less, and the gate-2 smoke reproduces
the phase-2 F2 result end to end (the unguarded counterfactual manufactures
`kv_pressure`; the guarded one does not). What no smoke can show is whether
the ranks of a real boot agree, whether the live thresholds fire on real
traffic, or what the signals' real noise is. That is the whole content of the
window.

One thing the desk smoke DID catch, and it changed the trace format: the
observer originally recorded `occupancy` (a ratio) and not `held_tokens` (its
numerator). Gate 2's counterfactual varies the capacity denominator — that is
the self-conditioning mechanism — so a replay rebuilding held tokens from a
recorded occupancy moves the numerator with the denominator and every
counterfactual comes back clean for the wrong reason. The trace now records
both, and the tool refuses a trace carrying only the ratio.
