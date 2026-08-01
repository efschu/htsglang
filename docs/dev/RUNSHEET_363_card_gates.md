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

## 0a. Corrections from the 2026-08-01 window (READ FIRST)

The first run of this runsheet cost three boots before the server came up.
All three are fixed in the commands below; they are recorded here because the
next reader will otherwise re-derive them.

* **sglang, not vLLM, spellings.** `--context-length`, not `--max-model-len`.
  Radix prefix caching and chunked prefill are ON by default -- there is no
  `--enable-prefix-caching` / `--enable-chunked-prefill`.
* **`--speculative-algorithm NEXTN` auto-chooses all three** spec params and
  ASSERTS the other two are unset. Do not pass
  `--speculative-num-draft-tokens`.
* **INT8-W8A8 does not boot on this build**: `NotImplementedError: No
  implemented int8_scaled_mm for current compute capability` (sgl_kernel
  0.3.21). It surfaces inside the cuda-graph cold-build window, so the
  visible error is a `ColdBuildWindowError` and the dispatch gap is the
  cause. The vehicle here is therefore **Qwen3.6-27B-FP8**, the documented
  reference arm.
* **`deep_gemm` hard-requires `libnvrtc.so.13`**, which the venv ships under
  `nvidia/cu13/lib` but does not put on the loader path. Without the
  `LD_LIBRARY_PATH` line below the FP8 path dies at the first GEMM.
* **Ready-wait patterns must be anchored.** Three separate over-matches cost
  a boot each: `Failed to` matches the benign `Ignore import error when
  loading ...` lines, `sigquit` matches `custom_sigquit_handler=None` inside
  the `server_args=ServerArgs(...)` dump, and any `pgrep -f`/`pkill -f`
  pattern naming the server also matches the checking shell itself (exit
  144, self-kill). Use the exclusion list and the PID-based kill below.

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
LD_LIBRARY_PATH=/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH \
python3 -m sglang.launch_server \
  --model-path $MODEL_ROOT/Qwen3.6-27B-FP8 \
  --served-model-name Qwen3.6-27B \
  --tp-size 3 \
  --rank-gpu-id 0,1,2 \
  --rank-tp-ratio auto-performance \
  --rank-auto-reserve-mib 5500,3800,3800 \
  --context-length 32768 \
  --port 30000 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --speculative-algorithm NEXTN \
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

### 1a. The ready-wait, copy-pasteable

Do not hand-roll this. Three separate over-matches cost one boot each in the
first window, and a bare `until curl` with no deadline is how the #377 window
lost 20 of its 25 minutes.

```bash
DEADLINE=$(( $(date +%s) + 540 ))          # wall clock, always
PAT='Received sigquit|Scheduler hit an exception|Initialization failed|CUDA out of memory|torch\.OutOfMemoryError|serve: error:|NotImplementedError'
EXCL='Ignore import error|server_args=ServerArgs'
while :; do
  [ $(date +%s) -ge $DEADLINE ] && { echo "VERDICT=DEADLINE"; break; }
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health)" = "200" ] \
    && { echo "VERDICT=READY"; break; }
  if grep -vE "$EXCL" "$OUT/boot.log" | grep -qE "$PAT"; then
    echo "VERDICT=TERMINAL"; grep -vE "$EXCL" "$OUT/boot.log" | grep -E "$PAT" | tail -2 | cut -c1-200
    break
  fi
  ps -eo pid,cmd | grep -E "python.*launch_server" | grep -v grep >/dev/null \
    || { echo "VERDICT=PROCESS_GONE"; break; }
  sleep 10
done
```

Why each piece is the way it is (all three learned by losing a boot):

| trap | what happened | the guard |
|---|---|---|
| `Failed to` as a death pattern | matches the benign `Ignore import error when loading sglang.srt.models.* : ... Failed to load dynamic shared library ...` lines every boot emits | the `EXCL` list |
| `sigquit` case-insensitively | matches `custom_sigquit_handler=None` inside the one-line `server_args=ServerArgs(...)` dump | `Received sigquit`, and `server_args=ServerArgs` in `EXCL` |
| `pgrep -f`/`pkill -f "...launch_server..."` | **matches the checking shell itself** -- the pattern is in your own command line. Exit 144, self-kill. It bit twice, once while releasing the arb files | `ps -eo pid,cmd \| grep -E ... \| grep -v grep`, and kill by **PID captured at launch**, never by pattern |

Capture the PID at launch so teardown never needs a pattern:

```bash
setsid nohup env ... python3 -m sglang.launch_server ... > $OUT/boot.log 2>&1 &
sleep 2; ps -eo pid,cmd | grep -E "python.*launch_server" | grep -v grep \
  | awk '{print $1}' > $OUT/server.pids
```

Shut down with `kill -INT $(cat $OUT/server.pids)`. SIGINT and SIGTERM both
write the trace's summary line since the #363 shutdown hook landed; SIGKILL
does not, and a summary-less trace is refused by `readout.py` on purpose.

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

The first window reached only **6.2 % occupancy** of a 519 670-token pool, so
nothing came near the admissibility interlock. The re-run wants a HEAVIER KV
load -- raise `--burst` and `--drain`, and give the drain arm enough
`--drain-tokens` that held tokens approach the 85 % ascend mark -- or the
occupancy axis stays untested for a second window running.

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

**Shut the server down cleanly**: `kill -INT $(cat $OUT/server.pids)`, or
SIGTERM -- both write the summary line now that the #363 shutdown hook is in
(clean exit, exception/KeyboardInterrupt, and SIGTERM are all covered).
`SIGKILL` is not, by design: `readout.py` refuses a summary-less trace because
"zero desyncs" and "zero desyncs so far" are different claims.

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

## 6. Gate 3 — two identical boots, per-signal bands

The band script EXISTS now: `scripts/regime_gates/bands.py`, written against
the real traces and smoke-green card-less. Alignment is settled (see its
docstring); what the window supplies is the second boot.

**The statistic is chosen per signal, and that is not cosmetic.** Since #388
the shares are near-binary at `window_rounds = 64` — a window is essentially
all-prefill or all-decode — so a pointwise A-vs-A band on them is 1 whenever
the two boots' bursts land on different boundary indices, which they always
do. Four signals (`prefill_share`, `decode_share`, `occupancy`,
`queued_prompt_tokens`) are therefore compared DISTRIBUTIONALLY: per-run
summaries (peak, quantiles, and the duty cycle at each constant's own value)
with the band `|summary_a - summary_b|`. `rank_ms_spread_pct` keeps the
pointwise band. Every signal is read on the ACTIVE boundaries only.

```bash
for run in a b; do
  OUT3=/spinning/gpu-battery-results/$(date +%F)_363_gate3_$run
  mkdir -p $OUT3
  # EXACTLY the section-1 boot, only --regime-trace changes.
  #   ... --regime-controller observe --regime-trace $OUT3/regime.jsonl
  # then EXACTLY the section-2 workload, same flags, same order:
  python scripts/regime_gates/workload.py --repeats 2 --burst 8 \
      --burst-tokens 900 --drain 12 --drain-tokens 900 --mixed 8 --idle-s 25 \
      | tee $OUT3/workload.log
  kill -INT $(cat $OUT3/server.pids)
done

python scripts/regime_gates/bands.py \
  --arm-a .../363_gate3_a/regime.rank0.jsonl \
  --arm-b .../363_gate3_b/regime.rank0.jsonl \
  --evidence $OUT/regime-gate-evidence.json \
  --note "two identical boots, workload.py identical flags"
```

**Identical means identical.** Same recipe, same workload flags, same order.
The band is what the arm measures against ITSELF; any difference you introduce
between the arms is measured as noise and inflates every threshold's bar.

Card-less, before pointing it at anything:

```bash
python scripts/regime_gates/bands.py --smoke     # the pipeline
python scripts/regime_gates/bands.py --falsify   # the statistic itself
```

`--falsify` is the argument for the distributional statistic, in three cases
that can each fail: a same-duty burst shift (the retained pointwise path still
calls it `ARMS_DISSIMILAR` — that is the false alarm of record), a genuine
duty difference (the guard has to survive the reformulation), and a
barely-moving signal (old and new must report the same band).

### What the report says, and the three ways it can refuse

Per signal it prints the statistic, the band, the two active-boundary counts
and the observed max; then the duty cycles at every constant's own threshold;
then a verdict per section-3.4 constant. Only `CLEARS` is a pass, and the
other four are different problems with different fixes:

| verdict | meaning | what to do |
|---|---|---|
| `CLEARS` | pointwise: the gap it defends exceeds 2x its band. Distributional: its crossing rate exceeds 2x the run-to-run disagreement about that rate | nothing |
| `INSIDE_BAND` | the quantity it is judged on is inside the signal's own noise | re-derive the constant from the band |
| `UNREACHED` | the signal never approached the constant in either run | the regime it gates cannot be entered. Either the constant is mis-set or the workload never made the shape — the report does not choose, and does not re-tune |
| `UNDERPOWERED` | fewer than 8 active boundaries in the smaller arm | a longer or busier workload; a band from 2 samples is a number, not a measurement |
| `ARMS_DISSIMILAR` | pointwise: the band is as large as one arm's own movement. Distributional: the two runs' duty cycles at the constants' own thresholds differ by more than sampling one workload twice should produce | the two boots were not doing the same thing. Check both ran the same workload and neither was truncated |

### What the archived traces already say, re-analysed with this statistic

All four archived arms (the pre-#388 pair and the post-#388 pair) have been
re-run through the distributional statistic at the desk, no cards. Expect
these again:

* `enter_prefill = 0.35` -> **CLEARS** on the post-#388 pair: crossing rate
  0.266 of active boundaries (arm A 0.317, arm B 0.214) against 2x the
  disagreement 0.103 = 0.206. It clears by 1.29x, which is a pass and a thin
  one — a busier workload is the way to widen it, not a new number.
* `enter_decode = 0.90` -> **CLEARS**: rate 0.565 against 2x 0.155 = 0.310.
* `spread_veto_pct = 25` -> **UNREACHED**: the measured spread peaked at
  **0.68 %** here, 0.61 % and 12.5 % in the two earlier windows. Recorded, not
  re-tuned — calibration is a separate decision with its own evidence.
* `kv_ascend_mark = 0.85` -> **UNREACHED** at 16.5 % peak occupancy, unless
  the workload is heavier than section 2's. This one is INHERITED from #287
  and is reported for information only: a failure here is a finding for #287,
  not a licence to set a second mark on the same pool.
* `PRESTAGE_SINGLE_PROMPT_TOKENS = 8192` -> **CLEARS**, also thinly (rate
  0.160 against 2x 0.070 = 0.140).

So the window's job for gate 3 is the two `UNREACHED` reachability findings,
not the statistic. None of these are reasons to change a number by hand. They
are the gate-3 output.

## 7. Gate 4, for the window after

Three arms (`off` / `observe` / `act`) over one workload, judged on ms/verify
**and** ms/prefill against the `off` arm's own band, at equal or better
`max_total_num_tokens`. It needs `act`, so it needs gates 1–3 in the evidence
file first, and it needs a boot with `--kv-reshard-vectors` covering a real
flip target — otherwise the act arm has nowhere to go and the comparison
measures nothing.

---

## 8. The RE-RECORD window after #388 (do this before gate 4)

#388 changed what the classifier emits: the phase is now read off the batch
that RAN (`last_batch`) instead of off `running_batch.is_prefill_only`, which
was a request kind and False on every generating batch. `prefill_share` can
move for the first time, so **every gate recorded under the old attribution
describes the old classifier**.

What has to be re-recorded, and what does not:

| gate | status | why |
|---|---|---|
| 1 (desyncs) | **RE-RECORD** | the regime histogram and the transition count both change; a desync count is about the verdicts, and the verdicts are different now |
| 2 (F2 replay) | **RE-RECORD** | it replays the trace of gate 1. Same boot, so it costs nothing extra |
| 3 (`enter_prefill`) | **RE-RECORDED and re-analysed** | UNREACHED was a reading of a signal that could not move. On the #388 pair it CLEARS |
| 3 (`occupancy`, `queued_prompt_tokens`) | already re-analysed, no cards | #388 restricted them to ACTIVE boundaries and re-ran the SAME two archived traces. ARMS_DISSIMILAR was an alignment artifact and is gone; see DESIGN §17.3 |
| 3 (all five signals, the STATISTIC) | already re-analysed, no cards | the #388 pair came back ARMS_DISSIMILAR on every signal, which was a pointwise statistic applied to a near-binary one. Fixed and all four archived arms re-analysed at the desk; see DESIGN §19 |
| 4 | not yet startable | needs 1-3 in the evidence file |

**The boots.** Gates 1+2 ride on ONE boot; gate 3 needs the pair. Three boots
total, and all three are the §1 recipe with only `--regime-trace` differing —
nothing else may change, or the gate-3 arms stop being an A-vs-A pair.

```bash
# Boot R1 -- gates 1+2. Section 1 verbatim, new output directory.
OUT=/spinning/gpu-battery-results/$(date +%F)_363_gates_388
# Boots R2/R3 -- gate 3, the pair. Section 6 verbatim.
OUT3=/spinning/gpu-battery-results/$(date +%F)_363_gate3_388_$run   # run in a b
```

Flags, unchanged from §1 and §6 — **#388 added no flag and changed no
default**, which is the point: the same launch line now records a different
(and correct) phase:

```
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance
  --rank-auto-reserve-mib 5500,3800,3800
  --model-path $MODEL_ROOT/Qwen3.6-27B-FP8 --context-length 32768
  --kv-cache-dtype fp8_e4m3 --speculative-algorithm NEXTN --trust-remote-code
  --enable-metrics --enable-metrics-for-all-schedulers
  --regime-controller observe --regime-trace $OUT/regime.jsonl
  SGLANG_ENABLE_METRICS_DEVICE_TIMER=1  LD_LIBRARY_PATH=<venv>/nvidia/cu13/lib
```

Workload, unchanged, and the same flags on all three boots:

```bash
python scripts/regime_gates/workload.py --repeats 2 --burst 8     --burst-tokens 900 --drain 12 --drain-tokens 900 --mixed 8 --idle-s 25
```

**One thing to add if the window allows it, and it is not free.** The pre-#388
gate-3 arms produced only **29 active boundaries** out of 15-19 k, and the
re-record's produced **41 and 56**. That is above `MIN_PAIRED_SAMPLES = 8` but
not by much, and `UNDERPOWERED` BLOCKS. It is also what makes the two passing
distributional verdicts thin: a duty cycle estimated from 41 windows has a
standard error near 0.07, which is most of the 0.103 disagreement
`enter_prefill` had to clear. A longer or busier workload buys margin on every
signal at once, and it is the only thing that will. Do not change the workload
flags BETWEEN the two gate-3 arms.

**What to check first, before spending the window on a workload.** The whole
point of the re-record is that `prefill_share` moves. Confirm it early:

```bash
grep -m5 REGIME-OBSERVE $OUT/boot.log | grep -o 'prefill [0-9]*%'
```

A run that still reports `prefill 0%` through a `prefill_burst` phase means
the fix did not reach this boot — check the branch before spending the rest of
the window.

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
