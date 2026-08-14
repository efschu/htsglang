# RUNSHEET — #363 ACT window: measure the stages, then run the rule live

**One file, executable top to bottom.** Preflight, claim, four measurement
boots, the measurement pass, the act leg with the decision rule live,
extraction, teardown, sanctioned restore.

**Derived from** the merged line `ac0d1f36f3` (`feat/route-a-631` ==
`integration/r2` at the time of writing) plus branch `feat/desk-363-act`,
worktree `/spinning/wt-desk-363-act`,
`PYTHONPATH=/spinning/wt-desk-363-act/python`. Every flag, env var and script
below was verified against the REAL parser and the REAL files on that branch
(§11); nothing here was grepped.

> Read `docs/rig-runbook.md` before the first boot — it is the source of truth
> for launch mechanics and it outranks any command line copied from a runsheet,
> including this one.

---

## 0. WHY THIS WINDOW EXISTS, AND WHAT CHANGED SINCE THE LAST ONE

`TICKET_363_WINDOW_VERDICT.md` failed with two blockers.

**Blocker one is now addressable.** The axis had zero flip targets because
`build_stage_table` refuses a planner-solved candidate that carries no
measurement, and nothing could produce one:

```
RegimeError: stage table refused (#578): the planner solved 1 stage(s) --
solved-enc -- but they carry no measurement. Each needs measured_gain_pct,
measured_band_pct and flip_cost_s ... The solver cannot predict any of the three.
```

`#584`'s first half put measured card rates on disk and cleared the planner
feed (`WINDOW_VERDICT_584_M584.md` §6: *"with per-stage measurements present:
2 stages, 1 flip target, reach=reshard"*). This branch adds the second half —
the per-stage measurement canon (`planner/stage_measure_store.py`), the pass
that fills it from the controller's own traces
(`planner/stage_measure_pass.py`), and the promotion seam
(`regime_stages.apply_measurements`). **This window's first job is to take
those measurements on metal.** Until they exist the act leg has nowhere to go
and must not be run — that is §5's STOP.

**Blocker two is NOT closed and this window may not close it.** Gate 3 blocks
on `spread_veto_pct = 25.0`, which peaked at 0.407 % on a balanced rig and
**does not exist in the runtime at all** (`grep -rn 'spread_veto_pct' python/`
returns nothing; the only act-mode interlock on that signal vetoes on
`rank_ms_spread_pct is None`, never on a magnitude). So gate 3 cannot pass and
`--regime-controller act` is refused at parse time. The verdict's item 2 —
*decide what that constant is for: wire it, or take it out of gate 3's blocking
set* — is a decision about the GATE and is owed by a shift with that mandate.
**This runsheet does not re-tune it.** §5 uses the bootstrap the previous
runsheet already sanctions, under the same four rules, and the window log
records that the act leg ran under one.

---

## 1. WHAT THE WINDOW PRODUCES

| # | Artefact | Produced by |
|---|---|---|
| M1 | A **stage measurement record** (gain, band, flip cost) in the canon, keyed by this rig's card-UUID set and this checkpoint | boots B1–B3 + the pass (§6) |
| M2 | A boot log line reading **`stage table: 2 stage(s), 2 reachable at runtime, 1 flip target(s)`** | boot B4 (§7) |
| M3 | An **act trace** whose `ms_decision` rows carry `signal_pct`, `band_pct`, `flip_cost_pct` and `threshold_pct` — the decision rule, per boundary | boot B4 |
| M4 | A **corridor verdict**: zero samples below 1024 MiB on every card, 100 ms sampling, every boot | §8 |

---

## 2. THE DECISION RULE THAT IS BEING PUT ON METAL

Verbatim from `managers/regime_ms_clock.py` (module docstring and
`decision_threshold_pct`). A flip to candidate C from the stage in force I is
proposed at a boundary only when ALL of:

```
(1) the ms window is ready (>= 8 rounds) and I is MEASURED;
(2) signal(C) > threshold(C), where

        signal(C)        = 100 * (mean_total_ms - predicted_ms(C)) / mean_total_ms
        predicted_ms(C)  = compute_ms + wait_ms * (1 + g_I/100) / (1 + g_C/100)
        threshold(C)     = max( enter_margin_pct,
                                band(C) + flip_cost_pct(C) )
        band(C)          = sqrt( band(I)^2 + band(C)^2 )
        flip_cost_pct(C) = 100 * flip_cost_s(C) / flip_payback_s

(3) C is the same best candidate for enter_window (2) consecutive boundaries;
(4) some candidate clears exit_margin_pct (2 %) for exit_window (4)
    consecutive boundaries.
```

Then the five act interlocks (`regime_runtime._act_interlocks`) still apply:
selectability, dwell, group agreement, the one-boundary-stale spread veto, and
corridor admission.

**Every term is measured except two policy constants**, and both are named as
policy in the code: `enter_margin_pct = 5.0` and `flip_payback_s = 60.0`. The
second is new with this slice: it is the horizon a flip must repay itself
within, and it is what converts an instrumented cost in SECONDS into the same
unit as a gain in PERCENT OF A ROUND. It is deliberately not the dwell — using
the dwell would make `flip_cost_pct` collapse to the constant `100 /
amortization`, i.e. the measured flip cost cancelling out of its own price.

**Watermark policy, unchanged from the previous runsheet:** if the A-vs-A floor
measured in §4 exceeds 5.0 %, the enter watermark moves ONCE, before the act
leg, recorded with its measurement — never afterwards until the result is
agreeable.

---

## 3. PREFLIGHT — CPU only, before claiming any card

```bash
cd /spinning/wt-desk-363-act
export PYTHONPATH=/spinning/wt-desk-363-act/python
PY=/spinning/htsglang-gpu/.venv/bin/python

# 3a. the window preflight (flags, gate evidence, census, stage table).
#     Pass ALL of these: each omitted argument turns a check into a SKIP, and
#     a SKIP is what let the last window claim cards with the blocker unseen.
CUDA_VISIBLE_DEVICES=99 $PY scripts/regime_363_window/preflight_363_window.py \
  --evidence $EV --boot-log $LAST_BOOT_LOG --census-dir $OUT/census \
  --kv-reshard-vectors '2,11,10;3,10,10' --stage solved-enc --strict

# 3b. the corridor instrument's own can-fail arm -- 3/3 or do not trust it
$PY scripts/regime_363_window/corridor_report.py --smoke

# 3c. the measurement canon: what is already on disk for this rig
$PY -m sglang.srt.planner.stage_measure_pass --show

# 3d. the card rates #584 measured (the canon's precondition)
$PY -m sglang.srt.planner.card_rate_pass --show
```

`3c` on a rig that has never measured a stage prints `0 record(s)` — that is
the expected state and it is what §4 fixes. If it already lists the candidate
as **USABLE**, skip to §7.

---

## 4. CLAIM, THEN THE MEASUREMENT BOOTS

### 4.0 Claim

```bash
mkdir -p /spinning/gpu-arb
echo "363-act $$ $(date -Is)" > /spinning/gpu-arb/holder
# heartbeat in its OWN process; STOP THE HEARTBEAT BEFORE RELEASING
```

Whoever stops serving owns bringing it back (§10).

### 4.1 Common launch line (verified against the parser, §11)

```bash
NVRTC=/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="$NVRTC:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/spinning/wt-desk-363-act/python
export SGLANG_UNEVEN_DCP=1 SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
export SGLANG_TRANSIENT_CENSUS=1 SGLANG_RESIDENCY_CENSUS_DIR=$OUT/census
export SGLANG_STAGE_MEASUREMENTS=/spinning/evidence-363-act/stage_measurements.json

MODEL=/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8
COMMON="--model-path $MODEL --tp 3 --rank-gpu-id 0,1,2
  --rank-tp-ratio auto-performance --rank-auto-reserve-mib 3000,2700,2700
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code
  --max-running-requests 16
  --speculative-algorithm NEXTN --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --enable-metrics --host 127.0.0.1 --port 30030"
```

`--rank-auto-reserve-mib 2200` on the 3080s **tips** (OOM in the GDN prefill
scratch; the 80-token warm-up survives and fakes success). 2700 carries.

### 4.2 The four boots

`$OUT` is a fresh directory per boot. **Nothing may differ between B1 and B2
except `$OUT`**, or they stop being an A-vs-A pair.

| Boot | Adds to `$COMMON` | Purpose |
|---|---|---|
| **B1 — floor A** | `--regime-controller observe --regime-stage-clock --regime-trace $OUT/trace.jsonl --kv-reshard-vectors '2,11,10;3,10,10'` | the reference stage, repeat 1 |
| **B2 — floor B** | identical to B1, new `$OUT` only | the reference stage, repeat 2. **The floor comes from this pair, before any delta is read.** |
| **B3 — the candidate stage** | identical to B1; the candidate layout is reached IN THIS BOOT by `POST /kv_reshard` (4.4) | the stage arm, and the boot the flip cost is instrumented on |
| **B4 — the act leg** | §7 | the rule live |

**Why B3 reshards rather than booting the candidate layout.** The KV token
vector is not a boot flag: `--rank-kv-ratio` takes MODES (`coupled`,
`capacity`), and the vector itself is what `--kv-reshard-vectors` declares and
`/kv_reshard` installs. So the stage arm is a SEGMENT of one boot's trace, and
the pass takes `--stage-from-round` / `--reference-to-round` for exactly that
— the round index is the only monotone index the trace carries (there is no
wall clock in it, #363 window F3) and it is replicated across ranks, so the
same bounds select the same window on every rank's file.

**Consequence to respect:** the reference segment and the stage segment are
from ONE boot, which removes boot-to-boot variance from the delta and is the
better comparison — but the A-vs-A floor must still come from the B1/B2 PAIR,
because that pair is what measures the variance the delta has to beat. Do not
substitute a within-boot split for the floor.

### 4.3 Load — one window per boot, >= 10 s of covered device time per arm

The floor is 10 s of DEVICE time (`MIN_MEASURE_SECONDS`, refused by the record
itself). The profile below produces several hundred, which is the margin the
band needs to be worth reading.

| Phase | Duration | Shape |
|---|---|---|
| Settle | 120 s | decode-heavy, short prompts |
| SHIFT | 300 s | deep-prefill burst: long prompts, few outputs |
| Return | 300 s | decode-heavy again, identical to settle |

```bash
$PY scripts/regime_gates/workload.py --base http://127.0.0.1:30030 \
  --repeats 2 --burst 16 --burst-tokens 6000 \
  --drain 12 --drain-tokens 900 --mixed 8 --idle-s 25
```

**Same driver flags on every boot.** The gate-3 pair proved what a workload
change between arms costs; a floor measured under one load and a delta under
another is not a floor.

Check early, before spending the window:

```bash
grep -m5 REGIME-OBSERVE $OUT/boot.log | grep -o 'prefill [0-9]*%'
```

`prefill 0%` through a prefill burst means the #388 attribution did not reach
this boot — stop and check the branch.

### 4.4 B3's shape: reference segment, the move, stage segment, flip samples

One boot, one driver run, four phases. Keep the driver running throughout —
a reshard commits at the next boundary where every rank is idle, and a server
with no traffic gives a flip cost measured on an empty pool, which is not the
cost the controller will pay.

```bash
# (1) REFERENCE SEGMENT -- >= 5 min on the booted vector, under the §4.3 driver
#     note the last round before the move:
grep -o '"round": [0-9]*' $OUT/trace.rank0.jsonl | tail -1     # -> $REF_TO

# (2) THE MOVE -- arm the candidate vector; returns immediately
curl -s -X POST http://127.0.0.1:30030/kv_reshard \
     -H 'content-type: application/json' -d '{"target_vector": [3,10,10]}'
grep 'KV-RESHARD DONE' $OUT/boot.log | tail -1          # carries total_ms
grep -o '"round": [0-9]*' $OUT/trace.rank0.jsonl | tail -1     # -> $STAGE_FROM (+2 boundaries)

# (3) STAGE SEGMENT -- >= 5 min on the candidate vector, SAME driver flags

# (4) FLIP SAMPLES -- three round trips, still under load
for i in 1 2 3; do
  curl -s -X POST http://127.0.0.1:30030/kv_reshard -H 'content-type: application/json' \
       -d '{"target_vector": [2,11,10]}'; sleep 45
  curl -s -X POST http://127.0.0.1:30030/kv_reshard -H 'content-type: application/json' \
       -d '{"target_vector": [3,10,10]}'; sleep 45
done
grep -c 'KV-RESHARD DONE' $OUT/boot.log                  # expect >= 7
```

Give `$STAGE_FROM` two boundaries of slack after the DONE line: the ms window
holds samples taken under the OLD vector and averaging them into the new one
judges the new configuration partly by the one it replaced.

The flip cost is the ONE term the record refuses to default. It comes from the
actuator's own report — the DONE line's `total_ms`, which the mover times
around its whole read/exchange/write/cutover walk — never from a stopwatch
around two log lines. The pass takes the **MAXIMUM** of the samples: the
controller pays the cost of the flip it is about to make, and pricing by the
mean under-charges exactly the expensive one.

---

## 5. STOP CONDITIONS — read these before boot B4

1. **No usable record after §6 ⇒ STOP.** The act leg would be an expensive
   observe under a misleading name; that is the verdict's own §1 and it is why
   three boots were spent last time instead of seven.
2. **Gate 3 is still blocked** (§0). The act leg therefore runs under the
   sanctioned bootstrap, with all four rules of
   `RUNSHEET_363_STAGE_CLOCK_WINDOW.md` §3 in force: the `source` says it is a
   bootstrap in those words, it is rewritten with the measured verdict
   immediately after, a failing gate returns the file to three items, and the
   window log records that the first act leg ran under one.
3. **A corridor breach on any boot fails the window regardless of everything
   else**, and it is a code defect rather than a tight margin.

---

## 6. THE MEASUREMENT PASS — turn the boots into the canon

```bash
cd /spinning/wt-desk-363-act && export PYTHONPATH=/spinning/wt-desk-363-act/python
export SGLANG_STAGE_MEASUREMENTS=/spinning/evidence-363-act/stage_measurements.json

$PY -m sglang.srt.planner.stage_measure_pass \
  --stage solved-enc --regime prefill_heavy --reference booted \
  --model-key /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8 \
  --reference-trace $B3/trace.rank0.jsonl --reference-trace $B3/trace.rank1.jsonl \
                    --reference-trace $B3/trace.rank2.jsonl \
  --reference-to-round $REF_TO \
  --stage-trace     $B3/trace.rank0.jsonl --stage-trace $B3/trace.rank1.jsonl \
                    --stage-trace $B3/trace.rank2.jsonl \
  --stage-from-round $STAGE_FROM \
  --floor-a $B1/trace.rank0.jsonl --floor-b $B2/trace.rank0.jsonl \
  --flip-log $B3/boot.log \
  --phase-regime prefill_heavy --warmup 20 \
  --source "363-act window <date>, boots B1/B2/B3, driver burst16/6000" \
  --write
```

Both arms are segments of B3's own trace (§4.4); the FLOOR is the B1/B2 pair,
and substituting a within-boot split for it would measure the delta against
itself. The rig key is resolved from NVML unless `--rig-uuid` is given. The
pass REFUSES, with the arm named, on: a trace with no summary line (the server was
killed, so "zero so far" is all it supports), a boot with no
`--regime-stage-clock` (`ms_decision` is `None` on every record), ranks that
disagree by more than 1 % on the mean round, fewer than 8 boundaries, an arm
below the 10 s floor, and no instrumented flip.

Then confirm what the canon will admit:

```bash
$PY -m sglang.srt.planner.stage_measure_pass --show
```

A row printed with `REFUSED:` lines is a measurement that was TAKEN and is not
usable — the commonest case is a gain inside its own band, which means the
stage is not better on this rig and the honest outcome is that the axis has
nothing to select. **That is a result, not a failure of the window.**

---

## 7. BOOT B4 — the act leg, with the rule live

```bash
EV=/spinning/evidence-363-act/gate_evidence.json     # §5 rule 2
$COMMON \
  --regime-controller act \
  --regime-gate-evidence $EV \
  --regime-stage-clock \
  --regime-trace $OUT/trace.jsonl \
  --kv-reshard-vectors '2,11,10;3,10,10'
```

`act` is refused at parse time unless the evidence file names all four gate
items with a non-empty `source`, AND at least one actuator is wired
(`--kv-reshard-vectors` or `--enable-vram-dial`) — both refusals are in
`server_args._handle_regime_controller`.

**First check, within 60 s of boot:**

```bash
grep -m1 'stage table:' $OUT/boot.log
# want: stage table: 2 stage(s), 2 reachable at runtime, 1 flip target(s), booted on 'booted'
grep 'stage measurement:' $OUT/boot.log     # every refusal, by stage and reason
grep 'stage measurement canon' $OUT/boot.log
```

`1 flip target(s)` is the line the whole slice exists to produce. `0 flip
target(s)` with a `stage measurement:` refusal beside it means the canon did
not match this boot — read the reason: it names which of the four misses
applied (never measured / other card set / other checkpoint / self-refused).

Run the SAME driver as §4.3, and sample the corridor (§8) for the whole boot.

---

## 8. CORRIDOR SAMPLING — every boot, 100 ms, NVML FREE

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  -lms 100 > $OUT/corridor.csv &
CORRIDOR_PID=$!
# ... boot + driver ...
kill $CORRIDOR_PID
$PY scripts/regime_363_window/corridor_report.py --csv $OUT/corridor.csv
```

Free is the NVML **FREE** column, never `total - used` (a carve-out of
~424/518 MiB is invisible to the subtraction), and the verdict is the
time-series MINIMUM under load, never a boot snapshot. The reader exits 1 on
any sample below 1024 MiB and on a series too short to be a time series; its
can-fail arm is `--smoke` (3/3, §3b).

---

## 9. EXTRACTION AND PASS CRITERIA

```bash
# A1/A2/A5 -- proposals, actuations, flips, desyncs, summary present
$PY scripts/regime_gates/readout.py --trace "$B4/trace.rank*.jsonl" --ranks 3

# A3 -- ms/round, compute vs wait, judged against the B1/B2 floor
$PY scripts/regime_363_window/msround_split.py \
  --arm $B4/trace.rank0.jsonl --arm $B4/trace.rank1.jsonl --arm $B4/trace.rank2.jsonl \
  --control $B1/trace.rank0.jsonl --control $B1/trace.rank1.jsonl --control $B1/trace.rank2.jsonl \
  --floor-a $B1/trace.rank0.jsonl --floor-b $B2/trace.rank0.jsonl \
  --phase-regime prefill_heavy --warmup 20 --min-samples 30 \
  --require-summary --json

# A6 -- the rule, per boundary, from the act trace
$PY - "$B4"/trace.rank*.jsonl <<'EOF'
import json, sys
for path in sys.argv[1:]:
    for line in open(path):
        row = json.loads(line)
        d = (row.get("ms_decision") or {}) if row.get("kind") == "verdict" else {}
        if d.get("signal_pct") is not None:
            print(path.split('/')[-1], row["round"], f"signal={d['signal_pct']:.2f}",
                  f"band={d['band_pct']:.2f}", f"flip={d['flip_cost_pct']:.2f}",
                  f"threshold={d['threshold_pct']:.2f}", "->", d["target"])
EOF
```

**On `--control`:** `msround_split.py` documents the control as a boot WITHOUT
`--regime-stage-clock`, because that was the previous window's shape and its F4
finding was that such a control carries no compute/wait split at all. Here B1
carries the clock, so both sides of the comparison have the split and the
"was it the wait term that moved" question is answerable on both — strictly
better than F4's situation, and worth stating so a reader does not read the
tool's help as a contradiction.

| ID | Criterion |
|---|---|
| **A1** | `stage_clock_proposals > 0` **and** `actuations > 0`, at least one inside the SHIFT phase. Read from the trace, not a log line. |
| **A2** | Total flips over the window **<= 4**. Two are structural (into the prefill stage, back out); more is thrash. |
| **A3** | ms/round in SHIFT beats the B1 control by **more than the B1/B2 floor**, and the movement is in the **wait** term. |
| **A4** | NVML free per card, 100 ms, **zero samples below 1024 MiB**, every boot. One breach fails the window regardless of A1–A3. |
| **A5** | `desyncs == 0` in every rank's trace **and the summary line present**. |
| **A6** | Every `ms_decision` row carries `flip_cost_pct` and `threshold_pct`, and every ADOPTED target has `signal_pct > threshold_pct`. A single adopted row that does not is a defect in the rule, not a tight call. |

Four of six is not a pass.

---

## 10. TEARDOWN AND SANCTIONED RESTORE

1. **Stop the corridor sampler** and confirm `corridor.csv` has a final line —
   a truncated series has no minimum.
2. **Stop the server**, confirm the trace summary line on EVERY rank file.
   Without it A5 is unanswerable and the boot must be repeated.
3. **Rewrite the gate evidence file** if a bootstrap was used (§5 rule 2).
   This is the step most likely to be forgotten and has the longest tail: a
   bootstrap left in place silently authorizes every later act boot.
4. **Park the canon**: `stage_measurements.json` is the window's durable
   output. Copy it to the evidence directory next to the traces, and record in
   the window log which boots produced each record's `source`.
5. **Stop the heartbeat BEFORE releasing** the arbitration holder, then remove
   `/spinning/gpu-arb/holder`.
6. **Restore serving** per the standing rotation state: production 30030 and
   its watchdog are OFF by operator order while `/spinning/PRODUCTION_STOPPED`
   exists, and no strand owes a restore while it does. If that guard file is
   gone, whoever stopped serving brings it back and verifies with a real
   request, not a health probe.
7. **Never touch port 30099** (the local router), and never `pkill` broadly —
   only PIDs this window started.

---

## 11. DESK VALIDATION ALREADY DONE (CPU, branch `feat/desk-363-act`)

| Item | Result |
|---|---|
| `--regime-trace PATH` writes **`PATH.rank<N>.jsonl`**, one file per rank | verified in `regime_runtime._rank_trace_path`: the flag stays a single path and the reader gets its per-rank files. A `{}` placeholder is NOT substituted — an older runsheet's `rank{}.jsonl` would produce files literally named `rank{}.rank0.jsonl`. |
| Flags `--regime-controller`, `--regime-trace`, `--regime-gate-evidence`, `--regime-stage-clock`, `--kv-reshard-vectors`, `--enable-vram-dial`, `--rank-gpu-id`, `--rank-tp-ratio`, `--rank-auto-reserve-mib`, `--max-running-requests`, `--speculative-*`, `--enable-metrics` | **PASS — verified by BUILDING the parser** (`ServerArgs.add_cli_args`) and reading `_actions`, not by grep: the flags are derived from annotated dataclass field names, so a literal search reports a false MISSING (the AUDIT-251 trap). |
| `--regime-interval`, `--regime-log-every` | **DO NOT EXIST** on this line. Any runsheet naming them is stale. |
| `POST /kv_reshard` `{"target_vector": [...]}` | present, `http_server.py:1244`; body type `io_struct.KvReshardReqInput` |
| `corridor_report.py --smoke` | **3/3** — clean PASS, planted 900 MiB sample FAILS, all-999 series FAILS (the string-compare trap that produced a false reassurance in the last window) |
| `preflight_363_window.py --evidence <absent> --strict` | ran on this branch: `flags-parse` **PASS**, `entry-gate` FAIL (all four items missing), `actuator-wired` FAIL, two SKIPs — i.e. it refuses correctly and names what is missing |
| `stage_measure_pass` / `stage_measure_store` | 43 hermetic tests, 4 executed can-fail arms (branch `feat/desk-363-act`; the arms and their red counts are in the commit message) |
| ms-clock decision rule | 14 hermetic tests; the pre-slice threshold reverted as a can-fail arm turns 3 of them red |

**No GPU, no serving process and no model were touched to produce this
runsheet.**

---

## 12. OUT OF SCOPE

* Combination with PP / DP / EP — the axis inherits #656's scope.
* **Re-tuning `spread_veto_pct` or `kv_ascend_mark` to make gate 3 pass.**
  Both are decisions owed elsewhere (`bands.py`'s own note assigns
  `kv_ascend_mark` to #287); a measurement window that re-tunes its own gate
  has measured the tuning.
* The weight-cut axis: no runtime actuator, so a stage differing in it is
  reported and never selected (#354/#357).
* Any second measurement store. The canon is one file with one identity; a
  second one is how two numbers for the same stage start disagreeing.
