# RUNSHEET — #363 stage-clock measurement window

**One file, executable top to bottom.** Claim, preflight, boots in order,
extraction, teardown, restore. Everything desk-checkable has been desk-checked
on the merged line and the results are below; everything else is a card step.

**Derived from** `integration/r2` @ `9cedf43811` (re-resolved 2026-08-14 — the
line had advanced from `99a4b0a493`). Branch `chore/ticket-363-window`,
worktree `/spinning/wt-363-window`,
`PYTHONPATH=/spinning/wt-363-window/python`.

> `TICKET_363_STAGE_CLOCK.md` still points at branch
> `feat/regime-stage-actuator-363` and worktree `/spinning/wt-363-stages`.
> **That pointer is stale** — the actuator merged as `b2f0a749ac` and rides the
> merged line flag-gated OFF (`c3919fe1cb`). Use this runsheet's paths.

---

## 0. FINDINGS FIRST — four things that would have cost the window

All four were found at the desk, against the merged line's own code. Each one
would have surfaced as a refusal or a meaningless result *after* the cards were
claimed.

### F1 — `act` is refused at parse time, and gate 4 is CIRCULAR. **BLOCKING.**

`--regime-controller act` is refused until `--regime-gate-evidence` names
**all four** entry-gate items with `passed: true` **and** a non-empty `source`
(`regime_stages.py:465-511`; `EntryGate.open` at `:509-511` is
`not missing and not malformed` — no subset is accepted).

The fourth item, `f4_card_comparison`, is produced by *"three arms (off /
observe / act) over one workload"* (`regime_stages.py:452-459`). **So gate 4
needs an `act` arm, and `act` needs gate 4.**

`RUNSHEET_363_card_gates.md` §7 says gate 4 *"needs gates 1–3 in the evidence
file first"*. **That does not match the code**, which needs four. A window that
trusts §7 will hit a parse-time refusal. §3 below is the honest way through.

### F2 — the ticket's boot line is missing a required actuator flag. **BLOCKING.**

`act` additionally refuses unless at least one of `--kv-reshard-vectors` /
`--enable-vram-dial` is given (`server_args.py:7784-7793`) — otherwise "every
proposal would be refused for want of an actuator and 'act' would be an
expensive 'observe' under a misleading name". `TICKET_363_STAGE_CLOCK.md` §4
lists neither. The boots in §4 below add `--kv-reshard-vectors`.

### F3 — the trace has no wall clock, so A3's "last 120 s" is not selectable.

Trace records carry `round` and `epoch` and **no timestamp**
(`regime_runtime.py:424-459`). "The mean ms/round over the last 120 s of the
SHIFT phase" (ticket A3) cannot be evaluated from the trace alone. Segment
instead by `--phase-regime prefill_heavy` (the label the controller itself
assigned) or by an explicit `--from-round/--to-round` taken from the workload
driver's phase log. Both are supported by the extractor; neither is a guess.

### F4 — A3's compute/wait split cannot be settled by arm-vs-control.

`ms_decision` is `None` on every boot without `--regime-stage-clock`
(`regime_runtime.py:456-457`) — which is precisely the control arm. So the
control carries **no** compute/wait split, and the arm-vs-control pair can
answer "did ms/round improve" but **not** "was it the wait term that moved".
The split claim must be read from arms that carry the clock: the A-vs-A pair
(§4 boots B2/B3) and the arm's own across-phase movement. The extractor prints
this rather than silently comparing a number against nothing.

---

## 1. Pre-steps the GPU shift must do BEFORE this window

This window is **not** first in the queue. In order:

| # | Pre-step | Why | Owner |
|---|---|---|---|
| P0 | Run the CPU preflight (§2). Costs seconds, no cards. | Catches F1/F2 before a claim | anyone |
| P1 | **Gate 3 reachability**: close the two `UNREACHED` findings — `spread_veto_pct = 25` (measured peak **0.68 %**) and `kv_ascend_mark = 0.85` (at **16.5 %** peak occupancy). | `f3_bands_measured` is one of the four gate items | GPU shift |
| P2 | **A busier/longer workload** for the gate-3 pair. The re-record produced **41 and 56** active boundaries against `MIN_PAIRED_SAMPLES = 8`; a duty cycle from 41 windows has SE ≈ 0.07, most of the 0.103 disagreement `enter_prefill` had to clear. Both passing verdicts are thin (**1.29×** and **1.14×**). The runsheet's own words: a longer or busier workload "is the only thing that will" buy margin. **Do not change workload flags between the two gate-3 arms.** | Makes gate 3 defensible rather than thin | GPU shift |
| P3 | **Gate 4** (`off`/`observe`/`act`, one workload) — needs the §3 bootstrap. | Produces `f4_card_comparison` | GPU shift |
| P4 | **Stage table with ≥1 MEASURED non-reference stage** (ticket P1). A planner-only table runs clean and proves nothing. If every non-reference stage is unmeasured, **STOP** — that is #584's measurement pass, not this ticket. | The axis never proposes an UNMEASURED stage | GPU shift |
| P5 | **Transient census** per stage, under **both** load states the window visits (deep-prefill and decode-heavy): `SGLANG_TRANSIENT_CENSUS=1` (`environ.py:667`), `SGLANG_RESIDENCY_CENSUS_DIR=<dir>` (`:659`). A partial set is refused loudly by `pp_cut_calibration` and that refusal is correct. | Without it every flip is REFUSED, not priced at zero | GPU shift |

### The gate state, MEASURED on this box (2026-08-14, CPU, read-only)

Not inferred from the runsheets — the preflight was pointed at the real
evidence files on disk:

| Evidence file | Verdict |
|---|---|
| `/spinning/gpu-battery-results/2026-08-01_363_gates_388/regime-gate-evidence.json` | **CLOSED** — missing `f3_bands_measured`, `f4_card_comparison` |
| `/spinning/gpu-battery-results/2026-08-01_363_gates_rerun/regime-gate-evidence.json` | **CLOSED** — missing `f3_bands_measured`, `f4_card_comparison` |

Both carry gates 1+2 (`desyncs_zero`, `f2_live_replay`) with sources. **No
evidence file on this box opens the gate today**, and `/spinning/evidence-363/`
does not exist — the path used in §2's example is a placeholder the shift must
point at the real file. So `--regime-controller act` would be refused right
now, on either file, exactly as F1 predicts. **Two items outstanding, and they
are P1 and P3 below.**

**Only when P1–P5 are done does this window have anything to measure.**

---

## 2. Preflight — CPU only, run before claiming cards

```bash
cd /spinning/wt-363-window
export PYTHONPATH=/spinning/wt-363-window/python
CUDA_VISIBLE_DEVICES=99 python scripts/regime_363_window/preflight_363_window.py \
  --evidence   /spinning/evidence-363/gate_evidence.json \
  --stage-table $OUT/stage_table.json \
  --census-dir /spinning/evidence-363/census \
  --stage prefill_deep --stage decode_heavy \
  --kv-reshard-vectors '2,11,10' \
  --strict
```

Exit 0 = clear to claim. Exit 1 = the reason is printed and named.

**Already verified on this line (2026-08-14, CPU):**

* `flags-parse` **PASS** — all 5 window flags accepted by `server_args`.
  Verified by *parsing*, not grepping: a literal search for
  `"--regime-stage-clock"` finds **nothing**, because the flags are derived
  from annotated dataclass field names (`regime_stage_clock: A[bool, ...]`,
  `server_args.py:5327`). This is the AUDIT-251 assembled-name trap; a grep-based
  "flag exists" check reports a false MISSING here.
* `preflight --smoke` **6/6** cases red-on-demand, driven against the real
  `EntryGate` (not a stub) — including the gate-4-only-missing case, which is
  how F1 was proven rather than argued.

---

## 3. The gate-4 bootstrap — the only honest way past F1

Gate 4 cannot be produced without `act`, and `act` cannot start without gate 4.
The gate is **deliberately openable** — its own docstring says a gate that can
only refuse "is not a gate, it is a disabled feature with extra words".

So the bootstrap is: write `f4_card_comparison` with a `source` that **says it
is a bootstrap**, run the three arms, then **rewrite the file with the real
result**. The `source` field is free text and is the whole audit trail.

```jsonc
{
  "desyncs_zero":      {"passed": true, "source": "R1 gates1+2 re-record, 147105 verdicts, 0 desyncs, 36 transitions"},
  "f2_live_replay":    {"passed": true, "source": "R1 f2_replay.py, open-loop replay matched"},
  "f3_bands_measured": {"passed": true, "source": "R2/R3 distributional bands, <fill in verdicts>"},
  "f4_card_comparison":{"passed": true, "source": "BOOTSTRAP ONLY — authorizes the act arm that PRODUCES gate 4. Not a result. Replace with the measured verdict before any stage-clock window."}
}
```

**Rules on this file, and they are not optional.**

1. The bootstrap `source` must say it is a bootstrap, in those words. An
   unattributed pass is already refused by the code; a *misattributed* one is
   worse, because it reads as evidence.
2. **Rewrite it immediately after the gate-4 run** with the measured verdict.
   A bootstrap left in place silently authorizes every later `act` boot.
3. If gate 4 **fails**, the file goes back to three items. Do not leave a
   passing bootstrap next to a failing measurement.
4. Record the bootstrap in the window log, so a later reader sees that the
   first `act` arm ran under one.

---

## 4. Window claim, then the boots in order

### 4.0 Claim

```bash
mkdir -p /spinning/gpu-arb
echo "363-stage-clock $$ $(date -Is)" > /spinning/gpu-arb/holder
# heartbeat in its own process; STOP THE HEARTBEAT BEFORE RELEASING
```

Whoever stops serving owns bringing it back.

### 4.1 Common launch line

Verified against `server_args` on this line (§2). `$OUT` is a fresh directory
per boot.

```bash
MODEL=$MODEL_ROOT/Qwen3.6-27B-FP8
COMMON="--tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance
  --rank-auto-reserve-mib 5500,3800,3800
  --model-path $MODEL --context-length 32768
  --kv-cache-dtype fp8_e4m3 --speculative-algorithm NEXTN --trust-remote-code
  --enable-metrics --enable-metrics-for-all-schedulers"
ENVV="SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 LD_LIBRARY_PATH=<venv>/nvidia/cu13/lib"
```

### 4.2 The boots

Nothing may differ between B2 and B3 except the output path, or they stop
being an A-vs-A pair.

| Boot | Adds | Purpose |
|---|---|---|
| **B1 — control** | `--regime-controller act --regime-gate-evidence $EV --kv-reshard-vectors '2,11,10' --regime-trace $OUT/b1-rank{}.jsonl` (**no** `--regime-stage-clock`) | A3's control arm. Carries no `ms_decision` by construction (F4). |
| **B2 — A-vs-A #1** | B1 **plus** `--regime-stage-clock` | Floor repeat 1, and carries the compute/wait split. |
| **B3 — A-vs-A #2** | identical to B2, new `$OUT` only | Floor repeat 2. **The floor comes from this pair, before any arm is judged.** |

**Load profile — ONE window per boot, the shift happens inside it.** Do not
restart between phases; the claim is about the TRANSITION, and a restart
replaces it with two steady states.

| Phase | Duration | Shape |
|---|---|---|
| Settle | 120 s | decode-heavy, bs 4, short prompts |
| SHIFT | 300 s | deep-prefill burst: long prompts, few outputs |
| Return | 300 s | decode-heavy again, identical to settle |

Driver (same flags on every boot; heavier than the gate-3 default per P2):

```bash
python scripts/regime_gates/workload.py \
  --repeats 2 --burst 8 --burst-tokens 900 \
  --drain 12 --drain-tokens 900 --mixed 8 --idle-s 25
```

**Check early, before spending the window:**

```bash
grep -m5 REGIME-OBSERVE $OUT/boot.log | grep -o 'prefill [0-9]*%'
```

A run still reporting `prefill 0%` through a prefill burst means the #388
attribution did not reach this boot — check the branch before continuing.
(#388 landed on this line: `f7ce5b36ca` item A, `9f8e658e15` item B, merged
`98495afc4c`.)

### 4.3 Corridor sampling, for A4

Sample NVML **FREE** per card at 100 ms for the whole window — a time-series
minimum, never a boot snapshot, and never `total - used` (a carve-out of
~424/518 MiB is invisible to the subtraction).

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  -lms 100 > $OUT/corridor.csv &
```

---

## 5. Extraction

```bash
cd /spinning/wt-363-window && export PYTHONPATH=/spinning/wt-363-window/python

# A1/A2/A5 -- proposals, actuations, flips, desyncs, summary present
python scripts/regime_gates/readout.py --trace "$OUT/b2-rank*.jsonl" --ranks 3

# A3 -- ms/round, compute vs wait, judged against the A-vs-A floor
python scripts/regime_363_window/msround_split.py \
  --arm     $OUT_B2/b2-rank0.jsonl --arm $OUT_B2/b2-rank1.jsonl --arm $OUT_B2/b2-rank2.jsonl \
  --control $OUT_B1/b1-rank0.jsonl --control $OUT_B1/b1-rank1.jsonl --control $OUT_B1/b1-rank2.jsonl \
  --floor-a $OUT_B2/b2-rank0.jsonl --floor-b $OUT_B3/b3-rank0.jsonl \
  --phase-regime prefill_heavy --warmup 20 --min-samples 30 \
  --require-summary --json

# A4 -- corridor minimum per card, from the time series
awk -F, '{gsub(/ /,"");if($2<m[$1]||!($1 in m))m[$1]=$2}END{for(g in m)print g, m[g]}' \
  $OUT/corridor.csv
```

The extractor **refuses** rather than reporting: a thin segment, a truncated
trace (no summary line), a phase label matching nothing, and any cross-arm
delta with no floor supplied. Proven by `--smoke` (5/5 cases, §7).

---

## 6. PASS criteria — all five, or the window has not passed

Four of five is not a pass.

| ID | Criterion | Source of the number |
|---|---|---|
| **A1** | `stage_clock_proposals > 0` **and** `actuations > 0`, with ≥1 actuation inside the SHIFT phase. Read off the trace's `ms_decision`/`actuated`, not a log line. | ticket A1 |
| **A2** | Total flips over the window **≤ 4**. Two are structural (into the prefill stage, back out). More is thrash. | ticket A2; `flip_cost_s` is what makes it expensive |
| **A3** | ms/round in SHIFT beats the B1 control by **more than the B2/B3 floor**, and the movement is in the **wait** term. Report compute and wait separately. | ticket A3; floor per the ms/round canon. **Read F4**: the split comes from the clock-carrying arms, not from the control |
| **A4** | NVML free per card, 100 ms sampling, **minimum ≥ 1024 MiB** on every card. One breach fails the window regardless of A1–A3 — the gate exists to make it impossible, so a breach is a code defect, not a tight margin. | corridor law; ticket A4 |
| **A5** | `desyncs == 0` in every rank's trace **and the summary line present**. Absent summary ⇒ the run did not end cleanly, so "zero so far" is all it supports. | ticket A5 |

**Watermark policy.** If the measured A-vs-A floor exceeds
`DEFAULT_ENTER_MARGIN_PCT = 5.0` (`regime_ms_clock.py:214`), the watermark
moves **ONCE, before the window**, recorded with its measurement — never
adjusted afterwards until the result is agreeable. Related shipped constants:
`DEFAULT_EXIT_MARGIN_PCT = 2.0` (`:218`), `DEFAULT_ENTER_WINDOW = 2` (`:221`),
`DEFAULT_EXIT_WINDOW = 4` (`:227`). The failure mode of a too-low watermark is
"never flips", not "flips on noise" — the code refuses a flip whose signal does
not clear the measured band.

**If it fails**, the ticket's §6 table maps symptom → cause → next move; the
refusal reason in the trace (`admission.last.reason`) names the load state,
both price terms, and whether the refusal was local or group.

---

## 7. Desk validation already done (CPU, this branch)

| Item | Result |
|---|---|
| `preflight_363_window.py --smoke` | **6/6** red-on-demand against the real `EntryGate` |
| `preflight_363_window.py` on the tree | `flags-parse` **PASS**, all 5 flags accepted |
| `msround_split.py --smoke` | **5/5**: clears a real delta, refuses one inside the floor, refuses thin segments, truncated traces, and non-matching labels |
| `workload.py --help` | imports, parses, exit 0 |
| `readout.py --help` | imports, parses, exit 0 |
| `f2_replay.py --help` | imports, parses, exit 0 |
| `bands.py --help` | imports, parses, exit 0 |

Every script above provably parses and reaches argv assembly under
`CUDA_VISIBLE_DEVICES=99`. No GPU, no serving process, no model was touched.

---

## 8. Teardown and restore

1. **Stop the corridor sampler** (`kill %1`) and confirm `$OUT/corridor.csv`
   has a final line — a truncated series has no minimum.
2. **Stop the server**, confirm the trace summary line exists on **every**
   rank file. Without it A5 is unanswerable and the boot must be repeated.
3. **Rewrite the evidence file** if a bootstrap was used (§3 rule 2). This is
   the step most likely to be forgotten and the one with the longest tail.
4. **Stop the heartbeat BEFORE releasing** the arbitration holder, then remove
   `/spinning/gpu-arb/holder`.
5. **Restore serving** if this window stopped it — whoever stopped it owns
   bringing it back. Verify with a real request, not a health probe alone.
6. Park artifacts: traces, `corridor.csv`, `boot.log`, the evidence file as it
   stood at the end, and the extractor's `--json` output.

---

## 9. Out of scope

* Combination with PP/DP/EP — the axis inherits #656's scope.
* **Tuning the watermarks to make the window pass.** If the floor is wider than
  the mark, the mark moves once, before the window, recorded with its
  measurement.
* The weight-cut axis: no runtime actuator, so a stage differing in it is
  reported and never selected (#354/#357).
