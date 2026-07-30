# Task 285 -- DFLASH against NEXTN on structured output

Persistent record for the s16 arm: what was surveyed upstream before building,
which decisions the harness encodes, and the exact command the GPU window runs.
Written so the next agent does not have to reconstruct any of it from a chat
transcript.

## 1. Upstream survey (sgl-project/sglang, checked 2026-07-30)

Scope of the check: does upstream already carry (a) a DFLASH-like draft, (b)
runtime selection between draft algorithms, (c) adaptive k / draft length,
(d) standalone or solo draft placement. Only sglang was surveyed.

| capability | upstream equivalent | links | maturity | coverage vs our fork | recommendation |
|---|---|---|---|---|---|
| (a) DFLASH-like draft | Yes, `SpeculativeAlgorithm.DFLASH`, bound to the Inkling draft-checkpoint architecture (target hidden-state taps, block-size overrides) | [#31840](https://github.com/sgl-project/sglang/pull/31840) merged 2026-07-27, [#31372](https://github.com/sgl-project/sglang/pull/31372); follow-ons [#32595](https://github.com/sgl-project/sglang/pull/32595), [#32417](https://github.com/sgl-project/sglang/pull/32417), [#32798](https://github.com/sgl-project/sglang/pull/32798), [#32175](https://github.com/sgl-project/sglang/pull/32175) | merged, several PRs per day through 2026-07-29 | Name identical, mechanism not verified line by line. Our fork's DFLASH commits sit ON TOP of the upstream worker (GGUF-target lm_head sampling, solo draft-KV pool, ladder integration) | Ours is a delta on upstream, not a parallel implementation. Keep tracking upstream; do not describe DFLASH itself as a fork feature |
| (b) runtime switching between algorithms | Not found | -- | -- | Upstream's adaptive machinery varies the step count WITHIN one fixed algorithm; nothing swaps EAGLE/DFLASH/NGRAM at runtime | Fork delta stands (#156 cross-algo ladder) |
| (c) adaptive k / draft length | Yes, mature | [#23994](https://github.com/sgl-project/sglang/pull/23994), [#27493](https://github.com/sgl-project/sglang/pull/27493), [#24055](https://github.com/sgl-project/sglang/pull/24055), plus `adaptive_runtime_state.py` / `AdaptiveController` on main | merged, iterated April-July 2026 | Architecturally close to #75/#138 | Build on the upstream primitives rather than a parallel path; this matches the existing provenance note that the adaptive controller is upstream, and our slice is graph-offload plus the high-accept ladder |
| (d) standalone / solo placement | Partial, and the word means something else | [#12625](https://github.com/sgl-project/sglang/pull/12625) merged 2025-12-30, [#29622](https://github.com/sgl-project/sglang/pull/29622) | merged, stable | Upstream `STANDALONE` = a draft that does not share embeddings/lm_head with the target, still in-process and in the same TP group. No dedicated-GPU or separate-process placement upstream | Fork delta stands (#153/#155 `--speculative-draft-placement solo` + `--speculative-draft-gpu`). Do not conflate the two meanings of "standalone" in documentation |

Caveats worth carrying forward: (a) was not diffed line by line, so "coverage"
for the DFLASH row is unresolved; the search terms for (b) were our vocabulary
("bandit", "runtime switch"), not necessarily upstream's.

## 2. What the s16 arm measures, and why it is built this way

The claim under test is narrow: DFLASH is documented in this fork as weak on
prose and strong on format-constrained text, and every number behind that
documentation was measured on prose or on multiturn chat. The crossover suite
put DFLASH at +6 % short-context, parity long, and 18.9-20.9 % behind NEXTN in
the multiturn regime. Nothing measured it on code, JSON or tables.

Decisions the harness encodes:

* **FP8 vehicle, not GGUF.** `Qwen3.6-27B-FP8` as target, `qwen3.6-27b-dflash`
  as drafter. #290 contaminates the GGUF path and would sit inside the
  difference this step reads.
* **Split placement on both arms.** Where the drafter lives is a separate
  question (#153/#155); mixing it in would make every cell of the table need
  two explanations.
* **Interleaved by rounds, not blocks.** The speculative algorithm is a boot
  flag, so request-level interleaving is impossible without the cross-algo
  ladder -- and the ladder carries a standing 13-15 % capture tax that would
  land inside the reported number. Rounds walk the arm list in the same order,
  so drift hits both arms alike.
* **A-vs-A floor first.** Round 0 boots the NEXTN recipe twice under two names;
  their per-cell spread is the gate. Anything under it is `~` and is not a
  finding. Without the floor round the analysis marks every delta unverdicted.
* **The content class is a row, never an average.** Three classes, each its own
  cell at each batch size.
* **Every point is output-validated.** Python/Bash parse, strict JSON plus
  required keys, row count of the requested kind. A point under
  `--min-valid-ratio` is written out with `counted: false` and dropped from the
  tables but named in the report.
* **Prompts are independent.** No model output is ever fed back in as a
  follow-up prompt (#156 self-conditioning trap), and a unit test asserts it.
* **The KV pool can be pinned.** The DFLASH drafter costs weights the NEXTN
  head does not, so at equal reserve the arms end up with different KV
  capacities. `S16_MAX_TOTAL_TOKENS` equalises them; the analysis prints both
  capacities and says so when they differ.

## 3. Files

| file | role |
|---|---|
| `scripts/gpu_battery/prompts/structured_v1.json` | 18 prompts, 3 classes (6 each), mixed de/en, one validator spec per prompt |
| `scripts/gpu_battery/s16_dflash_structured.sh` | orchestrator: floor round, comparison rounds, boot template, budget wall |
| `scripts/gpu_battery/s16_structured_point.py` | one (arm, bs, class) point: prompt pool, window, tick harvest, meta_info accept, output validation |
| `scripts/gpu_battery/s16_analysis.py` | floor gate, per-class comparison tables, validation and reject tables, KV-pool proof |
| `test/registered/unit/distributed/test_s16_structured_arm.py` | hermetic CPU test (24 cases) |

## 4. The GPU window

Calibration boot first (one arm, one class, to read the DFLASH KV capacity):

```bash
export BATTERY_RUN=/spinning/gpu-battery-results/$(date +%F)_dflash_structured
export BATTERY_STEP=s16_dflash_structured
export BATTERY_STEP_DIR=$BATTERY_RUN/$BATTERY_STEP
export WT=/spinning/wt-dflash-structured
mkdir -p "$BATTERY_STEP_DIR"
S16_ONLY=dflash S16_FLOOR=0 S16_ROUNDS=1 S16_BS=1 S16_CLASSES=code_completion \
  bash /spinning/wt-dflash-structured/scripts/gpu_battery/s16_dflash_structured.sh
grep max_total_num_tokens "$BATTERY_STEP_DIR/proofs/dflash_r1.txt"
```

Then the full run with both arms on the same pool:

```bash
export S16_MAX_TOTAL_TOKENS=<the number from the calibration boot>
bash /spinning/wt-dflash-structured/scripts/gpu_battery/s16_dflash_structured.sh
/spinning/htsglang-gpu/.venv/bin/python \
  /spinning/wt-dflash-structured/scripts/gpu_battery/s16_analysis.py \
  --step-dir "$BATTERY_STEP_DIR" --json "$BATTERY_STEP_DIR/summary.json"
```

Open risk for that window: the reserve vector `4500,4200,4200` is carried over
from the proven bar1_hi recipe and has not been booted with a DFLASH drafter
sharded on top. If the DFLASH arm fails to boot, the reserve is the first thing
to raise (`S16_RESERVE`), and the VRAM corridor rule decides how far.
