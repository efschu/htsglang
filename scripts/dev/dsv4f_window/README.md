# DSV4F window, 2026-08-04 — run order and paste-ready commands

Everything in this directory is **DESK-WRITTEN**. Only the hermetic smokes in
`MOCK_SMOKE.md` have been executed; no arm has touched a GPU.

Worktree `/spinning/wt-dsv4f-window`, branch `feat/dsv4f-window-2026-08-04`.
Artifacts land in `$RUN`, default
`/spinning/gpu-battery-results/2026-08-04_dsv4f_window/`.

---

## 0. Read this first: NVML index ≠ CUDA ordinal on this rig

Permanently, not as boot-to-boot drift. `nvidia-smi`/NVML enumerates by PCI
bus; torch defaults to `CUDA_DEVICE_ORDER=FASTEST_FIRST` and puts the 5090
first.

| | NVML 0 | NVML 1 | NVML 2 |
|---|---|---|---|
| card | RTX 3080 (05:00.0) | **RTX 5090 (0A:00.0)** | RTX 3080 (0B:00.0) |
| CUDA ordinal | **1** | **0** | **2** |

Which space each flag takes:

| flag / tool | space | source |
|---|---|---|
| `--rank-gpu-id` | CUDA ordinals | `server_args.py:8476-8477` (`gpu_id_for_rank` returns `rank_gpu_id[world_rank]`) |
| `--rank-auto-reserve-mib` | per RANK, zipped positionally against `--rank-gpu-id` | `server_args.py:9111` |
| `--rank-moe-resident-fraction` | per RANK, same zip | same |
| `--speculative-draft-gpu` | **CUDA ordinals** | `server_args.py:3581-3589`, verbatim: *"CUDA device index (torch.cuda order, same space as `--rank-gpu-id`)"* |
| `nvidia-smi --query-gpu=...` | NVML indices | — |

Consequences, both load-bearing:

* The proven recipe's `--rank-gpu-id 0,1,2` is **correct as written**: rank 0 →
  cuda:0 → the 5090, which is why rank 0 carries the largest reserve and
  resident fraction (and is the clock rank, #439). Do not "fix" it.
* `--speculative-draft-gpu` must be **0**, the 5090's CUDA ordinal — *not* 1,
  its NVML index. Passing 1 does **not** error (rank 1 legitimately maps to
  cuda:1); it silently puts the DSpark solo draft head on a 3080, where the
  MXFP4 Marlin path does not exist (SM90/SM120 only). Silent wrongness, so
  `lib.sh::assert_draft_gpu_is_5090` asserts it and prints both orderings on
  failure.

Never line a per-rank quantity up against an `nvidia-smi` reading directly.
Use `lib.sh::nvml_index_for_rank` / `vram_free_mib_for_rank`, which bridge by
UUID. `resolve_cards` writes both orderings to `$RUN/device_order.json` (plus a
per-arm copy) for every arm, so every artifact records which space it used.

---

## 1. WINDOW CHECKLIST

Arm priority, highest first. Time is the scarce resource; take them in order
and stop where the window ends.

| # | arm | gates on | why this priority |
|---|---|---|---|
| 1 | **#478 quant swap** (`boot_478_quant.sh`) | nothing | Two boots, both must be in the SAME window at the SAME power state or the comparison is void. Highest information per boot. |
| 2 | **#470 DSpark** (`boot_470_dspark.sh`) | Boot A before Boot B, enforced | Carries a **correctness** question (ANALYSE_447 §2.4) that outranks every perf number in the window. |
| 3 | **#462 F2** (`boot_462_f2.sh`) | eager control → F2 → §5 A/B, enforced | Gates default-on for the breakable route and every performance claim about it. |
| 4 | **#390/#394 expert stats** | — | **No boot of its own.** `SGLANG_EXPERT_STATS=1` is armed in EVERY arm's env by `export_base_env` because it is free; arm 4 is harvested from the other three boots. Each boot copies its dump to `*.preteardown` *before* teardown — the SIGTERM revision left on disk is not the headline artifact. |

Standing rules that apply to every arm:

* Hold `/spinning/gpu-arb/` for the whole window. Stop the heartbeat **before**
  releasing. `arb_claim` preserves the incumbent SERVING holder line verbatim
  to `$RUN/holder_serving_original.txt` — `restore_serving.sh` refuses without it.
* `power_tag <arm> start` and `power_tag <arm> end` on every arm. The user
  lowered all power targets on 2026-08-03; **old full-power baselines are dead**
  and any number without its power state compares to nothing.
* Every boot carries `--enable-metrics` (`assert_metrics_flag` refuses otherwise),
  `--reasoning-parser deepseek-v4`, `--tool-call-parser deepseekv4` and
  `--chat-template`.
* A-vs-A floor before any delta. `probes.py report_delta` refuses to print a
  delta smaller than its own point's floor.
* `py-spy dump` before any kill; TERM the recorded pgid only; never `pkill`.
* If a `cachetrim.sh` is started, **stop it at server-ready**. Leaving it
  running during serving inflated the A-vs-A floor from 2.55 % to 39.91 %.
  Nothing in this directory starts one.

---

## 2. Files

| file | what it is |
|---|---|
| `lib.sh` | shared helpers: `preflight`, `power_tag`, `resolve_cards`, `wait_ready`, `stop_server`, `rammon_start/stop`, `arb_claim`/`arb_heartbeat_stop`, the index-space assertions |
| `dsv4f_chat_template.jinja` | the model's chat template, **extracted from the GGUF metadata**, not written by hand |
| `extract_chat_template.py` | extracts / verifies / self-tests that template |
| `logging_break_debug.json` | `SGLANG_LOGGING_CONFIG_PATH` config: DEBUG on exactly one module, for the #462 independent break count |
| `boot_478_quant.sh` | arm 1 |
| `boot_470_dspark.sh` | arm 2 (two-boot gate, ordering enforced) |
| `boot_462_f2.sh` | arm 3 (§-order enforced) |
| `probes.py` | all measurement; `--selftest` is hermetic |
| `restore_serving.sh` | mandatory end-of-window restore |
| `MOCK_SMOKE.md` | exactly what was executed at desk time, and what was not |

### The chat template — a correction to the briefing

The briefing said the GGUF checkpoints carry no chat template and told me to
*produce* one. Half right:

* **TRUE**: neither quant dir's `tokenizer_config.json` has a `chat_template`
  key (and both have empty `added_tokens_decoder`), so the tokenizer sglang
  loads carries none and `--chat-template` really is required.
* **FALSE**: the GGUF files themselves **do** carry
  `tokenizer.chat_template` — 13698 bytes of Unsloth-fixed DeepSeek-V4 jinja,
  sha256 `e643c31f…`, **byte-identical between UD-IQ3_XXS and UD-Q3_K_XL**.

So the template is extracted, not invented. A desk-written one would have been
an unvalidated guess about the fullwidth `<｜User｜>`/`<｜Assistant｜>` markers,
the `<think>` handling and the `<｜DSML｜>` tool-call block, with no way to tell
a wrong guess from a right one. That the two quants agree also matters for arm
1: if they disagreed, the #478 quant swap would silently also be a prompt-format
swap.

The chat-format probe (`probes.py chatprobe`) proves the template was *applied*,
not merely that the server answered: it renders the same conversation locally
with the template file the boot passed, sends that string through native
`/generate`, sends the raw messages through `/v1/chat/completions`, and requires
the two greedy outputs to agree — with a **negative control** (naive role-less
concatenation) that must *disagree*. If the negative control agrees too, the
probe reports `instrument void` rather than a pass.

---

## 3. Commands, in order

### Setup (once, at the top of the window)

```bash
export RUN=/spinning/gpu-battery-results/2026-08-04_dsv4f_window
export PYTHONPATH=/spinning/wt-dsv4f-window/python
cd /spinning/wt-dsv4f-window/scripts/dev/dsv4f_window

# claim the cards (preserves the incumbent SERVING holder line verbatim)
bash -c '. ./lib.sh; arb_claim "DSV4F window 2026-08-04 (#478/#470/#462)"'

# instruments must pass their own can-discriminate check before any arm
/spinning/htsglang-gpu/.venv/bin/python probes.py --selftest
/spinning/htsglang-gpu/.venv/bin/python extract_chat_template.py --selftest
/spinning/htsglang-gpu/.venv/bin/python extract_chat_template.py --verify
```

### Arm 1 — #478 quant swap (both arms, same window, same power state)

```bash
ARM=iq3xxs ./boot_478_quant.sh
ARM=q3kxl  ./boot_478_quant.sh
```

Optional overrides (the per-rank vectors are DEFAULTS ONLY — recompute them
from the GGUF footprint analysis):

```bash
RESIDENT_FRACTION=0.485,0.42,0.42 AUTO_RESERVE_MIB=2200,1400,1400 \
  ARM=q3kxl ./boot_478_quant.sh
```

Before quoting anything, diff the power states:

```bash
diff <(jq -S . "$RUN/powerstate_478_iq3xxs.json") <(jq -S . "$RUN/powerstate_478_q3kxl.json")
```

### Arm 2 — #470 DSpark (two-boot gate)

```bash
SUBARM=a_base   ./boot_470_dspark.sh      # baseline residency, no draft
SUBARM=a_cut    ./boot_470_dspark.sh      # residency cut by ~11 GiB, no draft
SUBARM=b_dspark ./boot_470_dspark.sh      # the DSpark arm (REFUSES without both above)
```

`b_dspark` refuses unless `probes_470_a_base_all.json`,
`probes_470_a_cut_all.json` and `idem_reference_470_a_cut.json` all exist.
TICKET_470 §5: *"If Boot A cannot be run at all, do not run Boot B: an
unattributed multiplier is not a result."*

**Boot A deliverable is ONE number**: the decode cost, in ms/round, of making
room for the head (`a_cut` vs `a_base`), gated on the larger of the two arms'
own A-vs-A floors.

**Deviation, stated**: TICKET_470 §2 wants the cut measured in the *same* boot.
`--rank-moe-resident-fraction` is a launch-time `ServerArgs` field and there is
no runtime endpoint that changes it (searched `http_server.py` and
`srt/managers/` — none exists). The ticket's own escape clause applies: *"If the
cut cannot be expressed as a budget knob on this build, say so and price it by
the closest available lever rather than skipping the boot."* So Boot A is two
boots, each carrying its own floor — strictly more conservative than a
within-boot delta.

`RESIDENT_FRACTION_CUT` defaults to `0.383,0.42,0.42` (= 0.485 × 0.79, from
TICKET_470 §0's ~21 % figure). That is **arithmetic from the ticket, not a
measurement** — §7.6 says so itself. Override it:

```bash
RESIDENT_FRACTION_CUT=0.NNN,0.42,0.42 SUBARM=a_cut ./boot_470_dspark.sh
```

The script asserts §3.1 checks 1 and 3 hard (solo placement announced; the
marlin runner reached the draft's expert layers via `Preparing MXFP4 experts
for Marlin backend`, verified at `mxfp4_marlin_moe.py:133`). Check 2 (markov_w2
TP-shard disabled) only **warns**, because that log line's exact wording is not
pinned anywhere in this tree — read the log and record which it was.

### Arm 3 — #462 F2 (§-order enforced)

```bash
ARM=eager           ./boot_462_f2.sh    # §3 control. Must run first.
ARM=f2              ./boot_462_f2.sh    # §3 F2 + probe ON, then §4 evidence
ARM=breakable_clean ./boot_462_f2.sh    # §5 ms/verify A/B, probe OFF
```

`f2` refuses without the eager control; `breakable_clean` refuses without
`$RUN/F2_break_cost.txt`.

Readout (the boot script runs it; here it is for a manual re-run):

```bash
python3 /spinning/wt-dsv4f-window/scripts/dev/494_break_cost/summarise.py \
    --drop-rounds 20 "$RUN"/break_cost.rank*.jsonl | tee "$RUN/F2_break_cost.txt"
```

**Contradiction, resolved in favour of the code.** Both the briefing
(`SGLANG_BREAK_COST_PATH="$RUN/break_cost"`) and TICKET_462 §3
(`"$RUN/break_cost.jsonl"  # becomes one file per rank`) assume the path is
expanded per rank. **It is not.** `break_cost_clock.py:513` reads

```python
path = os.environ.get(ENV_PATH) or f"/tmp/break_cost.{rank_tag}.jsonl"
```

and uses a user-supplied value **verbatim** — no rank tag is interpolated. Set
it and all three TP ranks append to one file, and the ticket's own glob
`"$RUN"/break_cost.rank*.jsonl` matches nothing. So `boot_462_f2.sh` leaves
`SGLANG_BREAK_COST_PATH` **unset**, takes the documented per-rank default in
`/tmp`, clears stale files before the boot, and copies the results into `$RUN`
afterwards.

The script asserts `crossings/round == 43` per rank and cross-checks it against
an **independent** DEBUG count of `Break graph due to function:
_moe_offload_fetch_step` — enabled on exactly that one module via
`logging_break_debug.json`, not by a global `--log-level debug`.

**Verdict rule**: `43 × (break + rendezvous + planning + publish)` against the
launch-overhead saving the graph buys. Report both numbers and the ratio.
**No kill threshold** — Aufwand/Ertrag decides, and a small win that is cheap
to keep is still a win. Do not quote F1's 5.3–8.4× (a *ceiling* measured on
Qwen3.6-35B-A3B). The 43 rendezvous/step are irreducible on this route
(DESIGN_462 §4); their removal is not an achievable optimisation.

### End of window — MANDATORY

```bash
./restore_serving.sh
```

Re-boots INT8-W8A8 serving via `/tmp/w530_boot.sh` (re-executed, not
reimplemented, so there is one recipe and not two that drift), smokes `/health`
and one MT probe with a discriminating check that the answer is actually
translated, checks the translator front door on 30800, **stops our heartbeat
before releasing**, then restores the SERVING holder line byte for byte from
`$RUN/holder_serving_original.txt`.

The translator front door runs as its own process
(`/spinning/wt-466-translator/scripts/translator/`); the engine restart does
**not** restart it. If smoke 3/3 warns, bring it back and run
`front_door_test.py --url ws://127.0.0.1:30800` before reporting the window closed.

---

## 4. Reporting rules

* **ms/verify and ms/prefill. Never tok/s** as a headline (`probes.py` carries
  tok/s only as `tok_per_s_secondary`).
* Every quoted delta names its own point's A-vs-A floor. A delta inside the
  floor is reported as *"inside the floor"* — that is the result.
* Accept length is `meta_info.spec_accept_length` from native `/generate`, or
  the `Decode batch` tick line. **Never** `spec_ema_accept_len`, which is a
  server-lifetime EMA and is recorded only as provenance.
* The llama.cpp 0.49–0.77 accept band (PR #25784) is *their* domains, order of
  magnitude only, never a 1:1 comparison against
  `2026-08-03_447_dspark/prompts.json` (whose own provenance block says so).
* Corridor readings sampled during load, per card, minimum stated.
* Every arm's numbers carry the power state from `powerstate_<arm>.json`.
* Anything unmeasured stays labelled unmeasured.
