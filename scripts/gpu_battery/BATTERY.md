# GPU test battery

Thirteen steps, ~5 h 09 min of card time, each with a machine-checkable
success criterion. Built so a cheap model can drive it mechanically: run the
step, run the check, PASS = continue, FAIL/STOP = halt and report in
structured form. The executor never judges. Every decision sits in a check
script up front.

Whoever drives the battery reads **EXECUTOR_PROTOCOL.md** — this file is the
reference, the protocol is the instruction.

---

## One-time setup

```bash
export BATTERY_RUN=/spinning/gpu-battery-results/$(date +%Y-%m-%d_%H%M)
export WT=/spinning/wt-gpu-battery              # the worktree under test
mkdir -p "$BATTERY_RUN"
cd "$WT/scripts/gpu_battery"
/spinning/htsglang-gpu/.venv/bin/python battery_state.py --run-dir "$BATTERY_RUN" init
```

`BATTERY_RUN` is the bracket around everything: artifacts, logs, state. A
second run on the same day gets its own directory; a RESUME gets the old one.

## Every step, always the same

```bash
BATTERY_RUN=$BATTERY_RUN bash run_step.sh <step>
```

`run_step.sh` does the following, in this order and without asking: resume
gate, VRAM corridor, take locks (or deliberately not), run the step under a
hard timeout, py-spy dump before every kill on a timeout, release locks, run
the check, write the verdict into `state.json`, print the verdict line.

Exit code 0 = PASS or SKIP, 1 = FAIL, 2 = STOP.

---

## Resume and selection

Green steps are **not repeated by default**. A run that aborted at step 6 is
continued after the bugfix like this:

```bash
export BATTERY_RUN=/spinning/gpu-battery-results/<the old run>
bash run_step.sh s06        # continues at s06, s00-s05 stay green
```

What gets run is freely selectable. The plan says in advance what would
happen:

```bash
PY=/spinning/htsglang-gpu/.venv/bin/python
$PY battery_state.py plan                       # everything still open
$PY battery_state.py plan --only s01,s06,s08    # just these three
$PY battery_state.py plan --from s06            # from here on
$PY battery_state.py plan --to s05              # just the boot queue
$PY battery_state.py plan --skip s03,s04        # without the expensive boots
$PY battery_state.py status                     # what stands where
```

**Legitimate doubt about a green step** — a green run nobody trusts any more
after a fix or a driver change — is forced explicitly, not by deleting files:

```bash
bash run_step.sh s02 --force          # runs again despite PASS
$PY battery_state.py plan --force s02,s03
$PY battery_state.py plan --rerun-all # everything once more
```

Every repeat run counts in `state.json` as a new attempt and pushes the old
verdict into `history`. A forced run is therefore distinguishable after the
fact from a first run.

**Dependencies are artifact dependencies, not an ordering.** s08 needs the
files from s01 and s06 — nothing else. It can be run on its own weeks later
without repeating a single boot. A step whose precondition is not PASS
reports `BLOCKED` instead of running silently.

**Exception: the BAR1 block (s10-s12) is a chain.** Without the patched
driver there is no direct path, and without a run that demonstrably went over
it there is no curve to measure. s11 requires s10 PASS, s12 requires s11
PASS. Conversely, s00-s09 hang off **no** BAR1 step: the battery stays fully
drivable on a stock driver.

---

## The steps

Order as below. Card identity is resolved at runtime via PCI/NVML
everywhere; nowhere is there a fixed index.

### S0 — `s00_preflight` · haiku · ~3 min · repeatable

Card inventory (PCI, UUID, NVML↔CUDA join), VRAM corridor, lock state,
mandatory files, driver/torch/NCCL version.

* **Precondition** none. This is the precondition for all the others.
* **Command** `bash run_step.sh s00`
* **Success** `check_s00_preflight.py`: ≥2 cards, each with PCI+UUID+CUDA
  index (the PCI join MUST succeed — without it, every card index named
  later may mean a different piece of silicon), each with ≥400 MiB free, no
  foreign lock, all mandatory files present, nvidia-smi/curl/py-spy
  available.
* **Abort** every finding here is STOP, never FAIL: nothing has been tested
  yet, so nothing can have failed.
* **Artifacts** `s00_preflight/preflight.json`, `inventory.json`

### S1 — `s01_p2p_reprobe` · haiku · ~10 min · repeatable

The re-probe package after the driver update: capability matrix → d2d bench →
NCCL transport check.

* **Precondition** s00 PASS. **run_all.sh takes the locks itself** —
  run_step.sh only checks that they are free. If it held them, the tool would
  abort on its own lock acquisition.
* **Command** `bash run_step.sh s01`
  (optionally `P2P_BASELINE=<old nccl_transport.json>` for the diff column)
* **Success** `check_s01_p2p_reprobe.py`: envelopes are correct;
  `can_access_peer` decided for EVERY directed pair (None ≠ False); where peer
  access is possible, the **effective aperture fields are populated** (the
  nominal 256 MiB is an upper bound, not a usability guarantee, and every
  consumer ignores it); the 255/256/257 MiB bracket is in the ladder; every
  NCCL pair has a transport finding; and the files load with the **real #279
  loaders** — zero apertures or zero profiles is FAIL.
* **Abort** timeout/crash of a pair = FAIL. No results directory = STOP.
* **Not judged** whether P2P engages. "No P2P anywhere" is a fully collected
  result. `verdict_diff.md` is filled in by the reader, not the executor.
* **Artifacts** `s01_p2p_reprobe/results/{capability_matrix,d2d_bench,nccl_transport}.json`,
  `run.log`, `verdict_diff.md`

### S2 — `s02_boot_a` · sonnet · ~35 min · NOT repeatable · **REPORT GATE**

r7c boot A, Qwen3.6-27B-FP8: the single-axis falsifier. A goes first because
it is the only boot whose outcome changes what the others mean.

* **Precondition** s00 PASS, corridor green, locks free.
* **Command** `bash run_step.sh s02`
* **Success** `check_s02_boot_a.py`: an arm for all five contents (alphabet,
  squares, repeat, code, prose); `accept_len_mean` a real number per arm
  (None is the recipe's own abort criterion: probe off, or the spec path is
  not running); `rounds > 0`; **position curve present** and positions 0..K-1
  covered; **reference column** present and named with its source; MIN free
  MiB per card collected; no OOM/NCCL/traceback in the log.
* **Abort** server not up, OOM, traceback = FAIL (the boot ran and said
  something). No server.log = STOP (the recipe never ran at all).
* **Not judged** the HEIGHT of the accept numbers. Reproduction (2.6–3.3,
  position 0 ~65%) and falsification (~1.5, position 0 24–45%) are both
  results.
* **REPORT GATE** After S2 the run halts and reports, PASS or not, before S3
  starts. That is how the R7c queue specifies it.
* **Artifacts** `s02_boot_a/{accept.json,accept.txt,reference_column.json,vram.csv,vram_summary.txt,cards.txt,server_info.json,server.log,step.log}`

### S3 — `s03_boot_b` · sonnet · ~40 min · NOT repeatable

r7c boot B, Huihui-AWQ-MTP (INT4 body, BF16 head): the second half of A's
question. A moves the target quantization, B lifts only the head.

* **Precondition** s00 PASS. (Artifact-wise independent of S2; the ordering
  is about content, not technology.)
* **Command** `bash run_step.sh s03`
* **Success / abort** as in S2.
* **Known risk up front** AWQ × uneven TP × MTP has never been booted on this
  branch. If the load rejects the shape, that is ONE spent boot and the answer
  is "not on this vehicle". The executor does not tune the ratio inside the
  window — that would be a different experiment.
* **Artifacts** as in S2, under `s03_boot_b/`

### S4 — `s04_boot_c` · sonnet · ~45 min · NOT repeatable

r7c boot C, GGUF-Q3 target plus a quantized DFLASH-Q8_0 drafter solo on one
3080. Third, because it is the boot with the highest retry probability — and a
retry is cheaper once the accept questions are settled.

* **Precondition** s00 PASS.
* **Command** `bash run_step.sh s04`
* **Success** as in S2, **plus**: `loader_lines.txt` proves a loaded drafter.
  A server that only came up on the target would pass every accept check and
  still not have answered C's question.
* **Abort** drafter load throws on a tensor name → FAIL, report the name. OOM
  on the host card → FAIL; raising `RESERVE_HOST` is the business of the NEXT
  run, not of a retry inside the window.
* **Explicitly not an abort** incoherent output. That is a result.
* **Artifacts** as in S2 plus `loader_lines.txt`, under `s04_boot_c/`

### S5 — `s05_boot_d` · sonnet · ~30 min · NOT repeatable

r7c boot D, lane re-seed A/B on the round-7b configuration. The one boot that
is known to come up.

* **Precondition** s00 PASS.
* **Command** `bash run_step.sh s05`
* **Success** `check_s05_boot_d.py`: three contents (squares, code, prose),
  **both** arms per content (an A/B with one arm is not an A/B),
  `accept_len_mean` and `decode_ms_mean` real numbers in both arms (the price
  is the point of this boot and it lives in `decode_ms_mean`),
  `reseed_forwards` set in the re-seed arm, `output_identical` present as a
  bool; MIN free per card; no fatal in the log.
* **Explicitly not an error** `output_identical == False`. That IS the
  measurement.
* **Artifacts** `s05_boot_d/{reseed.json,reseed.txt,vram_summary.txt,server.log,…}`

### S6 — `s06_nccl_reference` · haiku · ~15 min · repeatable

The NCCL / system-RAM reference measurement in #279 format
(`new_nccl_reference_envelope`, schema_version 1).

* **Precondition** s00 PASS.
* **Command** `bash run_step.sh s06`
* **What is measured** per card pair, in pinned subprocesses: `all_reduce`
  over the ladder 64 KiB / 1 MiB / 8 MiB / 64 MiB and `send_recv` in **both**
  directions, each in two arms (`idle` and `host_stream_64mib`), **p50 AND p99
  in every row**. Both directions, because the #278 wrap-up failed on exactly
  this: the load axis had been collected asymmetrically, p50 against p99, and
  became unusable.
* **Success** `check_s06_nccl_reference.py`: envelope correct; every row has
  all ten mandatory fields (a row with nine is discarded by the loader, and a
  file made of such rows loads as empty); p99 ≥ p50; `transport` populated;
  both arms over the SAME (op, pair, size) keys; `send_recv` in both
  directions per pair; and `load_nccl_reference()` returns measured profiles
  without error.
* **Abort** an aborted pair = FAIL.
* **Artifacts** `s06_nccl_reference/{nccl_reference.json,nccl_debug.log}`

### S7 — `s07_offload_register_gpu` · haiku · ~12 min · repeatable

The offload register on real silicon: `CudaDeviceOps`, real item sizes,
fetch-back latencies per class (#286 remainder items 1 and 4).

* **Precondition** s00 PASS.
* **Command** `bash run_step.sh s07`
* **What runs** all three movement routes on the largest card (resolved at
  runtime): `tensor` (pinned pool + async H2D) for
  lane_workspaces/kv_shadow/experts, `tag` (#93 tag pools over the real
  memory-saver) for graph_rungs/gdn_state_sets, `suspend` (#89) for cold_lane.
  256 MiB per item, 5 park/wave_in cycles, p50/p99.
  The park is requested EXPLICITLY: the register parks on demand, and under
  `auto` `park()` refuses without saturation pressure — correct, but then the
  run measures the gate instead of the movement. The probe therefore sets
  every measured class to `ram` via the granular per-class knob
  (`--lane-offload-class-policy` syntax) and puts the resolved policy map into
  the artifact. The planning phase (`--dry-run`) runs the same sequence
  against `FakeDeviceOps` — without a card, as a self-test of the probe.
* **Success** `check_s07_offload_register_gpu.py`: `device_ops ==
  "CudaDeviceOps"` (a validation with FakeDeviceOps validates nothing); **all
  three routes** green; a real size per row, confirmed by
  `resolve_size_bytes`; the state sequence really contains `parked` (a
  silently no-op'ing park returns the same thing as a working one); fetch-back
  latency > 0 over ≥3 cycles; zero park/wave_in errors; the negative control
  refused (one item stays on `auto` without a sensor and must be rejected —
  otherwise the run does not prove that it was the explicit policy that
  allowed the park); `latency_term_ms` collected — this is exactly the number
  the #279 dispatcher reads.
* **Abort** memory-saver unavailable = STOP (two of three routes would be
  untested, and a green verdict on a third of the register would be worse than
  none). The memory-saver needs its preload hook in LD_PRELOAD, otherwise
  every `region()` dies; the probe sets it itself and restarts once (the
  linker reads LD_PRELOAD at process start).
* **Artifacts** `s07_offload_register_gpu/offload_register_gpu.json`

### S8 — `s08_dispatcher_tables` · haiku · ~3 min · repeatable · **CPU-only**

Load the measured rate tables into the #279 dispatcher and verify placeholder
neutrality.

* **Precondition** s01 AND s06 PASS. No card, no lock, no corridor.
* **Command** `bash run_step.sh s08` (optionally `GDR_TSV=<#278 matrix>`)
* **Success** `check_s08_dispatcher_tables.py`: all three sources present and
  loaded without error; >0 measured profiles AND >0 effective apertures (zero
  apertures means the capability matrix contributed nothing and every direct
  path is unbounded — the silent variant of the failure); the deliberately
  placeholder-contaminated class decides STATUS_QUO everywhere, including for
  `protected`; **the saturation sensor and the latency term were NOT consulted
  while doing so** (the probe hooks THROW on contact, so that "not consulted"
  is proven and not assumed); and the fully measured class decides at least
  one real path — otherwise the tables were loaded and ignored.
* **Why this step exists** `load_rate_tables` degrades missing sources loudly,
  but without error, to placeholders, and rule 1 then holds everything at the
  status quo. Correct at runtime — but indistinguishable here from a run with
  no cards at all. And: neutrality has so far been tested ONLY with
  placeholders. Only now are there measured candidates that could violate the
  rule.
* **Artifacts** `s08_dispatcher_tables/dispatcher_tables.json`

### S9 — `s09_sensor_smoke` · sonnet · ~15 min · repeatable

gdn/KV pressure ladders: flags on a real boot, sensor against real occupancy.
Small model (Qwen3.5-4B), one card, TP=1.

* **Precondition** s00 PASS.
* **Command** `bash run_step.sh s09`
* **Success** `check_s09_sensor_smoke.py`: all four ladder flags come back
  from `/get_server_info` (argument validation is CPU-tested — what is new is
  that the values reach the scheduler); two identical greedy generations are
  identical (the ladders are supposed to be inert); ≥20 occupancy samples from
  `sglang:token_usage` with a maximum > 0 (a flat zero line would mean the load
  never reached the pool); the sensor returns a reading with a verdict, a
  finite occupancy and a trend — and twice the same one from the same series.
* **Explicitly NOT tested** the wiring of the sensor to scheduler occupancy,
  and any real movement of state sets or KV. Neither exists yet; they are the
  open items these numbers prepare.
* **Artifacts** `s09_sensor_smoke/{sensor_smoke.json,server_info.json,server.log}`

---

## The BAR1 block (s10-s12) — runs on the PVE HOST

These three steps do not run where the battery runs. The patched driver, the
holder module and `/dev/dmabuf_holder` live on the PVE host; CT999 cannot even
open the holder device (major 10 is not in the container's device allowlist).
So the steps drive the host over ssh, while battery, state and artifacts stay
in the container.

What follows from that, and what is written in each of the three scripts:

* **Two `/tmp` namespaces.** `/tmp/gpu-card-N.lock` in CT999 and the
  identically named directory on the host are **different** locks. Both are
  therefore taken: the container side by `run_step.sh` (`locks="battery"`),
  the host side by the step itself via `battery_host.sh`. A foreign lock is
  broken on **neither** side.
* **Every ssh has a deadline.** A blocking ssh inside a bash call leaves the
  executor unable to act without anyone seeing it. `host_ssh` always carries
  `timeout`, `ConnectTimeout` and keepalives.
* **py-spy runs on the HOST**, before every kill, out of the container venv
  (the host has none of its own). Only the step's own process group is killed,
  never a pattern.
* **Server logs stay on the host** (`/root/battery-bar1/`). What lands in the
  run directory is the grep result and a bounded tail — never the whole log,
  never in an agent's context.
* **Path mapping.** The container's root filesystem is
  `/spinning/subvol-999-disk-0` on the host, writable in both directions. That
  is why the measurement driver on the host writes straight into the run
  directory — it is the only way s12's live table is live.

Environment variables (all with a default, all documented in
`battery_host.sh`): `BAR1_HOST`, `BAR1_HOST_KEY`, `BAR1_HOST_SUBVOL`,
`BAR1_HOST_WT`, `BAR1_NV_SOURCE`, `BAR1_HOLDER_KO`, `BAR1_EXTCACHE`,
`BAR1_PORT`, `BAR1_VIEWER_KILL_OK`, `S12_SESSIONS`, `S12_POINT_SECONDS`,
`S12_BASELINE_TOL_PCT`.

### S10 — `s10_bar1_driver` · sonnet · ~6 min · repeatable

Patched driver plus the direct path's preconditions. Recipe from
`04_BETRIEB.md` of the P2P handover, with one deliberate difference: the PCI
addresses for the reset are resolved at runtime from `lspci` instead of being
written in — enumeration shifts between boots and driver states.

* **Precondition** s00 PASS.
* **Command** `bash run_step.sh s10`
* **Idempotent** If the patched module is already in place **with** regkey and
  holder, nothing is unloaded: the step checks and records
  `reload_performed=false`. Swapping the driver just to show that it can be
  swapped would cost exactly the state s11 and s12 need.
* **Success** `check_s10_bar1_driver.py`: regkey in
  `/proc/driver/nvidia/params`; `strings nvidia.ko | grep -c SMALLBAR_P2P` = 37
  (the full patch, not the 1 of the minimal one); **the srcversion of the
  LOADED module matches that of the `.ko`** — the regkey line on its own proves
  a parameter, not the module's identity; `dmabuf_holder` loaded **and**
  `/dev/dmabuf_holder` present; `nvidia_uvm` loaded; all three cards enumerate
  with UUID and PCI address.
* **Abort** host unreachable, compute processes on the cards, or viewers
  holding the modules = STOP (environment). Missing regkey, minimal patch,
  foreign module, missing holder, lost card = FAIL.
* **Viewers** `nvtop` and its relatives hold the modules. They are terminated
  **only** with `BAR1_VIEWER_KILL_OK=1`, and that clearance is **not
  permanent** (05_FALLEN). Without it the step halts and names the PIDs.
* **Restore** The battery ends by default **without** a restore: s11 and s12
  run under the patch, and NCCL coexistence has been measured (37.11 / 64.65 /
  359.35 µs against 37.89 / 65.25 / 357.63 µs — all within the spread).
  Whoever wants the stock driver back: `bash s10_restore.sh` — its own
  command, not a battery step.
* **Artifacts** `s10_bar1_driver/{driver_state.json,remote_probe.sh,remote_reload.sh,host/*}`

### S11 — `s11_bar1_e2e` · sonnet · ~25 min · repeatable

One standard run over the direct path, end to end. The question is **not**
whether it boots.

* **Precondition** s10 PASS. The BAR1 integration must be present in the
  working tree under test (`barlink_bar1.py`, `benchmark/bar1_graph_check.py`),
  otherwise STOP with a pointer to `BAR1_HOST_WT`.
* **Command** `bash run_step.sh s11`
* **Gate first** `benchmark/bar1_graph_check.py 0,1,2`. `GRAPH_FREIGABE=1`
  without that proof yields numbers from an operating point nobody can defend.
* **Success** `check_s11_bar1_e2e.py`: all gate cases of the gate passed;
  `ACHIEVED=bar1` **per group** — with `SGLANG_UNEVEN_DCP=1` there are two
  (`tp:0`, `dcp:0`), and one of them on gloo makes the run a mixed one (which
  happened exactly once and cost a whole measurement); per group one
  `barlink-BAR1: setup in` line; smoke answer coherent (the numbers 1..20 in
  sequence, **counted**, not judged) and `spec_accept_length` a number; no
  OOM/NCCL/watchdog in the grepped log.
* **The bolt as a special case** `barlink._select` aborts **loudly** instead of
  silently falling back to the host-staged gloo layer under a graph capture.
  As long as `all_gather` is missing from `barlink_bar1`, the standard run ends
  with
  `RuntimeError: barlink: 'all_gather' with <n> bytes during a CUDA graph
  capture ...`. The check reports that as its **own FAIL message** carrying
  `RIEGEL`, the operation and the size — correct behaviour of the code and a
  FAIL of this step at the same time. This is exactly the scenario the
  parallel BAR1 integration signs off.
* **Not judged** the HEIGHT of `spec_accept_length`, the setup duration, any
  throughput.
* **Artifacts** `s11_bar1_e2e/{bar1_e2e.json,graph_check.txt,barlink_lines.txt,smoke.json,server_info.json,server.log,remote_*.sh}`

### S12 — `s12_prefill_kurve` · sonnet · ~70 min · NOT repeatable

**The** measurement. Over the host path the prefill curve is flat
(1190/1097/1144/1105/1122 tok/s over 1/2/4/8/16 sessions) at 65-90% collective
time — more sessions buy nothing, because it is the collectives that saturate,
not the compute units. If the detour is the ceiling, the ceiling falls with
it.

* **Precondition** s11 PASS.
* **Command** `bash run_step.sh s12`
* **Interleaved, not blockwise** Per session count, bar1 and baseline run back
  to back (A,B,A,B), then the next session count. Eight boots for four points
  per arm — that is the price of not comparing two different afternoons
  (measurement rule 5). Blockwise would be one boot per arm, and worthless.
* **The arms differ in exactly three variables** (`SGLANG_BARLINK`,
  `SGLANG_BARLINK_TRANSPORT`, `SGLANG_BARLINK_GRAPH_ENABLE`, plus the driver
  source). Both boot scripts come from **one** template so they cannot drift
  apart; a test diffs them and allows exactly two lines of difference.
* **What is measured** per arm and session count 1/4/8/16 one prefill point
  (time box 15 s, warm-up 8 s, the counter is `prompt_tokens` **from the
  response**, a unique head per request plus `/flush_cache` so that the radix
  cache is not what gets measured), plus per boot one decode point at bs=1 and
  bs=16 with `ms/Token`, `ms/Verify` and `spec_accept_length`. Raw per-request
  rows are persisted, not just the final number.
* **Success** `check_s12_prefill_kurve.py`: one number in **both** arms per
  planned session count; the arms really did alternate (otherwise FAIL
  "blockwise"); **the transport proof per point** — bar1 points with all groups
  on `ACHIEVED=bar1`, baseline points without a single barlink group; one decode
  point per arm at bs=1 and bs=16; one persisted output sample per arm (a fast
  garbage run looks good in a throughput table); and the **baseline
  reproduces** the known numbers within `±5%`.
* **Abort** If the baseline does not reproduce, that is **STOP**: this
  environment is not the one the known numbers came from. Caveat: the known
  values come from a **different** measurement program. If the gate breaks, the
  first suspect is the measurement program, not the rig — hence STOP (the
  operator decides) and not FAIL, and hence the tolerance is adjustable via
  `S12_BASELINE_TOL_PCT`. The A/B comparison itself does not depend on it:
  both arms are measured with the same program.
* **Explicitly NOT judged** whether the curve stays flat or rises, and by how
  much bar1 beats the baseline. Flat is a finding, rising is a finding. A check
  that judged it would decide the very question the measurement exists for. The
  result JSON carries both curves side by side.
* **Read along live** After each session-count pair, `zwischentabelle.md` is
  rewritten from the persisted points. The executor prints it
  (EXECUTOR_PROTOCOL 6b/6c).
* **Not covered** compute versus wait time per rank. That needs the profiler
  and is an exercise of its own.
* **Artifacts** `s12_prefill_kurve/{prefill_kurve.json,zwischentabelle.md,punkte.jsonl,roh_*.jsonl,belege/*,logs/*,remote_*.sh}`

---

## Time budget

| Step | Model | expected | hard budget |
|---|---|---:|---:|
| s00_preflight | haiku | 3 min | 5 min |
| s01_p2p_reprobe | haiku | 10 min | 30 min |
| s02_boot_a | sonnet | 35 min | 60 min |
| s03_boot_b | sonnet | 40 min | 65 min |
| s04_boot_c | sonnet | 45 min | 70 min |
| s05_boot_d | sonnet | 30 min | 50 min |
| s06_nccl_reference | haiku | 15 min | 30 min |
| s07_offload_register_gpu | haiku | 12 min | 20 min |
| s08_dispatcher_tables | haiku | 3 min | 10 min |
| s09_sensor_smoke | sonnet | 15 min | 30 min |
| s10_bar1_driver | sonnet | 6 min | 15 min |
| s11_bar1_e2e | sonnet | 25 min | 45 min |
| s12_prefill_kurve | sonnet | 70 min | 150 min |
| **Total** | | **5 h 09 min** | |

The hard budgets sit well above the expectations so that a slow load is not
mistaken for a hang. An overrun is STOP, never a reason to wait longer — with
a py-spy dump of every registered process before the kill.

## What the battery deliberately does NOT do

* **It does not judge measured values.** No check has an opinion on an accept
  height, a bandwidth, or whether P2P engages. What is checked is that the
  measurement took place, is complete, and can be loaded by its real consumer.
  Interpretation is the reader's work — which is why it is not buried in a
  script that passes it silently.
* **It does not fill in `verdict_diff.md`.** The eight "no P2P" legacy
  verdicts need a judgement per line; an overturned verdict gets a task of its
  own before any placement code is touched. That file records the delta, it
  does not authorize rebuilds.
* **It tunes nothing inside the window.** No ratio, no `RESERVE_HOST`, no
  context. A boot that rejects the shape is a spent boot with an answer — not
  a starting point for a parameter search.
* **It does not test wiring that does not exist.** Sensor to scheduler
  (addendum 9), admission hook on the scheduler (addendum 8), real movement of
  the GDN state sets: all open items. "Testing" them would mean handing out a
  green verdict for code that is not there.
* **It does not run a perf regression.** ms/round against a noise floor is its
  own, longer exercise with interleaved measurement and a fixed clock. A
  battery with 20-second measurement points cannot deliver that and should not
  pretend to. **s12 is the one exception** and pays the price for it: eight
  boots, interleaved arms, a reproduction gate against the known baseline —
  and still no verdict on the shape of the curve.
* **It does not restore the driver.** After s10 the patched driver stays
  loaded; that is intentional and backed by evidence (NCCL coexistence
  measured). `s10_restore.sh` is an operator command, not the end of a step.
* **It breaks no foreign locks** and kills nothing it did not start itself.
