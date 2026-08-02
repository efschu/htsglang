# Task #428 boot checks — the GPU half of the #421 F1/F2 fixes

The two HIGH-severity #421 findings were fixed on the desk, with hermetic
falsifiers that prove the wiring exists. Neither fix is DONE until it has
booted: F1's `auto` table is built from NVML and the launcher's rank→card
vector, and F2's register is configured inside `ModelRunner.__init__` — two
places a CPU test can only stub. Both fixes are therefore reported
**BOOT-PENDING** until the scripts here pass in a GPU window.

| script | proves |
|---|---|
| `f1_kv_pressure_ladder_auto.sh` | `--kv-pressure-ladder auto` boots on the standard 27B recipe, the table is computed from the rig profile, KV pressure drives a rung flip, and serving stays coherent across it |
| `f2_lane_offload_register.sh` | the three `--lane-offload-*` flags reach the process-global offload register at runner init, and a planted typo refuses the boot instead of degrading to the default preset |

Both scripts are self-contained, take the rig facts from `/root/rig-env.sh`
the way `docs/rig-runbook.md` §4 does, write their log **outside** the repo,
and print a single `PASS`/`FAIL` verdict line per check. Neither one reads a
server log into anybody's context; they grep and report.

## Before running

* One GPU window, exclusive. Publish it in `/spinning/gpu-arb/` first.
* `MODEL_ROOT`, `VENV`, `REPO_ROOT` set (or `/root/rig-env.sh` present).
* `PORT` free; the scripts default to 30428 and fail fast if it is taken.
* Kill only your own PIDs on the way out. The scripts write their PID to
  `/tmp/428_<check>.pid` and use it; they never `pkill sglang`.

## Reserve

`5500,3800,3800` — the reserve validated against real long prompts on the
reference rig (runbook §4.9; RIG EXAMPLE, see there — a different
rig/model/context needs its own probe). F1 deliberately drives the KV pool
towards full, so the run must not be sitting on a reserve that only survives
a warmup.

## What a failure means

* **F1 boots but no flip fires** — the ladder was built but the rungs it
  inventoried have no reachable actuator on this configuration. Check the
  `wired reliefs` list in the `[kv-pressure]` INFO line the fix emits: it is
  computed from the flags, so an empty list means the recipe passed none of
  `--max-running-requests-ceiling`, `--enable-kv-session-offload`,
  `--kv-reshard-vectors`. That is a recipe bug, not a code bug.
* **F1 refuses at boot with "cannot map ranks to cards"** — the rank→card
  UUID vector was not published. Expected only if `--rank-gpu-id` is absent
  on a mixed rig; the standard recipe passes it, so this is a real finding.
* **F2 shows the latency profile under `--lane-offload-profile capacity`** —
  the configure call was reached too late (after the first adapter read) or
  not at all. That is exactly the #421 F2 defect returning.
