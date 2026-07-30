#!/usr/bin/env bash
# Shared setup for the #274 round 7c boot queue.
#
# Sourced by every r7c_boot_*.sh. Holds only what is IDENTICAL across the
# queue: environment, paths, the card-order resolution, the VRAM sampler and
# the arbitration bookkeeping. Anything a single boot varies stays in that
# boot's own file, so a diff between two recipes shows the experiment and
# nothing else.
#
# NOT executable on its own.

set -uo pipefail

source /root/rig-env.sh 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:-/spinning/htsglang}"
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"
WT="${WT:-/spinning/wt-lane-r7c}"

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

ARB=/spinning/gpu-arb
OWNER=agent-lane-r7c

# --- dry run ----------------------------------------------------------------
# R7C_DRY_RUN=1 walks a recipe from the top to the launch line without touching
# a GPU, the arbitration files or the server, and prints the assembled launch
# command instead of running it. It exists because the only failure this queue
# has produced so far was an unbound variable at the line that builds those
# flags -- a class of bug that needs no hardware to find, and now has a test
# that finds it (test/registered/unit/test_r7c_recipe_dry_run.py).
R7C_DRY_RUN="${R7C_DRY_RUN:-0}"

dry_run() { [ "$R7C_DRY_RUN" = "1" ]; }

# Missing inputs abort a real boot. In a dry run they are a warning, so the
# flag assembly stays checkable on a machine that has neither cards nor models.
require_file() {  # $1 = path, $2 = abort message
  [ -f "$1" ] && return 0
  if dry_run; then
    echo "DRY RUN: fehlt, ignoriert: $1" >&2
    return 0
  fi
  echo "ABBRUCH: $2" >&2
  return 1
}

# --- card order -------------------------------------------------------------
# CUDA order (every flag) and NVML order (nvidia-smi) differ on this rig, and
# the runbook says explicitly not to hardcode either: the mapping can shift
# with a driver or boot change. Resolved here, once, and printed into the log
# so every recipe's --speculative-draft-gpu / --rank-gpu-id can be read back
# against the card it actually meant.
resolve_cards() {
  # A dry run has no cards to ask. R7C_CARDS_FILE feeds a recorded resolution
  # in instead, so the parsing below and everything downstream of it runs on
  # the same text a real rig produces.
  if [ -n "${R7C_CARDS_FILE:-}" ]; then
    cat "$R7C_CARDS_FILE"
    return
  fi
  "$VENV/bin/python" - <<'PY'
import subprocess
import torch

names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total",
     "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip().splitlines()
print("CUDA order (what the flags mean):")
for i, n in enumerate(names):
    print(f"  cuda:{i}  {n}")
print("NVML order (what nvidia-smi prints):")
for line in smi:
    print(f"  {line.strip()}")
big = max(range(len(names)), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
small = [i for i in range(len(names)) if i != big]
print(f"CUDA_BIG={big}")
print(f"CUDA_SMALL={','.join(str(i) for i in small)}")
PY
}

# Sets CUDA_BIG (the 5090) and CUDA_SMALL (comma list of the 3080s) in the
# CALLING shell, and echoes the resolution -- into $1 as well if given.
#
# The copy into the log file has to happen in HERE. The call sites used to
# write `load_card_order | tee "$OUT/cards.txt"`, which runs the function in a
# subshell of the pipeline: the echoed lines still reached the log, so the
# resolution looked successful, while the two assignments died with that
# subshell and the first use of CUDA_SMALL hit `set -u` as an unbound variable.
# Taking the file as an argument keeps the assignments in the caller's shell,
# which is the only place they are of any use.
load_card_order() {  # $1 = optional file to copy the resolution into
  local out
  if ! out="$(resolve_cards)"; then
    echo "ABBRUCH: Kartenreihenfolge nicht aufloesbar" >&2
    return 1
  fi
  if [ -n "${1:-}" ]; then
    printf '%s\n' "$out" | tee "$1"
  else
    printf '%s\n' "$out"
  fi
  CUDA_BIG="$(printf '%s\n' "$out" | sed -n 's/^CUDA_BIG=//p')"
  CUDA_SMALL="$(printf '%s\n' "$out" | sed -n 's/^CUDA_SMALL=//p')"
  # `set -u` catches an UNSET variable, never an empty one. A resolver that
  # printed nothing usable would otherwise pass silently and put an empty
  # string into --speculative-draft-gpu, so the emptiness is checked here.
  if [ -z "$CUDA_BIG" ] || [ -z "$CUDA_SMALL" ]; then
    echo "ABBRUCH: CUDA_BIG/CUDA_SMALL fehlen in der Kartenaufloesung" >&2
    return 1
  fi
  export CUDA_BIG CUDA_SMALL
}

# --- safety net -------------------------------------------------------------
# The hardware is always right and the arbitration files can go stale, so this
# runs before every boot regardless of what free-until says.
assert_cards_free() {
  local busy
  dry_run && { echo "DRY RUN: Kartenpruefung uebersprungen"; return 0; }
  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
          | awk -F'[, ]+' '$2 > 500 {print $1}')"
  if [ -n "$busy" ]; then
    echo "ABBRUCH: Karten belegt (>500 MiB): $busy" >&2
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2
    return 1
  fi
  return 0
}

claim_cards() {  # $1 = purpose
  dry_run && { echo "DRY RUN: keine Arbitrierung, purpose=$1"; return 0; }
  printf 'session=operator  cards=0,1,2  purpose=%s  since=%s\n' \
    "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ARB/holder"
  printf '%s %s BELEGT cards=0,1,2 purpose=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OWNER" "$1" >> "$ARB/log"
}

release_cards() {  # $1 = result line
  dry_run && { echo "DRY RUN: keine Freigabe, result=$1"; return 0; }
  rm -f "$ARB/holder"
  printf '%s %s FREI cards=0,1,2 %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OWNER" "$1" >> "$ARB/log"
}

# --- VRAM sampler (queue item 4) --------------------------------------------
# The missing input for every multi-card placement decision is "how many MiB
# are actually free on each 3080 while this vehicle is serving". It costs
# nothing to answer during a boot that is happening anyway, so every recipe
# samples it rather than leaving it for a boot of its own.
#
# Sampled DURING load, not just at the end: the peak is what decides whether a
# drafter fits, and the steady state after warmup hides the prefill scratch
# that the 2700 MiB reserve exists for.
start_vram_sampler() {  # $1 = output csv
  local out="$1"
  echo "ts_utc,gpu_index,name,mem_used_mib,mem_total_mib,util_pct" > "$out"
  if dry_run; then
    echo "DRY RUN: kein VRAM-Sampler"
    VRAM_SAMPLER_PID=""
    export VRAM_SAMPLER_PID
    return 0
  fi
  (
    while true; do
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
                 --format=csv,noheader,nounits \
        | sed "s/^/$(date -u +%Y-%m-%dT%H:%M:%SZ),/" >> "$out"
      sleep 5
    done
  ) &
  VRAM_SAMPLER_PID=$!
  export VRAM_SAMPLER_PID
}

stop_vram_sampler() {
  [ -n "${VRAM_SAMPLER_PID:-}" ] && kill "$VRAM_SAMPLER_PID" 2>/dev/null
  VRAM_SAMPLER_PID=""
}

# Free MiB per card at the point of lowest headroom over the whole run --
# the number a placement decision needs.
summarise_vram() {  # $1 = csv
  "$VENV/bin/python" - "$1" <<'PY'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
peak = defaultdict(int)
total = {}
name = {}
for r in rows:
    i = int(r["gpu_index"])
    peak[i] = max(peak[i], int(r["mem_used_mib"]))
    total[i] = int(r["mem_total_mib"])
    name[i] = r["name"]
print(f"{'NVML':>4} {'Karte':28} {'Peak benutzt':>13} {'Total':>8} {'MIN frei':>9}")
for i in sorted(peak):
    print(f"{i:>4} {name[i]:28} {peak[i]:>13} {total[i]:>8} {total[i]-peak[i]:>9}")
print(f"\n({len(rows)} Samples)")
PY
}

# Starts the launcher in the background, or -- in a dry run -- prints the fully
# assembled command and starts nothing. Every recipe routes its launch through
# here so the dry run covers the exact flags a real boot would pass, including
# everything the recipe derived from CUDA_BIG / CUDA_SMALL.
launch_server() {  # $1 = server log, $2 = pid file, rest = command + flags
  local log="$1" pidfile="$2"
  shift 2
  if dry_run; then
    printf 'DRY RUN LAUNCH: %s\n' "$*"
    return 0
  fi
  setsid "$@" > "$log" 2>&1 &
  echo $! > "$pidfile"
}

wait_for_server() {  # $1 = port, $2 = timeout_s
  local port="$1" budget="${2:-1800}" t0
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "$budget" ]; do
    if curl -sf -m 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "server up nach $(( $(date +%s) - t0 ))s"
      return 0
    fi
    sleep 10
  done
  echo "ABBRUCH: server nicht oben in ${budget}s" >&2
  return 1
}
