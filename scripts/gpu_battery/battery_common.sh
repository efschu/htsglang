#!/usr/bin/env bash
# Shared plumbing for the GPU test battery (scripts/gpu_battery/).
#
# Holds ONLY what every step needs identically: the results layout, the
# cross-session card arbitration, the VRAM corridor gate and the NVML/PCI card
# inventory. Anything a single step varies stays in that step's own file.
#
# NOT executable on its own. Sourced by the step scripts.
#
# Two arbitration mechanisms exist on this rig and they are NOT the same:
#   * /tmp/gpu-card-N.lock  -- DIRECTORIES, mkdir is the atomic acquire, an
#     info file inside carries holder + heartbeat. Cross-session. This is the
#     one the battery takes.
#   * /spinning/gpu-arb/holder -- the operator-level holder file the r7c boot
#     recipes write themselves. The battery never touches it.
# A held lock is NEVER stolen, stale or not. Reporting and stopping is the
# only correct answer; breaking someone else's lock needs the operator.

set -uo pipefail

BATTERY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BATTERY_DIR

# The worktree the battery runs against. Every step exports this into the
# recipes so they never fall back to their own default checkout.
WT="${WT:-$(cd "$BATTERY_DIR/../.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$WT}"
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"
PY="${PY:-$VENV/bin/python}"
export WT REPO_ROOT VENV MODEL_ROOT PY

BATTERY_RESULTS_ROOT="${BATTERY_RESULTS_ROOT:-/spinning/gpu-battery-results}"
export BATTERY_RESULTS_ROOT

# Absolute floor, per the VRAM corridor rule. Not a tunable knob: below this
# no step may start, and a step that wants to argue about it is a step that
# must stop instead.
BATTERY_MIN_FREE_MIB="${BATTERY_MIN_FREE_MIB:-400}"

# --- results layout ---------------------------------------------------------
# One run directory, one sub-directory per step, everything a check needs and
# everything the handoff quotes lives inside it. Nothing is written into the
# repository.
battery_run_dir() {
    if [ -n "${BATTERY_RUN:-}" ]; then
        printf '%s\n' "$BATTERY_RUN"
        return 0
    fi
    echo "STOP: BATTERY_RUN is not set (see BATTERY.md, step S0)" >&2
    return 1
}

battery_step_dir() {  # $1 = step id, e.g. s02_boot_a
    local run
    run="$(battery_run_dir)" || return 1
    mkdir -p "$run/$1"
    printf '%s\n' "$run/$1"
}

# --- card inventory ---------------------------------------------------------
# Cards are identified by PCI bus id and UUID, never by a bare index: torch's
# CUDA order and NVML's order differ on this rig and the mapping shifts with
# driver and boot state. Every step that names a card resolves it here first.
battery_card_inventory() {  # $1 = output json path
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" - "$1" <<'PY'
import json
import subprocess
import sys

out = {"cards": [], "errors": []}

fields = "index,name,uuid,pci.bus_id,memory.total,memory.used"
try:
    smi = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60,
    )
    lines = [line for line in smi.stdout.strip().splitlines() if line.strip()]
except Exception as exc:  # nvidia-smi missing or wedged is a STOP, not a crash
    lines = []
    out["errors"].append(f"nvidia-smi: {exc!r}")

for line in lines:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        out["errors"].append(f"unparsable nvidia-smi row: {line!r}")
        continue
    idx, name, uuid, pci, total, used = parts
    total_mib, used_mib = int(total), int(used)
    out["cards"].append({
        "nvml_index": int(idx),
        "name": name,
        "uuid": uuid,
        "pci_bus_id": pci.lower(),
        "cuda_index": None,
        "vram_total_mib": total_mib,
        "vram_used_mib": used_mib,
        "vram_free_mib": total_mib - used_mib,
    })

# The PCI join is the whole point: it is what makes a CUDA index quotable.
try:
    import torch

    if torch.cuda.is_available():
        by_pci = {c["pci_bus_id"]: c for c in out["cards"]}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            pci = "%08x:%02x:%02x.0" % (
                getattr(props, "pci_domain_id", 0),
                getattr(props, "pci_bus_id", 0),
                getattr(props, "pci_device_id", 0),
            )
            card = by_pci.get(pci.lower())
            if card is None:
                out["errors"].append(
                    f"cuda:{i} pci {pci} has no NVML row -- device-order join failed"
                )
            else:
                card["cuda_index"] = i
        out["torch"] = torch.__version__
        try:
            out["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
        except Exception as exc:
            out["errors"].append(f"nccl version: {exc!r}")
    else:
        out["errors"].append("torch.cuda.is_available() is False")
except Exception as exc:
    out["errors"].append(f"torch: {exc!r}")

try:
    drv = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=60,
    )
    out["driver"] = drv.stdout.strip().splitlines()[0].strip() if drv.stdout.strip() else None
except Exception as exc:
    out["errors"].append(f"driver_version: {exc!r}")

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")

for c in out["cards"]:
    print(
        f"  nvml:{c['nvml_index']} cuda:{c['cuda_index']} {c['pci_bus_id']} "
        f"{c['name']}  frei {c['vram_free_mib']} / {c['vram_total_mib']} MiB"
    )
for e in out["errors"]:
    print(f"  ERROR {e}")
PY
}

# --- VRAM corridor ----------------------------------------------------------
# Never test on red. The hardware is always right and the arbitration files can
# go stale, so this runs before every step that touches a card, regardless of
# what any lock or holder file says.
battery_assert_corridor() {
    local bad=0 seen=0 line idx free
    while IFS=, read -r idx total used; do
        idx="${idx// /}"; total="${total// /}"; used="${used// /}"
        [ -z "$idx" ] && continue
        seen=$((seen + 1))
        free=$((total - used))
        if [ "$free" -lt "$BATTERY_MIN_FREE_MIB" ]; then
            echo "KORRIDOR: Karte $idx nur $free MiB frei (< $BATTERY_MIN_FREE_MIB)" >&2
            bad=1
        fi
    done < <(nvidia-smi --query-gpu=index,memory.total,memory.used \
             --format=csv,noheader,nounits 2>/dev/null)
    # No rows is NOT an empty corridor, it is a blind one. A card step that
    # starts because nvidia-smi said nothing is the worst of both: it runs, and
    # nobody knows what it ran on. This case is reachable -- s10 reloads the
    # host driver, and a container whose device nodes went stale afterwards
    # reports exactly nothing here.
    if [ "$seen" -eq 0 ]; then
        echo "KORRIDOR: nvidia-smi liefert keine Karten-Zeile -- keine Sicht auf die Karten" >&2
        bad=1
    fi
    if [ "$bad" != 0 ]; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total \
            --format=csv,noheader >&2
        return 1
    fi
    return 0
}

# --- locks ------------------------------------------------------------------
BATTERY_HELD_LOCKS=()
BATTERY_HEARTBEAT_PID=""

battery_release_locks() {
    if [ -n "$BATTERY_HEARTBEAT_PID" ]; then
        kill "$BATTERY_HEARTBEAT_PID" 2>/dev/null
        BATTERY_HEARTBEAT_PID=""
    fi
    local d
    for d in ${BATTERY_HELD_LOCKS[@]+"${BATTERY_HELD_LOCKS[@]}"}; do
        rm -rf "$d"
    done
    BATTERY_HELD_LOCKS=()
}

battery_acquire_locks() {  # $1 = step id (goes into the info file)
    local step="$1" n i lock
    n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')" || n=0
    if [ "${n:-0}" -lt 1 ]; then
        echo "STOP: keine GPU sichtbar (nvidia-smi -L leer)" >&2
        return 2
    fi
    for i in $(seq 0 $((n - 1))); do
        lock="/tmp/gpu-card-$i.lock"
        if mkdir "$lock" 2>/dev/null; then
            {
                echo "holder=gpu_battery"
                echo "step=$step"
                echo "pid=$$"
                echo "acquired=$(date -Is)"
                echo "heartbeat=$(date -Is)"
            } > "$lock/info"
            BATTERY_HELD_LOCKS+=("$lock")
        else
            echo "STOP: $lock ist belegt:" >&2
            sed 's/^/    /' "$lock/info" 2>/dev/null >&2
            echo "fremde Locks werden nie gebrochen -- Operator fragen." >&2
            battery_release_locks
            return 2
        fi
    done
    (
        while true; do
            for d in ${BATTERY_HELD_LOCKS[@]+"${BATTERY_HELD_LOCKS[@]}"}; do
                sed -i "s/^heartbeat=.*/heartbeat=$(date -Is)/" "$d/info" 2>/dev/null
            done
            sleep 30
        done
    ) &
    BATTERY_HEARTBEAT_PID=$!
    return 0
}

# Reports whether ANY card lock is currently held, without taking anything.
# Used by the steps whose own tool takes the locks (p2p run_all.sh).
battery_locks_are_free() {
    local n i lock
    n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')" || n=0
    for i in $(seq 0 $((${n:-0} - 1))); do
        lock="/tmp/gpu-card-$i.lock"
        if [ -d "$lock" ]; then
            echo "STOP: $lock ist belegt:" >&2
            sed 's/^/    /' "$lock/info" 2>/dev/null >&2
            return 2
        fi
    done
    return 0
}

# --- process hygiene --------------------------------------------------------
# py-spy BEFORE any kill, always, and only ever our own pid. A dump costs
# seconds; a killed hang that nobody dumped costs the whole run.
battery_dump_and_kill() {  # $1 = pid, $2 = dump path
    local pid="$1" dump="$2"
    [ -z "$pid" ] && return 0
    kill -0 "$pid" 2>/dev/null || return 0
    if command -v py-spy >/dev/null 2>&1; then
        timeout 60 py-spy dump --pid "$pid" > "$dump" 2>&1 || true
    elif [ -x "$VENV/bin/py-spy" ]; then
        timeout 60 "$VENV/bin/py-spy" dump --pid "$pid" > "$dump" 2>&1 || true
    else
        echo "py-spy nicht gefunden -- kein Dump moeglich" > "$dump"
    fi
    kill "$pid" 2>/dev/null
    sleep 5
    kill -9 "$pid" 2>/dev/null
    return 0
}

# The boot recipes write their server pid to a file of their own. run_step.sh
# can only py-spy what it knows about, so a step that spawns a long-lived
# process registers it here. Harvesting instead of parsing the recipe keeps the
# recipes untouched.
battery_harvest_pidfile() {  # $1 = pidfile the step will write, $2 = pids file
    local pidfile="$1" pids="$2"
    rm -f "$pidfile"
    (
        local i
        for i in $(seq 1 720); do
            if [ -s "$pidfile" ]; then
                cat "$pidfile" >> "$pids"
                break
            fi
            sleep 5
        done
    ) &
    BATTERY_HARVEST_PID=$!
}

battery_stop_harvest() {
    [ -n "${BATTERY_HARVEST_PID:-}" ] && kill "$BATTERY_HARVEST_PID" 2>/dev/null
    BATTERY_HARVEST_PID=""
}

# --- server helper ----------------------------------------------------------
# Never an unbounded wait inside a single call: every curl carries -m and the
# loop carries a budget. An agent that blocks forever in one bash call is a
# wedged agent, not a patient one.
battery_wait_for_server() {  # $1 = port, $2 = budget_s
    local port="$1" budget="${2:-900}" t0
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$budget" ]; do
        if curl -sf -m 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "server up nach $(( $(date +%s) - t0 ))s"
            return 0
        fi
        sleep 10
    done
    echo "server nicht oben in ${budget}s" >&2
    return 1
}
