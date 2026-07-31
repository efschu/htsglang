#!/usr/bin/env bash
# S0 -- preflight. The only step that is allowed to discover that the rig is
# not ready; every later step assumes what this one established.
#
# Touches no card beyond querying it: no allocation, no boot, no process that
# outlives the script.
#
# Answers, in one artifact:
#   * which physical card is which (PCI + UUID + the NVML<->CUDA join),
#   * is the VRAM corridor green on every card,
#   * are all card locks free,
#   * do all recipes, probes, models and the venv exist,
#   * what driver / torch / NCCL is this, so the handoff can name it.
#
# Usage:  BATTERY_RUN=<run dir> bash s00_preflight.sh

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh

STEP=s00_preflight
DIR="$(battery_step_dir "$STEP")" || exit 2

echo "== card inventory (PCI/UUID, NVML<->CUDA join) =="
battery_card_inventory "$DIR/inventory.json"

"$PY" - "$DIR" "$WT" "$VENV" "$MODEL_ROOT" "$BATTERY_MIN_FREE_MIB" <<'PY'
import datetime
import json
import os
import shutil
import sys

step_dir, wt, venv, model_root, min_free = sys.argv[1:6]

with open(os.path.join(step_dir, "inventory.json")) as f:
    inv = json.load(f)

required = [
    f"{wt}/scripts/dual_group/r7c/common.sh",
    f"{wt}/scripts/dual_group/r7c/boot_a_fp8_reference.sh",
    f"{wt}/scripts/dual_group/r7c/boot_b_dense_head.sh",
    f"{wt}/scripts/dual_group/r7c/boot_c_dflash_solo_q8.sh",
    f"{wt}/scripts/dual_group/r7c/boot_d_lane_reseed.sh",
    f"{wt}/scripts/dual_group/lane_accept_probe.py",
    f"{wt}/scripts/p2p_readiness/run_all.sh",
    f"{wt}/scripts/p2p_readiness/capability_matrix.py",
    f"{wt}/scripts/p2p_readiness/d2d_bench.py",
    f"{wt}/scripts/p2p_readiness/nccl_transport_check.py",
    f"{wt}/python/sglang/srt/distributed/device_communicators/barlink_path_rates.py",
    f"{venv}/bin/python",
    f"{model_root}/Qwen3.6-27B-FP8",
    f"{model_root}/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP",
    f"{model_root}/Qwen3.6-27B-MTP-Q3_K_M-GGUF/Qwen3.6-27B-Q3_K_M.gguf",
    f"{model_root}/qwen3.6-27b-dflash-gguf",
    f"{model_root}/Qwen3.5-4B",
]

locks_held = []
for i in range(len(inv.get("cards", [])) or 8):
    lock = f"/tmp/gpu-card-{i}.lock"
    if os.path.isdir(lock):
        info = ""
        try:
            with open(os.path.join(lock, "info")) as f:
                info = " ".join(f.read().split())
        except OSError:
            info = "<no info file>"
        locks_held.append({"lock": lock, "info": info})

payload = {
    "kind": "gpu_battery_preflight",
    "schema_version": 1,
    "timestamp": datetime.datetime.now().isoformat(),
    "min_free_mib": int(min_free),
    "cards": inv.get("cards", []),
    "inventory_errors": inv.get("errors", []),
    "driver": inv.get("driver"),
    "torch": inv.get("torch"),
    "nccl": inv.get("nccl"),
    "locks_held": locks_held,
    "required_files": {p: os.path.exists(p) for p in required},
    "tools": {
        "nvidia-smi": bool(shutil.which("nvidia-smi")),
        "py-spy": bool(shutil.which("py-spy") or os.path.exists(f"{venv}/bin/py-spy")),
        "curl": bool(shutil.which("curl")),
    },
    "arb_holder": None,
}

holder = "/spinning/gpu-arb/holder"
if os.path.exists(holder):
    try:
        with open(holder) as f:
            payload["arb_holder"] = " ".join(f.read().split())
    except OSError:
        payload["arb_holder"] = "<unreadable>"

with open(os.path.join(step_dir, "preflight.json"), "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")

missing = [p for p, ok in payload["required_files"].items() if not ok]
print(f"required files: {len(required) - len(missing)}/{len(required)} present")
for p in missing:
    print(f"  MISSING {p}")
print(f"card locks held: {len(locks_held)}")
print(f"arb holder: {payload['arb_holder']}")
PY

echo "== VRAM corridor =="
if battery_assert_corridor; then
    echo "corridor green (>= $BATTERY_MIN_FREE_MIB MiB free per card)"
else
    echo "corridor RED -- the check will report STOP"
fi

echo "done: $DIR"
