#!/usr/bin/env bash
# #631 phase-flip test family -- THE canonical list (operator request
# 2026-08-08: two independent under-collections happened from ad-hoc
# globs; the family spans three directories, one glob does NOT cover it).
# CPU-only desk run: correct PYTHONPATH (worktree, not the serving tree)
# and no GPU (CUDA_VISIBLE_DEVICES=99).
set -euo pipefail

WT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"

FAMILY=(
    test/registered/scheduler/test_phase_flip_plan.py
    test/registered/scheduler/test_phase_flip_runtime.py
    test/registered/scheduler/test_phase_flip_boot.py
    test/registered/scheduler/test_phase_flip_protocol.py
    test/registered/scheduler/test_gdn_flip_plan.py
    test/registered/scheduler/test_gdn_flip_mover.py
    test/registered/scheduler/test_weights_arena.py
    test/registered/scheduler/test_kv_reshard.py
    test/registered/scheduler/test_step6_harness.py
    test/registered/unit/managers/test_regime_act.py
    test/registered/unit/server_args/test_phase_flip_args.py
    test/registered/unit/distributed/test_phase_flip_groups.py
)

cd "$WT/test/registered/scheduler"
PYTHONPATH="$WT/python" CUDA_VISIBLE_DEVICES=99 \
    "$PY" -m pytest "${FAMILY[@]/#/$WT/}" "$@"
