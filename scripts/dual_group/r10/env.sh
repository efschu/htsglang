#!/usr/bin/env bash
# Shared environment for the #340 card window (lane device guard, r10).
# Base is /spinning/wt-340 -- the worktree that carries the guard fix.
source /root/rig-env.sh 2>/dev/null || true
export WT=${WT:-/spinning/wt-340}
export OUT=${OUT:-/spinning/gpu-battery-results/2026-07-31_340_gpu}
export PY="${PY:-$VENV/bin/python}"
export MODEL_ROOT=${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export FLASHINFER_DISABLE_VERSION_CHECK=1

mkdir -p "$OUT/logs" "$OUT/posten1" "$OUT/posten2"
