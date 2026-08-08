#!/usr/bin/env bash
# #631 Route A step-6 boot: ONE instance, THREE ranks, PP=3 prefill boot
# topology with the TP=3/DCP=3 flip stack built beside it
# (--enable-phase-flip). Supersedes route_a_631_pd_boot.sh (the PD pair
# was the pre-correction architecture; kept only as reference).
#
# DEVICE SELECTION IS BY UUID, NEVER BY NVML INDEX (NVML drift moved the
# 5090 between indices across boots on this rig -- measured). The 5090 is
# ordered FIRST in CUDA_VISIBLE_DEVICES: stage 0 carries 32/64 layers and
# TP rank 0 carries 30/64 of the token vector, both sized for the big
# card; --rank-gpu-id then indexes INTO the visible set, which is
# ordering-independent.
#
# Recipe provenance: the measured #625 PP=3 arm (runbook 4.7.1) plus the
# #631 flip flags. --disable-overlap-schedule matches the measured
# recipe; the flip's TP phase then runs event_loop_normal (the
# re-dispatch wrapper's non-overlap branch).
#
# FIRST-BOOT MEASUREMENT OBLIGATIONS (DESIGN_631 3.4b/4 -- a first boot
# that does not produce these numbers is incomplete):
#   * SGLANG_NCCL_BUFFER_DUMP armed below -> per-group NCCL buffer cost
#     incl. the flip_tp/flip_dcp/flip_pp secondary set (read the dump,
#     update the 3.4a ledger's ~500 MiB guess).
#   * activation term: read the post-capture leftover log lines (the 4016
#     MiB heuristic is known-falsified-high; update the ledger).
#   * NVML-free >= 1024 MiB per card CONTINUOUSLY under load: run the
#     100-ms sampler alongside every load window (time-series minimum,
#     NVML FREE column, never total-used).
#
# MEMORY BUDGETS: the #625 PP arm ran 28500,17500,17500 with pool-luxury
# sizing. Under the flip the SAME cards also carry the TP stack's pools +
# the weights arena delta (+2.34/0/+0.75 GB), so the PP-side budgets
# below are cut to the 3.4a ledger's conservative first-boot point.
# Override via RANK_MIB for tuning; the pin-3 boot assert and the
# physical-fit checks fail fast on a wrong guess.
set -euo pipefail

WT="${WT:-/spinning/wt-631-routea}"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
MODEL="${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8}"
PORT="${PORT:-30023}"
RANK_MIB="${RANK_MIB:-22000,13000,13000}"
LOGDIR=/tmp/route-a-631
mkdir -p "$LOGDIR"

export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH=/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
# The flip's TP phase token-shards KV under the WEIGHTED owner rule; the
# boot builder REFUSES without this pair (phase_flip_boot).
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
# Mixed 5090+3080 group is the CONFIGURATION, not a symptom.
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
# First-boot ledger measurement: per-group NCCL buffer costs, incl. the
# flip secondary set (mem_ledger/nccl_probe brackets every pynccl ctor).
export SGLANG_NCCL_BUFFER_DUMP="${SGLANG_NCCL_BUFFER_DUMP:-$LOGDIR/nccl_buffers.json}"

# --- resolve cards by NAME -> UUID, 5090 FIRST -------------------------------
BIG_UUID=""; SMALL_UUID=()
while IFS=, read -r _idx name uuid _tot; do
    name="$(echo "$name" | xargs)"; uuid="$(echo "$uuid" | xargs)"
    case "$name" in *5090*) BIG_UUID="$uuid" ;; *3080*) SMALL_UUID+=("$uuid") ;; esac
done < <(nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader)
[[ -n "$BIG_UUID" && ${#SMALL_UUID[@]} -ge 2 ]] || {
    echo "FATAL: expected one 5090 and two 3080s from NVML" >&2; exit 1; }
echo "cuda:0=5090=$BIG_UUID"
echo "cuda:1=3080a=${SMALL_UUID[0]}"
echo "cuda:2=3080b=${SMALL_UUID[1]}"

CUDA_VISIBLE_DEVICES="$BIG_UUID,${SMALL_UUID[0]},${SMALL_UUID[1]}" \
exec "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --trust-remote-code \
    --tp-size 1 --pp-size 3 \
    --pp-stage-ratio 2,1,1 \
    --rank-gpu-id 0,1,2 \
    --rank-gpu-memory-mib "$RANK_MIB" \
    --enable-phase-flip \
    --phase-flip-tp-vector 30,17,17 \
    --disable-overlap-schedule \
    --kv-cache-dtype fp8_e4m3 --context-length 65536 \
    --enable-metrics --host 127.0.0.1 --port "$PORT" \
    "$@"
