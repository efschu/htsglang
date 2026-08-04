#!/usr/bin/env bash
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
#
# Serve Qwen3-TTS behind the stock OpenAI /v1/audio/speech surface.
#
# Ports are placeholders resolved from the environment, per the rig-env
# convention -- nothing real belongs in the repository.
#
#   source /root/rig-env.sh
#   scripts/translator/serve_tts.sh
#
# The card is pinned by NVML UUID before the process starts, so `cuda:0` inside
# it is unambiguous. Same doctrine as every other tenant here: isolate at the
# process level, never with an in-process device map.
#
set -euo pipefail

VENV="${TRANSLATOR_TTS_VENV:-/spinning/llm_stuff/translator-models/tts-venv}"
MODEL_DIR="${TRANSLATOR_TTS_MODEL_DIR:-/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base}"
HOST="${TRANSLATOR_TTS_HOST:-127.0.0.1}"
PORT="${TRANSLATOR_TTS_PORT:-30810}"
CARD_UUID="${TRANSLATOR_TTS_CARD_UUID:-<GPU-UUID-of-the-5090>}"
BUDGET_FRACTION="${TRANSLATOR_TTS_GPU_FRACTION:-0.12}"
LOG_DIR="${TRANSLATOR_LOG_DIR:-/spinning/llm_stuff/translator-models/logs}"
PID_FILE="${LOG_DIR}/tts.pid"

die() { echo "error: $*" >&2; exit 1; }

[ -x "$VENV/bin/vllm-omni" ] || die "no vllm-omni in $VENV; run scripts/translator/setup_tts_venv.sh first"
[ -d "$MODEL_DIR" ] || die "no checkpoint at $MODEL_DIR"

case "$CARD_UUID" in
  "<GPU-UUID"*)
    die "TRANSLATOR_TTS_CARD_UUID is unset. Resolve it via NVML -- never a torch
     ordinal, they disagree on this rig:
       nvidia-smi --query-gpu=index,name,uuid --format=csv" ;;
esac

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  die "already running as pid $(cat "$PID_FILE"); stop it first"
fi

echo "serving $(basename "$MODEL_DIR") on ${HOST}:${PORT}, card ${CARD_UUID}"

# `--omni` is what adds the audio surfaces: POST /v1/audio/speech and the
# /v1/audio/voices registry the cloning path needs.
#
# Observed on first contact (2026-08-03), correcting the binary this script was
# written against: the flag belongs to the `vllm-omni` console script, not to
# stock `vllm`. vllm-omni is a PLUGIN whose entry point registers omni models
# into vLLM; its CLI wrapper decides which server to run by scanning argv for
# the literal "--omni" BEFORE argparse runs, and delegates to stock vLLM when
# it is absent. Two consequences worth knowing, because both cost time here:
# the flag is invisible to every `--help` listing, and omitting it does not
# fail -- it silently boots a plain LLM server whose only symptom is that
# /v1/audio/* is missing from /openapi.json.
CUDA_VISIBLE_DEVICES="$CARD_UUID" setsid "$VENV/bin/vllm-omni" serve "$MODEL_DIR" \
  --omni \
  --served-model-name "${TRANSLATOR_TTS_MODEL_NAME:-qwen3-tts}" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$BUDGET_FRACTION" \
  --max-model-len "${TRANSLATOR_TTS_MAX_LEN:-4096}" \
  > "${LOG_DIR}/tts-serve.log" 2>&1 &

echo $! > "$PID_FILE"
echo "pid $(cat "$PID_FILE"), log ${LOG_DIR}/tts-serve.log"

# Bounded wait -- never an unbounded spin, per the robustness canon.
for _ in $(seq 1 120); do
  if curl -sf -m 2 "http://${HOST}:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "ready"
    curl -s -m 5 "http://${HOST}:${PORT}/v1/models"
    exit 0
  fi
  sleep 5
done

echo "did not become ready within 600 s; last log lines:" >&2
tail -20 "${LOG_DIR}/tts-serve.log" >&2
exit 1
