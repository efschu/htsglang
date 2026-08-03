#!/usr/bin/env bash
# Prefill satellite arm (#212) -- runs ON the satellite box.
#
# The satellite is the weaker machine; it owns prefill only. Its KV and (for a
# hybrid GDN model) its mamba state are pushed to the decode arm by the PD
# transfer backend, so this side never decodes past the handoff token.
#
# Transport: mooncake_tcp. The RDMA line is not reachable from every location
# on this rig (see docs/rig-runbook.md section 1) and TCP needs no HCA; the
# arg hook rewrites the backend to mooncake with MC_FORCE_TCP=1 and drops the
# IB device. Swap to --disaggregation-transfer-backend mooncake plus
# --disaggregation-ib-device <hca> when both arms sit on the fast line.
#
# Env (all have defaults; override on the command line):
#   SAT_SGLANG_SRC   python tree to run (PYTHONPATH)
#   SAT_VENV_PY      interpreter
#   SAT_MODEL        model path -- MUST be the same checkpoint as the decode arm
#   SAT_PORT         http port          (default 31212)
#   SAT_BOOTSTRAP    bootstrap port     (default 8998)
#   SAT_MEM_FRAC     --mem-fraction-static
#   SAT_CUDART12     directory holding libcudart.so.12 for the mooncake wheel
set -euo pipefail

SAT_SGLANG_SRC="${SAT_SGLANG_SRC:-<SATELLITE_SGLANG_SRC>}"
SAT_VENV_PY="${SAT_VENV_PY:-<SATELLITE_PYTHON>}"
SAT_MODEL="${SAT_MODEL:-<SATELLITE_MODEL_DIR>}"
SAT_PORT="${SAT_PORT:-31212}"
SAT_BOOTSTRAP="${SAT_BOOTSTRAP:-8998}"
SAT_MEM_FRAC="${SAT_MEM_FRAC:-0.88}"
SAT_CTX="${SAT_CTX:-16384}"
SAT_LOG="${SAT_LOG:-/tmp/t212_satellite_prefill.log}"
SAT_CUDART12="${SAT_CUDART12:-}"
# Turing (sm75) has a 64 KiB shared-memory ceiling per block. flashinfer's
# paged prefill kernel asks for 65616 bytes at head_dim 256 (Qwen3.5/3.6) and
# is rejected at the first prefill -- after a clean weight load and a sized KV
# pool, so it reads as a runtime fault rather than a capability one. Triton
# is the backend that fits. Override only on a card that is not sm75.
SAT_ATTN="${SAT_ATTN:-triton}"
# Both arms must answer with the same served name, or the preflight in
# prefill_offload.py cannot tell "same model, two mounts" from "two models".
# The default is the model path, which necessarily differs across boxes.
SAT_SERVED_NAME="${SAT_SERVED_NAME:-satellite-pair}"
# The satellite is memory-poor by definition, and the KV-budget profiler does
# not see the Triton GDN prefill scratch: the boot succeeds, reports its free
# memory, and then dies on the first REAL prefill with
# "Triton Error [CUDA]: out of memory" while a 4-token warmup passed. The
# scratch scales with the chunk, so shrink the chunk rather than the context.
SAT_CHUNK="${SAT_CHUNK:-512}"
# --dtype float16 below is NOT a preference. Turing has no bfloat16, so the
# satellite would cast to fp16 on its own; the decode arm would then keep the
# checkpoint's bfloat16 and the two would exchange KV bytes of different
# widths. The pair is fp16 because the weakest member is.

export PYTHONPATH="$SAT_SGLANG_SRC"
# The mooncake transfer-engine wheel links CUDA 12 while the torch stack here
# is CUDA 13; without this the import dies on libcudart.so.12 and the PD arm
# falls back to no transport at all.
if [ -n "$SAT_CUDART12" ]; then
  export LD_LIBRARY_PATH="$SAT_CUDART12:${LD_LIBRARY_PATH:-}"
fi
# Turing has no bfloat16; the SSM state is the one tensor that must stay wide.
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-float32}"

exec "$SAT_VENV_PY" -u -m sglang.launch_server \
  --model-path "$SAT_MODEL" \
  --served-model-name "$SAT_SERVED_NAME" \
  --tp-size 1 --base-gpu-id 0 \
  --dtype float16 \
  --mem-fraction-static "$SAT_MEM_FRAC" \
  --context-length "$SAT_CTX" \
  --max-running-requests 4 \
  --chunked-prefill-size "$SAT_CHUNK" \
  --page-size 1 \
  --attention-backend "$SAT_ATTN" --sampling-backend pytorch \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port "$SAT_BOOTSTRAP" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --trust-remote-code --enable-metrics \
  --host 0.0.0.0 --port "$SAT_PORT" \
  >> "$SAT_LOG" 2>&1
