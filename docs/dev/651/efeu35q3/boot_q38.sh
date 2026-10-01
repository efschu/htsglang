#!/bin/bash
# efeu-TP14: boot for Qwen3.8-35B-A3B-Distill (Q3_K_M) on the 780M (gfx1103).
#
# Derived from #651/#655 boot_ondemand.sh (the proven operating point) with the
# differences this checkpoint and the 2026-10-01 findings require:
#
#   GGUF EXTENSION    the GGUF kernels come from a build WITHOUT real-true16
#                     (GGUF_EXT_DIR, prepended to PYTHONPATH). Root cause of the
#                     gfx1103 Q6_K/Q5_K "per-launch fault": clang-21 true16
#                     codegen puts a D16 load into one half of a VGPR while a
#                     VALU op writes the other half; the write is lost when the
#                     load returns in that window. The default venv build
#                     (gfx1100, true16) stays installed for the old service.
#   GUARD             guard v2 runs against THE SERVING MODULE (not the old
#                     gguf_rocm_probe build) and, with GUARD_STRICT=1, refuses
#                     any launch-to-launch difference -- the fixed build is
#                     deterministic, so a transient is now a failure, not noise.
#   NO kt OFFLOAD     the Q3_K_M weights leave ~5.5 GiB more than Q4 did, so
#                     decode graphs come back (kt forced them off).
#   TOKENIZER         Distill tokenizer + chat template, the proven config.json.
#
# Everything else (wedge policy, mamba slots, memfrac, tool parser) is the
# #655 operating point.
set -u

export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.0.0}
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
export SGLANG_NUM_THREADS=${SGLANG_NUM_THREADS:-12}

source /root/lh/venv/bin/activate
GGUF_EXT_DIR=${GGUF_EXT_DIR:-/root/efeu35q3/ext_v2}
SGL_SRC=${SGL_SRC:-/root/efeu35q3/sglang_src/python}
export PYTHONPATH="$GGUF_EXT_DIR:$SGL_SRC"
if [ -d /opt/ktk ]; then
  export PYTHONPATH="$PYTHONPATH:/opt/ktk"
fi

export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-/var/lib/hicache/kv}
export SGLANG_PINNED_HOST_RESERVE_MIB=${SGLANG_PINNED_HOST_RESERVE_MIB:-512}

sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

# Guard v2 on the module that will actually serve. GGUF_PROBE_MODULE /
# GGUF_FIXTURE_DIR are read by the guard; the fixtures are the #651 K-quant
# slices (q4_K/q5_K/q6_K) plus GUARD_EXTRA_GGUF slices of the served file.
GGUF_PROBE_MODULE=sglang_gguf_rocm GGUF_FIXTURE_DIR=${GGUF_FIXTURE_DIR:-/root/lh/ggufbuild} \
GUARD_STRICT=${GUARD_STRICT:-1} GUARD_NRUNS=${GUARD_NRUNS:-10} GUARD_REQUIRE=${GUARD_REQUIRE:-q4_K,q5_K,q6_K} \
  python ${GUARD_SCRIPT:-/root/efeu35q3/gpu_sanity_guard_v2.py} || {
  echo "GPU sanity guard v2 failed - dequantize is not fit to serve"; exit 1; }

CHUNKED_PREFILL=${CHUNKED_PREFILL:-256}
python /root/651-p2/scripts/wedge_policy.py "$CHUNKED_PREFILL" || {
  if [ "${WEDGE_POLICY_MEASURE:-0}" = "1" ]; then
    # Measurement boots only: the policy's premise (bf16 GEMM M=1024 wedges at
    # ~3 % free GTT) predates the true16 fix and the 4+ GiB headroom of this
    # checkpoint; the chunk-size envelope is re-measured with dmesg watched.
    echo "WEDGE-POLICY: refusal OVERRIDDEN for a measurement boot (cp=$CHUNKED_PREFILL)"
  else
    echo "Wedge policy refused this configuration"; exit 1
  fi; }

MODEL=${MODEL:-/root/efeu35q3/models/Qwen3.8-35B-A3B-Q3_K_M.gguf}
TOKENIZER=${TOKENIZER:-/root/efeu35q3/hf}
MEMFRAC=${MEMFRAC:-0.97}
PORT=${PORT:-31661}
CTX=${CTX:-32768}

MAMBASLOTS=${MAMBASLOTS:-4}
MAMBA_ARGS=()
[ -n "$MAMBASLOTS" ] && MAMBA_ARGS=(--max-mamba-cache-size "$MAMBASLOTS")

GRAPH_ARGS=()
[ "${EAGER:-0}" = "1" ] && GRAPH_ARGS=(--disable-cuda-graph)

TOOLPARSER=${TOOLPARSER:-qwen3_coder}
TOOLPARSER_ARGS=()
[ -n "$TOOLPARSER" ] && TOOLPARSER_ARGS=(--tool-call-parser "$TOOLPARSER")

REASONING_ARGS=()
[ -n "${REASONING_PARSER:-}" ] && REASONING_ARGS=(--reasoning-parser "$REASONING_PARSER")

HICACHE_ARGS=()
if [ "${HICACHE:-0}" = "1" ]; then
  HICACHE_ARGS=(
    --enable-hierarchical-cache
    --hicache-storage-backend file
    --hicache-size 0
    --hicache-ratio "${HICACHE_RATIO:-0.1}"
    --hicache-write-policy "${HICACHE_WRITE_POLICY:-write_through}"
    --hicache-mem-layout page_first_direct
  )
fi

KT_ARGS=()
if [ -n "${KTMETHOD:-}" ]; then
  KT_ARGS=(
    --kt-method "$KTMETHOD" --kt-weight-path "${KTWEIGHTS:-$MODEL}"
    --kt-num-gpu-experts "${KTEXPERTS:-0}" --kt-cpuinfer "${KTCPUINFER:-8}"
    --kt-threadpool-count "${KTPOOLS:-1}" --kt-max-deferred-experts-per-token "${KTDEFER:-0}"
    --disable-cuda-graph
  )
fi

EXTRA_ARGS=()
[ -n "${EXTRA:-}" ] && read -r -a EXTRA_ARGS <<< "$EXTRA"

CMD=(python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "${SERVED_NAME:-qwen38-35b-a3b}" \
  --tokenizer-path "$TOKENIZER" \
  --load-format gguf \
  --quantization gguf \
  --device cuda \
  --tp-size 1 \
  --context-length "$CTX" \
  --max-running-requests 1 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --mamba-radix-cache-strategy no_buffer \
  --disable-overlap-schedule \
  --page-size 1 \
  --mem-fraction-static "$MEMFRAC" \
  "${MAMBA_ARGS[@]}" \
  --chunked-prefill-size "$CHUNKED_PREFILL" \
  "${GRAPH_ARGS[@]}" \
  "${HICACHE_ARGS[@]}" \
  "${KT_ARGS[@]}" \
  "${TOOLPARSER_ARGS[@]}" \
  "${REASONING_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --enable-metrics \
  --host "${HOST:-127.0.0.1}" --port "$PORT" \
  --log-level info)

# POWER PROFILE (measured 2026-10-01, same boot, bs1 decode with graphs):
#   low-power 11.3 tok/s, balanced 20.2 tok/s, performance 20.5 tok/s.
# The laptop sits on low-power; with POWER_PROFILE set, the profile is switched
# while the model is loaded and the previous one is restored when the server
# exits (the on-demand front door parks with SIGTERM to the process group).
if [ -n "${POWER_PROFILE:-}" ] && command -v powerprofilesctl >/dev/null 2>&1; then
  PREV_PROFILE=$(powerprofilesctl get 2>/dev/null || echo balanced)
  powerprofilesctl set "$POWER_PROFILE" && \
    echo "power profile: $PREV_PROFILE -> $POWER_PROFILE (restored on exit)"
  restore_profile() { powerprofilesctl set "$PREV_PROFILE" && echo "power profile restored: $PREV_PROFILE"; }
  trap restore_profile EXIT
  trap 'kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD"; exit 143' TERM INT
  "${CMD[@]}" &
  CHILD=$!
  wait "$CHILD"
  exit $?
fi
exec "${CMD[@]}"
