#!/usr/bin/env bash
# V4 boot proof -- step 4 of ANALYSE_321 §9.2, plus the collective-floor eraser.
#
# WHAT THIS PROVES
# ----------------
# "V4" is the all-Linear NVFP4 variant (compressed-tensors
# `nvfp4-pack-quantized`, MLP + GDN + full-attn at 4 bits, 24.35e9 params,
# embed/lm_head/MTP left in bf16). ANALYSE_321 §2 identifies it as the ONLY
# published NVFP4 flavour that is simultaneously VRAM-positive (-7736 MiB),
# context-positive (1.57x), decode-positive (-27 %) and prefill-neutral
# (-3.1 %) -- and, before #291-S3, the only one that could not boot at all on
# a rig with a pre-Blackwell rank (`get_min_capability() == 100`, no fallback).
#
# Two arms, in this order:
#
#   ARM 1  TP=3 uneven, after the #291-S3 fix.
#          What it proves: the sm_86 ranks now take the Marlin E2M1 lane
#          instead of raising NotImplementedError, and the #323a coarsening
#          makes the uneven shards Marlin-tile-valid. Expect the 3080 ranks to
#          log the Marlin backend and the 5090 rank a native FP4 backend --
#          two different lanes in one model, which is the point.
#
#   ARM 2  solo 5090.
#          What it proves: the §6.2 result -- V4 is the FIRST weight format
#          under which Qwen3.6-27B fits on one 5090 with a real KV pool.
#          Expected: ~17.9 GiB of weights and ~326k KV tokens, against FP8's
#          78k. This arm deletes the 64-80 % collective floor that dominates
#          the TP=3 window on this interconnect (all PHB, no P2P, no NVLink),
#          and it is the enabling condition for the weightless KV lane.
#          Predicted prefill for a 2048-token chunk: ~161 ms solo vs ~1206 ms
#          under the best TP=3 plan -- a 7.5x difference that is entirely
#          transport, not compute.
#
# FALSIFIERS (do not skip -- a boot that runs is not a boot that is right)
# -----------------------------------------------------------------------
#   * Coherence: the #289 five-prompt set must come back coherent on BOTH
#     arms. A wrong global-scale inversion in the Marlin branch does not
#     crash -- it scales every output by global_scale**2 and degrades quietly.
#   * The two arms must agree semantically. They run different lanes on
#     different cards; divergence points at the mixed-arch W4A4 determinism
#     risk of ANALYSE_321 §7(e), not at a boot bug.
#   * ARM 1's prefill should land within ~3 % of the FP8 baseline window. The
#     analysis predicts -3.1 %; a large positive delta means the §5 model is
#     wrong and everything downstream needs re-deriving. Good falsifier.
#
# BEFORE RUNNING
# --------------
#   * Take the GPU window (/spinning/gpu-arb) -- this needs all three cards
#     exclusively for ARM 1.
#   * Run scripts/nvfp4/phi0_lane_microbench.py FIRST. It costs seconds and it
#     tells you which backend each rank will resolve, which is the first thing
#     you will want when reading these logs.
#   * A V4 checkpoint must be present. ANALYSE_321 §7(d) ranks them:
#       1. ocicek/Qwen3.6-27B-NVFP4              (MTP cleanly separated as
#                                                 model-mtp-bf16.safetensors --
#                                                 the pattern to prefer)
#       2. AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
#       3. llmfan46/...-Native-MTP-Preserved-NVFP4
#     Per the #318 lesson, read `mtp.*` out of the checkpoint's own
#     hf_quant_config.json `ignore` list -- the `-MTP` suffix in a repo name
#     implies nothing about whether the drafter is quantised. Prefer a bf16
#     drafter: the MTP layer runs at shapes where a 4-bit GEMM has no
#     arithmetic advantage and accept length is directly quality-sensitive.
#
# USAGE
#   scripts/nvfp4/v4_boot_proof.sh <arm>   with arm in {tp3,solo,both}
#   MODEL=/path/to/v4-checkpoint scripts/nvfp4/v4_boot_proof.sh both

set -euo pipefail

ARM="${1:-both}"
MODEL="${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-NVFP4}"
PORT="${PORT:-30000}"
OUT="${OUT:-/spinning/gpu-battery-results/nvfp4_v4_boot}"
# Time-boxed, per the standing rule: bound the run by TIME, not by token count.
RUN_SECONDS="${RUN_SECONDS:-20}"

mkdir -p "$OUT"

# --- NVML-resolved card indices --------------------------------------------
# Never assume physical index 0 is the 5090; NVML/nvidia-smi order can shift
# between boots and driver states.
resolve_cards() {
  python3 - <<'PY'
import pynvml
pynvml.nvmlInit()
five, threes = None, []
for i in range(pynvml.nvmlDeviceGetCount()):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()
    if "5090" in name:
        five = i
    elif "3080" in name:
        threes.append(i)
    print(f"# nvml[{i}] {name}", flush=True)
pynvml.nvmlShutdown()
assert five is not None, "no RTX 5090 found via NVML"
print(f"FIVE={five}")
print(f"THREES={','.join(str(i) for i in threes)}")
PY
}

eval "$(resolve_cards | tee "$OUT/nvml_inventory.txt" | grep -E '^(FIVE|THREES)=')"
echo "resolved: 5090 -> ${FIVE}, 3080s -> ${THREES}"

COMMON=(
  --model-path "$MODEL"
  --served-model-name Qwen3.6-27B-NVFP4
  --trust-remote-code
  --port "$PORT"
  --mem-fraction-static 0.9
  --context-length -1
)

wait_ready() {
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if curl -s -m 5 "http://127.0.0.1:${PORT}/health_generate" >/dev/null 2>&1; then
      return 0
    fi
    if ! (kill -0 "$1" 2>/dev/null); then
      echo "server pid $1 died before becoming ready" >&2
      return 1
    fi
    sleep 5
  done
  echo "server did not become ready within the deadline" >&2
  return 1
}

# The #289 coherence set: five prompts, checked for coherence and for
# agreement between the two arms.
COHERENCE_PROMPTS=(
  "Explain in three sentences why 4-bit weight quantisation helps decode latency more than prefill throughput."
  "Write a Python function that returns the n-th triangular number."
  "What is the capital of Australia, and why is it not Sydney?"
  "Summarise the difference between tensor parallelism and pipeline parallelism."
  "List four causes of memory fragmentation in a paged KV cache."
)

probe() {
  local tag="$1"
  {
    echo "## ${tag}"
    curl -s -m 30 "http://127.0.0.1:${PORT}/get_server_info" || true
    echo
    for prompt in "${COHERENCE_PROMPTS[@]}"; do
      echo "### ${prompt}"
      curl -s -m 120 "http://127.0.0.1:${PORT}/generate" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1], "sampling_params": {"temperature": 0, "max_new_tokens": 160}}))' "$prompt")" \
        || true
      echo
    done
  } | tee "$OUT/${tag}_probe.json"
}

boot() {
  local tag="$1"; shift
  local log="$OUT/${tag}_server.log"
  echo "--- ${tag}: booting, log -> ${log}"
  # setsid so the server survives this script's own process group and a stray
  # Ctrl-C never leaves a half-killed rank holding VRAM.
  setsid python3 -m sglang.launch_server "${COMMON[@]}" "$@" >"$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$OUT/${tag}.pid"
  if wait_ready "$pid"; then
    probe "$tag"
    # Time-boxed steady-state sample rather than a fixed token count.
    sleep "$RUN_SECONDS"
    curl -s -m 30 "http://127.0.0.1:${PORT}/metrics" > "$OUT/${tag}_metrics.txt" || true
  fi
  # Kill only our own pid -- never a broad pkill, other sessions share this box.
  kill -TERM "-$(ps -o pgid= "$pid" | tr -d ' ')" 2>/dev/null || true
  local deadline=$((SECONDS + 120))
  while (kill -0 "$pid" 2>/dev/null) && (( SECONDS < deadline )); do sleep 2; done
  kill -KILL "$pid" 2>/dev/null || true
  echo "--- ${tag}: done"
}

if [[ "$ARM" == "tp3" || "$ARM" == "both" ]]; then
  IFS=, read -r T0 T1 <<< "$THREES"
  # ARM 1: the #291-S3 proof. Two Marlin ranks + one native-FP4 rank in one
  # model, with the uneven plan that #323a makes tile-valid.
  CUDA_VISIBLE_DEVICES="${FIVE},${T0},${T1}" \
  boot v4_tp3_uneven \
    --tensor-parallel-size 3 \
    --rank-tp-ratio auto \
    --rank-auto-reserve-mib 3000,2700,2700
fi

if [[ "$ARM" == "solo" || "$ARM" == "both" ]]; then
  # ARM 2: the collective-floor eraser. §6.2 expects ~17.9 GiB of weights and
  # ~326k KV tokens on the 5090 alone. Grep the log for max_total_num_tokens
  # and the weight footprint and put both against those two numbers.
  CUDA_VISIBLE_DEVICES="${FIVE}" \
  boot v4_solo_5090 \
    --tensor-parallel-size 1
fi

echo
echo "=== what to read out of ${OUT} ==="
echo "  1. grep -E 'fp4|nvfp4|marlin|backend' *_server.log"
echo "     ARM 1 must show TWO different FP4 lanes across the three ranks."
echo "  2. grep -E 'max_total_num_tokens|KV Cache is allocated|weight' v4_solo_5090_server.log"
echo "     expected ~17.9 GiB weights and ~326k tokens (FP8 leaves 78k)."
echo "  3. diff the five coherence answers between the two arms."
echo "     Divergence is the mixed-arch W4A4 determinism risk (ANALYSE_321"
echo "     §7 e), not a boot bug -- it needs the #50 battery, not a retry."
echo "  4. ARM 1 prefill vs the FP8 baseline window: predicted -3.1 %."
echo "     A large POSITIVE delta falsifies the §5 cost model."
