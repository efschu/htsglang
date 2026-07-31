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
# RESULT OF THE FIRST RUN (2026-07-31, ocicek/Qwen3.6-27B-NVFP4)
# ---------------------------------------------------------------
#   * ARM 1 does NOT boot, and not for a reason #291-S3 or #323a can fix.
#     The checkpoint quantises the GDN b/a gate (`create_ba_proj`), whose full
#     width is 96 rows -- below the Marlin 64-tile at TP=1 already. Confirmed
#     by falsifier: a TP=1 boot on one 3080, with no shard plan at all, fails
#     with the same message. sm_86 + all-Linear NVFP4 is blocked at the
#     checkpoint, not at the split.
#   * ARM 2 boots, serves and is coherent. 18.81 GiB of weights, 153,007 KV
#     tokens, 7.70x the TP=3 fp8 prefill throughput.
#   * NEXTN is not available on this checkpoint: its compressed-tensors
#     `ignore` list names `mtp.*` unfused, the draft module is built as
#     `model.layers.0.*` fused, so the drafter comes out quantised against a
#     bf16 shard. Run ARM 2 without `--speculative-algorithm`.
# Details: docs/dev/INTEGRATION_R3_VALIDATION.md, section "NVFP4-Beleg".
#
# WHAT #332 CHANGED, AND WHAT TO EXPECT THIS TIME
# -----------------------------------------------
# All three findings above were code defects, and all three are fixed. This
# script is the acceptance program for them. Every expectation below is
# derived from the numbers the first run measured, so a miss is readable.
#
#   POSTEN 1 -- ARM 1 must now BOOT. A compressed-tensors NVFP4 layer whose
#     UNSHARDED width misses the Marlin tile is loaded packed and materialised
#     dense at load instead of aborting. Expect, on EVERY rank, one warning
#     line per affected layer:
#       "NVFP4: layer '...linear_attn.in_proj_ba' has no Marlin form at any TP
#        size (the unsharded output width is 96, ...) ... DEQUANTISED ..."
#     Count them: 48 GDN layers x the number of MARLIN ranks. On this rig that
#     is 96, not 144 -- MEASURED 2026-07-31, 48 on TP1 and 48 on TP2 and ZERO
#     on TP0. The zero is correct and not a miss: the guard asks whether the
#     unsharded width has a MARLIN form, and the 5090 rank does not use Marlin
#     at all, it uses the native FP4 lane. The old "144" assumed three sm_86
#     ranks, i.e. a different rig. More than 48 per Marlin rank means an MLP
#     layer fell into the fallback and the VRAM figures below will be wrong.
#     Cost check: the dense b/a gates add ~16 MiB across the whole model.
#     Anything larger is a routing bug, not a rounding difference.
#
#   POSTEN 2 -- ARM 1 runs WITH `--speculative-algorithm NEXTN`. The
#     checkpoint's `ignore` list names the drafter's projections unfused; the
#     fused-name match now resolves them, so the drafter is built dense.
#     Expect: no #318 raise, ZERO unloaded draft parameters, and an accept
#     length WELL above 1.0 -- 1.0052 with 0/573 accepted drafts is the exact
#     signature of a drafter on uninitialised weights (#316's card proof), so
#     it is the falsifier, not a disappointment. Read
#     `meta_info.spec_accept_length`, never `spec_ema_accept_len`.
#
#   POSTEN 3 -- ARM 2 is sized by RESERVE, not by fraction. The first run left
#     2.44 GiB unused at `--mem-fraction-static 0.90` (~80k KV tokens at that
#     run's 31.9 KiB/token), so 153,007 was not the arm's ceiling; §5 puts the
#     ceiling near 233k. `--rank-auto-reserve-mib <MiB>` now works without
#     `--rank-gpu-id` and sets the budget to NVML total minus exactly that
#     reserve. At the default 2048 MiB below, expect roughly
#       153,007 + (3261 - 2048) MiB / 31.9 KiB ~= 192k tokens,
#     i.e. +25 %, with weights unchanged at 18.81 GiB. Push the reserve lower
#     to approach 233k -- that is deliberately the user's risk, no margin is
#     added on top, and the VRAM corridor rule (>= 400 MiB free) is the floor
#     to check afterwards. Note one interaction: on the mamba post-capture
#     path the slack is floored by the pre-capture reserve, so a reserve below
#     that floor stops buying tokens; the boot log names both.
#
# USAGE
#   scripts/nvfp4/v4_boot_proof.sh <arm>   with arm in {tp3,solo,both}
#   MODEL=/path/to/v4-checkpoint scripts/nvfp4/v4_boot_proof.sh both
#   SOLO_RESERVE_MIB=1500 scripts/nvfp4/v4_boot_proof.sh solo
#   NEXTN=0 scripts/nvfp4/v4_boot_proof.sh tp3      # posten-1-only control

set -euo pipefail

# Rig environment (docs/rig-runbook.md sections 1.1 and 2). Without the venv
# interpreter, PYTHONPATH and the cu13 library path this script silently runs
# a different checkout, or dies in the first deep_gemm JIT compile.
source /root/rig-env.sh 2>/dev/null || true
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
PY="${PY:-$VENV/bin/python}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

ARM="${1:-both}"
MODEL="${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-NVFP4}"
PORT="${PORT:-30000}"
OUT="${OUT:-/spinning/gpu-battery-results/nvfp4_v4_boot}"
# Time-boxed, per the standing rule: bound the run by TIME, not by token count.
RUN_SECONDS="${RUN_SECONDS:-20}"
# #332 posten 3: the solo arm's ENTIRE headroom, in MiB. 0.90 of this card
# withheld 3261 MiB; 2048 is the first honest step down from that. Lower it to
# walk towards the ~233k ceiling -- nothing is added on top of what you name.
SOLO_RESERVE_MIB="${SOLO_RESERVE_MIB:-2048}"
# #332 posten 2: ARM 1 with speculation, which the ignore-match fix unblocks.
NEXTN="${NEXTN:-1}"

mkdir -p "$OUT"

# --- NVML-resolved card identity --------------------------------------------
# Never assume physical index 0 is the 5090; NVML/nvidia-smi order can shift
# between boots and driver states.
#
# Resolve to UUIDs, not to indices. `CUDA_VISIBLE_DEVICES` is interpreted in
# CUDA enumeration order (FASTEST_FIRST on this rig), which is NOT NVML order:
# NVML index 1 is the 5090 but CUDA index 0 is (docs/rig-runbook.md 6.1).
# Feeding an NVML index into CUDA_VISIBLE_DEVICES silently pins the wrong card.
# A UUID means the same card in both order systems.
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
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    if isinstance(uuid, bytes):
        uuid = uuid.decode()
    if "5090" in name:
        five = uuid
    elif "3080" in name:
        threes.append(uuid)
    print(f"# nvml[{i}] {name} {uuid}", flush=True)
pynvml.nvmlShutdown()
assert five is not None, "no RTX 5090 found via NVML"
print(f"FIVE={five}")
print(f"THREES={','.join(threes)}")
PY
}

eval "$(resolve_cards | tee "$OUT/nvml_inventory.txt" | grep -E '^(FIVE|THREES)=')"
echo "resolved: 5090 -> ${FIVE}, 3080s -> ${THREES}"

COMMON=(
  --model-path "$MODEL"
  --served-model-name Qwen3.6-27B-NVFP4
  --trust-remote-code
  --port "$PORT"
  --context-length -1
  # Mandatory on every launch_server call on this rig (rig-runbook.md 3).
  --enable-metrics
)
# Sizing is per arm now, not shared: ARM 1 keeps the per-GPU placement
# reserves it always had, ARM 2 takes the #332 card-wide one. The two are
# mutually exclusive with --mem-fraction-static by design.

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
  setsid "$PY" -m sglang.launch_server "${COMMON[@]}" "$@" >"$log" 2>&1 &
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
  # ARM 1: the #291-S3 proof, now reachable. Two Marlin ranks + one native-FP4
  # rank in one model, with the uneven plan that #323a makes tile-valid, the
  # #332 dequant lane carrying the 96-row GDN gates that used to abort the
  # load, and NEXTN on top now that the ignore match sees fused names.
  SPEC=()
  if [[ "$NEXTN" == "1" ]]; then
    SPEC=(--speculative-algorithm NEXTN --speculative-num-steps 3
          --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
  fi
  # Placement is --rank-gpu-id, not CUDA_VISIBLE_DEVICES. `--rank-tp-ratio auto`
  # derives its per-rank budgets from NVML totals and therefore REQUIRES
  # --rank-gpu-id; pinning by UUID and passing `auto` alone aborts at argument
  # validation ("--rank-tp-ratio auto requires --rank-gpu-id") without ever
  # reaching a card. All three cards are in play here, so there is nothing to
  # hide, and index 0 resolves to the 5090 (budgets [29607, 17780, 17780] MiB),
  # which is what the 3000/2700/2700 reserve list assumes.
  boot v4_tp3_uneven \
    --tensor-parallel-size 3 \
    --rank-gpu-id 0,1,2 \
    --rank-tp-ratio auto \
    --rank-auto-reserve-mib 3000,2700,2700 \
    "${SPEC[@]}"
fi

if [[ "$ARM" == "solo" || "$ARM" == "both" ]]; then
  # ARM 2: the collective-floor eraser, sized honestly. The first run got
  # 18.81 GiB of weights and 153,007 KV tokens at --mem-fraction-static 0.90
  # and left 2.44 GiB on the card; this arm names the headroom instead.
  CUDA_VISIBLE_DEVICES="${FIVE}" \
  boot v4_solo_5090 \
    --tensor-parallel-size 1 \
    --rank-auto-reserve-mib "$SOLO_RESERVE_MIB"
fi

echo
echo "=== what to read out of ${OUT} (expectations from the 2026-07-31 beleg) ==="
echo "  1. grep -E 'fp4|nvfp4|marlin|backend' *_server.log"
echo "     ARM 1 must show TWO different FP4 lanes across the three ranks."
echo "  2. #332 posten 1 -- grep -c 'DEQUANTISED' v4_tp3_uneven_server.log"
echo "     expected 96 on this rig (48 GDN layers x the TWO Marlin ranks), each"
echo "     naming in_proj_ba and the unsharded width 96; the 5090 rank"
echo "     contributes ZERO because it takes the native FP4 lane. Zero overall"
echo "     means ARM 1 died the old death; more than 48 per Marlin rank means"
echo "     an MLP layer fell into the dense lane."
echo "  3. #332 posten 2 -- grep -E 'unloaded|accept' v4_tp3_uneven_server.log"
echo "     expected: no #318 raise, zero unloaded draft parameters, and"
echo "     meta_info.spec_accept_length well above 1.0. Exactly 1.0052 with"
echo "     0 accepted drafts is the drafter-on-uninitialised-weights"
echo "     signature -- that is the falsifier."
echo "  4. #332 posten 3 -- grep -E 'max_total_num_tokens|Reserve-based sizing|weight' \\"
echo "       v4_solo_5090_server.log"
echo "     expected 18.81 GiB weights (unchanged) and, at the default"
echo "     2048 MiB reserve, ~192k tokens against the first run's 153,007."
echo "     The sizing line must report total - reserve with no extra margin."
echo "     Then check the VRAM corridor: >= 400 MiB free after the boot."
echo "  5. diff the five coherence answers between the two arms."
echo "     Divergence is the mixed-arch W4A4 determinism risk (ANALYSE_321"
echo "     §7 e), not a boot bug -- it needs the #50 battery, not a retry."
echo "     ARM 1 now boots, so §7(e) is finally answerable in one window."
echo "  6. ARM 1 prefill vs the FP8 baseline window: predicted -3.1 %."
echo "     A large POSITIVE delta falsifies the §5 cost model."
