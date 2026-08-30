#!/bin/bash
# #855 BOOT PROOF -- arm B: the GDN-covered W8A8 artifact, on the shipping tree.
#
# Derived from boot_969flip_nogrid_s3.sh with the launch-flag block kept
# BYTE-IDENTICAL apart from ONE line: --model-path. That is deliberate --
# arm A is the 2026-08-30 03:07 boot (boot_969nogrid_a51e5e8f28_0830_030705.log)
# on the SAME tree with the SAME flags and the incumbent
# Qwen3.8-27B-INT8-vocabint8-embed, so the KV-pool delta is attributable to the
# checkpoint and to nothing else.
#
# ARTIFACT: Qwen3.8-27B-INT8-gdncov-vocabembed, 27.541 GiB
#   = the #855 GDN requant (tools/requant_gdn_int8_855.py, commit 3dc2af094d)
#     applied ON TOP OF the live #727 int8-embed checkpoint, i.e. the UNION of
#     both INT8 axes rather than a swap.  The plain Qwen3.8-27B-INT8-gdncov of
#     3dc2af094d was built from the PLAIN incumbent and would have GIVEN BACK
#     the #727 embed win (-1.18 GiB) to buy the GDN win.
#   Live vocabint8-embed 32.695 GiB -> 27.541 GiB = -5.154 GiB per rank image.
#
# ACCOUNTABILITY (user law int8-ueberall-mit-rechenschaft, 2026-08-30):
#   Quality cost adopted here: 144 GDN dense projections (in_proj_qkv,
#   in_proj_z, out_proj x 48 layers) move BF16 -> int8 per-channel symmetric
#   weights with dynamic per-token int8 activations.  Measured weight error on
#   the plain gdncov build: rel-Frobenius median 1.018 %, worst 1.447 %
#   (layers.57.out_proj, SNR 36.8 dB) -- NOTE_855_gdncov_artifact.md.  Crest
#   factors on GDN (median 4.0-4.3) match the MLP tensors the incumbent already
#   quantizes (median 3.8-4.5), so this is the same population, not a
#   pathological one.  Bought with it: -5.154 GiB of weight image per rank,
#   which becomes KV pool, plus a measured 1.39x (sm120) / 1.46x (sm86) prefill
#   linear-layer gain (NOTE_855_microbench.md, GPU window W25).  The
#   ACTIVATION axis is the honestly-named residual risk: 144 paths that were
#   BF16 now run per_token_quant_int8, and the ~1 % weight error does not bound
#   that -- which is exactly what this boot measures.
#   NOT taken here: lm_head int8 (the Qwen3.8-27B-INT8-vocabint8-both axis,
#   a further -1.18 GiB).  Reason: it is a SECOND unproven axis and stacking two
#   unproven quality deltas in one boot makes an attribution impossible.  It is
#   the next lever if this boot is clean, not a rejection.
#
# DEVIATION from the flip script, stated per speed-mode rules:
#   SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION is left at its default (1) here.
#   The flip strand forced it to 0 because a flip could delay the reply past
#   the curl timeout.  The standing dead-man order (2026-08-30) makes
#   /health_generate the only honest liveness probe, and the deadman requires
#   TWO consecutive failures at m=25s, so a single flip-delayed probe does not
#   fire it.  Liveness detection outranks the false-positive risk.
set -u

TREE=/spinning/wt-855-int8
VENV=/spinning/htsglang-gpu/.venv
# MODEL / TAG / RANK_MIB are overridable so the SAME flag block serves every arm
# of this strand -- the arm-A KLD-capture boot, the footprint-calibration boot,
# and the retuned final boot differ ONLY in these, which is what makes the
# comparisons attributable (Patchstand vor Last).
MODEL=${MODEL_ARG:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed}
TAG=${TAG_ARG:-gdncov}
RANK_MIB=${RANK_MIB_ARG:-31800,18800,19800}
# FLIP_POLICY: "auto" is the production shape. The MEASUREMENT arms use "manual"
# so a long chunked prefill cannot race an auto-arm -- see the 05:05Z wedge
# (W855-WEDGE-SPECIMEN.md): flip armed tp_to_pp while a chunked prefill was
# incomplete, strict drain refused, and the group livelocked. That is the
# flip strand's #856/#819 fault domain, not the checkpoint's; holding the
# layout still is what makes the INT8 ladder attributable to the checkpoint.
FLIP_POLICY=${FLIP_POLICY_ARG:-auto}
STAMP=$(date -u +%m%d_%H%M%S)
LOG=/spinning/evidence-665-f1/boot_855_${TAG}_0840f82601_${STAMP}.log

NVRTC="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$NVRTC:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
. /root/rig-env.sh
set -a
# shellcheck disable=SC1091
. /root/boot800_env.txt
set +a
export PYTHONPATH="$TREE/python"

export SGLANG_ARMING_FLOOR_SOLVED=1
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1
# FRESH HiCache store: the model changed, and a store keyed on the previous
# checkpoint's geometry would be a two-geometry key (HiCache-Phasen-Uniform).
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/tmp/hicache_855_${TAG}
mkdir -p "/tmp/hicache_855_${TAG}"

mkdir -p /spinning/evidence-665-f1

{
  echo "=== #855 BOOT (arm B, gdncov+vocabembed union) ==="
  echo "date        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "tree        : $TREE @ $(git -C "$TREE" rev-parse --short=10 HEAD) ($(git -C "$TREE" rev-parse --abbrev-ref HEAD))"
  echo "tag         : $TAG"
  echo "rank_mib    : $RANK_MIB"
  echo "flip_policy : $FLIP_POLICY"
  echo "model       : $MODEL"
  echo "footprint   : ${SGLANG_PHASE_FOOTPRINT_DUMP:-<not armed>}"
  echo "model bytes : $(du -sbL "$MODEL" | awk '{printf "%s (%.3f GiB)", $1, $1/1073741824}')"
  echo "arm A ref   : /spinning/evidence-665-f1/boot_969nogrid_a51e5e8f28_0830_030705.log"
  echo "arm A model : /spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-vocabint8-embed"
  echo "arm A bytes : $(du -sbL /spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-vocabint8-embed | awk '{printf "%s (%.3f GiB)", $1, $1/1073741824}')"
  echo "CVD         : ${CUDA_VISIBLE_DEVICES}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,uuid --format=csv,noheader | sed 's/^/  nvml: /'
  echo "SEAM-CACHE RECORDS CONSUMED (WINDOW-PROTOCOL gate 2, named):"
  for f in /root/.cache/sglang/kv_budget-*-seam-rank*.json; do
    [ -f "$f" ] && echo "  $f  mtime=$(date -u -r "$f" +%Y-%m-%dT%H:%M:%SZ)"
  done
  echo "  NOTE: the seam reserve is read from the PREVIOUS boot's record, so"
  echo "  the arm A / arm B pool comparison is not a clean A/B of the"
  echo "  checkpoints alone (WINDOW-PROTOCOL gate 2). The WEIGHT-IMAGE delta"
  echo "  (-5.154 GiB) is exact; the KV-token delta carries this caveat."
} | tee -a "$LOG"

setsid choom -n 1000 -- "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --served-model-name Qwen3.8-27B \
  --tp-size 1 --pp-size 3 \
  --pp-stage-ratio 32,18,14 \
  --pp-attn-stage-ratio 8,4,4 \
  --rank-gpu-id 0,1,2 \
  --rank-gpu-memory-mib "$RANK_MIB" \
  --skip-server-warmup \
  --enable-phase-flip \
  --phase-flip-tp-vector 32,16,16 \
  --phase-flip-policy "$FLIP_POLICY" \
  --phase-flip-purity strict:3 \
  --phase-flip-spill-depth arena \
  --disable-overlap-schedule \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 262144 \
  --max-running-requests 8 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --chat-template-default-kwargs '{"preserve_thinking": true}' \
  --enable-cache-report \
  --enable-metrics \
  --enable-hierarchical-cache \
  --hicache-ratio 1.5 \
  --hicache-write-policy write_through \
  --hicache-storage-backend file \
  --hicache-mem-layout layer_first \
  --hicache-io-backend direct \
  --host 127.0.0.1 --port 30030 \
  --chunked-prefill-size 4096 \
  --scheduler-distributed-teardown \
  --page-size 1 \
  --random-seed 785500001 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-slot-reorder \
  --uneven-dcp --uneven-dcp-weighted \
  --kv-backing-relief \
  --barlink --barlink-transport bar1 \
  --barlink-bar1-window-mib 24,PP_0=96,FLIP_TP_0=48,FLIP_DCP_0=32 \
  --barlink-bar1-cap-cycles 300000000000 \
  --seam-entry-margin-mib 512 \
  --seam-entry-delay-budget-s 2 \
  --flip-seam-chunk-mib 8 \
  --collective-census-interval 50 \
  --phase-policy-drain-mode \
  --phase-policy-drain-mode-strict \
  --phase-policy-min-dwell-s 3 \
  --phase-policy-tp-decode-floor-s 10 \
  --phase-policy-pp-window-s 15 \
  --phase-policy-decode-stall-slo-s 180 \
  --phase-policy-decode-contention 1.0 \
  --uneven-token-vector 29,19,16 \
  --uneven-token-vector-role seed \
  --phase-flip-canonical-kv-page \
  --phase-flip-writeback \
  --phase-flip-rebind-hicache \
  >> "$LOG" 2>&1 &

echo $! > /spinning/gpu-arb/boot855.launchpid

# DEAD-MAN ARMED BY THE LAUNCHER, not by the operator's memory (standing order
# 2026-08-30, after an arming was missed). One-shot: it EXITS when it fires, so
# an empty pgrep AFTER a verdict is correct behaviour, not a lost watcher.
setsid /spinning/gpu-arb/devtools/boot_deadman.sh "$LOG" 30030 \
  > "/spinning/gpu-arb/deadman_855_${TAG}.out" 2>&1 &
echo "  deadman armed: pid $!, verdict -> /spinning/gpu-arb/deadman_855_${TAG}.out"
sleep 1
echo "  deadman pgrep proof: $(pgrep -c -f "[b]oot_deadman.sh" 2>/dev/null || echo 0) process(es)"
ln -sfn "$LOG" /root/current_boot.log
echo "$LOG" > /spinning/gpu-arb/boot855.logpath
echo "launched, pid $(cat /spinning/gpu-arb/boot855.launchpid), log $LOG"
