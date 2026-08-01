#!/usr/bin/env bash
# #404 window 2 -- ONE boot: the residue bracket on the CAPTURED verify graph,
# the mixed-rung job, and the per-round pool checksum.
#
# WHAT THE PREVIOUS WINDOW LEFT
#   2026-08-01_404_bracket_window returned "B clean and C clean": 18 arms, 738
#   rejected candidate rows, a flat dose response -- residue VOLUME does not
#   reach the pool-axis leak that #399 proved by elimination. It named its own
#   two gaps exactly, and this boot closes both.
#
#     GAP 1  The recipe pinned --dual-group-lane-spec-steps 1, so the boot
#            captured exactly ONE verify shape (2 tokens). The K=3 arms -- the
#            only ones with 3 rejected rows per round -- ran EAGER
#            (verify_graph_rounds 0). Retained static capture buffers exist
#            only on the captured path and that is where the whole _hidden
#            defect family lives, so the highest-residue arm and the surface
#            most likely to leak had never met.
#            FIX: --dual-group-lane-spec-rungs 0,1,3 records verify shapes 2
#            AND 4 (verify_rungs = tuple(k+1 for k in rungs if k>=1),
#            dual_group_lane.py). Both K=1 and K=3 arms then REPLAY, and every
#            arm asserts it (--strict) instead of reporting it.
#
#     GAP 2  No measured run has ever reached the READ side of the _kv_len
#            advance: a verify round taking n_cached right after a K=0 round.
#            A scalar rung pin is constant for the whole job, so no recipe
#            could produce a mixed-rung one.
#            FIX: spec_steps as a cycled SCHEDULE ([0,1] / [0,3]), plus the
#            adaptive arm the previous window proposed. The schedule carries
#            the verdict because it is the same on every boot; the adaptive arm
#            says whether the policy finds the shape on its own.
#
#   NEW INSTRUMENT  SGLANG_LANE_POOL_CHECKSUM=1 makes every arm carry per-round
#   digests of the committed slot mapping, the committed KV rows and the
#   committed conv/ssm state. Read two ways (append-only within the arm, and
#   against the arm's own no-spec reference), a divergence lands on a SURFACE
#   and a ROUND instead of on a token index.
#
# COST OF THE PROBE, so it is not discovered in the timings: one D2H of the
#   committed prefix per round per full-attention layer. Round times are not
#   affected (the probe runs after the round's wall clock is taken) but the
#   job's total duration is, and it grows with the position. This boot is a
#   CORRECTNESS window; do not read a throughput number off it.
#
# VRAM  the second verify graph is a NEW post on the lane's card. The previous
#   window peaked at 29347/32607 MiB on the 5090 (3260 MiB free), so there is
#   headroom -- but check the "lane budget ... MiB =" and "verify graph
#   captured" lines before the driver runs, and abort if the corridor
#   (free >= 400 MiB) is not respected.
#
# ABORT  server not up in 20 min -> abort, keep the log, count the boot.
#   A divergent arm is NOT an abort. That is the measurement. An UNMET
#   EXPECTATION (an arm that ran eager, a mixed arm that was single-rung) IS a
#   broken vehicle: --strict makes the driver exit non-zero on it.

set -uo pipefail
WT="${WT:-/spinning/wt-404b-checksum}"
OWNER="${OWNER:-agent-404b-checksum}"
export WT OWNER
source "$WT/scripts/dual_group/r7c/common.sh"

TARGET_DIR="$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF"
TARGET="$TARGET_DIR/Qwen3.6-27B-Q3_K_M.gguf"
PORT="${PORT:-30085}"
OUT="${OUT:-/spinning/gpu-battery-results/2026-08-02_404_steps3_checksum}"
LOG="${LOG:-/tmp/404b-checksum.server.log}"
PIDFILE=/tmp/404b-checksum.pid
LANE_BUDGET="${LANE_BUDGET:-700}"
RANK_MIB="${RANK_MIB:-21000,17780,17780}"
STEPS="${STEPS:-3}"
RUNGS="${RUNGS:-0,1,3}"
TOKENS="${TOKENS:-64}"
BRACKET="$WT/scripts/dual_group/r404/bracket_arms.py"

mkdir -p "$OUT"

assert_cards_free || exit 1
load_card_order "$OUT/cards.txt" || exit 1
claim_cards "#404 window 2: steps=3 captured + checksum"
trap 'stop_vram_sampler; kill "$(cat $PIDFILE 2>/dev/null)" 2>/dev/null; release_cards "#404b abgebrochen"; exit 1' INT TERM
start_vram_sampler "$OUT/vram.csv"
export SGLANG_ACCEPT_POSITION_PROBE=1
export SGLANG_LANE_MARGIN_PROBE=1
# The #404 probe. The PATH value is a PREFIX: each lane/rank appends
# .lane<L>.rank<R>.jsonl, because under TP every rank runs this code with the
# same environment and one shared file would interleave rounds from processes
# that are not at the same round.
export SGLANG_LANE_POOL_CHECKSUM=1
export SGLANG_LANE_POOL_CHECKSUM_PATH="$OUT/pool_checksum"
# Per-position digests: narrows a KV difference from "the prefix" to "this
# token". Costs jsonl size, not device traffic (the host copies are already
# made), so it rides along.
export SGLANG_LANE_POOL_CHECKSUM_PER_POS=1

( while true; do touch "$ARB/holder" 2>/dev/null; sleep 300; done ) &
HB_PID=$!

cd "$WT" || exit 1
launch_server "$LOG" "$PIDFILE" \
  "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$TARGET" \
  --tokenizer-path "$TARGET_DIR" \
  --tp-size 3 --rank-gpu-id 0,1,2 \
  --rank-tp-ratio 2,1,1 --rank-mlp-ratio 6,1,1 --rank-vocab-ratio 6,1,1 \
  --rank-gpu-memory-mib "$RANK_MIB" \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e4m3 --context-length 16384 --trust-remote-code \
  --max-running-requests 4 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --dual-group-lane --dual-group-lane-budget-mib "$LANE_BUDGET" \
  --dual-group-lane-concurrent --dual-group-lane-admission-ms 2.0 \
  --dual-group-lane-spec --dual-group-lane-spec-steps "$STEPS" \
  --dual-group-lane-spec-rungs "$RUNGS" --dual-group-lane-spec-adaptive \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT"

if dry_run; then
  kill $HB_PID 2>/dev/null
  stop_vram_sampler
  release_cards "#404b dry run"
  echo "DRY RUN ok: #404 window 2"
  exit 0
fi

if ! wait_for_server "$PORT" 1200; then
  echo "--- letzte 40 Zeilen Serverlog (nur bei Abbruch) ---" >&2
  tail -40 "$LOG" >&2
  kill $HB_PID 2>/dev/null
  stop_vram_sampler
  release_cards "#404b: server nicht oben, verbraucht"
  exit 1
fi

grep -E "lane budget [0-9]+ MiB =|verify graph captured|NEXTN head graph captured|HEAD pool sizing|lane 0 ready" \
  "$LOG" | tee "$OUT/contract_lines.txt"

# THE ACCEPTANCE GATE FOR GAP 1, checked before a single arm runs. Two verify
# shapes must have been recorded; one of them is the K=3 arm's. Without this
# the window would repeat the previous one's mistake -- reading the K=3 rows
# as captured when they were eager.
if [ "$(grep -c 'verify graph captured (bs 1, 4 tokens' "$LOG")" -lt 1 ]; then
  echo "ABORT: the K=3 verify shape (4 tokens) was not captured; the recipe" >&2
  echo "       did not close gap 1. Check --dual-group-lane-spec-rungs." >&2
  kill $HB_PID 2>/dev/null
  kill "$(cat $PIDFILE)" 2>/dev/null
  stop_vram_sampler
  release_cards "#404b: K=3 verify shape not captured, verbraucht"
  exit 1
fi

run_bracket() {  # $1 = tag, rest = extra flags
  local tag="$1"; shift
  echo "== run $tag ==" | tee -a "$OUT/driver.txt"
  "$VENV/bin/python" "$BRACKET" \
    --port "$PORT" --tokenizer "$TARGET_DIR" \
    --tokens "$TOKENS" --steps "$STEPS" --strict \
    --out "$OUT/$tag.json" "$@" 2>&1 | tee -a "$OUT/driver.txt"
  return "${PIPESTATUS[0]}"
}

RC=0
# run1  the residue ladder ON THE CAPTURED GRAPH -- gap 1, the headline.
run_bracket run1_k3_captured --prompts squares --prelude none \
  --arms k3_plain,k3_tv0 || RC=$?
# run2  the mixed-rung job -- gap 2, the READ side of the _kv_len advance.
run_bracket run2_mixed_rung --prompts squares --prelude none \
  --arms mixed_0_1,mixed_0_3,adaptive,adaptive_tv0 || RC=$?
# run3  the K=1 bracket again, captured, as the direct tie to the previous
#       window's numbers (plain 1.145/55 there; a divergence here would be
#       comparable to it without a second boot).
run_bracket run3_k1_control --prompts squares --prelude none || RC=$?
# run4  other content families at the top rung.
run_bracket run4_prompts --prompts alphabet,repeat --prelude none \
  --arms k3_plain,k3_tv0,mixed_0_3 || RC=$?

curl -sf -m 30 "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json"

kill $HB_PID 2>/dev/null
"$VENV/bin/py-spy" dump --pid "$(cat $PIDFILE)" > "$OUT/pyspy_before_kill.txt" 2>&1
kill "$(cat $PIDFILE)" 2>/dev/null
sleep 20
stop_vram_sampler
summarise_vram "$OUT/vram.csv" | tee "$OUT/vram_summary.txt"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$OUT/final_vram.txt"
release_cards "#404 window 2 DONE (rc=$RC), Rohdaten in $OUT"
echo "fertig: $OUT (rc=$RC)"
exit "$RC"
