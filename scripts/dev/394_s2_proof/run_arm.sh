#!/usr/bin/env bash
# Run ONE arm of the #394/#439 proof window end to end.
#
#   [RUN=<dir>] [PORT=<port>] scripts/dev/394_s2_proof/run_arm.sh <arm> \
#       [max_tokens] [runs] [warmups]
#
# Boots the arm through boot_ab.sh, waits with a BOUNDED loop, records the
# VRAM/corridor facts, runs bench-length generations, tears the boot down
# verifying the cards come back to 0 MiB, and reads the #390 dump with
# read_arm.py TWICE: once before teardown as a liveness check (read_$ARM.txt,
# not quotable) and once after it on the final, work-matched revision
# (read_final_$ARM.txt), which is the one a result is quoted from.
#
# WHY THIS FILE IS IN THE REPO (#439). The 2026-08-02 ARM3 battery drove
# boot_ab.sh from an ad-hoc copy of this script that assigned ARM without
# EXPORTING it. boot_ab.sh reads ARM from the environment, so every arm booted
# the baseline, and the failure is invisible in the output: an A/B between two
# baselines reports a clean null. The export below is the fix, boot_ab.sh now
# refuses an unset ARM rather than defaulting to it, and
# test_expert_compute_placement_439.TestTheArmHarnessCarriesTheArm pins both
# through DRY_RUN=1 with a can-fail arm that drops the export.
set -uo pipefail

ARM="${1:?usage: run_arm.sh <equal|proportional|compute|compute-cal> [max_tokens] [runs] [warmups]}"
export ARM
MAXTOK="${2:-900}"
RUNS="${3:-3}"
WARMUPS="${4:-1}"

source /root/rig-env.sh 2>/dev/null || true
# EXPORTED, every one of them. boot_ab.sh runs as a child process and reads all
# of these from the environment; a plain assignment here reaches its own body
# and nothing else, which is the same mistake as the unexported ARM one line
# up. Defaulted-then-exported so a caller may override any of them.
export REPO_ROOT="${REPO_ROOT:?set REPO_ROOT or provide /root/rig-env.sh}"
export VENV="${VENV:?set VENV}"
export WT="${WT:-$REPO_ROOT}"
export PORT="${PORT:-30394}"
export RUN="${RUN:-/spinning/gpu-battery-results/$(date +%F)_394_s2}"
# The decode probe is carried with the battery results, not with the repo: it
# is the harness the published numbers were produced by, and re-implementing it
# per window is how two windows stop being comparable.
PROBE="${PROBE:-/spinning/gpu-battery-results/2026-08-02_394_linkshards/decode_probe.py}"

mkdir -p "$RUN"

echo "=== ARM $ARM: boot ==="
bash "$WT/scripts/dev/394_s2_proof/boot_ab.sh" || exit 1
PID=$(cat "$RUN/boot_$ARM.launchpid")

ok=0
i=0
for i in $(seq 1 80); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  kill -0 "$PID" 2>/dev/null || { echo "PROCESS GROUP GONE during boot"; break; }
  sleep 15
done
echo "ready=$ok after $((i * 15))s"

{
  echo "=== arm=$ARM ready=$ok $(date -u +%FT%TZ) ==="
  grep -ohE "max_total_num_tokens=[0-9]+" "$RUN/boot_$ARM.log" | tail -1
  grep -ohE "resolved --rank-tp-ratio[^)]*|--rank-tp-ratio auto: derived[^.]*\." \
    "$RUN/boot_$ARM.log" | tail -2
  grep -ohE "rank-moe-ratio[^.]*\.|moe_compute_policy[^,]*|link-proportional[^ ]*" \
    "$RUN/boot_$ARM.log" | tail -4
  # #439: the residency correction identifies itself per rank. Absent on the
  # baseline arms by construction; absent on a compute arm means the arm is
  # NOT VRAM-neutral and its corridor numbers are not arm 1's.
  grep -hE "resident fraction held at the base plan" "$RUN/boot_$ARM.log" | tail -3
  grep -ohiE "WARNING.*SGLANG_MOE_HOST_SHARD_RATIO.*" "$RUN/boot_$ARM.log" | tail -2
  echo "--- VRAM after boot ---"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader
} > "$RUN/facts_$ARM.txt" 2>&1
cat "$RUN/facts_$ARM.txt"

if [ "$ok" = "1" ]; then
  echo "=== ARM $ARM: bench-length generations (${MAXTOK} tok x ${RUNS}) ==="
  "$VENV/bin/python" "$PROBE" --url "http://127.0.0.1:$PORT" \
    --runs "$RUNS" --warmups "$WARMUPS" --max-tokens "$MAXTOK" \
    --label "$ARM" --out "$RUN/decode_$ARM.json" 2>&1 | tee "$RUN/decode_$ARM.log"
  # give the 45 s interval dump one more tick so the last generations land
  sleep 50
  # NOT the quotable readout -- see the post-teardown read below. This one is a
  # liveness/self-identification check on a live server; its counters are a
  # mid-run snapshot per rank and must never be divided by another arm's.
  echo "=== ARM $ARM: #390 readout (pre-teardown snapshot, NOT quotable) ==="
  PYTHONPATH="$WT/python" "$VENV/bin/python" \
    "$WT/scripts/dev/394_s2_proof/read_arm.py" "$RUN" "$ARM" 2>&1 \
    | tee "$RUN/read_$ARM.txt"
else
  echo "BOOT FAILED for $ARM"; tail -25 "$RUN/boot_$ARM.log"
fi

kill -TERM -"$PID" 2>/dev/null
sleep 25
kill -9 -"$PID" 2>/dev/null
sleep 10
echo "--- VRAM after teardown ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# THE QUOTABLE READOUT (rule of 2026-08-03, replaces "quote the pre-teardown
# numbers"). SIGTERM makes every rank write a FINAL dump revision at a common,
# well-defined endpoint, so the arms of a window are work-matched and their
# counters may be divided by each other. The pre-teardown revisions cannot:
# each rank writes on its own 45 s timer, so two arms get caught at different
# fractions of their runs -- 96.8 % against 91.9 % in the #439 green-corridor
# window, which inflated that window's transfer-term ratio by ~5 % (1.5028x
# reported against 1.4307x work-matched). See ARM3_COMPUTE.md, "Which revision
# to read". Compare the `work point of this arm:` lines of the two arms before
# quoting any ratio; read_arm.py warns when a revision is not final.
if [ "$ok" = "1" ]; then
  echo "=== ARM $ARM: #390 readout (FINAL, work-matched -- quote THIS) ==="
  PYTHONPATH="$WT/python" "$VENV/bin/python" \
    "$WT/scripts/dev/394_s2_proof/read_arm.py" "$RUN" "$ARM" 2>&1 \
    | tee "$RUN/read_final_$ARM.txt"
fi
