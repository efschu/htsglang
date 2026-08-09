#!/usr/bin/env bash
# #631 Route A -- THE ONE UNMANNED ACCEPTANCE RUN.
#
# The acceptance bar for this feature was never "each property was seen at
# some point". It is that ALL of them hold SIMULTANEOUSLY, in a single
# unattended run, on one instance, with nobody steering it:
#
#   1. POLICY=auto drives flips in BOTH directions -- no client here ever
#      posts /phase_flip.
#   2. CUDA graphs are live (replay activity in the decode phase).
#   3. Speculative accept length comes off the WIRE (meta_info), not only
#      out of the scheduler log.
#   4. The KV pool size of each phase is recorded.
#   5. The VRAM corridor HOLDS: per-card time-series MINIMUM free >= 1024
#      MiB, sampled at 100 ms THROUGH the whole run, never a snapshot.
#   6. The long-context leg: ONE bs=1 session past the model's native
#      262144 ceiling, with the planted content at that depth verified.
#
# Order matters. The mixed load runs FIRST, so the corridor is sampled
# under the flip traffic that stresses it; the single long session runs
# AFTERWARDS and alone, because a bs=1 measurement with other traffic in
# flight is not a bs=1 measurement.
#
# Everything lands in one timestamped evidence directory. The corridor
# sampler is stopped with SIGINT (that is how it prints its report), and
# it is stopped BEFORE the collection step so its JSON is complete.
set -uo pipefail

WT="${WT:-/spinning/wt-631-routea}"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
PORT="${PORT:-30030}"
SERVING_LOG="${SERVING_LOG:-/spinning/serving-30030.boot.log}"
MIXED_SECONDS="${MIXED_SECONDS:-300}"
IDLE_SECONDS="${IDLE_SECONDS:-60}"
NEEDLE_TOKENS="${NEEDLE_TOKENS:-300000}"
MODEL_DIR="${MODEL_DIR:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8-yarn1.5}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/spinning/evidence-631/unmanned_acceptance_${STAMP}}"
mkdir -p "$OUT"

MAIN="$OUT/run.log"
exec > >(tee -a "$MAIN") 2>&1

say() { printf '\n=== [%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

say "UNMANNED ACCEPTANCE START  out=$OUT"
say "provenance"
git -C "$WT" rev-parse HEAD
git -C "$WT" status --porcelain | head -20
curl -s -m 5 "http://127.0.0.1:$PORT/get_server_info" -o "$OUT/server_info.json" \
  && "$PY" - "$OUT/server_info.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
keys = ("model_path", "context_length", "max_total_num_tokens",
        "max_running_requests", "speculative_algorithm", "phase_flip_policy",
        "enable_phase_flip", "max_mamba_cache_size")
flat = d if isinstance(d, dict) else {}
for k in keys:
    if k in flat:
        print(f"  {k} = {flat[k]}")
PYEOF

# The line the whole run is measured against. Started first, killed last.
say "corridor sampler up (100 ms, floor 1024)"
"$PY" "$WT/scripts/route_a_631_corridor.py" --interval-ms 100 \
      --out "$OUT/corridor.json" > "$OUT/corridor.log" 2>&1 &
CORRIDOR_PID=$!

# Mark the log position so collection reads only THIS run's lines.
LOG_START=$(wc -l < "$SERVING_LOG")
echo "log_start_line=$LOG_START" | tee "$OUT/log_start"

say "PHASE 1 -- mixed load under POLICY=auto (${MIXED_SECONDS}s + ${IDLE_SECONDS}s idle)"
"$PY" "$WT/scripts/route_a_631_policy_acceptance.py" \
      --port "$PORT" --seconds "$MIXED_SECONDS" --idle-seconds "$IDLE_SECONDS" \
      --out "$OUT/policy_acceptance.json" > "$OUT/policy_acceptance.log" 2>&1
echo "policy acceptance rc=$?"
tail -25 "$OUT/policy_acceptance.log"

say "PHASE 2 -- bs=1 long-context needle probe past 262144 (alone)"
"$PY" "$WT/scripts/route_a_631_yarn_needle_probe.py" \
      --port "$PORT" --model-dir "$MODEL_DIR" \
      --target-tokens "$NEEDLE_TOKENS" --out "$OUT/needle.json" \
      > "$OUT/needle.log" 2>&1
echo "needle probe rc=$?"
tail -20 "$OUT/needle.log"

say "corridor sampler down (SIGINT prints the report)"
kill -INT "$CORRIDOR_PID" 2>/dev/null
wait "$CORRIDOR_PID" 2>/dev/null
tail -20 "$OUT/corridor.log"

say "COLLECT -- evidence slices from this run's log window only"
sed -n "$((LOG_START+1)),\$p" "$SERVING_LOG" > "$OUT/serving_window.log"
{
  echo "--- phase transitions (policy-driven; no client posted /phase_flip)"
  grep -aiE "phase flip (commit|committed)|flip .*(pp->tp|tp->pp)|phase_flip.*commit" \
       "$OUT/serving_window.log" | tail -60
  echo
  echo "--- counts"
  printf 'flip commits      : %s\n' "$(grep -acEi 'phase flip commit|flip commit' "$OUT/serving_window.log")"
  printf 'accept-len lines  : %s\n' "$(grep -acEi 'accept.?len' "$OUT/serving_window.log")"
  printf 'cuda graph lines  : %s\n' "$(grep -acEi 'cuda graph|graph capture|capture_bs|replay' "$OUT/serving_window.log")"
  echo
  echo "--- KV pool per phase"
  grep -aoE "max_total_num_tokens[= ]+[0-9]+" "$OUT/serving_window.log" | sort | uniq -c | tail -20
  echo
  echo "--- CUDA graph activity"
  grep -aiE "cuda graph|graph capture" "$OUT/serving_window.log" | tail -20
} > "$OUT/evidence_slices.txt" 2>&1
tail -40 "$OUT/evidence_slices.txt"

say "UNMANNED ACCEPTANCE END  out=$OUT"
