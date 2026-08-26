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

# #910: REFUSE AT THE TOP, not 300 s of mixed load later. The default above is
# the Qwen3.6 checkpoint the #631 Route A acceptance ran on, and it is no
# longer on this box. Every property this script measures -- the derived KV
# pool per phase, the corridor floor, the 262144 native ceiling the long-context
# leg goes past -- is a number FOR THAT CHECKPOINT, so the default is not
# repointed at a surviving build: a run against another model would produce a
# full evidence directory whose numbers mean something else. Naming another
# checkpoint explicitly via MODEL_DIR is the operator's call and is still
# allowed; silently accepting an absent one is not.
if [ ! -d "$MODEL_DIR" ]; then
  echo "REFUSING: MODEL_DIR does not exist: $MODEL_DIR" >&2
  echo "  This is the #631 Route A acceptance checkpoint. It is gone from this" >&2
  echo "  box, and the acceptance numbers (per-phase KV pool, 1024 MiB corridor" >&2
  echo "  floor, the 262144 native ceiling the bs=1 leg exceeds) are numbers for" >&2
  echo "  THAT checkpoint -- an evidence directory produced against a different" >&2
  echo "  model is not this acceptance run." >&2
  echo "  Pass MODEL_DIR=<path> explicitly to accept a different specimen." >&2
  exit 2
fi

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

# ORDER IS THE POINT: this runs AFTER the long session, because concurrent
# /generate minutes after a >262k session is the exact recipe that killed
# all three ranks in the live-slot enumeration. It is therefore both the
# wire-accept-len measurement and that defect's standing reproduction.
say "PHASE 3 -- wire accept-len in TP, and the post-long-session flip"
"$PY" "$WT/scripts/route_a_631_wire_accept_probe.py" \
      --port "$PORT" --out "$OUT/wire_accept.json" \
      > "$OUT/wire_accept.log" 2>&1
echo "wire probe rc=$?"
tail -12 "$OUT/wire_accept.log"

say "corridor sampler down (SIGINT prints the report)"
kill -INT "$CORRIDOR_PID" 2>/dev/null
wait "$CORRIDOR_PID" 2>/dev/null
tail -20 "$OUT/corridor.log"

say "COLLECT -- evidence slices from this run's log window only"
sed -n "$((LOG_START+1)),\$p" "$SERVING_LOG" > "$OUT/serving_window.log"
{
  # The wording is the SERVER'S, not a guess. A first version of this
  # collector grepped for "phase flip commit" and reported 0 commits for a
  # run that had made 54 -- the log says "PHASE-FLIP DONE <phase>". A
  # collector that under-reports is worse than none: it turns a passing run
  # into an apparent failure.
  echo "--- flip commits, BOTH directions (policy-driven; no client posted /phase_flip)"
  printf 'PHASE-FLIP DONE pp (returned to PP) : %s\n' \
    "$(grep -acE 'PHASE-FLIP DONE pp' "$OUT/serving_window.log")"
  printf 'PHASE-FLIP DONE tp (entered TP)     : %s\n' \
    "$(grep -acE 'PHASE-FLIP DONE tp' "$OUT/serving_window.log")"
  printf 'flips armed by the policy           : %s\n' \
    "$(grep -acE 'phase flip armed' "$OUT/serving_window.log")"
  printf 'flips ABANDONED (any reason)        : %s\n' \
    "$(grep -acE 'FLIP ABANDONED' "$OUT/serving_window.log")"
  printf 'abandoned for STAGING room          : %s\n' \
    "$(grep -acE 'staging [0-9]+ MiB needed' "$OUT/serving_window.log")"
  echo
  echo "--- speculation on the wire and CUDA graphs"
  printf 'accept-len lines                    : %s\n' \
    "$(grep -acE 'accept len:' "$OUT/serving_window.log")"
  printf 'decode passes WITH a cuda graph     : %s\n' \
    "$(grep -acE 'cuda graph: True' "$OUT/serving_window.log")"
  grep -aoE 'accept len: [0-9.]+, accept rate: [0-9.]+' "$OUT/serving_window.log" \
    | tail -5
  echo
  echo "--- KV pool per phase (the flip's own census, at arm)"
  grep -aoE 'POOL CENSUS at-arm [a-z_]+: size=[0-9]+ free=[0-9]+' \
    "$OUT/serving_window.log" | sort | uniq -c | tail -10
  echo
  echo "--- the long single session (bs=1, past the native ceiling)"
  grep -aoE '#full token: [0-9]{6,}, full token usage: [0-9.]+' \
    "$OUT/serving_window.log" | tail -5
  echo
  echo "--- staging refusals in full (the guard doing its job)"
  grep -aE 'staging [0-9]+ MiB needed' "$OUT/serving_window.log" | tail -3
  echo
  echo "--- accept-len ON THE WIRE (meta_info, native /generate)"
  grep -aE 'VERDICT healthy_after|wire_accept' "$OUT/wire_accept.log" | tail -3
  echo
  echo "--- live-slot enumeration after the long session (the fixed defect)"
  printf 'requests skipped for having no req_pool_idx yet : %s\n' \
    "$(grep -acE 'live-slot enumeration skipped' "$OUT/serving_window.log")"
  printf 'live-slot dimension errors (must be 0)          : %s\n' \
    "$(grep -acE 'must have same number of dimensions' "$OUT/serving_window.log")"
  printf 'server health at the end of the run             : %s\n' \
    "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")"
} > "$OUT/evidence_slices.txt" 2>&1
tail -40 "$OUT/evidence_slices.txt"

say "UNMANNED ACCEPTANCE END  out=$OUT"
