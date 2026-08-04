#!/bin/bash
# Hermetic smoke for the spill-matrix harness. No GPU work, no model load.
#
# Why this exists: desk-written-never-executed code is unvalidated code. Every
# arm of the window runs its own execution smoke here first, and the most
# valuable one is the LAST: each recipe's argv is pushed through the REAL
# server-args validator, so a recipe that the code refuses is discovered at the
# desk instead of costing a boot inside the window.
set -u

WT=/spinning/wt-spill-matrix
VENV=/spinning/htsglang-gpu/.venv
HERE="$WT/scripts/dev/spill_matrix"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fails=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }

echo "== 1. cards.sh identity =="
if bash "$HERE/cards.sh" identity | grep -q 'NVIDIA'; then ok "identity lists cards"; else bad "identity"; fi

echo "== 2. cards.sh corridor + verdict (3 s) =="
bash "$HERE/cards.sh" corridor "$TMP/corr.tsv" 3 >/dev/null 2>&1
if [ -s "$TMP/corr.tsv" ]; then ok "corridor sampled $(wc -l < "$TMP/corr.tsv") rows"; else bad "corridor empty"; fi
bash "$HERE/cards.sh" verdict "$TMP/corr.tsv" >/dev/null 2>&1
rc=$?
if [ $rc -eq 0 ] || [ $rc -eq 1 ]; then ok "verdict ran (rc=$rc)"; else bad "verdict rc=$rc"; fi
# Can-fail arm: a synthetic sample below the floor MUST be called RED.
printf 'epoch\tidx\tfree_mib\tused_mib\n1\t0\t100\t900\n' > "$TMP/red.tsv"
if bash "$HERE/cards.sh" verdict "$TMP/red.tsv" 2>&1 | grep -q CORRIDOR-RED; then
    ok "can-fail: 100 MiB free is called CORRIDOR-RED"
else
    bad "can-fail: floor violation NOT detected"
fi

echo "== 3. drive.py signal regexes vs the real log strings =="
# Synthetic log carrying the EXACT strings from kv_session_offload.py. If a
# message is renamed upstream, this smoke goes red rather than the window
# silently reporting a missing signal as a feature failure.
cat > "$TMP/fake.log" <<'EOF'
[TP0] kv-session-offload (S4) armed: mode=deep S=3 prefix=[0] rank=0
[TP0] kv-session-offload SPILL(partial): rid=abc arrival_seq=7 L=1200 host_rows=64
[TP0] kv-session-offload (P1): wave-back THRESHOLD armed -- a host block
[TP0] kv-session-offload: draft-KV bundle armed (draft pool rows=32)
[TP0] kv-session-offload spec-in-tick: reserved 16 draft-read rows
[TP0] kv-session-offload spec-in-tick: rid=abc spill batch armed with 4 draft tokens
[TP0] kv-session-offload SPILL BUDGET (#236) armed: total=65536
[TP0] kv-session-offload BUDGET: DEMOTING rid=abc (budget 'total' exhausted)
[TP0] kv-session-offload: SELF-CALIBRATING spill-tick cadence armed (floor=8)
[TP0] kv-session-offload prefill-spill (born-spilled) ENABLED: a prompt
[TP0] kv-session-offload tick build: rid=abc has no output token yet
[TP0] closing slot 3: restored to device
EOF
for cell in H1 H2 H4 H5 H6 H8 H9 H11 H12 H14 H3; do
    if "$VENV/bin/python" "$HERE/drive.py" signals "$TMP/fake.log" "$cell" 2>&1 | grep -q 'ALL SIGNALS PRESENT'; then
        ok "signals $cell match"
    else
        bad "signals $cell DID NOT match the real log strings"
    fi
done
# Can-fail arm: an empty log must report the signal missing, never a pass.
: > "$TMP/empty.log"
if "$VENV/bin/python" "$HERE/drive.py" signals "$TMP/empty.log" H2 2>&1 | grep -q 'SIGNAL MISSING'; then
    ok "can-fail: empty log reports SIGNAL MISSING"
else
    bad "can-fail: empty log did not report a missing signal"
fi

echo "== 4. drive.py compare =="
printf '{"0":{"text":"alpha beta"},"1":{"text":"same"}}' > "$TMP/a.json"
printf '{"0":{"text":"alpha beta"},"1":{"text":"same"}}' > "$TMP/b.json"
printf '{"0":{"text":"alpha DELTA"},"1":{"text":"same"}}' > "$TMP/c.json"
if "$VENV/bin/python" "$HERE/drive.py" compare "$TMP/a.json" "$TMP/b.json" | grep -q 'identical=2 diverged=0'; then
    ok "compare: identical arms agree"
else
    bad "compare: identical arms not recognised"
fi
if "$VENV/bin/python" "$HERE/drive.py" compare "$TMP/a.json" "$TMP/c.json" | grep -q 'diverged=1'; then
    ok "can-fail: compare detects a divergence"
else
    bad "can-fail: compare missed a divergence"
fi

echo "== 5. drive.py ready is bounded on a dead port =="
t0=$(date +%s)
"$VENV/bin/python" "$HERE/drive.py" ready 39999 4 >/dev/null 2>&1
t1=$(date +%s)
if [ $((t1 - t0)) -le 12 ]; then ok "ready returned in $((t1 - t0))s (bounded)"; else bad "ready hung $((t1 - t0))s"; fi

echo "== 6. every recipe through the REAL server-args validator =="
# This is the point of the whole file: find out which recipes the code refuses
# BEFORE the window, and record the refusal text as a matrix result.
export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_UNEVEN_DCP=1 SGLANG_UNEVEN_DCP_WEIGHTED=1 SGLANG_MAMBA_SSM_DTYPE=bfloat16
for r in K0 K1 K2 K3 K4 L1 C1; do
    # Re-derive the argv exactly as boot.sh would, including its env exports.
    line=$(DRY=1 bash "$HERE/boot.sh" "$r" | tail -1)
    argv=${line#*-m sglang.launch_server }
    envs=""
    case "$r" in
        K2) envs="KVSO_ALLOW_SPEC=1 KVSO_RESUME=1" ;;
        K3) envs="KVSO_ALLOW_SPEC=1 KVSO_RESUME=1 SGLANG_KVSO_SPILL_GRAPH=1" ;;
    esac
    out=$(env $envs timeout 180 "$VENV/bin/python" -c '
import sys
from sglang.srt.server_args import prepare_server_args
try:
    prepare_server_args(sys.argv[1:])
    print("ACCEPTED")
except Exception as e:
    print("REFUSED: " + " ".join(str(e).split())[:400])
' $argv 2>&1 | grep -E '^(ACCEPTED|REFUSED)' | head -1)
    printf '  %-3s %s\n' "$r" "${out:-<no verdict: parser died, see below>}"
    [ -n "$out" ] || fails=$((fails + 1))
done

echo
if [ "$fails" -eq 0 ]; then echo "SMOKE GREEN"; else echo "SMOKE RED ($fails failures)"; fi
exit $((fails > 0))
