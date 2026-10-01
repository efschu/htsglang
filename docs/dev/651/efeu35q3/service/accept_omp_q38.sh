#!/bin/bash
# efeu-TP14 2026-10-01: coding-agent acceptance for Qwen3.8-35B-A3B on the
# on-demand service, FULL default omp tool surface (~17k-token system prompt),
# i.e. the real use case. Derived from #655 accept_omp655c.sh (which had to cut
# to 2 tools because ~10k prompts wedged the GPU). Run as root; omp runs as efeu.
#   phase 1: create fib.py, run it (expects 6765)
#   phase 2: run a broken file, read the traceback, fix it (expects 15)
# Each omp call is a fresh process with the same system prompt: phase 2 should
# hit the prefix cache (device or L3) instead of re-prefilling ~17k tokens.
set -u
TS=$(date +%H%M%S)
OUT=/root/efeu35q3/results/accept_omp_q38_${TS}.txt
exec > >(tee -a "$OUT") 2>&1
MODEL=${MODEL_ID:-local/qwen38-35b-a3b}
FLAGS="--no-lsp --no-skills --auto-approve"
resets() { dmesg 2>/dev/null | grep -cE "GPU reset\("; }
R0=$(resets)
echo "=== omp acceptance q38 (full tools) $(date -Is) model=$MODEL flags=$FLAGS resets=$R0 ==="
echo "service: $(curl -s -m 10 localhost:31651/ondemand/status)"
W=/home/efeu/omp-q38-$TS
mkdir -p "$W"
cat > "$W/buggy.py" <<'PYEOF'
def sum_list(values):
    total = 0
    for v in values:
        total += v
    return total


data = "1,2,3,4,5"
print("sum is", sum_list(data))
PYEOF
cp "$W/buggy.py" "$W/buggy.py.original"
chown -R efeu:efeu "$W"
fail=0
run_omp() {
  su - efeu -c "export PATH=\$HOME/.local/bin:\$PATH; cd '$W'; timeout 1800 omp --model $MODEL $FLAGS -p '$1'"
}
echo; echo "=========== PHASE 1 ==========="
T0=$(date +%s)
run_omp 'Write a file fib.py that prints the 20th Fibonacci number (sequence starts 1, 1, 2, 3), then run it with python3 and tell me what it printed.'
echo "--- phase 1 elapsed=$(( $(date +%s) - T0 ))s resets=$(resets) ---"
if [ -f "$W/fib.py" ]; then
  A=$(cd "$W" && timeout 60 python3 fib.py 2>&1 | tr -d '\n'); echo "independent run: $A"
  case "$A" in *6765*) echo "P1-CORRECT-6765 yes";; *) echo "P1-CORRECT-6765 no"; fail=1;; esac
else echo "P1-WROTE-FILE no"; fail=1; fi
echo; echo "=========== PHASE 2 ==========="
T2=$(date +%s)
run_omp 'Run buggy.py with python3. It crashes. Read the error, then write a fixed buggy.py that prints the sum of the five numbers, and run it again to confirm. Tell me the final output.'
echo "--- phase 2 elapsed=$(( $(date +%s) - T2 ))s resets=$(resets) ---"
diff -q "$W/buggy.py" "$W/buggy.py.original" >/dev/null && { echo "P2-FILE-EDITED no"; fail=1; } || echo "P2-FILE-EDITED yes"
F=$(cd "$W" && timeout 60 python3 buggy.py 2>&1 | tr -d '\n'); echo "independent run: $F"
case "$F" in *15*) echo "P2-CORRECT-15 yes";; *) echo "P2-CORRECT-15 no"; fail=1;; esac
R1=$(resets); echo "resets final=$R1 (baseline $R0)"
echo "desktop: $(systemctl is-active gdm3 2>/dev/null || systemctl is-active display-manager)"
BL=$(curl -s -m 10 localhost:31651/ondemand/status | python3 -c 'import json,sys;print(json.load(sys.stdin).get("backend_log") or "")')
[ -n "$BL" ] && { echo "--- prefix-cache evidence (backend log) ---"; grep -E "Prefill batch" "$BL" | awk '{for(i=1;i<=NF;i++) if($i=="#new-token:"||$i=="#cached-token:") printf "%s %s ", $i, $(i+1); print ""}' | tail -25; }
if [ "$fail" = "0" ]; then echo "OMP-Q38-ACCEPTANCE: PASS"; else echo "OMP-Q38-ACCEPTANCE: FAIL"; fi
exit "$fail"
