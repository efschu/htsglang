#!/bin/bash
# #655: end-to-end acceptance for the coding agent, run AS USER EFEU.
#
# The bar is a genuine tool loop against the local model -- file edit plus
# shell execution -- not a chat reply that happens to contain code. So there
# are two phases and the second one is the real test:
#
#   phase 1  create a program from nothing, run it, report its output
#   phase 2  a file with a DELIBERATE bug is planted; the agent has to run it,
#            read the failure, fix the file and re-run
#
# Phase 2 is what separates a tool loop from a code completion: the fix is only
# reachable by executing the broken file and reading what came back.
#
# Detached by design -- decode here is ~8.6 tok/s with CPU expert offload, so a
# multi-turn agent session runs in minutes, not seconds.
set -u
TS=$(date +%H%M%S)
OUT=/root/651-p2/results/accept_omp655_${TS}.txt
exec > >(tee -a "$OUT") 2>&1

resets() { dmesg -T 2>/dev/null | grep -cE "GPU reset\("; }
R0=$(resets)
echo "=== oh-my-pi end-to-end acceptance $(date -Is) ==="
echo "baseline GPU resets=$R0"
echo "service status: $(curl -s -m 10 localhost:31651/ondemand/status)"

W=/home/efeu/omp655-$TS
mkdir -p "$W"
chown efeu:efeu "$W"

# The planted bug: sum_list is called with a string, and the function assumes a
# list of numbers. Running it raises TypeError; the traceback names the line.
cat > "$W/buggy.py" <<'PYEOF'
def sum_list(values):
    total = 0
    for v in values:
        total += v
    return total


# BUG: the caller passes a comma-separated string, not a list of numbers.
data = "1,2,3,4,5"
print("sum is", sum_list(data))
PYEOF
chown efeu:efeu "$W/buggy.py"
cp "$W/buggy.py" "$W/buggy.py.original"

fail=0

echo
echo "=========== PHASE 1: create and run ==========="
T0=$(date +%s)
su - efeu -c "
export PATH=\$HOME/.local/bin:\$PATH
cd '$W'
timeout 2400 omp --model local/qwen36-35b-a3b --no-lsp --auto-approve -p \
  'Create a file fib.py that prints the 20th Fibonacci number (starting 1, 1, 2, 3), then run it with python3 and tell me exactly what it printed.'
"
T1=$(date +%s)
echo "--- phase 1 elapsed=$((T1-T0))s ---"
echo "FILES: $(ls -A "$W")"
[ -f "$W/fib.py" ] && echo "P1-WROTE-FILE yes" || { echo "P1-WROTE-FILE no"; fail=1; }
if [ -f "$W/fib.py" ]; then
  echo "--- fib.py ---"; cat "$W/fib.py"
  ACTUAL=$(cd "$W" && timeout 60 python3 fib.py 2>&1 | tr -d '\n')
  echo "--- independent run of fib.py: $ACTUAL"
  echo "$ACTUAL" | grep -q 6765 && echo "P1-CORRECT-6765 yes" || { echo "P1-CORRECT-6765 no"; fail=1; }
fi
echo "resets after phase 1: $(resets) (baseline $R0)"

echo
echo "=========== PHASE 2: run, diagnose, fix ==========="
T2=$(date +%s)
su - efeu -c "
export PATH=\$HOME/.local/bin:\$PATH
cd '$W'
timeout 2400 omp --model local/qwen36-35b-a3b --no-lsp --auto-approve -p \
  'The file buggy.py in this directory crashes. Run it with python3, read the error, fix buggy.py so it prints the sum of the five numbers, and run it again to confirm. Tell me what the final output was.'
"
T3=$(date +%s)
echo "--- phase 2 elapsed=$((T3-T2))s ---"
echo "--- buggy.py after the agent ---"; cat "$W/buggy.py"
if diff -q "$W/buggy.py" "$W/buggy.py.original" >/dev/null; then
  echo "P2-FILE-EDITED no"; fail=1
else
  echo "P2-FILE-EDITED yes"
fi
FIXED=$(cd "$W" && timeout 60 python3 buggy.py 2>&1 | tr -d '\n')
echo "--- independent run of buggy.py: $FIXED"
echo "$FIXED" | grep -q 15 && echo "P2-CORRECT-15 yes" || { echo "P2-CORRECT-15 no"; fail=1; }

echo
echo "=== desktop still alive? ==="
systemctl is-active gdm3 2>/dev/null || systemctl is-active display-manager
R1=$(resets)
echo "resets final=$R1 (baseline $R0)"
[ "$R1" != "$R0" ] && { echo "NOTE: GPU reset occurred during the run"; fail=1; }

echo
echo "workdir=$W"
if [ "$fail" = "0" ]; then echo "OMP655-ACCEPTANCE: PASS"; else echo "OMP655-ACCEPTANCE: FAIL"; fi
exit "$fail"
