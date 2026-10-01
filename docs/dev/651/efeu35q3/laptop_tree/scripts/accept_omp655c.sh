#!/bin/bash
# #655: coding-agent acceptance, MINIMAL configuration. Run AS USER EFEU.
#
# Sizing, measured rather than guessed (results/omp_promptsize_*.txt, taken
# offline against a capture server so no GPU time was spent iterating):
#     default (11 tools)   17423 tokens
#     4 tools               6593 tokens
#     2 tools (write,bash)  3530 tokens   <- this configuration
# The tool SCHEMAS dominate: 11 tools cost 11706 tokens of schema on every
# single turn. Cutting to the two tools a file-edit-and-run loop actually needs
# takes the whole request to 3530.
#
# Why that matters here specifically: this iGPU's amdgpu MES fault wedges the
# GPU as prompts grow, and a ~10k prompt wedged it repeatedly today while ~1k
# prompts served. 3530 is chosen to sit near the low end that has served.
#
# write + bash is a complete loop: write creates and rewrites files, bash runs
# them and reads them back (cat). No `read` tool is needed for that.
set -u
TS=$(date +%H%M%S)
OUT=/root/651-p2/results/accept_omp655c_${TS}.txt
exec > >(tee -a "$OUT") 2>&1

FLAGS="--no-lsp --no-skills --tools=write,bash --auto-approve"

resets() { dmesg -T 2>/dev/null | grep -cE "GPU reset\("; }
R0=$(resets)
echo "=== oh-my-pi acceptance (minimal) $(date -Is) ==="
echo "flags: $FLAGS"
echo "baseline GPU resets=$R0"
echo "service: $(curl -s -m 10 localhost:31651/ondemand/status)"

W=/home/efeu/omp655c-$TS
mkdir -p "$W"

# Deliberate bug: sum_list is handed a comma-separated STRING but assumes a
# list of numbers, so `total += v` raises TypeError. The fix is only reachable
# by running the file and reading the traceback.
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

echo
echo "=========== PHASE 1: create a program and run it ==========="
T0=$(date +%s)
su - efeu -c "
export PATH=\$HOME/.local/bin:\$PATH
cd '$W'
timeout 1500 omp --model local/qwen36-35b-a3b $FLAGS -p \
  'Write a file fib.py that prints the 20th Fibonacci number (sequence starts 1, 1, 2, 3), then run it with python3 and tell me what it printed.'
"
echo "--- phase 1 elapsed=$(( $(date +%s) - T0 ))s resets=$(resets) ---"
echo "FILES: $(ls -A "$W")"
if [ -f "$W/fib.py" ]; then
  echo "P1-WROTE-FILE yes"
  echo "--- fib.py as written by the agent ---"; cat "$W/fib.py"
  A=$(cd "$W" && timeout 60 python3 fib.py 2>&1 | tr -d '\n')
  echo "--- independent run: $A"
  case "$A" in *6765*) echo "P1-CORRECT-6765 yes";; *) echo "P1-CORRECT-6765 no"; fail=1;; esac
else
  echo "P1-WROTE-FILE no"; fail=1
fi

echo
echo "=========== PHASE 2: run a broken file, diagnose, fix ==========="
T2=$(date +%s)
su - efeu -c "
export PATH=\$HOME/.local/bin:\$PATH
cd '$W'
timeout 1500 omp --model local/qwen36-35b-a3b $FLAGS -p \
  'Run buggy.py with python3. It crashes. Read the error, then write a fixed buggy.py that prints the sum of the five numbers, and run it again to confirm. Tell me the final output.'
"
echo "--- phase 2 elapsed=$(( $(date +%s) - T2 ))s resets=$(resets) ---"
echo "--- buggy.py after the agent ---"; cat "$W/buggy.py"
if diff -q "$W/buggy.py" "$W/buggy.py.original" >/dev/null; then
  echo "P2-FILE-EDITED no"; fail=1
else
  echo "P2-FILE-EDITED yes"
fi
F=$(cd "$W" && timeout 60 python3 buggy.py 2>&1 | tr -d '\n')
echo "--- independent run: $F"
case "$F" in *15*) echo "P2-CORRECT-15 yes";; *) echo "P2-CORRECT-15 no"; fail=1;; esac

echo
echo "=== desktop still alive? ==="
systemctl is-active gdm3 2>/dev/null || systemctl is-active display-manager
R1=$(resets)
echo "resets final=$R1 (baseline $R0)"
[ "$R1" != "$R0" ] && echo "NOTE: GPU reset during this run"

echo
echo "workdir=$W"
echo "transcript=$OUT"
if [ "$fail" = "0" ]; then echo "OMP655C-ACCEPTANCE: PASS"; else echo "OMP655C-ACCEPTANCE: FAIL"; fi
exit "$fail"
