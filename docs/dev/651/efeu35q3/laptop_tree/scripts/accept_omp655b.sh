#!/bin/bash
# #655: coding-agent acceptance, TRIMMED tool set. Run AS USER EFEU.
#
# Why trimmed. The full omp request to this endpoint measured 17029 tokens, and
# the breakdown says the prompt is not the problem:
#     system prompt   22421 chars  (~5.6k tokens)
#     11 tool schemas 40452 chars  (~10.1k tokens)
# The TOOL SCHEMAS are two thirds of it. Two independent limits sit below that
# number on this server: max_prefill_tokens is 16384, and -- the harder one --
# a ~10k prompt took GPU reset(6) on this part's MES firmware. So the tool set
# is cut to the four tools a file-edit-and-run loop actually needs, which is
# also all this acceptance exercises. --no-lsp and --no-skills trim the system
# prompt further.
#
# This is a real limitation of the machine, not a preference: on this iGPU the
# agent has to be kept small, and every tool added is prefill on every turn.
set -u
TS=$(date +%H%M%S)
OUT=/root/651-p2/results/accept_omp655b_${TS}.txt
exec > >(tee -a "$OUT") 2>&1

OMP_FLAGS="--no-lsp --no-skills --tools=read,write,edit,bash --auto-approve"

resets() { dmesg -T 2>/dev/null | grep -cE "GPU reset\("; }
R0=$(resets)
echo "=== oh-my-pi acceptance (trimmed) $(date -Is) ==="
echo "flags: $OMP_FLAGS"
echo "baseline GPU resets=$R0"
echo "service: $(curl -s -m 10 localhost:31651/ondemand/status)"

W=/home/efeu/omp655b-$TS
mkdir -p "$W"

# The planted bug: sum_list is handed a comma-separated STRING, but it assumes
# a list of numbers, so `total += v` raises TypeError. The fix is only
# reachable by running the file and reading the traceback.
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
echo "=========== PHASE 1: create and run ==========="
T0=$(date +%s)
su - efeu -c "
export PATH=\$HOME/.local/bin:\$PATH
cd '$W'
timeout 1800 omp --model local/qwen36-35b-a3b $OMP_FLAGS -p \
  'Create a file fib.py that prints the 20th Fibonacci number (the sequence starting 1, 1, 2, 3), then run it with python3 and tell me exactly what it printed.'
"
echo "--- phase 1 elapsed=$(( $(date +%s) - T0 ))s, resets=$(resets) ---"
echo "FILES: $(ls -A "$W")"
if [ -f "$W/fib.py" ]; then
  echo "P1-WROTE-FILE yes"
  echo "--- fib.py ---"; cat "$W/fib.py"
  ACTUAL=$(cd "$W" && timeout 60 python3 fib.py 2>&1 | tr -d '\n')
  echo "--- independent run: $ACTUAL"
  case "$ACTUAL" in *6765*) echo "P1-CORRECT-6765 yes";; *) echo "P1-CORRECT-6765 no"; fail=1;; esac
else
  echo "P1-WROTE-FILE no"; fail=1
fi

echo
echo "=========== PHASE 2: run, diagnose, fix ==========="
T2=$(date +%s)
su - efeu -c "
export PATH=\$HOME/.local/bin:\$PATH
cd '$W'
timeout 1800 omp --model local/qwen36-35b-a3b $OMP_FLAGS -p \
  'The file buggy.py in this directory crashes when run. Run it with python3, read the error, fix buggy.py so that it prints the sum of the five numbers, and run it again to confirm. Tell me the final output.'
"
echo "--- phase 2 elapsed=$(( $(date +%s) - T2 ))s, resets=$(resets) ---"
echo "--- buggy.py after the agent ---"; cat "$W/buggy.py"
if diff -q "$W/buggy.py" "$W/buggy.py.original" >/dev/null; then
  echo "P2-FILE-EDITED no"; fail=1
else
  echo "P2-FILE-EDITED yes"
fi
FIXED=$(cd "$W" && timeout 60 python3 buggy.py 2>&1 | tr -d '\n')
echo "--- independent run: $FIXED"
case "$FIXED" in *15*) echo "P2-CORRECT-15 yes";; *) echo "P2-CORRECT-15 no"; fail=1;; esac

echo
echo "=== desktop still alive? ==="
systemctl is-active gdm3 2>/dev/null || systemctl is-active display-manager
R1=$(resets)
echo "resets final=$R1 (baseline $R0)"
[ "$R1" != "$R0" ] && echo "NOTE: GPU reset occurred during this run"

echo
echo "workdir=$W"
echo "transcript=$OUT"
if [ "$fail" = "0" ]; then echo "OMP655B-ACCEPTANCE: PASS"; else echo "OMP655B-ACCEPTANCE: FAIL"; fi
exit "$fail"
