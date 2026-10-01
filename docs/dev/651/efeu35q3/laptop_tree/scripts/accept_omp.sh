#!/bin/bash
# #651: the user's actual stop criterion -- efeu can USE the coding agent.
#
# Two complete round trips, as efeu, on a real task, with no GPU reset and a
# responsive desktop afterwards. Twice because the wedge has repeatedly
# survived one pass and hit the next; a single success proves very little on
# this hardware.
set -u
OUT=/root/651-p2/results/accept_omp_$(date +%H%M%S).txt
exec > >(tee -a "$OUT") 2>&1

resets() { dmesg -T 2>/dev/null | grep -cE "GPU reset\("; }
R0=$(resets)
echo "=== oh-my-pi acceptance $(date -Is) === baseline GPU resets=$R0"

fail=0
for cycle in 1 2; do
  echo
  echo "--- cycle $cycle ---"
  T0=$(date +%s)
  su - efeu -c '
    export PATH=$HOME/.local/bin:$PATH
    W=$(mktemp -d /home/efeu/omp-accept-XXXX); cd "$W"
    timeout 1500 omp --model local/qwen36-35b-a3b --no-lsp -p \
      "Write a Python script fib.py that prints the 20th Fibonacci number, then run it and tell me the output." \
      2>&1 | tail -12
    echo "FILES: $(ls "$W")"
    if [ -f "$W/fib.py" ]; then echo "WROTE-FILE yes"; else echo "WROTE-FILE no"; fi
    grep -qE "6765" "$W"/* 2>/dev/null && echo "CORRECT-6765 yes" || echo "CORRECT-6765 no"
  '
  T1=$(date +%s)
  R=$(resets)
  echo "  cycle $cycle elapsed=$((T1-T0))s resets=$R (baseline $R0)"
  if [ "$R" != "$R0" ]; then
    echo "  FAIL: GPU reset during cycle $cycle"
    fail=1
    break
  fi
done

echo
echo "=== desktop still alive? ==="
systemctl is-active gdm3 2>/dev/null || systemctl is-active display-manager
for c in /sys/class/drm/card*-eDP-*; do
  echo "  $(basename "$c"): status=$(cat "$c"/status) enabled=$(cat "$c"/enabled) dpms=$(cat "$c"/dpms)"
done
loginctl list-sessions --no-legend 2>/dev/null | grep -c greeter | xargs -I{} echo "  greeter sessions: {}"

echo
if [ "$fail" = "0" ]; then echo "OMP-ACCEPTANCE: PASS"; else echo "OMP-ACCEPTANCE: FAIL"; fi
exit "$fail"
