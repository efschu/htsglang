#!/bin/bash
# #651: acceptance for the minimal coding agent -- the user's actual criterion.
# Two complete round trips as efeu on a real task. A cycle passes only if the
# file was written AND running it gives the right answer AND no GPU reset
# occurred; checking resets alone once reported PASS for a run that produced
# nothing, which is exactly the false green this has to avoid.
set -u
OUT=/root/651-p2/results/accept_efeucode_$(date +%H%M%S).txt
exec > >(tee -a "$OUT") 2>&1
resets() { dmesg -T 2>/dev/null | grep -cE "GPU reset\("; }
R0=$(resets)
echo "=== efeu-code acceptance $(date -Is) === baseline GPU resets=$R0"
fail=0
for cycle in 1 2; do
  echo; echo "--- cycle $cycle ---"
  T0=$(date +%s)
  RES=$(su - efeu -c '
    export PATH=$HOME/.local/bin:$PATH
    W=$(mktemp -d /home/efeu/code-accept-XXXX); cd "$W"
    timeout 1700 efeu-code --cwd "$W" "Write a Python script fib.py that prints the 20th Fibonacci number, then run it." 2>&1 | tail -18
    echo "FILES: $(ls "$W")"
    if [ -f "$W/fib.py" ]; then
      O=$(cd "$W" && timeout 60 python3 fib.py 2>&1)
      echo "SCRIPT-OUTPUT: $O"
      case "$O" in *6765*) echo "VERDICT ok";; *) echo "VERDICT wrong";; esac
    else
      echo "VERDICT nofile"
    fi
  ')
  echo "$RES"
  T1=$(date +%s); R=$(resets)
  echo "  cycle $cycle elapsed=$((T1-T0))s resets=$R (baseline $R0)"
  case "$RES" in *"VERDICT ok"*) ;; *) echo "  FAIL: no correct result in cycle $cycle"; fail=1;; esac
  [ "$R" != "$R0" ] && { echo "  FAIL: GPU reset in cycle $cycle"; fail=1; }
  [ "$fail" = "1" ] && break
done
echo; echo "=== desktop ==="
systemctl is-active gdm3 2>/dev/null || systemctl is-active display-manager
for c in /sys/class/drm/card*-eDP-*; do
  echo "  $(basename "$c"): enabled=$(cat "$c"/enabled) dpms=$(cat "$c"/dpms)"
done
echo
[ "$fail" = "0" ] && echo "EFEUCODE-ACCEPTANCE: PASS" || echo "EFEUCODE-ACCEPTANCE: FAIL"
exit "$fail"
