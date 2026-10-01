#!/bin/bash
# efeu-TP14: idle energy probe -- READ-ONLY, touches nothing.
# Over SECONDS (default 30): package energy from RAPL (intel-rapl:0, AMD RAPL via
# powercap; the APU package = CPU cores + iGPU + uncore), the amdgpu-reported
# socket power, CPU% per thread of the sglang processes (top -H, 1 sample per
# interval) and total system CPU%. Run with the model LOADED and idle, then
# PARKED, to get the cost of keeping it loaded.
#   idle_probe.sh [seconds] [label]
set -u
S=${1:-30}; L=${2:-probe}
R=/sys/class/powercap/intel-rapl:0
E0=$(cat $R/energy_uj); T0=$(date +%s.%N)
GPUW=0; N=0
cpu0=$(awk '/^cpu /{print $2+$3+$4+$6+$7+$8, $5}' /proc/stat)
TOPF=$(mktemp)
top -H -b -d $S -n 2 -w 200 > $TOPF &
TP=$!
for i in $(seq 1 $S); do
  w=$(cat /sys/class/hwmon/hwmon12/power1_average 2>/dev/null || cat /sys/class/hwmon/hwmon12/power1_input)
  GPUW=$((GPUW + w)); N=$((N + 1)); sleep 1
done
wait $TP
E1=$(cat $R/energy_uj); T1=$(date +%s.%N)
cpu1=$(awk '/^cpu /{print $2+$3+$4+$6+$7+$8, $5}' /proc/stat)
python3 - "$E0" "$E1" "$T0" "$T1" "$GPUW" "$N" "$cpu0" "$cpu1" "$L" "$TOPF" <<'PY'
import sys
e0, e1, t0, t1, gw, n = map(float, sys.argv[1:7])
b0, i0 = map(float, sys.argv[7].split()); b1, i1 = map(float, sys.argv[8].split())
label, topf = sys.argv[9], sys.argv[10]
dt = t1 - t0
pkg = (e1 - e0) / 1e6 / dt if e1 >= e0 else float("nan")
busy = (b1 - b0) / max(1.0, (b1 - b0) + (i1 - i0)) * 100
print(f"[{label}] {dt:.0f}s  package {pkg:.2f} W (RAPL)  amdgpu socket {gw / n / 1e6:.2f} W  system CPU busy {busy:.1f} %")
blocks = open(topf).read().split("top - ")
last = blocks[-1].splitlines()
rows = []
for ln in last:
    p = ln.split()
    if len(p) >= 12 and p[0].isdigit():
        try:
            cpu = float(p[8].replace(",", "."))
        except ValueError:
            continue
        if cpu >= 1.0:
            rows.append((cpu, p[0], p[1], " ".join(p[11:])))
rows.sort(reverse=True)
print(f"[{label}] threads >= 1 % CPU (top -H, {dt:.0f}s average): {len(rows)}")
for cpu, pid, user, cmd in rows[:15]:
    print(f"   {cpu:6.1f} %  tid {pid:>7} {user:8s} {cmd[:60]}")
PY
rm -f $TOPF
