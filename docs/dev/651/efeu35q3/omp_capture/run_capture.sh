#!/bin/bash
# efeu-TP14: capture omp's system prompt for two cwds and two moments, against a
# local capture endpoint (never the user's model service), with a scratch HOME
# that is a copy of efeu's omp config with baseUrl pointed at the capture port.
set -u
C=/tmp/efeu_omp_capture
mkdir -p $C && cp /root/efeu35q3/omp_capture/capture_server.py $C/
rm -rf $C/home $C/out; mkdir -p $C/home $C/out; chmod 755 $C
cp -a /home/efeu/.omp $C/home/.omp 2>/dev/null
rm -rf $C/home/.omp/agent/sessions
sed -i 's#http://127.0.0.1:31651/v1#http://127.0.0.1:31699/v1#' $C/home/.omp/agent/models.yml
chown -R efeu:efeu $C/home $C/out
su efeu -c "python3 $C/capture_server.py 31699 $C/out" &
SP=$!
sleep 1
run() {  # $1 cwd  $2 label
  su efeu -c "export HOME=$C/home PATH=/home/efeu/.local/bin:\$PATH; cd $1; timeout 120 omp --model local/qwen38-35b-a3b -p 'cpu last aktuell?'" > $C/out/omp_$2.txt 2>&1
  ls -S $C/out/req_*.json | head -1 | xargs -I{} cp {} $C/out/sys_$2.json; mkdir -p $C/out/$2; mv $C/out/req_*.json $C/out/$2/
}
run /tmp tmp_a
sleep 61
run /tmp tmp_b
run /home/efeu home
kill $SP; pkill -f "^python3 /tmp/efeu_omp_capture/capture_server.py"
python3 - $C/out <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
def sysmsg(f):
    r = json.load(open(d / f))
    m = r["messages"]
    s = m[0]["content"] if isinstance(m[0]["content"], str) else "".join(x.get("text", "") for x in m[0]["content"])
    return s, r
a, ra = sysmsg("sys_tmp_a.json"); b, _ = sysmsg("sys_tmp_b.json"); h, _ = sysmsg("sys_home.json")
def firstdiff(x, y):
    i = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    return i
print("system prompt chars:", len(a), "| tools:", len(ra.get("tools") or []), "| messages:", len(ra["messages"]))
for name, y in (("same cwd, +61 s", b), ("cwd /home/efeu", h)):
    i = firstdiff(a, y)
    if i == len(a) == len(y):
        print(f"{name}: IDENTICAL")
    else:
        print(f"{name}: first difference at char {i} of {len(a)} ({100*i/len(a):.1f} %)")
        print("   A:", repr(a[max(0, i-80):i+60]))
        print("   B:", repr(y[max(0, i-80):i+60]))
PY
