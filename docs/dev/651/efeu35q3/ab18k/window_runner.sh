#!/bin/bash
# efeu-TP14: measurements that need the whole machine (18k greedy A/B of the
# out_proj lever, then the llama.cpp PP2 sweep), run ONLY in a quiet window
# (service up, inflight 0, idle >= 900 s), with the user's service guaranteed
# back afterwards:
#   * trap EXIT/INT/TERM -> systemctl start htsglang-ondemand (always)
#   * while the service is down, a sentinel on :31651 answers 503 "back in a few
#     minutes" and records the knock; a knock aborts the measurement at once and
#     the trap brings the service back.
set -u
cd /root/efeu35q3
LOG=logs/window_runner.log
exec >> $LOG 2>&1
KNOCK=/tmp/efeu_user_knock
st() { curl -s -m5 localhost:31651/ondemand/status; }
quiet() { st | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d["state"]=="up" and d["inflight"]==0 and d["idle_seconds"]>=900 else 1)' 2>/dev/null; }
SENT=""
restore() {
  [ -n "$SENT" ] && kill $SENT 2>/dev/null
  pkill -f "^python -m sglang.launch_server" ; pkill -f "^/root/651-p2/llama.cpp/build/bin/llama-server|^/root/efeu35q3/llama-hip-build/bin/llama-"
  sleep 3
  powerprofilesctl set power-saver
  systemctl start htsglang-ondemand
  echo "$(date +%T) service restored: $(systemctl is-active htsglang-ondemand)"
}
knocked() { [ -e $KNOCK ]; }
run_guarded() {  # run "$@" in its own process group; kill it on a knock
  setsid "$@" &
  local p=$!
  while kill -0 $p 2>/dev/null; do
    if knocked; then pkill -KILL -g $(ps -o pgid= $p | tr -d " "); echo "$(date +%T) KNOCK -> aborted: $*"; return 1; fi
    sleep 2
  done
  wait $p
}
echo "=== window_runner start $(date -Is)"
# the staged production config is applied first (apply_when_quiet.sh)
until grep -q "APPLY DONE" logs/apply_when_quiet.log 2>/dev/null; do sleep 30; done
until quiet; do sleep 30; done
echo "$(date +%T) quiet window: $(st)"
rm -f $KNOCK
trap restore EXIT INT TERM
systemctl stop htsglang-ondemand
python3 - <<'PY' &
import http.server, pathlib
class H(http.server.BaseHTTPRequestHandler):
    def _r(self):
        pathlib.Path("/tmp/efeu_user_knock").touch()
        b = b'{"error":"model service in a short maintenance window, back in ~3 minutes"}'
        self.send_response(503); self.send_header("Content-Type", "application/json")
        self.send_header("Retry-After", "180"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    do_GET = do_POST = _r
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", 31651), H).serve_forever()
PY
SENT=$!
powerprofilesctl set balanced
mkdir -p results/ab18k
for MODE in permute dense; do
  knocked && exit 0
  [ -s results/ab18k/sglang_$MODE.json ] && continue
  P=1; [ $MODE = dense ] && P=0
  BL=logs/boot_ab18k_$MODE.log
  SGLANG_GGUF_GDN_OUTPROJ_PERMUTE=$P PORT=31671 CTX=32768 MEMFRAC=0.92 CHUNKED_PREFILL=512 WEDGE_CP_MEASURED=512 \
    setsid ./boot_q38.sh > $BL 2>&1 &
  for i in $(seq 1 90); do grep -q "fired up" $BL && break; knocked && exit 0; sleep 5; done
  run_guarded /root/lh/venv/bin/python ab18k/greedy_long.py 31671 sglang_$MODE results/ab18k/sglang_$MODE.json || exit 0
  pkill -f "^python -m sglang.launch_server"; sleep 8
done
if [ ! -s results/ab18k/llamacpp_cpu.json ] && ! knocked; then
  /root/651-p2/llama.cpp/build/bin/llama-server -m models/Qwen3.8-35B-A3B-Q3_K_M.gguf -ngl 0 --jinja -c 20480 -t 8 \
    --port 31690 --host 127.0.0.1 > logs/llama_ab18k.log 2>&1 &
  for i in $(seq 1 60); do curl -sf -m 3 localhost:31690/health >/dev/null && break; sleep 3; done
  run_guarded /root/lh/venv/bin/python ab18k/greedy_long.py 31690 llamacpp_cpu results/ab18k/llamacpp_cpu.json || exit 0
  pkill -f "^/root/651-p2/llama.cpp/build/bin/llama-server"; sleep 3
fi
# PP2 (llama.cpp HIP, -ngl k), after the A/B
if [ ! -x llama-hip-build/bin/llama-bench ] && ! knocked; then
  run_guarded bash pp2/build_llama_hip.sh || exit 0
fi
for NGL in 41 33 25 0; do
  knocked && exit 0
  [ -s results/pp2/coh_ngl$NGL.json ] && continue
  NGLS=$NGL run_guarded bash pp2/pp2_llama.sh || exit 0
done
echo "=== WINDOW DONE $(date -Is)"
