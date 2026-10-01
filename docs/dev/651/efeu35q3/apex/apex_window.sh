#!/bin/bash
# efeu-TP14: APEX-I-MiniPlus-V2.1 bring-up measurements that need the whole
# machine (the live Q3_K_M service and APEX do not fit in RAM together):
#   A  boot APEX (no spec) :31671  -> probe 8/8 + greedy outputs, decode, prefill
#   B  boot APEX + MTP k=1 (NEXTN, mtp Q4_0 head) -> greedy identity vs A,
#      decode, py-spy record of spec steps (sgl_kernel op census)
#   B2/B3 MTP k=1 / k=3 on the native spec ops (sgl_spec_rocm) -> identity, decode
#   C  llama.cpp CPU reference on the same APEX file -> compare vs A
# Gate: runs only after an explicit RELEASE file (night window, operator) AND
# service up, inflight 0, idle >= 900 s. While the service is down a sentinel
# on :31651 answers 503; a POST (a real request -- GET health probes and the
# dashboard do not count) is a knock: the measurement aborts at once and the
# trap brings the service back.
set -u
cd /root/efeu35q3
LOG=logs/apex_window.log
exec >> $LOG 2>&1
R=results/apex
mkdir -p $R
KNOCK=/tmp/efeu_user_knock
RELEASE=/root/efeu35q3/apex/RELEASE_NIGHT
A=/root/efeu35q3/models_apex/Qwen3.8-35B-A3B-Distill.APEX-I-MiniPlus-V2.1-Abliterated.gguf
M=/root/efeu35q3/models_apex/mtp-Qwen3.8-35B-A3B-Distill-Q4_0.gguf
PY=/root/lh/venv/bin/python
st() { curl -s -m5 localhost:31651/ondemand/status; }
quiet() { st | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d["state"]=="up" and d["inflight"]==0 and d["idle_seconds"]>=900 else 1)' 2>/dev/null; }
knocked() { [ -e $KNOCK ]; }
SENT=""
restore() {
  [ -n "$SENT" ] && kill $SENT 2>/dev/null
  pkill -f "^python -m sglang.launch_server" ; pkill -f "^/root/651-p2/llama.cpp/build/bin/llama-server"
  pkill -f "^py-spy record"
  sleep 3
  systemctl start htsglang-ondemand
  echo "$(date +%T) service restored: $(systemctl is-active htsglang-ondemand)"
}
guard() {  # run "$@" in its own process group; kill it on a knock
  setsid "$@" &
  local p=$!
  while kill -0 $p 2>/dev/null; do
    if knocked; then pkill -KILL -g $(ps -o pgid= $p | tr -d " "); echo "$(date +%T) KNOCK -> aborted: $*"; return 1; fi
    sleep 2
  done
  wait $p
}
boot() {  # $1 label, rest = env assignments; waits for "fired up"
  local L=$1; shift
  local BL=logs/boot_apex_$L.log
  env "$@" MODEL=$A TOKENIZER=/root/efeu35q3/hf_apex PORT=31671 CTX=32768 MEMFRAC=0.92 \
      CHUNKED_PREFILL=512 WEDGE_CP_MEASURED=512 MAXRUN=1 MAMBASLOTS=4 SERVED_NAME=apex \
      GUARD_EXTRA_GGUF=$A setsid ./boot_q38.sh > $BL 2>&1 &
  for i in $(seq 1 120); do
    grep -q "fired up" $BL && { echo "$(date +%T) $L up"; return 0; }
    grep -qE "Traceback|guard v2 failed|policy refused" $BL && { echo "$(date +%T) $L BOOT FAILED"; tail -5 $BL; return 1; }
    knocked && return 1
    sleep 5
  done
  echo "$(date +%T) $L boot timeout"; return 1
}
stop_sglang() { pkill -f "^python -m sglang.launch_server"; sleep 8; }

echo "=== apex_window armed $(date -Is)"
until [ -e $RELEASE ]; do sleep 60; done
until quiet; do sleep 30; done
echo "$(date +%T) quiet window: $(st)"
rm -f $KNOCK
trap restore EXIT INT TERM
systemctl stop htsglang-ondemand
python3 - <<'PY' &
import http.server, pathlib
class H(http.server.BaseHTTPRequestHandler):
    def _r(self, knock):
        if knock:
            pathlib.Path("/tmp/efeu_user_knock").touch()
        b = b'{"error":"model service in a short maintenance window, back in a few minutes"}'
        self.send_response(503); self.send_header("Content-Type", "application/json")
        self.send_header("Retry-After", "180"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._r(False)
    def do_POST(self): self._r(True)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", 31651), H).serve_forever()
PY
SENT=$!
powerprofilesctl set balanced

# --- A: APEX without spec
boot nospec || exit 0
guard $PY probe_q38.py 31671 apex --json $R/probe_apex_nospec.json || exit 0
guard $PY ab18k/greedy_long.py 31671 apex_nospec $R/greedy_long_apex_nospec.json apex || exit 0
guard $PY bench_decode.py --port 31671 --model apex --label apex_nospec --out $R/decode_apex_nospec.json || exit 0
guard $PY bench_prefill.py --port 31671 --model apex --label apex_nospec --lengths 512,2048,8192 --out $R/prefill_apex_nospec.json || exit 0
grep -E "There is no support for fast MoE|Traceback" logs/boot_apex_nospec.log | head -3
stop_sglang

# --- B: APEX + MTP k=1
if boot mtp1 "EXTRA=--speculative-algorithm NEXTN --speculative-draft-model-path $M --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2"; then
  guard $PY probe_q38.py 31671 apex --json $R/probe_apex_mtp1.json || exit 0
  guard $PY ab18k/greedy_long.py 31671 apex_mtp1 $R/greedy_long_apex_mtp1.json apex || exit 0
  P=$(pgrep -f "sglang::scheduler" | head -1); [ -z "$P" ] && P=$(pgrep -f "^python -m sglang.launch_server" | head -1)
  ( py-spy record --pid $P --subprocesses --duration 25 --rate 200 --format raw -o $R/pyspy_apex_mtp1.txt > /dev/null 2>&1 & )
  guard $PY bench_decode.py --port 31671 --model apex --label apex_mtp1 --out $R/decode_apex_mtp1.json || exit 0
  grep -E "accept|spec" logs/boot_apex_mtp1.log | tail -5 | cut -c1-200
else
  echo "$(date +%T) MTP boot failed -- see logs/boot_apex_mtp1.log; continuing with C"
  knocked && exit 0
fi
stop_sglang

# --- B2/B3: native spec ops (sgl_spec_rocm, gfx1103 build of upstream
# eagle_utils.cu) on the import path -> decide_spec_kernel_backend = native.
# k=1 (steps 1, draft 2) and k=3 (steps 3, draft 4), each with greedy identity
# vs A and decode tok/s; py-spy census for k=3.
SPECEXT="GGUF_EXT_DIR=/root/efeu35q3/ext_v2:/root/efeu35q3/spec_rocm"
for K in 1 3; do
  if boot mtp${K}native "$SPECEXT" "EXTRA=--speculative-algorithm NEXTN --speculative-draft-model-path $M --speculative-num-steps $K --speculative-eagle-topk 1 --speculative-num-draft-tokens $((K+1))"; then
    grep -m1 "Spec kernel backend" logs/boot_apex_mtp${K}native.log | cut -c1-160
    guard $PY probe_q38.py 31671 apex --json $R/probe_apex_mtp${K}native.json || exit 0
    guard $PY ab18k/greedy_long.py 31671 apex_mtp${K}native $R/greedy_long_apex_mtp${K}native.json apex || exit 0
    if [ $K = 3 ]; then
      P=$(pgrep -f "sglang::scheduler" | head -1); [ -z "$P" ] && P=$(pgrep -f "^python -m sglang.launch_server" | head -1)
      ( py-spy record --pid $P --subprocesses --duration 25 --rate 200 --format raw -o $R/pyspy_apex_mtp3native.txt > /dev/null 2>&1 & )
    fi
    guard $PY bench_decode.py --port 31671 --model apex --label apex_mtp${K}native --out $R/decode_apex_mtp${K}native.json || exit 0
    grep -E "accept len|accept_len|spec" logs/boot_apex_mtp${K}native.log | tail -3 | cut -c1-200
  else
    echo "$(date +%T) mtp${K}native boot failed -- see logs/boot_apex_mtp${K}native.log"
    knocked && exit 0
  fi
  stop_sglang
done

# --- C: llama.cpp CPU reference, same file
/root/651-p2/llama.cpp/build/bin/llama-server -m $A -ngl 0 --jinja --chat-template-file /root/efeu35q3/hf_apex/chat_template.jinja \
  -c 8192 -t 8 --port 31690 --host 127.0.0.1 > logs/llama_apex.log 2>&1 &
for i in $(seq 1 90); do curl -sf -m 3 localhost:31690/health >/dev/null && break; knocked && exit 0; sleep 3; done
guard $PY llama_ref_compare.py 31690 $R/probe_apex_nospec.json --json $R/llama_ref_compare_apex.json || exit 0
pkill -f "^/root/651-p2/llama.cpp/build/bin/llama-server"; sleep 3
echo "=== APEX WINDOW DONE $(date -Is)"
