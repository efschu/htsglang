#!/bin/bash
# #655: find the largest CTX this GPU will actually SERVE, offload on.
#
# Why this exists. A 978-token prompt ran clean at CTX=8192 with the offload
# on (bench_kt655_kt0_033057.txt, resets unchanged). After CTX was raised to
# 196608 the SAME small prompt class took GPU reset(6) at ~10k and reset(7,8)
# at ~1k. Prompt depth therefore does not explain it and context size does:
# the variable that changed between the clean runs and the wedging runs is
# --context-length, not the request.
#
# So this walks CTX upward with a small fixed 1k probe as the canary and
# records, per value, whether the machine survives a single request. The probe
# is deliberately SMALL: the question here is not how deep a prompt can go, it
# is whether the server is usable at all at that context setting.
#
# Each iteration costs a restart plus a ~195 s cold load, so this is detached.
set -u
TS=$(date +%H%M%S)
LOG=/root/651-p2/results/ctxwall655_${TS}.log
exec > >(tee -a "$LOG") 2>&1

D=/etc/systemd/system/htsglang-ondemand.service.d
TESTCONF=$D/50-ctxtest.conf

resets() { dmesg -T 2>/dev/null | grep -cE "GPU reset\("; }
mark() { echo "[$(date -Is)] $*"; }

wait_ready() {   # returns 0 when up, 1 on timeout/park
  for _ in $(seq 1 80); do
    case "$(curl -s -m 8 localhost:31651/ondemand/status)" in
      *'"state": "up"'*) return 0;;
    esac
    sleep 5
  done
  return 1
}

mark "ctxwall start; resets=$(resets)"
BEST=""

for ctx in 8192 32768 65536 131072 196608; do
  printf '[Service]\n# temporary: written by ctxwall655.sh\nEnvironment=CTX=%s\n' "$ctx" > "$TESTCONF"
  systemctl daemon-reload
  systemctl restart htsglang-ondemand
  sleep 3
  R0=$(resets)
  mark "--- CTX=$ctx (resets before=$R0) ---"

  # The probe itself triggers the lazy load; its own timeout covers it.
  timeout 900 python3 /root/651-p2/scripts/needle655.py 1000 \
    "/root/651-p2/results/ctxwall655_${ctx}_${TS}.txt" 2>&1 | \
    grep -E "achieved_prompt_tokens|prefill_wall_s|prefill_tok_s|decode_tok_s|NEEDLE-PROBE|HTTPError|URLError"
  RC=${PIPESTATUS[0]}
  sleep 8
  R1=$(resets)

  POOL=$(curl -s -m 8 localhost:31651/ondemand/status | grep -o '"kv_tokens": [0-9]*' | head -1)
  if [ "$R1" != "$R0" ]; then
    mark "CTX=$ctx WEDGE (resets $R0 -> $R1) rc=$RC $POOL"
    mark "stopping: higher values cannot be better"
    break
  else
    mark "CTX=$ctx CLEAN rc=$RC $POOL"
    [ "$RC" = "0" ] && BEST=$ctx
  fi
done

mark "=== ctxwall done. largest CTX that served a clean probe: ${BEST:-NONE} ==="
mark "resets now $(resets)"
mark "NOTE: 50-ctxtest.conf is still in place; the caller decides the final value."
mark "log=$LOG"
