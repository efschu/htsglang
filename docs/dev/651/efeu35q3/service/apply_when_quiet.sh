#!/bin/bash
# efeu-TP14: apply a staged 50-q38.conf with ONE service restart, only when the
# user is not working (inflight 0 and idle >= IDLE_MIN s).
#   PROOF=1 (default 0): afterwards omp turn / side request / omp turn as active
#           proof. PROOF=0: passive -- no request is sent; evidence comes from
#           the user's own next turn.
#   Rollback: if the new config does not come up (e.g. "Pinned host RAM
#   over-committed") within 6 min, the previous drop-in is restored and the
#   service restarted on it, so the user is never left without a model.
set -u
IDLE_MIN=${IDLE_MIN:-300}
PROOF=${PROOF:-0}
cd /root/efeu35q3
LOG=logs/apply_when_quiet.log
exec >> $LOG 2>&1
D=/etc/systemd/system/htsglang-ondemand.service.d/50-q38.conf
st() { curl -s -m5 localhost:31651/ondemand/status; }
wait_up() {  # $1 seconds
  local t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt $1 ]; do
    st | grep -q '"state": "up"' && return 0
    sleep 5
  done
  return 1
}
echo "=== apply_when_quiet start $(date -Is) (idle >= $IDLE_MIN s, inflight 0, PROOF=$PROOF)"
until st | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d['inflight']==0 and d['idle_seconds']>=$IDLE_MIN else 1)" 2>/dev/null; do sleep 15; done
echo "$(date +%T) quiet: $(st)"
echo "--- before"; free -m | sed -n 2,3p
cp $D /root/efeu35q3/50-q38.conf.previous
install -m 0644 /root/efeu35q3/50-q38.conf $D
systemctl daemon-reload
systemctl restart htsglang-ondemand
T0=$(date +%s)
if ! wait_up 360; then
  echo "$(date +%T) NEW CONFIG DID NOT COME UP -> rollback"
  B=$(st | python3 -c 'import json,sys; print(json.load(sys.stdin).get("backend_log") or "")')
  [ -n "$B" ] && grep -E "over-committed|Error|Traceback" $B | tail -3 | cut -c1-200
  install -m 0644 /root/efeu35q3/50-q38.conf.previous $D
  systemctl daemon-reload; systemctl restart htsglang-ondemand
  wait_up 360 && echo "$(date +%T) rolled back, service up" || echo "$(date +%T) ROLLBACK ALSO FAILED"
  echo "=== APPLY DONE (ROLLED BACK) $(date -Is)"; exit 1
fi
echo "$(date +%T) up after $(( $(date +%s)-T0 )) s: $(st)"
B=$(st | python3 -c 'import json,sys; print(json.load(sys.stdin)["backend_log"])')
grep -E "Mamba Cache is allocated|pinned host memory|host KV pool|max_total_num_tokens=|over-committed" $B | cut -c1-200
echo "--- after load"; free -m | sed -n 2,3p
if [ "$PROOF" = "1" ]; then
  mkdir -p /tmp/omp-proof2 && chown efeu /tmp/omp-proof2
  omp_turn() { su - efeu -c "export PATH=\$HOME/.local/bin:\$PATH; cd /tmp/omp-proof2; timeout 900 omp --model local/qwen38-35b-a3b --no-lsp --no-skills -p '$1'" 2>&1 | tail -1; }
  T1=$(date +%s); omp_turn "say hi (turn 1)"; echo "turn 1: $(( $(date +%s)-T1 )) s"
  curl -s -m 120 localhost:31651/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"qwen38-35b-a3b","messages":[{"role":"user","content":"Give a 3-word title for: say hi"}],"max_tokens":12,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import json,sys; print("side request:", json.load(sys.stdin)["choices"][0]["message"]["content"])'
  T2=$(date +%s); omp_turn "say hi again (turn 2)"; echo "turn 2: $(( $(date +%s)-T2 )) s"
  grep -E "Prefill batch" $B | sed -E "s/.*(\[2026[^]]*\]).*#new-token: ([0-9]+), #cached-token: ([0-9]+).*/\1 new=\2 cached=\3/" | grep -v "new=512 cached=0" | tail -12
  echo "--- after turns"; free -m | sed -n 2,3p
fi
echo "=== APPLY DONE $(date -Is)"
