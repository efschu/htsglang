#!/bin/bash
# Nordstern L0 preflight -- everything that can be checked WITHOUT starting a
# rank or touching a GPU. Run this before asking for the coordinated GPU
# window; it should be all-green first.
set -u
SECOND=${SECOND:-192.168.0.89}
KEY=${KEY:-/root/.ssh/id_ed25519_192.168.0.89}
SSH="ssh -i $KEY -o IdentitiesOnly=yes -o ConnectTimeout=10"
MASTER=${MASTER:-192.168.0.101}
PORT=${PORT:-31900}
MODEL_MAIN=${MODEL_MAIN:-/spinning/llm_stuff/club-3090/models-cache/Llama-3.1-8B-Instruct}
MODEL_SECOND=${MODEL_SECOND:-/root/models/llama-3.1-8b}
fail=0
ok(){ echo "  OK   $*"; }; bad(){ echo "  FAIL $*"; fail=1; }

echo "== 1. reachability =="
ping -c3 -W2 "$SECOND" >/dev/null 2>&1 && ok "ping $SECOND" || bad "ping $SECOND"
$SSH root@$SECOND true >/dev/null 2>&1 && ok "ssh $SECOND" || bad "ssh $SECOND"
$SSH root@$SECOND "getent hosts $MASTER >/dev/null || ping -c2 -W2 $MASTER >/dev/null" 2>/dev/null \
  && ok "second host can reach master $MASTER" || bad "second host cannot reach $MASTER"

echo "== 2. rendezvous port $PORT free on master =="
ss -ltn 2>/dev/null | grep -q ":$PORT " && bad "port $PORT already bound on master" || ok "port $PORT free"

echo "== 3. nothing else holding the GPUs =="
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
[ "$n" -eq 0 ] && ok "main rig GPUs idle" || bad "main rig has $n compute process(es) -- coordinate first"
$SSH root@$SECOND 'pgrep -f "launch_serve[r]" >/dev/null' 2>/dev/null \
  && bad "second host still running a server" || ok "second host idle"

echo "== 4. model present on BOTH sides =="
[ -d "$MODEL_MAIN" ] && ok "main: $MODEL_MAIN" || bad "main: $MODEL_MAIN missing"
$SSH root@$SECOND "[ -d $MODEL_SECOND ]" 2>/dev/null && ok "second: $MODEL_SECOND" || \
  bad "second: $MODEL_SECOND missing -- copy it first (see l0_copy_model.sh)"

echo "== 5. same code on both sides =="
for f in python/sglang/srt/distributed/utils.py python/sglang/srt/layers/quantization/fp8.py; do
  a=$(md5sum /spinning/wt-htccl/$f 2>/dev/null | cut -d' ' -f1)
  b=$($SSH root@$SECOND "md5sum /root/sglang-src/sglang/${f#python/sglang/}" 2>/dev/null | cut -d' ' -f1)
  [ -n "$a" ] && [ "$a" = "$b" ] && ok "$(basename $f) identical" || bad "$(basename $f) DIFFERS (a=$a b=$b)"
done

echo
[ $fail -eq 0 ] && echo "### PREFLIGHT GREEN" || echo "### PREFLIGHT RED -- fix the FAILs above"
exit $fail
