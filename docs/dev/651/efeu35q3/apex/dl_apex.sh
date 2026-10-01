#!/bin/bash
# efeu-TP14: download the IsValorum APEX-I-MiniPlus-V2.1 GGUF + MTP Q4_0 head +
# chat template. Low priority (nice/ionice), resumable; checks the LFS sha256
# from the HF tree API. Does not touch the service.
set -u
D=/root/efeu35q3/models_apex
mkdir -p $D && cd $D
R=IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-MiniPlus-V2.1-Abliterated-GGUF
U=https://huggingface.co/$R/resolve/main
curl -s -m30 "https://huggingface.co/api/models/$R/tree/main" > tree.json
for f in chat_template.jinja README.md mtp-Qwen3.8-35B-A3B-Distill-Q4_0.gguf Qwen3.8-35B-A3B-Distill.APEX-I-MiniPlus-V2.1-Abliterated.gguf; do
  for i in 1 2 3 4 5 6 7 8; do
    nice -n 19 ionice -c3 curl -sS -L --fail -C - --retry 5 --retry-delay 5 --limit-rate 60M -o "$f" "$U/$f" && break
    echo "$f: curl rc=$? retry $i"; sleep 15
  done
  want=$(python3 -c "import json,sys; d={x['path']:x for x in json.load(open('tree.json'))}; print((d['$f'].get('lfs') or {}).get('oid',''))")
  if [ -n "$want" ]; then
    got=$(nice -n 19 ionice -c3 sha256sum "$f" | cut -d' ' -f1)
    [ "$got" = "$want" ] && echo "$f SHA-OK $got" || echo "$f SHA-FAIL got=$got want=$want"
  else
    echo "$f (no lfs) size=$(stat -c %s "$f")"
  fi
done
echo "=== DL-DONE $(date -Is)"
