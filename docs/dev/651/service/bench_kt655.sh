#!/bin/bash
# #655: decode/prefill throughput, tagged so a kt run and a no-kt run land in
# separate files. Same streaming + ignore_eos method as bench_tokens_per_s.sh:
# TTFT is the prefill cost, everything after it is decode, and ignore_eos is
# what makes the decode budget bind (without it the model stops at ~10 tokens
# and max_tokens measures nothing).
set -u

BASE=http://127.0.0.1:31651
MODEL=qwen36-35b-a3b
TAG=${TAG:-untagged}
STAMP=$(date +%H%M%S)
RESULT=/root/651-p2/results/bench_kt655_${TAG}_${STAMP}.txt
mkdir -p /root/651-p2/results
exec > >(tee -a "$RESULT") 2>&1

echo "=== $TAG $STAMP ==="
echo "resets_before=$(dmesg -T | grep -cE 'GPU reset\(')"

bench() {
  local words="$1" decode="$2"
  python3 - "$BASE" "$MODEL" "$words" "$decode" <<'PY'
import json, sys, time, urllib.request

base, model, words, decode = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
filler = " ".join(["alpha bravo charlie delta echo"] * max(1, words // 5)) if words else ""
prompt = ("Summarize this text. " + filler) if words else "Write a long description of the sea."
req = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "chat_template_kwargs": {"enable_thinking": False},
    "temperature": 0,
    "max_tokens": decode,
    "ignore_eos": True,
    "stream": True,
    "stream_options": {"include_usage": True},
}
r = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(req).encode(),
                           headers={"Content-Type": "application/json"})
t0 = time.time(); ttft = None; ntok = 0; ptok = None
with urllib.request.urlopen(r, timeout=900) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        d = json.loads(payload)
        if d.get("usage"):
            ptok = d["usage"].get("prompt_tokens", ptok)
        for ch in d.get("choices", []):
            if ch.get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                ntok += 1
total = time.time() - t0
if ttft is None:
    print("no tokens streamed"); sys.exit(1)
dec_s = total - ttft
rate = (ntok - 1) / dec_s if dec_s > 0 else 0
print(f"prompt_tokens={ptok} decode_tokens={ntok} ttft={ttft:.2f}s "
      f"prefill_tok_s={(ptok or 0)/ttft:.1f} decode_s={dec_s:.2f} "
      f"decode_tok_s={rate:.2f} ms_per_token={(1000/rate if rate else 0):.1f} total={total:.2f}s")
PY
}

echo "--- decode at short context (3 runs, first is warm-up) ---"
bench 0 128
bench 0 256
bench 0 256
echo "--- prefill, 800-word prompt, decode pinned to 16 ---"
bench 800 16
echo "resets_after=$(dmesg -T | grep -cE 'GPU reset\(')"
echo "=== done $TAG ==="
