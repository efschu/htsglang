#!/bin/bash
# #655: honest prefill / decode throughput on the 780M iGPU, same boot.
#
# Streaming is what separates the two phases. Time-to-first-token is the prefill
# cost for a known prompt length; every later chunk is decode. Measuring both
# from one non-streaming elapsed number conflates them, which is why the wedge
# sweep's timings are only good to a factor.
#
# ignore_eos pins the decode length so the rate is not hostage to where the
# model chooses to stop.
#
# NOTE: this is the NO-OFFLOAD baseline. kt_kernel does not import on this box,
# so no expert is served from CPU or disk. These are the numbers an offload lane
# would have to beat.
set -u

BASE=http://127.0.0.1:31651
MODEL=qwen36-35b-a3b
STAMP=$(date +%H%M%S)
RESULT=/root/651-p2/results/bench_tps_${STAMP}.txt
mkdir -p /root/651-p2/results
exec > >(tee -a "$RESULT") 2>&1

echo "=== prefill/decode throughput ${STAMP} (no expert offload) ==="
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
r = urllib.request.Request(base + "/v1/chat/completions",
                           data=json.dumps(req).encode(),
                           headers={"Content-Type": "application/json"})
t0 = time.time()
ttft = None
ntok = 0
ptok = None
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
print(f"prompt_tokens={ptok} decode_tokens={ntok} "
      f"ttft={ttft:.2f}s prefill_tok_s={(ptok or 0)/ttft:.1f} "
      f"decode_s={dec_s:.2f} decode_tok_s={(ntok-1)/dec_s if dec_s > 0 else 0:.2f} "
      f"total={total:.2f}s")
PY
}

echo "--- decode rate at short context ---"
bench 0 256
bench 0 256
echo "--- prefill rate, rising prompt length (decode pinned to 16) ---"
for w in 200 800 1600 3200; do bench "$w" 16; done
echo "--- decode rate at long context ---"
bench 1600 512
echo "resets_after=$(dmesg -T | grep -cE 'GPU reset\(')"
