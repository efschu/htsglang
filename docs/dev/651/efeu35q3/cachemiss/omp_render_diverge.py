"""efeu-TP14: passive proof of where consecutive omp prompts diverge.

Rebuilds the OpenAI message list omp sends for each request of one omp session
(from ~efeu/.omp/agent/sessions/*.jsonl), renders it with the served chat
template, and compares consecutive renders token by token. The unknown system
prompt + tool block is a placeholder; its token length is calibrated as
(prompt_tokens from the backend log) - (rendered length), and must come out the
same for every request, which validates the reconstruction.

usage: omp_render_diverge.py SESSION.jsonl HF_DIR [preserve_thinking=0|1]
"""
import json
import sys

from transformers import AutoTokenizer

sess, hf = sys.argv[1], sys.argv[2]
preserve = len(sys.argv) > 3 and sys.argv[3] == "1"
tok = AutoTokenizer.from_pretrained(hf)

entries = [json.loads(l) for l in open(sess)]
msgs = []          # OpenAI-shaped history
requests = []      # (label, list of messages at the time a request was sent)
last_reset = 0
for e in entries:
    if e.get("type") == "reset_boundary":
        last_reset = len(msgs)
    if e.get("type") != "message":
        continue
    m = e["message"]
    role = m["role"]
    c = m.get("content") or []
    if role == "user":
        msgs.append({"role": "user", "content": "".join(x.get("text", "") for x in c if x.get("type") == "text")})
        requests.append((e["timestamp"][11:19] + " user", list(msgs[last_reset:])))
    elif role == "assistant":
        if not c:  # errored / aborted turn: nothing is replayed
            continue
        text = "".join(x.get("text", "") for x in c if x.get("type") == "text")
        calls = [{"id": x["id"], "type": "function",
                  "function": {"name": x["name"], "arguments": x.get("arguments") or {}}}
                 for x in c if x.get("type") == "toolCall"]
        am = {"role": "assistant", "content": text}
        if calls:
            am["tool_calls"] = calls
        msgs.append(am)
    elif role == "toolResult":
        text = "".join(x.get("text", "") for x in c if x.get("type") == "text")
        msgs.append({"role": "tool", "tool_call_id": m.get("toolCallId", ""), "content": text})
        requests.append((e["timestamp"][11:19] + " tool", list(msgs[last_reset:])))

SYS = {"role": "system", "content": "SYSTEM-PLACEHOLDER"}
prev = None
for label, ms in requests:
    kw = {"preserve_thinking": True} if preserve else {}
    ids = tok.apply_chat_template([SYS] + ms, tokenize=True, add_generation_prompt=True, **kw)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    d = None
    if prev is not None:
        n = min(len(prev), len(ids))
        d = next((i for i in range(n) if prev[i] != ids[i]), n)
    info = ""
    if d is not None and d < len(prev):
        info = f" DIVERGES at rel {d} (prev len {len(prev)}): prev[{d}:+6]={tok.decode(prev[d:d+6])!r} new={tok.decode(ids[d:d+6])!r}"
    elif d is not None:
        info = " append-only"
    print(f"{label}: msgs {len(ms)} rel_len {len(ids)}{info}")
    prev = ids
