"""Determinism vs prompt length: <=8-token prompts prefill via MMVQ,
longer ones via dequantize+rocBLAS (SGLANG_GGUF_MMQ_MAX_TOKENS default 8)."""
import json, sys, urllib.request, collections

PORT = int(sys.argv[1]); N = 8
CASES = {"short(4tok)": "2+2=", "long(17tok)": "Question: What is 14 * 3? Answer with just the number.\nAnswer:"}
for label, prompt in CASES.items():
    firsts = collections.Counter(); texts = collections.Counter()
    for i in range(N):
        body = json.dumps({"text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 8},
            "return_logprob": True, "top_logprobs_num": 4}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        tl = d.get("meta_info", {}).get("output_top_logprobs")
        firsts[json.dumps(tl[0] if tl else None)] += 1
        texts[d["text"]] += 1
    print(f"{label}: distinct first-token logprob sets {len(firsts)}/{N}, distinct texts {len(texts)}/{N}")
