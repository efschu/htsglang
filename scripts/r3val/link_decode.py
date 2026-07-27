"""Cross-rig link A/B: pure DECODE rate, measured by streaming.

Task #204. The decode rate is taken as

    (n_tokens - 1) / (t_last_token - t_first_token)

i.e. the time between the FIRST and LAST streamed token. This measures decode
only. It is immune both to prefill/TTFT and to the radix prefix-cache artefact
that invalidated the earlier e2e-slope numbers (a repeated identical prompt
hits the cache, so the constant prefill term does NOT cancel in an e2e slope).

Per point: ~10-20 s of decode, 3 content classes, greedy, degeneration check,
per-GPU clock/temp annotated. Short runs deliberately: on this rig the cards
stay cool and unthrottled in a short burst, which is both the honest regime
for a link comparison and closer to real bursty chat load than a sustained
battery.

Usage: link_decode.py <base_url> <label> [reps] [max_new_tokens]
"""

import json
import os
import statistics as st
import subprocess
import sys
import time

import requests

BASE = sys.argv[1].rstrip("/")
LABEL = sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
NTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 200

# Records land beside this harness unless R3VAL_LOGS points elsewhere.
LOGS = os.environ.get("R3VAL_LOGS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs"
)
os.makedirs(LOGS, exist_ok=True)

PROMPTS = {
    "code": "Write a complete, well-commented Python implementation of a red-black tree "
            "with insert, delete, and search, plus a short explanation of each rebalancing case.",
    "prosa": "Write a long, vivid narrative essay about a lighthouse keeper on the Atlantic "
             "coast across four decades, focusing on weather, solitude, and memory.",
    "misch": "Explain how paged attention works in a modern LLM inference server, then give "
             "an annotated Python sketch of the block table, then discuss the trade-offs in prose.",
}

# Rig 1's ssh key and address come from the environment (source your local rig
# env file); the fallbacks are placeholders so an unsourced run fails at ssh
# instead of probing some other machine.
RIG1_KEY = os.environ.get("RIG1_KEY", "<RIG1_SSH_KEY>")
RIG1_HOST = os.environ.get("RIG1_HOST", "<RIG1_IP>")

REMOTE_SMI = ["ssh", "-n", "-i", RIG1_KEY, "-o", "IdentitiesOnly=yes",
              "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5", f"root@{RIG1_HOST}",
              "nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,"
              "clocks_throttle_reasons.active --format=csv,noheader,nounits"]


def gpu_state():
    """Rig1 clock/temp/throttle. Annotated per point, never used to justify
    a placement decision -- throttling is a condition, not a design input."""
    try:
        out = subprocess.run(REMOTE_SMI, capture_output=True, text=True,
                             timeout=15).stdout.strip()
        st_ = {}
        for line in out.splitlines():
            p = [x.strip() for x in line.split(",")]
            st_[f"gpu{p[0]}"] = {"sm_mhz": int(p[1]), "temp": int(p[2]),
                                 "throttled": p[3] != "0x0000000000000000"}
        return st_
    except Exception as e:
        return {"error": str(e)}


def stream_once(prompt, ntok):
    """Returns (decode_rate, n, first_tok_s, decode_s, text)."""
    r = requests.post(
        f"{BASE}/generate",
        json={"text": prompt,
              "sampling_params": {"max_new_tokens": ntok, "temperature": 0.0,
                                  "ignore_eos": True},
              "stream": True},
        stream=True, timeout=900)
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    n = 0
    text = ""
    for raw in r.iter_lines():
        if not raw:
            continue
        s = raw.decode("utf-8", "ignore")
        if s.startswith("data: "):
            s = s[6:]
        if s.strip() == "[DONE]":
            break
        try:
            d = json.loads(s)
        except Exception:
            continue
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        t_last = now
        n = d.get("meta_info", {}).get("completion_tokens", n)
        text = d.get("text", text)
    if t_first is None or t_last is None or t_last <= t_first or n < 2:
        return None
    return ((n - 1) / (t_last - t_first), n, t_first - t0, t_last - t_first, text)


def degenerate(txt):
    w = txt.split()
    if len(w) < 60:
        return False
    tail = w[-60:]
    return len(set(tail)) < 8


out = {"label": LABEL, "base": BASE, "ntok": NTOK, "reps": REPS, "classes": {}}
print(f"# {LABEL}  ntok={NTOK} reps={REPS}", flush=True)
print(f"{'class':6s} {'rep':>3s} {'decode tok/s':>13s} {'n':>5s} {'ttft_s':>8s} "
      f"{'decode_s':>9s}  gpu clocks", flush=True)

for cls, p in PROMPTS.items():
    rates, raws = [], []
    for i in range(REPS):
        g0 = gpu_state()
        res = stream_once(p, NTOK)
        if res is None:
            print(f"{cls:6s} {i:3d}  FAILED/too short", flush=True)
            continue
        rate, n, ttft, dsec, text = res
        deg = degenerate(text)
        rates.append(rate)
        raws.append({"rep": i, "rate": rate, "n": n, "ttft_s": ttft,
                     "decode_s": dsec, "degenerate": deg,
                     "text_head": text[:120], "gpu": g0})
        clk = " ".join(f"{k}:{v['sm_mhz']}MHz/{v['temp']}C"
                       + ("(thr)" if v["throttled"] else "")
                       for k, v in sorted(g0.items()) if isinstance(v, dict))
        print(f"{cls:6s} {i:3d} {rate:13.2f} {n:5d} {ttft:8.2f} {dsec:9.2f}  {clk}"
              + ("  !!DEGENERATE" if deg else ""), flush=True)
    if rates:
        out["classes"][cls] = {
            "rates": rates, "median": round(st.median(rates), 3),
            "mean": round(st.mean(rates), 3),
            "sd": round(st.stdev(rates), 3) if len(rates) > 1 else 0.0,
            "raw": raws}
        print(f"{cls:6s}  -> median {st.median(rates):.2f} tok/s "
              f"(n={len(rates)}, sd={out['classes'][cls]['sd']:.2f})", flush=True)

path = os.path.join(LOGS, f"link_{LABEL}.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print(f"\nraw -> {path}")
print("SUMMARY " + json.dumps({c: v["median"] for c, v in out["classes"].items()}))
