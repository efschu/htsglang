"""A-vs-A noise-floor + thermal-drift probe (Task #103, methodology gate).

WHY THIS EXISTS
---------------
Clock pinning (`nvidia-smi -lgc`) is REFUSED by the driver on the GeForce
cards in this rig, even as root. GPU2 (3080) was observed at 88 C with
sw_thermal_slowdown ACTIVE and SM clock 1620 MHz while the identical GPU0
(3080) ran 1920 MHz -- an 18 % clock spread between two same-model cards,
purely thermal. Under lock-step TP the slowest rank sets the pace, so this
drift lands directly on end-to-end tok/s.

Therefore the detection limit must be MEASURED, not assumed. This probe runs
one single unchanged configuration (A vs A) as a long series of identical
measurement blocks while sampling clock / temperature / throttle state, so
that we can read off:
  1. the warm-up transient duration -> the mandatory pre-conditioning time,
  2. the residual scatter in the thermal plateau -> THE DETECTION LIMIT.

Any effect reported later by this campaign must beat that limit. Anything
below it is "below the detection limit", never "a small win".

Usage: noise_floor.py <port> <label> <n_blocks> [class]
Env:   NF_SHORT / NF_LONG decode window.
"""

import json
import os
import statistics
import subprocess
import sys
import threading
import time

import requests

PORT = int(sys.argv[1])
LABEL = sys.argv[2]
NBLOCKS = int(sys.argv[3])
CLS = sys.argv[4] if len(sys.argv) > 4 else "code"
BASE = f"http://127.0.0.1:{PORT}"

N_SHORT = int(os.environ.get("NF_SHORT", "200"))
N_LONG = int(os.environ.get("NF_LONG", "1000"))

PROMPTS = {
    "code": "Write a complete, well-commented Python implementation of a red-black tree "
            "with insert, delete, and search, plus a short explanation of each rebalancing case.",
    "prosa": "Write a long, vivid narrative essay about a lighthouse keeper on the Atlantic "
             "coast across four decades, focusing on weather, solitude, and memory.",
    "misch": "Explain how paged attention works in a modern LLM inference server, then give "
             "an annotated Python sketch of the block table, then discuss the trade-offs in prose.",
}
PROMPT = PROMPTS[CLS]

_stop = threading.Event()
_samples = []


def sampler():
    while not _stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,clocks.sm,temperature.gpu,utilization.gpu,"
                 "power.draw,clocks_throttle_reasons.active",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            row = {"t": time.time(), "gpu": {}}
            for line in out.splitlines():
                p = [x.strip() for x in line.split(",")]
                row["gpu"][int(p[0])] = {
                    "sm_mhz": int(p[1]), "temp": int(p[2]), "util": int(p[3]),
                    "power": float(p[4]), "throttle": p[5],
                }
            _samples.append(row)
        except Exception:
            pass
        _stop.wait(0.5)


def window(t0, t1):
    """Aggregate the thermal samples inside a measurement window."""
    sel = [s for s in _samples if t0 <= s["t"] <= t1]
    if not sel:
        return {}
    idxs = sorted(sel[0]["gpu"])
    agg = {}
    for i in idxs:
        vals = [s["gpu"][i] for s in sel if i in s["gpu"]]
        thr = [v["throttle"] for v in vals]
        agg[f"gpu{i}"] = {
            "sm_mhz_mean": round(statistics.mean(v["sm_mhz"] for v in vals), 1),
            "sm_mhz_min": min(v["sm_mhz"] for v in vals),
            "temp_mean": round(statistics.mean(v["temp"] for v in vals), 1),
            "temp_max": max(v["temp"] for v in vals),
            "power_mean": round(statistics.mean(v["power"] for v in vals), 1),
            # any non-zero throttle mask during the window invalidates the point
            "throttled_frac": round(sum(1 for x in thr if x != "0x0000000000000000")
                                    / len(thr), 3),
        }
    return agg


DUAL = os.environ.get("NF_DUAL", "0") == "1"


def one(ntok):
    r = requests.post(
        f"{BASE}/generate",
        json={"text": PROMPT,
              "sampling_params": {"max_new_tokens": ntok, "temperature": 0.0,
                                  "ignore_eos": True}},
        timeout=1800)
    d = r.json()
    m = d["meta_info"]
    return (m["e2e_latency"], m["completion_tokens"], m.get("spec_accept_length"),
            d["text"])


def one_dual(ntok):
    """Two concurrent identical streams; returns aggregate wall-clock rate inputs."""
    res = {}

    def run(j):
        res[j] = one(ntok)

    ths = [threading.Thread(target=run, args=(j,)) for j in range(2)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    ntot = sum(res[j][1] for j in range(2))
    acc = statistics.median([res[j][2] for j in range(2)])
    return wall, ntot, acc, res[0][3]


def measure(ntok):
    """Uniform interface: (elapsed, tokens, accept, text)."""
    if DUAL:
        return one_dual(ntok)
    return one(ntok)


th = threading.Thread(target=sampler, daemon=True)
th.start()

blocks = []
print(f"# {LABEL}: {NBLOCKS} identical blocks, class={CLS}, "
      f"window {N_SHORT}->{N_LONG}", flush=True)
print(f"{'blk':>3} {'tok/s':>8} {'accept':>7} {'ntok':>6} {'hash':>8} "
      f"{'g0MHz':>6} {'g0C':>4} {'g1MHz':>6} {'g1C':>4} {'g2MHz':>6} {'g2C':>4} "
      f"{'thr':>5} {'elapsed':>8}", flush=True)

T0 = time.time()
for b in range(NBLOCKS):
    t0 = time.time()
    e1, n1, a1, x1 = measure(N_SHORT)
    e2, n2, a2, x2 = measure(N_LONG)
    t1 = time.time()
    rate = (n2 - n1) / (e2 - e1)
    w = window(t0, t1)
    # content fingerprint: if two points produce different text, the tok/s
    # comparison between them is contaminated by content variance (r=0.90)
    h = format(abs(hash(x2)) % (16 ** 8), "08x")
    thr = max(w[g]["throttled_frac"] for g in w) if w else 0.0
    rec = {"blk": b, "rate": rate, "accept": a2, "n_long": n2,
           "text_hash": h, "text_len_chars": len(x2),
           "t_rel": round(t0 - T0, 1), "thermal": w}
    blocks.append(rec)
    # append incrementally: a killed run must never lose its collected points
    with open(f"/spinning/r3val/logs/nf_{LABEL}.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"{b:3d} {rate:8.2f} {a2:7.3f} {n2:6d} {h:>8} "
          + " ".join(f"{w[f'gpu{i}']['sm_mhz_mean']:6.0f} {w[f'gpu{i}']['temp_mean']:4.0f}"
                     for i in range(3) if f"gpu{i}" in w)
          + f" {thr:5.2f} {t0-T0:8.1f}", flush=True)

out = {"label": LABEL, "cls": CLS, "n_short": N_SHORT, "n_long": N_LONG,
       "blocks": blocks}
path = f"/spinning/r3val/logs/nf_{LABEL}.json"
with open(path, "w") as f:
    json.dump(out, f, indent=1)

r = [b["rate"] for b in blocks]
print(f"\nALL blocks: n={len(r)} mean={statistics.mean(r):.2f} "
      f"sd={statistics.stdev(r):.2f} ({100*statistics.stdev(r)/statistics.mean(r):.2f}%) "
      f"min={min(r):.2f} max={max(r):.2f} spread={100*(max(r)-min(r))/statistics.mean(r):.1f}%")
half = len(r) // 2
r2 = r[half:]
if len(r2) > 1:
    print(f"PLATEAU (2nd half): n={len(r2)} mean={statistics.mean(r2):.2f} "
          f"sd={statistics.stdev(r2):.2f} "
          f"({100*statistics.stdev(r2)/statistics.mean(r2):.2f}%) "
          f"min={min(r2):.2f} max={max(r2):.2f} "
          f"spread={100*(max(r2)-min(r2))/statistics.mean(r2):.1f}%")
hashes = set(b["text_hash"] for b in blocks)
print(f"distinct output texts: {len(hashes)} (1 = content variance fully controlled)")

# --- the two-axis decomposition -------------------------------------------
# tok/s = (verify rounds per second) x (accepted tokens per verify round)
#            ^ hardware/pipeline axis      ^ content/speculation axis
# The plateau A-vs-A control showed r(tok/s, accept) = 0.98, i.e. almost all
# residual end-to-end scatter is CONTENT variance. Dividing it out isolates
# the hardware axis, where the noise floor is ~5x tighter.
acc = [b["accept"] for b in blocks]
rr = [b["rate"] / b["accept"] for b in blocks]
for nm, v in (("accept (content axis)", acc), ("round_rate (hw axis)", rr)):
    if len(v) > 1:
        print(f"{nm:24s} mean={statistics.mean(v):7.3f} sd={statistics.stdev(v):6.3f} "
              f"({100*statistics.stdev(v)/statistics.mean(v):5.2f}%) "
              f"median={statistics.median(v):7.3f}")
print(f"raw -> {path}  (+ .jsonl incremental)")
