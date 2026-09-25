#!/usr/bin/env python3
"""Metal probe for the 27B blocked uneven-DCP LSE merge (RC7c, 25.09.).

The crash it proves gone: boot weg2rc7_186c022f80 D TP0 (5090) OOMed in the
a2a LSE merge (`Tried to allocate 144.00 MiB` = recv (3*12, 4096, 256) fp32)
on a PREFIX-BEARING D forward at full chunk width. The same forward is reached
two ways, and this probe drives both in ONE boot (X ceiling 12288):

  S  multi-turn follow-up: round 1 lays an ~8k-token context (with needle 1)
     down (the front picks P or D), round 2 repeats it verbatim plus ~3.9k NEW
     tokens (needle 2). D holds round 1 in its radix tree, so round 2's
     uncached extent is ~3.9k <= 4096: D-direct, ONE extend of ~3.9k rows
     over an ~8k prefix -- the X=4096 form of the crash.
  L  the X-ceiling form: the same, with min(--large-new, X_live - 300) new
     tokens (~8-12k): D-direct over several 4096-row chunks, every one of them
     prefix-bearing.

Per case it reads D's own lines for its rid and the window of the case:
  - 'WEG2 X-GATE rid=R uncached=U X=.. verdict=admit'           (D-direct, U)
  - '#969 EXTENT ... (start, end, prefix, extend)' on TP0        (the widths)
  - 'Prefill rank batch, #new-token: T, ...dcp.all_gather ../Nx' on TP0:
      prefix-bearing with T > block_tokens -> N = 16 * (2 + ceil(T/block))
      (kv gather + q gather + one LSE gather per block, 16 full-attn layers);
      T <= block -> 48 (unchanged); no prefix -> 16 (unchanged)
  - 'DCP-MERGE-BLOCKED ... T=T blocks=B block_tokens=W'          (B = ceil(T/W))
  - '[vram-peak] ... card free F of .. -> transient headroom used = H GiB' (TP0)
  - no 'OutOfMemoryError', no 'Traceback' in the window
and checks both needles in the answer (the prefix needle is read THROUGH the
blocked merge; a wrong merge loses it).

Output: 'DCPMERGE CASE=<S|L> VERDIKT OK|FAIL ...' plus 'DCPMERGE ev:' lines and
'DCPMERGE GESAMT OK|FAIL'. Exit code 0 only when every case is OK.
"""
import argparse
import json
import math
import random
import re
import sys
import time
import urllib.request

CHARS_PER_TOKEN = 3.0  # the front's estimator (weg2/front.py CHARS_PER_TOKEN)
FULL_ATTN_LAYERS = 16  # Qwen3.8-27B: 64 layers, full_attention_interval 4


def state(front):
    with urllib.request.urlopen(front + "/weg2/state", timeout=10) as r:
        return json.loads(r.read())


def x_live(front):
    try:
        v = state(front).get("x_tokens")
        return int(v) if isinstance(v, (int, float)) and v > 0 else None
    except Exception:  # noqa: BLE001
        return None


def chat(front, model, messages, max_tokens, timeout=1200):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(front + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return {"wall": time.time() - t0, "text": resp["choices"][0]["message"]["content"] or "",
            "usage": resp.get("usage", {})}


def filler(target_tokens, rnd, tag):
    places = ["Hang", "Kai", "Markt", "Bahnhof", "Deich", "Wald", "Hafen", "Platz"]
    lines, i = [], 0
    while len(" ".join(lines)) / CHARS_PER_TOKEN < target_tokens:
        i += 1
        lines.append(f"{tag} {i}: Das Lager {rnd.randint(10000, 99999)} liegt am {rnd.choice(places)} "
                     f"{rnd.randint(1, 999)} und fuehrt {rnd.randint(2, 90)} Kisten.")
    return lines


def code(rnd):
    return f"{rnd.randint(1000, 9999)}-{rnd.choice('BCDFGHKMNPRSTVWXZ')}{rnd.choice('BCDFGHKMNPRSTVWXZ')}"


def n_lines(path):
    with open(path, errors="replace") as f:
        return sum(1 for _ in f)


def lines_since(path, n0):
    with open(path, errors="replace") as f:
        return f.read().splitlines()[n0:]


def tp0(lines):
    return [ln for ln in lines if " TP0]" in ln]


def boot_width(dlog):
    """TP0's derived width from the boot line 'DCP-MERGE-BLOCK block_tokens=W
    source=... heads=[...]' of the TARGET backend (the first one on TP0)."""
    with open(dlog, errors="replace") as f:
        for ln in f:
            if " TP0]" in ln and "DCP-MERGE-BLOCK block_tokens=" in ln:
                m = re.search(r"DCP-MERGE-BLOCK block_tokens=(\d+)", ln)
                return int(m.group(1)) if m else None
    return None


def run_case(a, name, new_tokens):
    rnd = random.Random(f"{a.tag}-{name}-{time.time_ns()}")
    c1, c2 = code(rnd), code(rnd)
    ctx = filler(a.ctx_tokens, rnd, "Zeile")
    ctx.insert(len(ctx) // 3, f"Merke dir den ersten Kontrollcode: {c1}.")
    user1 = " ".join(ctx) + "\n\nBestaetige nur mit: gelesen."
    n0d, n0f = n_lines(a.dlog), n_lines(a.flog)
    r1 = chat(a.front, a.model, [{"role": "user", "content": user1}], 8)
    time.sleep(a.settle_s)
    new = filler(new_tokens, rnd, "Posten")
    new.insert(len(new) // 2, f"Merke dir den zweiten Kontrollcode: {c2}.")
    user2 = (" ".join(new) + "\n\nNenne den ersten und den zweiten Kontrollcode, "
             "sonst nichts.")
    msgs = [{"role": "user", "content": user1}, {"role": "assistant", "content": r1["text"]},
            {"role": "user", "content": user2}]
    n1d = n_lines(a.dlog)
    r2 = chat(a.front, a.model, msgs, 48)
    time.sleep(a.settle_s)
    dl = lines_since(a.dlog, n1d)
    dl_all = lines_since(a.dlog, n0d)
    fl = lines_since(a.flog, n0f)
    gates = [ln for ln in tp0(dl) if "WEG2 X-GATE" in ln]
    adm = [ln for ln in gates if "verdict=admit" in ln]
    unc = [int(m.group(1)) for ln in adm for m in [re.search(r"uncached=(\d+)", ln)] if m]
    ext = [ln for ln in tp0(dl) if "#969 EXTENT" in ln]
    pre = []  # (T, n_gathers) of prefix-bearing prefill rank rows on TP0
    for ln in tp0(dl):
        m = re.search(r"Prefill rank batch, #new-token: (\d+),.*dcp\.all_gather [0-9.]+/(\d+)x", ln)
        if m and int(m.group(2)) > FULL_ATTN_LAYERS:
            pre.append((int(m.group(1)), int(m.group(2))))
    blk = [ln for ln in tp0(dl) if "DCP-MERGE-BLOCKED" in ln]
    width = a.width if a.width is not None else boot_width(a.dlog)
    peaks = [ln for ln in tp0(dl_all) if "[vram-peak]" in ln]
    head = [float(m.group(1)) for ln in peaks for m in [re.search(r"headroom used = peak - allocated ([0-9.]+) GiB", ln)] if m]
    free = [float(m.group(1)) for ln in peaks for m in [re.search(r"card free ([0-9.]+) of", ln)] if m]
    boom = [ln for ln in dl_all if "OutOfMemoryError" in ln or "Traceback" in ln]
    # the pass rule: every prefix-bearing TP0 row wider than the width shows the
    # blocked gather count 16 * (2 + ceil(T / width)); rows <= width show 48
    bad_rows = []
    for T, n in pre:
        want = FULL_ATTN_LAYERS * (2 + math.ceil(T / width)) if width and T > width else 3 * FULL_ATTN_LAYERS
        if n != want:
            bad_rows.append((T, n, want))
    wide = [T for T, _ in pre if width and T > width]
    match = c1 in r2["text"] and c2 in r2["text"]
    ok = bool(adm and wide and not bad_rows and not boom and match and width)
    for ln in (gates[-2:] + ext[-4:] + blk[-4:]):
        print("DCPMERGE ev:", ln[:260])
    route = [ln for ln in fl if "WEG2-ROUTE" in ln][-2:]
    for ln in route:
        print("DCPMERGE ev:", ln[:260])
    print(f"DCPMERGE CASE={name} VERDIKT {'OK' if ok else 'FAIL'}: new_tokens~{new_tokens} "
          f"uncached={unc[-1:] or '-'} admit={len(adm)} block_tokens={width} "
          f"prefix_rows(T,gathers)={pre} wide={wide} bad={bad_rows} blocked_lines={len(blk)} "
          f"headroom_max_GiB={max(head) if head else '-'} card_free_min_GiB={min(free) if free else '-'} "
          f"oom_or_tb={len(boom)} needles={'MATCH' if match else 'MISS'} "
          f"r1_wall={r1['wall']:.1f}s r2_wall={r2['wall']:.1f}s usage2={r2['usage']}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--front", default="http://127.0.0.1:30030")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dlog", required=True, help="group D log of this boot")
    ap.add_argument("--flog", required=True, help="front log of this boot")
    ap.add_argument("--tag", default="dcpmerge")
    ap.add_argument("--cases", default="S,L")
    ap.add_argument("--ctx-tokens", type=int, default=8000)
    ap.add_argument("--small-new", type=int, default=3900)
    ap.add_argument("--large-new", type=int, default=9000)
    ap.add_argument("--width", type=int, default=None,
                    help="block width if the boot line is not in the log head (default: read it)")
    ap.add_argument("--settle-s", type=float, default=3.0)
    a = ap.parse_args()
    results = {}
    for name in [c.strip().upper() for c in a.cases.split(",") if c.strip()]:
        if name == "S":
            n = a.small_new
        else:
            x = x_live(a.front)
            n = min(a.large_new, (x - 300) if x else a.large_new)
            print(f"DCPMERGE CASE=L X_live={x} -> new_tokens {n}")
        try:
            results[name] = run_case(a, name, n)
        except Exception as e:  # noqa: BLE001
            print(f"DCPMERGE CASE={name} VERDIKT FAIL: {type(e).__name__}: {e}")
            results[name] = False
    ok = bool(results) and all(results.values())
    print(f"DCPMERGE GESAMT {'OK' if ok else 'FAIL'} {results}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
