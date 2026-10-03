#!/usr/bin/env python3
"""DUAL-TP3PP3: read one dual boot's logs and answer the questions that decide
whether the double layout RUNS.

  eval_dual_boot.py <evidence-dir-or-prefix>   (the boot_weg2_<tag>_... files)

Answers, each with the lines it counted:
1. Did the dual path engage? (launcher WEG2-DUAL*, front WEG2 DUAL-LAYOUT on,
   P stage assembled, union OWNER/PEER + bound bytes, D ratios published)
2. Were there flips? (WEG2-FLIP begin) -- must be 0.
3. Did P prefill WHILE D decoded? Overlap of PP0 prefill seconds with TP0
   decode seconds (1-s buckets), and D's per-round gpu-ms in buckets with and
   without P activity (the price D pays; risk-1 predicted ~2x at P 50 %).
4. Errors: Traceback / REFUSED / W-codes / CUDA errors, first lines only.
"""
from __future__ import annotations

import collections
import glob
import os
import re
import statistics
import sys

TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _files(arg):
    if os.path.isdir(arg):
        c = sorted(glob.glob(os.path.join(arg, "boot_weg2_*.front.log")), key=os.path.getmtime)
        if not c:
            sys.exit(f"no boot_weg2_*.front.log in {arg}")
        arg = c[-1][: -len(".front.log")]
    return {k: f"{arg}.{k}.log" for k in ("front", "P", "D")}


def _sec(line):
    m = TS.match(line)
    if not m:
        return None
    import datetime

    return int(datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp())


def _grep(path, pats, limit=6):
    out = collections.defaultdict(list)
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            for name, rx in pats.items():
                if rx.search(line):
                    if len(out[name]) < limit:
                        out[name].append(line.rstrip()[:260])
                    out[name + "#"].append(1)
    return out


# D rounds are placed by their own ``t:`` field (epoch seconds, taken as the
# round's END; the round spans gpu-ms before it), never by the log line's stamp:
# the D log flushes 'Decode rank batch' lines in bursts (boot dual1i
# ...10010932: 6453 rounds over 457 s of t landed in 109 log seconds, lag p50
# 1.4 s / p99 19 s), which made the old 1-s buckets report "5 s with BOTH" for
# ~55 s of real overlap. P activity: per rank, the chunk window
# [WEG2-VRAM-PEAK t_unix_ms - 'Prefill rank batch' gpu-ms, t_unix_ms] (paired by
# TIME, see pair_peaks_with_batches: a peak without a batch line is dropped and
# counted, never shifts the later pairs as zip() did); without those lines, the old PP0 'Prefill batch' log seconds.
_VP = re.compile(r"PP(\d)\] WEG2-VRAM-PEAK rank=\d+ phase=chunk rows=(\d+) .*?t_unix_ms=(\d+)")
_RB = re.compile(r"PP(\d)\] Prefill rank batch, #new-token: (\d+), .*gpu-ms: ([\d.]+)")
_DR = re.compile(r"TP0\] Decode rank batch, rank: 0, #round: \d+, t: ([\d.]+), .*gpu-ms: ([\d.]+)")


#: A 'Prefill rank batch' line is stamped to the whole second and follows its
#: chunk's WEG2-VRAM-PEAK line by milliseconds; further back than this it is a
#: different chunk's line (ITEM 200, 03.10.).
MAX_PAIR_GAP_S = 10.0


def pair_peaks_with_batches(peaks, batches, max_gap=MAX_PAIR_GAP_S):
    """Pair VRAM-PEAK chunk ends with 'Prefill rank batch' lines BY TIME.

    ``peaks``: chunk end times (epoch s). ``batches``: (log stamp epoch s,
    gpu-ms). For each peak, in time order, take the LAST not-yet-used batch whose
    stamp is <= the peak end and at most ``max_gap`` s before it. A peak with no
    such batch is dropped (the old zip() paired the n-th peak with the n-th
    batch, so ONE missing line shifted every later pair by a whole chunk).
    Returns (list of (end, gpu_ms), n_dropped_peaks)."""
    import bisect

    bs = sorted(range(len(batches)), key=lambda i: (batches[i][0], i))
    stamps = [batches[i][0] for i in bs]
    used = [False] * len(bs)
    out, dropped = [], 0
    for end in sorted(peaks):
        k = bisect.bisect_right(stamps, end) - 1
        while k >= 0 and used[k]:
            k -= 1
        if k < 0 or end - stamps[k] > max_gap:
            dropped += 1
            continue
        used[k] = True
        out.append((end, batches[bs[k]][1]))
    return out, dropped


def _union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _cover(a, b, merged, starts):
    import bisect

    k = max(0, bisect.bisect_right(starts, b) - 1)
    s = 0.0
    while k >= 0 and k < len(merged) and merged[k][1] >= a:
        s += max(0.0, min(b, merged[k][1]) - max(a, merged[k][0]))
        k -= 1
    return s


def overlap(p_path, d_path):
    vp, rb, batch_sec, tokens = collections.defaultdict(list), collections.defaultdict(list), set(), 0
    if os.path.exists(p_path):
        with open(p_path, errors="replace") as f:
            for line in f:
                m = _VP.search(line)
                if m:
                    vp[int(m.group(1))].append(int(m.group(3)) / 1000.0)
                    continue
                m = _RB.search(line)
                if m:
                    st = _sec(line)
                    if st is not None:
                        rb[int(m.group(1))].append((float(st), float(m.group(3))))
                    continue
                if "PP0] Prefill batch" in line:
                    s = _sec(line)
                    t = re.search(r"#new-token: (\d+)", line)
                    if s is not None and t:
                        batch_sec.add(s)
                        tokens += int(t.group(1))
    p_iv, dropped_peaks, paired_peaks = [], 0, 0
    for r in vp:
        pairs, dr = pair_peaks_with_batches(vp[r], rb.get(r, []))
        dropped_peaks += dr
        paired_peaks += len(pairs)
        for end, g in pairs:
            p_iv.append((end - g / 1000.0, end))
    if p_iv:
        p_source = "chunk-windows"
    else:
        p_source = "batch-line-seconds"
        p_iv = [(float(s), float(s) + 1.0) for s in batch_sec]
    merged = _union(p_iv)
    starts = [a for a, _ in merged]
    rounds = []
    if os.path.exists(d_path):
        with open(d_path, errors="replace") as f:
            for line in f:
                m = _DR.search(line)
                if m:
                    t, g = float(m.group(1)), float(m.group(2))
                    rounds.append((t - g / 1000.0, t, g))
    d_merged = _union([(a, b) for a, b, _ in rounds])
    both = sum(_cover(a, b, merged, starts) for a, b in d_merged)
    during, idle = [], []
    for a, b, g in rounds:
        if _cover(a, b, merged, starts) > 0:
            during.append(g)
        elif _cover(a - 1.0, b + 1.0, merged, starts) == 0:
            idle.append(g)
    return {"p_source": p_source, "d_source": "t-field", "p_tokens": tokens,
            "p_s": sum(b - a for a, b in merged), "d_s": sum(b - a for a, b in d_merged),
            "both_s": both, "during": during, "idle": idle,
            "peaks_paired": paired_peaks, "peaks_dropped": dropped_peaks}


def main():
    fs = _files(sys.argv[1])
    print("files:", fs)
    pats = {
        "dual_launch": re.compile(r"WEG2-DUAL"),
        "front_dual": re.compile(r"WEG2 DUAL-LAYOUT on"),
        "p_stage": re.compile(r"DUAL-TP3PP3 P stage \d+/\d+ assembled"),
        "p_bound": re.compile(r"DUAL-TP3PP3 P: shared part bound"),
        "union_owner": re.compile(r"WEG2-UNION OWNER"),
        "union_peer": re.compile(r"WEG2-UNION PEER"),
        "d_ratios": re.compile(r"WEG2-UNION D ratios published"),
        "flip": re.compile(r"WEG2-FLIP begin"),
        "error": re.compile(r"Traceback|REFUSED:|CUDA error|out of memory|OutOfMemory|Weg2Stop|DualShareError|UnionShareError|STOP "),
        "mps": re.compile(r"MPS"),
        "duty": re.compile(r"duty throttle armed"),
        # unified KV per card (dual1g)
        "p_kv_join": re.compile(r"DUAL-TP3PP3 P-KV JOIN"),
        "p_kv_grant": re.compile(r"DUAL-TP3PP3 P-KV PP0 GRANT"),
        "p_kv_wait": re.compile(r"DUAL-TP3PP3 P-KV (PP0 )?WAIT"),
        "p_kv_mapped": re.compile(r"DUAL-TP3PP3 P-KV MAPPED-BY-GRANT"),
        "p_kv_release": re.compile(r"DUAL-TP3PP3 P-KV RELEASE"),
        "d_kv_join": re.compile(r"DUAL-TP3PP3 D-KV JOIN"),
        "d_kv_grow": re.compile(r"DUAL-TP3PP3 D-KV GROW"),
        "d_kv_shrink": re.compile(r"DUAL-TP3PP3 D-KV SHRINK"),
        "d_kv_wait": re.compile(r"DUAL-TP3PP3 D-KV GROUP-WAIT"),
        "p_pause": re.compile(r"WEG2 DUAL P-PAUSE rid"),
        "p_paused": re.compile(r"WEG2 DUAL P-PAUSED rid"),
        "terminate": re.compile(r"terminate called"),
    }
    for k, p in fs.items():
        g = _grep(p, pats)
        print(f"\n== {k}: " + ", ".join(f"{n}={len(g.get(n + '#', []))}" for n in pats))
        for n in ("dual_launch", "front_dual", "p_stage", "p_bound", "union_owner", "union_peer", "d_ratios",
                  "flip", "error", "duty", "p_kv_join", "p_kv_grant", "p_kv_wait", "p_kv_release", "d_kv_join",
                  "d_kv_grow", "d_kv_shrink", "d_kv_wait", "p_pause", "p_paused", "terminate"):
            for line in g.get(n, [])[:3]:
                print(f"   {n}: {line}")
    o = overlap(fs["P"], fs["D"])
    print(f"\n== overlap (P: {o['p_source']}, D: {o['d_source']}): {o['p_s']:.0f} s with P prefill "
          f"({o['p_tokens']} tokens), {o['d_s']:.0f} s with D decode, {o['both_s']:.0f} s with BOTH")
    if o["p_source"] == "chunk-windows":
        print(f"   chunk windows: {o['peaks_paired']} VRAM-PEAK ends paired with a 'Prefill rank batch' line by time, "
              f"{o['peaks_dropped']} unmatched peaks dropped")
    for name, vals in (("D round gpu-ms while P prefills", o["during"]), ("D round gpu-ms, P idle (+-1 s)", o["idle"])):
        if vals:
            vals = sorted(vals)
            print(f"   {name}: n={len(vals)} p50={statistics.median(vals):.1f} "
                  f"p90={vals[int(0.9 * (len(vals) - 1))]:.1f}")
        else:
            print(f"   {name}: n=0")


if __name__ == "__main__":
    main()
