#!/usr/bin/env python3
"""Build the window-10 per-arm table from the artifacts.

Reads only files the arms actually wrote. Anything it cannot find is printed as
"-" rather than guessed, so a gap in the table is a gap in the evidence.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

OUT = "/spinning/gpu-battery-results/2026-08-05_window10"

ARMS = [
    ("arm1_pin_record_bl", "pin 84fff442e1", "record", "barlink dev, gate registry EMPTY"),
    ("arm2_int_record_bl", "int 548f4cee5c", "record", "barlink dev, #517 on"),
    ("arm0_int_today_every32", "int 548f4cee5c", "today", "barlink dev, EVERY=32"),
    ("arm0_int_today_517on", "int 548f4cee5c", "today", "barlink dev, #517 on"),
    ("arm0_int_today_wdoff", "int 548f4cee5c", "today", "barlink dev, watchdog OFF"),
]

PT = re.compile(
    r"tick ([\d.]+) tok/s, accept ([\d.]+), ms/Verify ([\d.]+).*?client ([\d.]+) tok/s"
)


def s14_points(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        m = PT.search(line)
        if m:
            out.append(dict(tick=float(m.group(1)), accept=float(m.group(2)),
                            msv=float(m.group(3)), client=float(m.group(4))))
    return out


def bench(path):
    """Return {'narrative': (mean, cv), 'code': (...), 'pp': mean}."""
    if not os.path.exists(path):
        return {}
    txt = open(path, errors="replace").read()
    res = {}
    for name in ("narrative", "code"):
        m = re.search(
            r"=== summary \[" + name + r"\].*?decode_TPS\s+mean=\s*([\d.]+)\s+std=\s*[\d.]+\s+CV=\s*([\d.]+)%",
            txt, re.S)
        if m:
            res[name] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"=== summary \[prompt-processing\].*?PP tok/s\s+mean=\s*([\d.]+)", txt, re.S)
    if m:
        res["pp"] = float(m.group(1))
    return res


def pyspy(path, n=3):
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path, errors="replace"):
        m = re.match(r"\s+(\d+)/(\d+)\s+(\S+) \((.*):(\d+)\)", line)
        if m:
            rows.append(f"{m.group(1)}/{m.group(2)} {m.group(3)} ({os.path.basename(m.group(4))}:{m.group(5)})")
        if len(rows) >= n:
            break
    return rows


def jsonl_point(arm):
    """The granular record of the MEASURED draw: ms/round and the within-window
    tick spread, which is what makes the bs=1 tick instrument what it is."""
    p = f"{OUT}/raw/decode_punkte.jsonl"
    if not os.path.exists(p):
        return None
    hit = None
    for line in open(p):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("arm") == arm:
            hit = d
    return hit


def tokens(arm):
    p = f"{OUT}/plan_{arm}.txt"
    if not os.path.exists(p):
        return "-"
    m = re.search(r"max_total_num_tokens=(\d+)", open(p, errors="replace").read())
    return m.group(1) if m else "-"


def main():
    print("# Window 10 -- per-arm results\n")
    print("Record reference (#424 int8_decode, barlink BAR1, pin 1960957e3b):")
    print("  s14 bs=1 tick 126.8 / client 120.5 tok/s, ms/Verify 30.37, accept 3.85")
    print("  s14 bs=1 A-vs-A floor IN ITS OWN BOOT: 104.46 / 118.65 / 106.07 (33.9 %)")
    print("  bench.sh narrative decode_TPS 86.46, code 112.18, PP 1638.99")
    print("  max_total_num_tokens 431360, ctx 131072, mrr 16\n")

    hdr = ("| arm | tree | config | s14 tick | s14 client | ms/Verify | accept | "
           "floor spread | bench narr | bench code | PP | tokens |")
    print(hdr)
    print("|" + "---|" * 12)
    for arm, tree, cfg, note in ARMS:
        pts = s14_points(f"{OUT}/messen_{arm}.log")
        fl = s14_points(f"{OUT}/floor_{arm}.log")
        b = bench(f"{OUT}/bench_{arm}.txt")
        if pts:
            p = pts[-1]
            tick, cl, msv, acc = f"{p['tick']:.1f}", f"{p['client']:.1f}", f"{p['msv']:.2f}", f"{p['accept']:.2f}"
        else:
            tick = cl = msv = acc = "-"
        if fl:
            v = [x["tick"] for x in fl]
            spread = f"{min(v):.1f}-{max(v):.1f} ({(max(v)-min(v))/max(min(v),1e-9)*100:.1f} %)"
        else:
            spread = "-"
        nb = f"{b['narrative'][0]:.2f} (CV {b['narrative'][1]:.1f} %)" if "narrative" in b else "-"
        cb = f"{b['code'][0]:.2f} (CV {b['code'][1]:.1f} %)" if "code" in b else "-"
        pp = f"{b['pp']:.0f}" if "pp" in b else "-"
        print(f"| {arm} | {tree} | {cfg} | {tick} | {cl} | {msv} | {acc} | {spread} | {nb} | {cb} | {pp} | {tokens(arm)} |")

    print("\n## py-spy leaf census (20 samples of the TP0 scheduler thread under a bs=1 load)\n")
    for arm, _, _, note in ARMS:
        rows = pyspy(f"{OUT}/pyspy_{arm}.txt")
        print(f"* **{arm}** ({note})")
        if rows:
            for r in rows:
                print(f"    * {r}")
        else:
            print("    * -")
    return 0


if __name__ == "__main__":
    sys.exit(main())
