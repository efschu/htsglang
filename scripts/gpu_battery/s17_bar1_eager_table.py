#!/usr/bin/env python3
"""#366 eager format x transport table.

Both transports are measured in this window, in the EAGER regime
(--disable-cuda-graph), because bar1 + CUDA graphs + NEXTN cold-captures for
>35 min on the 27B FP8 vehicle (the 5090 lacks tuned W8A8 block-fp8 configs, so
every GEMM shape cold-autotunes -- the #255/#368 gap). Eager skips capture and
puts bar1 and NCCL in the SAME regime, so the bar1-vs-NCCL question is answered
honestly. These numbers are NOT comparable to #354's graph-mode NCCL baseline,
and every cell is labelled 'eager'.

Arm jsonl is named ``<fmt>_<phase>_<transport>`` (e.g. fp8_prefopt_bar1).
Prefill points come from the *prefopt* arms, decode from the *auto* arms -- the
#354 phase recipe. Two levers are reported separately:
  transport = bar1 vs NCCL at fixed format
  format    = INT8 vs FP8 at fixed transport

Noise floors are CARRIED from #354's methodology (prefill s=1 2.71 %,
prefill s>=2 3.18 %, decode 2.72 %); a delta under the row floor prints as
"within noise". They are throughput-measurement floors and regime-independent
to first order; a same-arm A-vs-A was not re-run here (boot budget went to
covering both transports x both formats).
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Optional, Tuple

NOISE_PCT = {"prefill_s1": 2.71, "prefill_s8": 3.18,
             "decode_bs1": 2.72, "decode_bs8": 2.72}
POINTS = ("prefill_s1", "prefill_s8", "decode_bs1", "decode_bs8")
POINT_ARM = {"prefill_s1": "prefopt", "prefill_s8": "prefopt",
             "decode_bs1": "auto", "decode_bs8": "auto"}


def _read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load(out_dir: str) -> Dict[Tuple[str, str, str], Optional[float]]:
    pre = _read_jsonl(os.path.join(out_dir, "punkte.jsonl"))
    dec = _read_jsonl(os.path.join(out_dir, "decode_punkte.jsonl"))
    out: Dict[Tuple[str, str, str], Optional[float]] = {}
    for fmt in ("fp8", "int8"):
        for tr in ("bar1", "nccl"):
            for point in POINTS:
                arm = f"{fmt}_{POINT_ARM[point]}_{tr}"
                if point.startswith("prefill"):
                    n = int(point.split("_s")[1])
                    rows = [r for r in pre if r.get("arm") == arm
                            and r.get("sessions") == n]
                    v = rows[-1]["prefill"]["prefill_tok_s"] if rows else None
                else:
                    b = int(point.split("_bs")[1])
                    rows = [r for r in dec if r.get("arm") == arm
                            and r.get("bs") == b]
                    # Primary is the scheduler tick rate, but under eager +
                    # docker-logs buffering the server-log tick parse yields 0
                    # ticks; the client-side token rate (klient_tok_s) is the
                    # reliable decode metric here and is used for every arm so
                    # the column is consistent.
                    if rows:
                        v = rows[-1].get("tick_gen_tok_s_median")
                        if v is None:
                            v = rows[-1].get("klient_tok_s")
                    else:
                        v = None
                out[(fmt, tr, point)] = round(v, 1) if isinstance(
                    v, (int, float)) else None
    return out


def _delta(new: Optional[float], ref: Optional[float], point: str) -> str:
    if new is None or ref is None or not ref:
        return "-"
    pct = (new - ref) / ref * 100.0
    if abs(pct) < NOISE_PCT[point]:
        return "within noise"
    return f"{pct:+.1f}%"


def build(out_dir: str) -> dict:
    v = load(out_dir)
    t = {"regime": "eager (--disable-cuda-graph)", "points": list(POINTS),
         "noise_pct": NOISE_PCT, "cells": {},
         "transport_effect": {}, "format_effect": {}}
    for fmt in ("fp8", "int8"):
        for tr in ("bar1", "nccl"):
            for p in POINTS:
                t["cells"][f"{fmt}|{tr}|{p}"] = v[(fmt, tr, p)]
    for fmt in ("fp8", "int8"):
        for p in POINTS:  # transport lever: bar1 vs nccl
            t["transport_effect"][f"{fmt}|{p}"] = _delta(
                v[(fmt, "bar1", p)], v[(fmt, "nccl", p)], p)
    for tr in ("bar1", "nccl"):
        for p in POINTS:  # format lever: int8 vs fp8
            t["format_effect"][f"{tr}|{p}"] = _delta(
                v[("int8", tr, p)], v[("fp8", tr, p)], p)
    return t


def render(t: dict) -> str:
    L = ["#366 Qwen3.6-27B TP=3 uneven (5090 + 2x 3080), phase-optimal points, "
         "tok/s", f"REGIME: {t['regime']} -- NOT comparable to #354 graph-mode "
         "NCCL. Both transports measured this window.",
         "Noise floors carried from #354: prefill s=1 2.71%, s>=2 3.18%, "
         "decode 2.72%.", ""]
    head = ("point".ljust(12) + "FP8 NCCL".rjust(10) + "FP8 bar1".rjust(10)
            + "  d(tr)".ljust(15) + "INT8 NCCL".rjust(10) + "INT8 bar1".rjust(10)
            + "  d(tr)")
    L += [head, "-" * len(head)]
    for p in t["points"]:
        def c(fmt, tr):
            x = t["cells"][f"{fmt}|{tr}|{p}"]
            return (f"{x:.1f}" if x is not None else "-").rjust(10)
        L.append(p.ljust(12) + c("fp8", "nccl") + c("fp8", "bar1")
                 + ("  " + t["transport_effect"][f"fp8|{p}"]).ljust(15)
                 + c("int8", "nccl") + c("int8", "bar1")
                 + "  " + t["transport_effect"][f"int8|{p}"])
    L += ["", "LEVER 1 -- transport (bar1 vs NCCL, format fixed), EAGER:"]
    for fmt in ("fp8", "int8"):
        L.append("  " + fmt.upper().ljust(5) + "  ".join(
            f"{p}: {t['transport_effect'][f'{fmt}|{p}']}" for p in t["points"]))
    L += ["", "LEVER 2 -- format (INT8 vs FP8, transport fixed), EAGER:"]
    for tr in ("nccl", "bar1"):
        L.append("  " + tr.upper().ljust(5) + "  ".join(
            f"{p}: {t['format_effect'][f'{tr}|{p}']}" for p in t["points"]))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    t = build(a.out_dir)
    with open(os.path.join(a.out_dir, "table_366_eager.json"), "w") as fh:
        json.dump(t, fh, indent=1)
    txt = render(t)
    with open(os.path.join(a.out_dir, "tabelle_366.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
