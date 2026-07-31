#!/usr/bin/env python3
"""s17 -- the #366 decision table: format x transport, reported separately.

#354 measured Qwen3.6-27B at its phase optimum over NCCL and #366 measures the
same four points over the barlink BAR1 direct path. Two levers moved between
those two studies -- the checkpoint FORMAT (FP8 vs INT8-W8A8) and the collective
TRANSPORT (NCCL vs bar1) -- and the interesting question is what each one is
worth on its own. A table that only shows the diagonal (FP8+NCCL vs INT8+bar1)
cannot answer that, so this one prints the full 2x2 per point and then names
the two marginal effects explicitly.

The NCCL column is QUOTED from #354, never re-measured: same rig, same recipe,
same points, and its noise floors are reused rather than re-derived
(prefill s=1 2.71 %, prefill s>=2 3.18 %, decode 2.72 %). A delta smaller than
the floor of its row is reported as "within noise" and not as a number, because
below the floor the sign is not evidence.

Reads the JSONL that s12_prefill_kurve.py and s14_decode_punkt.py wrote for the
bar1 arms; prefill comes from the concentrated boots and decode from the
VRAM-auto boots, which is the #354 phase recipe.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional, Tuple

#: #354, tabelle_354.txt. The phase-optimal column: prefill from the
#: concentrated boot, decode from the VRAM-auto boot.
NCCL_BASELINE: Dict[Tuple[str, str], float] = {
    ("fp8", "prefill_s1"): 1540.3,
    ("fp8", "prefill_s8"): 1322.5,
    ("fp8", "decode_bs1"): 122.2,
    ("fp8", "decode_bs8"): 447.3,
    ("int8", "prefill_s1"): 1787.5,
    ("int8", "prefill_s8"): 1483.6,
    ("int8", "decode_bs1"): 112.0,
    ("int8", "decode_bs8"): 426.1,
}

#: Reused from #354, not re-derived.
NOISE_PCT = {
    "prefill_s1": 2.71,
    "prefill_s8": 3.18,
    "decode_bs1": 2.72,
    "decode_bs8": 2.72,
}

POINTS = ("prefill_s1", "prefill_s8", "decode_bs1", "decode_bs8")

#: Which boot each point is read from (the phase recipe).
POINT_ARM = {
    "prefill_s1": "prefopt",
    "prefill_s8": "prefopt",
    "decode_bs1": "auto",
    "decode_bs8": "auto",
}


def _read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load_bar1(out_dir: str) -> Dict[Tuple[str, str], Optional[float]]:
    """The measured bar1 numbers, keyed like the baseline."""
    pre = _read_jsonl(os.path.join(out_dir, "punkte.jsonl"))
    dec = _read_jsonl(os.path.join(out_dir, "decode_punkte.jsonl"))
    out: Dict[Tuple[str, str], Optional[float]] = {}
    for fmt in ("fp8", "int8"):
        for point in POINTS:
            arm = f"{fmt}_{POINT_ARM[point]}"
            if point.startswith("prefill"):
                n = int(point.split("_s")[1])
                rows = [r for r in pre if r.get("arm") == arm
                        and r.get("sessions") == n]
                val = (rows[-1]["prefill"]["prefill_tok_s"] if rows else None)
            else:
                b = int(point.split("_bs")[1])
                rows = [r for r in dec if r.get("arm") == arm
                        and r.get("bs") == b]
                val = rows[-1].get("tick_gen_tok_s_median") if rows else None
            out[(fmt, point)] = round(val, 1) if isinstance(val, (int, float)) \
                else None
    return out


def _delta(new: Optional[float], ref: Optional[float],
           point: str) -> str:
    """Percent change, or 'within noise' when it does not clear the floor."""
    if new is None or ref is None or not ref:
        return "-"
    pct = (new - ref) / ref * 100.0
    if abs(pct) < NOISE_PCT[point]:
        return "within noise"
    return f"{pct:+.1f}%"


def build(out_dir: str) -> dict:
    bar1 = load_bar1(out_dir)
    table = {
        "points": list(POINTS),
        "noise_pct": NOISE_PCT,
        "nccl_quoted_from": "#354 tabelle_354.txt (phase-optimal column)",
        "cells": {},
        "transport_effect": {},
        "format_effect": {},
    }
    for fmt in ("fp8", "int8"):
        for point in POINTS:
            table["cells"][f"{fmt}|{point}|nccl"] = NCCL_BASELINE[(fmt, point)]
            table["cells"][f"{fmt}|{point}|bar1"] = bar1[(fmt, point)]

    # Lever 1: transport, held format constant.
    for fmt in ("fp8", "int8"):
        for point in POINTS:
            table["transport_effect"][f"{fmt}|{point}"] = _delta(
                bar1[(fmt, point)], NCCL_BASELINE[(fmt, point)], point)
    # Lever 2: format, held transport constant.
    for transport in ("nccl", "bar1"):
        src = NCCL_BASELINE if transport == "nccl" else bar1
        for point in POINTS:
            table["format_effect"][f"{transport}|{point}"] = _delta(
                src[("int8", point)], src[("fp8", point)], point)
    return table


def render(t: dict) -> str:
    L = []
    L.append("#366 Qwen3.6-27B TP=3 uneven (5090 + 2x 3080), phase-optimal "
             "points, tok/s")
    L.append("NCCL column quoted from #354; bar1 column measured on the PVE "
             "host (CT999 cannot open /dev/dmabuf_holder).")
    L.append("Noise floors reused from #354: prefill s=1 2.71%, s>=2 3.18%, "
             "decode 2.72%.")
    L.append("")
    head = ("point".ljust(13) + "FP8 NCCL".rjust(10) + "FP8 bar1".rjust(10)
            + "  d(transport)".ljust(16)
            + "INT8 NCCL".rjust(11) + "INT8 bar1".rjust(11)
            + "  d(transport)")
    L.append(head)
    L.append("-" * len(head))
    for p in t["points"]:
        def cell(f, tr):
            v = t["cells"][f"{f}|{p}|{tr}"]
            return f"{v:.1f}".rjust(10 if f == "fp8" else 11) if v is not None \
                else "-".rjust(10 if f == "fp8" else 11)
        L.append(p.ljust(13) + cell("fp8", "nccl") + cell("fp8", "bar1")
                 + ("  " + t["transport_effect"][f"fp8|{p}"]).ljust(16)
                 + cell("int8", "nccl") + cell("int8", "bar1")
                 + "  " + t["transport_effect"][f"int8|{p}"])
    L.append("")
    L.append("LEVER 1 -- transport (format held constant), bar1 vs NCCL:")
    for f in ("fp8", "int8"):
        L.append("  " + f.upper().ljust(5)
                 + "  ".join(f"{p}: {t['transport_effect'][f'{f}|{p}']}"
                             for p in t["points"]))
    L.append("")
    L.append("LEVER 2 -- format (transport held constant), INT8 vs FP8:")
    for tr in ("nccl", "bar1"):
        L.append("  " + tr.upper().ljust(5)
                 + "  ".join(f"{p}: {t['format_effect'][f'{tr}|{p}']}"
                             for p in t["points"]))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    t = build(args.out_dir)
    with open(os.path.join(args.out_dir, "table_366.json"), "w") as fh:
        json.dump(t, fh, indent=1)
    text = render(t)
    with open(os.path.join(args.out_dir, "tabelle_366.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
