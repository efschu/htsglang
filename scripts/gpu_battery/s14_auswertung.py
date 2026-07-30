#!/usr/bin/env python3
"""s14 -- decode_punkte.jsonl to the tables the #294 verdict is written from.

Runs in the container, reads nothing but the run directory. It does not judge:
it prints the floor, the per-point table and the ratios, and the verdict is
written by hand into docs/dev/INTEGRATION_R3_VALIDATION.md.

THE FLOOR IS PRINTED BEFORE THE RATIOS, and every ratio is printed next to it,
because a ratio smaller than the floor of its own arm is not a finding. The
floor here is the spread of REPEATS OF THE SAME ARM at the same batch size --
within one boot and across boots -- expressed as (max-min)/median, which is the
quantity a between-arm difference has to clear.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

METRIKEN = (
    ("tick_ms_pro_verify", "ms/Verify"),
    ("tick_gen_tok_s_median", "tok/s (tick)"),
    ("klient_tok_s", "tok/s (klient)"),
    ("tick_accept_len_median", "accept"),
    ("tick_ms_pro_schritt", "ms/Schritt"),
)


def lade(pfad: str) -> list:
    punkte = []
    if not os.path.exists(pfad):
        return punkte
    with open(pfad, errors="replace") as f:
        for zeile in f:
            zeile = zeile.strip()
            if not zeile:
                continue
            try:
                punkte.append(json.loads(zeile))
            except json.JSONDecodeError:
                continue
    return punkte


def _arm_basis(arm: str) -> str:
    """`bar1_hi_r2` -> `bar1_hi`. The round belongs to the sample, not the arm."""
    return arm.rsplit("_r", 1)[0] if "_r" in arm else arm


def _spanne(werte: list) -> dict:
    werte = [w for w in werte if isinstance(w, (int, float))]
    if not werte:
        return {"n": 0}
    med = statistics.median(werte)
    out = {
        "n": len(werte),
        "median": med,
        "min": min(werte),
        "max": max(werte),
        "spanne_rel": (max(werte) - min(werte)) / med if med else None,
    }
    if len(werte) > 2:
        out["stdev_rel"] = statistics.stdev(werte) / med if med else None
    return out


def _fmt(x, nk=2) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "ja" if x else "NEIN"
    if isinstance(x, float):
        return f"{x:.{nk}f}"
    return str(x)


def tabelle_punkte(punkte: list) -> str:
    z = [
        "| Arm | bs | Wdh | ms/Verify | ms/Schritt | tok/s tick | tok/s klient "
        "| accept (tick) | accept (klient) | Ticks gew./bs | fremde bs | Graph |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    zaehler: dict = {}
    for p in sorted(punkte, key=lambda q: (q.get("arm", ""), q.get("folge", 0))):
        key = (p.get("arm"), p.get("bs"))
        zaehler[key] = zaehler.get(key, 0) + 1
        z.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} |".format(
                p.get("arm"),
                p.get("bs"),
                zaehler[key],
                _fmt(p.get("tick_ms_pro_verify")),
                _fmt(p.get("tick_ms_pro_schritt"), 1),
                _fmt(p.get("tick_gen_tok_s_median"), 1),
                _fmt(p.get("klient_tok_s"), 1),
                _fmt(p.get("tick_accept_len_median")),
                _fmt(p.get("klient_accept_len_gesamt")),
                _fmt(p.get("tick_ticks_gewertet")),
                _fmt(p.get("tick_ticks_bs")),
                _fmt(p.get("tick_ticks_fremde_bs")),
                _fmt(p.get("tick_cuda_graph")),
            )
        )
    return "\n".join(z)


def tabelle_boden(punkte: list) -> str:
    """Repeat spread per (arm, bs). This is the floor, and it comes first."""
    gruppen: dict = {}
    for p in punkte:
        gruppen.setdefault((_arm_basis(p.get("arm", "")), p.get("bs")), []).append(p)
    z = [
        "| Arm | bs | Wdh | Metrik | Median | min | max | Spanne (max-min)/Median "
        "| rel. Stdev |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for (arm, bs), gruppe in sorted(gruppen.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        if len(gruppe) < 2:
            continue
        for feld, name in METRIKEN:
            s = _spanne([g.get(feld) for g in gruppe])
            if s.get("n", 0) < 2:
                continue
            z.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    arm,
                    bs,
                    s["n"],
                    name,
                    _fmt(s["median"]),
                    _fmt(s["min"]),
                    _fmt(s["max"]),
                    _fmt(100.0 * s["spanne_rel"], 2) + " %" if s["spanne_rel"] is not None else "-",
                    _fmt(100.0 * s["stdev_rel"], 2) + " %" if s.get("stdev_rel") is not None else "-",
                )
            )
    return "\n".join(z) if len(z) > 2 else "(keine Wiederholung im Lauf)"


def tabelle_verhaeltnis(punkte: list, arm_a: str, arm_b: str) -> str:
    """arm_a against arm_b per batch size, with both floors next to the ratio."""
    je: dict = {}
    for p in punkte:
        je.setdefault((_arm_basis(p.get("arm", "")), p.get("bs")), []).append(p)
    bs_werte = sorted({bs for (_, bs) in je if bs is not None})
    z = [
        f"| bs | ms/Verify {arm_a} | ms/Verify {arm_b} | Faktor | tok/s {arm_a} "
        f"| tok/s {arm_b} | Faktor | accept {arm_a} | accept {arm_b} "
        "| Boden ms/Verify (max beider Arme) | ueber Boden |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for bs in bs_werte:
        a = je.get((arm_a, bs)) or []
        b = je.get((arm_b, bs)) or []
        if not a or not b:
            continue
        va = _spanne([p.get("tick_ms_pro_verify") for p in a])
        vb = _spanne([p.get("tick_ms_pro_verify") for p in b])
        ra = _spanne([p.get("tick_gen_tok_s_median") for p in a])
        rb = _spanne([p.get("tick_gen_tok_s_median") for p in b])
        aa = _spanne([p.get("tick_accept_len_median") for p in a])
        ab = _spanne([p.get("tick_accept_len_median") for p in b])
        if not va.get("median") or not vb.get("median"):
            continue
        faktor = vb["median"] / va["median"]
        # A single sample has no spread, and _spanne reports 0.0 for it. Taking
        # that as a floor would clear every difference against a floor of zero,
        # which is the opposite of what a floor is for: a point without a
        # repetition has NO floor and says so.
        spannen = [
            s["spanne_rel"]
            for s in (va, vb)
            if s.get("n", 0) >= 2 and s.get("spanne_rel") is not None
        ]
        if len(spannen) < 2:
            boden_text, ueber = "-", "?"
        else:
            # The difference has to clear the floor of the noisier of the two
            # arms; the floor is a relative spread, so the comparison is on
            # |faktor - 1|.
            boden = max(spannen)
            boden_text = _fmt(100.0 * boden, 2) + " %"
            ueber = "ja" if abs(faktor - 1.0) > boden else "NEIN"
        z.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                bs,
                _fmt(va["median"]),
                _fmt(vb["median"]),
                _fmt(faktor, 3),
                _fmt(ra.get("median"), 1),
                _fmt(rb.get("median"), 1),
                _fmt((ra["median"] / rb["median"]) if ra.get("median") and rb.get("median") else None, 3),
                _fmt(aa.get("median")),
                _fmt(ab.get("median")),
                boden_text,
                ueber,
            )
        )
    return "\n".join(z)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step-dir", required=True)
    p.add_argument("--arm-a", default="bar1_hi")
    p.add_argument("--arm-b", default="nccl_hi")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    punkte = lade(os.path.join(args.step_dir, "decode_punkte.jsonl"))
    if not punkte:
        print("keine Punkte in decode_punkte.jsonl", file=sys.stderr)
        return 1

    teile = [
        f"### Rauschboden -- Wiederholungen desselben Arms ({len(punkte)} Punkte)",
        "",
        tabelle_boden(punkte),
        "",
        "### Punkte einzeln",
        "",
        tabelle_punkte(punkte),
        "",
        f"### {args.arm_a} gegen {args.arm_b}",
        "",
        tabelle_verhaeltnis(punkte, args.arm_a, args.arm_b),
        "",
    ]
    text = "\n".join(teile)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
