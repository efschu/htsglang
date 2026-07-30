#!/usr/bin/env python3
"""s13 -- the tables for #293 step 2, built from what the run persisted.

Reads three kinds of artifact out of one step directory and prints markdown:

  punkte.jsonl        one line per measured point, arm name carrying the round
                      as a suffix ("bar1pipe_r2")
  wait/<arm>.json     the compute/wait split of that boot's primary point,
                      produced on the host by s12_log_analyse
  belege/<arm>.txt    the ERREICHT lines and the prefill-graph lines of that
                      boot -- evidence, not numbers

THE NOISE FLOOR IS COMPUTED, NOT ASSUMED. Every arm ran in every round, so the
spread of one arm across rounds is an A-vs-A measurement of the same
configuration. The largest such spread over all arms is the floor, and a
difference between two arms that does not clear it is printed with a marker
rather than as a result. Nothing here decides what the levers are worth; it
prints the numbers with their uncertainty attached so the verdict can be
argued with.

Stdlib only: it runs in the container against the run directory, but nothing
stops it running on the host.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

REFERENZ = "nccl"


def arm_und_runde(name: str) -> tuple:
    m = re.match(r"^(.*)_r(\d+)$", name or "")
    if not m:
        return (name, 0)
    return (m.group(1), int(m.group(2)))


def lade_punkte(step_dir: str) -> list:
    pfad = os.path.join(step_dir, "punkte.jsonl")
    out = []
    if not os.path.exists(pfad):
        return out
    with open(pfad, errors="replace") as f:
        for zeile in f:
            zeile = zeile.strip()
            if not zeile:
                continue
            try:
                out.append(json.loads(zeile))
            except json.JSONDecodeError:
                continue
    return out


def lade_wait(step_dir: str) -> dict:
    """{(arm, runde): [rang-aggregate]} out of wait/*.json."""
    out: dict = {}
    d = os.path.join(step_dir, "wait")
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, name), errors="replace") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for w in payload.get("wait") or []:
            arm, runde = arm_und_runde(w.get("arm"))
            out.setdefault((arm, runde), []).append(w)
    return out


def lade_belege(step_dir: str) -> dict:
    """{(arm, runde): {'bar1_gruppen': n, 'prefill_graph': str}}."""
    out: dict = {}
    d = os.path.join(step_dir, "belege")
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".txt"):
            continue
        arm, runde = arm_und_runde(name[:-4])
        try:
            with open(os.path.join(d, name), errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        pg = "aus"
        if "prefill CUDA graph end" in text or "prefill CUDA graph begin" in text:
            pg = "AN"
        elif "isabling prefill CUDA graph" in text or "Disable prefill CUDA" in text:
            pg = "aus (auto-disable)"
        out[(arm, runde)] = {
            "bar1_gruppen": text.count("HTCCL-BAR1: Aufbau in"),
            "erreicht": len(re.findall(r"ERREICHT=", text)),
            "pipe_zeilen": text.count("HTCCL-BAR1-PIPE:"),
            "vorrat_leer": "Graph-Vorrat des Ergebnisrings ist erschoepft" in text,
            "prefill_graph": pg,
        }
    return out


def sammle(punkte: list) -> dict:
    """{(arm, sessions): {runde: {...}}}"""
    out: dict = {}
    for p in punkte:
        arm, runde = arm_und_runde(p.get("arm"))
        sess = p.get("sessions")
        rate = (p.get("prefill") or {}).get("prefill_tok_s")
        eintrag = {"prefill_tok_s": rate}
        for d in p.get("decode") or []:
            bs = d.get("batch")
            eintrag[f"tick_tok_s_bs{bs}"] = d.get("tick_gen_tok_s_median")
            eintrag[f"accept_bs{bs}"] = d.get("tick_accept_len_median")
            eintrag[f"ms_verify_bs{bs}"] = d.get("tick_ms_pro_verify")
        out.setdefault((arm, sess), {})[runde] = eintrag
    return out


def _mittel(werte: list):
    werte = [w for w in werte if isinstance(w, (int, float))]
    if not werte:
        return None
    return sum(werte) / len(werte)


def _spanne_pct(werte: list):
    """Relative spread of repeated measurements of the SAME configuration."""
    werte = [w for w in werte if isinstance(w, (int, float))]
    if len(werte) < 2:
        return None
    m = _mittel(werte)
    if not m:
        return None
    return (max(werte) - min(werte)) / m * 100.0


def _f(v, nk=1):
    return "-" if not isinstance(v, (int, float)) else format(v, f".{nk}f")


def bericht(step_dir: str) -> str:
    punkte = lade_punkte(step_dir)
    daten = sammle(punkte)
    wait = lade_wait(step_dir)
    belege = lade_belege(step_dir)

    arme = []
    for arm, _ in sorted(daten):
        if arm not in arme:
            arme.append(arm)
    sessions = sorted({s for _, s in daten if isinstance(s, int)})
    runden = sorted({r for d in daten.values() for r in d})

    zeilen = []

    # --- noise floor first, because it decides what may be reported ---------
    # PER SESSION COUNT, not one number for the whole table. The two points
    # are different measurements -- one session is a latency measurement with
    # a single stream feeding it, eight is a saturated pipeline -- and they do
    # not have the same repeatability. Folding them into one maximum would
    # hold the tight point (0,2-0,8 % at eight sessions) to the loose point's
    # floor and throw away real differences.
    spannen = []
    boden_je_sess: dict = {}
    for (arm, sess), je_runde in sorted(daten.items()):
        s = _spanne_pct([e.get("prefill_tok_s") for e in je_runde.values()])
        if s is not None:
            spannen.append((s, arm, sess))
            boden_je_sess[sess] = max(boden_je_sess.get(sess, 0.0), s)
    boden = max((s for s, _, _ in spannen), default=None)
    median_spanne = statistics.median([s for s, _, _ in spannen]) if spannen else None

    zeilen.append("### Rauschboden (A gegen A, derselbe Arm ueber die Runden)")
    zeilen.append("")
    zeilen.append("| Arm | Sess. | " + " | ".join(f"R{r}" for r in runden) + " | Spanne % |")
    zeilen.append("|---|---:|" + "---:|" * (len(runden) + 1))
    for (arm, sess), je_runde in sorted(daten.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        werte = [je_runde.get(r, {}).get("prefill_tok_s") for r in runden]
        zeilen.append(
            f"| {arm} | {sess} | "
            + " | ".join(_f(w) for w in werte)
            + f" | {_f(_spanne_pct(werte), 2)} |"
        )
    zeilen.append("")
    zeilen.append(
        f"Groesste A-gegen-A-Spanne insgesamt: **{_f(boden, 2)} %**, "
        f"Median der Spannen {_f(median_spanne, 2)} %. "
        "Massstab ist aber der Boden DES JEWEILIGEN PUNKTES: "
        + ", ".join(
            f"{s} Session(s) {_f(b, 2)} %" for s, b in sorted(boden_je_sess.items())
        )
        + ". Ein Verhaeltnis, das weniger als diesen Boden von 1,000 abweicht, "
        "ist unten mit `~` markiert und ist keine Aussage."
    )
    zeilen.append("")

    # --- the main table ----------------------------------------------------
    zeilen.append("### Arm x Sessions: Prefill-Durchsatz und Verhaeltnis zu NCCL")
    zeilen.append("")
    kopf = "| Arm |"
    trenn = "|---|"
    for sess in sessions:
        kopf += f" tok/s (s={sess}) | vs. nccl |"
        trenn += "---:|---:|"
    kopf += " Prefill-Graph | BAR1-Gruppen |"
    trenn += "---|---:|"
    zeilen.append(kopf)
    zeilen.append(trenn)

    for arm in arme:
        zeile = f"| {arm} |"
        for sess in sessions:
            je_runde = daten.get((arm, sess), {})
            m = _mittel([e.get("prefill_tok_s") for e in je_runde.values()])
            ref = _mittel(
                [
                    e.get("prefill_tok_s")
                    for e in daten.get((REFERENZ, sess), {}).values()
                ]
            )
            if m is None or not ref:
                zeile += f" {_f(m)} | - |"
                continue
            v = m / ref
            marke = ""
            grenze = boden_je_sess.get(sess)
            if grenze is not None and abs(v - 1.0) * 100.0 < grenze:
                marke = "~"
            zeile += f" {_f(m)} | {marke}{v:.3f} |"
        bel = belege.get((arm, runden[0] if runden else 1), {})
        zeile += f" {bel.get('prefill_graph', '-')} | {bel.get('bar1_gruppen', '-')} |"
        zeilen.append(zeile)
    zeilen.append("")

    # --- compute / wait per rank ------------------------------------------
    zeilen.append("### compute / wait je Rang am Primaerpunkt (sessions=8)")
    zeilen.append("")
    zeilen.append(
        "| Arm | Runde | TP0 comp | TP0 wait | TP1 comp | TP1 wait | "
        "TP2 comp | TP2 wait | gpu-ms TP1 | wait-Anteil TP1 |"
    )
    zeilen.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arme:
        for runde in runden:
            rows = wait.get((arm, runde))
            if not rows:
                continue
            je_rang = {r.get("rang"): r for r in rows}
            zelle = []
            for rang in (0, 1, 2):
                r = je_rang.get(rang) or {}
                zelle.append(_f(r.get("compute_ms_median")))
                zelle.append(_f(r.get("wait_ms_median")))
            tp1 = je_rang.get(1) or {}
            anteil = tp1.get("wait_anteil")
            zeilen.append(
                f"| {arm} | {runde} | "
                + " | ".join(zelle)
                + f" | {_f(tp1.get('gpu_ms_median'))} | "
                + (f"{anteil * 100:.1f} %" if isinstance(anteil, float) else "-")
                + " |"
            )
    zeilen.append("")

    # --- decode -----------------------------------------------------------
    zeilen.append("### Decode-Ticks am selben Boot (sessions=8)")
    zeilen.append("")
    zeilen.append(
        "| Arm | bs=1 tok/s | bs=1 accept | bs=16 tok/s | bs=16 accept | "
        "bs=16 ms/Verify |"
    )
    zeilen.append("|---|---:|---:|---:|---:|---:|")
    for arm in arme:
        je_runde = daten.get((arm, 8), {})
        if not je_runde:
            continue
        def mm(key):
            return _mittel([e.get(key) for e in je_runde.values()])
        zeilen.append(
            f"| {arm} | {_f(mm('tick_tok_s_bs1'))} | {_f(mm('accept_bs1'), 2)} | "
            f"{_f(mm('tick_tok_s_bs16'))} | {_f(mm('accept_bs16'), 2)} | "
            f"{_f(mm('ms_verify_bs16'), 2)} |"
        )
    zeilen.append("")

    # --- evidence ---------------------------------------------------------
    zeilen.append("### Belege je Boot")
    zeilen.append("")
    zeilen.append(
        "| Arm | Runde | ERREICHT-Zeilen | BAR1-Aufbau | PIPE-Zeilen | "
        "Vorrat leer | Prefill-Graph |"
    )
    zeilen.append("|---|---:|---:|---:|---:|---|---|")
    for arm in arme:
        for runde in runden:
            b = belege.get((arm, runde))
            if not b:
                continue
            zeilen.append(
                f"| {arm} | {runde} | {b['erreicht']} | {b['bar1_gruppen']} | "
                f"{b['pipe_zeilen']} | {'ja' if b['vorrat_leer'] else 'nein'} | "
                f"{b['prefill_graph']} |"
            )
    zeilen.append("")
    return "\n".join(zeilen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    print(bericht(args.step_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
