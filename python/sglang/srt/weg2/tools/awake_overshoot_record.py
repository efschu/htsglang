#!/usr/bin/env python3
"""H94: das Awake-Overshoot-Record EINES Boots aus seinen eigenen Logs.

    python -m sglang.srt.weg2.tools.awake_overshoot_record \\
        --front boot_weg2_<tag>_<tip>_<stamp>.front.log [--group P|D|both] \\
        [--append /spinning/evidence-665-f1/weg2_measured_record.json]

Liest aus dem Front-Log Tag, Commit, WEG2-FORM (Modell), die Zuordnung
Ordinal -> Karte, die Budgetzeilen (die geltende ``measured_awake_overshoot``-
Buchung je Ordinal) und die CORRIDOR-FLOOR-Zeilen (``verdict_floor`` je Karte);
aus den Geschwister-Logs ``.P.log`` / ``.D.log`` die H55-Fenster
``WEG2-VRAM-PEAK`` der Lastphasen. Die Regel steht in
``weg2.awake_overshoot`` (SLACK / SATURATED / SHORT). Ohne ``--append``
schreibt es nichts: es druckt die Herleitung je Karte und das JSON des Records.

Das Record gilt nur fuer das Modell dieses Boots (``model`` im Record, beim
Lesen zusaetzlich ``weg2_form.same_model_sample``): ein NF-Record wird nie
einem 27B-Boot berechnet und umgekehrt.

Nur Standardbibliothek.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

from sglang.srt.weg2 import awake_overshoot as ao


def _sibling(front: str, suffix: str) -> str:
    if not front.endswith(".front.log"):
        raise SystemExit(f"--front must name a *.front.log, got {front!r}")
    return front[: -len(".front.log")] + suffix


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--front", required=True, help="the boot's *.front.log")
    ap.add_argument("--group", choices=("P", "D", "both"), default="both")
    ap.add_argument("--append", default=None,
                    help="measured-record sidecar to append the record(s) to")
    ns = ap.parse_args(argv)

    with open(ns.front, errors="replace") as f:
        ff = ao.parse_front(f)
    if not ff.model:
        print(f"{ao.LINE_TAG} REFUSED: {ns.front} carries no WEG2-FORM line -- the record "
              f"could not name the checkpoint it was measured on", file=sys.stderr)
        return 2
    if not ff.ordinals:
        print(f"{ao.LINE_TAG} REFUSED: {ns.front} carries no USER-RESERVE PROVENANCE "
              f"line -- ordinals cannot be mapped to cards", file=sys.stderr)
        return 2
    groups = ("P", "D") if ns.group == "both" else (ns.group,)
    records: List[dict] = []
    rc = 0
    for g in groups:
        path = _sibling(ns.front, f".{g}.log")
        try:
            with open(path, errors="replace") as f:
                minima = ao.load_minima(f, g)
        except OSError as e:
            print(f"{ao.LINE_TAG} group={g} REFUSED: {path}: {e}", file=sys.stderr)
            rc = 2
            continue
        derivs, gaps = ao.derive_group(ff, g, minima)
        print(f"{ao.LINE_TAG} DERIVE group={g} model={ff.model} boot={ff.tag} @ {ff.commit}")
        for d in derivs:
            print("  " + d.text())
        for gap in gaps:
            print("  GAP " + gap)
        if gaps:
            print(f"{ao.LINE_TAG} group={g} REFUSED: {len(gaps)} card(s) not derivable -- "
                  f"a partial record would charge 0 by omission", file=sys.stderr)
            rc = 2
            continue
        records.append(ao.build_record(
            group=g, model=ff.model, boot_tag=ff.tag, commit=ff.commit, at=ff.at,
            derivations=derivs, form=ff.form, source=os.path.basename(ns.front)))
    print(json.dumps(records, indent=1))
    if ns.append:
        for rec in records:
            ao.append_record(ns.append, rec)
        print(f"{ao.LINE_TAG} appended {len(records)} record(s) to {ns.append}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
