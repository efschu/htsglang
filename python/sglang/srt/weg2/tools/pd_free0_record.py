# SPDX-License-Identifier: Apache-2.0
"""H92d: der gemessene Flip-Start P->D eines Boots als Record (free0).

    python -m sglang.srt.weg2.tools.pd_free0_record <boot>.front.log [--append]

Liest ``<boot>.front.log`` und die Geschwister ``.P.log``/``.D.log``: je Wake
P->D (hoechstens die ersten zwei: ``/0`` und ``/1``, die Positionen, die
``wake_credit_pd.plan_wake_credit_pd`` rechnet) ``driver_free`` am Flip-Start,
P's Pufferzeilen und die D-Sitze (``wake_credit_pd.pd_free0_from_logs``). Die
Form ist die, die der Planer DIESES Boots selbst gewaehlt hat (``gegen die
Referenz fnFL2xNNN/1`` in seiner eigenen WAKE-CREDIT-Zeile), das Modell das
seines Front-Logs, Commit und Tag aus dem Dateinamen. Ohne ``--append`` nur
gedruckt; mit ``--append`` an das Sidecar ``weg2_measured_record.json`` im
Verzeichnis des Logs (``--record`` ueberschreibt). Ein Boot, dem eine Zeile
fehlt, schreibt NICHTS (ein halber Eintrag waere eine erfundene Luft)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List

from sglang.srt.weg2 import wake_credit_pd as _wpd

_RX_FORM = re.compile(r"WAKE-CREDIT \(#H14\) P->D \S+: Wake P->D .*?gegen die Referenz (\S+?)/1 ")


def records_from_boot(front_log: str, max_flips: int = 2) -> List[Dict[str, object]]:
    from sglang.srt.weg2 import form as _form

    stem = front_log[: -len(".front.log")] if front_log.endswith(".front.log") else front_log
    texts = {}
    for kind in ("front", "P", "D"):
        with open("%s.%s.log" % (stem, kind), errors="replace") as fh:
            texts[kind] = fh.read()
    m = _RX_FORM.search(texts["front"])
    if m is None:
        raise ValueError("%s: keine WAKE-CREDIT P->D-Zeile mit Referenz -- die Form ist "
                         "ungenannt, kein Record" % front_log)
    name = _form._FRONT_LOG_RE.match(os.path.basename(stem + ".front.log"))
    tag = name.group("tag") if name else os.path.basename(stem)
    commit = name.group("tip") if name else None
    model = _form.log_identity(stem + ".front.log").model
    out = []
    for flip in range(int(max_flips)):
        try:
            rec = _wpd.pd_free0_from_logs(texts["P"], texts["D"], texts["front"],
                                          source="fnFL2%s/%d" % (tag[len("fnFL2"):], flip)
                                          if tag.startswith("fnFL2") else "%s/%d" % (tag, flip),
                                          flip=flip)
        except ValueError:
            if flip == 0:
                raise
            break
        rec.update(form_key=m.group(1), commit=commit, boot_tag=tag, model=model)
        out.append(rec)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("front_log")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--record", default=None)
    ns = ap.parse_args(argv)
    recs = records_from_boot(ns.front_log)
    for r in recs:
        print(json.dumps(r, sort_keys=True))
    if ns.append:
        from sglang.srt.weg2 import host_ledger

        path = ns.record or os.path.join(os.path.dirname(os.path.abspath(ns.front_log)),
                                         host_ledger.MEASURED_RECORD_NAME)
        for r in recs:
            host_ledger.append_measured_record(path, r)
        print("appended %d pd_free0 record(s) to %s" % (len(recs), path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
