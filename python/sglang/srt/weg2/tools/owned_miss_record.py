# SPDX-License-Identifier: Apache-2.0
"""#239 S3f Miss-Record BOOTSTRAP (transition, "aus Log (Uebergang)").

THE RECORD IS THE RANKS' OWN (``layers.moe.pool_miss_cost``, written at every
D sleep with ``SGLANG_WEG2_OWNED_MISS_RECORD=<dir>``). This tool reads the same
two numbers from ONE D log of a boot that had no rank record yet; its entry
ranks under every rank record (``resolve_owned_miss_ms``) and is dropped as
soon as one exists. Nothing steers on it once the ranks have measured.

    python -m sglang.srt.weg2.tools.owned_miss_record <boot>.D.log [--append]

The owned solve (``planner.expert_residency.solve_owned_cut``) prices a rank's
round as ``missed rows x MoE layers x ms per row`` and ran on the seed 0.1 ms
(5090) / 0.2 ms (3080), UNMEASURED. z30w (29.09.) showed the cost of that: the
round rule moved misses from the 5090 to TP1 and the round got slower at bs1
and bs2. This tool reads the two numbers a D log already carries and divides
them, per rank:

* ``pool.fetch`` device ms per decode round -- the family the collective
  clock's graph reader splits out of ``Decode rank batch`` (only with
  ``SGLANG_DEBUG_COLLECTIVE_CLOCK_GRAPH_NODES=1``; without it the line says
  ``split unavailable`` and this tool REFUSES, it never guesses);
* missed rows per forward and MoE layer -- ``MoE expert pool layer N: F
  decode forwards since last sync, M misses`` (layers 0/23/47 are sampled;
  their mean stands for every MoE layer, named).

``ms per row = sum(pool.fetch ms) / (misses per forward and layer x MoE
layers x forwards)``, host = the attention host's rank, worker = the
miss-weighted mean of the other ranks. Without ``--append`` the entry is
printed; with it, appended to the sidecar ``weg2_measured_record.json`` next
to the log (``--record`` overrides). A log that lacks either number writes
NOTHING -- half an entry would be an invented cost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

from sglang.srt.planner.expert_residency import OWNED_MISS_KIND

_RX_ROUND = re.compile(
    r"^\[(?P<ts>[^\]]+?) TP(?P<tp>\d+)\] Decode rank batch, rank: (?P<rank>\d+), .*?"
    r"#fwd: (?P<fwd>\d+), gpu-ms: (?P<gpu>[0-9.]+)(?P<rest>.*)$")
_RX_FAMILY = re.compile(r"([\w:.]+) ([0-9.]+)/(\d+)x")
_RX_MISS = re.compile(
    r" TP(?P<tp>\d+)\] MoE expert pool layer (?P<layer>\d+): (?P<fwd>\d+) decode forwards "
    r"since last sync, (?P<miss>\d+) misses")
FETCH_FAMILY = "pool.fetch"


def owned_miss_from_log(d_text: str, *, n_moe_layers: int, host_rank: int = 0,
                        source: str = "", min_rounds: int = 50) -> Dict[str, object]:
    """The ``owned_miss_ms`` entry of one D log, or ``ValueError`` naming what
    is missing."""
    fetch: Dict[int, float] = {}
    fwd: Dict[int, int] = {}
    rounds: Dict[int, int] = {}
    unsplit = 0
    last_ts = ""
    for line in d_text.splitlines():
        m = _RX_ROUND.search(line)
        if m is None:
            continue
        rest = m.group("rest")
        if "wait by family:" not in rest:
            unsplit += 1
            continue
        r = int(m.group("rank"))
        ms = sum(float(v) for name, v, _n in _RX_FAMILY.findall(rest)
                 if name.split(":")[-1] == FETCH_FAMILY)
        fetch[r] = fetch.get(r, 0.0) + ms
        fwd[r] = fwd.get(r, 0) + int(m.group("fwd"))
        rounds[r] = rounds.get(r, 0) + 1
        last_ts = m.group("ts")
    if not rounds:
        raise ValueError(
            "%s: no split 'Decode rank batch' line (%d unsplit) -- the collective clock's "
            "graph reader was off (SGLANG_DEBUG_COLLECTIVE_CLOCK_GRAPH_NODES=1 in group D)"
            % (source or "D log", unsplit))
    miss_rows: Dict[int, List[int]] = {}
    for m in _RX_MISS.finditer(d_text):
        acc = miss_rows.setdefault(int(m.group("tp")), [0, 0])
        acc[0] += int(m.group("miss"))
        acc[1] += int(m.group("fwd"))
    per_rank: Dict[str, Dict[str, float]] = {}
    host_ms: Optional[float] = None
    w_fetch = 0.0
    w_rows = 0.0
    for r in sorted(rounds):
        if rounds[r] < int(min_rounds):
            raise ValueError("%s: rank %d has %d split rounds (< %d)"
                             % (source or "D log", r, rounds[r], min_rounds))
        misses, mfwd = miss_rows.get(r, (0, 0))
        if mfwd <= 0:
            raise ValueError("%s: rank %d has no 'MoE expert pool layer' miss line"
                             % (source or "D log", r))
        per_fwd_layer = misses / mfwd
        rows = per_fwd_layer * int(n_moe_layers) * fwd[r]
        if rows <= 0:
            raise ValueError("%s: rank %d missed no row -- no cost to divide"
                             % (source or "D log", r))
        ms_row = fetch[r] / rows
        per_rank[str(r)] = {"ms_per_row": round(ms_row, 5), "fetch_ms": round(fetch[r], 1),
                            "forwards": fwd[r], "rounds": rounds[r],
                            "miss_per_fwd_layer": round(per_fwd_layer, 4)}
        if r == int(host_rank):
            host_ms = ms_row
        else:
            w_fetch += fetch[r]
            w_rows += rows
    if host_ms is None or w_rows <= 0:
        raise ValueError("%s: host rank %d or the workers are missing"
                         % (source or "D log", host_rank))
    return {
        "kind": OWNED_MISS_KIND, "source": source, "at": last_ts,
        "miss_ms_per_row": [round(host_ms, 5), round(w_fetch / w_rows, 5)],
        "per_rank": per_rank, "rounds": min(rounds.values()), "unsplit_rounds": unsplit,
        "n_moe_layers": int(n_moe_layers), "host_rank": int(host_rank),
        "method": "sum(pool.fetch ms) / (misses per forward and layer, layers 0/23/47 "
                  "mean) x MoE layers x forwards",
        "provenance": "aus Log (Uebergang)",
    }


def record_from_boot(d_log: str, *, n_moe_layers: int, host_rank: int) -> Dict[str, object]:
    from sglang.srt.weg2 import form as _form

    stem = d_log[: -len(".D.log")] if d_log.endswith(".D.log") else d_log
    with open(d_log, errors="replace") as fh:
        text = fh.read()
    name = _form._FRONT_LOG_RE.match(os.path.basename(stem + ".front.log"))
    tag = name.group("tag") if name else os.path.basename(stem)
    rec = owned_miss_from_log(text, n_moe_layers=n_moe_layers, host_rank=host_rank, source=tag)
    model = None
    if os.path.exists(stem + ".front.log"):
        model = _form.log_identity(stem + ".front.log").model
    rec.update(model=model, boot_tag=tag, commit=name.group("tip") if name else None)
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("d_log")
    ap.add_argument("--moe-layers", type=int, default=48)
    ap.add_argument("--host-rank", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--record", default=None)
    a = ap.parse_args(argv)
    try:
        rec = record_from_boot(a.d_log, n_moe_layers=a.moe_layers, host_rank=a.host_rank)
    except ValueError as exc:
        print("OWNED-MISS-RECORD REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(rec, indent=1))
    if a.append:
        from sglang.srt.weg2 import host_ledger

        path = a.record or os.path.join(os.path.dirname(os.path.abspath(a.d_log)),
                                        host_ledger.MEASURED_RECORD_NAME)
        host_ledger.append_measured_record(path, rec)
        print("OWNED-MISS-RECORD appended to %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
