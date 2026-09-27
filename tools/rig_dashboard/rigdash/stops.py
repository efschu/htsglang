"""Planned stop or death?  Read from the acceptance harness's own log.

Operator order 2026-09-27 ~18Z: NF rc12s (09271719) was stopped ON PURPOSE
(stop file of the NF endless loop, switch to rc12t) and the dashboard showed
"Gruppe D tot" -- the teardown after a planned stop kills the processes, and
WEG2-HEALTH then reports process_alive=False like a death.

Both the NF endless loop (gpu-arb/docker/nf_dauer_loop.sh) and the 27B arms
run host_acceptance.sh, which writes ``<model dir>/abnahme_cu130.log`` next to
``<model dir>/evidence/boot_*.log``.  The lines that decide it, verbatim:

  planned  [nf-dauer 2026-09-27T17:39:28Z] Stop-Datei -> Hold-Ende, Schleife endet
  planned  [host-acc 17:39:36Z] AGENT-HOLD (h91dprbar1dauer) Ende: Stop-Datei
  planned  [host-acc ...Z] AGENT-HOLD (<tag>) Ende: Zeit            (the hold ran out)
  death    [host-acc 15:39:54Z] AGENT-HOLD (releasedraftbar1w1) Ende: Container-tot
  death    [nf-rc11b 16:28:55Z] DEADMAN-VERDICT deadman_dkrnfh91dprbar1dauer09271603_P: DEADMAN[CRASH] 2026-09-27T16:28:41+00:00 ...

A death kicks the hold by the same stop file (the loop touches agent_hold_stop
after a deadman verdict), so "Ende: Stop-Datei" alone is NOT proof of a
planned stop: any death marker of the boot wins.  The nf-dauer "Stop-Datei"
line is written only when the operator set the loop's stop file.
"""

from __future__ import annotations

import calendar
import os
import re
import threading
from typing import Dict, List, Optional

HARNESS_LOG = "abnahme_cu130.log"
MATCH_AFTER_S = 600.0     # a marker up to this long after the boot's last log line still belongs to it

RE_DAUER_STOP = re.compile(r"^\[nf-dauer (\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)Z\] Stop-Datei -> Hold-Ende")
RE_HOLD_END = re.compile(r"^\[host-acc (\d\d):(\d\d):(\d\d)Z\] AGENT-HOLD \(([^)\s]+)\) Ende: (Stop-Datei|Zeit|Container-tot)")
RE_DEADMAN = re.compile(r"DEADMAN-VERDICT deadman_(\S+?)_(P|D|front): DEADMAN\[([A-Z_-]+)\]"
                        r"(?: (\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d))?")


def parse_marker(line: str) -> Optional[dict]:
    """One harness line -> marker dict, or None.  ``t`` is epoch UTC when the
    line carries a date, else ``sod`` (seconds of the UTC day) is set and the
    date is resolved against the boot (resolve_t)."""
    m = RE_DAUER_STOP.match(line)
    if m:
        y, mo, d, hh, mi, ss = (int(x) for x in m.groups())
        return {"kind": "planned", "src": "nf-dauer", "t": float(calendar.timegm((y, mo, d, hh, mi, ss))),
                "tag": None, "text": line.strip()[:240]}
    m = RE_HOLD_END.match(line)
    if m:
        hh, mi, ss, tag, why = m.groups()
        return {"kind": "death" if why == "Container-tot" else "planned", "src": "host-acc",
                "sod": int(hh) * 3600 + int(mi) * 60 + int(ss), "t": None, "tag": tag, "why": why,
                "text": line.strip()[:240]}
    m = RE_DEADMAN.search(line)
    if m:
        ident, group, verdict = m.group(1), m.group(2), m.group(3)
        t = None
        if m.group(4):
            t = float(calendar.timegm(tuple(int(x) for x in m.groups()[3:9])))
        return {"kind": "death", "src": "deadman", "t": t, "ident": ident, "group": group,
                "verdict": verdict, "text": line.strip()[:240]}
    return None


def resolve_t(mk: dict, first_t: float) -> Optional[float]:
    """Epoch of a marker; a time-of-day-only line takes the boot's UTC day
    (the next day when that lands more than an hour before the boot)."""
    if mk.get("t") is not None:
        return mk["t"]
    if mk.get("sod") is None or first_t is None:
        return None
    day = first_t - (first_t % 86400)
    t = day + mk["sod"]
    if t < first_t - 3600:
        t += 86400
    return t


def classify(stem: str, first_t: Optional[float], last_t: Optional[float], markers: List[dict]) -> dict:
    """``{"planned": {t, text} | None, "death": {t, text} | None}`` for one boot.
    Pure, unit-tested.  A death marker of the boot always wins."""
    out = {"planned": None, "death": None}
    if first_t is None:
        return out
    hi = (last_t or first_t) + MATCH_AFTER_S
    for mk in markers:
        if mk["src"] == "deadman":
            # the deadman id is the boot's tag with its date stamp -> exact
            if mk["ident"] not in stem:
                continue
            t = mk["t"] if mk["t"] is not None else last_t
        else:
            if mk.get("tag") and mk["tag"] not in stem:
                continue
            t = resolve_t(mk, first_t)
            if t is None or t < first_t or t > hi:
                continue
        slot = "death" if mk["kind"] == "death" else "planned"
        cur = out[slot]
        if cur is None or t < cur["t"]:
            out[slot] = {"t": t, "text": mk["text"], "src": mk["src"]}
    if out["death"] is not None:
        out["planned"] = None
    return out


class HarnessLogs:
    """Tails ``<model dir>/abnahme_cu130.log`` for every model dir with boots."""

    def __init__(self):
        from .live import Tail          # late: live imports nothing from here
        self._Tail = Tail
        self.tails: Dict[str, object] = {}
        self.markers: Dict[str, List[dict]] = {}
        self.lock = threading.Lock()

    @staticmethod
    def path_for(evidence_dir: str) -> str:
        return os.path.join(os.path.dirname(evidence_dir.rstrip("/")), HARNESS_LOG)

    def poll(self, evidence_dirs):
        for ed in set(evidence_dirs):
            p = self.path_for(ed)
            if not os.path.exists(p):
                continue
            t = self.tails.get(p)
            if t is None:
                t = self.tails[p] = self._Tail(p, None)
            new = []
            for line in t.poll():
                mk = parse_marker(line)
                if mk:
                    new.append(mk)
            if new:
                with self.lock:
                    lst = self.markers.setdefault(p, [])
                    lst.extend(new)
                    del lst[:-2000]

    def for_dir(self, evidence_dir: str) -> List[dict]:
        with self.lock:
            return list(self.markers.get(self.path_for(evidence_dir), ()))
