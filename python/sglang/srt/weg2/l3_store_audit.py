# SPDX-License-Identifier: Apache-2.0
"""L3-REVOKE (28.09.): read a persistent L3 store from outside, and name a
revoked write window in it.

    python -m sglang.srt.weg2.l3_store_audit audit <store>
    python -m sglang.srt.weg2.l3_store_audit revoke <store> --from 2026-09-28T10:51:24Z \
        --to 2026-09-28T10:57:19Z --reason "rc12z17 S1 live-shrink wiped the expert bank"

``audit`` only reads: the identity records, the pages by kind (KV / QSA index /
other component), the KV pages whose QSA sibling is missing (a KV hit without
its index is capped to nothing, L3-REUSE 0928), the write-time histogram per
minute and the pages inside every revoked window. ``revoke`` only writes the
window into ``L3_REVOKED.json``; the pages are MOVED by the next boot's
attach (``launcher.l3_persist_attach``), never here and never deleted. Run
neither against a store a live boot is writing -- the audit would read a
moving target, and the record is only read at attach anyway.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

QSA_COMPONENT = "qsa_indexer"
_HEX = frozenset("0123456789abcdef")


def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def parse_utc(text: str) -> float:
    """``2026-09-28T10:51:24Z`` (or without the Z) -> unix seconds, UTC."""
    t = text.strip().rstrip("Z")
    return float(calendar.timegm(time.strptime(t, "%Y-%m-%dT%H:%M:%S")))


def split_stem(stem: str, suffixes: Tuple[str, ...] = ()) -> Tuple[str, str, str]:
    """``(hash, component, suffix)`` of a page stem ``{hash}[.{component}]{suffix}``.

    The suffix is one the groups recorded (``L3_SUFFIXES.<group>.json``,
    longest first); without a record the hash is the leading hex run and the
    suffix starts after the component (the model name in the suffix carries
    dots -- ``Qwen3.8`` -- and a component name carries ``_``: ``qsa_indexer``)."""
    for sfx in sorted(suffixes, key=len, reverse=True):
        if sfx and stem.endswith(sfx):
            head = stem[: len(stem) - len(sfx)]
            h, _, comp = head.partition(".")
            return h, comp, sfx
    i = 0
    while i < len(stem) and stem[i] in _HEX:
        i += 1
    h, rest = stem[:i], stem[i:]
    if not rest.startswith("."):
        return h, "", rest
    rest = rest[1:]
    for comp in (QSA_COMPONENT,):
        if rest.startswith(comp):
            return h, comp, rest[len(comp):]
    cut = rest.find("_")
    return (h, rest, "") if cut < 0 else (h, rest[:cut], rest[cut:])


def _pages(directory: str):
    for root, _dirs, names in os.walk(directory):
        for name in names:
            if name.endswith(".bin") and ".tmp." not in name:
                p = os.path.join(root, name)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                yield p, name[:-4], st


def audit(directory: str, windows: Optional[List[Tuple[float, float, str]]] = None) -> Dict:
    """Everything ``audit`` prints, as data (read-only)."""
    out: Dict = {"dir": directory, "records": {}}
    for n in sorted(os.listdir(directory)):
        if n.startswith("L3_") and n.endswith(".json"):
            try:
                with open(os.path.join(directory, n)) as f:
                    out["records"][n] = json.load(f)
            except (OSError, ValueError) as exc:
                out["records"][n] = f"UNREADABLE {type(exc).__name__}: {exc}"
    if windows is None:
        from sglang.srt.weg2.launcher import l3_revoked_windows

        windows = l3_revoked_windows(directory)
    suffixes: Tuple[str, ...] = ()
    for rec in out["records"].values():
        if isinstance(rec, dict) and isinstance(rec.get("suffixes"), list):
            suffixes += tuple(str(x) for x in rec["suffixes"])
    kinds: Counter = Counter()
    kind_bytes: Counter = Counter()
    minutes: Counter = Counter()
    kv: Dict[Tuple[str, str], int] = {}
    qsa = set()
    in_window = [0] * len(windows)
    in_window_bytes = [0] * len(windows)
    for _p, stem, st in _pages(directory):
        h, comp, sfx = split_stem(stem, suffixes)
        kind = comp or "kv"
        size = st.st_blocks * 512
        kinds[kind] += 1
        kind_bytes[kind] += size
        minutes[time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(st.st_mtime))] += 1
        if not comp:
            kv[(h, sfx)] = 1
        elif comp == QSA_COMPONENT:
            qsa.add((h, sfx))
        for i, (t0, t1, _r) in enumerate(windows):
            if t0 <= st.st_mtime <= t1:
                in_window[i] += 1
                in_window_bytes[i] += size
    out["kinds"] = {k: {"pages": kinds[k], "gib": round(kind_bytes[k] / 2**30, 3)} for k in sorted(kinds)}
    out["kv_without_qsa"] = sum(1 for key in kv if key not in qsa) if qsa else None
    out["minutes"] = dict(sorted(minutes.items()))
    out["windows"] = [
        {"from": _utc(t0), "to": _utc(t1), "reason": r, "pages": in_window[i],
         "gib": round(in_window_bytes[i] / 2**30, 3)}
        for i, (t0, t1, r) in enumerate(windows)
    ]
    return out


def _print_audit(rep: Dict) -> None:
    print(f"L3-AUDIT dir={rep['dir']}")
    for n, rec in rep["records"].items():
        print(f"  record {n}: {json.dumps(rec, sort_keys=True) if not isinstance(rec, str) else rec}")
    for k, v in rep["kinds"].items():
        print(f"  pages {k}: {v['pages']} ({v['gib']} GiB)")
    kvq = rep["kv_without_qsa"]
    print(f"  kv pages without their {QSA_COMPONENT} sibling: "
          + ("n/a (no QSA pages in this store)" if kvq is None else str(kvq)))
    for m, n in rep["minutes"].items():
        print(f"  written {m}: {n}")
    for w in rep["windows"]:
        print(f"  REVOKED {w['from']} .. {w['to']} ({w['reason']}): {w['pages']} pages "
              f"({w['gib']} GiB) -- moved to <store>.revoked at the next attach")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="l3_store_audit", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="read a store (nothing is written)")
    a.add_argument("store")
    a.add_argument("--json", action="store_true")
    r = sub.add_parser("revoke", help="record a revoked write window (moved at the next attach)")
    r.add_argument("store")
    r.add_argument("--from", dest="t_from", required=True, help="UTC, e.g. 2026-09-28T10:51:24Z")
    r.add_argument("--to", dest="t_to", required=True, help="UTC")
    r.add_argument("--reason", required=True)
    ns = ap.parse_args(argv)
    if ns.cmd == "audit":
        rep = audit(ns.store)
        if ns.json:
            print(json.dumps(rep, sort_keys=True, indent=1))
        else:
            _print_audit(rep)
        return 0
    from sglang.srt.weg2.launcher import l3_revoke_window

    rec = l3_revoke_window(ns.store, parse_utc(ns.t_from), parse_utc(ns.t_to), ns.reason)
    print(f"L3-REVOKE recorded in {os.path.join(ns.store, 'L3_REVOKED.json')}: "
          f"{json.dumps(rec['windows'][-1], sort_keys=True)} -- pages move at the next attach")
    return 0


if __name__ == "__main__":
    sys.exit(main())
