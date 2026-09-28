"""VRAM loop P1a (28.09.): the per-state VRAM peak record of one boot.

WEG2-VRAM-PEAK (H55) measures every rank x phase window (allocated, reserved,
peak_allocated, card_free), but the planner read it back for exactly one
quantity (P_ACTIVATION_MIB). This module turns one boot's logs into ONE record
``VRAM_PEAK_MIB`` keyed ``group -> rank -> phase -> state``, where the state is
the load the window ran under:

* D ``phase=round``: ``bs<n>/S<k>`` -- the running seats of the nearest
  preceding ``Decode batch`` line of that rank's group (TP0 carries it) and the
  KV stage the rank last reported (``#251 WAKE-RESHARD ... stage=S<k>``, one per
  wake; S0 when none) -- the
  "stage x seat count" evidence (P1a (d));
* every other phase: the phase itself (``chunk``, ``idle``, ...).

Per key: ``n``, ``peak_allocated`` p99/max, ``cache_unused`` (reserved -
allocated) median/max, ``card_free`` min, ``slack_at_peak`` min (card_free +
reserved - peak_allocated: unused even at the window's peak).

A pure instrument: nothing consumes it yet (P1c/P2 will). ``--write`` merges it
into ``profile_records_data/<profile>.json`` like any measured record.

PURE: stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from bisect import bisect_right
from typing import Dict, List, Optional, Sequence, Tuple

RECORD_NAME = "VRAM_PEAK_MIB"

_RX_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
_RX_KV = re.compile(r"(\w+)=(\S+)")
_RX_DECODE = re.compile(r" TP0\] Decode batch, #running-req: (\d+)")
_RX_STAGE = re.compile(r" TP(\d+)\] #251 WAKE-RESHARD .*?\bstage=S([0-9])\b")


def _ts(line: str) -> Optional[str]:
    m = _RX_TS.match(line)
    return m.group(1) if m else None


def _int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _q(vals: Sequence[int], p: float) -> Optional[int]:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    return v[min(len(v) - 1, int(p * len(v)))]


def samples(group: str, lines) -> List[dict]:
    """The WEG2-VRAM-PEAK samples of one group log, with the load state joined."""
    decode_ts: List[str] = []
    decode_bs: List[int] = []
    stage_ts: Dict[int, List[str]] = {}
    stage_val: Dict[int, List[int]] = {}
    peaks: List[Tuple[str, dict]] = []
    for line in lines:
        ts = _ts(line)
        if ts is None:
            continue
        m = _RX_DECODE.search(line)
        if m:
            decode_ts.append(ts)
            decode_bs.append(int(m.group(1)))
            continue
        m = _RX_STAGE.search(line)
        if m:
            r = int(m.group(1))
            stage_ts.setdefault(r, []).append(ts)
            stage_val.setdefault(r, []).append(int(m.group(2)))
            continue
        if "WEG2-VRAM-PEAK" in line:
            peaks.append((ts, dict(_RX_KV.findall(line))))
    out = []
    for ts, kv in peaks:
        rank = _int(kv.get("rank"))
        phase = kv.get("phase")
        if rank is None or not phase:
            continue
        state = phase
        if group == "D" and phase == "round":
            i = bisect_right(decode_ts, ts) - 1
            bs = decode_bs[i] if i >= 0 else None
            j = bisect_right(stage_ts.get(rank, []), ts) - 1
            stage = stage_val[rank][j] if j >= 0 else 0
            state = f"bs{bs if bs is not None else '?'}/S{stage}"
        out.append(dict(group=group, rank=rank, phase=phase, state=state,
                        allocated=_int(kv.get("allocated_mib")),
                        reserved=_int(kv.get("reserved_mib")),
                        peak_allocated=_int(kv.get("peak_allocated_mib")),
                        card_free=_int(kv.get("card_free_mib"))))
    return out


def aggregate(rows: Sequence[dict]) -> Dict[str, dict]:
    agg: Dict[Tuple[str, int, str, str], List[dict]] = {}
    for r in rows:
        agg.setdefault((r["group"], r["rank"], r["phase"], r["state"]), []).append(r)
    out: Dict[str, dict] = {}
    for (g, rank, phase, state), rs in sorted(agg.items()):
        cache = [x["reserved"] - x["allocated"] for x in rs
                 if x["reserved"] is not None and x["allocated"] is not None]
        slack = [x["card_free"] + x["reserved"] - x["peak_allocated"] for x in rs
                 if None not in (x["card_free"], x["reserved"], x["peak_allocated"])]
        pk = [x["peak_allocated"] for x in rs]
        out.setdefault(g, {}).setdefault(str(rank), {}).setdefault(phase, {})[state] = {
            "n": len(rs),
            "peak_allocated_p99": _q(pk, 0.99),
            "peak_allocated_max": _q(pk, 1.0),
            "cache_unused_med": _q(cache, 0.5),
            "cache_unused_max": _q(cache, 1.0),
            "card_free_min": _q([x["card_free"] for x in rs], 0.0),
            "slack_at_peak_min": _q(slack, 0.0),
        }
    return out


def record_for(stem: str, profile: str) -> dict:
    rows: List[dict] = []
    for g in ("P", "D"):
        path = f"{stem}.{g}.log"
        if os.path.exists(path):
            with open(path, errors="replace") as f:
                rows.extend(samples(g, f))
    tag = os.path.basename(stem)
    return {
        "name": RECORD_NAME,
        "value": aggregate(rows),
        "provenance": (f"VRAM loop P1a: WEG2-VRAM-PEAK of boot {tag} per group x rank x phase x "
                       f"load state (D rounds: bs<n>/S<stage>); {len(rows)} windows. MiB. "
                       "slack_at_peak = card_free + reserved - peak_allocated."),
        "boots": [tag],
        "kind": "memory",
        "power_limit_w": None,
        "power_limit_source": "not a power-dependent quantity (bytes)",
    }


def merge_into(path: str, rec: dict) -> None:
    with open(path) as f:
        data = json.load(f)
    recs = [r for r in data.get("records", []) if r.get("name") != rec["name"]]
    recs.append(rec)
    data["records"] = recs
    with open(path, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stem", required=True, help="boot log stem (without .P.log/.D.log)")
    ap.add_argument("--profile", required=True, help="profile records file, e.g. nextflash")
    ap.add_argument("--write", action="store_true", help="merge into profile_records_data")
    a = ap.parse_args(argv)
    rec = record_for(a.stem, a.profile)
    if a.write:
        from sglang.srt.weg2.profile_records import RECORDS_DIR

        merge_into(os.path.join(RECORDS_DIR, f"{a.profile}.json"), rec)
    json.dump(rec, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
