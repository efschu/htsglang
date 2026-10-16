"""ROOM-VERDICTS 1016 (F2, NF h16s P-gap root): a room wait that cannot be answered ends on FACTS, not on time.

THE GAP (boot ..._1f4b3e73d4_1016_103428, P): READ-ROOM's room wait ends when no slot came for its idle bound
(``SGLANG_WEG2_READ_ROOM_WAIT_MS``, 10 s), JOIN-LIVENESS-style. At 11:27:25 and 11:27:35 the reads of PP1/PP2 ended
``READ-ROOM SHORT class=served_nothing_to_spill spilled_pages=0`` after ``waited_ms=10019..10456`` each -- every
serve of the whole wait had answered ``released_pages=0`` (``READ-ROOM SERVE ... free=0 clock=0``) and ``W3-ARENA
SPILL candidates=0``: from the first answered round on, nothing could ever have come; the 10 s were pure idleness
with 5 requests queued (25 s without a prefill batch, two waits in a row). Twelve P gaps of 10-13 s since 11:19Z
match the bound.

THE RULE: every serve (``read_room.serve``, the scheduler round of each P rank) writes THIS rank's latest verdict
into the arena's shared ``.room2`` file: pages it released, nodes whose host copy is held only by I/O in flight
(``transient``: a write-through or store write not acked, a load-back's host lock, a direct mamba row write -- they
end on their own and may make a copy spillable), and whether the spill was in its empty backoff (``quiet``: it did
not look). The fill's room wait ends at once when, since its last progress, EVERY P rank (``pp_size`` distinct
live processes) has answered with released=0, transient=0 and quiet=0 -- a slot frees only when every rank lets go,
and no rank can. The idle bound stays, as the lifesign limit of a wait that does not get its answers.

Not counted as transient, on purpose: a device lock of a request (it does not move while nothing runs), the
spill's keep-payable filter. Switch ``SGLANG_WEG2_ROOM_VERDICTS`` (default on; 0 = the wait byte for byte as before).
"""

from __future__ import annotations

import logging
import os
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MARK = "READ-ROOM"
ENV = "SGLANG_WEG2_ROOM_VERDICTS"

#: one slot per process: pid, t_ms, released, transient, quiet, serve seq
_V = struct.Struct("<QQQQQQ")
_SLOTS = 64
REGION_SIZE = _SLOTS * _V.size
_STALE_MS = 30_000
_LOCK = threading.Lock()


class _State:
    seq = 0
    futile = 0
    notes = 0


_S = _State


def enabled() -> bool:
    raw = (os.environ.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _off() -> int:
    from sglang.srt.weg2 import room_wait_arbiter as _arb

    return _arb._SIZE_F3


def _map(arena: Any, create: bool):
    from sglang.srt.weg2 import room_wait_arbiter as _arb

    return _arb.mapped(arena, create=create)


def _now_ms() -> int:
    return int(time.time() * 1000.0)


# --- the serve side (scheduler round of each P rank) -------------------------------------------------

def transient_blocked(cache: Any) -> int:
    """Nodes whose host copy (an arena reference) is held by I/O IN FLIGHT -- refused by the spill now, possibly
    spillable once it lands. Read-only walk of the tree; 0 for a tree without nodes (a stand-in)."""
    walk = getattr(cache, "_collect_all_nodes", None)
    if not callable(walk):
        return 0
    try:
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE
    except Exception:  # noqa: BLE001
        return 0
    wt = getattr(cache, "ongoing_write_through", None) or {}
    direct = getattr(cache, "_weg2_direct_mamba_rows", None) or {}
    root = getattr(cache, "root_node", None)
    n = 0
    try:
        for node in walk():
            if node is root:
                continue
            try:
                hv = node.component_data[BASE_COMPONENT_TYPE].host_value
            except Exception:  # noqa: BLE001
                continue
            if hv is None:
                continue
            nid = getattr(node, "id", None)
            if (nid in wt or getattr(node, "write_through_pending_id", None) is not None or nid in direct
                    or any(int(getattr(cd, "host_lock_ref", 0) or 0) > 0 for cd in node.component_data)):
                n += 1
    except Exception:  # noqa: BLE001 - an unreadable tree is no proof of futility
        return 1
    return n


def spill_quiet(cache: Any) -> bool:
    """The W3 spill is in its empty backoff: a serve now returns 0 without having looked."""
    try:
        return time.monotonic() < float(getattr(cache, "_w3_spill_quiet_until", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False


def note_verdict(arena: Any, released: int, transient: int, quiet: bool, pid: Optional[int] = None) -> None:
    """This rank's latest serve verdict into the shared file (one slot per process)."""
    if not enabled():
        return
    m = _map(arena, create=True)
    if m is None:
        return
    me = int(pid or os.getpid())
    base = _off()
    now = _now_ms()
    with _LOCK:
        _S.seq += 1
        free = oldest = None
        slot = None
        for s in range(_SLOTS):
            off = base + s * _V.size
            p, t, _r, _tr, _q, _sq = _V.unpack(m[off:off + _V.size])
            if p == me:
                slot = s
                break
            if free is None and (not p or now - t > _STALE_MS):
                free = s
            if oldest is None or t < oldest[1]:
                oldest = (s, t)
        if slot is None:
            slot = free if free is not None else oldest[0]
        off = base + slot * _V.size
        m[off:off + _V.size] = _V.pack(me, now, max(0, int(released)), max(0, int(transient)), int(bool(quiet)),
                                       _S.seq)


def verdicts(arena: Any) -> List[Tuple[int, int, int, int, int]]:
    """``[(pid, t_ms, released, transient, quiet)]`` of every live process's latest serve."""
    from sglang.srt.weg2 import room_wait_arbiter as _arb

    m = _map(arena, create=False)
    if m is None:
        return []
    base = _off()
    now = _now_ms()
    out = []
    for s in range(_SLOTS):
        off = base + s * _V.size
        try:
            p, t, r, tr, q, _sq = _V.unpack(m[off:off + _V.size])
        except (ValueError, struct.error):
            continue
        if not p or now - t > _STALE_MS or not _arb._alive(int(p)):
            continue
        out.append((int(p), int(t), int(r), int(tr), int(q)))
    return out


# --- the fill side -----------------------------------------------------------------------------------

def futile(arena: Any, since_ms: int, pp_size: Optional[int] = None) -> Optional[str]:
    """None = room may still come (a rank has not answered since ``since_ms``, released pages, has I/O in flight,
    or did not look). Else the evidence text: every one of the ``pp_size`` P ranks answered after the wait's last
    progress with nothing released, nothing in flight, the spill having looked."""
    if not enabled():
        return None
    if pp_size is None:
        from sglang.srt.weg2 import room_wait_arbiter as _arb

        pp_size = int(_arb._A.pp_size or 0)
    if pp_size <= 0:
        return None
    fresh: Dict[int, Tuple[int, int, int, int, int]] = {}
    for v in verdicts(arena):
        if v[1] > int(since_ms):
            fresh[v[0]] = v
    if len(fresh) < pp_size:
        return None
    if any(r or tr or q for _p, _t, r, tr, q in fresh.values()):
        return None
    return " ".join("pid%d:released=0,transient=0,age_ms=%d" % (p, max(0, _now_ms() - t))
                    for p, t, _r, _tr, _q in sorted(fresh.values()))


def note_futile(stems: int, waited_ms: float, evidence: str) -> None:
    _S.futile += 1
    n = _S.futile
    if n <= 32 or n % 64 == 0:
        logger.warning("%s FUTILE n=%d stems=%d waited_ms=%.0f %s (every P rank answered the room request after this "
                       "wait's last progress: nothing released, nothing in flight -- no slot can free, the wait ends "
                       "now instead of at its idle bound)", MARK, n, int(stems), float(waited_ms), evidence)


def reset_for_tests() -> None:
    _S.seq = 0
    _S.futile = 0
    _S.notes = 0
