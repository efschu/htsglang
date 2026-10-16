"""ROOM-VERDICTS 1016 (F2, NF h16s P-gap root): a room wait that cannot be answered ends on FACTS, not on time.

THE GAP (boot ..._1f4b3e73d4_1016_103428, P): READ-ROOM's room wait ends when no slot came for its idle bound
(``SGLANG_WEG2_READ_ROOM_WAIT_MS``, 10 s), JOIN-LIVENESS-style. At 11:27:25 and 11:27:35 the reads of PP1/PP2 ended
``READ-ROOM SHORT class=served_nothing_to_spill spilled_pages=0`` after ``waited_ms=10019..10456`` each -- every
serve of the whole wait had answered ``released_pages=0`` (``READ-ROOM SERVE ... free=0 clock=0``) and ``W3-ARENA
SPILL candidates=0``: from the first answered round on, nothing could ever have come; the 10 s were pure idleness
with 5 requests queued (25 s without a prefill batch, two waits in a row). Twelve P gaps of 10-13 s since 11:19Z
match the bound.

THE RULE: every serve (``read_room.serve``, the scheduler round of each P rank) writes THIS rank's latest verdict
into the arena's shared ``.room2`` file: pages released, I/O in flight that can make a copy spillable (``transient``,
an O(1) census: write-throughs, store writes, load-backs, direct mamba rows, device-locked rows of a running chunk),
whether the spill was in its empty backoff (``quiet``), whether the serve found room already there or on its way
(``room_seen``), and this process's own arena references (``own_refs``). The fill's room wait ends when, after its
last progress and BEFORE the claim round that followed and gained nothing, every P rank (``pp_size`` distinct live
processes) answered with released=0, transient=0, quiet=0, room_seen=0 -- AND no yielded fill's give-back is pending,
no waiting fill of another request holds landed references, and the P ranks' own references account for every
reference on the arena (no group D holder could still let go). The idle bound stays as the lifesign limit.

Review 1016 (nf-review-f3f2-1016, A1/A2/A7) made it so: before, a serve that found room answered like one that found
none (probe P4), and a yielded fill's references on their way back were invisible (P3/P7).
Switch ``SGLANG_WEG2_ROOM_VERDICTS`` (default on; 0 = the wait byte for byte as before).
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

#: one slot per process: pid, t_ms, released, transient, quiet, serve seq, room_seen, own_refs (-1 = unknown)
_V = struct.Struct("<QQQQQQQq")
_SLOTS = 64
#: pending give-backs of fills that yielded (F3 x F2, review A2): pid, rid hash, t_ms, landed references, active
_G = struct.Struct("<QQQQQ")
_G_SLOTS = 64
REGION_SIZE = _SLOTS * _V.size + _G_SLOTS * _G.size
_STALE_MS = 30_000
_LOCK = threading.Lock()


class _State:
    seq = 0
    futile = 0
    notes = 0
    givebacks: Dict[str, int] = {}


_S = _State


def enabled() -> bool:
    raw = (os.environ.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _off() -> int:
    from sglang.srt.weg2 import room_wait_arbiter as _arb

    return _arb._SIZE_F3


def _g_off() -> int:
    return _off() + _SLOTS * _V.size


def _map(arena: Any, create: bool):
    from sglang.srt.weg2 import room_wait_arbiter as _arb

    return _arb.mapped(arena, create=create)


def _now_ms() -> int:
    return int(time.time() * 1000.0)


# --- the serve side (scheduler round of each P rank) -------------------------------------------------

def transient_blocked(cache: Any) -> int:
    """Review A7: I/O in flight or a device lock that can make a host copy spillable once it ends -- O(1) from the
    tree's own counters (the old walk over every node cost a tree walk per serve on the scheduler thread):
    write-throughs not acked, store writes not acked (they hold host locks), load-backs in flight (host locks),
    direct mamba row writes, and device-locked rows (a running chunk's request: its lock drops when the chunk ends,
    then device node and host copy become spillable). Conservative on purpose: in flight anywhere counts, so a wait
    ends as futile only when NOTHING of the kind is in flight on this rank."""
    n = 0
    for attr in ("ongoing_write_through", "ongoing_backup", "ongoing_load_back", "_weg2_direct_mamba_rows"):
        try:
            n += len(getattr(cache, attr, None) or ())
        except TypeError:
            n += 1
    prot = getattr(cache, "component_protected_size_", None)
    if isinstance(prot, dict):
        try:
            n += sum(1 for v in prot.values() if int(v or 0) > 0)
        except (TypeError, ValueError):
            n += 1
    return n


def spill_quiet(cache: Any) -> bool:
    """The W3 spill is in its empty backoff: a serve now returns 0 without having looked."""
    try:
        return time.monotonic() < float(getattr(cache, "_w3_spill_quiet_until", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False


def own_refs(arena: Any) -> int:
    """This process's reader references on the arena (its RefLedger; -1 = no ledger: unknown)."""
    led = getattr(arena, "_ledger", None)
    if led is None:
        return -1
    try:
        return int(led.held.sum())
    except Exception:  # noqa: BLE001
        return -1


def note_verdict(arena: Any, released: int, transient: int, quiet: bool, pid: Optional[int] = None,
                 room_seen: bool = False, refs: Optional[int] = None) -> None:
    """This rank's latest serve verdict into the shared file (one slot per process). ``room_seen`` (review A1): the
    serve found room already there (free + clock) or on its way (a peer still holds pages this rank let go for the
    read) -- never a futility answer. ``refs``: this process's own arena references (None = read the ledger)."""
    if not enabled():
        return
    m = _map(arena, create=True)
    if m is None:
        return
    me = int(pid or os.getpid())
    held = own_refs(arena) if refs is None else int(refs)
    base = _off()
    now = _now_ms()
    with _LOCK:
        _S.seq += 1
        free = oldest = None
        slot = None
        for s in range(_SLOTS):
            off = base + s * _V.size
            p, t, _r, _tr, _q, _sq, _rs, _h = _V.unpack(m[off:off + _V.size])
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
                                       _S.seq, int(bool(room_seen)), held)


def verdicts(arena: Any) -> List[Tuple[int, int, int, int, int, int, int]]:
    """``[(pid, t_ms, released, transient, quiet, room_seen, own_refs)]`` of every live process's latest serve."""
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
            p, t, r, tr, q, _sq, rs, h = _V.unpack(m[off:off + _V.size])
        except (ValueError, struct.error):
            continue
        if not p or now - t > _STALE_MS or not _arb._alive(int(p)):
            continue
        out.append((int(p), int(t), int(r), int(tr), int(q), int(rs), int(h)))
    return out


# --- pending give-backs (review A2) ------------------------------------------------------------------

def note_giveback(arena: Any, rid: str, landed: int, pid: Optional[int] = None) -> None:
    """A fill of ``rid`` yielded (F3) holding ``landed`` references; the reap of its read gives the ones beyond its
    anchor back. Until this rank's round sees the read reaped, it is room ON ITS WAY: no wait is futile."""
    from sglang.srt.weg2 import room_wait_arbiter as _arb

    m = _map(arena, create=True)
    if m is None:
        return
    me, h, now = int(pid or os.getpid()), _arb.rid_hash(rid), _now_ms()
    base = _g_off()
    with _LOCK:
        free = oldest = None
        slot = None
        for s in range(_G_SLOTS):
            off = base + s * _G.size
            p, gh, t, _n, act = _G.unpack(m[off:off + _G.size])
            if p == me and gh == h:
                slot = s
                break
            if free is None and (not act or now - t > _STALE_MS):
                free = s
            if oldest is None or t < oldest[1]:
                oldest = (s, t)
        if slot is None:
            slot = free if free is not None else oldest[0]
        off = base + slot * _G.size
        m[off:off + _G.size] = _G.pack(me, h, now, max(0, int(landed)), 1)
        _S.givebacks[str(rid)] = h


def givebacks(arena: Any) -> List[Tuple[int, int, int]]:
    """``[(pid, rid hash, landed)]`` pending on every live process."""
    from sglang.srt.weg2 import room_wait_arbiter as _arb

    m = _map(arena, create=False)
    if m is None:
        return []
    base, now, out = _g_off(), _now_ms(), []
    for s in range(_G_SLOTS):
        off = base + s * _G.size
        try:
            p, h, t, n, act = _G.unpack(m[off:off + _G.size])
        except (ValueError, struct.error):
            continue
        if not act or not p or now - t > _STALE_MS or not _arb._alive(int(p)):
            continue
        out.append((int(p), int(h), int(n)))
    return out


def reap_givebacks(arena: Any, cache: Any) -> int:
    """Scheduler round of the rank that yielded: a give-back whose read left ``ongoing_prefetch`` was reaped (the
    #257 cut released what lay beyond the anchor) -- the record goes. Returns the records cleared."""
    if not _S.givebacks:
        return 0
    # pgap re-review A-GB: the reap only QUEUES the release (append_host_mem_release); the references return when the
    # queue is drained (every round, or every N-th under the gated drain) -- not before
    q = getattr(getattr(cache, "cache_controller", None), "host_mem_release_queue", None)
    try:
        if q is not None and not q.empty():
            return 0
    except Exception:  # noqa: BLE001
        return 0
    ongoing = getattr(cache, "ongoing_prefetch", None)
    done = [rid for rid in list(_S.givebacks) if ongoing is None or rid not in ongoing]
    if not done:
        return 0
    m = _map(arena, create=False)
    me = os.getpid()
    base = _g_off()
    with _LOCK:
        for rid in done:
            h = _S.givebacks.pop(rid, None)
            if m is None or h is None:
                continue
            for s in range(_G_SLOTS):
                off = base + s * _G.size
                p, gh, _t, _n, act = _G.unpack(m[off:off + _G.size])
                if p == me and gh == h:
                    m[off:off + _G.size] = _G.pack(0, 0, 0, 0, 0)
    return len(done)


# --- the fill side -----------------------------------------------------------------------------------

def futile(arena: Any, since_ms: int, pp_size: Optional[int] = None, until_ms: Optional[int] = None,
           rid: Optional[str] = None, ignore_holders: bool = False) -> Optional[str]:
    """None = room may still come. Else the evidence text: every one of the ``pp_size`` P ranks answered in
    (``since_ms``, ``until_ms``] -- after the wait's last progress and BEFORE the claim round that followed (review
    A1: a claim round always runs between the answers and the end) -- with nothing released, nothing in flight, the
    spill having looked, no room seen; no yielded fill's give-back is pending (A2); and the P ranks' own references
    account for every reference on the arena (no other holder -- group D -- could still let go)."""
    if not enabled():
        return None
    if pp_size is None:
        from sglang.srt.weg2 import room_wait_arbiter as _arb

        pp_size = int(_arb._A.pp_size or 0)
    if pp_size <= 0:
        return None
    fresh: Dict[int, Tuple[int, ...]] = {}
    for v in verdicts(arena):
        if v[1] > int(since_ms) and (until_ms is None or v[1] <= int(until_ms)):
            fresh[v[0]] = v
    if len(fresh) < pp_size:
        return None
    if any(r or tr or q or rs for _p, _t, r, tr, q, rs, _h in fresh.values()):
        return None
    if givebacks(arena):
        return None
    from sglang.srt.weg2 import room_wait_arbiter as _arb

    if not ignore_holders and _arb.holders(arena, exclude_rid=rid):
        return None   # another request's waiting fill holds landed references: the arbitration can still free them
    held = [h for *_x, h in fresh.values()]
    if any(h < 0 for h in held):
        return None
    rc = getattr(arena, "ref_census", None)
    if callable(rc):
        try:
            _pinned, refs_now, _c = rc()
        except Exception:  # noqa: BLE001
            return None
        if int(refs_now) > sum(held):
            return None
    return " ".join("pid%d:released=0,transient=0,own_refs=%d,age_ms=%d" % (p, h, max(0, _now_ms() - t))
                    for p, t, _r, _tr, _q, _rs, h in sorted(fresh.values()))


def note_futile(stems: int, waited_ms: float, evidence: str) -> None:
    _S.futile += 1
    n = _S.futile
    if n <= 32 or n % 64 == 0:
        logger.warning("%s FUTILE n=%d stems=%d waited_ms=%.0f %s (every P rank answered the room request after this "
                       "wait's last progress and before its last claim round: nothing released, nothing in flight, no "
                       "room seen, no give-back pending, no foreign holder -- no slot can free, the wait ends now "
                       "instead of at its idle bound)", MARK, n, int(stems), float(waited_ms), evidence)


def reset_for_tests() -> None:
    _S.seq = 0
    _S.futile = 0
    _S.notes = 0
    _S.givebacks = {}
