"""Q-610 DUAL-ANCHOR-RELEASE: group P of the dual layout gives its mamba arena
references back -- the end anchor once the front ended its rid, the settled
prefix-cache anchors when a claim finds the arena full.

THE CLASS (dual y8t 10031114, 5d12a95e92, 60-min acceptance; same chain on
0145 4a632a3a94 and 1009 55c95a89c7, never under that much load): the mamba
arena holds 112 anchor slots shared by P and D. A tree node keeps ONE reader
reference per anchor its host value addresses; the only point that gives them
back is the reset before P's sleep (``_release_host_values_before_reset``,
H81), with the end anchors held one D phase longer (CARRIER-HOLD, released at
P's wake). The dual layout never resets P while it serves. From 11:24:25 on
P PP0 held 92 of the 112 slots (``ARENA-REF-HOLDERS ... arena-78446592.bin
tree=92 ... arena_pinned=112``), every claim's room-making found nothing
(``#1427 ARENA-DROP need=1 freed=0`` x429, ``ARENA-CLAIM REFUSED statuses=[4]``),
every new END anchor was refused (``#1421 BACKUP-REFUSED why=mamba_claim``,
then ``anchor_only_claim``; ``P-FUND EVICT KV-ONLY``), D found the tail's
KV but no anchor in range (``#1028B FETCH CAP ... claimed=0``, ``#1035c
ZERO-ANSWER ... CAPPED by=mamba``), W31/W50, the front re-routed through P
leg 1 into the same full arena, and the second refusal ended as W35 (long) or
W53 (short) -- every request after 11:24:25.

THE FIX, bound to consumption (the #243 rule, handoff_pending.py), group P of
the dual layout only:

* **retain** (:func:`release_ended`, called at every retain publish): an END
  anchor registered at its END-ANCHOR mark gives its tree reference back once
  its rid ENDED at the front (the terminal rid-end tombstone,
  ``handoff_pending.ended``) -- D does not read it any more -- or once it is
  neither pending nor ended for longer than the hand-off expire bound (the
  module's own garbage rule). A hand-off still pending is never touched.
* **claim** (:func:`claim_room`, a refused mamba claim): the same, then every
  SETTLED anchor of a node no running request holds (device lock 0) that is
  not an end anchor of a live rid -- forks and grid anchors, i.e. P's prefix
  cache. One pass, the same set on every PP rank (the trees are replicas), so
  the slots free once the last rank gave its reference.

A released anchor stays COMPLETE in the arena and findable by stem; the
claim's room-making (``_evict_for_claim``) takes it only when it needs the
slot, and writes it to L3 first (#257 d). Host bookkeeping only: one ``free``
per node, a few ``stat`` calls, no device work, no collective.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

MARKER = "Q-610 DUAL-ANCHOR-RELEASE"


def armed(env=None) -> bool:
    """Group P of the dual layout, switch on (SGLANG_WEG2_ENABLE_DUAL_ANCHOR_RELEASE)."""
    e = os.environ if env is None else env
    if (e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() != "1":
        return False
    if (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() != "P":
        return False
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_ENABLE_DUAL_ANCHOR_RELEASE.get())
    except Exception:  # noqa: BLE001
        return True


def _expire_s() -> float:
    try:
        from sglang.srt.weg2 import handoff_pending as _hp

        return float(_hp._expire_s())
    except Exception:  # noqa: BLE001
        return 900.0


def rid_done(rid: Optional[str], registered_at: Optional[float] = None, now: Optional[float] = None) -> bool:
    """D will not read rid's end anchor any more: the front ended the rid, or
    the rid is neither pending nor ended past the hand-off expire bound. A
    pending hand-off is never done. Unknown rid (None) is never done."""
    if not rid:
        return False
    try:
        from sglang.srt.weg2 import handoff_pending as _hp

        if _hp.status(rid).get("state") == "pending":
            return False
        if _hp.ended(rid):
            return True
    except Exception:  # noqa: BLE001 -- in doubt keep the reference
        return False
    if registered_at is None:
        return False
    now = time.monotonic() if now is None else now
    return (now - float(registered_at)) > _expire_s()


class Registry:
    """rid -> (end-anchor node, registration time) of this rank, from the
    END-ANCHOR mark to the release. Bounded by the rids the front has not
    ended yet (plus the expire bound)."""

    def __init__(self) -> None:
        self.entries: dict = {}

    def note(self, rid: str, node) -> None:
        if rid and node is not None:
            self.entries[str(rid)] = (node, time.monotonic())

    def done(self, now: Optional[float] = None) -> list:
        """[(rid, node)] whose rid is done (removed from the registry)."""
        out = []
        for rid, (node, t0) in list(self.entries.items()):
            if rid_done(rid, t0, now):
                out.append((rid, node))
                del self.entries[rid]
        return out


_N = [0]


def log_batch(*, at: str, released: int, ended: int, prefix: int, kept_pending: int,
              claim_ok: Optional[bool] = None, depth: Optional[int] = None) -> None:
    """One line per batch that released anything (rate-limited on the retain
    path, always on the claim path)."""
    if not released and at != "claim":
        return
    _N[0] += 1
    n = _N[0]
    if at == "claim" or n <= 16 or n % 64 == 0:
        logger.info(
            "%s n=%d at=%s released=%d ended=%d prefix=%d kept_pending=%d%s%s (P's tree gives "
            "its mamba arena references back -- the dual layout never resets P; the pages stay "
            "COMPLETE in the arena until a claim needs the slot, #257 writes them to L3 first)",
            MARKER, n, at, released, ended, prefix, kept_pending,
            "" if claim_ok is None else f" claim={'ok' if claim_ok else 'refused'}",
            "" if depth is None else f" depth={depth}")
