"""TOLD-ANCHOR-HOLD (dual y9d4, 04.10. 09:34:14Z, b73e808a0f): between the
moment a request's TOLD is on a rank and its ADMISSION, nothing in the tree
held the anchor the told names. Group P of the dual layout queues a request for
seconds (P at 25 % SM, 4 s on y9d4) and in that window the tree moves on:

  09:34:10  PP0 told=22528 for weg2-0-77; PP1/PP2 "#1400 FOLLOWER SATISFIED
            LOCALLY told=22528" and stamp ``#1042 EXTENT set extent=22528``
  09:34:10  END-ANCHOR of weg2-0-72 (anchor=26955) -- every rank's insert runs
            the per-path cap (``WEG2 PATH-CAP``, cap=4)
  09:34:11  PP1/PP2 stamp extent=16665 (the next END anchor up the path)
  09:34:14  PP1 loads 16665, adopts the GDN anchor at 16665 -> ``#968 PREFIX
            MATERIALISATION SHORTFALL`` (deficit 22528) -> DEBUG-HOLD, W17.

The told is PP0's decision, every rank admits at it, and an anchor the tree
gave up in between is a shortfall no rank can repair (raenge-nie-uneins: the
only safe form is that the tree does not give it up).

THE HOLD. The registry is the scheduler's own told table
(``scheduler._weg2_store_told``: rid -> told, written at PP0's publish and at
the follower's absorb, popped at the admission, at ``Q-580 TOLD-FORGET`` and at
the abort) -- no second lifecycle to keep in step: the entry is born and dies
where the told is. The tree reads it through :class:`Hold` and excludes from
every anchor-GIVING-UP decision the anchor whose END depth is a told of a
request still standing between told and admission:

* the per-path cap (``_weg2_cap_path_states``: the held anchor is skipped like
  a fork or an END anchor, the cap takes the next eligible one);
* Q-610 (``_weg2_dual_releasable``: neither the claim-room walk nor the
  END-anchor release gives its reference back);
* the P inner-anchor release (``_weg2_release_inner_anchor``).

A tree reference an anchor keeps is also what keeps its arena slot out of the
claim-time ``ARENA-DROP`` (stage i takes only slots NO reader references): the
exemption at the three funnels IS the ARENA-DROP exemption -- the arena does
not know trees, so a guard there would have to guess.

RANK-UNIFORM BY CONSTRUCTION. The decision reads (a) the told table, whose
order and content are the ring's -- PP0 publishes, every follower absorbs the
same rid -> told in the same order -- (b) the node's END depth (inserts), and
(c) a COUNT of cap runs (one per anchoring insert, the same step on every
rank). No clock, no lock, no arena state, no rank-local copy. The two bounds
against a standing block are counts too: at most ``MAX`` held anchors at once
(the oldest told entries first, i.e. insertion order) and a hold lasts at most
``RUNS`` cap runs; past either the OLD way applies and a named WARNING says so.

DEPTH-ONLY MATCH. The hold names a depth, not a node: a request's anchor at
told T is found by END depth on the path the cap/release walks. Another path
carrying an anchor at exactly T is held too (at most one extra anchor per held
told, bounded by MAX) -- cheaper and safer than resolving the node by a tree
walk that would split nodes and touch the LRU.

Default ON in the dual layout, group P only; ``SGLANG_WEG2_DUAL_TOLD_ANCHOR_HOLD
=0`` is the old way. Everywhere else (flip, NF, INT8, group D, no dual layout)
:func:`attach` is a no-op and the tree never sees a hold.

F3 INSTRUMENTS (live in the same group-P gate, independent of the switch):
``#1042 EXTENT REGRESSED`` when a request's extent falls below its told, with the
last anchor takes at that depth (path, node, depth, END anchor y/n); the
PATH-CAP line for every run that touches a held depth.
"""
from __future__ import annotations

import collections
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_DUAL_TOLD_ANCHOR_HOLD"
ENV_MAX = "SGLANG_WEG2_DUAL_TOLD_ANCHOR_HOLD_MAX"
ENV_RUNS = "SGLANG_WEG2_DUAL_TOLD_ANCHOR_HOLD_RUNS"
DEFAULT_MAX = 8
DEFAULT_RUNS = 256
MARKER = "#1400 TOLD-ANCHOR-HOLD"
TAKES_KEEP = 64


def dual_p(env=None) -> bool:
    """Group P of the dual layout (the gate every dual fix lives behind)."""
    e = os.environ if env is None else env
    return ((e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() == "1"
            and (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "P")


def armed(env=None) -> bool:
    """The hold switch: dual P and SGLANG_WEG2_DUAL_TOLD_ANCHOR_HOLD not 0."""
    e = os.environ if env is None else env
    if not dual_p(e):
        return False
    return (e.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _int(env, name: str, default: int) -> int:
    try:
        return max(0, int((env.get(name, "") or "").strip() or default))
    except ValueError:
        return default


class Hold:
    """The tree's view of the told table of its scheduler (rank-local object,
    rank-uniform content). ``source``: rid -> told."""

    def __init__(self, source: Dict[str, int], env=None) -> None:
        self.source = source
        self._env = os.environ if env is None else env
        self.age: Dict[str, int] = {}
        self.expired: set = set()
        self.takes: collections.deque = collections.deque(maxlen=TAKES_KEEP)
        self.last_extent: Dict[str, int] = {}
        self._warned: set = set()
        self.expired_total = 0

    # -- the exemption ------------------------------------------------------
    def armed(self) -> bool:
        return armed(self._env)

    def depths(self, tick: bool = False) -> Dict[int, List[str]]:
        """{END depth: [rids]} of the anchors the hold keeps NOW (empty when the
        switch is off). ``tick`` = one cap run passed (the uniform clock)."""
        if not self.armed():
            return {}
        mx = _int(self._env, ENV_MAX, DEFAULT_MAX)
        runs = _int(self._env, ENV_RUNS, DEFAULT_RUNS)
        out: Dict[int, List[str]] = {}
        live = [(r, int(t)) for r, t in list(self.source.items()) if int(t) > 0]
        if tick:
            for stale in [r for r in self.age if r not in self.source]:
                self.age.pop(stale, None)
                self.expired.discard(stale)
        taken = 0
        for rid, told in live:
            if rid in self.expired:
                continue
            if tick:
                a = self.age.get(rid, 0) + 1
                self.age[rid] = a
                if runs and a > runs:
                    self.expired.add(rid)
                    self.expired_total += 1
                    logger.warning(
                        "%s EXPIRED rid=%s told=%d after %d cap runs: the anchor is not held any "
                        "more (the old way applies; the request has stood between told and "
                        "admission for too long) (expired_total=%d)",
                        MARKER, rid, told, a, self.expired_total)
                    continue
            if taken >= mx:
                if tick and ("max", rid) not in self._warned:
                    self._warned.add(("max", rid))
                    logger.warning(
                        "%s OVER-MAX rid=%s told=%d: %d anchors are held already, this one is not "
                        "(the old way applies; MAX=%d)", MARKER, rid, told, taken, mx)
                continue
            taken += 1
            out.setdefault(told, []).append(rid)
        if len(self._warned) > 4096:
            self._warned.clear()
        return out

    def told_depths(self) -> Dict[int, List[str]]:
        """{told: [rids]} of every told in the table, hold switch or not (the
        F3 instruments' view)."""
        out: Dict[int, List[str]] = {}
        for rid, told in list(self.source.items()):
            if int(told) > 0:
                out.setdefault(int(told), []).append(rid)
        return out

    # -- F3 instruments ------------------------------------------------------
    def note_take(self, kind: str, node, depth: Optional[int], end_anchor: bool) -> None:
        """An anchor was given up (cap / Q-610 / inner release): kept for the
        EXTENT-REGRESSED line. ``depth`` None = not computed."""
        self.takes.append((kind, getattr(node, "id", "?"), depth, bool(end_anchor)))

    def check_extent(self, rid, extent: Optional[int]) -> bool:
        """``#1042 EXTENT`` stamp of ``rid``: WARN once per rid when it falls from
        >= told to < told while the told still stands (told .. admission).
        Returns True when it warned."""
        rid = str(rid)
        told = self.source.get(rid)
        if told is None or extent is None:
            self.last_extent.pop(rid, None)
            return False
        told = int(told)
        prev = self.last_extent.get(rid)
        self.last_extent[rid] = int(extent)
        if len(self.last_extent) > 4096:
            for r in [r for r in self.last_extent if r not in self.source]:
                self.last_extent.pop(r, None)
        if prev is None or prev < told or int(extent) >= told or ("ext", rid) in self._warned:
            return False
        self._warned.add(("ext", rid))
        same = [t for t in self.takes if t[2] == told]
        recent = same or list(self.takes)[-4:]
        logger.warning(
            "#1042 EXTENT REGRESSED rid=%s told=%d extent %d -> %d between told and admission "
            "(hold=%s): the anchor at %d was given up. Takes at that depth: %s%s",
            rid, told, prev, int(extent), "on" if self.armed() else "OFF", told,
            ["%s node=%s depth=%s end_anchor=%s" % t for t in same] or "none recorded",
            "" if same else " ; last takes (other depths, or an unattributed path: ARENA-DROP / "
            "evict / LRU): %s" % (["%s node=%s depth=%s end_anchor=%s" % t for t in recent],))
        return True


def attach(tree, source, env=None) -> Optional[Hold]:
    """Give ``tree`` its hold over ``source`` (the scheduler's told table).
    Dual P only; a no-op (None) everywhere else. Idempotent."""
    if tree is None or source is None or not dual_p(env):
        return None
    cur = getattr(tree, "_weg2_told_hold", None)
    if cur is not None and cur.source is source:
        return cur
    h = Hold(source, env)
    try:
        tree._weg2_told_hold = h
    except Exception:  # noqa: BLE001 -- a double that refuses attributes holds nothing
        return None
    register_active(h)
    logger.info("%s attached armed=%d max=%d runs=%d (the anchor a told names is kept from "
                "told to admission; %s)", MARKER, int(h.armed()),
                _int(h._env, ENV_MAX, DEFAULT_MAX), _int(h._env, ENV_RUNS, DEFAULT_RUNS), ENV)
    return h


_ACTIVE: Optional[Hold] = None


def register_active(h: Optional[Hold]) -> None:
    """The process-wide hold the extent stamp reports to (one scheduler per process)."""
    global _ACTIVE
    _ACTIVE = h


def report_extent(req, extent) -> None:
    """Called by the ``#1042`` extent stamp: never raises, free when no hold exists."""
    h = _ACTIVE
    if h is None:
        return
    try:
        h.check_extent(getattr(req, "rid", None), extent)
    except Exception:  # noqa: BLE001 -- an instrument never breaks a match walk
        pass
