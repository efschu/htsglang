"""TOLD-ANCHOR-HOLD (dual y9d4, 04.10. 09:34:14Z, b73e808a0f): between the
moment a request's TOLD is on a rank and its ADMISSION, nothing in the tree
held the anchor the told names. Group P of the dual layout queues a request for
seconds (P at 25 % SM, 4 s on y9d4) and in that window the tree moves on:

  09:34:10  PP0 told=22528 for pdflip-0-77; PP1/PP2 "#1400 FOLLOWER SATISFIED
            LOCALLY told=22528" and stamp ``#1042 EXTENT set extent=22528``
  09:34:10  END-ANCHOR of pdflip-0-72 (anchor=26955) -- every rank's insert runs
            the per-path cap (``PDFLIP PATH-CAP``, cap=4)
  09:34:11  PP1/PP2 stamp extent=16665 (the next END anchor up the path)
  09:34:14  PP1 loads 16665, adopts the GDN anchor at 16665 -> ``#968 PREFIX
            MATERIALISATION SHORTFALL`` (deficit 22528) -> DEBUG-HOLD, W17.

The told is PP0's decision, every rank admits at it, and an anchor the tree
gave up in between is a shortfall no rank can repair (raenge-nie-uneins: the
only safe form is that the tree does not give it up).

THE HOLD. The registry is the UNION of the scheduler's own told records
(:class:`ToldView`): ``_pdflip_store_told`` (written at the Admit), and -- the
paced/PF form of y9d4 (FLLIPER_PDFLIP_TOLD_PACED=1) writes nothing else until the
Admit -- ``_pdflip_told_pacing`` (PP0, read-ahead published), ``_pdflip_told_early``
(follower, read-ahead absorbed) and ``_pdflip_store_told_satisfied`` (follower held
the span). They are born where the told is born (PP0's publish, the follower's
absorb) and die together (the Admit pops them, ``Q-580 TOLD-FORGET`` / the abort
drop all of ``_TOLD_RECORDS``) -- no second lifecycle to keep in step. The tree reads it through :class:`Hold` and excludes from
every anchor-GIVING-UP decision the anchor whose END depth is a told of a
request still standing between told and admission:

* the per-path cap (``_pdflip_cap_path_states``: the held anchor is skipped like
  a fork or an END anchor, the cap takes the next eligible one);
* Q-610 (``_pdflip_dual_releasable``: neither the claim-room walk nor the
  END-anchor release gives its reference back);
* the P inner-anchor release (``_pdflip_release_inner_anchor``).

A tree reference an anchor keeps is also what keeps its arena slot out of the
claim-time ``ARENA-DROP`` (stage i takes only slots NO reader references): the
exemption at the three funnels IS the ARENA-DROP exemption -- the arena does
not know trees, so a guard there would have to guess.

RANK-UNIFORM BY CONSTRUCTION. The decision reads (a) the told records, whose
CONTENT is the ring's -- PP0 publishes, every follower absorbs the same
rid -> told (the follower's record starts one ring hop after PP0's and ends one
hop after PP0's Admit: at the start that is the one residual window, see the
report; the MAX bound picks by (-told, rid), never by the rank-local dict order) -- (b) the node's END depth (inserts), and
(c) a COUNT of cap runs (one per anchoring insert, the same step on every
rank). No clock, no lock, no arena state, no rank-local copy. The two bounds
against a standing block are counts too: at most ``MAX`` held anchors at once
(by (-told, rid) -- content, never the rank-local order of a dict) and a hold lasts at most
``RUNS`` cap runs; past either the OLD way applies and a named WARNING says so.

DEPTH-ONLY MATCH. The hold names a depth, not a node: a request's anchor at
told T is found by END depth on the path the cap/release walks. Another path
carrying an anchor at exactly T is held too (at most one extra anchor per held
told, bounded by MAX) -- cheaper and safer than resolving the node by a tree
walk that would split nodes and touch the LRU.

Default ON in the dual layout, group P only; ``FLLIPER_PDFLIP_DUAL_TOLD_ANCHOR_HOLD
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

ENV = "FLLIPER_PDFLIP_DUAL_TOLD_ANCHOR_HOLD"
ENV_MAX = "FLLIPER_PDFLIP_DUAL_TOLD_ANCHOR_HOLD_MAX"
ENV_RUNS = "FLLIPER_PDFLIP_DUAL_TOLD_ANCHOR_HOLD_RUNS"
DEFAULT_MAX = 24
DEFAULT_RUNS = 256
MARKER = "#1400 TOLD-ANCHOR-HOLD"
TAKES_KEEP = 64


def dual_p(env=None) -> bool:
    """Group P of the dual layout (the gate every dual fix lives behind)."""
    e = os.environ if env is None else env
    return ((e.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() == "1"
            and (e.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper() == "P")


def armed(env=None) -> bool:
    """The hold switch: dual P and FLLIPER_PDFLIP_DUAL_TOLD_ANCHOR_HOLD not 0."""
    e = os.environ if env is None else env
    if not dual_p(e):
        return False
    return (e.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _int(env, name: str, default: int) -> int:
    try:
        return max(0, int((env.get(name, "") or "").strip() or default))
    except ValueError:
        return default


#: the scheduler's told records a told lives in from PP0's publish to the admission, in
#: priority order (the first record that names a rid wins -- all four carry the same value for
#: one rid; the admitted table is the final one):
#:   _pdflip_store_told            rid -> told   PP0 at the Admit's publish, follower at the Admit's absorb
#:   _pdflip_told_pacing           rid -> _Pace  PP0 from the read-ahead's publish to the Admit (``.told``)
#:   _pdflip_told_early            rid -> told   follower from the read-ahead's absorb to the Admit's absorb
#:   _pdflip_store_told_satisfied  rid -> told   follower that held the told span locally, until admission
#:   _pdflip_told_kept             rid -> _Kept  y9d4c (75 % SM, 17 s in the queue): the adder's FIRST visit
#:                                             (p_intake.told_admission) POPS the four records above and keeps
#:                                             the verdict here (``.told``) until settle_told sees the rid
#:                                             leave the queue -- on a follower nothing else stands then
TOLD_RECORDS = ("_pdflip_store_told", "_pdflip_told_pacing", "_pdflip_told_early", "_pdflip_store_told_satisfied",
                "_pdflip_told_kept")


class ToldView:
    """Read-only, dict-like union of the scheduler's told records (:data:`TOLD_RECORDS`).

    WHY NOT ``_pdflip_store_told`` ALONE (reviewer, y9d4 paced/PF form, FLLIPER_PDFLIP_TOLD_PACED=1):
    the paced form writes ``_pdflip_store_told`` only at the Admit. Between the read-ahead and the
    Admit the told stands in ``_pdflip_told_pacing`` (PP0) and ``_pdflip_told_early`` /
    ``_pdflip_store_told_satisfied`` (follower) -- and the END-anchor insert of pdflip-0-72 (the cap
    run) fell INSIDE that window (P 29035 SATISFIED -> 29067 END-ANCHOR -> PACED-ADMIT ABSORBED
    later). The records are looked up by NAME on every read (the scheduler creates the dicts
    lazily). Their lifecycle is already one: ``_TOLD_RECORDS`` / Q-580 TOLD-FORGET drop all of them
    when the request leaves the queue, the Admit pops pacing/early/satisfied."""

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def _merged(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for attr in TOLD_RECORDS:
            d = getattr(self.scheduler, attr, None)
            if not isinstance(d, dict):
                continue
            for rid, v in list(d.items()):
                if rid in out:
                    continue
                t = getattr(v, "told", v)   # _Pace carries it as an attribute
                try:
                    out[rid] = int(t)
                except (TypeError, ValueError):
                    continue
        return out

    def items(self):
        return list(self._merged().items())

    def get(self, rid, default=None):
        return self._merged().get(rid, default)

    def __contains__(self, rid) -> bool:
        return rid in self._merged()

    def __iter__(self):
        return iter(self._merged())

    def __len__(self) -> int:
        return len(self._merged())


def view_of(scheduler) -> ToldView:
    """The scheduler's one view (stable identity: ``attach`` is idempotent on it)."""
    v = getattr(scheduler, "_pdflip_told_view", None)
    if v is None:
        v = ToldView(scheduler)
        try:
            scheduler._pdflip_told_view = v
        except Exception:  # noqa: BLE001
            pass
    return v


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
        # sorted DEEPEST told first, then rid: the records' own order is rank-local (PP0's pacing
        # dict vs the follower's early dict), so the MAX bound picks by CONTENT, the same set on
        # every rank -- and past MAX it is the SHALLOW (cheap to redo) anchors that fall out, never
        # the deepest (y9d4c: queue 13-16, ascending order dropped the most expensive tolds)
        live = sorted(((r, int(t)) for r, t in list(self.source.items()) if int(t) > 0),
                      key=lambda x: (-x[1], x[0]))
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
        >= told to < told while the told still stands (told .. admission). ``extent``
        None = the ``hitless_clear`` form (y9d4c: the anchor was gone, the match found no
        host hit, the stamp cleared instead of setting a smaller number) counts as a fall
        to 0. Returns True when it warned."""
        rid = str(rid)
        told = self.source.get(rid)
        if told is None:
            self.last_extent.pop(rid, None)
            return False
        told = int(told)
        prev = self.last_extent.get(rid)
        now = 0 if extent is None else int(extent)
        cleared = extent is None
        if cleared:
            self.last_extent.pop(rid, None)      # nothing stands any more; a later stamp re-arms
        else:
            self.last_extent[rid] = now
        if len(self.last_extent) > 4096:
            for r in [r for r in self.last_extent if r not in self.source]:
                self.last_extent.pop(r, None)
        if prev is None or prev < told or now >= told or ("ext", rid) in self._warned:
            return False
        self._warned.add(("ext", rid))
        same = [t for t in self.takes if t[2] == told]
        recent = same or list(self.takes)[-4:]
        logger.warning(
            "#1042 EXTENT REGRESSED rid=%s told=%d extent %d -> %s%s between told and admission "
            "(hold=%s): the anchor at %d was given up. Takes at that depth: %s%s",
            rid, told, prev, "None" if cleared else now, " (hitless_clear)" if cleared else "",
            "on" if self.armed() else "OFF", told,
            ["%s node=%s depth=%s end_anchor=%s" % t for t in same] or "none recorded",
            "" if same else " ; last takes (other depths, or an unattributed path: ARENA-DROP / "
            "evict / LRU): %s" % (["%s node=%s depth=%s end_anchor=%s" % t for t in recent],))
        return True


def attach(tree, source, env=None) -> Optional[Hold]:
    """Give ``tree`` its hold over ``source`` (the scheduler's told table).
    Dual P only; a no-op (None) everywhere else. Idempotent."""
    if tree is None or source is None or not dual_p(env):
        return None
    cur = getattr(tree, "_pdflip_told_hold", None)
    if cur is not None and cur.source is source:
        return cur
    h = Hold(source, env)
    try:
        tree._pdflip_told_hold = h
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
