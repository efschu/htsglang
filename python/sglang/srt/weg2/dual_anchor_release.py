"""Q-610 DUAL-ANCHOR-RELEASE: group P of the dual layout gives its mamba arena
references back -- the end anchor once the front ended its rid, the settled
prefix-cache anchors when a claim finds the arena full. Q-650 (bottom of the
module): group D does the same under a cap, and publishes the room it leaves.

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


# -- Q-650 DUAL-ANCHOR-RELEASE-D -----------------------------------------------
# Dual y8v 10031504 (673cc89f6a, 60-min acceptance, 15:18-15:21): Q-610 emptied
# P's share (own_held 49 -> 0-4) and group D filled the arena alone -- every D
# rank held 111-112 of the 112 slots (``ARENA-REF-CENSUS complete=112``, D
# own_held 111-112, refs=335 = 3 D ranks x 111). D never resets either, so its
# tree's host-backed anchors kept their reader references forever. P's END
# anchor of weg2-0-121 found no slot (``WEG2-PUBLISH-CHUNK stopped=mamba_full``
# x6, ``#1421 BACKUP-REFUSED why=anchor_only_claim``), D refused the tail
# (W50), the front re-routed through P leg 1 into the same full arena and the
# second refusal ended W53 -- every long request after 15:20:18, D idle from
# 15:20:53 (``ADMISSION-WEDGE`` 1 queued 0 running).
#
# THE FIX, group D of the dual layout: a capacity rule over D's OWN tree -- a
# rank-uniform quantity (the D ranks run the same matches and inserts, so their
# trees and access orders are replicas) -- plus the shared pin count read from
# the arena header. D keeps at most ``slots - 2*p_room`` references; when the
# arena has fewer than ``p_room`` unpinned slots left for P's claims, or D is
# over its cap, D gives back its least recently used SETTLED anchors that no
# running request holds (lock 0: the host-backed resume state of a cached
# prefix). A released anchor stays COMPLETE and findable by stem until a P
# claim takes the slot (Q-610's semantics); D re-adopts it by its prefetch if
# it is still there.
#
# The room D leaves is published to a tiny /dev/shm file the front reads: a
# rid whose first refusal would send it through P leg 1 again while the arena
# has no room for P's END anchor waits for that room first (ANCHOR-OWED),
# instead of spending a second P prefill into the same full arena (W53).

MARKER_D = "Q-650 DUAL-ANCHOR-RELEASE-D"

#: the D tick runs every this many HICACHE rounds (the round counter is
#: rank-lockstep; ~1 s at group D's measured ~300 rounds/s)
D_TICK_ROUNDS = 256
#: a room file older than this is no reading (D gone, tick stalled)
ROOM_MAX_AGE_S = 30.0
#: the front's ANCHOR-OWED hold: bound and poll period
ANCHOR_OWED_WAIT_S = 60.0
ANCHOR_OWED_POLL_S = 0.5


def armed_d(env=None) -> bool:
    """Group D of the dual layout, the Q-610 switch (SGLANG_WEG2_ENABLE_DUAL_ANCHOR_RELEASE)."""
    e = os.environ if env is None else env
    if (e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() != "1":
        return False
    if (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() != "D":
        return False
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_ENABLE_DUAL_ANCHOR_RELEASE.get())
    except Exception:  # noqa: BLE001
        return True


def p_room(slots: int) -> int:
    """The unpinned slots D leaves for P's claims: 1/8 of the arena, at least 2."""
    return max(2, int(slots) // 8)


def d_cap(slots: int) -> int:
    """The most references D's tree keeps on an arena of `slots`."""
    return max(0, int(slots) - 2 * p_room(slots))


def d_release_need(*, slots: int, pinned: int, d_held: int) -> int:
    """How many of its references D gives back now: down to its cap, and at
    least enough that P finds `p_room` unpinned slots (when D holds that many)."""
    slots, pinned, d_held = int(slots), int(pinned), int(d_held)
    if slots <= 0 or d_held <= 0:
        return 0
    over_cap = max(0, d_held - d_cap(slots))
    short_room = max(0, p_room(slots) - (slots - pinned))
    return min(d_held, max(over_cap, short_room))


def d_release_pass(candidates, *, need: int, releasable, release, age) -> int:
    """Give back up to `need` of `candidates` (D's arena-backed tree nodes),
    least recently used first (`age(node)`, smaller = older), skipping every
    node `releasable(node)` refuses (a running request, a write in flight).
    Returns the references given back."""
    if need <= 0:
        return 0
    done = 0
    for node in sorted(candidates, key=age):
        if done >= need:
            break
        if not releasable(node):
            continue
        release(node)
        done += 1
    return done


def _tag(env=None) -> str:
    e = os.environ if env is None else env
    return e.get("SGLANG_WEG2_DUAL_KV_TAG", "") or e.get("SGLANG_WEG2_TAG", "weg2")


def room_file(tag: Optional[str] = None, root: str = "/dev/shm") -> str:
    import hashlib

    t = _tag() if tag is None else str(tag)
    return os.path.join(root, "wdar-%s.json" % hashlib.sha1(t.encode()).hexdigest()[:10])


def publish_room(*, slots: int, pinned: int, d_held: int, tag: Optional[str] = None,
                 root: str = "/dev/shm", now: Optional[float] = None) -> Optional[str]:
    """D's reading of the arena for the front (atomic replace). None on failure."""
    import json

    path = room_file(tag, root)
    tmp = "%s.%d.tmp" % (path, os.getpid())
    try:
        with open(tmp, "w") as f:
            json.dump({"slots": int(slots), "pinned": int(pinned), "d_held": int(d_held),
                       "t": time.time() if now is None else float(now)}, f)
        os.replace(tmp, path)
        return path
    except OSError:
        return None


def read_room(tag: Optional[str] = None, root: str = "/dev/shm", now: Optional[float] = None,
              max_age_s: float = ROOM_MAX_AGE_S) -> Optional[dict]:
    """The last fresh room reading, or None (absent, unreadable, stale)."""
    import json

    try:
        with open(room_file(tag, root)) as f:
            r = json.load(f)
    except (OSError, ValueError):
        return None
    now = time.time() if now is None else float(now)
    if not isinstance(r, dict) or now - float(r.get("t", 0.0)) > max_age_s:
        return None
    return r


def room_blocked(r: Optional[dict]) -> bool:
    """P's END anchor finds no slot: fewer than 2 unpinned slots. No reading = not blocked."""
    if not r:
        return False
    return int(r.get("slots", 0)) - int(r.get("pinned", 0)) < 2


async def wait_anchor_room(read, *, sleep, clock, max_s: float = ANCHOR_OWED_WAIT_S,
                           poll_s: float = ANCHOR_OWED_POLL_S) -> tuple:
    """ANCHOR-OWED: wait until `read()` shows room for P's END anchor.
    Returns (outcome, held_s): 'free' (never blocked), 'room' (room came),
    'timeout' (bound reached -- the caller proceeds as before)."""
    t0 = clock()
    if not room_blocked(read()):
        return "free", 0.0
    while clock() - t0 < max_s:
        await sleep(poll_s)
        if not room_blocked(read()):
            return "room", clock() - t0
    return "timeout", clock() - t0


_ND = [0]


def log_d(*, released: int, need: int, d_held: int, pinned: int, slots: int, candidates: int) -> None:
    """One line per D tick that gave anything back (all of them up to 64, then every 64th)."""
    if not released:
        return
    _ND[0] += 1
    n = _ND[0]
    if n <= 64 or n % 64 == 0:
        logger.info(
            "%s n=%d released=%d need=%d d_held=%d cap=%d pinned=%d slots=%d p_room=%d "
            "candidates=%d (D's tree gives its least recently used settled anchors back -- "
            "the dual layout never resets D; the pages stay COMPLETE until a P claim needs the slot)",
            MARKER_D, n, released, need, d_held, d_cap(slots), pinned, slots, p_room(slots), candidates)


class OncePer:
    """Log-throttle by key: True the first time a key is seen (bounded memory,
    oldest key dropped) and on every `every`-th repeat after; `count(key)` the
    occurrences so far."""

    def __init__(self, every: int = 4096, cap: int = 4096) -> None:
        import collections

        self.every, self.cap = int(every), int(cap)
        self.seen = collections.OrderedDict()

    def __call__(self, key) -> bool:
        c = self.seen.pop(key, 0) + 1
        self.seen[key] = c
        while len(self.seen) > self.cap:
            self.seen.popitem(last=False)
        return c == 1 or (self.every > 0 and c % self.every == 0)

    def count(self, key) -> int:
        return int(self.seen.get(key, 0))
