"""PF (26.09.): the GROUP told=0 fallback of the paced told (#1416e).

THE FAILURE IT REMOVES. With ``FLLIPER_PDFLIP_TOLD_PACED`` PP0 publishes a
read-ahead ``PdFlipStoreTold(paced=True)`` and, once its pacing window (an
ESTIMATE of the followers' store read, capped at 10 s) has passed, the
membership verdict ``PdFlipStoreAdmit(rid, told)``. A follower whose read is
still running at the Admit falls into the bounded busy-wait of
``pdflip_store_told.admission`` -- the whole stage stops -- and after
``WAIT_CAP_S`` raises ``#1400 STORE-TOLD WAIT EXCEEDED``: the group dies. A
read that terminated SHORT (prefetch policy timeout, a declined
registration) dies one line later as ``STORE-TOLD MISMATCH``.

WHY PP0 MUST DECIDE. On the carrierless form told IS the membership signal
and every rank plans locally; a follower that skipped (or admitted at 0) on
its own would plan a different batch -- the W27 width split. So the only
legal rescue is PP0 switching the request to told=0 for EVERY rank, and for
that PP0 needs each follower's read state, which no live channel carried
(#1175's return trip has no caller; the output ring only runs on passes with
a batch; the #1268 home hop belongs to the idle vote).

THE CHANNEL (``PDFLIP_TOLD_ACK_TAG``). Each follower sends PP0 a
``PdFlipToldReadAck(rank, seq, reads=[(rid, own_prefix), ...])`` for every
paced rid whose read has terminated, point to point on the gloo world group,
on its OWN tag (tag 0 carries the typed proxy/output channel, demultiplexed
in band -- a standing receive there would misframe it; 1268 is the vote's).
Size-then-payload, exactly ``send_home``'s framing, so PP0 receives with
``ObjectRecvFrame`` -- one standing frame per follower, driven by
``ObjectRecvFrame.poll``: a completion-flag read per pass, no join, no
WARNING per expiry, the receive stays posted (the two alternatives are
measured dead on this build: ``Work.is_completed()`` never turns True --
corpse F -- and ``wait(timeout=)`` closes the gloo pair, #829). The sender
never blocks either: one message in flight per follower, each ``isend``
parked on its own thread (``ParkedWait``), completion read as a flag in a
later pass; new reads wait in an outbox meanwhile.

``own_prefix`` is computed exactly as ``admission`` will compare it
(completed prefix, plus the registered head on a twin/absolute told, told
itself for a follower satisfied locally), WITHOUT consuming anything.

PP0'S VERDICT, per paced rid, at the top of each pass (after the harvest):
  * every follower acked and every ``own == told``  -> ``Admit(told)`` now,
    even before the pacing window ends (the ack is the fact the window only
    estimated);
  * some follower acked ``own != told``             -> ``Admit(0)`` now
    (its admission would raise STORE-TOLD MISMATCH);
  * the Frist passed with an ack still missing       -> ``Admit(0)``;
  * otherwise                                        -> keep skipping.
``Admit(0)`` carries the wire attribute ``fallback=1``. PP0 applies it in the
same pass (its own store record is released, so its admission compares 0
with 0); every follower applies it in the pass that absorbs it: its read is
cut through the upstream abort path (``tree.release_aborted_request``: the
operation is terminated, the host lock dropped, the rows go back through the
host release queue -- whose arena rows return their reader references with
``FLLIPER_HICACHE_ARENA_QUEUE_REFS``, release-table row 30 / #718-stray), its
records and span pin are dropped, its twin mark taken. All ranks then admit
the request with told=0 -- P recomputes the prefix itself -- and the prefix
cap is 0 on every rank. Nobody waits: the admission loop finds no read.

THE FRIST (from the read-ahead's publication; PP0-local, never on the wire):
    min(window + GRACE_S, CAP_S), and never later than intake + TOTAL_S
GRACE 2 s, CAP 12 s, TOTAL 16 s by default -- PP0's own read time is on the
#699 admission-wedge clock too (20 s of queued / 0 running), so the told=0
recompute must start before that alarm.

WIRE, AND WHY "OFF" IS BYTE-IDENTICAL. Nothing new rides the request wire
when the switch is off: the two markers are INSTANCE attributes set only in
the armed path (``ack`` on the paced read-ahead: "send me your read state";
``fallback`` on a told-0 Admit), so the pickled dataclasses are unchanged
otherwise; the ack stream does not exist. Followers follow the wire, never
their own env (the #1416e rule): a follower acks only read-aheads that carry
``ack``, and releases only on an Admit that carries ``fallback``.

SWITCH ``FLLIPER_PDFLIP_TOLD_GROUP_FALLBACK`` (unset = the profile row: ON on
nextflash since 14.10., off on qwen27b), read once on PP0,
effective only with ``FLLIPER_PDFLIP_TOLD_PACED`` on: only there does PP0 hold
admission back until an Admit.

H64 (NF h16n 17c2563647, boot ..._1015_042125, P 04:30:45 pdflip-6-44 and
04:31:14 pdflip-8-47 -- 130k tokens recomputed the store held). Three parts,
each PP0-only or wire-following, so no rank can leave its peers:

* FRIST-PASSES (``FLLIPER_PDFLIP_TOLD_FALLBACK_MIN_PASSES``, default pp_size+1;
  0 = the wall Frist alone). The Frist was WALL time, but an ack can only
  move in PASSES: the read-ahead reaches stage k k hops down the request
  chain, the follower acks in the pass that absorbs it, and PP0 takes the ack
  in two of its own passes (the size frame, then the payload irecv it posts
  only on seeing the size -- gloo completes a send only against a posted
  receive). While P recomputes, one pass is 3-7 s (PP0 pass_ms=5201 and 6879,
  PP1/PP2 ``CHAIN-RECV blocked 2131 ms``): the 2.16 s / 9.88 s Frists expired
  after one / three PP0 passes, before any ack could have arrived. The Frist
  now passes only once the wall time is up AND PP0 has run ``MIN_PASSES``
  passes since the read-ahead -- a PP0-local count; nobody else decides.
* ACK-DRAIN (``FLLIPER_PDFLIP_TOLD_ACK_DRAIN``, default on). PP0 harvested the
  ack stream only while it had a paced rid open; an ack whose size frame was
  taken in the deciding pass left its payload unposted, the follower's send
  never completed and every later ack of that follower queued behind it --
  PP2's pdflip-8-47 ack surfaced 3.5 min later as ``PF TOLD-ACK LATE``. PP0 now
  polls the standing frames on every pass once the stream exists (a flag read
  per follower).
* GROUP-FLOOR (``FLLIPER_PDFLIP_TOLD_GROUP_FLOOR``, default on). An ack != told
  used to mean told=0. When EVERY follower acked and all acks name the SAME
  depth ``m`` with ``0 < m < told`` -- each ack is that stage's own resumable
  reach (``own_prefix``: rank_resumable / OVER-UNRESUMABLE / NO-ROOM) -- and
  PP0's own TF verdict admits ``m``, PP0 admits ``m`` for every rank:
  ``Admit(m)`` with the wire marker ``floor`` (instance attribute, armed path
  only). Every rank applies it alike (``apply_floor``): no read is released,
  the credit is popped with the span pin kept, the rid is SATISFIED at ``m``
  -- the admission caps its match at ``m`` and the row materialises it. Acks
  that differ from each other, an ack of 0 or above told, a missing ack at the
  Frist, or PP0's TF refusing ``m``: told=0 as before. The floor is the
  group-agreed minimum each stage PROVES it holds; a stage that acked more
  than ``m`` proved nothing about ``m``, so differing acks are never floored.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from flliper.srt.managers.pdflip_idle_vote import FIXED_P2P_TAG_BASE

logger = logging.getLogger(__name__)

ENV_FALLBACK = "FLLIPER_PDFLIP_TOLD_GROUP_FALLBACK"
ENV_GRACE_S = "FLLIPER_PDFLIP_TOLD_FALLBACK_GRACE_S"
ENV_CAP_S = "FLLIPER_PDFLIP_TOLD_FALLBACK_CAP_S"
ENV_TOTAL_S = "FLLIPER_PDFLIP_TOLD_FALLBACK_TOTAL_S"
GRACE_S_DEFAULT = 2.0
CAP_S_DEFAULT = 12.0
#: PP0 intake -> verdict, below the #699 ADMISSION_WEDGE_SECONDS (20 s) with
#: room for the Admit's own lag down the pipe and the told=0 prefill start.
TOTAL_S_DEFAULT = 16.0

#: the ack stream's own point-to-point tag (1268 = idle vote, 580 = the
#: prefetch vote collective, 0 = the typed proxy/output channel).
#: 1538: shifted above ``FIXED_P2P_TAG_BASE`` (see ``pdflip_idle_vote``): the
#: group fence's ``monitored_barrier`` uses the collective counter as a raw p2p
#: slot on this same world cpu group, and at ``t1 == 1416`` PP0's standing ack
#: frames swallowed both followers' barrier messages (fence #337, boot
#: fs10052155, 27B with PF on).
PDFLIP_TOLD_ACK_TAG = FIXED_P2P_TAG_BASE + 1416
#: wire markers: INSTANCE attributes, present only on the armed path.
WIRE_ACK = "ack"
WIRE_FALLBACK = "fallback"
#: at most this many acks taken off one follower's stream per pass.
HARVEST_MAX_PER_SRC = 8

REASON_ACKS = "acks"
REASON_MISMATCH = "mismatch"
REASON_FRIST = "frist"
#: H64 GROUP-FLOOR: every follower acked the same depth below told
REASON_FLOOR = "floor"
WIRE_FLOOR = "floor"
ENV_MIN_PASSES = "FLLIPER_PDFLIP_TOLD_FALLBACK_MIN_PASSES"
ENV_ACK_DRAIN = "FLLIPER_PDFLIP_TOLD_ACK_DRAIN"
ENV_FLOOR = "FLLIPER_PDFLIP_TOLD_GROUP_FLOOR"
#: B5 ANCHOR-ACKS (NF h16p 226aad7105, 15.10. 12:04:24 / 12:16:46Z): a follower's ack carries the resumable anchor
#: depths below its reach; PP0 floors at the deepest depth EVERY stage (itself included) can resume from.
ENV_ANCHOR_ACKS = "FLLIPER_PDFLIP_TOLD_ANCHOR_ACKS"
#: at most this many anchor depths per ack entry (deepest first)
ANCHOR_ACKS_MAX = 16
#: T2 WAIT-PROGRESS (NF h16p 226aad7105, 15.10.: the 4 ``reason=frist`` cases, 365k tokens): a missing ack is LAG,
#: not absence. Every late ack the log shows (PP2 pdflip-33-81 own=121344 at +6 s, PP2 pdflip-4-25 own=43392 at +4 s) named
#: EXACTLY told -- PP0 had thrown the whole prefix away for an ack that was on its way. The Frist stays the bound of an
#: IDLE box (the #699 wedge clock runs while nothing serves); while the box is SERVING -- the detector that guards that
#: very clock (``invariant_checker.admission_wedge_verdict``, at the alarm threshold less ``ENV_WAIT_MARGIN_S``) would
#: not call it wedged -- PP0 keeps waiting: the chain lag that held the ack IS the serving work. ``ENV_WAIT_HARD_S``
#: from the read-ahead bounds a dead stream (frist_hard).
ENV_WAIT_PROGRESS = "FLLIPER_PDFLIP_TOLD_WAIT_PROGRESS"
ENV_WAIT_HARD_S = "FLLIPER_PDFLIP_TOLD_WAIT_HARD_S"
ENV_WAIT_MARGIN_S = "FLLIPER_PDFLIP_TOLD_WAIT_MARGIN_S"
WAIT_HARD_S_DEFAULT = 60.0
WAIT_MARGIN_S_DEFAULT = 5.0
REASON_FRIST_HARD = "frist_hard"
#: T2 ROOM-DEFER (same boots, pdflip-0-3 12:46:20 / pdflip-0-10 12:33:29): ``PF TOLD-ACK NO-ROOM loadback_rows=184768
#: room=162176`` acked 0 and PP0 re-prefilled 185k the store held. The room was short because a RUNNING prefill held
#: 100k rows ("full token usage: 0.38", one batch later 0.06): a shortage that lasts as long as that request. A follower
#: whose room would suffice once the rows other requests hold are released keeps its ack back (PP0 reads a missing ack
#: as lag and waits, WAIT-PROGRESS) and acks the real reach when the room is there. Not enough even then (rows >
#: room + locked): the ack says 0 as before (B5's anchors below still apply).
ENV_ROOM_DEFER = "FLLIPER_PDFLIP_TOLD_ROOM_DEFER"
#: a deferred ack re-reads the room at most this often (seconds)
ROOM_DEFER_RECHECK_S = 0.05
#: T2 DEFER-LIFESIGN (review A1, B1): the hard bound of a wait is a LIFESIGN, not 60 s of wall clock from the read-ahead.
#: A follower that defers its ack sends a defer entry in the ack message it already has (``(rid, 0, (), ("defer", rows,
#: room, locked, unpayable))`` -- B5 lengthened the same entry; no new wire message) at most every
#: ``ENV_DEFER_BEAT_S``. PP0 counts the hard bound (``ENV_WAIT_HARD_S``) from the last beat of every rank whose ack is
#: still missing; a rank that neither acked nor beats is counted from the read-ahead as before (dead = frist_hard).
#: ``ENV_WAIT_ABS_S`` is the absolute ceiling against a follower that beats forever (mutual waiting, a leak).
ENV_DEFER_LIFESIGN = "FLLIPER_PDFLIP_TOLD_DEFER_LIFESIGN"
ENV_DEFER_BEAT_S = "FLLIPER_PDFLIP_TOLD_DEFER_BEAT_S"
ENV_WAIT_ABS_S = "FLLIPER_PDFLIP_TOLD_WAIT_ABS_S"
DEFER_BEAT_S_DEFAULT = 1.0
#: a beat younger than this many beat periods is "live" (the follower defers NOW)
DEFER_LIVE_BEATS = 4
WAIT_ABS_S_DEFAULT = 300.0
REASON_FRIST_ABS = "frist_abs"
#: the marker of a defer entry (entry[3][0]); it is a lifesign, never an ack
DEFER_MARK = "defer"
#: T2 ROOM-DEFER-UNPAYABLE (review A2, B2): room + locked + unpayable == the whole pool, so on a pool that holds the
#: told at all the shortage is ALWAYS one of two transient things: rows other requests hold, or evictable rows no peel
#: can pay YET (the backups they need wait for arena slots -- the full-arena state). Both defer; the second is the
#: class the user's order names (a full L2 arena is no reason). It is bounded by PROGRESS (room grows), not by serving.
ENV_ROOM_DEFER_UNPAYABLE = "FLLIPER_PDFLIP_TOLD_ROOM_DEFER_UNPAYABLE"
#: hourly census line (K2): cause=arena_full reads and ARENA-CLAIM REFUSED against what the waits pin
ENV_HOURLY_S = "FLLIPER_PDFLIP_TOLD_CENSUS_S"
HOURLY_S_DEFAULT = 3600.0
_perf = time.perf_counter
_mono = time.monotonic
#: item 180: where a follower's re-ask of PP0 went (follower_reask)
ROUTE_PF = "pf-ack"
ROUTE_NONE = "none"

_LOG_FIRST = 8
_LOG_EVERY = 256


def env_on() -> bool:
    # RG 26.09.: unset = the registry row (pdflip/form.py PREFIX_SWITCHES; PF is
    # on on the nextflash row since 14.10., off on qwen27b).
    from flliper.srt.pdflip.form import prefix_switch_armed

    return prefix_switch_armed(ENV_FALLBACK)


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value >= 0 else default


def frist_s(window_s: float) -> float:
    """Seconds from the read-ahead's publication PP0 waits for every
    follower's ack before it sends told=0."""
    grace = _env_float(ENV_GRACE_S, GRACE_S_DEFAULT)
    cap = _env_float(ENV_CAP_S, CAP_S_DEFAULT)
    return min(cap, max(0.0, float(window_s)) + grace)


def _env_switch(name: str) -> bool:
    raw = (os.environ.get(name, "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def ack_drain_on() -> bool:
    """H64 ACK-DRAIN (default on): PP0 polls the ack stream on every pass."""
    return _env_switch(ENV_ACK_DRAIN)


def floor_on() -> bool:
    """H64 GROUP-FLOOR (default on): equal acks below told admit that depth."""
    return _env_switch(ENV_FLOOR)


def anchor_acks_on() -> bool:
    """B5 ANCHOR-ACKS (default on): acks carry the resumable anchors below the reach; the floor is the deepest
    depth common to every stage. ``0`` = H64's equal-acks floor alone."""
    return _env_switch(ENV_ANCHOR_ACKS)


def wait_progress_on() -> bool:
    """T2 WAIT-PROGRESS (default on): a late ack is waited for while the box serves (see ENV_WAIT_PROGRESS)."""
    return _env_switch(ENV_WAIT_PROGRESS)


def defer_lifesign_on() -> bool:
    """T2 DEFER-LIFESIGN (default on): see ENV_DEFER_LIFESIGN."""
    return _env_switch(ENV_DEFER_LIFESIGN)


def room_defer_unpayable_on() -> bool:
    """T2 ROOM-DEFER-UNPAYABLE (default on): see ENV_ROOM_DEFER_UNPAYABLE."""
    return _env_switch(ENV_ROOM_DEFER_UNPAYABLE)


def room_defer_on() -> bool:
    """T2 ROOM-DEFER (default on): a follower whose load-back room is short only while other requests hold rows
    defers its ack instead of acking 0 (see ENV_ROOM_DEFER)."""
    return _env_switch(ENV_ROOM_DEFER)


def min_passes(pp_size: int) -> int:
    """H64 FRIST-PASSES: PP0 passes after the read-ahead before the Frist may
    fire. Default pp_size + 1: pp_size - 1 chain hops to the last follower,
    then the two PP0 passes that take its ack (size frame, payload frame).
    0 = the wall Frist alone (pre-H64)."""
    default = max(0, int(pp_size) + 1)
    raw = (os.environ.get(ENV_MIN_PASSES, "") or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def deadline(published_at: float, own_read_s: float, window_s: float, room_wait_s: float = 0.0) -> float:
    """The Frist: ``min(published + frist_s(window), intake + TOTAL_S)``, never before publication.

    READ-ROOM (review A1 of arenaread, coupled to T2 DEFER-LIFESIGN): ``room_wait_s`` = the time PP0's own reads
    of this rid spent inside ARMED room waits (``read_room.lifesign_s``: a named, bounded wait on a full arena
    whose holders were asked to hand back L3-backed copies) -- a lifesign of the read, not idleness, exactly like
    a follower's defer beat. The ``intake + TOTAL_S`` clamp (the #699 wedge clock, which only logs) moves by it,
    capped by told2's absolute ceiling ``FLLIPER_PDFLIP_TOLD_WAIT_ABS_S``; without it 3 reads x 10 s of room wait put
    the Frist BEFORE the publication and the acks on their way met ``told=0 reason=frist`` (124245 pdflip-4-25:
    reads 5114/7748/10032 ms, ``PF TOLD-FALLBACK told=43392 -> 0 reason=frist after=0.16s``). 0 (no armed wait,
    READ-ROOM off) = the old Frist, byte for byte."""
    total = _env_float(ENV_TOTAL_S, TOTAL_S_DEFAULT)
    intake_at = float(published_at) - max(0.0, float(own_read_s))
    room = min(max(0.0, float(room_wait_s)), max(0.0, float(own_read_s)),
               _env_float(ENV_WAIT_ABS_S, WAIT_ABS_S_DEFAULT))
    return max(float(published_at), min(float(published_at) + frist_s(window_s), intake_at + total + room))


def _say(n: int) -> bool:
    return n <= _LOG_FIRST or n % _LOG_EVERY == 0


def _bump(obj, attr: str) -> int:
    n = int(getattr(obj, attr, 0) or 0) + 1
    setattr(obj, attr, n)
    return n


@dataclass
class PdFlipToldReadAck:
    """Follower -> PP0 on ``PDFLIP_TOLD_ACK_TAG``: the reads this rank has
    terminated since its last ack, as ``(rid, own_prefix)`` -- or, B5
    ANCHOR-ACKS, ``(rid, own_prefix, anchors)``: ``anchors`` = the depths below
    ``own_prefix`` this rank can ALSO resume from (state-bearing boundaries on
    its matched path, deepest first). The same message, a longer entry; PP0
    reads both forms."""

    rank: int
    seq: int
    reads: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class PdFlipStreamControl:
    """K20 (17.10.): follower -> PP0 on any GlooAckChannel stream, consumed INSIDE the channel (never handed to a
    caller). ``bye`` = this rank goes to sleep: it completes PP0's standing receive and PP0 does not re-arm it until
    ``wake()``. ``beat`` = idle heartbeat: completes the standing receive so gloo's receive clock (7200 s from the
    post) restarts long before it can fire."""

    rank: int
    kind: str  # "bye" | "beat"
    #: K20d: sleep epoch of a bye (= the sender's wake count + 1); PP0 takes a bye only of ITS current epoch. -1 = none.
    epoch: int = -1


#: K20: an idle follower completes PP0's standing receive at least this often (s); 0 = off. Vorgabe AN. Far below the
#: gloo timeout (7200 s) that killed PP0 at 00:36:54Z (boot 1016_172204: the last ack at 22:36:54Z re-armed the frame,
#: P slept 22:36:57Z, nothing was ever sent again).
ENV_HEARTBEAT_S = "FLLIPER_PDFLIP_PF_ACK_HEARTBEAT_S"
HEARTBEAT_S_DEFAULT = 600.0
#: K20: how long PP0 waits at sleep for the followers' byes (s). Far below the gloo timeout.
ENV_SLEEP_COLLECT_S = "FLLIPER_PDFLIP_PF_TOLD_ACK_TIMEOUT_S"
SLEEP_COLLECT_S_DEFAULT = 10.0
ENV_WAKE_DRAIN_S = "FLLIPER_PDFLIP_PF_WAKE_DRAIN_S"
WAKE_DRAIN_S_DEFAULT = 5.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


def _world_ranks(scheduler) -> List[int]:
    """Global rank of every PP stage, index = pp_rank (tp_size 1 form)."""
    ranks = getattr(getattr(scheduler, "pp_group", None), "ranks", None)
    if ranks:
        return [int(r) for r in ranks]
    ps = scheduler.ps
    tp = int(getattr(ps, "tp_size", 1) or 1)
    return [r * tp for r in range(int(ps.pp_size))]


class GlooAckChannel:
    """One process's end of the ack stream. Follower: ``send_nowait`` /
    ``pump``. PP0: ``harvest``. Nothing here ever blocks the caller."""

    def __init__(self, group: Any, pp_rank: int, world_ranks: List[int],
                 tag: int = PDFLIP_TOLD_ACK_TAG, label: str = "told-ack"):
        self.group = group
        self.pp_rank = int(pp_rank)
        self.world_ranks = list(world_ranks)
        #: PR (pdflip/pp_room_vote.py) runs a second stream of the same shape on
        #: its own tag; the told-ack defaults are unchanged.
        self.tag = int(tag)
        self.label = str(label)
        self._inflight: Optional[List[Tuple[Any, Any]]] = None  # (ParkedWait, tensor)
        self._frames: Dict[int, Any] = {}
        self.errors = 0
        #: a size header went out without its payload: the stream is
        #: misframed for good, so this follower sends nothing more (PP0's
        #: Frist answers told=0 for its reads from then on -- alive, slower).
        self.dead = False
        #: K20: PP0 side: ranks that said bye (their standing frame is not re-armed until wake()); follower side:
        #: monotonic time of the last post (heartbeat clock) and the bye sent flag.
        self._sleeping: set = set()
        self._last_post = time.monotonic()
        self.byes_taken = 0
        #: K20d: wakes seen by this rank (lockstep on every rank: one per resume leg); a bye is stamped wakes+1
        self.wakes = 0
        self.byes_stale = 0
        #: K20b: follower: after its own bye NO send leaves until wake (a send without a posted receive blocks 7200 s)
        self._gated = False
        #: K20c: the bye could not be posted inside the budget (an earlier send still hangs): pump() posts it
        self._bye_pending = False
        self.held = 0
        #: K20b: PP0: sleep_quiesce left byes missing -> wake() drains them before re-arming
        self._bye_missing = False
        self.beats_taken = 0

    @classmethod
    def for_scheduler(cls, scheduler) -> "GlooAckChannel":
        group = getattr(getattr(scheduler, "world_group", None), "cpu_group", None)
        return cls(group, int(scheduler.ps.pp_rank), _world_ranks(scheduler))

    # -- follower ------------------------------------------------------
    def pump(self, beat: bool = True) -> bool:
        """True when nothing is in flight any more. A completed send is
        joined (instantly: its flag is set) only to surface its error.
        ``beat``: an idle follower posts the K20 heartbeat here (once per HEARTBEAT_S)."""
        if self._inflight is None:
            if beat and self._bye_pending:
                self._bye_pending = False
                if not self.send_nowait(PdFlipStreamControl(rank=self.pp_rank, kind="bye", epoch=self.wakes + 1)):
                    self._bye_pending = True
            elif beat and self._beat_due():
                self.send_nowait(PdFlipStreamControl(rank=self.pp_rank, kind="beat"))
            return self._inflight is None
        if not all(pw.completed for pw, _t in self._inflight):
            return False
        done, self._inflight = self._inflight, None
        for pw, _t in done:
            try:
                pw.join(None)
            except Exception as exc:  # noqa: BLE001 - PP0's Frist covers a lost ack
                self._error("send", exc)
        return True

    def _error(self, what: str, exc: BaseException) -> None:
        self.errors += 1
        if _say(self.errors):
            logger.warning(
                "PF %s %s failed rank pp=%d (%d so far): %r -- PP0 sends "
                "told=0 at its Frist for every rid an ack could not carry",
                self.label, what, self.pp_rank, self.errors, exc,
            )

    def send_nowait(self, ack: PdFlipToldReadAck) -> bool:
        """Post ``ack`` unless the previous one is still in flight. True =
        the ack left the outbox (posted, or dropped on a dead stream)."""
        if self.dead:
            return True
        if self._gated and not (isinstance(ack, PdFlipStreamControl) and ack.kind == "bye"):
            self.held += 1
            return False
        if not self.pump(beat=False):
            return False
        self._last_post = time.monotonic()
        import numpy as np
        import torch
        import torch.distributed as dist

        from flliper.srt.mem_cache.hicache_collective import ParkedWait

        payload = pickle.dumps(ack)
        size_t = torch.tensor([len(payload)], dtype=torch.long)
        data_t = torch.ByteTensor(np.frombuffer(payload, dtype=np.uint8).copy())
        dst = self.world_ranks[0]
        works = []
        try:
            for label, t in (("size", size_t), ("payload", data_t)):
                w = dist.isend(t, dst, group=self.group, tag=self.tag)
                works.append((ParkedWait(w, f"pdflip/{self.label}/{label}"), t))
        except Exception as exc:  # noqa: BLE001 - never into the follower's pass
            self._error("send-post", exc)
            if works:
                # the size is on the wire without its payload: PP0 would read
                # the next size AS this payload. Never send on this stream again.
                self.dead = True
                logger.warning(
                    "PF TOLD-ACK stream DEAD rank pp=%d: half a frame posted; "
                    "no more acks from this rank, PP0's Frist decides told=0",
                    self.pp_rank,
                )
            self._inflight = works or None
            return True
        self._inflight = works
        return True

    # -- PP0 -----------------------------------------------------------
    def _frame(self, pp_rank: int):
        frame = self._frames.get(pp_rank)
        if frame is None:
            from flliper.srt.distributed.pp_object_recv import ObjectRecvFrame

            src = self.world_ranks[pp_rank]
            frame = self._frames[pp_rank] = ObjectRecvFrame(
                group=self.group,
                src_global=src,
                tag=self.tag,
                site=f"pdflip/{self.label}[pp{pp_rank}]",
                rank_desc="pp_rank=0",
            )
        return frame

    def _beat_due(self) -> bool:
        if self.dead or self.pp_rank == 0 or self._gated:
            return False
        hb = _env_float(ENV_HEARTBEAT_S, HEARTBEAT_S_DEFAULT)
        return hb > 0 and time.monotonic() - self._last_post >= hb

    def send_bye(self, budget_s: float) -> bool:
        """Follower, at sleep: complete PP0's standing receive, then GATE every send until wake(). Bounded; True = the
        bye has LEFT (PP0 received it). A bye still in flight at the budget stays in flight (PP0's frame is posted:
        its wake() drain takes it) but the gate is closed either way."""
        if self.dead or self.pp_rank == 0:
            return True
        end = time.monotonic() + budget_s
        msg = PdFlipStreamControl(rank=self.pp_rank, kind="bye", epoch=self.wakes + 1)
        while not self.send_nowait(msg):
            if time.monotonic() >= end:
                # an earlier send still hangs: the bye is NOT posted. Gate, but remember it; pump() posts it as soon
                # as that send completes (K20c) -- PP0's receive must not stay posted through the sleep.
                self._gated = True
                self._bye_pending = True
                return False
            time.sleep(0.01)
        self._gated = True
        while not self.pump(beat=False):
            if time.monotonic() >= end:
                return False
            time.sleep(0.01)
        return True

    def open_sends(self) -> int:
        """Sends of this process not completed (in flight, blocked on a missing receive) -- a gated, held fact is
        not a send but counts as held."""
        if self._bye_pending:
            return 1
        if self._inflight is None:
            return 0
        return 0 if all(pw.completed for pw, _t in self._inflight) else 1

    def open_receives(self) -> int:
        """PP0: standing receives that are posted right now (a frame past IDLE and not COMPLETE)."""
        n = 0
        for f in self._frames.values():
            if f.state in ("awaiting_size", "awaiting_data"):
                n += 1
        return n

    def wake(self) -> None:
        """Every rank: the group is awake again. Follower: the send gate opens. PP0: a bye that came after its
        sleep budget is taken FIRST (bounded, ENV_WAKE_DRAIN_S) so no follower is 'asleep' through the awake phase;
        then the frames are re-armed by the next harvest."""
        self._gated = False
        self._bye_pending = False
        if self.pp_rank == 0:
            if self._bye_missing:
                end = time.monotonic() + _env_float(ENV_WAKE_DRAIN_S, WAKE_DRAIN_S_DEFAULT)
                n = len(self.world_ranks) - 1
                while len(self._sleeping) < n and time.monotonic() < end:
                    self.harvest()
                    time.sleep(0.005)
                if len(self._sleeping) < n:
                    logger.error("PDFLIP PP-RECEIVE-CENSUS at=wake %s: %d bye(s) still missing", self.label,
                                 n - len(self._sleeping))
            self._bye_missing = False
            self._sleeping.clear()
        self.wakes += 1  # K20d: after the drain (a late bye of THIS sleep is still current while draining)

    def harvest(self) -> List[Any]:
        out: List[Any] = []
        for r in range(1, len(self.world_ranks)):
            if r in self._sleeping:
                continue
            frame = self._frame(r)
            for _ in range(HARVEST_MAX_PER_SRC):
                try:
                    if not frame.poll():
                        break
                    item = frame.take()
                    if isinstance(item, PdFlipStreamControl):
                        if item.kind == "bye" and item.epoch not in (-1, self.wakes + 1):
                            self.byes_stale += 1
                            logger.warning(
                                "PDFLIP PP-RECEIVE-CENSUS %s: STALE bye from pp%d epoch=%d (PP0 expects %d) discarded; "
                                "the follower is awake", self.label, r, item.epoch, self.wakes + 1)
                            continue
                        if item.kind == "bye":
                            self._sleeping.add(r)
                            self.byes_taken += 1
                            break
                        self.beats_taken += 1
                        continue
                    out.append(item)
                except Exception as exc:  # noqa: BLE001 - the Frist covers it
                    self._error(f"receive[pp{r}]", exc)
                    break
        return out


def _channel(scheduler):
    ch = getattr(scheduler, "_pdflip_fb_channel", None)
    if ch is None:
        ch = scheduler._pdflip_fb_channel = GlooAckChannel.for_scheduler(scheduler)
    return ch


# ---------------------------------------------------------------------------
# PP0
# ---------------------------------------------------------------------------


@dataclass
class _Beat:
    """T2 DEFER-LIFESIGN: a follower's last defer entry, on PP0's clock. ``progress_at`` = the last beat that showed
    progress (the room grew, or the rows others hold / the unpayable evictable shrank; the first beat counts)."""

    at: float
    progress_at: float
    rows: int
    room: int
    locked: int
    unpayable: int
    n: int = 1
    #: the shortage is unpayable evictable (room + locked < rows): bounded by progress even while the box serves
    unpayable_class: bool = False


@dataclass
class _Open:
    told: int
    deadline: float
    acks: Dict[int, int] = field(default_factory=dict)
    #: H64 FRIST-PASSES: PP0's pass number at the read-ahead (-1 = unknown:
    #: the wall Frist alone, as a caller without a pass count had it)
    published_pass: int = -1
    #: H64: the Frist was held by the pass floor at least once (named once)
    frist_held: bool = False
    #: B5 ANCHOR-ACKS: per follower rank, the anchor depths below its ack
    anchors: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    #: T2 WAIT-PROGRESS: PP0's clock at the read-ahead, the moment the wait began (None = never waited), its passes
    published_at: float = 0.0
    waited_since: Optional[float] = None
    wait_n: int = 0
    #: T2 DEFER-LIFESIGN: per follower rank whose ack is still missing, its last defer beat
    beats: Dict[int, "_Beat"] = field(default_factory=dict)


def _pp0_open_map(scheduler) -> Dict[str, _Open]:
    d = getattr(scheduler, "_pdflip_fb_open", None)
    if d is None:
        d = scheduler._pdflip_fb_open = {}
    return d


def pp0_open(scheduler, rid: str, told: int, published_at: float, own_read_s: float, window_s: float,
             published_pass: Optional[int] = None, room_wait_s: float = 0.0) -> float:
    """PP0 put a paced read-ahead with ``ack`` on the wire for ``rid``. ``room_wait_s``: see :func:`deadline`."""
    dl = deadline(published_at, own_read_s, window_s, room_wait_s)
    _pp0_open_map(scheduler)[str(rid)] = _Open(
        told=int(told), deadline=dl,
        published_pass=-1 if published_pass is None else int(published_pass),
        published_at=float(published_at),
    )
    return dl - float(published_at)


def pp0_has_stream(scheduler) -> bool:
    """H64 ACK-DRAIN: the ack stream exists on PP0 (its frames were posted)."""
    return getattr(scheduler, "_pdflip_fb_channel", None) is not None and ack_drain_on()


def pp0_forget(scheduler, rid: str) -> None:
    _pp0_open_map(scheduler).pop(str(rid), None)


#: item 220: ``rid -> told`` PP0 admitted (told > 0) and has not seated yet
RETRACT_ADMITTED_MAX = 1024


def _admitted_map(scheduler) -> Dict[str, int]:
    d = getattr(scheduler, "_pdflip_fb_admitted", None)
    if d is None:
        d = scheduler._pdflip_fb_admitted = {}
    return d


def _late_map(scheduler) -> Dict[str, int]:
    d = getattr(scheduler, "_pdflip_fb_late", None)
    if d is None:
        d = scheduler._pdflip_fb_late = {}
    return d


def pp0_watching(scheduler) -> bool:
    """True while PP0 holds an admitted-at-told rid a late ack could still
    retract: the publish pass keeps harvesting the ack stream for it."""
    return bool(getattr(scheduler, "_pdflip_fb_admitted", None))


def pp0_retract_due(scheduler, queued, parked) -> List[Tuple[str, int, int]]:
    """W27-UNIFORM (item 220, residual of item 180): the answer path for a
    short reach a follower finds AFTER the Admit. Returns ``(rid, told, own)``
    for every rid PP0 retracts to told=0 in THIS pass (the caller puts one
    ``Admit(0, fallback)`` on the wire and applies it to PP0 itself, exactly
    like a Frist fallback).

    Rank-uniform by construction: only PP0 decides, and only from PP0-local
    facts -- a follower's late ack differing from the told PP0 admitted, and
    the rid still in PP0's own waiting queue (PP0 has not seated it, so no
    rank has: every rank seats at PP0's pass plus its plan lag, and the wire
    keeps its order). A rid PP0 seated already is TOO LATE: nothing is put on
    the wire (a retract after PP0's seat would be the rank-local split), the
    follower's hold stays and the #1233 W27 guard names it. No reserve, and
    PP0 never waits for a follower, so the follower's hold cannot deadlock
    against it. Admitted rids that left the queue are forgotten here."""
    admitted = _admitted_map(scheduler)
    late = _late_map(scheduler)
    out: List[Tuple[str, int, int]] = []
    for rid in list(late):
        own = late[rid]
        told = admitted.get(rid)
        if told is None:
            late.pop(rid, None)
            continue
        if rid in queued:
            late.pop(rid, None)
            admitted.pop(rid, None)
            out.append((rid, int(told), int(own)))
        elif rid not in parked:
            late.pop(rid, None)
            admitted.pop(rid, None)
            n = _bump(scheduler, "_pf_retract_late_n")
            if _say(n):
                logger.error(
                    "PF TOLD-RETRACT TOO LATE rid=%s told=%d follower_reach=%d (n=%d): PP0 "
                    "seated this rid already -- no rank-uniform answer is possible any more, "
                    "nothing is put on the wire; the follower's hold stays and the #1233 W27 "
                    "guard names the split", rid, told, own, n,
                )
    for rid in list(admitted):
        if rid not in queued and rid not in parked:
            admitted.pop(rid, None)  # seated or gone: nothing left to retract
    while len(admitted) > RETRACT_ADMITTED_MAX:
        admitted.pop(next(iter(admitted)), None)
    return out


def pp0_retract_applied(scheduler, rid: str, told: int, own: int) -> None:
    """PP0 applies its own retract (same as the Frist fallback: its read is
    released, its admission compares 0 with 0) and names it."""
    rid = str(rid)
    release_own_read(scheduler, rid)
    try:
        from flliper.srt.managers import pdflip_store_told as _st

        _st.forget_unreached(scheduler, rid)
    except Exception:  # noqa: BLE001 - bookkeeping
        pass
    (getattr(scheduler, "_pdflip_told_kept", None) or {}).pop(rid, None)
    (getattr(scheduler, "_pdflip_store_told_satisfied", None) or {}).pop(rid, None)
    n = _bump(scheduler, "_pf_retract_n")
    if n <= 32 or n % _LOG_EVERY == 0:
        logger.warning(
            "PF TOLD-RETRACT rid=%s told=%d -> 0 follower_reach=%d (n=%d): a follower's tree "
            "stayed short of the admitted told after the Admit and PP0 had not seated the "
            "rid yet -- PP0 switches it to told=0 for EVERY rank on the Admit channel "
            "(fallback marker); P recomputes the prefix", rid, told, own, n,
        )


def _census_now(scheduler) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        rd = sys.modules.get("flliper.srt.pdflip.read_depth")
        out["arena_full_reads"] = int((rd.census() if rd is not None else {}).get("arena_full", 0))
    except Exception:  # noqa: BLE001
        out["arena_full_reads"] = -1
    try:
        ap = sys.modules.get("flliper.srt.mem_cache.pool_host.arena_pool")
        out["arena_claim_refused"] = int(getattr(getattr(ap, "ArenaMHAHostPool", None), "_1427_full_n", 0) or 0)
    except Exception:  # noqa: BLE001
        out["arena_claim_refused"] = -1
    for key, attr in (
        ("waited", "_pf_waited_n"), ("wait_started", "_pf_wait_n"), ("room_defer", "_pf_ack_room_defer_n"),
        ("no_room", "_pf_ack_no_room_n"), ("fallback", "_pf_fallback_n"),
    ):
        out[key] = int(getattr(scheduler, attr, 0) or 0)
    return out


def census_tick(scheduler) -> None:
    """Review A3 (K2): once per ``ENV_HOURLY_S`` (default an hour) one line with the DELTA since the last line of
    the reads cut for ``cause=arena_full``, the ``ARENA-CLAIM REFUSED`` count and the waits / defers / fallbacks of
    this stage -- the same line on h16p (without the T2 terms) is the comparison: do the longer L2 pins of a wait push
    other reads into arena_full? Never raises."""
    try:
        now = _mono()
        st = getattr(scheduler, "_pf_census", None)
        cur = _census_now(scheduler)
        if st is None:
            scheduler._pf_census = (now, cur)
            return
        t0, prev = st
        if now - t0 < _env_float(ENV_HOURLY_S, HOURLY_S_DEFAULT):
            return
        scheduler._pf_census = (now, cur)
        logger.info(
            "PF TOLD-CENSUS rank pp=%s window=%.0fs %s totals=%s: reads cut for cause=arena_full and "
            "ARENA-CLAIM REFUSED per window -- compare with h16p (T2 waits pin the span's L2 slots longer)",
            getattr(getattr(scheduler, "ps", None), "pp_rank", "?"), now - t0,
            {k: cur[k] - prev.get(k, 0) for k in cur}, cur,
        )
    except Exception:  # noqa: BLE001 - an instrument never breaks a pass
        return


def _note_beat(o: Optional[_Open], rank: int, body: Any, now: float) -> None:
    if o is None or rank in o.acks:
        return
    try:
        rows, room, locked, unpayable = (int(x) for x in tuple(body)[1:5])
    except Exception:  # noqa: BLE001 - a malformed beat is no lifesign
        return
    prev = o.beats.get(rank)
    if prev is None:
        o.beats[rank] = _Beat(
            at=now, progress_at=now, rows=rows, room=room, locked=locked, unpayable=unpayable,
            unpayable_class=room + locked < rows,
        )
        return
    better = room > prev.room or locked < prev.locked or unpayable < prev.unpayable
    o.beats[rank] = _Beat(
        at=now, progress_at=now if better else prev.progress_at, rows=rows,
        # the best of each term seen, so a wobble does not read as progress next time
        room=max(room, prev.room), locked=min(locked, prev.locked), unpayable=min(unpayable, prev.unpayable),
        n=prev.n + 1, unpayable_class=room + locked < rows,
    )


def pp0_harvest(scheduler, now: Optional[float] = None) -> int:
    """Take every ack that has landed off the standing receives (no wait). ``now`` = PP0's monotonic clock of the
    pass (the clock ``pp0_decide`` reads its ``published_at`` on); a defer beat is stamped with it."""
    open_map = _pp0_open_map(scheduler)
    now = _mono() if now is None else float(now)
    census_tick(scheduler)
    n = 0
    for ack in _channel(scheduler).harvest():
        if not isinstance(ack, PdFlipToldReadAck):
            logger.warning("PF TOLD-ACK object %s on the ack stream dropped", type(ack).__name__)
            continue
        for entry in ack.reads:
            rid, own = entry[0], entry[1]
            if len(entry) > 3 and entry[3] and entry[3][0] == DEFER_MARK:
                # T2 DEFER-LIFESIGN: not an ack -- the follower lives and defers; PP0 only notes it. A beat for a
                # rid PP0 decided (or never opened) changes nothing and is not a late ack.
                _note_beat(open_map.get(str(rid)), int(ack.rank), entry[3], now)
                continue
            anchors = tuple(int(a) for a in entry[2]) if len(entry) > 2 and entry[2] else ()
            o = open_map.get(str(rid))
            if o is None:
                # item 220: a follower's re-ask (item 180) of a rid PP0
                # admitted at told that has NOT been seated yet is a retract
                # candidate; PP0 decides it at its next publish
                # (pp0_retract_due), never here
                adm = _admitted_map(scheduler).get(str(rid))
                if adm is not None and int(own) != int(adm):
                    _late_map(scheduler)[str(rid)] = int(own)
                    continue
                # decided (or dropped) already: late ack (item 180: also a
                # follower's re-ask of an already admitted rid -- named)
                k = _bump(scheduler, "_pf_ack_late_n")
                if _say(k):
                    logger.warning(
                        "PF TOLD-ACK LATE rank=%s rid=%s own=%s (n=%d): PP0 decided this "
                        "rid already -- the ack changes nothing", ack.rank, str(rid), own, k,
                    )
                continue
            o.acks[int(ack.rank)] = int(own)
            o.anchors[int(ack.rank)] = anchors
            o.beats.pop(int(ack.rank), None)  # the ack ends the deferral of that rank
            n += 1
    if n:
        k = _bump(scheduler, "_pf_ack_harvest_n")
        if _say(k):
            logger.info("PF TOLD-ACK HARVEST acks=%d open=%d (n=%d)", n, len(open_map), k)
    return n


def common_depths(o: _Open) -> List[int]:
    """B5: the depths in (0, told) EVERY acked follower can resume from -- its ack plus the anchors it sent below
    it -- deepest first. Without anchors (switch off, an old entry) each follower contributes its ack alone, i.e.
    H64's equal-acks floor."""
    told = int(o.told)
    common: Optional[set] = None
    for rank, own in o.acks.items():
        mine = {int(own)}
        if anchor_acks_on():
            mine |= {int(a) for a in o.anchors.get(rank, ()) if 0 < int(a) and (int(own) <= 0 or int(a) <= int(own))}
        common = mine if common is None else (common & mine)
    if not common:
        return []
    return sorted((d for d in common if 0 < d < told), reverse=True)


def _floor_candidate(o: _Open) -> Optional[int]:
    """H64 GROUP-FLOOR: the deepest depth every ack names (B5: every ack or one of its anchors), when it lies in
    (0, told); None = no floor is possible (no common depth, 0, over told)."""
    depths = common_depths(o)
    return depths[0] if depths else None


def pp0_floor_below(scheduler, rid: str, req, below: int, absolute: bool) -> int:
    """B5: PP0's own tree refused the floor ``below`` (TF). The next deeper common depth PP0 ITSELF can resume from
    (``pdflip_told_fidelity.pp0_admissible`` at that depth, no side effect), else 0. PP0 alone decides; the followers
    proved every candidate by acking it. Read before ``pp0_note_verdict`` pops the open entry."""
    if not anchor_acks_on():
        return 0
    o = _pp0_open_map(scheduler).get(str(rid))
    if o is None:
        return 0
    try:
        from flliper.srt.managers import pdflip_told_fidelity as _tf
    except Exception:  # noqa: BLE001
        return 0
    reads_left = _tf_anchor_read_budget()
    for d in common_depths(o):
        if d >= int(below):
            continue
        depth = _tf.absolute_depth(req, d, absolute)
        own = _tf.pp0_admissible(scheduler, req, depth)
        if own is not None and int(own) >= depth:
            return int(d)
        if own is not None and reads_left > 0:
            # S43 TF-ANCHOR-READ: every follower proved ``d``; PP0's tree holds
            # the KV there but its Mamba anchor only in the store -> read just
            # the blob (e1 at this depth) and ask PP0's tree again. PP0 alone
            # decides, the followers move nothing (SOLL R11).
            reads_left -= 1
            got = _tf_anchor_read(scheduler, req, depth)
            if got > 0:
                own2 = _tf.pp0_admissible(scheduler, req, depth)
                k = _bump(scheduler, "_pf_tf_anchor_read_n")
                if _say(k) or (own2 is not None and int(own2) >= depth):
                    logger.warning(
                        "S43 TF-ANCHOR-READ rid=%s depth=%d below=%d own_before=%d own_after=%s (n=%d): PP0's tree "
                        "held the KV at a depth every follower proved, its Mamba anchor only in the store -- the "
                        "blob was read so PP0 resumes there", str(rid), int(depth), int(below), int(own),
                        own2, k)
                if own2 is not None and int(own2) >= depth:
                    return int(d)
    return 0


def _tf_anchor_read_budget() -> int:
    """S43: anchor reads one floor search may spend (0 = the switch is off)."""
    try:
        from flliper.srt.environ import envs as _envs

        if not _envs.FLLIPER_PDFLIP_TF_ANCHOR_READ.get():
            return 0
        return max(0, int(_envs.FLLIPER_PDFLIP_TF_ANCHOR_READ_MAX.get()))
    except Exception:  # noqa: BLE001 - an unreadable switch reads nothing
        return 0


def _tf_anchor_read(scheduler, req, depth: int) -> int:
    """S43: PP0's tree reads the Mamba anchor blob of the node at ``depth``
    (``UnifiedRadixCache.pdflip_anchor_only_read_at``); 0 when the tree has no
    such reader or nothing was attached. Never raises."""
    fn = getattr(getattr(scheduler, "tree_cache", None), "pdflip_anchor_only_read_at", None)
    if not callable(fn):
        return 0
    try:
        return int(fn(req, int(depth), site="tf_floor") or 0)
    except Exception:  # noqa: BLE001 - a failed read leaves the floor where it was
        logger.warning("S43 TF-ANCHOR-READ failed rid=%s depth=%d", getattr(req, "rid", "?"), int(depth),
                       exc_info=True)
        return 0


def _frist_due(scheduler, rid: str, o: _Open, now: float, pass_n: Optional[int]) -> bool:
    """The wall Frist AND (H64) the PP0 pass floor since the read-ahead."""
    if now < o.deadline:
        return False
    if pass_n is None or o.published_pass < 0:
        return True
    need = min_passes(int(scheduler.ps.pp_size))
    done = int(pass_n) - int(o.published_pass)
    if done >= need:
        return True
    if not o.frist_held:
        o.frist_held = True
        n = _bump(scheduler, "_pf_frist_held_n")
        if _say(n):
            logger.info(
                "PF TOLD-FRIST HELD rid=%s told=%d passes=%d/%d acks=%s (n=%d): the wall Frist is "
                "up but PP0 has not run the passes an ack needs to reach it (chain hops + size "
                "+ payload frame) -- the verdict waits for the passes, not for the clock",
                rid, o.told, done, need, dict(o.acks), n,
            )
    return False


def _wait_hard_s() -> float:
    return _env_float(ENV_WAIT_HARD_S, WAIT_HARD_S_DEFAULT)


def box_serving(scheduler) -> Tuple[bool, str]:
    """T2 WAIT-PROGRESS: True = the #699 admission-wedge detector, asked at its alarm threshold less
    ``ENV_WAIT_MARGIN_S``, would NOT call this box wedged -- something serves (a decode round runs) or progressed
    lately (a first token, a retired prefill chunk). It is the verdict the wedge guard itself uses, on the clocks it
    reads (``last_first_token_progress_time`` / ``last_prefill_progress_time``), not a second rule. PP0-local; no clock
    or no queue to read = (False, ...): the Frist stands, as before."""
    try:
        # the scheduler imports the module at its top (scheduler.py); a process that did not has no wedge guard to
        # ask -- and must not pay the first import (torch & co.) inside PP0's pass
        _ic = sys.modules.get("flliper.srt.managers.scheduler_components.invariant_checker")
        if _ic is None:
            return False, "invariant_checker not loaded"
        queued = len(scheduler.waiting_queue)
        running = len(scheduler.running_batch.reqs)
        first = float(scheduler.last_first_token_progress_time)
        threshold = max(1.0, float(_ic._wedge_alarm_threshold()) - _env_float(ENV_WAIT_MARGIN_S, WAIT_MARGIN_S_DEFAULT))
        now = _perf()
        prefill_stamp = getattr(scheduler, "last_prefill_progress_time", None)
        since_prefill = None if prefill_stamp is None else now - float(prefill_stamp)
        alarm, detail = _ic.admission_wedge_verdict(
            queued, running, now - first, threshold=threshold, seconds_since_prefill_progress=since_prefill,
        )
    except Exception as exc:  # noqa: BLE001 - no clock to read: no proof of progress
        return False, f"no progress clock: {exc!r}"
    return (not alarm), detail


def _wake_floor(scheduler, now: float) -> float:
    """The moment of the last sleep wake, on the clock of ``now`` (-inf = none): a wait that spans a sleep is not
    aged by the sleep (review point 4)."""
    wake = getattr(scheduler, "_pdflip_last_wake_t", None)
    if wake is None:
        return float("-inf")
    try:
        age = _perf() - float(wake)
    except Exception:  # noqa: BLE001
        return float("-inf")
    return float(now) - age if age >= 0 else float("-inf")


def _lifesign(scheduler, o: _Open, now: float, serving: bool) -> Tuple[float, bool, str]:
    """T2 DEFER-LIFESIGN: ``(oldest lifesign over the ranks whose ack is missing, any of them defers NOW, text)``.
    A rank that beats counts from its last beat -- while the box serves (the chain moves, the rows come back); from
    its last PROGRESS beat when the box idles or when its shortage is unpayable evictable (nothing but a growing
    room helps). A rank without a beat (an ack on its way, or dead) counts from the read-ahead, as before. The wake
    of a sleep is a floor for all."""
    floor = max(float(o.published_at), _wake_floor(scheduler, now))
    live = False
    oldest: Optional[float] = None
    parts: List[str] = []
    stale = DEFER_LIVE_BEATS * _env_float(ENV_DEFER_BEAT_S, DEFER_BEAT_S_DEFAULT)
    try:
        pp_size = int(scheduler.ps.pp_size)
    except Exception:  # noqa: BLE001
        pp_size = 1 + len(o.acks)
    for r in range(1, pp_size):
        if r in o.acks:
            continue
        b = o.beats.get(r) if defer_lifesign_on() else None
        if b is None:
            sign = floor
        else:
            by_progress = (not serving) or b.unpayable_class
            sign = max(floor, b.progress_at if by_progress else b.at)
            if now - b.at <= stale:
                live = True
            parts.append(
                f"pp{r}:beats={b.n} rows={b.rows} room={b.room} locked={b.locked} unpayable={b.unpayable} "
                f"beat_age={now - b.at:.1f}s progress_age={now - b.progress_at:.1f}s"
            )
        oldest = sign if oldest is None else min(oldest, sign)
    return (floor if oldest is None else oldest), live, "; ".join(parts)


def _wait_context(scheduler, rid: str) -> str:
    """Review A3: what a wait costs at the metal -- the L2 slots the rid's pinned span holds on this stage, the queue
    and the named idle blockers (a flip / sleep / seat gate waits for them, K10)."""
    slots = _pinned_slots(scheduler, rid)
    try:
        queued = len(scheduler.waiting_queue)
    except Exception:  # noqa: BLE001
        queued = -1
    blockers = "n/a"
    fn = getattr(scheduler, "idle_blockers", None)
    if callable(fn):
        try:
            blockers = ",".join(str(b) for b in fn())[:160] or "none"
        except Exception:  # noqa: BLE001
            blockers = "n/a"
    return f"pinned_l2_slots={slots} queued={queued} idle_blockers=[{blockers}]"


def _pinned_slots(scheduler, rid: str) -> int:
    """L2 arena slots held by the pinned span of ``rid`` on this stage (-1 = not readable). An instrument."""
    try:
        from flliper.srt.pdflip import park_hold_yield as _phy

        tree = scheduler.tree_cache
        pool = _phy._kv_pool(tree)
        if pool is None:
            return -1
        return sum(len(_phy._node_slots(pool, n)) for n in _phy._span_nodes(tree, rid))
    except Exception:  # noqa: BLE001 - an instrument never breaks a verdict
        return -1


def _frist_verdict(scheduler, rid: str, o: _Open, now: float, pass_n: Optional[int]) -> Optional[Tuple[int, str]]:
    """The acks are not all in. ``None`` = keep waiting (the Frist is not due, or T2 WAIT-PROGRESS: it is due but the
    box is serving -- the late ack is chain lag, and the lag IS the serving work -- or a follower defers); else
    ``(0, reason)``. The hard bound hangs on lifesigns (DEFER-LIFESIGN), the absolute ceiling on the read-ahead."""
    if not _frist_due(scheduler, rid, o, now, pass_n):
        return None
    if not wait_progress_on():
        return 0, REASON_FRIST
    serving, detail = box_serving(scheduler)
    oldest, live, beats = _lifesign(scheduler, o, now, serving)
    if not serving and not live:
        return 0, REASON_FRIST
    hard = _wait_hard_s()
    floor = max(float(o.published_at), _wake_floor(scheduler, now))
    if now - floor >= max(hard, _env_float(ENV_WAIT_ABS_S, WAIT_ABS_S_DEFAULT)):
        return 0, REASON_FRIST_ABS
    if now - oldest >= hard:
        return 0, REASON_FRIST_HARD
    if o.waited_since is None:
        o.waited_since = float(now)
    o.wait_n += 1
    if o.wait_n == 1:
        n = _bump(scheduler, "_pf_wait_n")
        if _say(n):
            logger.info(
                "PF TOLD-WAIT rid=%s told=%d after=%.2fs acks=%s (n=%d): the Frist is up but the box is serving "
                "(%s)%s -- a missing ack is chain lag or a deferring follower, PP0 keeps the rid and waits for it "
                "instead of told=0; %s",
                rid, o.told, now - o.published_at, dict(o.acks), n, detail[:160],
                f" defer[{beats}]" if beats else "", _wait_context(scheduler, rid),
            )
    return None


def pp0_decide(scheduler, rid: str, now: float, pass_n: Optional[int] = None) -> Optional[Tuple[int, str]]:
    """``None`` = keep waiting; else ``(told to admit, reason)``. ``pass_n`` =
    PP0's current pass number (H64 FRIST-PASSES; None = the wall Frist alone).
    PP0-local facts only: the acks it holds, its clock, its pass count."""
    o = _pp0_open_map(scheduler).get(str(rid))
    if o is None:
        return 0, REASON_FRIST  # untracked paced rid: the always-uniform answer
    followers = range(1, int(scheduler.ps.pp_size))
    complete = all(r in o.acks for r in followers)
    if any(own != o.told for own in o.acks.values()):
        if not floor_on():
            return 0, REASON_MISMATCH
        m = _floor_candidate(o)
        if m is None:
            return 0, REASON_MISMATCH
        if complete:
            return m, REASON_FLOOR
        # a floor is still possible: wait for the missing acks (bounded by the Frist, T2: by progress)
        return _frist_verdict(scheduler, rid, o, now, pass_n)
    if complete:
        return o.told, REASON_ACKS
    return _frist_verdict(scheduler, rid, o, now, pass_n)


def pp0_note_verdict(scheduler, rid: str, told: int, told_final: int, reason: str, now: float, published_at: float) -> None:
    o = _pp0_open_map(scheduler).pop(str(rid), None)
    acks = dict(o.acks) if o is not None else {}
    if int(told_final) > 0:
        # item 220: PP0 admitted at told; until it seats the rid, a follower's
        # late re-ask can still switch it to told=0 (pp0_retract_due)
        _admitted_map(scheduler)[str(rid)] = int(told_final)
    if o is not None and o.waited_since is not None:
        # T2 WAIT-PROGRESS: the proof marker -- what used to be told=0 reason=frist ended with a verdict that keeps the
        # prefix (``kept`` > 0) or, named, still fell back
        n_w = _bump(scheduler, "_pf_waited_n")
        if n_w <= 32 or n_w % _LOG_EVERY == 0:
            logger.info(
                "PF TOLD-WAITED rid=%s told=%d -> %d reason=%s waited=%.2fs polls=%d acks=%s beats=%s (n=%d): the "
                "verdict came after the Frist while the box served; %s", rid, told, int(told_final), reason,
                now - o.waited_since, o.wait_n, acks, {r: b.n for r, b in o.beats.items()}, n_w,
                _wait_context(scheduler, rid),
            )
    if reason == REASON_ACKS:
        n = _bump(scheduler, "_pf_admit_acks_n")
        if _say(n):
            logger.info(
                "PF TOLD-ACKED rid=%s told=%d after=%.2fs acks=%s (n=%d): every "
                "follower's read reproduced told -- admitted without the window",
                rid, told, now - published_at, acks, n,
            )
        return
    if reason == "raise" and int(told_final) > int(told):
        # S64 S-1400 (b): pdflip_told_raise names the admit itself (S64 TOLD-RAISE ADMIT)
        return
    if reason == REASON_FLOOR and int(told_final) > 0:
        n = _bump(scheduler, "_pf_floor_n")
        if n <= 32 or n % _LOG_EVERY == 0:
            logger.warning(
                "PF TOLD-FLOOR rid=%s told=%d -> %d after=%.2fs acks=%s (n=%d): every follower "
                "proved the same depth below told and PP0's own tree admits it -- PP0 admits "
                "that depth for EVERY rank (floor marker) instead of told=0",
                rid, told, int(told_final), now - published_at, acks, n,
            )
        return
    n = _bump(scheduler, "_pf_fallback_n")
    if n <= 32 or n % _LOG_EVERY == 0:
        logger.warning(
            "PF TOLD-FALLBACK rid=%s told=%d -> 0 reason=%s after=%.2fs acks=%s "
            "followers=%d (n=%d): PP0 switches the request to told=0 for EVERY "
            "rank; P recomputes the prefix instead of the group dying in "
            "STORE-TOLD WAIT EXCEEDED / MISMATCH",
            rid, told, reason, now - published_at, acks,
            int(scheduler.ps.pp_size) - 1, n,
        )


def apply_floor(scheduler, rid: str, floor: int) -> None:
    """H64 GROUP-FLOOR on THIS rank -- PP0 at its verdict, a follower at the
    ``Admit(floor, floor=1)`` it absorbs; the same steps on every rank. The
    rid is SATISFIED at ``floor``: nothing is released (the rows ARE the held
    prefix), this rank's loaded credit is popped with the span pin kept (the
    #1417 pin stays until the rid leaves the queue, as for any told), no read
    is awaited, and the admission caps its match at ``floor`` (its told). A
    follower twin keeps its mark, so its admission still checks its live tree
    at ``floor`` (W27-UNIFORM) exactly as at any absolute told."""
    from flliper.srt.managers import pdflip_store_told as _st

    rid = str(rid)
    follower_forget(scheduler, rid)
    try:
        _st._pop_credit_keep_pin(scheduler.tree_cache, rid)
    except Exception:  # noqa: BLE001 - bookkeeping; the satisfied mark decides
        pass
    sat = getattr(scheduler, "_pdflip_store_told_satisfied", None)
    if sat is None:
        sat = scheduler._pdflip_store_told_satisfied = {}
    sat[rid] = int(floor)
    note_floor_applied(scheduler, rid, int(floor))
    n = _bump(scheduler, "_pf_floor_applied_n")
    if n <= 32 or n % _LOG_EVERY == 0:
        logger.info(
            "PF TOLD-FLOOR APPLIED rank pp=%s rid=%s floor=%d (n=%d): satisfied at PP0's floor, "
            "nothing released, admits at that depth like every rank",
            getattr(scheduler.ps, "pp_rank", "?"), rid, int(floor), n,
        )


#: H16Q review A4 (metal tripwire, B5 M1): a B5 anchor shallower than the #1417 span pin is not pinned -- evicted
#: between the ack and the admission, the follower's tree falls short of the floor (W27-UNIFORM SHORT, then HELD).
#: Not a silent fault (W27 names it, its guard stops the group), but a new way into that class: a W27 SHORT/HELD
#: for a rid within this many seconds after its TOLD-FLOOR is named once per (rid, kind). No stop, no decision.
FLOOR_W27_WINDOW_S = 30.0


def note_floor_applied(scheduler, rid: str, floor: int, now: Optional[float] = None) -> None:
    """A4: remember the floor this rank applied for ``rid`` (bookkeeping only; never raises)."""
    try:
        m = getattr(scheduler, "_pdflip_floor_w27_watch", None)
        if not isinstance(m, dict):
            m = scheduler._pdflip_floor_w27_watch = {}
        t = time.time() if now is None else float(now)
        if len(m) > 256:
            for k in [k for k, v in m.items() if t - v[1] > FLOOR_W27_WINDOW_S]:
                m.pop(k, None)
        m[str(rid)] = (int(floor), t, set())
    except Exception:  # noqa: BLE001 - an instrument
        pass


def floor_then_w27(scheduler, rid: str, kind: str, told: int, live: int, now: Optional[float] = None) -> bool:
    """A4: called at the W27-UNIFORM SHORT / HELD lines; names a W27 that follows a TOLD-FLOOR of the same rid
    within ``FLOOR_W27_WINDOW_S``. True = named. Never raises, never decides."""
    try:
        m = getattr(scheduler, "_pdflip_floor_w27_watch", None)
        ent = m.get(str(rid)) if isinstance(m, dict) else None
        if ent is None:
            return False
        floor, t_floor, named = ent
        t = time.time() if now is None else float(now)
        if t - t_floor > FLOOR_W27_WINDOW_S or kind in named:
            return False
        named.add(kind)
        n = _bump(scheduler, "_pf_floor_w27_n")
        logger.warning(
            "PF TOLD-FLOOR-THEN-W27 rank pp=%s rid=%s floor=%d w27=%s told=%d live=%d after_s=%.1f (n=%d): this "
            "rank's tree fell short of the floor PP0 set from the acks -- a B5 anchor above the span pin left the "
            "tree between ack and admit (review M1); named, no stop -- metal count must be 0",
            getattr(getattr(scheduler, "ps", None), "pp_rank", "?"), str(rid), int(floor), kind, int(told),
            int(live), t - t_floor, n,
        )
        return True
    except Exception:  # noqa: BLE001 - an instrument never breaks the admission
        return False


def release_own_read(scheduler, rid: str) -> None:
    """Drop this rank's store read and records for ``rid`` through the
    upstream abort path (see the module docstring); PP0 and followers alike."""
    tree = getattr(scheduler, "tree_cache", None)
    rel = getattr(tree, "release_aborted_request", None)
    if callable(rel):
        try:
            rel(str(rid))
        except Exception as exc:  # noqa: BLE001 - never leave the verdict unapplied
            logger.warning("PF TOLD-FALLBACK release_aborted_request(%s) raised: %r", str(rid), exc)
    else:
        # a tree without the abort path: at least drop the records admission reads
        for attr in ("_prefetch_completed_tokens", "prefetch_loaded_tokens_by_reqid"):
            d = getattr(tree, attr, None)
            if isinstance(d, dict):
                d.pop(str(rid), None)


# ---------------------------------------------------------------------------
# follower
# ---------------------------------------------------------------------------


@dataclass
class _FState:
    expect: Dict[str, int] = field(default_factory=dict)  # rid -> read-ahead told
    registered: Dict[str, Any] = field(default_factory=dict)  # rid -> req
    outbox: List[Tuple[Any, ...]] = field(default_factory=list)
    seq: int = 0
    #: T2 ROOM-DEFER: rid -> (first deferred at, next re-check at, last defer beat at, beats sent), perf clock
    deferred: Dict[str, Tuple[float, float, float, int]] = field(default_factory=dict)


def _fstate(scheduler, create: bool = False) -> Optional[_FState]:
    st = getattr(scheduler, "_pdflip_fb_follower", None)
    if st is None and create:
        st = scheduler._pdflip_fb_follower = _FState()
    return st


def follower_expect(scheduler, rid: str, told: int) -> None:
    """A read-ahead carrying ``ack`` arrived: report this rid's read."""
    st = _fstate(scheduler, create=True)
    st.expect[str(rid)] = int(told)
    while len(st.expect) > 1024:
        old = next(iter(st.expect))
        st.expect.pop(old, None)
        st.registered.pop(old, None)


def follower_note_registered(scheduler, req) -> None:
    st = _fstate(scheduler)
    if st is None:
        return
    rid = str(getattr(req, "rid", ""))
    if rid in st.expect:
        st.registered[rid] = req


def follower_reask(scheduler, rid: str, told: int, live: int) -> str:
    """W27-UNIFORM (item 180): a follower whose LIVE tree stays short of
    PP0's told (``pdflip_store_told.follower_hold_unreached``) re-asks PP0
    instead of seating on its own tree: its reach (``live``) goes out on the
    PF ack stream like any terminated read's, and PP0's rule stands -- an ack
    that differs from told means told=0 for EVERY rank (``pp0_decide``). The
    follower decides nothing; it holds until PP0 answers
    (``follower_release``) or its tree reaches told. ``ROUTE_NONE`` = the
    boot has no ack stream (PF off: no follower state), nothing is sent."""
    st = _fstate(scheduler)
    if st is None:
        return ROUTE_NONE
    st.outbox = [e for e in st.outbox if e[0] != str(rid)]
    st.outbox.append((str(rid), int(live)))
    return ROUTE_PF


def follower_forget(scheduler, rid: str) -> None:
    st = _fstate(scheduler)
    if st is None:
        return
    rid = str(rid)
    st.expect.pop(rid, None)
    st.registered.pop(rid, None)
    st.deferred.pop(rid, None)
    _room_defer_map(scheduler).pop(rid, None)
    (getattr(scheduler, "_pdflip_fb_room_defer_seen", None) or set()).discard(rid)
    if st.outbox:
        st.outbox = [e for e in st.outbox if e[0] != rid]


def follower_release(scheduler, rid: str) -> None:
    """``Admit(0, fallback)`` absorbed: cut this rank's read of ``rid``."""
    from flliper.srt.pdflip import p_twin_defer as _twin

    rid = str(rid)
    follower_forget(scheduler, rid)
    release_own_read(scheduler, rid)
    # item 180: the rank-uniform answer ends a hold for a tree short of told
    try:
        from flliper.srt.managers import pdflip_store_told as _st

        _st.forget_unreached(scheduler, rid)
    except Exception:  # noqa: BLE001 - bookkeeping
        pass
    (getattr(scheduler, "_pdflip_told_kept", None) or {}).pop(rid, None)
    satisfied = getattr(scheduler, "_pdflip_store_told_satisfied", None)
    if satisfied:
        satisfied.pop(rid, None)
    _twin.take_follower_twin(scheduler, rid)
    digests = getattr(scheduler, "_pdflip_told_keys_digest", None)
    if digests:
        digests.pop(rid, None)
    n = _bump(scheduler, "_pf_follower_release_n")
    if n <= 32 or n % _LOG_EVERY == 0:
        logger.warning(
            "PF TOLD-FALLBACK ABSORBED rank pp=%s rid=%s (n=%d): PP0 admitted at "
            "told=0; this rank's store read was released (abort path) and it "
            "admits at 0 like every rank", getattr(scheduler.ps, "pp_rank", "?"), rid, n,
        )


def _is_follower_twin(scheduler, rid: str) -> bool:
    from flliper.srt.pdflip import p_twin_defer as _twin

    st = getattr(scheduler, _twin._ATTR, None)
    return bool(st) and str(rid) in st.twin_follower


def own_prefix(scheduler, req, rid: str, told: int) -> int:
    """What ``pdflip_store_told.admission`` will compare with told -- read,
    never consumed."""
    from flliper.srt.managers import pdflip_store_told as _st
    from flliper.srt.pdflip import p_twin_defer as _twin

    satisfied = getattr(scheduler, "_pdflip_store_told_satisfied", None) or {}
    if rid in satisfied:
        return _satisfied_resumable(scheduler, req, rid, int(told))
    own = int(_st._completed_prefix(scheduler.tree_cache, rid))
    if _is_follower_twin(scheduler, rid):
        own += _twin.registered_head(req)
    return _resumable_own(scheduler, req, rid, own)


def _satisfied_resumable(scheduler, req, rid: str, told: int) -> int:
    """OVER-UNRESUMABLE (#968, 27B 673cc89f6a PP1 17:23:06Z; NF port): a SATISFIED
    rid (early read past told, or a local prefix covering it) used to ack told
    unprobed -- "holds the span" is not "can resume at told" on the hybrid tree
    (one node 0..N, recurrent state at N only: ``#904 refused ... Mamba:absent``,
    then ``#968 SHORTFALL`` after 45 s). The settle already refuses to mark an
    unresumable over-read satisfied (``pdflip_store_told.follower_early_settle``);
    this is the guard on the ack for every other way into ``satisfied``: the
    rank's own admission reach AT told (``rank_resumable``) below told acks that
    reach, so PP0 answers told=0 for every rank. No probe / dual layout /
    fidelity off = told, as before."""
    if told <= 0 or (os.environ.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() == "1":
        return told
    if not env_on():   # the ack only exists on the PF ring; kept for callers
        return told
    try:
        from flliper.srt.managers import pdflip_told_fidelity as _tf

        if not _tf.enabled():
            return told
        res = _tf.rank_resumable(scheduler, req, told)
    except Exception:  # noqa: BLE001 - a probe never breaks the ack
        return told
    if res is None or int(res) >= told:
        return told
    # H64-AUFLAGE 1 (B1, ARENAFULL-1015): GROUP-FLOOR admits equal follower acks below told AT that depth --
    # the reach acked here must be one this rank can also LOAD BACK, or a floor m runs into ROW-ROOM SHORT
    res = _room_own(scheduler, req, rid, int(res))
    n = _bump(scheduler, "_pf_ack_satisfied_unresumable_n")
    if n <= 32 or n % _LOG_EVERY == 0:
        logger.warning(
            "PF TOLD-ACK SATISFIED-UNRESUMABLE rank pp=%s rid=%s told=%d resumable=%d (n=%d): "
            "this rank counts the told span as held but its tree cannot resume there -- the "
            "ack says so and PP0 answers told=0 for every rank",
            getattr(scheduler.ps, "pp_rank", "?"), str(rid), told, int(res), n,
        )
    return int(res)


def _resumable_own(scheduler, req, rid: str, own: int) -> int:
    """ACK-RESUMABLE (N1 dkr27browauthoritybar1fs10010740, PP1 07:46:10Z,
    pdflip-10-16): a follower's read can complete the told KV span while its
    tree holds no recurrent state at that depth -- PP1 acked 17406 and then
    refused its own resume (``#928 ... best_value_len=0``), so PP0 admitted at
    told and the group died in #968 after a 19 s #1175 wait. The ack names
    what this rank's admission can actually RESUME from (the told-fidelity
    probe, the #928 rule read-only): KV without an anchor at its end acks
    less than told, and PP0 answers told=0 for EVERY rank (PF) -- a
    rank-agreed re-prefill instead of a group death. No probe = no verdict
    (the KV count stands, as before)."""
    if own <= 0:
        return own
    try:
        from flliper.srt.managers import pdflip_told_fidelity as _tf

        # W27-UNIFORM (open point (b) of item 025): the rank's OWN admission
        # reach -- capped match plus the #988 load-back's state-aligned extent
        # -- not PP0's plain TF probe: host KV whose end carries no recurrent
        # state loads back only to the anchor below it, and the ack must say so
        # (PP0 then answers told=0 for every rank) instead of acking a depth
        # this rank's admission stops short of (START-SPLIT, PPWidthDivergence).
        res = _tf.rank_resumable(scheduler, req, int(own))
    except Exception:  # noqa: BLE001 - a probe never breaks the ack
        return own
    if res is None or int(res) >= own:
        return _room_own(scheduler, req, rid, own)
    n = _bump(scheduler, "_pf_ack_unresumable_n")
    if n <= 32 or n % _LOG_EVERY == 0:
        logger.warning(
            "PF TOLD-ACK UNRESUMABLE rank pp=%s rid=%s kv=%d resumable=%d (n=%d): this "
            "rank's read completed the KV span but its tree cannot resume there (no "
            "recurrent state at the end) -- the ack says so and PP0 answers told=0 for "
            "every rank instead of admitting a prefix this rank cannot materialise",
            getattr(scheduler.ps, "pp_rank", "?"), str(rid), own, int(res), n,
        )
    # H64-AUFLAGE 1 (B1, ARENAFULL-1015): the reach below own is what GROUP-FLOOR may admit -- room-checked too
    return _room_own(scheduler, req, rid, int(res))


def _loadback_rows(scheduler, req, told: int) -> Optional[int]:
    """Rows this rank's admission must load back to hold ``told``: the matched
    depth that is HOST-only (the device part is held already). None = no probe."""
    try:
        tree = getattr(scheduler, "tree_cache", None)
        match = getattr(tree, "match_prefix", None)
        ids = getattr(req, "full_untruncated_fill_ids", None)
        if ids is None or len(ids) == 0:
            ids = getattr(req, "origin_input_ids", None)
        if tree is None or not callable(match) or ids is None or len(ids) == 0:
            return None
        from flliper.srt.managers.pdflip_store_told import _probe_key
        from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams

        key, _bigram = _probe_key(scheduler, req, ids, int(told))
        mr = match(MatchPrefixParams(key=key))
        _di = getattr(mr, "device_indices", None)
        dev = 0 if _di is None else int(_di.numel() if hasattr(_di, "numel") else len(_di))
        return max(0, min(int(told), dev + int(getattr(mr, "host_hit_length", 0) or 0)) - dev)
    except Exception:  # noqa: BLE001 - a probe never breaks the ack
        return None


def _ack_evictable(tree) -> int:
    """ACK-ROOM-PAYABLE (ARENAFULL-1015, NF h16n 07:30:35 PP1 pdflip-16-53): the evictable part of the ack's room is
    what ONE peel can pay -- the PR walk (``pp_room_vote.estimate_payable``: locked chains, un-backable leaves with
    host-locked children and the free arena room for backups all counted as the peel meets them), capped by the
    tree's reported size. The reported size alone said 255040 where the peel paid 48192, and the ack let PP0 admit a
    load-back this stage could not hold. Off (``FLLIPER_PDFLIP_ACK_ROOM_PAYABLE=0``) or unreadable walk: the reported
    evictable size, as before. Read-only."""
    reported = int(tree.evictable_size())
    try:
        from flliper.srt.environ import envs

        if not envs.FLLIPER_PDFLIP_ACK_ROOM_PAYABLE.get():
            return reported
        from flliper.srt.pdflip import pp_room_vote as _pr
        from flliper.srt.pdflip import pp_slot_fidelity as _sf

        cc = getattr(tree, "cache_controller", None)
        host_free = _pr.backup_room_tokens(getattr(cc, "mem_pool_host", None) if cc is not None else None)
        walked = _pr.estimate_payable(tree, local_floor=bool(getattr(tree, _sf.FLOOR_LOCAL_PP_ATTR, False)),
                                      host_free=host_free)
        return max(0, min(reported, int(walked)))
    except Exception:  # noqa: BLE001 - no walk: the reported size
        return reported


def _room_defer_map(scheduler) -> Dict[str, Tuple[int, int, int, int]]:
    d = getattr(scheduler, "_pdflip_fb_room_defer", None)
    if d is None:
        d = scheduler._pdflip_fb_room_defer = {}
    return d


def room_deferred(scheduler, rid: str) -> Optional[Tuple[int, int, int, int]]:
    """T2 ROOM-DEFER: ``(rows, room, locked, unpayable)`` when the LAST ``_room_own`` of ``rid`` found the room short
    only transiently -- while other requests hold rows (``room + locked >= rows``) or, ROOM-DEFER-UNPAYABLE, while
    evictable rows wait for a peel nobody can pay yet (``room + locked + unpayable`` = the pool >= rows); else None.
    Read once per ack formation."""
    return _room_defer_map(scheduler).pop(str(rid), None)


def _locked_rows(scheduler, available: int, evictable: int) -> int:
    """Rows other requests hold on this stage's device pool: ``max_total_num_tokens - (available + evictable)`` --
    the very ``num_used`` the scheduler's ``full token usage`` line prints (h16p: 0.38 = 99.6k rows at the NO-ROOM
    ack, 0.06 one batch later). They return to the pool when those requests finish. 0 = not readable."""
    total = int(getattr(scheduler, "max_total_num_tokens", 0) or 0)
    return max(0, total - int(available) - int(evictable)) if total > 0 else 0


def _room_own(scheduler, req, rid: str, own: int, mark: bool = True) -> int:
    """ACK-ROOM (dual1k dkr27bnvfp4dual1kbar1fs10010950, PP1 09:55:20Z, rid
    pdflip-0-10, a fork twin): the follower's read reproduced told=16383 and its
    tree could resume there (host KV + anchor), but the head was HOST-only on
    this rank (PP0/PP2 held it on the device) and the load-back found no room:
    ``SF LOADBACK-ROOM PP-RESIDUAL kv_tokens=12288 avail=1717 evictable=0`` --
    a concurrent 61440-token prefill held the pool. PP0 had admitted at told,
    so #968 followed at once. The ack names 0 when this rank cannot hold the
    told's load-back even with every evictable row freed: PP0 answers told=0
    for EVERY rank (a rank-agreed re-prefill, chunked from 0) instead of a
    group death. Read-only (nothing evicted here); no probe = no verdict.

    T2 ROOM-DEFER (NF h16p 226aad7105, PP1/PP2 pdflip-0-3 12:46:20 ``loadback_rows=184768 room=162176``, pdflip-0-10
    12:33:29 ``188928 / 129472``): the 0 re-prefilled what the store held on EVERY stage -- for a shortage that was a
    RUNNING request's rows (usage 0.38, then 0.06 one batch later). When room + the rows others hold would suffice,
    the shortage is transient: the call still returns 0 (nothing may be loaded back NOW), but with ``mark`` it
    records ``(rows, room, locked)`` for ``room_deferred`` -- the ack is then held back instead of sent as 0."""
    if mark:
        _room_defer_map(scheduler).pop(str(rid), None)
    rows = _loadback_rows(scheduler, req, own)
    if not rows:
        return own
    try:
        tree = scheduler.tree_cache
        alloc = getattr(tree, "token_to_kv_pool_allocator", None)
        available = int(alloc.available_size())
        reported = int(tree.evictable_size())
        room = available + _ack_evictable(tree)
    except Exception:  # noqa: BLE001 - no allocator readable: no verdict
        return own
    if room >= int(rows):
        return own
    locked = _locked_rows(scheduler, available, reported)
    # review A2 (B2): room + locked + unpayable == the pool. ``unpayable`` = reported evictable rows no peel pays yet
    # (the backups they need wait for arena slots: the FULL-ARENA state). A pool that holds the told at all gets it
    # once those clear -- transient, like the rows other requests hold. Only a told that exceeds the pool (or a pool
    # that is unreadable) is permanent.
    unpayable = max(0, int(reported) - (room - int(available)))
    total = int(getattr(scheduler, "max_total_num_tokens", 0) or 0)
    transient = bool(
        room_defer_on()
        and (room + locked >= int(rows) or (room_defer_unpayable_on() and total > 0 and int(rows) <= total))
    )
    if mark and transient:
        _room_defer_map(scheduler)[str(rid)] = (int(rows), int(room), int(locked), int(unpayable))
        seen = getattr(scheduler, "_pdflip_fb_room_defer_seen", None)
        if seen is None:
            seen = scheduler._pdflip_fb_room_defer_seen = set()
        if str(rid) in seen:
            return 0  # named on the first formation; a re-check every pass would only repeat it
        seen.add(str(rid))
        n = _bump(scheduler, "_pf_ack_room_defer_n")
        if _say(n):
            logger.warning(
                "PF TOLD-ACK ROOM-DEFER rank pp=%s rid=%s told=%d loadback_rows=%d room=%d locked=%d unpayable=%d "
                "class=%s (n=%d): the load-back does not fit now but fits once the %d rows other requests hold are "
                "released%s -- the ack is held back (a defer beat tells PP0 this rank lives; PP0 waits, it does not "
                "answer told=0) and goes out when the room is there",
                getattr(scheduler.ps, "pp_rank", "?"), str(rid), own, int(rows), room, locked, unpayable,
                "locked" if room + locked >= int(rows) else "unpayable", n, locked,
                "" if room + locked >= int(rows) else f" and the {unpayable} evictable rows no peel pays yet clear "
                "(the full-arena state: their backups wait for arena slots)",
            )
        return 0
    n = _bump(scheduler, "_pf_ack_no_room_n")
    if n <= 32 or n % _LOG_EVERY == 0:
        logger.warning(
            "PF TOLD-ACK NO-ROOM rank pp=%s rid=%s told=%d loadback_rows=%d room=%d (n=%d): "
            "this rank holds the told span on its HOST only and cannot load it back even "
            "with every evictable row freed -- the ack says 0 and PP0 answers told=0 for "
            "every rank instead of an SF LOADBACK-ROOM residual and #968 after PP0 admitted",
            getattr(scheduler.ps, "pp_rank", "?"), str(rid), own, int(rows), room, n,
        )
    return 0


def _path_anchor_depths(tree, node) -> List[int]:
    """B5: the state-bearing boundaries on the path root..``node``, deepest first, in KEY units (the walk's
    ``cum_tokens``): every node the tree's own match validators accept (mamba's ``is_resume_candidate`` among them --
    the predicate ``_state_anchor_depth`` reads off, never a second rule). Read-only."""
    comps = getattr(tree, "_components_tuple", None)
    root = getattr(tree, "root_node", None)
    if not comps or node is None or root is None:
        return []
    device_only = getattr(tree, "cache_controller", None) is None
    validators = tuple(c.create_match_validator(match_device_only=device_only) for c in comps)
    path = []
    n = node
    while n is not None and n is not root:
        path.append(n)
        n = n.parent
    depth = sum(len(x.key) for x in path)
    out: List[int] = []
    for x in path:  # deepest first
        try:
            if all(v(x, depth) for v in validators):
                out.append(int(depth))
        except Exception:  # noqa: BLE001 - a node a validator cannot read is no anchor
            pass
        depth -= len(x.key)
    return out


def ack_anchors(scheduler, req, rid: str, own: int, told: int) -> List[int]:
    """B5 ANCHOR-ACKS: the depths below ``own`` (and below ``told``) this rank can resume ``rid`` from AND load
    back -- the anchors on its matched path to ``max(own, told)``, room-checked like the ack itself (H64-AUFLAGE 1:
    a floor this rank cannot load back would run into ROW-ROOM SHORT). Deepest first, at most ``ANCHOR_ACKS_MAX``.
    No probe = none (the ack alone, as before). Read-only."""
    try:
        tree = getattr(scheduler, "tree_cache", None)
        match = getattr(tree, "match_prefix", None)
        ids = getattr(req, "full_untruncated_fill_ids", None)
        if ids is None or len(ids) == 0:
            ids = getattr(req, "origin_input_ids", None)
        if tree is None or not callable(match) or ids is None or len(ids) == 0:
            return []
        from flliper.srt.managers.pdflip_store_told import _probe_key
        from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams

        reach = max(int(own), int(told))
        if reach <= 0:
            return []
        key, _bigram = _probe_key(scheduler, req, ids, reach)
        mr = match(MatchPrefixParams(key=key))
        node = getattr(mr, "best_match_node", None)
        cap = int(own) if int(own) > 0 else int(told)
        depths = [d for d in _path_anchor_depths(tree, node) if 0 < d < int(told) and (d < cap or int(own) <= 0)]
    except Exception:  # noqa: BLE001 - a probe never breaks the ack
        return []
    out: List[int] = []
    for d in depths:
        if int(own) > 0 and d == int(own):
            continue
        if _room_own(scheduler, req, rid, d, mark=False) != d:
            continue  # this depth's load-back does not fit: not a depth this rank proves
        out.append(int(d))
        if len(out) >= ANCHOR_ACKS_MAX:
            break
    return out


def _progress_free(tree) -> bool:
    fn = getattr(tree, "prefetch_progress_is_collective_free", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001
            return False
    return True  # the armed form is tp_size 1 by construction


def follower_pump(scheduler) -> None:
    """Every follower pass (fallback read-aheads seen): report terminated
    reads to PP0 and finish the previous send -- never waits."""
    st = _fstate(scheduler)
    if st is None:
        return
    census_tick(scheduler)
    tree = scheduler.tree_cache
    free = None
    for rid in list(st.registered):
        told = st.expect.get(rid)
        if told is None:
            st.registered.pop(rid, None)
            continue
        if free is None:
            free = _progress_free(tree)
        if not free:
            break  # a collective here would be the #580 class; the Frist decides
        if not tree.check_prefetch_progress(rid):
            continue
        if getattr(st.registered[rid], "_pdflip_early_told", None) is not None:
            # DP-NACHLAUF (N5p): an early read -- settle it against told first
            from flliper.srt.managers import pdflip_store_told as _st

            if not _st.follower_early_settle_now(scheduler, st.registered[rid], rid, told):
                continue  # short: the told-limited read acks when it ends
        req = st.registered[rid]
        # T2 ROOM-DEFER: a deferred rid re-reads its room at most every ROOM_DEFER_RECHECK_S
        _held = st.deferred.get(rid)
        if _held is not None and _perf() < _held[1]:
            continue
        own = own_prefix(scheduler, req, rid, told)
        _dfr = room_deferred(scheduler, rid) if room_defer_on() else None
        if _dfr is not None:
            # the room is short only transiently: no ack yet -- PP0 reads the missing ack as lag and waits
            # (WAIT-PROGRESS); the real reach goes out when the room is there. The read stays registered. DEFER-
            # LIFESIGN: a defer entry in the ack message tells PP0 this rank lives and what it waits for.
            _now = _perf()
            _beat_at = _held[2] if _held is not None else float("-inf")
            _beats = _held[3] if _held is not None else 0
            if defer_lifesign_on() and _now - _beat_at >= _env_float(ENV_DEFER_BEAT_S, DEFER_BEAT_S_DEFAULT):
                st.outbox = [e for e in st.outbox if e[0] != rid]
                st.outbox.append((rid, 0, (), (DEFER_MARK,) + tuple(int(x) for x in _dfr)))
                _beat_at = _now
                _beats += 1
            st.deferred[rid] = (
                _held[0] if _held is not None else _now, _now + ROOM_DEFER_RECHECK_S, _beat_at, _beats,
            )
            continue
        if _held is not None:
            n_r = _bump(scheduler, "_pf_ack_room_deferred_released_n")
            if _say(n_r):
                logger.info(
                    "PF TOLD-ACK ROOM-DEFER RELEASED rank pp=%s rid=%s told=%d own=%d after=%.2fs beats=%d (n=%d): "
                    "the room is there, the ack goes out with the real reach; %s",
                    getattr(scheduler.ps, "pp_rank", "?"), rid, told, int(own), _perf() - _held[0],
                    int(_held[3]), n_r, _wait_context(scheduler, rid),
                )
        st.deferred.pop(rid, None)
        (getattr(scheduler, "_pdflip_fb_room_defer_seen", None) or set()).discard(rid)
        st.registered.pop(rid, None)
        st.expect.pop(rid, None)
        # the real ack replaces any defer beat still waiting to go out for this rid
        st.outbox = [e for e in st.outbox if e[0] != rid]
        if anchor_acks_on():
            st.outbox.append((rid, own, tuple(ack_anchors(scheduler, req, rid, own, told))))
        else:
            st.outbox.append((rid, own))
    ch = _channel(scheduler)
    if not st.outbox:
        ch.pump()
        return
    ack = PdFlipToldReadAck(rank=int(scheduler.ps.pp_rank), seq=st.seq, reads=list(st.outbox))
    if ch.send_nowait(ack):
        st.outbox = []
        st.seq += 1
        n = _bump(scheduler, "_pf_ack_sent_n")
        if _say(n):
            logger.info(
                "PF TOLD-ACK SENT rank pp=%s seq=%d reads=%s (n=%d)",
                scheduler.ps.pp_rank, ack.seq, list(ack.reads), n,
            )


# ---------------------------------------------------------------------------
# K20 (17.10.): the standing receives at sleep
# ---------------------------------------------------------------------------

#: scheduler attributes holding the GlooAckChannel of each stream (told-ack, pp-room)
_STREAM_ATTRS = ("_pdflip_fb_channel", "_pdflip_pr_channel")


def _streams(scheduler) -> List[GlooAckChannel]:
    out = []
    for a in _STREAM_ATTRS:
        ch = getattr(scheduler, "__dict__", {}).get(a) if hasattr(scheduler, "__dict__") else getattr(scheduler, a, None)
        if ch is not None:
            out.append(ch)
    return out


def sleep_quiesce(scheduler, budget_s: Optional[float] = None) -> int:
    """K20 root fix. PP0's standing ``ObjectRecvFrame`` on each ack stream is a posted gloo receive; gloo's clock on a
    posted receive is the process-group timeout (7200 s here) and nothing ever stops it: boot 1016_172204 re-armed the
    frame on the last ack of rid pdflip-178-2512 (22:36:54Z), P slept at 22:36:57Z, no pass ran, and at 00:36:54Z PP0
    died on ``PF told-ack receive[pp1] ... Timed out waiting 7200000ms`` (the pp-room receive followed as a cascade).

    At sleep every follower completes PP0's standing receive with a ``bye`` (its own rank, bounded), PP0 takes the byes
    and does NOT re-arm those frames until ``wake``. PP0 decides, the followers only report: no rank-local abort.
    Returns the number of receives still open on THIS rank (PP0; followers return 0). The census line is emitted by
    every rank; ``open`` must be 0."""
    budget = _env_float(ENV_SLEEP_COLLECT_S, SLEEP_COLLECT_S_DEFAULT) if budget_s is None else float(budget_s)
    rank = int(scheduler.ps.pp_rank)
    chans = _streams(scheduler)
    if int(getattr(scheduler.ps, "pp_size", 1) or 1) <= 1 or not chans:
        logger.info("PDFLIP PP-RECEIVE-CENSUS at=sleep rank=pp%d open=0 streams=0", rank)
        return 0
    t0 = time.monotonic()
    open_n = 0
    detail = []
    if rank != 0:
        for ch in chans:
            ok = ch.send_bye(budget)
            detail.append(f"{ch.label}:bye={'sent' if ok else 'UNSENT'},held={ch.held}")
            if not ok:
                open_n += 1
    else:
        end = t0 + budget
        while True:
            for ch in chans:
                ch.harvest()  # byes are consumed inside the channel; an ack still in flight at sleep is dropped (PP0's Frist: told=0)
            open_n = sum(ch.open_receives() for ch in chans)
            if open_n == 0 or time.monotonic() >= end:
                break
            time.sleep(0.005)
        for ch in chans:
            ch._bye_missing = len(ch._sleeping) < len(ch.world_ranks) - 1
        detail = [f"{ch.label}:open={ch.open_receives()},byes={ch.byes_taken},beats={ch.beats_taken}" for ch in chans]
    open_sends = sum(ch.open_sends() for ch in chans)
    logger.log(
        logging.INFO if (open_n == 0 and open_sends == 0) else logging.ERROR,
        "PDFLIP PP-RECEIVE-CENSUS at=sleep rank=pp%d open=%d open_sends=%d streams=%d waited_s=%.2f %s",
        rank, open_n, open_sends, len(chans), time.monotonic() - t0, " ".join(detail),
    )
    return open_n + open_sends


def wake(scheduler) -> None:
    """PP0: re-arm the standing receives after the group woke (the next harvest posts them)."""
    for ch in _streams(scheduler):
        ch.wake()
