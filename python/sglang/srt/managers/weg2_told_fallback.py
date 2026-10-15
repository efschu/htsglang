"""PF (26.09.): the GROUP told=0 fallback of the paced told (#1416e).

THE FAILURE IT REMOVES. With ``SGLANG_WEG2_TOLD_PACED`` PP0 publishes a
read-ahead ``Weg2StoreTold(paced=True)`` and, once its pacing window (an
ESTIMATE of the followers' store read, capped at 10 s) has passed, the
membership verdict ``Weg2StoreAdmit(rid, told)``. A follower whose read is
still running at the Admit falls into the bounded busy-wait of
``weg2_store_told.admission`` -- the whole stage stops -- and after
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

THE CHANNEL (``WEG2_TOLD_ACK_TAG``). Each follower sends PP0 a
``Weg2ToldReadAck(rank, seq, reads=[(rid, own_prefix), ...])`` for every
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
``SGLANG_HICACHE_ARENA_QUEUE_REFS``, release-table row 30 / #718-stray), its
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

SWITCH ``SGLANG_WEG2_TOLD_GROUP_FALLBACK`` (unset = the profile row: ON on
nextflash since 14.10., off on qwen27b), read once on PP0,
effective only with ``SGLANG_WEG2_TOLD_PACED`` on: only there does PP0 hold
admission back until an Admit.

H64 (NF h16n 17c2563647, boot ..._1015_042125, P 04:30:45 weg2-6-44 and
04:31:14 weg2-8-47 -- 130k tokens recomputed the store held). Three parts,
each PP0-only or wire-following, so no rank can leave its peers:

* FRIST-PASSES (``SGLANG_WEG2_TOLD_FALLBACK_MIN_PASSES``, default pp_size+1;
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
* ACK-DRAIN (``SGLANG_WEG2_TOLD_ACK_DRAIN``, default on). PP0 harvested the
  ack stream only while it had a paced rid open; an ack whose size frame was
  taken in the deciding pass left its payload unposted, the follower's send
  never completed and every later ack of that follower queued behind it --
  PP2's weg2-8-47 ack surfaced 3.5 min later as ``PF TOLD-ACK LATE``. PP0 now
  polls the standing frames on every pass once the stream exists (a flag read
  per follower).
* GROUP-FLOOR (``SGLANG_WEG2_TOLD_GROUP_FLOOR``, default on). An ack != told
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

from sglang.srt.managers.weg2_idle_vote import FIXED_P2P_TAG_BASE

logger = logging.getLogger(__name__)

ENV_FALLBACK = "SGLANG_WEG2_TOLD_GROUP_FALLBACK"
ENV_GRACE_S = "SGLANG_WEG2_TOLD_FALLBACK_GRACE_S"
ENV_CAP_S = "SGLANG_WEG2_TOLD_FALLBACK_CAP_S"
ENV_TOTAL_S = "SGLANG_WEG2_TOLD_FALLBACK_TOTAL_S"
GRACE_S_DEFAULT = 2.0
CAP_S_DEFAULT = 12.0
#: PP0 intake -> verdict, below the #699 ADMISSION_WEDGE_SECONDS (20 s) with
#: room for the Admit's own lag down the pipe and the told=0 prefill start.
TOTAL_S_DEFAULT = 16.0

#: the ack stream's own point-to-point tag (1268 = idle vote, 580 = the
#: prefetch vote collective, 0 = the typed proxy/output channel).
#: 1538: shifted above ``FIXED_P2P_TAG_BASE`` (see ``weg2_idle_vote``): the
#: group fence's ``monitored_barrier`` uses the collective counter as a raw p2p
#: slot on this same world cpu group, and at ``t1 == 1416`` PP0's standing ack
#: frames swallowed both followers' barrier messages (fence #337, boot
#: fs10052155, 27B with PF on).
WEG2_TOLD_ACK_TAG = FIXED_P2P_TAG_BASE + 1416
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
ENV_MIN_PASSES = "SGLANG_WEG2_TOLD_FALLBACK_MIN_PASSES"
ENV_ACK_DRAIN = "SGLANG_WEG2_TOLD_ACK_DRAIN"
ENV_FLOOR = "SGLANG_WEG2_TOLD_GROUP_FLOOR"
#: B5 ANCHOR-ACKS (NF h16p 226aad7105, 15.10. 12:04:24 / 12:16:46Z): a follower's ack carries the resumable anchor
#: depths below its reach; PP0 floors at the deepest depth EVERY stage (itself included) can resume from.
ENV_ANCHOR_ACKS = "SGLANG_WEG2_TOLD_ANCHOR_ACKS"
#: at most this many anchor depths per ack entry (deepest first)
ANCHOR_ACKS_MAX = 16
#: T2 WAIT-PROGRESS (NF h16p 226aad7105, 15.10.: the 4 ``reason=frist`` cases, 365k tokens): a missing ack is LAG,
#: not absence. Every late ack the log shows (PP2 weg2-33-81 own=121344 at +6 s, PP2 weg2-4-25 own=43392 at +4 s) named
#: EXACTLY told -- PP0 had thrown the whole prefix away for an ack that was on its way. The Frist stays the bound of an
#: IDLE box (the #699 wedge clock runs while nothing serves); while the box is SERVING -- the detector that guards that
#: very clock (``invariant_checker.admission_wedge_verdict``, at the alarm threshold less ``ENV_WAIT_MARGIN_S``) would
#: not call it wedged -- PP0 keeps waiting: the chain lag that held the ack IS the serving work. ``ENV_WAIT_HARD_S``
#: from the read-ahead bounds a dead stream (frist_hard).
ENV_WAIT_PROGRESS = "SGLANG_WEG2_TOLD_WAIT_PROGRESS"
ENV_WAIT_HARD_S = "SGLANG_WEG2_TOLD_WAIT_HARD_S"
ENV_WAIT_MARGIN_S = "SGLANG_WEG2_TOLD_WAIT_MARGIN_S"
WAIT_HARD_S_DEFAULT = 60.0
WAIT_MARGIN_S_DEFAULT = 5.0
REASON_FRIST_HARD = "frist_hard"
#: T2 ROOM-DEFER (same boots, weg2-0-3 12:46:20 / weg2-0-10 12:33:29): ``PF TOLD-ACK NO-ROOM loadback_rows=184768
#: room=162176`` acked 0 and PP0 re-prefilled 185k the store held. The room was short because a RUNNING prefill held
#: 100k rows ("full token usage: 0.38", one batch later 0.06): a shortage that lasts as long as that request. A follower
#: whose room would suffice once the rows other requests hold are released keeps its ack back (PP0 reads a missing ack
#: as lag and waits, WAIT-PROGRESS) and acks the real reach when the room is there. Not enough even then (rows >
#: room + locked): the ack says 0 as before (B5's anchors below still apply).
ENV_ROOM_DEFER = "SGLANG_WEG2_TOLD_ROOM_DEFER"
#: a deferred ack re-reads the room at most this often (seconds)
ROOM_DEFER_RECHECK_S = 0.05
_perf = time.perf_counter
#: item 180: where a follower's re-ask of PP0 went (follower_reask)
ROUTE_PF = "pf-ack"
ROUTE_NONE = "none"

_LOG_FIRST = 8
_LOG_EVERY = 256


def env_on() -> bool:
    # RG 26.09.: unset = the registry row (weg2/form.py PREFIX_SWITCHES; PF is
    # on on the nextflash row since 14.10., off on qwen27b).
    from sglang.srt.weg2.form import prefix_switch_armed

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


def deadline(published_at: float, own_read_s: float, window_s: float) -> float:
    total = _env_float(ENV_TOTAL_S, TOTAL_S_DEFAULT)
    intake_at = float(published_at) - max(0.0, float(own_read_s))
    return max(float(published_at), min(float(published_at) + frist_s(window_s), intake_at + total))


def _say(n: int) -> bool:
    return n <= _LOG_FIRST or n % _LOG_EVERY == 0


def _bump(obj, attr: str) -> int:
    n = int(getattr(obj, attr, 0) or 0) + 1
    setattr(obj, attr, n)
    return n


@dataclass
class Weg2ToldReadAck:
    """Follower -> PP0 on ``WEG2_TOLD_ACK_TAG``: the reads this rank has
    terminated since its last ack, as ``(rid, own_prefix)`` -- or, B5
    ANCHOR-ACKS, ``(rid, own_prefix, anchors)``: ``anchors`` = the depths below
    ``own_prefix`` this rank can ALSO resume from (state-bearing boundaries on
    its matched path, deepest first). The same message, a longer entry; PP0
    reads both forms."""

    rank: int
    seq: int
    reads: List[Tuple[str, int]] = field(default_factory=list)


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
                 tag: int = WEG2_TOLD_ACK_TAG, label: str = "told-ack"):
        self.group = group
        self.pp_rank = int(pp_rank)
        self.world_ranks = list(world_ranks)
        #: PR (weg2/pp_room_vote.py) runs a second stream of the same shape on
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

    @classmethod
    def for_scheduler(cls, scheduler) -> "GlooAckChannel":
        group = getattr(getattr(scheduler, "world_group", None), "cpu_group", None)
        return cls(group, int(scheduler.ps.pp_rank), _world_ranks(scheduler))

    # -- follower ------------------------------------------------------
    def pump(self) -> bool:
        """True when nothing is in flight any more. A completed send is
        joined (instantly: its flag is set) only to surface its error."""
        if self._inflight is None:
            return True
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

    def send_nowait(self, ack: Weg2ToldReadAck) -> bool:
        """Post ``ack`` unless the previous one is still in flight. True =
        the ack left the outbox (posted, or dropped on a dead stream)."""
        if self.dead:
            return True
        if not self.pump():
            return False
        import numpy as np
        import torch
        import torch.distributed as dist

        from sglang.srt.mem_cache.hicache_collective import ParkedWait

        payload = pickle.dumps(ack)
        size_t = torch.tensor([len(payload)], dtype=torch.long)
        data_t = torch.ByteTensor(np.frombuffer(payload, dtype=np.uint8).copy())
        dst = self.world_ranks[0]
        works = []
        try:
            for label, t in (("size", size_t), ("payload", data_t)):
                w = dist.isend(t, dst, group=self.group, tag=self.tag)
                works.append((ParkedWait(w, f"weg2/{self.label}/{label}"), t))
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
            from sglang.srt.distributed.pp_object_recv import ObjectRecvFrame

            src = self.world_ranks[pp_rank]
            frame = self._frames[pp_rank] = ObjectRecvFrame(
                group=self.group,
                src_global=src,
                tag=self.tag,
                site=f"weg2/{self.label}[pp{pp_rank}]",
                rank_desc="pp_rank=0",
            )
        return frame

    def harvest(self) -> List[Any]:
        out: List[Any] = []
        for r in range(1, len(self.world_ranks)):
            frame = self._frame(r)
            for _ in range(HARVEST_MAX_PER_SRC):
                try:
                    if not frame.poll():
                        break
                    out.append(frame.take())
                except Exception as exc:  # noqa: BLE001 - the Frist covers it
                    self._error(f"receive[pp{r}]", exc)
                    break
        return out


def _channel(scheduler):
    ch = getattr(scheduler, "_weg2_fb_channel", None)
    if ch is None:
        ch = scheduler._weg2_fb_channel = GlooAckChannel.for_scheduler(scheduler)
    return ch


# ---------------------------------------------------------------------------
# PP0
# ---------------------------------------------------------------------------


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


def _pp0_open_map(scheduler) -> Dict[str, _Open]:
    d = getattr(scheduler, "_weg2_fb_open", None)
    if d is None:
        d = scheduler._weg2_fb_open = {}
    return d


def pp0_open(scheduler, rid: str, told: int, published_at: float, own_read_s: float, window_s: float,
             published_pass: Optional[int] = None) -> float:
    """PP0 put a paced read-ahead with ``ack`` on the wire for ``rid``."""
    dl = deadline(published_at, own_read_s, window_s)
    _pp0_open_map(scheduler)[str(rid)] = _Open(
        told=int(told), deadline=dl,
        published_pass=-1 if published_pass is None else int(published_pass),
        published_at=float(published_at),
    )
    return dl - float(published_at)


def pp0_has_stream(scheduler) -> bool:
    """H64 ACK-DRAIN: the ack stream exists on PP0 (its frames were posted)."""
    return getattr(scheduler, "_weg2_fb_channel", None) is not None and ack_drain_on()


def pp0_forget(scheduler, rid: str) -> None:
    _pp0_open_map(scheduler).pop(str(rid), None)


#: item 220: ``rid -> told`` PP0 admitted (told > 0) and has not seated yet
RETRACT_ADMITTED_MAX = 1024


def _admitted_map(scheduler) -> Dict[str, int]:
    d = getattr(scheduler, "_weg2_fb_admitted", None)
    if d is None:
        d = scheduler._weg2_fb_admitted = {}
    return d


def _late_map(scheduler) -> Dict[str, int]:
    d = getattr(scheduler, "_weg2_fb_late", None)
    if d is None:
        d = scheduler._weg2_fb_late = {}
    return d


def pp0_watching(scheduler) -> bool:
    """True while PP0 holds an admitted-at-told rid a late ack could still
    retract: the publish pass keeps harvesting the ack stream for it."""
    return bool(getattr(scheduler, "_weg2_fb_admitted", None))


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
        from sglang.srt.managers import weg2_store_told as _st

        _st.forget_unreached(scheduler, rid)
    except Exception:  # noqa: BLE001 - bookkeeping
        pass
    (getattr(scheduler, "_weg2_told_kept", None) or {}).pop(rid, None)
    (getattr(scheduler, "_weg2_store_told_satisfied", None) or {}).pop(rid, None)
    n = _bump(scheduler, "_pf_retract_n")
    if n <= 32 or n % _LOG_EVERY == 0:
        logger.warning(
            "PF TOLD-RETRACT rid=%s told=%d -> 0 follower_reach=%d (n=%d): a follower's tree "
            "stayed short of the admitted told after the Admit and PP0 had not seated the "
            "rid yet -- PP0 switches it to told=0 for EVERY rank on the Admit channel "
            "(fallback marker); P recomputes the prefix", rid, told, own, n,
        )


def pp0_harvest(scheduler) -> int:
    """Take every ack that has landed off the standing receives (no wait)."""
    open_map = _pp0_open_map(scheduler)
    n = 0
    for ack in _channel(scheduler).harvest():
        if not isinstance(ack, Weg2ToldReadAck):
            logger.warning("PF TOLD-ACK object %s on the ack stream dropped", type(ack).__name__)
            continue
        for entry in ack.reads:
            rid, own = entry[0], entry[1]
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
    (``weg2_told_fidelity.pp0_admissible`` at that depth, no side effect), else 0. PP0 alone decides; the followers
    proved every candidate by acking it. Read before ``pp0_note_verdict`` pops the open entry."""
    if not anchor_acks_on():
        return 0
    o = _pp0_open_map(scheduler).get(str(rid))
    if o is None:
        return 0
    try:
        from sglang.srt.managers import weg2_told_fidelity as _tf
    except Exception:  # noqa: BLE001
        return 0
    for d in common_depths(o):
        if d >= int(below):
            continue
        depth = _tf.absolute_depth(req, d, absolute)
        own = _tf.pp0_admissible(scheduler, req, depth)
        if own is not None and int(own) >= depth:
            return int(d)
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
        _ic = sys.modules.get("sglang.srt.managers.scheduler_components.invariant_checker")
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


def _frist_verdict(scheduler, rid: str, o: _Open, now: float, pass_n: Optional[int]) -> Optional[Tuple[int, str]]:
    """The acks are not all in. ``None`` = keep waiting (the Frist is not due, or T2 WAIT-PROGRESS: it is due but the
    box is serving -- the late ack is chain lag, and the lag IS the serving work); else ``(0, reason)``."""
    if not _frist_due(scheduler, rid, o, now, pass_n):
        return None
    if not wait_progress_on():
        return 0, REASON_FRIST
    serving, detail = box_serving(scheduler)
    if not serving:
        return 0, REASON_FRIST
    if now - o.published_at >= _wait_hard_s():
        return 0, REASON_FRIST_HARD
    if o.waited_since is None:
        o.waited_since = float(now)
    o.wait_n += 1
    if o.wait_n == 1:
        n = _bump(scheduler, "_pf_wait_n")
        if _say(n):
            logger.info(
                "PF TOLD-WAIT rid=%s told=%d after=%.2fs acks=%s (n=%d): the Frist is up but the box is serving "
                "(%s) -- a missing ack is chain lag, PP0 keeps the rid and waits for it instead of told=0",
                rid, o.told, now - o.published_at, dict(o.acks), n, detail[:160],
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
                "PF TOLD-WAITED rid=%s told=%d -> %d reason=%s waited=%.2fs polls=%d acks=%s (n=%d): the verdict "
                "came after the Frist while the box served", rid, told, int(told_final), reason,
                now - o.waited_since, o.wait_n, acks, n_w,
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
    from sglang.srt.managers import weg2_store_told as _st

    rid = str(rid)
    follower_forget(scheduler, rid)
    try:
        _st._pop_credit_keep_pin(scheduler.tree_cache, rid)
    except Exception:  # noqa: BLE001 - bookkeeping; the satisfied mark decides
        pass
    sat = getattr(scheduler, "_weg2_store_told_satisfied", None)
    if sat is None:
        sat = scheduler._weg2_store_told_satisfied = {}
    sat[rid] = int(floor)
    n = _bump(scheduler, "_pf_floor_applied_n")
    if n <= 32 or n % _LOG_EVERY == 0:
        logger.info(
            "PF TOLD-FLOOR APPLIED rank pp=%s rid=%s floor=%d (n=%d): satisfied at PP0's floor, "
            "nothing released, admits at that depth like every rank",
            getattr(scheduler.ps, "pp_rank", "?"), rid, int(floor), n,
        )


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
    #: T2 ROOM-DEFER: rid -> (first deferred at, next re-check at), perf clock
    deferred: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def _fstate(scheduler, create: bool = False) -> Optional[_FState]:
    st = getattr(scheduler, "_weg2_fb_follower", None)
    if st is None and create:
        st = scheduler._weg2_fb_follower = _FState()
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
    PP0's told (``weg2_store_told.follower_hold_unreached``) re-asks PP0
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
    (getattr(scheduler, "_weg2_fb_room_defer_seen", None) or set()).discard(rid)
    if st.outbox:
        st.outbox = [e for e in st.outbox if e[0] != rid]


def follower_release(scheduler, rid: str) -> None:
    """``Admit(0, fallback)`` absorbed: cut this rank's read of ``rid``."""
    from sglang.srt.weg2 import p_twin_defer as _twin

    rid = str(rid)
    follower_forget(scheduler, rid)
    release_own_read(scheduler, rid)
    # item 180: the rank-uniform answer ends a hold for a tree short of told
    try:
        from sglang.srt.managers import weg2_store_told as _st

        _st.forget_unreached(scheduler, rid)
    except Exception:  # noqa: BLE001 - bookkeeping
        pass
    (getattr(scheduler, "_weg2_told_kept", None) or {}).pop(rid, None)
    satisfied = getattr(scheduler, "_weg2_store_told_satisfied", None)
    if satisfied:
        satisfied.pop(rid, None)
    _twin.take_follower_twin(scheduler, rid)
    digests = getattr(scheduler, "_weg2_told_keys_digest", None)
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
    from sglang.srt.weg2 import p_twin_defer as _twin

    st = getattr(scheduler, _twin._ATTR, None)
    return bool(st) and str(rid) in st.twin_follower


def own_prefix(scheduler, req, rid: str, told: int) -> int:
    """What ``weg2_store_told.admission`` will compare with told -- read,
    never consumed."""
    from sglang.srt.managers import weg2_store_told as _st
    from sglang.srt.weg2 import p_twin_defer as _twin

    satisfied = getattr(scheduler, "_weg2_store_told_satisfied", None) or {}
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
    unresumable over-read satisfied (``weg2_store_told.follower_early_settle``);
    this is the guard on the ack for every other way into ``satisfied``: the
    rank's own admission reach AT told (``rank_resumable``) below told acks that
    reach, so PP0 answers told=0 for every rank. No probe / dual layout /
    fidelity off = told, as before."""
    if told <= 0 or (os.environ.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() == "1":
        return told
    if not env_on():   # the ack only exists on the PF ring; kept for callers
        return told
    try:
        from sglang.srt.managers import weg2_told_fidelity as _tf

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
    weg2-10-16): a follower's read can complete the told KV span while its
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
        from sglang.srt.managers import weg2_told_fidelity as _tf

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
        from sglang.srt.managers.weg2_store_told import _probe_key
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

        key, _bigram = _probe_key(scheduler, req, ids, int(told))
        mr = match(MatchPrefixParams(key=key))
        _di = getattr(mr, "device_indices", None)
        dev = 0 if _di is None else int(_di.numel() if hasattr(_di, "numel") else len(_di))
        return max(0, min(int(told), dev + int(getattr(mr, "host_hit_length", 0) or 0)) - dev)
    except Exception:  # noqa: BLE001 - a probe never breaks the ack
        return None


def _ack_evictable(tree) -> int:
    """ACK-ROOM-PAYABLE (ARENAFULL-1015, NF h16n 07:30:35 PP1 weg2-16-53): the evictable part of the ack's room is
    what ONE peel can pay -- the PR walk (``pp_room_vote.estimate_payable``: locked chains, un-backable leaves with
    host-locked children and the free arena room for backups all counted as the peel meets them), capped by the
    tree's reported size. The reported size alone said 255040 where the peel paid 48192, and the ack let PP0 admit a
    load-back this stage could not hold. Off (``SGLANG_WEG2_ACK_ROOM_PAYABLE=0``) or unreadable walk: the reported
    evictable size, as before. Read-only."""
    reported = int(tree.evictable_size())
    try:
        from sglang.srt.environ import envs

        if not envs.SGLANG_WEG2_ACK_ROOM_PAYABLE.get():
            return reported
        from sglang.srt.weg2 import pp_room_vote as _pr
        from sglang.srt.weg2 import pp_slot_fidelity as _sf

        cc = getattr(tree, "cache_controller", None)
        host_free = _pr.backup_room_tokens(getattr(cc, "mem_pool_host", None) if cc is not None else None)
        walked = _pr.estimate_payable(tree, local_floor=bool(getattr(tree, _sf.FLOOR_LOCAL_PP_ATTR, False)),
                                      host_free=host_free)
        return max(0, min(reported, int(walked)))
    except Exception:  # noqa: BLE001 - no walk: the reported size
        return reported


def _room_defer_map(scheduler) -> Dict[str, Tuple[int, int, int]]:
    d = getattr(scheduler, "_weg2_fb_room_defer", None)
    if d is None:
        d = scheduler._weg2_fb_room_defer = {}
    return d


def room_deferred(scheduler, rid: str) -> Optional[Tuple[int, int, int]]:
    """T2 ROOM-DEFER: ``(rows, room, locked)`` when the LAST ``_room_own`` of ``rid`` found the room short only for as
    long as other requests hold rows (``room + locked >= rows``); else None. Read once per ack formation."""
    return _room_defer_map(scheduler).pop(str(rid), None)


def _locked_rows(scheduler, available: int, evictable: int) -> int:
    """Rows other requests hold on this stage's device pool: ``max_total_num_tokens - (available + evictable)`` --
    the very ``num_used`` the scheduler's ``full token usage`` line prints (h16p: 0.38 = 99.6k rows at the NO-ROOM
    ack, 0.06 one batch later). They return to the pool when those requests finish. 0 = not readable."""
    total = int(getattr(scheduler, "max_total_num_tokens", 0) or 0)
    return max(0, total - int(available) - int(evictable)) if total > 0 else 0


def _room_own(scheduler, req, rid: str, own: int, mark: bool = True) -> int:
    """ACK-ROOM (dual1k dkr27bnvfp4dual1kbar1fs10010950, PP1 09:55:20Z, rid
    weg2-0-10, a fork twin): the follower's read reproduced told=16383 and its
    tree could resume there (host KV + anchor), but the head was HOST-only on
    this rank (PP0/PP2 held it on the device) and the load-back found no room:
    ``SF LOADBACK-ROOM PP-RESIDUAL kv_tokens=12288 avail=1717 evictable=0`` --
    a concurrent 61440-token prefill held the pool. PP0 had admitted at told,
    so #968 followed at once. The ack names 0 when this rank cannot hold the
    told's load-back even with every evictable row freed: PP0 answers told=0
    for EVERY rank (a rank-agreed re-prefill, chunked from 0) instead of a
    group death. Read-only (nothing evicted here); no probe = no verdict.

    T2 ROOM-DEFER (NF h16p 226aad7105, PP1/PP2 weg2-0-3 12:46:20 ``loadback_rows=184768 room=162176``, weg2-0-10
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
    transient = bool(room_defer_on() and room + locked >= int(rows))
    if mark and transient:
        _room_defer_map(scheduler)[str(rid)] = (int(rows), int(room), int(locked))
        seen = getattr(scheduler, "_weg2_fb_room_defer_seen", None)
        if seen is None:
            seen = scheduler._weg2_fb_room_defer_seen = set()
        if str(rid) in seen:
            return 0  # named on the first formation; a re-check every pass would only repeat it
        seen.add(str(rid))
        n = _bump(scheduler, "_pf_ack_room_defer_n")
        if _say(n):
            logger.warning(
                "PF TOLD-ACK ROOM-DEFER rank pp=%s rid=%s told=%d loadback_rows=%d room=%d locked=%d (n=%d): the "
                "load-back does not fit while other requests hold %d rows but fits once they are released -- the "
                "ack is held back (PP0 waits for it, it does not answer told=0) and goes out when the room is there",
                getattr(scheduler.ps, "pp_rank", "?"), str(rid), own, int(rows), room, locked, n, locked,
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
        from sglang.srt.managers.weg2_store_told import _probe_key
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

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
        if getattr(st.registered[rid], "_weg2_early_told", None) is not None:
            # DP-NACHLAUF (N5p): an early read -- settle it against told first
            from sglang.srt.managers import weg2_store_told as _st

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
            # the room is short only while other requests hold rows: no ack yet -- PP0 reads the missing ack as lag
            # and waits (WAIT-PROGRESS); the real reach goes out when the room is there. The read stays registered.
            _now = _perf()
            st.deferred[rid] = (_held[0] if _held is not None else _now, _now + ROOM_DEFER_RECHECK_S)
            continue
        if _held is not None:
            n_r = _bump(scheduler, "_pf_ack_room_deferred_released_n")
            if _say(n_r):
                logger.info(
                    "PF TOLD-ACK ROOM-DEFER RELEASED rank pp=%s rid=%s told=%d own=%d after=%.2fs (n=%d): the "
                    "room is there, the ack goes out with the real reach", getattr(scheduler.ps, "pp_rank", "?"),
                    rid, told, int(own), _perf() - _held[0], n_r,
                )
        st.deferred.pop(rid, None)
        (getattr(scheduler, "_weg2_fb_room_defer_seen", None) or set()).discard(rid)
        st.registered.pop(rid, None)
        st.expect.pop(rid, None)
        if anchor_acks_on():
            st.outbox.append((rid, own, tuple(ack_anchors(scheduler, req, rid, own, told))))
        else:
            st.outbox.append((rid, own))
    ch = _channel(scheduler)
    if not st.outbox:
        ch.pump()
        return
    ack = Weg2ToldReadAck(rank=int(scheduler.ps.pp_rank), seq=st.seq, reads=list(st.outbox))
    if ch.send_nowait(ack):
        st.outbox = []
        st.seq += 1
        n = _bump(scheduler, "_pf_ack_sent_n")
        if _say(n):
            logger.info(
                "PF TOLD-ACK SENT rank pp=%s seq=%d reads=%s (n=%d)",
                scheduler.ps.pp_rank, ack.seq, list(ack.reads), n,
            )
