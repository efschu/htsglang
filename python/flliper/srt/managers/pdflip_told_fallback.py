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

SWITCH ``FLLIPER_PDFLIP_TOLD_GROUP_FALLBACK`` (default off), read once on PP0,
effective only with ``FLLIPER_PDFLIP_TOLD_PACED`` on: only there does PP0 hold
admission back until an Admit.
"""

from __future__ import annotations

import logging
import os
import pickle
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
#: item 180: where a follower's re-ask of PP0 went (follower_reask)
ROUTE_PF = "pf-ack"
ROUTE_NONE = "none"

_LOG_FIRST = 8
_LOG_EVERY = 256


def env_on() -> bool:
    # RG 26.09.: unset = the registry row (pdflip/form.py PREFIX_SWITCHES; PF is
    # off on every row -- unproven on metal).
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
class PdFlipToldReadAck:
    """Follower -> PP0 on ``PDFLIP_TOLD_ACK_TAG``: the reads this rank has
    terminated since its last ack, as ``(rid, own_prefix)``."""

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

    def __init__(self, group: Any, pp_rank: int, world_ranks: List[int]):
        self.group = group
        self.pp_rank = int(pp_rank)
        self.world_ranks = list(world_ranks)
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
                "PF TOLD-ACK %s failed rank pp=%d (%d so far): %r -- PP0 sends "
                "told=0 at its Frist for every rid an ack could not carry",
                what, self.pp_rank, self.errors, exc,
            )

    def send_nowait(self, ack: PdFlipToldReadAck) -> bool:
        """Post ``ack`` unless the previous one is still in flight. True =
        the ack left the outbox (posted, or dropped on a dead stream)."""
        if self.dead:
            return True
        if not self.pump():
            return False
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
                w = dist.isend(t, dst, group=self.group, tag=PDFLIP_TOLD_ACK_TAG)
                works.append((ParkedWait(w, f"pdflip/told-ack/{label}"), t))
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
                tag=PDFLIP_TOLD_ACK_TAG,
                site=f"pdflip/told-ack[pp{pp_rank}]",
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
    ch = getattr(scheduler, "_pdflip_fb_channel", None)
    if ch is None:
        ch = scheduler._pdflip_fb_channel = GlooAckChannel.for_scheduler(scheduler)
    return ch


# ---------------------------------------------------------------------------
# PP0
# ---------------------------------------------------------------------------


@dataclass
class _Open:
    told: int
    deadline: float
    acks: Dict[int, int] = field(default_factory=dict)


def _pp0_open_map(scheduler) -> Dict[str, _Open]:
    d = getattr(scheduler, "_pdflip_fb_open", None)
    if d is None:
        d = scheduler._pdflip_fb_open = {}
    return d


def pp0_open(scheduler, rid: str, told: int, published_at: float, own_read_s: float, window_s: float) -> float:
    """PP0 put a paced read-ahead with ``ack`` on the wire for ``rid``."""
    dl = deadline(published_at, own_read_s, window_s)
    _pp0_open_map(scheduler)[str(rid)] = _Open(told=int(told), deadline=dl)
    return dl - float(published_at)


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


def pp0_harvest(scheduler) -> int:
    """Take every ack that has landed off the standing receives (no wait)."""
    open_map = _pp0_open_map(scheduler)
    n = 0
    for ack in _channel(scheduler).harvest():
        if not isinstance(ack, PdFlipToldReadAck):
            logger.warning("PF TOLD-ACK object %s on the ack stream dropped", type(ack).__name__)
            continue
        for rid, own in ack.reads:
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
            n += 1
    if n:
        k = _bump(scheduler, "_pf_ack_harvest_n")
        if _say(k):
            logger.info("PF TOLD-ACK HARVEST acks=%d open=%d (n=%d)", n, len(open_map), k)
    return n


def pp0_decide(scheduler, rid: str, now: float) -> Optional[Tuple[int, str]]:
    """``None`` = keep waiting; else ``(told to admit, reason)``."""
    o = _pp0_open_map(scheduler).get(str(rid))
    if o is None:
        return 0, REASON_FRIST  # untracked paced rid: the always-uniform answer
    followers = range(1, int(scheduler.ps.pp_size))
    if any(own != o.told for own in o.acks.values()):
        return 0, REASON_MISMATCH
    if all(r in o.acks for r in followers):
        return o.told, REASON_ACKS
    if now >= o.deadline:
        return 0, REASON_FRIST
    return None


def pp0_note_verdict(scheduler, rid: str, told: int, told_final: int, reason: str, now: float, published_at: float) -> None:
    o = _pp0_open_map(scheduler).pop(str(rid), None)
    acks = dict(o.acks) if o is not None else {}
    if int(told_final) > 0:
        # item 220: PP0 admitted at told; until it seats the rid, a follower's
        # late re-ask can still switch it to told=0 (pp0_retract_due)
        _admitted_map(scheduler)[str(rid)] = int(told_final)
    if reason == REASON_ACKS:
        n = _bump(scheduler, "_pf_admit_acks_n")
        if _say(n):
            logger.info(
                "PF TOLD-ACKED rid=%s told=%d after=%.2fs acks=%s (n=%d): every "
                "follower's read reproduced told -- admitted without the window",
                rid, told, now - published_at, acks, n,
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
    outbox: List[Tuple[str, int]] = field(default_factory=list)
    seq: int = 0


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
    return int(res)


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


def _room_own(scheduler, req, rid: str, own: int) -> int:
    """ACK-ROOM (dual1k dkr27bnvfp4dual1kbar1fs10010950, PP1 09:55:20Z, rid
    pdflip-0-10, a fork twin): the follower's read reproduced told=16383 and its
    tree could resume there (host KV + anchor), but the head was HOST-only on
    this rank (PP0/PP2 held it on the device) and the load-back found no room:
    ``SF LOADBACK-ROOM PP-RESIDUAL kv_tokens=12288 avail=1717 evictable=0`` --
    a concurrent 61440-token prefill held the pool. PP0 had admitted at told,
    so #968 followed at once. The ack names 0 when this rank cannot hold the
    told's load-back even with every evictable row freed: PP0 answers told=0
    for EVERY rank (a rank-agreed re-prefill, chunked from 0) instead of a
    group death. Read-only (nothing evicted here); no probe = no verdict."""
    rows = _loadback_rows(scheduler, req, own)
    if not rows:
        return own
    try:
        tree = scheduler.tree_cache
        alloc = getattr(tree, "token_to_kv_pool_allocator", None)
        room = int(alloc.available_size()) + int(tree.evictable_size())
    except Exception:  # noqa: BLE001 - no allocator readable: no verdict
        return own
    if room >= int(rows):
        return own
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
        if getattr(st.registered[rid], "_pdflip_early_told", None) is not None:
            # DP-NACHLAUF (N5p): an early read -- settle it against told first
            from flliper.srt.managers import pdflip_store_told as _st

            if not _st.follower_early_settle_now(scheduler, st.registered[rid], rid, told):
                continue  # short: the told-limited read acks when it ends
        req = st.registered.pop(rid)
        st.expect.pop(rid, None)
        st.outbox.append((rid, own_prefix(scheduler, req, rid, told)))
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
