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

SWITCH ``SGLANG_WEG2_TOLD_GROUP_FALLBACK`` (default off), read once on PP0,
effective only with ``SGLANG_WEG2_TOLD_PACED`` on: only there does PP0 hold
admission back until an Admit.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
WEG2_TOLD_ACK_TAG = 1416
#: wire markers: INSTANCE attributes, present only on the armed path.
WIRE_ACK = "ack"
WIRE_FALLBACK = "fallback"
#: at most this many acks taken off one follower's stream per pass.
HARVEST_MAX_PER_SRC = 8

REASON_ACKS = "acks"
REASON_MISMATCH = "mismatch"
REASON_FRIST = "frist"

_LOG_FIRST = 8
_LOG_EVERY = 256


def env_on() -> bool:
    return os.environ.get(ENV_FALLBACK, "0").strip().lower() in ("1", "true", "yes", "on")


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
class Weg2ToldReadAck:
    """Follower -> PP0 on ``WEG2_TOLD_ACK_TAG``: the reads this rank has
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
                w = dist.isend(t, dst, group=self.group, tag=WEG2_TOLD_ACK_TAG)
                works.append((ParkedWait(w, f"weg2/told-ack/{label}"), t))
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
                tag=WEG2_TOLD_ACK_TAG,
                site=f"weg2/told-ack[pp{pp_rank}]",
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


def _pp0_open_map(scheduler) -> Dict[str, _Open]:
    d = getattr(scheduler, "_weg2_fb_open", None)
    if d is None:
        d = scheduler._weg2_fb_open = {}
    return d


def pp0_open(scheduler, rid: str, told: int, published_at: float, own_read_s: float, window_s: float) -> float:
    """PP0 put a paced read-ahead with ``ack`` on the wire for ``rid``."""
    dl = deadline(published_at, own_read_s, window_s)
    _pp0_open_map(scheduler)[str(rid)] = _Open(told=int(told), deadline=dl)
    return dl - float(published_at)


def pp0_forget(scheduler, rid: str) -> None:
    _pp0_open_map(scheduler).pop(str(rid), None)


def pp0_harvest(scheduler) -> int:
    """Take every ack that has landed off the standing receives (no wait)."""
    open_map = _pp0_open_map(scheduler)
    n = 0
    for ack in _channel(scheduler).harvest():
        if not isinstance(ack, Weg2ToldReadAck):
            logger.warning("PF TOLD-ACK object %s on the ack stream dropped", type(ack).__name__)
            continue
        for rid, own in ack.reads:
            o = open_map.get(str(rid))
            if o is None:
                continue  # decided (or dropped) already: late ack
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
    if reason == REASON_ACKS:
        n = _bump(scheduler, "_pf_admit_acks_n")
        if _say(n):
            logger.info(
                "PF TOLD-ACKED rid=%s told=%d after=%.2fs acks=%s (n=%d): every "
                "follower's read reproduced told -- admitted without the window",
                rid[:8], told, now - published_at, acks, n,
            )
        return
    n = _bump(scheduler, "_pf_fallback_n")
    if n <= 32 or n % _LOG_EVERY == 0:
        logger.warning(
            "PF TOLD-FALLBACK rid=%s told=%d -> 0 reason=%s after=%.2fs acks=%s "
            "followers=%d (n=%d): PP0 switches the request to told=0 for EVERY "
            "rank; P recomputes the prefix instead of the group dying in "
            "STORE-TOLD WAIT EXCEEDED / MISMATCH",
            rid[:8], told, reason, now - published_at, acks,
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
            logger.warning("PF TOLD-FALLBACK release_aborted_request(%s) raised: %r", str(rid)[:8], exc)
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
    from sglang.srt.weg2 import p_twin_defer as _twin

    rid = str(rid)
    follower_forget(scheduler, rid)
    release_own_read(scheduler, rid)
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
            "admits at 0 like every rank", getattr(scheduler.ps, "pp_rank", "?"), rid[:8], n,
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
        return int(told)
    own = int(_st._completed_prefix(scheduler.tree_cache, rid))
    if _is_follower_twin(scheduler, rid):
        own += _twin.registered_head(req)
    return own


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
        req = st.registered.pop(rid)
        st.expect.pop(rid, None)
        st.outbox.append((rid, own_prefix(scheduler, req, rid, told)))
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
                scheduler.ps.pp_rank, ack.seq, [(r[:8], o) for r, o in ack.reads], n,
            )
