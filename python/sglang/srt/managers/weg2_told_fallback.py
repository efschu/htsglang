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
import time
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
    # RG 26.09.: unset = the registry row (weg2/form.py PREFIX_SWITCHES; PF is
    # off on every row -- unproven on metal).
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
    _room_hold_end(scheduler, rid, "verdict")  # Q-693: PP0 decided before the held ack
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


def own_prefix(scheduler, req, rid: str, told: int) -> Optional[int]:
    """What ``weg2_store_told.admission`` will compare with told -- read,
    never consumed. None = Q-693 ROOM-HOLD (dual P only): no ack yet."""
    from sglang.srt.managers import weg2_store_told as _st
    from sglang.srt.weg2 import p_twin_defer as _twin

    satisfied = getattr(scheduler, "_weg2_store_told_satisfied", None) or {}
    if rid in satisfied:
        return int(told)
    own = int(_st._completed_prefix(scheduler.tree_cache, rid))
    if _is_follower_twin(scheduler, rid):
        own += _twin.registered_head(req)
    return _resumable_own(scheduler, req, rid, own)


def _resumable_own(scheduler, req, rid: str, own: int) -> Optional[int]:
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

        res = _tf.pp0_admissible(scheduler, req, int(own))
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
            getattr(scheduler.ps, "pp_rank", "?"), str(rid)[:12], own, int(res), n,
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
        from sglang.srt.managers.weg2_store_told import _probe_key
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

        key, _bigram = _probe_key(scheduler, req, ids, int(told))
        mr = match(MatchPrefixParams(key=key))
        _di = getattr(mr, "device_indices", None)
        dev = 0 if _di is None else int(_di.numel() if hasattr(_di, "numel") else len(_di))
        return max(0, min(int(told), dev + int(getattr(mr, "host_hit_length", 0) or 0)) - dev)
    except Exception:  # noqa: BLE001 - a probe never breaks the ack
        return None


def _room_own(scheduler, req, rid: str, own: int) -> Optional[int]:
    """ACK-ROOM (dual1k dkr27bnvfp4dual1kbar1fs10010950, PP1 09:55:20Z, rid
    weg2-0-10, a fork twin): the follower's read reproduced told=16383 and its
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
        _room_hold_end(scheduler, rid, "room")
        return own
    if _room_hold(scheduler, req, rid, own, int(rows), room):
        return None  # Q-693 (dual only): ack held, re-read at the next pump
    n = _bump(scheduler, "_pf_ack_no_room_n")
    if n <= 32 or n % _LOG_EVERY == 0:
        logger.warning(
            "PF TOLD-ACK NO-ROOM rank pp=%s rid=%s told=%d loadback_rows=%d room=%d (n=%d): "
            "this rank holds the told span on its HOST only and cannot load it back even "
            "with every evictable row freed -- the ack says 0 and PP0 answers told=0 for "
            "every rank instead of an SF LOADBACK-ROOM residual and #968 after PP0 admitted%s",
            getattr(scheduler.ps, "pp_rank", "?"), str(rid)[:12], own, int(rows), room, n,
            _noroom_detail(scheduler, rid),
        )
    return 0


# Q-693 ACK-ROOM HOLD (27B NVFP4 dual fs10031727 bc2bd121c0, 37 NO-ROOM acks;
# PP1 17:42:58 weg2-0-183 told=68096 room=3735): the follower's pool was full of
# its PREDECESSOR weg2-0-182, whose last chunk (extend=147) this rank had
# admitted in the same second and which released its rows one pass later (full
# token usage 0.98 -> 0.51). The ack said 0 on that transient and PP0 re-
# prefilled 68664 tokens (~15 s). With a predecessor in flight whose rows cover
# the shortfall the ack is HELD (the read stays registered) and re-read at the
# next pump; it says 0 only on a stable shortage. PP0's Frist bounds the hold:
# an ack that never comes is PP0's told=0 at the Frist, the verdict of today.
# The dual1k check stays: a CONCURRENT prefill (chunks still to admit on a P
# that runs more than one request) is not a predecessor and is not counted.
ROOM_HOLD_MARK = "PF TOLD-ACK ROOM-HOLD"
_ROOM_HOLD_ATTR = "_q693_room_hold"


def _inflight_reqs(scheduler, req) -> List[Any]:
    """Every request this rank holds rows for, other than ``req``, once."""
    out: List[Any] = []
    seen = {id(req)}

    def _take(r) -> None:
        if r is not None and id(r) not in seen:
            seen.add(id(r))
            out.append(r)

    for ring in ("mbs", "running_mbs"):
        for b in getattr(scheduler, ring, None) or ():
            for r in getattr(b, "reqs", None) or ():
                _take(r)
    for r in getattr(getattr(scheduler, "running_batch", None), "reqs", None) or ():
        _take(r)
    _take(getattr(scheduler, "chunked_req", None))
    for r in getattr(scheduler, "_pp_chunked_req_before_by_slot", None) or ():
        _take(r)
    return out


def _serial(scheduler) -> bool:
    mrr = getattr(scheduler, "max_running_requests", None)
    if mrr is None:
        mrr = getattr(getattr(scheduler, "server_args", None), "max_running_requests", None)
    try:
        return mrr is not None and int(mrr) <= 1
    except (TypeError, ValueError):
        return False


def inflight_release_rows(scheduler, req) -> int:
    """Rows the PREDECESSORS in flight on this rank hold now and hand back to the
    tree (evictable) when they finish, before ``req`` can be admitted: requests
    whose last chunk this rank has admitted, and -- on a serial P (max running
    requests 1), where nothing runs beside ``req`` -- every request in flight. A
    predecessor with chunks to go allocates them first and frees all of it, so
    room + its held rows is the room after it (rows held = len(fill_ids))."""
    serial = _serial(scheduler)
    rows = 0
    for r in _inflight_reqs(scheduler, req):
        fill = len(getattr(r, "fill_ids", None) or ())
        origin = len(getattr(r, "origin_input_ids", None) or ())
        if fill <= 0:
            continue
        if serial or (origin > 0 and fill >= origin):
            rows += fill
    return rows


def _int_or_none(fn) -> Optional[int]:
    try:
        v = fn()
        if isinstance(v, (tuple, list)):
            v = v[0]
        return None if v is None else int(v)
    except Exception:  # noqa: BLE001 - an observation never breaks the ack
        return None


def pool_snapshot(scheduler) -> Dict[str, Any]:
    """Q-920 (observability, dual P only): what this rank's pool holds right now.
    ``available`` (allocator free list), ``evictable``, ``protected`` (rows locked
    by requests and publish pins), ``publish`` (write-through / backup nodes in
    flight), ``usage`` = 1 - (available + evictable) / total, and the last
    ``MAPPED-BY-GRANT``: tokens it added and its age. None = not readable."""
    tree = getattr(scheduler, "tree_cache", None)
    alloc = getattr(tree, "token_to_kv_pool_allocator", None)
    avail = _int_or_none(getattr(alloc, "available_size", lambda: None))
    evictable = _int_or_none(getattr(tree, "evictable_size", lambda: None))
    protected = _int_or_none(getattr(tree, "protected_size", lambda: None))
    publish = None
    try:
        publish = len(getattr(tree, "ongoing_write_through", None) or ()) + len(
            getattr(tree, "ongoing_backup", None) or ())
    except Exception:  # noqa: BLE001
        publish = None
    total = getattr(scheduler, "max_total_num_tokens", None)
    if total is None:
        total = getattr(alloc, "size", None)
    usage = None
    try:
        if avail is not None and evictable is not None and total:
            usage = round(1.0 - (avail + evictable) / float(total), 3)
    except Exception:  # noqa: BLE001
        usage = None
    grant_new, grant_age = None, None
    try:
        from sglang.srt.weg2 import dual_p_kv_stage as _dpk

        lg = getattr(_dpk._actor(scheduler), "last_grant", None)
        if lg:
            grant_age, grant_new = round(time.monotonic() - float(lg[0]), 2), int(lg[1])
    except Exception:  # noqa: BLE001
        pass
    return {"available": avail, "evictable": evictable, "protected": protected, "publish": publish,
            "usage": usage, "grant_new": grant_new, "grant_age": grant_age}


def _noroom_detail(scheduler, rid: str) -> str:
    """Q-920: the dual NO-ROOM line's extra terms; "" in the flip form (its line is unchanged)."""
    from sglang.srt.weg2 import dual_p_kv_stage as _dpk

    if not _dpk.armed():
        return ""
    p = pool_snapshot(scheduler)
    return (" | Q-920 full_rid=%s available=%s evictable=%s protected=%s pending_publish=%s usage=%s "
            "grant_new=%s grant_age_s=%s" % (str(rid), p["available"], p["evictable"], p["protected"],
                                              p["publish"], p["usage"], p["grant_new"], p["grant_age"]))


NOHOLD_MARK = "PF TOLD-ACK ROOM-NOHOLD"


def _room_nohold(scheduler, why: str, rid: str, rows: int, room: int, pending: int) -> None:
    """Q-920 (dual P only): WHY a NO-ROOM shortfall was not held -- one capped line per refusal
    (y8z had 0 ROOM-HOLD lines and no way to tell whether the check ran)."""
    n = _bump(scheduler, "_q920_nohold_n")
    if not (n <= 32 or n % _LOG_EVERY == 0):
        return
    p = pool_snapshot(scheduler)
    logger.warning(
        "%s why=%s rid=%s room=%d rows=%d pending=%d usage=%s protected=%s available=%s evictable=%s "
        "pending_publish=%s rank pp=%s (n=%d)",
        NOHOLD_MARK, why, str(rid), int(room), int(rows), int(pending), p["usage"], p["protected"],
        p["available"], p["evictable"], p["publish"], getattr(getattr(scheduler, "ps", None), "pp_rank", "?"), n,
    )


def _room_hold(scheduler, req, rid: str, own: int, rows: int, room: int) -> bool:
    """Q-693 (dual P only): True = hold this ack, the shortfall is a predecessor
    in flight that releases it."""
    from sglang.srt.weg2 import dual_p_kv_stage as _dpk

    if not _dpk.armed():
        return False
    if not any(r is req for r in (getattr(scheduler, "waiting_queue", None) or ())):
        _room_nohold(scheduler, "left_queue", rid, rows, room, 0)
        _room_hold_end(scheduler, rid, "left_queue")
        return False  # aborted / gone here: nothing to hold for
    pending = inflight_release_rows(scheduler, req)
    if pending <= 0 or room + pending < rows:
        _room_nohold(scheduler, "no_predecessor" if pending <= 0 else "predecessor_too_small",
                     rid, rows, room, pending)
        _room_hold_end(scheduler, rid, "stable")
        return False
    holds = getattr(scheduler, _ROOM_HOLD_ATTR, None)
    if holds is None:
        holds = {}
        setattr(scheduler, _ROOM_HOLD_ATTR, holds)
    if rid not in holds:
        holds[rid] = time.monotonic()
        n = _bump(scheduler, "_q693_room_hold_n")
        if n <= 32 or n % _LOG_EVERY == 0:
            logger.warning(
                "%s rank pp=%s rid=%s told=%d loadback_rows=%d room=%d predecessor_rows=%d (n=%d): "
                "the pool is short only by a predecessor still in flight on this rank -- the ack "
                "is held and re-read every pass instead of 0 (Q-693; PP0's Frist bounds it)",
                ROOM_HOLD_MARK, getattr(scheduler.ps, "pp_rank", "?"), str(rid)[:12], own, rows,
                room, pending, n,
            )
    return True


def _room_hold_end(scheduler, rid: str, how: str) -> None:
    holds = getattr(scheduler, _ROOM_HOLD_ATTR, None)
    if not holds or rid not in holds:
        if how == "stable":
            # Q-920: a 'stable' verdict without a hold used to be silent (y8z: 0 ROOM-HOLD lines)
            n = _bump(scheduler, "_q920_stable_nohold_n")
            if n <= 32 or n % _LOG_EVERY == 0:
                logger.warning(
                    "%s END rank pp=%s rid=%s how=stable held_s=none (no hold was taken: the shortage "
                    "was stable at the first look, ack 0) (n=%d)",
                    ROOM_HOLD_MARK, getattr(getattr(scheduler, "ps", None), "pp_rank", "?"), str(rid), n,
                )
        return
    t0 = holds.pop(rid)
    logger.warning(
        "%s END rank pp=%s rid=%s how=%s held_s=%.2f (room = the predecessor released; stable = "
        "the shortage outlived it, ack 0; verdict = PP0 decided first, its Frist)",
        ROOM_HOLD_MARK, getattr(getattr(scheduler, "ps", None), "pp_rank", "?"), str(rid)[:12], how,
        time.monotonic() - t0,
    )


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
        own = own_prefix(scheduler, st.registered[rid], rid, told)
        if own is None:
            continue  # Q-693 ROOM-HOLD (dual P only): re-read at the next pump
        st.registered.pop(rid)
        st.expect.pop(rid, None)
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
                scheduler.ps.pp_rank, ack.seq, [(r[:8], o) for r, o in ack.reads], n,
            )
