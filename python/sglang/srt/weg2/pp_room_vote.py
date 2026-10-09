"""PR: PP0 sizes a pass on the room EVERY stage can pay, not on its own alone.

NF int19 (617c6b6541), P log boot_weg2_dkrnfint4h6ablxcbar1dauer10081332_617c6b6541_
1008_133232.P.log 13:37:49Z: PP0 planned a 16384-token chunk on its own pool; PP2
(``#969N ADMIT ... extend=16384``, no veto: ``#1460 FOLLOWER-GATE gate=None``) had
``full_available_size=15232`` and a peel that paid nothing of
``reported_evictable=132224`` (``EVICT-FRONTIER-CENSUS on_frontier=3456
behind_device_child=128768``, ``EXTEND-RELIEF evicted=0``) -> ``Prefill out of
memory`` -> RANK-DEATH. Same death int18 12:57:14Z PP2 and int16 10:37:41Z PP0. The
stage trees diverge rank-locally (UD drops, arena refusals; at 13:37:49 ``full
token usage`` PP0 0.56 / PP1 0.50 / PP2 0.44), so PP0's own reading says nothing
about PP2's.

THE TRANSPORT. Each follower sends PP0 a :class:`Weg2PpRoomFact` point to point on
its own tag (``WEG2_ROOM_TAG``) through the told-fallback's non-blocking channel
(``weg2_told_fallback.GlooAckChannel``: one send in flight, parked on its own
thread; PP0 polls one standing ``ObjectRecvFrame`` per follower, never joins).
The ring's home trip carries nothing (``pp_output_payload_with_return_trip`` has
no caller) and the #1268 home hop belongs to the idle vote. Nothing blocks a
pass on either end; there is no collective.

THE FACT. ``room = available_size + payable`` measured at the follower's pass top,
with ``executed`` = its ``forward_ct`` (the batches whose rows it has already
allocated). ``payable`` is what ONE peel can pay without knowing the wall: a
post-order walk (:func:`estimate_payable`) that pays a device node only when every
device child is paid and the node itself can leave -- backed (demote), backed up
into the host room still free, or dropped on the local-PP floor (UD/UD-H: no
write-through in flight, host-only children clearable); a locked node or an
unpayable child stops its whole chain. It is capped by PW's measured wall
(``payable_evictable_or``). The walk is O(nodes) (86 nodes on PP2 at 13:37:49) and
its time rides the fact (``sim_us``).

PP0's CAP. ``cap = min over followers (room_k - rows PP0 launched after the
follower's executed count)`` -- the batches still in flight toward k, read off
``forward_iter``. A fact older than ``FACT_MAX_AGE_PASSES`` PP0 passes is dropped
(a silent follower costs nothing but the cap). The cap is published on PP0's tree
(``mem_cache.common.PP_ROOM_CAP_ATTR``): ``fundable_extend_tokens`` /
``published_fundable_floor`` take the MIN, so a new request's budget and the
chunked continuation (#679 park / narrowed chunk) both obey it, and the P intake
verdict (``weg2/p_intake.pool_terms``) reads the same number -- a P that runs
empty with a head no stage can hold reaches the named INTAKE-STALL (503, the
front re-routes and flips) instead of waiting for room no pass will free.

Scope: only followers whose tree is on the local-PP floor send (pp > 1, tp group
of one: NF P); D and every TP group never send, and PP0 without a fact publishes
nothing -- byte-identical."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import msgspec

from sglang.srt.managers.weg2_idle_vote import FIXED_P2P_TAG_BASE

logger = logging.getLogger(__name__)

#: own p2p stream (1268 = idle vote, 1416 = told-ack)
WEG2_ROOM_TAG = FIXED_P2P_TAG_BASE + 1513
#: a fact older than this many PP0 passes is not read
FACT_MAX_AGE_PASSES = 8
#: a follower re-sends an unchanged fact at most this often (seconds)
RESEND_S = 0.5


class Weg2PpRoomFact(msgspec.Struct, frozen=True):
    rank: int
    executed: int
    room: int
    available: int
    payable: int
    reported: int
    sim_us: float = 0.0


class Weg2PpRoomFactLane(Weg2PpRoomFact, frozen=True):
    """PRIORITY LANES 1008 (L3): the same fact plus the lane floor / epoch the sending stage APPLIES.  Only a stage that
    holds lane state sends this subclass; a boot without lanes sends the plain fact, wire unchanged."""

    lane_floor: int = -1
    lane_epoch: int = -1


class CapVerdict(msgspec.Struct, frozen=True):
    cap: int
    rank: int
    room: int
    inflight: int


# ---------------------------------------------------------------------------
# follower: what one peel can pay
# ---------------------------------------------------------------------------


def _locked(node) -> bool:
    return any(cd.lock_ref > 0 for cd in node.component_data)


def _host_clearable(node, state: Dict[int, Tuple[int, bool]]) -> bool:
    """A host-only node UD-H may remove: backed, no device/host lock, every
    child clearable (``_ud_clear_host_children``'s all-or-nothing test)."""
    if not node.evicted or not node.backuped:
        return False
    if any(cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in node.component_data):
        return False
    return all(state.get(id(c), (0, False))[1] for c in node.children.values())


def estimate_payable(tree, *, local_floor: bool, host_free: int) -> int:
    """Device tokens ONE peel can pay (see the module docstring)."""
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        BASE_COMPONENT_TYPE as ct,
    )

    root = tree.root_node
    cc = getattr(tree, "cache_controller", None)
    write_back = cc is not None and getattr(cc, "write_policy", None) == "write_back"
    ongoing = getattr(tree, "ongoing_write_through", None) or {}
    order, stack = [], [root]
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(n.children.values())
    state: Dict[int, Tuple[int, bool]] = {}
    budget = max(0, int(host_free))
    for n in reversed(order):  # children before their parent
        if n is root:
            continue
        value = n.component_data[ct].value
        if value is None:
            state[id(n)] = (0, _host_clearable(n, state))
            continue
        dev = [c for c in n.children.values() if c.component_data[ct].value is not None]
        host = [c for c in n.children.values() if c.component_data[ct].value is None]
        paid = sum(state[id(c)][0] for c in dev)
        if _locked(n) or not all(state[id(c)][1] for c in dev):
            state[id(n)] = (paid, False)
            continue
        tokens = len(value)
        if n.backuped:
            ok = True
        elif write_back and budget >= tokens:
            budget -= tokens
            ok = True
        elif write_back:
            ok = (local_floor and getattr(n, "id", None) not in ongoing
                  and all(state[id(c)][1] for c in host))
        else:
            ok = not host  # write_through deletes; #841 refuses with children
        state[id(n)] = (paid + (tokens if ok else 0), ok)
    return sum(state[id(c)][0] for c in root.children.values() if id(c) in state)


def follower_fact(*, tree, allocator, rank: int, executed: int, local_floor: bool,
                  lane_floor: int = -1, lane_epoch: int = -1) -> Weg2PpRoomFact:
    from sglang.srt.mem_cache.common import payable_evictable_or

    available = int(allocator.available_size())
    reported = int(tree.evictable_size())
    walled = int(payable_evictable_or(tree, tree.evictable_size))
    host_free = 0
    cc = getattr(tree, "cache_controller", None)
    pool = getattr(cc, "mem_pool_host", None) if cc is not None else None
    if pool is not None:
        try:
            host_free = int(pool.available_size())
        except Exception:  # noqa: BLE001 - no reading: no host room assumed
            host_free = 0
    t0 = time.perf_counter()
    walked = estimate_payable(tree, local_floor=local_floor, host_free=host_free)
    sim_us = (time.perf_counter() - t0) * 1e6
    payable = max(0, min(walked, walled))
    if int(lane_epoch) >= 0:
        return Weg2PpRoomFactLane(rank=int(rank), executed=int(executed), room=available + payable,
                                  available=available, payable=payable, reported=reported, sim_us=sim_us,
                                  lane_floor=int(lane_floor), lane_epoch=int(lane_epoch))
    return Weg2PpRoomFact(rank=int(rank), executed=int(executed), room=available + payable,
                          available=available, payable=payable, reported=reported, sim_us=sim_us)


class FollowerSender:
    """Sends a fact when it changed, else at most every ``RESEND_S``."""

    def __init__(self) -> None:
        self.last: Optional[Tuple[int, int]] = None
        self.last_t = 0.0
        self.pending: Optional[Weg2PpRoomFact] = None

    def offer(self, fact: Weg2PpRoomFact, now: float) -> bool:
        key = (fact.room, fact.executed)
        if key == self.last and now - self.last_t < RESEND_S:
            return False
        self.pending = fact
        return True

    def flush(self, channel, now: float) -> bool:
        """Post the pending fact unless one is still in flight (never blocks)."""
        if self.pending is None:
            return False
        if not channel.send_nowait(self.pending):
            return False
        self.last = (self.pending.room, self.pending.executed)
        self.last_t = now
        self.pending = None
        return True


# ---------------------------------------------------------------------------
# PP0: the cap
# ---------------------------------------------------------------------------


def batch_rows(batch) -> int:
    """Device rows a launched batch allocates on every stage."""
    try:
        if batch.forward_mode.is_extend():
            return int(batch.extend_num_tokens or 0)
        return len(batch.reqs)
    except Exception:  # noqa: BLE001 - an unreadable batch counts nothing
        return 0


class Weg2PpRoomCap(msgspec.Struct, frozen=True):
    """R_m: the room every stage applies in pass m, riding list m of the
    request chain (the H42c burst-clock convention: PP0 decides pass m on the
    SAME value its followers read in their pass m)."""

    cap: int
    binding_rank: int = 0


class RoomBook:
    def __init__(self) -> None:
        self.pass_no = 0
        self.rows_by_ct: Dict[int, int] = {}
        self.facts: Dict[int, Tuple[Weg2PpRoomFact, int]] = {}
        self.log_n = 0

    def begin_pass(self) -> None:
        self.pass_no += 1

    def lane_echo(self) -> Dict[int, Tuple[int, int]]:
        """PRIORITY LANES 1008 (L3): ``{rank: (lane_floor, lane_epoch)}`` each fresh follower fact says it applies (only
        facts that carry lane state).  PP0 reads whether every stage stands on one epoch."""
        out: Dict[int, Tuple[int, int]] = {}
        for rank, (f, seen) in sorted(self.facts.items()):
            if self.pass_no - seen > FACT_MAX_AGE_PASSES:
                continue
            if isinstance(f, Weg2PpRoomFactLane):
                out[int(rank)] = (int(f.lane_floor), int(f.lane_epoch))
        return out

    def note_batches(self, batches: Iterable[Any]) -> None:
        for b in batches:
            ct = getattr(b, "forward_iter", None) if b is not None else None
            if isinstance(ct, int) and ct not in self.rows_by_ct:
                self.rows_by_ct[ct] = batch_rows(b)

    def absorb(self, facts: Iterable[Any]) -> int:
        n = 0
        for f in facts:
            if isinstance(f, Weg2PpRoomFact):
                self.facts[int(f.rank)] = (f, self.pass_no)
                n += 1
        return n

    def cap(self, own_room: Optional[int] = None) -> Optional[CapVerdict]:
        """R_m = MIN(PP0's own payable room now, each fresh follower fact minus
        the rows PP0 launched after it). None without a fresh follower fact --
        the pass then runs exactly as before."""
        best: Optional[CapVerdict] = None
        for rank, (f, seen) in sorted(self.facts.items()):
            if self.pass_no - seen > FACT_MAX_AGE_PASSES:
                continue
            inflight = sum(r for ct, r in self.rows_by_ct.items() if ct > f.executed)
            v = CapVerdict(cap=max(0, f.room - inflight), rank=rank, room=f.room, inflight=inflight)
            if best is None or v.cap < best.cap:
                best = v
        if self.facts:
            oldest = min(f.executed for f, _s in self.facts.values())
            for ct in [c for c in self.rows_by_ct if c <= oldest]:
                self.rows_by_ct.pop(ct, None)
        if best is not None and own_room is not None and int(own_room) < best.cap:
            best = CapVerdict(cap=max(0, int(own_room)), rank=0, room=int(own_room), inflight=0)
        return best


def set_cap(tree, cap: Optional[int]) -> None:
    """Every stage, pass m: the agreed R_m on its tree (None = no agreement)."""
    from sglang.srt.mem_cache.common import PP_ROOM_CAP_ATTR

    if cap is None and getattr(tree, PP_ROOM_CAP_ATTR, None) is None:
        return  # nothing agreed, nothing written: byte-identical
    setattr(tree, PP_ROOM_CAP_ATTR, None if cap is None else int(cap))


def apply_cap(tree, cap: Optional[int]) -> None:
    """A follower, pass m: R_m on its tree; its own pool under R_m at the
    pass top is the residual, counted (``PR AGREED-ABOVE-LOCAL where=pass``)."""
    from sglang.srt.mem_cache.common import fundable_extend_tokens

    set_cap(tree, None)
    if cap is None:
        return
    local = int(fundable_extend_tokens(tree))
    if local < int(cap):
        agreed_short("pass", None, int(cap), int(cap), local)
    set_cap(tree, cap)


def stamp_cap(wire_reqs, verdict: Optional[CapVerdict]) -> list:
    """PP0, before the chain send: list m carries R_m (nothing without one)."""
    out = list(wire_reqs or ())
    if verdict is not None:
        out.append(Weg2PpRoomCap(cap=int(verdict.cap), binding_rank=int(verdict.rank)))
    return out


def absorb_cap(recv_reqs) -> Tuple[list, Optional[int]]:
    """A follower, after relaying list m: take R_m off it (None when absent)."""
    caps = [r for r in (recv_reqs or ()) if isinstance(r, Weg2PpRoomCap)]
    if not caps:
        return recv_reqs, None  # the list untouched: byte-identical
    rest = [r for r in recv_reqs if not isinstance(r, Weg2PpRoomCap)]
    return rest, int(caps[-1].cap)


class Weg2PpLaneFloor(msgspec.Struct, frozen=True):
    """PRIORITY LANES 1008 (L3): the lane floor every stage applies in pass m, riding list m of the request chain beside
    ``Weg2PpRoomCap`` (PP0 applies it in pass m and stamps the standing value; a follower takes it off after relaying).
    ``PR LANE-FLOOR floor=N epoch=E`` is logged by each stage when its applied value changes."""

    floor: int
    epoch: int


def stamp_lane_floor(wire_reqs, stamp: Optional["Weg2PpLaneFloor"]) -> list:
    """PP0, before the chain send: list m carries the lane floor (nothing without one: byte-identical)."""
    out = list(wire_reqs or ())
    if stamp is not None:
        out.append(stamp)
    return out


def absorb_lane_floor(recv_reqs) -> Tuple[list, Optional["Weg2PpLaneFloor"]]:
    """A follower, after relaying list m: take the lane floor off it (None when absent; the list untouched then)."""
    marks = [r for r in (recv_reqs or ()) if isinstance(r, Weg2PpLaneFloor)]
    if not marks:
        return recv_reqs, None
    rest = [r for r in recv_reqs if not isinstance(r, Weg2PpLaneFloor)]
    return rest, marks[-1]


def note_binding(verdict: Optional[CapVerdict], own_fundable: int, book: RoomBook) -> None:
    """PP0: name R_m when it binds below PP0's own fundable reading."""
    if verdict is None or verdict.cap >= int(own_fundable):
        return
    book.log_n += 1
    n = book.log_n
    if n <= 8 or (n & (n - 1)) == 0:
        logger.warning(
            "PR PP-ROOM-CAP binding_rank=%d cap=%d room=%d inflight=%d own_fundable=%d (n=%d): "
            "every stage sizes this pass on the room every stage can pay -- the pass "
            "narrows or the request waits by name instead of running a stage out of "
            "memory or one stage starting it alone (#1004)",
            verdict.rank, verdict.cap, verdict.room, verdict.inflight, int(own_fundable), n,
        )


_AGREED_SHORT = {"n": 0}


def agreed_short(where: str, rid, need: int, cap: int, local: int) -> int:
    """The residual: this stage's own pool is below the agreed R_m. Counted and
    named (``PR AGREED-ABOVE-LOCAL``) -- the metal count of the risk."""
    _AGREED_SHORT["n"] += 1
    n = _AGREED_SHORT["n"]
    if n <= 8 or (n & (n - 1)) == 0:
        logger.warning(
            "PR AGREED-ABOVE-LOCAL where=%s rid=%s need=%d agreed=%d local=%d (n=%d): this "
            "stage's own pool is below the room the stages agreed for this pass (a fact "
            "older than the pool's last change)",
            where, rid, int(need), int(cap), int(local), n,
        )
    return n
