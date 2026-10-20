"""L3-REUSE 0928: when the L3 write-behind may run in this process.

The write-behind (``HiCacheFile.l3_write_behind_pass``) copies COMPLETE L2
arena pages to the persistent disk store. It never competes with a flip: the
legs own the arena, the PCIe lanes and the host bandwidth (user law "HiCache
bremst Prefill/Decode nie").

No new clock -- the gate reads the flip state that already exists:

* the leg bracket: ``release_memory_occupation`` / ``resume_memory_occupation``
  run inside ``_pdflip_group_stop_on_leg_failure`` (weight_updater), which calls
  :func:`leg_enter` / :func:`leg_exit` here;
* ``scheduler.pdflip_dormant`` (W25): True from the sleep leg's kv pause to the
  wake's late site.

Quiet while a leg runs on this rank, after a sleep leg until the next wake leg
completes, and while this rank's scheduler is dormant.

HICACHE-NEVER-SLOW (28.09., 27B release-draft 13:51:00): quiet also while a
store->host LOAD is queued or in flight on this rank (``load``). The pass and
the load share the store's I/O; a 70142-token load of pdflip-1-17 waited
queue_ms=17016 for a read of 227 ms while six write-behind passes of ~1.4 s ran
-- 21 s without a prefill on an awake P with work queued. The write-behind is
background work and yields; the load is the request's critical path. So in Weg 2 exactly one
group writes -- the awake one -- and neither writes during a flip; the arena is
shared, so the awake group's owner rank sees the other group's pages too.

EG review (c): under a load that never stops the write-behind would never
write. Every ``load`` streak is timed (:func:`load_yield_stats`, an
``L3-WB LOAD-YIELD END`` line for streaks of ``_LONG_STREAK_S`` or more) and
bounded: after ``FLLIPER_HICACHE_L3_WB_LOAD_YIELD_MAX_S`` (30 s, 0 = no bound)
of one unbroken streak the gate opens for ``FLLIPER_HICACHE_L3_WB_LOAD_WRITE_S``
(2 s) -- ``L3-WB LOAD-YIELD CAP`` -- then the streak starts over. At most
2 s of writes per 30 s of continuous load.

S36 SOLL 2 ENDGUELTIG (18.10., user: "das sollte schon immer so sein, das von L2 auf L3 geschrieben wird ohne zu
blockieren" and "flip oder schlaf sind d2h oder h2d zwischen gpu und systemram. zurueckschreiben in L3 ist disk und
systemram. unabhaengig also."): the L2 -> L3 write NEVER holds still -- not during a leg, not while the group sleeps
or is dormant, not while loads are pending. Measured on the rc9 restart, D TP0: ``LOAD-YIELD`` 430 s in ~25 min,
single streaks up to 84 s, D's tree pages ``l3=on_disk:0,written:0`` -- a full arena that nothing could give back.
:func:`quiet_reason` answers None always; the leg / sleep / load bookkeeping stays (``awake_epoch`` and the
censuses read it) but gates nothing. The 28.09. problem (a 227 ms read queued 17 s behind write passes) is solved
by READ PRIORITY in the I/O order instead of silence: every store read brackets its I/O with :func:`read_enter` /
:func:`read_exit`; the writer asks :func:`write_turn` before each SMALL write chunk and waits while a read is in
flight -- at most ``FLLIPER_HICACHE_L3_WB_READ_YIELD_MS`` (20 ms) per chunk, then the chunk is written anyway, so
reads overtake and writes still move (>= one chunk per yield window).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import weakref
from typing import Optional

logger = logging.getLogger(__name__)

ENV_LOAD_YIELD_MAX_S = "FLLIPER_HICACHE_L3_WB_LOAD_YIELD_MAX_S"
ENV_LOAD_WRITE_S = "FLLIPER_HICACHE_L3_WB_LOAD_WRITE_S"
LOAD_YIELD_MAX_S_DEFAULT = 30.0
LOAD_WRITE_S_DEFAULT = 2.0
_LONG_STREAK_S = 5.0
#: S35 (18.10., rc9 restart 10:09:06-10:10:14Z): store reads of this process waiting in the arena ROOM-WAIT right now
#: (``HiCacheFile._await_fill_room``). See :func:`_load_gate`.
_room = {"waiting": 0, "starved_n": 0}

_clock = time.monotonic
_lock = threading.Lock()
_state = {"legs": 0, "released": False, "sched": None, "entered": 0}
#: EG review (c): the load-yield streak (see module docstring)
_load = {"since": None, "open_until": 0.0, "streaks": 0, "total_s": 0.0, "max_s": 0.0, "caps": 0}
#: S36: store reads with I/O in flight right now (read priority over the write-behind), and the census
_reads = {"inflight": 0, "yields": 0, "yield_s": 0.0, "max_s": 0.0, "forced": 0}
_read_cv = threading.Condition(threading.Lock())
ENV_READ_YIELD_MS = "FLLIPER_HICACHE_L3_WB_READ_YIELD_MS"
READ_YIELD_MS_DEFAULT = 20.0
#: callables answering "how many store->host loads are queued or in flight"
#: (the cache controller's prefetch queues registers one); any > 0 = quiet.
_load_probes: list = []


def register_load_probe(fn) -> None:
    """The cache controller registers its load-queue census (weak: a dead
    controller drops out). A probe that raises counts as no load."""
    try:
        ref = weakref.WeakMethod(fn) if hasattr(fn, "__self__") else (lambda f=fn: f)
    except TypeError:
        ref = lambda f=fn: f  # noqa: E731
    with _lock:
        _load_probes.append(ref)


def loads_pending() -> int:
    n = 0
    with _lock:
        probes = list(_load_probes)
    for ref in probes:
        fn = ref()
        if fn is None:
            continue
        try:
            n += int(fn() or 0)
        except Exception:  # noqa: BLE001 -- a broken probe must not stop the gate
            continue
    return n


def leg_enter(op: str, scheduler=None) -> None:
    """A leg RPC starts on this rank (``op``: the handler name)."""
    with _lock:
        _state["legs"] += 1
        _state["entered"] += 1
        if "release" in str(op):
            _state["released"] = True
        if scheduler is not None:
            try:
                _state["sched"] = weakref.ref(scheduler)
            except TypeError:
                pass


def leg_exit(op: str, ok: bool) -> None:
    """The leg RPC returned (``ok``) or raised on this rank."""
    with _lock:
        _state["legs"] = max(0, _state["legs"] - 1)
        if ok and "resume" in str(op):
            _state["released"] = False


def _env_s(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return v if v >= 0 else default


def _load_streak_end(now: float) -> None:
    since = _load["since"]
    if since is None:
        return
    held = max(0.0, now - since)
    _load["since"] = None
    _load["streaks"] += 1
    _load["total_s"] += held
    _load["max_s"] = max(_load["max_s"], held)
    if held >= _LONG_STREAK_S:
        logger.info("L3-WB LOAD-YIELD END held_s=%.1f streaks=%d total_s=%.1f max_s=%.1f caps=%d "
                    "(the write-behind yielded to store->host loads this long)",
                    held, _load["streaks"], _load["total_s"], _load["max_s"], _load["caps"])


def read_enter() -> None:
    """S36: a store read starts moving I/O (an L3 -> L2 fill, a KV-DURCHLAUF read)."""
    with _read_cv:
        _reads["inflight"] += 1


def read_exit() -> None:
    with _read_cv:
        _reads["inflight"] = max(0, _reads["inflight"] - 1)
        if _reads["inflight"] == 0:
            _read_cv.notify_all()


def reads_inflight() -> int:
    return int(_reads["inflight"])


def write_turn(max_wait_s: Optional[float] = None) -> float:
    """S36 read priority: called by the write-behind before each write chunk. Waits while a store read is in
    flight, at most ``max_wait_s`` (default ``FLLIPER_HICACHE_L3_WB_READ_YIELD_MS``), then returns -- the chunk is
    written either way. Returns the seconds waited."""
    if max_wait_s is None:
        max_wait_s = _env_s(ENV_READ_YIELD_MS, READ_YIELD_MS_DEFAULT) / 1000.0
    if _reads["inflight"] <= 0 or max_wait_s <= 0:
        return 0.0
    t0 = _clock()
    deadline = t0 + max_wait_s
    with _read_cv:
        while _reads["inflight"] > 0:
            left = deadline - _clock()
            if left <= 0:
                _reads["forced"] += 1
                break
            _read_cv.wait(left)
    held = max(0.0, _clock() - t0)
    _reads["yields"] += 1
    _reads["yield_s"] += held
    _reads["max_s"] = max(_reads["max_s"], held)
    return held


def read_priority_stats() -> dict:
    return {"inflight": _reads["inflight"], "yields": _reads["yields"], "yield_s": _reads["yield_s"],
            "max_s": _reads["max_s"], "forced": _reads["forced"]}


def room_wait_enter() -> None:
    """S35: a store read enters the arena ROOM-WAIT (the arena was FULL for its stems)."""
    with _lock:
        _room["waiting"] += 1


def room_wait_exit() -> None:
    with _lock:
        _room["waiting"] = max(0, _room["waiting"] - 1)


def room_starved() -> bool:
    """S35: True while a store read of this process waits for arena room."""
    return _room["waiting"] > 0


def _load_gate() -> Optional[str]:
    """``load`` while loads are pending, except the bounded write window.

    S35 (SOLL R7: "kann D einen Request unter X nicht nehmen, ist das ein Defekt"; fLLiper rc9 restart, D TP0
    10:09:06-10:10:14Z, 68 s without a D round at 18 open requests): NOT while a pending load is itself waiting for
    arena ROOM. The arena was full of D's own tree pages without an L3 copy (``slots=6485 pinned=6485``, ARENA-DROP
    ``l3=on_disk:0,written:0 freed=0``, READ-ROOM D-SPILL ``candidates=0 host_leaves=0``); a slot frees only once its
    page has an L3 copy, and the L3 copy is this write-behind -- which yielded to that very load (``L3-WB LOAD-YIELD END
    held_s=20.0 ... max_s=71.9``). A circular wait, ended there only by a client abort. A load in the room wait moves
    no store I/O, so the yield buys it nothing: the write-behind writes."""
    now = _clock()
    if loads_pending() <= 0:
        _load_streak_end(now)
        return None
    if room_starved():
        if _load["since"] is not None:
            _load_streak_end(now)
            _room["starved_n"] += 1
            n = _room["starved_n"]
            if n <= 16 or (n & (n - 1)) == 0:
                logger.info("L3-WB ROOM-STARVED n=%d: a pending store read waits for arena ROOM, which only an L3 "
                            "copy of a full arena page can free -- the write-behind writes instead of yielding (S35)", n)
        return None
    if now < _load["open_until"]:
        return None
    if _load["since"] is None:
        _load["since"] = now
        return "load"
    bound = _env_s(ENV_LOAD_YIELD_MAX_S, LOAD_YIELD_MAX_S_DEFAULT)
    held = now - _load["since"]
    if bound > 0 and held >= bound:
        _load["caps"] += 1
        _load_streak_end(now)
        _load["open_until"] = now + _env_s(ENV_LOAD_WRITE_S, LOAD_WRITE_S_DEFAULT)
        logger.info("L3-WB LOAD-YIELD CAP held_s=%.1f bound_s=%.1f write_s=%.1f caps=%d: loads were "
                    "pending for the whole bound, the write-behind writes this window anyway",
                    held, bound, _load["open_until"] - now, _load["caps"])
        return None
    return "load"


def load_yield_stats() -> dict:
    """The load-yield census: streaks, their total and longest, caps fired,
    and the running streak's age (0 when none runs)."""
    since = _load["since"]
    return {"streaks": _load["streaks"], "total_s": _load["total_s"], "max_s": _load["max_s"],
            "caps": _load["caps"], "running_s": 0.0 if since is None else max(0.0, _clock() - since)}


def quiet_reason() -> Optional[str]:
    """Why the write-behind must not run now -- S36 SOLL 2 ENDGUELTIG: never (None). Legs, sleep, dormancy and
    pending loads gate nothing any more (see the module docstring); reads get priority per chunk
    (:func:`write_turn`) instead. The former gates are kept readable as :func:`former_quiet_reason` for the
    censuses and the tests that pin that they no longer gate."""
    return None


def former_quiet_reason() -> Optional[str]:
    """What the pre-S36 gate would have answered now (instrument only, gates nothing)."""
    if _state["legs"] > 0:
        return "leg"
    if _state["released"]:
        return "asleep"
    ref = _state["sched"]
    sched = ref() if ref is not None else None
    if sched is not None and bool(getattr(sched, "pdflip_dormant", False)):
        return "dormant"
    return "load" if loads_pending() > 0 else None


def awake_epoch() -> Optional[int]:
    """P4b-cap (28.09.): the leg counter while this rank's scheduler is known
    and NOT dormant (W25 clears ``pdflip_dormant`` at the wake's late site, after
    the peer group released its KV); None while dormant or before any leg
    named the scheduler. Two equal answers bracket a span in which no leg
    began and the rank was awake at both ends -- the store read registered at
    the first answer probed a store the sleeping group no longer writes."""
    ref = _state["sched"]
    sched = ref() if ref is not None else None
    if sched is None or bool(getattr(sched, "pdflip_dormant", False)):
        return None
    return int(_state["entered"])


def quiet() -> bool:
    return quiet_reason() is not None


def _reset_for_tests() -> None:
    with _lock:
        _state.update(legs=0, released=False, sched=None, entered=0)
        _room.update(waiting=0, starved_n=0)
        _reads.update(inflight=0, yields=0, yield_s=0.0, max_s=0.0, forced=0)
        _load_probes.clear()
        _load.update(since=None, open_until=0.0, streaks=0, total_s=0.0, max_s=0.0, caps=0)
