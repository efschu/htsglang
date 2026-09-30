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

_clock = time.monotonic
_lock = threading.Lock()
_state = {"legs": 0, "released": False, "sched": None, "entered": 0}
#: EG review (c): the load-yield streak (see module docstring)
_load = {"since": None, "open_until": 0.0, "streaks": 0, "total_s": 0.0, "max_s": 0.0, "caps": 0}
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


def _load_gate() -> Optional[str]:
    """``load`` while loads are pending, except the bounded write window."""
    now = _clock()
    if loads_pending() <= 0:
        _load_streak_end(now)
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
    """Why the write-behind must not run now, or None."""
    if _state["legs"] > 0:
        return "leg"
    if _state["released"]:
        return "asleep"
    ref = _state["sched"]
    sched = ref() if ref is not None else None
    if sched is not None and bool(getattr(sched, "pdflip_dormant", False)):
        return "dormant"
    return _load_gate()


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
        _load_probes.clear()
        _load.update(since=None, open_until=0.0, streaks=0, total_s=0.0, max_s=0.0, caps=0)
