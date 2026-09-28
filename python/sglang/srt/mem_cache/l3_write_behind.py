"""L3-REUSE 0928: when the L3 write-behind may run in this process.

The write-behind (``HiCacheFile.l3_write_behind_pass``) copies COMPLETE L2
arena pages to the persistent disk store. It never competes with a flip: the
legs own the arena, the PCIe lanes and the host bandwidth (user law "HiCache
bremst Prefill/Decode nie").

No new clock -- the gate reads the flip state that already exists:

* the leg bracket: ``release_memory_occupation`` / ``resume_memory_occupation``
  run inside ``_weg2_group_stop_on_leg_failure`` (weight_updater), which calls
  :func:`leg_enter` / :func:`leg_exit` here;
* ``scheduler.weg2_dormant`` (W25): True from the sleep leg's kv pause to the
  wake's late site.

Quiet while a leg runs on this rank, after a sleep leg until the next wake leg
completes, and while this rank's scheduler is dormant.

HICACHE-NEVER-SLOW (28.09., 27B release-draft 13:51:00): quiet also while a
store->host LOAD is queued or in flight on this rank (``load``). The pass and
the load share the store's I/O; a 70142-token load of weg2-1-17 waited
queue_ms=17016 for a read of 227 ms while six write-behind passes of ~1.4 s ran
-- 21 s without a prefill on an awake P with work queued. The write-behind is
background work and yields; the load is the request's critical path. So in Weg 2 exactly one
group writes -- the awake one -- and neither writes during a flip; the arena is
shared, so the awake group's owner rank sees the other group's pages too.
"""

from __future__ import annotations

import threading
import weakref
from typing import Optional

_lock = threading.Lock()
_state = {"legs": 0, "released": False, "sched": None, "entered": 0}
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


def quiet_reason() -> Optional[str]:
    """Why the write-behind must not run now, or None."""
    if _state["legs"] > 0:
        return "leg"
    if _state["released"]:
        return "asleep"
    ref = _state["sched"]
    sched = ref() if ref is not None else None
    if sched is not None and bool(getattr(sched, "weg2_dormant", False)):
        return "dormant"
    if loads_pending() > 0:
        return "load"
    return None


def awake_epoch() -> Optional[int]:
    """P4b-cap (28.09.): the leg counter while this rank's scheduler is known
    and NOT dormant (W25 clears ``weg2_dormant`` at the wake's late site, after
    the peer group released its KV); None while dormant or before any leg
    named the scheduler. Two equal answers bracket a span in which no leg
    began and the rank was awake at both ends -- the store read registered at
    the first answer probed a store the sleeping group no longer writes."""
    ref = _state["sched"]
    sched = ref() if ref is not None else None
    if sched is None or bool(getattr(sched, "weg2_dormant", False)):
        return None
    return int(_state["entered"])


def quiet() -> bool:
    return quiet_reason() is not None


def _reset_for_tests() -> None:
    with _lock:
        _state.update(legs=0, released=False, sched=None, entered=0)
        _load_probes.clear()
