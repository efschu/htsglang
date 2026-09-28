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
completes, and while this rank's scheduler is dormant. So in Weg 2 exactly one
group writes -- the awake one -- and neither writes during a flip; the arena is
shared, so the awake group's owner rank sees the other group's pages too.
"""

from __future__ import annotations

import threading
import weakref
from typing import Optional

_lock = threading.Lock()
_state = {"legs": 0, "released": False, "sched": None, "entered": 0}


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
