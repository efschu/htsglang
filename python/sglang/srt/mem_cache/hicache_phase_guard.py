"""#718: disarm the device tier while the flip is in the phase that did not build it.

THE HAZARD, in one sentence: ``HiCacheController`` captures its device KV pool
ONCE, at construction, and the phase flip runs a second pool that the captured
one is not.

``cache_controller.py`` does the capture in ``__init__``
(``mem_pool_device = token_to_kv_pool_allocator.get_kvcache()``), and both
device transfers name that object directly --
``mem_pool_host.backup_from_device_all_layer(self.mem_pool_device, ...)`` on the
way out and ``load_to_device_per_layer(self.mem_pool_device, ...)`` on the way
back. The scheduler's allocator is assigned once too, and neither is reassigned
by the cutover. So the binding is to the BOOT (PP prefill) pool, permanently,
while the TP decode phase computes into the flip stack's own pool.

BOTH DIRECTIONS ARE CORRUPTION, not a dead cache. Worth stating precisely,
because "the cache just misses" would be a tolerable bug and this is not:

* WRITE (device -> host), during the TP phase: reads the PP pool's rows, which
  the model is no longer writing into -- at best the stale snapshot the cutover
  copy left there, at worst unbacked pages under the flip's VA-backed pools.
  Those bytes are then keyed by TOKEN CONTENT and, under #706, persisted to
  disk. A page that says "these tokens" while holding other tokens' KV is a
  wrong ANSWER later, in either phase, and it survives a reboot.
* LOAD (host -> device), during the TP phase: fills rows in the PP pool, which
  the model does not read. The radix tree nonetheless marks the prefix
  resident, so the scheduler SKIPS recomputing it and attention reads TP-pool
  rows that were never filled. Also a wrong answer, and a silent one.

THE CUT. Refuse device-tier I/O while the active phase is not the one the
binding belongs to. A refused write is a prefix that is not cached (a miss
later); a refused load is a prefetch that does not land (a miss now). Misses
are the correct failure mode here and they are cheap; both alternatives are
wrong output.

NO FLAG, DELIBERATELY. The predicate is false whenever the flip is not routing
(``phase_flip_tp_routing_active`` returns False when the secondary groups were
never built), so every deployment without ``--enable-phase-flip`` is
byte-identical to before by construction, and a flipping deployment gets the
guard unconditionally. A safety cut with an off switch is a footgun: the only
reason to turn it off would be to get back the behaviour that corrupts.

WHEN THE REAL FIX LANDS (rebinding the controller's pool at the cutover), this
predicate is the thing to update -- it encodes "the binding is the boot phase's"
in one place, on purpose.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#718 hicache-phase-guard]"

# One line per direction per process. The condition persists for a whole phase,
# so logging per operation would bury the log at decode rates.
_WARNED: set[str] = set()


def flip_routing_active() -> bool:
    """True while the phase flip routes collectives to its TP stack.

    False when the flip was never armed OR the secondary groups were never
    built, which is what makes this guard a no-op for every deployment that
    does not flip.
    """
    try:
        from sglang.srt.distributed.parallel_state import (
            phase_flip_tp_routing_active,
        )
    except Exception:  # pragma: no cover - import shape, not behaviour
        return False
    try:
        return bool(phase_flip_tp_routing_active())
    except Exception:  # pragma: no cover
        return False


def active_phase() -> str:
    """The phase the model is computing in right now."""
    return "tp" if flip_routing_active() else "pp"


def device_tier_disarmed(direction: Optional[str] = None) -> bool:
    """True when device-tier HiCache I/O must not run right now.

    The question is NOT "is the TP phase active" but "is the active phase the
    one this cache's pools are bound to". Those are the same question until
    #719's rebind is armed, and deliberately so: with no rebind the binding is
    always the boot (PP) phase, so this reduces to the #718 predicate exactly.
    Once a rebind moves the binding coherently, the tier is usable in the phase
    it was moved to, and this returns False there.

    ``direction`` only labels the log line ("write" / "load").
    """
    from sglang.srt.mem_cache.hicache_phase_binding import bound_phase

    if active_phase() == bound_phase():
        return False
    if direction is not None:
        warn_once(direction)
    return True


def warn_once(direction: str) -> None:
    if direction in _WARNED:
        return
    _WARNED.add(direction)
    logger.warning(
        "%s device-tier HiCache %s is DISARMED for the duration of the TP "
        "decode phase: this controller is bound to the pool it was built with "
        "(the boot PP stack's), which is not the pool the model is using now. "
        "Copying against it would %s. Prefixes cached in this phase are "
        "therefore not staged from device; the host and storage tiers are "
        "unaffected, and the PP phase resumes normal staging after the next "
        "flip.",
        LOG_PREFIX,
        direction,
        (
            "persist another row's bytes under a content-addressed key"
            if direction == "write"
            else "fill rows the model does not read, while the tree reports "
            "the prefix resident"
        ),
    )


def reset_warnings() -> None:
    """Test hook: forget which directions have already logged."""
    _WARNED.clear()
