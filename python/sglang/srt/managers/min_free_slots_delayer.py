import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "MIN-FREE-SLOTS"

#: The three things ``min_free_slots_verdict`` can do with what the operator
#: asked for (#894 S4). Named so a caller can say which one happened instead of
#: silently substituting a different number, or none.
MIN_FREE_SLOTS_HONOURED = "honoured"
MIN_FREE_SLOTS_CAPPED = "capped"
MIN_FREE_SLOTS_DISABLED_SMALL_POOL = "disabled_small_pool"

#: Below this many running-request slots the delayer is not built at all: with
#: a pool that small, holding admissions until N slots free would keep the
#: batch permanently under-filled rather than batch anything.
MIN_POOL_FOR_DELAY = 8


def _dflash_formula(max_running_requests: int) -> int:
    """The legacy DFlash heuristic, and the ceiling on any user value."""
    return min(4, max(2, (max_running_requests + 5) // 6))


def min_free_slots_verdict(
    user_value: Optional[int],
    max_running_requests: int,
    is_dflash_family: bool = False,
) -> Tuple[Optional[int], str]:
    """Resolve the threshold AND say what happened to the requested value.

    Returns ``(resolved, reason)``; ``resolved is None`` means no delayer is
    built. ``resolve_min_free_slots`` is the value half of this and is
    byte-for-byte the same number it always returned -- this function adds the
    second half, which is the half #894 S4 is about.

    THE DEFECT. ``--min-free-slots-delay`` was narrowed and, in one case,
    discarded outright without a word anywhere. There was no ``logger`` in this
    module and none at the call site, so an operator who wrote ``6`` and got
    ``4`` -- or who wrote ``6`` and got no admission gate at all, because the
    KV resolver had put ``max_running_requests`` below
    ``MIN_POOL_FOR_DELAY`` -- saw output identical to a request that was
    honoured. The help text stated the cap in prose, which is parse-time
    documentation and not a runtime answer about THIS boot's numbers; #889
    settled that those are not the same thing.

    The discard is the dangerous half: a capped value still delays, a discarded
    one leaves ``Scheduler.min_free_slots_delayer`` at ``None``, so the gate the
    operator configured is simply absent from the admission path.
    """
    max_running_requests = max(0, int(max_running_requests))
    formula = _dflash_formula(max_running_requests)
    requested = user_value
    if user_value is None:
        user_value = formula if is_dflash_family else None

    if user_value is None or user_value <= 1:
        # Nothing was asked for (or the request is the documented "off"), so
        # nothing was taken away.
        return None, MIN_FREE_SLOTS_HONOURED
    if max_running_requests < MIN_POOL_FOR_DELAY:
        return None, (
            MIN_FREE_SLOTS_DISABLED_SMALL_POOL
            if requested is not None
            else MIN_FREE_SLOTS_HONOURED
        )
    resolved = min(user_value, formula)
    if requested is not None and resolved < requested:
        return resolved, MIN_FREE_SLOTS_CAPPED
    return resolved, MIN_FREE_SLOTS_HONOURED


def narrowed_min_free_slots_warning(
    user_value: Optional[int],
    max_running_requests: int,
    is_dflash_family: bool = False,
) -> Optional[str]:
    """The line the scheduler must log when the request was not honoured (#894).

    ``None`` whenever the operator asked for nothing, or got what they asked
    for. A configuration that was never narrowed must not acquire a warning, or
    the warning stops being read -- same rule as ``superseded_pp_bound_warning``.

    WARNING, NOT REFUSAL, decided on the danger direction:

    * The discarding input is ``max_running_requests``, which is NOT a
      parse-time quantity. #287 made the ServerArgs field a CEILING and lets
      the KV-capacity resolver cut it further during scheduler init, so whether
      a written value survives is decided after the pools are built. A refusal
      there is a dead boot, mid-init, on a configuration that booted yesterday
      with a slightly larger KV budget.
    * The defect's blast radius is a wrong belief about how admissions are
      batched. A refusal's blast radius is the instance. Trading a
      documentation defect for an availability defect is the worse direction.
    * Nor may the code "honour it anyway" by building the delayer below the
      pool floor: with fewer than ``MIN_POOL_FOR_DELAY`` slots, a delayer that
      waits for N of them free keeps the running batch under-filled instead of
      batching anything. The cap and the floor are both deliberate. State what
      they did; do not undo them.
    """
    resolved, reason = min_free_slots_verdict(
        user_value, max_running_requests, is_dflash_family=is_dflash_family
    )
    if reason == MIN_FREE_SLOTS_HONOURED:
        return None
    pool = max(0, int(max_running_requests))
    if reason == MIN_FREE_SLOTS_DISABLED_SMALL_POOL:
        return (
            f"{LOG_PREFIX} #894 NARROWED KNOB: --min-free-slots-delay="
            f"{user_value} is DISCARDED, not capped -- there is NO ADMISSION "
            f"DELAY on this boot. The delayer is only built at "
            f"max_running_requests >= {MIN_POOL_FOR_DELAY}, and this rank "
            f"resolved {pool}. Note that value is not the ServerArgs ceiling: "
            f"the KV-capacity resolver may have cut it (#287), so a config "
            f"that kept the gate on a larger KV budget can lose it here. Raise "
            f"--max-running-requests / the KV budget above "
            f"{MIN_POOL_FOR_DELAY}, or drop the flag."
        )
    return (
        f"{LOG_PREFIX} #894 NARROWED KNOB: --min-free-slots-delay="
        f"{user_value} was CAPPED to {resolved}. The trigger may never delay "
        f"more aggressively than the DFlash formula min(4, max(2, "
        f"(max_running_requests + 5) // 6)), which is {_dflash_formula(pool)} "
        f"at max_running_requests={pool}. Read admission behaviour against "
        f"{resolved}, not against {user_value}."
    )


def resolve_min_free_slots(
    user_value: Optional[int],
    max_running_requests: int,
    is_dflash_family: bool = False,
) -> Optional[int]:
    """Resolve the min-free-slots threshold (None = disabled).

    A user value (>1) is capped to the DFlash formula so the trigger never
    delays more aggressively than the legacy heuristic. When unset, DFlash
    workloads fall back to the formula (preserving the always-on behavior);
    other workloads stay disabled. Also disabled when max_running_requests < 8.

    #894: the value half of ``min_free_slots_verdict``, kept as its own entry
    point because ``scheduler.py`` and the registered suites call it. It moves
    no number; the reason half is what was missing.
    """
    return min_free_slots_verdict(
        user_value, max_running_requests, is_dflash_family=is_dflash_family
    )[0]


class MinFreeSlotsDelayer:
    """Delay fresh prefill admissions until at least ``min_free_slots`` running-
    request slots free up, batching them into one admission instead of one at a
    time. Useful when each admission is expensive (e.g. DFlash's draft prefill).

    Per-rank local: running-batch slots are private to each DP rank, so a rank
    with free slots does not wait for a congested peer.
    """

    def __init__(self, min_free_slots: int):
        self._min_free_slots = min_free_slots

    def should_delay(self, *, running_bs: int, num_allocatable_reqs: int) -> bool:
        return running_bs > 0 and num_allocatable_reqs < self._min_free_slots
