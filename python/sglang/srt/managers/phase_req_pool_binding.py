"""#1040 -- per-phase ownership of the REQUEST-INDEX space.

The phase flip runs two model stacks. Each builds its own ``ReqToTokenPool``,
and until this cut the boot ALIASED them: the TP pool's ``req_to_token`` (and
its mamba index map) were pointed at the PP pool's tensors, and the scheduler's
``req_to_token_pool`` was never reassigned at the cutover. Every row id the TP
phase used was therefore minted by the PP allocator, and a row freed in one
phase was still named by the other.

Deleting the alias is only half the cut. The other half is here: at every
cutover the scheduler must be REBOUND to the incoming phase's own pool, the
outgoing pool must be censused (a row still held there is a request that
escaped the drain -- #919 on the request axis, never measured before), and the
incoming pool must start from a fresh-boot state.

THREE DESIGN POINTS, each the answer to a way this could have gone wrong:

* **Not gated on ``--phase-flip-rebind-hicache``.** That flag guards the
  HiCache pool rebind, which has a disarmed state that is merely a cache miss
  (#719: "a refused rebind is SAFE"). A request pool has no disarmed state. If
  the alias is deleted and the rebind does not happen, the TP phase reads a
  tensor nobody writes -- so this runs on every cutover, both flag states.
* **Raises, never logs.** Same reason. There is nothing to fall back to.
* **Refuses through its own resolver.** ``hicache_phase_binding.phase_pools_for``
  raises when a phase has no HOST pool, which is a legitimate HiCache-side
  refusal and has nothing to do with request rows. Routing the request-pool
  rebind through it would make the request-index space depend on the host
  budget -- the exact coupling the first design point forbids. Only the
  phase->runner walk is shared (``runner_for_phase``); the pool lookup is here.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Iterable, Optional, Sequence

from sglang.srt.mem_cache.hicache_phase_binding import runner_for_phase

logger = logging.getLogger(__name__)

LOG_PREFIX = "#1040 REQ-POOL"

# Emitted once per cutover on EVERY rank, whatever the census found. An
# unmeasured zero is not a zero (indicator law): a line that only appears when
# something is wrong cannot tell "nothing escaped" from "nobody looked".
REBIND_LOG_FORMAT = (
    "#1040 REQ-POOL REBOUND to the '%(phase)s' request pool "
    "(binding=%(binding)s, rows=%(rows)d, "
    "outgoing free=%(free)d/%(size)d, escapees=%(escapees)d rids=%(rids)s)"
)


class ReqPoolRebindRefused(RuntimeError):
    """The scheduler cannot be pointed at the incoming phase's request pool.

    Raised, not logged: there is no safe continuation. Carrying on would run
    the incoming phase against the outgoing phase's rows, which is the very
    defect this cut removes.
    """


@dataclasses.dataclass(frozen=True)
class ReqPoolCensus:
    """What the OUTGOING request pool still holds at the cutover.

    ``escapees`` is the count of rows the pool has not got back. ``free`` and
    ``size`` travel with it so the number carries its own denominator.

    ``rows`` are the HELD ROW IDS, derived from the pool's own free list, so
    they are present whenever the count is -- with or without a request list.
    ``rids`` names the subset of those rows a live request could be matched
    to, and ``unnamed`` is the rest. Boot 9 printed ``escapees=1 rids=[]
    rows=[]`` (log:2305/:2351): a guard that fired correctly and told the
    operator nothing (#1202).
    """

    size: int
    free: int
    escapees: int
    rows: tuple
    rids: tuple
    unnamed: int = 0


def census_outgoing_req_pool(
    pool: Any, reqs: Iterable[Any] = ()
) -> ReqPoolCensus:
    """Count the rows the outgoing phase did not return, AND NAME THEM.

    The pool's own arithmetic is the authority on HOW MANY (``free_slots``
    against ``size``) and, since #1202, on WHICH: ``free_slots`` is minted as
    ``range(1, size + 1)``, so the held rows are exactly that range minus the
    free list. That reading needs no request list at all, which matters
    because the request list is precisely what is missing in the failure this
    census exists to report.

    THE REQUEST LIST ONLY ADDS NAMES. Boot 9 refused with ``escapees=1,
    rids=[], rows=[]`` -- an operator reading that line learns a row escaped
    and nothing else, and the row id was available from the pool the whole
    time. Rows the list cannot account for are counted in ``unnamed`` rather
    than dropped: "one held row, none of them named" and "no held rows" must
    never print the same way (indicator law).
    """
    size = int(pool.size)
    free_slots = list(getattr(pool, "free_slots", ()) or ())
    free = len(free_slots)
    escapees = size - free
    held = sorted(set(range(1, size + 1)) - {int(x) for x in free_slots})
    by_row = {}
    for r in reqs or ():
        idx = getattr(r, "req_pool_idx", None)
        if idx is None:
            continue
        by_row.setdefault(int(idx), getattr(r, "rid", "<unknown>"))
    rids = [by_row[row] for row in held if row in by_row]
    return ReqPoolCensus(
        size=size,
        free=free,
        escapees=escapees,
        rows=tuple(held),
        rids=tuple(rids),
        unnamed=len(held) - len(rids),
    )


def _live_reqs(scheduler: Any) -> Sequence[Any]:
    """Every request the scheduler can still name, for the census's RIDS.

    ONE AUTHORITY, NOT A SECOND WALK (#1202). This used to enumerate
    ``running_batch`` plus ``waiting_queue`` -- two containers, while the
    flip's residency authority (``phase_flip_runtime._live_reqs``) walks six.
    A refusal that counts rows with the pool's arithmetic and names them with
    a narrower walk than the retraction used can only ever under-name, and
    boot 9 is what that looks like: ``escapees=1 rids=[]``, with the request
    reachable through ``last_mbs`` at the arm.

    So the walk is DELEGATED to that authority and this module keeps the one
    route the flip does not have: ``waiting_queue``, which holds requests that
    are admitted but not yet resident and can still carry a row.

    Naming stays best-effort by construction -- a row can outlive the object
    that held it, which is precisely the leak being measured -- so this never
    decides the count, only the labels. The import is deferred because
    ``phase_flip_runtime`` imports this module at its rebind site; a failed
    import degrades to the narrow walk and says so, rather than breaking a
    cutover for the sake of a label.
    """
    out = []
    try:
        from sglang.srt.managers.phase_flip_runtime import _live_reqs as authority

        out.extend(authority(scheduler))
    except Exception as exc:  # noqa: BLE001 - a label may never break a rebind
        logger.warning(
            "%s the flip's residency authority could not be consulted for the "
            "census labels (%s); falling back to the running batch alone, so "
            "an unnamed escapee may still be nameable",
            LOG_PREFIX,
            exc,
        )
        batch = getattr(scheduler, "running_batch", None)
        for r in getattr(batch, "reqs", None) or ():
            out.append(r)
    seen = {id(r) for r in out}
    for r in getattr(scheduler, "waiting_queue", None) or ():
        if id(r) not in seen:
            seen.add(id(r))
            out.append(r)
    return out


def req_pool_for_phase(scheduler: Any, phase: str) -> Optional[Any]:
    """The named phase's OWN request pool, or None when its stack is absent."""
    runner = runner_for_phase(scheduler, phase)
    return getattr(runner, "req_to_token_pool", None)


def rebind_req_pool_for_cutover(scheduler: Any, phase: str) -> ReqPoolCensus:
    """Point the scheduler at the incoming phase's request pool.

    Order is load-bearing:

    1. resolve the incoming pool -- refuse if the phase has none;
    2. census the OUTGOING pool while it is still the scheduler's, and refuse
       when rows escaped (a live row in a pool nobody will consult again is a
       request whose rows are lost, not a tidiness complaint);
    3. clear the incoming pool, so the phase starts from the fresh-boot state
       its consumers assume and any id carried across the seam is refused by
       the wrong-row guard rather than landing in range;
    4. rebind, then emit the line -- unconditionally.

    Called on EVERY cutover regardless of ``--phase-flip-rebind-hicache``; see
    the module docstring for why that flag must not reach this code.
    """
    incoming = req_pool_for_phase(scheduler, phase)
    if incoming is None:
        raise ReqPoolRebindRefused(
            f"{LOG_PREFIX} the '{phase}' phase has no request pool of its own. "
            "Since #1040 the two phases no longer share one req_to_token "
            "tensor, so a missing pool means the incoming phase would run "
            "against the outgoing phase's rows -- refused here rather than "
            "silently continuing on the wrong request-index space."
        )

    outgoing = getattr(scheduler, "req_to_token_pool", None)
    census = (
        census_outgoing_req_pool(outgoing, _live_reqs(scheduler))
        if outgoing is not None
        else ReqPoolCensus(size=0, free=0, escapees=0, rows=(), rids=())
    )

    if census.escapees:
        raise ReqPoolRebindRefused(
            f"{LOG_PREFIX} {census.escapees} of {census.size} rows are still "
            f"held in the OUTGOING request pool at the cutover "
            f"(free={census.free}, rows={list(census.rows)}, "
            f"rids={list(census.rids)}, unnamed={census.unnamed}). "
            f"The drain is supposed to return "
            "every row before the seam; a row left here belongs to a request "
            "the incoming phase can no longer reach, and the next allocation "
            "in this pool would hand that row to somebody else (#919 on the "
            "request axis)."
        )

    # The incoming pool must look like a fresh boot: clear() also re-mints its
    # binding tag, so a `req_pool_idx` carried across the seam is refused by
    # ReqToTokenPool.alloc instead of landing in range on another row.
    incoming.clear()
    scheduler.req_to_token_pool = incoming
    _restamp_phase_handles(scheduler, outgoing, incoming)

    logger.info(
        REBIND_LOG_FORMAT,
        {
            "phase": phase,
            "binding": getattr(incoming, "binding_tag", "<untagged>"),
            "rows": int(incoming.size),
            "free": census.free,
            "size": census.size,
            "escapees": census.escapees,
            "rids": list(census.rids),
        },
    )
    return census


# ---------------------------------------------------------------------------
# #1201: the OTHER holders of the request pool.
# ---------------------------------------------------------------------------
#
# The rebind above moves the SCHEDULER's handle. Three more objects cached a
# reference to the same pool at CONSTRUCTION, and none of them is on any
# cutover path:
#
#   * `UnifiedRadixCache.req_to_token_pool` -- answered by the read-at-use
#     property on that class; this file only has to name its owner;
#   * the pool's back-reference to the tree cache (`bind_tree_cache`), whose
#     only two callers are tree-cache constructors, so the INCOMING pool keeps
#     `tree_cache = None` for the whole phase. That disarms the evict-then-retry
#     in `_alloc_mamba_slots_or_evict` and silently reintroduces the #581/#773
#     regression the comment at `unified_radix_cache.py` was written to close;
#   * the pool's `layer_transfer_counter` (+ its mamba frame), registered once
#     by the assembler. A join on a pool nothing re-registered is a join that
#     never happens -- `_wait_for_mamba_layer` returns silently when the
#     counter is None.
#
# WHY RE-STAMP RATHER THAN REFUSE. The counter's None state is a LEGITIMATE
# steady state, not a defect: `register_layer_transfer_counter` documents
# `mamba_transfer_frame=None` as "the historic no-wait behaviour", and
# `swa_memory_pool` registers None deliberately. So a refusal at the join site
# would refuse every boot without a hierarchical cache. What is a defect is the
# handle failing to CROSS the seam, and that is decidable here, where both
# pools are in hand.
#
# The fourth holder, `FutureMap.pool` (built once in `Scheduler.init_overlap`),
# is NOT handled here and is registered as an open gap in
# `cutover_participants`.


def _restamp_phase_handles(scheduler: Any, outgoing: Any, incoming: Any) -> None:
    """Re-stamp onto the INCOMING pool every handle the boot stamped once.

    Duck-typed on purpose: `ReqToTokenPool` has neither hook, `HybridReqToTokenPool`
    has both, and a pool that cannot carry a handle simply does not get one.
    """
    tree_cache = getattr(scheduler, "tree_cache", None)

    if tree_cache is not None and hasattr(tree_cache, "bind_req_pool_owner"):
        tree_cache.bind_req_pool_owner(scheduler)

    if tree_cache is not None and hasattr(incoming, "bind_tree_cache"):
        incoming.bind_tree_cache(tree_cache)

    counter = getattr(outgoing, "layer_transfer_counter", None)
    if counter is not None and hasattr(incoming, "register_layer_transfer_counter"):
        if getattr(incoming, "layer_transfer_counter", None) is None:
            incoming.register_layer_transfer_counter(
                counter, getattr(outgoing, "_mamba_transfer_frame", None)
            )


def assert_req_pool_identity(scheduler: Any) -> None:
    """Every holder names ONE pool after the cutover, or the seam stops.

    Called at the END of the cutover, once, on every rank. It is not a
    re-implementation of the rebind: the rebind moves what it knows about, and
    this refuses when something it does NOT know about is still divergent --
    a fifth holder, a cache rebuilt from the outgoing pool, a hook that was
    added to one pool type and not the other.

    Loud on purpose. The divergence it looks for does not fail on its own:
    both pools hold the same number of rows, so a wrong-pool read lands in
    range and returns another phase's row contents (#1201). A boot that stops
    here is strictly better than a phase that answers wrongly.
    """
    pool = getattr(scheduler, "req_to_token_pool", None)
    if pool is None:
        raise ReqPoolRebindRefused(
            f"{LOG_PREFIX} the scheduler has no request pool after the cutover"
        )

    tree_cache = getattr(scheduler, "tree_cache", None)
    if tree_cache is not None:
        cache_pool = getattr(tree_cache, "req_to_token_pool", None)
        if cache_pool is not None and cache_pool is not pool:
            raise ReqPoolRebindRefused(
                f"{LOG_PREFIX} #1201 the tree cache still names a different "
                f"request pool than the scheduler after the cutover "
                f"(cache binding={getattr(cache_pool, 'binding_tag', '<untagged>')}, "
                f"scheduler binding={getattr(pool, 'binding_tag', '<untagged>')}). "
                "Both pools hold the same row count, so this does not fail on "
                "its own: the KV free path would read the wrong pool's rows "
                "and hand those indices to the allocator."
            )

        if hasattr(pool, "bind_tree_cache"):
            back = getattr(pool, "tree_cache", None)
            if back is not tree_cache:
                raise ReqPoolRebindRefused(
                    f"{LOG_PREFIX} #1201 the incoming request pool's tree-cache "
                    f"back-reference is {back!r}, not the scheduler's tree cache. "
                    "A REQUIRED mamba allocation on this pool cannot evict "
                    "cached checkpoints before failing (#581/#773)."
                )
