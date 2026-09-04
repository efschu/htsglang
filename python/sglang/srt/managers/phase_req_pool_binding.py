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
    """

    size: int
    free: int
    escapees: int
    rows: tuple
    rids: tuple


def census_outgoing_req_pool(
    pool: Any, reqs: Iterable[Any] = ()
) -> ReqPoolCensus:
    """Count the rows the outgoing phase did not return.

    The pool's own arithmetic is the authority on HOW MANY (``free_slots``
    against ``size``); the request list is only used to NAME them, so a census
    taken with no request list still reports the right count and simply cannot
    say which requests they were.
    """
    size = int(pool.size)
    free = len(pool.free_slots)
    escapees = size - free
    rows = []
    rids = []
    for r in reqs or ():
        idx = getattr(r, "req_pool_idx", None)
        if idx is None:
            continue
        rows.append(idx)
        rids.append(getattr(r, "rid", "<unknown>"))
    return ReqPoolCensus(
        size=size,
        free=free,
        escapees=escapees,
        rows=tuple(rows),
        rids=tuple(rids),
    )


def _live_reqs(scheduler: Any) -> Sequence[Any]:
    """Every request the scheduler can still name, for the census's RIDS.

    Naming is best-effort by construction -- a row can outlive the object that
    held it, which is precisely the leak being measured -- so this never
    decides the count, only the labels.
    """
    out = []
    batch = getattr(scheduler, "running_batch", None)
    for r in getattr(batch, "reqs", None) or ():
        out.append(r)
    for r in getattr(scheduler, "waiting_queue", None) or ():
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
            f"(free={census.free}, rids={list(census.rids)}, "
            f"rows={list(census.rows)}). The drain is supposed to return "
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
