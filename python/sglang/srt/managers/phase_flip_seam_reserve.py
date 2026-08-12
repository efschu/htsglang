# SPDX-License-Identifier: Apache-2.0
"""#656: size the KV pool so the FLIP SEAM can still be paid for.

THE DEFECT THIS EXISTS FOR (boot E, 2026-08-12)
-----------------------------------------------
With the TP pool sized to the PP id space, the pool stopped being an operator
number and became whatever the VRAM backs. It backed 683150 tokens, held the
1024 MiB corridor with zero breaches for its whole life -- and produced no
tokens at all. On rank1::

    staging 464 MiB needed but only 444 MiB is spendable
    (driver free 1468 MiB, reserve 1024 MiB kept free)

The sizer had filled the card down to the corridor floor and left nothing for
the seam. A flip is not free: crossing between the layouts needs device bytes
that exist NOWHERE in the pool arithmetic -- the weights-arena tail the TP
phase released and the PP phase must re-commit, the spilled drafter's restore,
and the wave-boundary slack of the backing swap. Short by 20 MiB, every
``tp_to_pp`` flip abandoned; under strict phase purity prefill can only run in
PP; so the instance answered /health with 200 and served nothing.

A pool that cannot afford to leave its own layout is not capacity.

THE LAW
-------
    pool = VRAM - corridor - staging

with ``staging`` the seam's AT-REST requirement -- what the seam costs with an
empty live set, which is the floor no amount of draining can lower. Anything
above that floor is genuine occupancy pressure and correctly answered by the
runtime's existing "retry when occupancy drops".

THE FIXED POINT, AND WHY IT IS SOLVED RATHER THAN ITERATED
----------------------------------------------------------
``staging`` is not a constant: the wave-boundary slack is
``drift_layers x row_nbytes x pool_rows``, so it GROWS with the pool it is
being subtracted from. Writing it with the pool-independent floor ``F``
(arena tail, draft restore) and the per-row coefficient ``a``::

    staging(T) = max(F, a*T)          T*cell + staging(T) <= R'

MAX, NOT SUM. That is what ``_staging_bytes`` computes, and it says why: the
wave staging and the arena/draft re-commits belong to different instants of
the seam and never coexist, so summing them "would model a peak that does not
occur and would abandon flips that fit". Reserving the sum here would give
away roughly 24k tokens of permanent pool for a peak the hardware never sees.

Two regimes, and exactly one of them holds for any given ``R'``::

    A. the floor binds  ->  T = (R' - F) / cell,   while a*T <= F
    B. the slack binds  ->  T = R' / (cell + a),   while a*T >  F

Both are closed forms; the branch is chosen by testing a candidate against
its own condition. Iterating instead lands on the wrong side -- it charges
the per-row term at a pool size that will not exist once the term is charged,
so it settles BELOW the fixed point and silently gives away pool, on a ticket
whose whole subject is pool that was silently given away.

WHERE F AND a COME FROM: MEASUREMENT, NOT ARITHMETIC
----------------------------------------------------
Both are properties of the two weight LAYOUTS and the wave plan, and neither
exists until the TP stack is built -- which happens AFTER the PP pool is
sized. There is no ordering that makes them knowable in time, so they are not
computed here at all: they are MEASURED at the end of the flip boot, from the
runtime's own methods (so this number cannot drift from the number the gate
will actually check), written to a fingerprinted record, and read by the next
boot with the same configuration.

That is the house pattern, not a new one -- ``note_post_capture_leftover``
already sizes this rig's KV from the previous boot's measured leftover, for
the same reason and with the same two-boot convergence.

A COLD RECORD IS TODAY'S BEHAVIOUR, EXACTLY. No record, no correction: the
first boot of a new configuration sizes as it does now, measures, and says so
in the log. The second boot is the one that is seam-safe. This is stated
loudly rather than smoothed over, because a capacity number that depends on
an on-disk record from a previous boot is a harness trap when it is silent
(#188 records what that cost the last time it was silent).

NOT A SAFETY MARGIN. F is the seam's own arithmetic, and it is subtracted
once. Nothing here adds a percentage, a fudge, or a rounding-down on top --
choosing to run closer to the edge than the seam's own numbers allow is not
something this can express, which is the point.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP-SEAM-RESERVE"

#: Turn the whole term off, restoring boot-E sizing exactly. A VALUE of the
#: same mechanism rather than a second code path, so the off switch cannot
#: drift from the on switch.
ENV_ENABLE = "SGLANG_PHASE_FLIP_SEAM_RESERVE"
#: Override the measured floor with an absolute MiB value. For bringing up a
#: configuration whose record does not exist yet without spending a boot on
#: it -- and for pinning a reproducible capacity when a byte gate needs one.
ENV_FIXED_MIB = "SGLANG_PHASE_FLIP_SEAM_RESERVE_MIB"

PROVENANCE_COLD = "cold"
PROVENANCE_STORED = "stored"
PROVENANCE_OVERRIDE = "override"
PROVENANCE_DISABLED = "disabled"
PROVENANCE_MALFORMED = "malformed"


@dataclass(frozen=True)
class SeamReserve:
    """What the seam costs this rank at rest, and where the number came from."""

    #: Pool-INDEPENDENT floor: the arena tail re-commit and the drafter's
    #: restore. Present with an empty live set and unaffected by draining.
    fixed_bytes: int = 0
    #: Device bytes per KV POOL ROW: the wave-boundary backing slack, which
    #: scales with the pool and is therefore the term that makes this a fixed
    #: point rather than a subtraction.
    per_row_bytes: float = 0.0
    provenance: str = PROVENANCE_COLD
    written_at: Optional[str] = None
    detail: str = ""

    @property
    def active(self) -> bool:
        return self.fixed_bytes > 0 or self.per_row_bytes > 0


def seam_reserve_enabled() -> bool:
    raw = os.environ.get(ENV_ENABLE)
    if raw is None or raw == "":
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def record_path(server_args, world_rank: int) -> str:
    """One record per (configuration, rank).

    PER RANK, and not per configuration alone: the arena tail is
    ``max(0, pp_bytes - tp_bytes)`` on THIS rank's two layouts, and on the
    ship boot that is 1436 MiB on rank2 against 466 MiB on rank1 and 0 on
    rank0. A shared record would size every rank from one rank's seam --
    exactly the cross-stage trap ``_measured_kv_budget_cache_path`` grew its
    ``-stage{pp_rank}`` suffix to close.
    """
    from sglang.srt.uneven_perf import measured_kv_budget_cache_path

    base = measured_kv_budget_cache_path(server_args)
    root, _ext = os.path.splitext(base)
    return f"{root}-seam-rank{int(world_rank)}.json"


def read_seam_reserve(server_args, world_rank: int) -> SeamReserve:
    """This rank's seam floor for THIS boot. Never raises."""
    if not seam_reserve_enabled():
        return SeamReserve(provenance=PROVENANCE_DISABLED)

    override = os.environ.get(ENV_FIXED_MIB)
    if override not in (None, ""):
        try:
            mib = int(float(override))
        except ValueError:
            logger.warning(
                "%s %s=%r is not a number; ignoring the override",
                LOG_PREFIX,
                ENV_FIXED_MIB,
                override,
            )
        else:
            return SeamReserve(
                fixed_bytes=max(0, mib) * (1 << 20),
                provenance=PROVENANCE_OVERRIDE,
                detail=f"{ENV_FIXED_MIB}={mib} MiB",
            )

    path = record_path(server_args, world_rank)
    if not os.path.exists(path):
        return SeamReserve(provenance=PROVENANCE_COLD, detail=path)
    try:
        with open(path) as fh:
            rec = json.load(fh)
        return SeamReserve(
            fixed_bytes=int(rec["fixed_bytes"]),
            per_row_bytes=float(rec["per_row_bytes"]),
            provenance=PROVENANCE_STORED,
            written_at=rec.get("written_at"),
            detail=rec.get("detail", ""),
        )
    except Exception as e:
        logger.warning("%s record %s unreadable (%s); sizing cold", LOG_PREFIX, path, e)
        return SeamReserve(provenance=PROVENANCE_MALFORMED, detail=path)


def write_seam_reserve(
    server_args,
    world_rank: int,
    fixed_bytes: int,
    per_row_bytes: float,
    detail: str,
) -> Optional[str]:
    """Persist this boot's measurement for the next one. Never raises.

    Written at the END of the flip boot, where both layouts, the arena
    carrier and the drafter all exist and the numbers are exact rather than
    predicted.
    """
    path = record_path(server_args, world_rank)
    payload = {
        "fixed_bytes": int(fixed_bytes),
        "per_row_bytes": float(per_row_bytes),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detail": detail,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=1)
        # Atomic, so a boot that dies mid-write leaves the previous record
        # intact rather than a truncated one the next boot must discard.
        os.replace(tmp, path)
        return path
    except Exception as e:
        logger.warning("%s could not write %s: %s", LOG_PREFIX, path, e)
        return None


# ---------------------------------------------------------------------------
# The arithmetic. Pure, so it is testable without a GPU, a model or a boot.
# ---------------------------------------------------------------------------


def required_free_bytes(
    headroom_bytes: int, corridor_bytes: int, seam_fixed_bytes: int
) -> int:
    """Bytes that must remain free after the pool is placed.

    MAX, NOT SUM, against the existing headroom. ``headroom_bytes`` is the
    activation reserve the sizer already leaves; the corridor law and the
    seam floor must BOTH be satisfiable out of what is left free, and on this
    rig the existing headroom already covers the corridor with room to spare
    (boot E rested at 1468 MiB free against a 1024 MiB law). Adding the terms
    would reserve the corridor twice and give back most of the capacity this
    change exists to keep. Taking the max reserves exactly the binding one --
    on boot E's rank1, 1024 + 464 = 1488 against a 1468 MiB headroom, i.e.
    the 20 MiB that were actually missing.
    """
    return max(int(headroom_bytes), int(corridor_bytes) + int(seam_fixed_bytes))


def solve_pool_tokens(
    corridor_relaxed_bytes: int,
    cell_bytes: int,
    fixed_bytes: int,
    per_row_bytes: float,
) -> int:
    """The largest T with ``T*cell + staging(T) <= R'``.

    ``R'`` is ``corridor_relaxed_bytes``: the KV byte budget plus whatever
    headroom the sizer already leaves, minus the corridor law -- i.e. every
    byte that could become KV if the corridor were the only thing to respect.

    ``staging(T) = max(F, a*T)``, a MAX and not a sum, because that is what
    ``_staging_bytes`` computes: the wave staging and the arena/draft
    re-commits belong to different instants of the seam and never coexist
    (the runtime says so at its ``max()`` and explains why summing them
    "would model a peak that does not occur and would abandon flips that
    fit"). Reserving their sum here would give away pool for a peak the
    hardware never sees -- roughly 24k tokens on this rig's numbers.

    Two regimes, one of which always holds:

      A. the floor binds  ->  T = (R' - F) / cell,   valid while a*T <= F
      B. the slack binds  ->  T = R' / (cell + a),   valid while a*T >  F

    Both are exact; the branch is chosen by checking the candidate against
    its own condition, so there is no iteration and no tolerance.
    """
    cell = int(cell_bytes)
    if cell <= 0:
        return 0
    R = max(0, int(corridor_relaxed_bytes))
    F = max(0, int(fixed_bytes))
    a = max(0.0, float(per_row_bytes))

    t_floor = max(0, (R - F) // cell)
    if a * t_floor <= F:
        return int(t_floor)
    return int(max(0, R // (cell + a)))


def corridor_relaxed_bytes(
    budget_bytes: int, headroom_bytes: int, corridor_bytes: int
) -> int:
    """``R' = budget + headroom - corridor``.

    The sizer hands out ``budget_bytes`` of KV and leaves ``headroom_bytes``
    free. The corridor law needs ``corridor_bytes`` of that free, and
    everything else is fungible with KV -- so this is the honest ceiling to
    solve the seam against, rather than pretending the existing headroom is
    untouchable. On boot E's rank1 the headroom (1468 MiB) exceeded the law
    (1024 MiB) by 444 MiB, and it was 464 MiB of staging that had to come out
    of it: the 20 MiB shortfall falls straight out of this subtraction.
    """
    return int(budget_bytes) + int(headroom_bytes) - int(corridor_bytes)


def seam_adjusted_budget_bytes(
    budget_bytes: int,
    headroom_bytes: int,
    corridor_bytes: int,
    cell_bytes: int,
    reserve: "SeamReserve",
) -> Tuple[int, int]:
    """(new_budget_bytes, tokens). The whole adjustment, in one place.

    Returns the budget unchanged when there is nothing to charge -- a cold
    record, a disabled term, or a configurator with no single per-token cell
    -- so every non-flip boot and every first boot is byte-identical.
    """
    if not reserve.active or int(cell_bytes) <= 0:
        return int(budget_bytes), 0
    relaxed = corridor_relaxed_bytes(budget_bytes, headroom_bytes, corridor_bytes)
    tokens = solve_pool_tokens(
        relaxed, cell_bytes, reserve.fixed_bytes, reserve.per_row_bytes
    )
    # Never GROW the budget: the seam term exists to reserve, and a headroom
    # that happens to exceed the corridor law by more than the seam needs is
    # not an invitation to spend the difference on KV. Other posts (the
    # activation reserve, the capture peak) live in that headroom too and
    # this function knows nothing about them.
    return min(int(budget_bytes), tokens * int(cell_bytes)), tokens


def measure_at_rest(runtime) -> Tuple[int, float, str]:
    """(fixed_bytes, per_row_bytes, detail) for THIS boot's two layouts.

    Computed from the runtime's OWN methods against an EMPTY live set, so the
    number recorded here is the number the gate will check -- not a parallel
    model of it that can drift. ``build_phase_flip_transition`` already
    handles a zero-row slot tensor (every send/recv guard evaluates false),
    which is what makes "at rest" expressible rather than approximated.

    ``_arena_tail_bytes`` and ``_draft_restore_bytes`` read static layout and
    carrier counters; ``_flip_waves`` is a pure function of the layer map and
    the vector; ``src.num_rows`` is the pool's physical row CAPACITY, not the
    live count. So none of this needs a round to have run, allocates
    anything, or performs a collective.
    """
    import torch

    from sglang.srt.layers.dcp.phase_flip_plan import build_phase_flip_transition

    empty = torch.empty(0, dtype=torch.int64)
    fixed = 0
    per_row = 0.0
    parts = []
    for direction in ("pp_to_tp", "tp_to_pp"):
        src, dst = runtime._src_dst(direction)
        waves = runtime._flip_waves(direction)
        tr = build_phase_flip_transition(
            empty,
            runtime._map,
            runtime._n_layers,
            runtime._vec,
            runtime._rank,
            direction,
        )
        total = int(runtime._staging_bytes(tr, direction, src, dst, waves))
        slack = int(runtime._backing_slack_bytes(direction, src, dst, waves))
        arena = int(runtime._arena_tail_bytes(direction))
        draft = int(runtime._draft_restore_bytes(direction))
        rows = max(1, int(src.num_rows))
        d_fixed = max(arena, draft)
        d_per_row = float(slack) / float(rows)
        fixed = max(fixed, d_fixed)
        per_row = max(per_row, d_per_row)
        mib = 1 << 20
        parts.append(
            f"{direction}: staging {total / mib:.0f} MiB at rest "
            f"(arena tail {arena / mib:.0f}, draft restore {draft / mib:.0f}, "
            f"wave slack {slack / mib:.0f} over {rows} rows = "
            f"{d_per_row:.1f} B/row)"
        )
    return fixed, per_row, "; ".join(parts)


def measure_and_record(scheduler, runtime) -> None:
    """Measure this boot's seam and leave the record for the next one.

    Never raises: a bookkeeping write must not be able to take down an
    instance that is otherwise serving.
    """
    if not seam_reserve_enabled():
        return
    try:
        fixed, per_row, detail = measure_at_rest(runtime)
    except Exception as e:
        logger.warning("%s could not measure the seam at rest: %s", LOG_PREFIX, e)
        return

    mib = 1 << 20
    path = write_seam_reserve(
        scheduler.server_args, int(runtime._rank), fixed, per_row, detail
    )
    logger.info(
        "%s MEASURED (rank %d): floor %.0f MiB, %.1f B/row. %s. Recorded in "
        "%s for the next boot with this configuration.",
        LOG_PREFIX,
        int(runtime._rank),
        fixed / mib,
        per_row,
        detail,
        path,
    )

    # LIVE VERDICT for the boot that is running right now. The record helps
    # the NEXT boot; an operator watching this one needs to know today
    # whether its flips can be funded, because the failure mode is an
    # instance that answers /health and serves nothing (#656 boot E).
    try:
        import torch

        free_bytes, _total = torch.cuda.mem_get_info()
        law = _corridor_law_bytes()
        spendable = int(free_bytes) - law
        if spendable < fixed:
            logger.error(
                "%s (rank %d): THIS BOOT CANNOT FUND ITS OWN FLIP. The seam "
                "needs %.0f MiB at rest and only %.0f MiB is spendable above "
                "the %.0f MiB corridor law (driver free %.0f MiB). Under "
                "strict phase purity the layout that cannot be reached will "
                "not run its work at all. Re-boot to pick up the record just "
                "written, or set %s.",
                LOG_PREFIX,
                int(runtime._rank),
                fixed / mib,
                spendable / mib,
                law / mib,
                int(free_bytes) / mib,
                ENV_FIXED_MIB,
            )
        else:
            logger.info(
                "%s (rank %d): seam fundable at rest -- needs %.0f MiB, has "
                "%.0f MiB above the corridor law.",
                LOG_PREFIX,
                int(runtime._rank),
                fixed / mib,
                spendable / mib,
            )
    except Exception as e:  # pragma: no cover - diagnosis only
        logger.warning("%s live verdict unavailable: %s", LOG_PREFIX, e)


def _corridor_law_bytes() -> int:
    from sglang.srt.managers.vram_dial import corridor_law_floor_bytes

    return int(corridor_law_floor_bytes())


def describe(reserve: SeamReserve, path: str) -> str:
    """The boot line. Emitted on EVERY flip boot, cold included (#188)."""
    mib = 1 << 20
    if reserve.provenance == PROVENANCE_DISABLED:
        return (
            f"seam reserve OFF ({ENV_ENABLE}); the pool is sized without a "
            f"flip-seam term, which is the sizing that produced an instance "
            f"that held the corridor and served nothing (#656 boot E)."
        )
    if reserve.provenance == PROVENANCE_OVERRIDE:
        return (
            f"seam reserve {reserve.fixed_bytes / mib:.0f} MiB from "
            f"{reserve.detail} -- an operator value, not this boot's "
            f"measurement. The measurement still lands in {path} for the "
            f"next boot."
        )
    if reserve.provenance == PROVENANCE_STORED:
        return (
            f"seam reserve {reserve.fixed_bytes / mib:.0f} MiB fixed + "
            f"{reserve.per_row_bytes:.1f} B/row, MEASURED BY A PREVIOUS BOOT "
            f"({path}, written {reserve.written_at}). {reserve.detail} "
            f"Identical commands against a different record state will size "
            f"differently; pin --max-total-tokens for a reproducible pool."
        )
    return (
        f"seam reserve is COLD (no record at {path}): this boot sizes with NO "
        f"flip-seam term and may produce an instance whose flips cannot be "
        f"funded (#656 boot E). It measures the seam at the end of the flip "
        f"boot and writes the record, so the NEXT identical boot is the "
        f"seam-safe one. Watch for 'FLIP ABANDONED' on this boot."
    )
