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
    #: Bytes that WERE spendable above the corridor law when this record was
    #: taken, and the id space they were measured at. The pair is what makes
    #: the correction need no model of the sizer's other posts: it is a
    #: measured position plus the slope along which moving the pool moves it.
    have_bytes: int = 0
    id_space: int = 0
    provenance: str = PROVENANCE_COLD
    written_at: Optional[str] = None
    detail: str = ""

    @property
    def active(self) -> bool:
        return self.id_space > 0 and (self.fixed_bytes > 0 or self.per_row_bytes > 0)


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
            have_bytes=int(rec.get("have_bytes", 0)),
            id_space=int(rec.get("id_space", 0)),
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
    have_bytes: int = 0,
    id_space: int = 0,
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
        "have_bytes": int(have_bytes),
        "id_space": int(id_space),
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


def seam_allowed_tokens(cell_bytes: int, reserve: "SeamReserve") -> int:
    """The largest id space whose seam this rank can still fund.

    ANCHORED ON A MEASUREMENT, NOT ON A MODEL OF THE SIZER. The previous
    boot recorded, at a known id space ``T_m``, how many bytes were actually
    spendable above the corridor law (``have_m``). Every KV row given back
    returns ``cell`` bytes to that pool, so along the pool axis::

        have(T) = have_m + (T_m - T) * cell
        need(T) = max(F, a*T)

    and the answer is the largest T with ``have(T) >= need(T)``:

        A. the floor binds  ->  T <= T_m + (have_m - F) / cell
        B. the slack binds  ->  T <= (have_m + T_m*cell) / (cell + a)

    This needs no term for the activation reserve, the capture peak, the
    arena, the TP stack or the invisible carve-out -- all of them are already
    inside ``have_m``, because it was measured with all of them resident.
    That is why it works on the pre-capture sizing path, where no headroom
    quantity exists to reason about (boot H: the post-capture path never ran,
    so a correction keyed to its headroom was never applied and the pool came
    back at 683150 unchanged).
    """
    cell = int(cell_bytes)
    if cell <= 0 or not reserve.active:
        return 0
    have_m = int(reserve.have_bytes)
    t_m = int(reserve.id_space)
    F = max(0, int(reserve.fixed_bytes))
    a = max(0.0, float(reserve.per_row_bytes))

    t_floor = t_m + (have_m - F) // cell
    if a * max(0, t_floor) <= F:
        return int(max(0, t_floor))
    return int(max(0, (have_m + t_m * cell) // (cell + a)))


def seam_adjusted_budget_bytes(
    budget_bytes: int,
    cell_bytes: int,
    reserve: "SeamReserve",
) -> Tuple[int, int]:
    """(new_budget_bytes, allowed_tokens). Never GROWS the budget.

    Unchanged when there is nothing to charge -- a cold record, a disabled
    term, or a configurator with no single per-token cell -- so every
    non-flip boot and every first boot is byte-identical.
    """
    if not reserve.active or int(cell_bytes) <= 0:
        return int(budget_bytes), 0
    allowed = seam_allowed_tokens(cell_bytes, reserve)
    return min(int(budget_bytes), allowed * int(cell_bytes)), allowed


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
