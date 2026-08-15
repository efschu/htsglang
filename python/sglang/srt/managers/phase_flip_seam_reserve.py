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
being subtracted from. Writing it with the ADDITIVE pool-independent floor
``A`` (the weights-arena tail), the ALTERNATIVE pool-independent floor ``F``
(the drafter's restore) and the per-row coefficient ``a``::

    staging(T) = A + max(F, a*T)      T*cell + staging(T) <= R'

TWO FLOORS, BECAUSE THEY COMBINE DIFFERENTLY, and the split is a measurement
rather than a refinement (#656, MERGE-R9 12.4). This was one flat
``max(F, a*T)`` with ``F = max(arena tail, draft restore)``, mirroring the
runtime's own ``_staging_bytes``, on the argument that the wave staging and
the re-commits belong to different instants of the seam and never coexist.

That argument holds for the DRAFTER and fails for the ARENA TAIL:
``stacks.refill`` is a PRE-cutover function, so its commit lands while the
wave state is still outstanding, while rung 2's restore runs inside
``_cutover`` after the waves' buffers are dead. The stage walk of one
``tp_to_pp`` cutover settles it -- entry 2464 MiB free, a 1386 MiB wave peak,
the refill reached at 1250 with 1214 MiB still outstanding, and the refill's
own 238 MiB taking the card to 1012: twelve MiB under a corridor law that
``max(1386, 238)`` called 54 MiB clear.

Setting ``A = 0`` reproduces the previous arithmetic byte for byte, which is
exactly what a record written before the split reads back as.

Two regimes, and exactly one of them holds for any given ``R'``::

    A. the floor binds  ->  T = (R' - A - F) / cell,   while a*T <= F
    B. the slack binds  ->  T = (R' - A) / (cell + a), while a*T >  F

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

#: Margin, in MiB, the solver keeps between the measured position and the seam
#: floor. WHY A MARGIN AT ALL (#656 R4, boot K3): ``seam_allowed_tokens``
#: returns the largest T with ``have(T) >= need(T)``, i.e. it targets EQUALITY,
#: so the boot it sizes lands exactly ON its own floor and has nothing to
#: absorb second-order error. Boot K3 derived 610942 and then re-measured
#: rank2 at 1430 MiB against a 1455 MiB floor -- 25 MiB the wrong side of its
#: own gate, logging ``CANNOT FUND ITS OWN FLIP`` while completing all 30
#: cutovers. The error is allocator granularity and arena drift along the pool
#: axis, and it is one-sided in the dangerous direction.
#:
#: THIS IS NOT THE GATE'S C20 ENTRY MARGIN. The flip gate wants 512 MiB of
#: entry headroom on top of the staging, but it satisfies that at flip time
#: from transient reclaim (measured: ``CORRIDOR-GUARD cleared on device 0:
#: ... reclaimed 136 MiB from [allocator-cache]``), so reserving the gate's
#: number here would charge the pool twice for one requirement -- roughly
#: 65k tokens on this rig's binding rank. What the SIZER owes is only the
#: measurement's own error bar, and 192 MiB is ~8x the largest observed
#: deviation.
ENV_MARGIN_MIB = "SGLANG_PHASE_FLIP_SEAM_MARGIN_MIB"
DEFAULT_MARGIN_MIB = 192

PROVENANCE_COLD = "cold"
PROVENANCE_STORED = "stored"
PROVENANCE_OVERRIDE = "override"
PROVENANCE_DISABLED = "disabled"
PROVENANCE_MALFORMED = "malformed"


@dataclass(frozen=True)
class SeamReserve:
    """What the seam costs this rank at rest, and where the number came from."""

    #: Pool-INDEPENDENT floor that is an ALTERNATIVE to the wave slack: the
    #: drafter's restore. Present with an empty live set and unaffected by
    #: draining. Rung 2's restore runs inside ``_cutover``, after the waves'
    #: buffers are dead, so it and the slack cannot both be resident -- which
    #: is why this one stays inside a ``max``.
    fixed_bytes: int = 0
    #: Pool-INDEPENDENT floor that is ADDED to whatever else the seam holds:
    #: the weights-arena tail (#656, MERGE-R9 12.4).
    #:
    #: SEPARATED FROM ``fixed_bytes`` BECAUSE THE TWO COMBINE DIFFERENTLY,
    #: and the measurement says so. ``stacks.refill`` is a PRE-cutover
    #: function, so its commit lands while the wave state is still
    #: outstanding; the stage walk of one tp_to_pp cutover
    #: (evidence-631/remediation-656/boot_m1.log, rank 1) entered at 2464 MiB
    #: free, reached the refill at 1250 with 1214 MiB outstanding, and the
    #: refill's own 238 MiB took the card to 1012 -- twelve MiB under the
    #: corridor law. Folded into one ``max`` the model predicted a 1250 MiB
    #: trough and could not see the breach.
    #:
    #: DEFAULT ZERO IS THE BACK-COMPAT PATH, exactly. A record written before
    #: this field existed carries ``fixed_bytes = max(arena, draft)`` and no
    #: arena entry, so it reads back as A=0 and the arithmetic below is
    #: byte-identical to what that record was written for. The next boot of
    #: that configuration re-measures and fills the field in -- the two-boot
    #: protocol this module already documents, carrying one more quantity.
    arena_fixed_bytes: int = 0
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
    #: The deepest CORRIDOR SHORTFALL this rank has been observed to make,
    #: in bytes: how far below the law its card went at the worst instant,
    #: recorded by the runtime's own corridor audit. ZERO until a boot
    #: actually measures one, so a rig that has never breached is sized
    #: exactly as before -- this term can only ever appear as a consequence
    #: of a measurement, which is what keeps it from becoming another
    #: constant carried between rigs (#656).
    corridor_shortfall_bytes: int = 0
    provenance: str = PROVENANCE_COLD
    written_at: Optional[str] = None
    detail: str = ""

    @property
    def total_fixed_bytes(self) -> int:
        """The seam's whole at-rest floor: the additive term plus the
        alternative one. What a caller wants when it needs ONE number for
        "the seam costs at least this much with an empty live set"."""
        return max(0, int(self.arena_fixed_bytes)) + max(0, int(self.fixed_bytes))

    @property
    def active(self) -> bool:
        return self.id_space > 0 and (
            self.fixed_bytes > 0 or self.arena_fixed_bytes > 0 or self.per_row_bytes > 0
        )


def seam_reserve_enabled() -> bool:
    raw = os.environ.get(ENV_ENABLE)
    if raw is None or raw == "":
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def seam_margin_bytes(reserve: Optional["SeamReserve"] = None) -> int:
    """Bytes the solver stands back from the measured position.

    TWO TERMS, and only the first is a constant:

    * the MEASUREMENT ERROR BAR (``DEFAULT_MARGIN_MIB``), which is what the
      sizer owes because ``seam_allowed_tokens`` targets equality;
    * the MEASURED CORRIDOR SHORTFALL for this rank, if any boot has ever
      recorded one. This is zero on a rig that has never breached, so it
      cannot travel between configurations as a number somebody picked.

    WHY THE SECOND TERM EXISTS (#656 acceptance, 2026-08-13). The sizer
    guarantees the seam is fundable AT REST. It says nothing about the
    level the card actually reaches while the seam RUNS, and on this rig
    that drawdown measured 1814-1852 MiB against a gate that assumes 512.
    Five cutovers entered through a gate with no objection and took the
    card to 886 MiB, 138 below the law. The shortfall is knowable only by
    measurement -- it is a property of the geometry, the load and the
    card -- so the honest term is the one the runtime writes down after it
    has seen it, and the two-boot protocol this file already documents is
    exactly the vehicle for carrying it to the next boot.

    ``SGLANG_PHASE_FLIP_SEAM_MARGIN_MIB`` still overrides BOTH terms, so a
    bring-up can pin the whole margin to one number.

    Read from the environment every call rather than cached: the same
    discipline as ``seam_reserve_enabled``, so a boot can be re-run with a
    different margin without a code change, and a test can set it per case.
    A malformed or negative value falls back to the default rather than
    disabling the margin -- the failure mode this exists to prevent is a
    zero-margin pool, so an unparseable override must not produce one.
    """
    raw = os.environ.get(ENV_MARGIN_MIB)
    if raw is not None:
        try:
            mib = int(raw)
        except (TypeError, ValueError):
            mib = -1
        if mib >= 0:
            return mib << 20
        # Malformed or negative: fall through to the derived margin rather
        # than to zero. The failure mode this exists to prevent is a
        # zero-margin pool, so an unparseable override must not produce one.
    measured = 0 if reserve is None else max(0, int(reserve.corridor_shortfall_bytes))
    return (DEFAULT_MARGIN_MIB << 20) + measured


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
            # Absent in records written before #656 MERGE-R9 12.4. Absent
            # means "this boot did not separate the two floors", and the
            # zero it defaults to reproduces that record's own arithmetic
            # exactly rather than guessing a split for it.
            arena_fixed_bytes=int(rec.get("arena_fixed_bytes", 0)),
            per_row_bytes=float(rec["per_row_bytes"]),
            have_bytes=int(rec.get("have_bytes", 0)),
            id_space=int(rec.get("id_space", 0)),
            # Absent in records written before #656; absent means "never
            # measured", which is exactly the zero the term defaults to.
            corridor_shortfall_bytes=int(rec.get("corridor_shortfall_bytes", 0)),
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
    arena_fixed_bytes: int = 0,
) -> Optional[str]:
    """Persist this boot's measurement for the next one. Never raises.

    Written at the END of the flip boot, where both layouts, the arena
    carrier and the drafter all exist and the numbers are exact rather than
    predicted.
    """
    path = record_path(server_args, world_rank)
    # PRESERVED, NOT OVERWRITTEN. The shortfall is written by a different
    # event (the runtime's corridor audit, mid-run) than this measurement
    # (end of the flip boot), so a boot that re-measures its seam must not
    # silently discard a breach an earlier boot paid to learn about.
    prior_shortfall = 0
    try:
        with open(path) as fh:
            prior_shortfall = int(json.load(fh).get("corridor_shortfall_bytes", 0))
    except Exception:
        prior_shortfall = 0
    payload = {
        "fixed_bytes": int(fixed_bytes),
        "arena_fixed_bytes": int(arena_fixed_bytes),
        "per_row_bytes": float(per_row_bytes),
        "have_bytes": int(have_bytes),
        "id_space": int(id_space),
        "corridor_shortfall_bytes": max(0, prior_shortfall),
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


def record_corridor_shortfall(
    server_args, world_rank: int, shortfall_bytes: int
) -> Optional[int]:
    """Persist a MEASURED corridor shortfall for this rank. Never raises.

    Called by the runtime's corridor audit when the continuous minimum on
    this rank's card has gone below the law. The next boot of the same
    configuration reads it back through :func:`read_seam_reserve` and the
    solver stands that much further back -- the two-boot protocol this file
    already documents, carrying one more measured quantity.

    A MONOTONIC MAXIMUM, deliberately. A shallower breach later does not
    mean the deeper one cannot recur; the pool must be sized for the worst
    instant that has ever been seen, which is the same rule the corridor law
    itself is stated with. Returns the value now on record, or None if it
    could not be written.
    """
    want = max(0, int(shortfall_bytes))
    if want <= 0:
        return None
    path = record_path(server_args, world_rank)
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except Exception:
        # No record yet: a cold boot has nothing to append to, and inventing
        # one here would hand the next boot a seam floor nobody measured.
        return None
    prior = int(rec.get("corridor_shortfall_bytes", 0))
    if want <= prior:
        return prior
    rec["corridor_shortfall_bytes"] = want
    rec["corridor_shortfall_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(
            "%s could not record the shortfall in %s: %s", LOG_PREFIX, path, e
        )
        return None
    logger.warning(
        "%s recorded a MEASURED corridor shortfall of %d MiB for rank %d "
        "(was %d MiB). The next boot of this configuration will stand that "
        "much further back when it sizes the pool, which is the two-boot "
        "protocol carrying one more measured quantity. This term is zero "
        "until a breach is actually observed, so it can never travel to "
        "another rig as a number somebody picked.",
        LOG_PREFIX,
        want >> 20,
        int(world_rank),
        prior >> 20,
    )
    return want


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
    arena_fixed_bytes: int = 0,
) -> int:
    """The largest T with ``T*cell + staging(T) <= R'``.

    ``R'`` is ``corridor_relaxed_bytes``: the KV byte budget plus whatever
    headroom the sizer already leaves, minus the corridor law -- i.e. every
    byte that could become KV if the corridor were the only thing to respect.

    ``staging(T) = A + max(F, a*T)``, and the SHAPE is the #656 MERGE-R9 12.4
    correction rather than an elaboration:

    * ``A`` (``arena_fixed_bytes``) is the weights-arena tail, committed by
      ``stacks.refill`` at the PRE-cutover seam and therefore held WHILE the
      wave state is still outstanding. It ADDS.
    * ``F`` (``fixed_bytes``) is the drafter's restore, which runs inside
      ``_cutover`` after the waves' buffers are dead. It and the slack cannot
      coexist, so it stays an ALTERNATIVE.

    The whole floor used to be one ``max`` against the slack, on the argument
    that the peaks belong to different instants and summing them "would model
    a peak that does not occur". The stage walk falsified that for the arena
    tail and only for the arena tail: one tp_to_pp cutover entered at 2464 MiB
    free, reached the refill at 1250 with 1214 MiB outstanding, and the
    refill's 238 MiB took it to 1012 -- under the 1024 MiB law that
    ``max(1214, 238)`` said was 226 MiB clear.

    ``A = 0`` restores the previous arithmetic exactly, which is what a
    pre-#656 record reads back as.

    Two regimes, one of which always holds:

      A. the floor binds  ->  T = (R' - A - F) / cell,   valid while a*T <= F
      B. the slack binds  ->  T = (R' - A) / (cell + a), valid while a*T >  F

    Both are exact; the branch is chosen by checking the candidate against
    its own condition, so there is no iteration and no tolerance.
    """
    cell = int(cell_bytes)
    if cell <= 0:
        return 0
    R = max(0, int(corridor_relaxed_bytes))
    F = max(0, int(fixed_bytes))
    A = max(0, int(arena_fixed_bytes))
    a = max(0.0, float(per_row_bytes))

    t_floor = max(0, (R - A - F) // cell)
    if a * t_floor <= F:
        return int(t_floor)
    return int(max(0, (R - A) // (cell + a)))


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


def seam_allowed_tokens(
    cell_bytes: int,
    reserve: "SeamReserve",
    *,
    abandon_is_survivable: bool = False,
) -> int:
    """The largest id space whose seam this rank can still fund.

    ``abandon_is_survivable`` says what a refused seam COSTS, and it is the
    difference between paying for the guarantee with a smaller pool and
    refusing to. Under strict purity a refused ``tp_to_pp`` means prefill
    never runs (boot E held the corridor and served nothing), so the pool
    pays. Where the other layout may do the work -- ``prefill_in_tp`` --
    the refusal is a runtime abandon that costs one flip, and the corridor
    law wins instead. Derived from the purity mode, never configured
    separately: it is a consequence of the boot, not a preference.

    ANCHORED ON A MEASUREMENT, NOT ON A MODEL OF THE SIZER. The previous
    boot recorded, at a known id space ``T_m``, how many bytes were actually
    spendable above the corridor law (``have_m``). Every KV row given back
    returns ``cell`` bytes to that pool, so along the pool axis::

        have(T) = have_m + (T_m - T) * cell
        need(T) = A + max(F, a*T)

    and the answer is the largest T with ``have(T) >= need(T)``:

        A. the floor binds  ->  T <= T_m + (have_m - A - F) / cell
        B. the slack binds  ->  T <= (have_m + T_m*cell - A) / (cell + a)

    ``A`` is the ADDITIVE floor -- the weights-arena tail, committed at the
    pre-cutover seam while the wave state is still outstanding (#656,
    MERGE-R9 12.4, and ``solve_pool_tokens`` for the walk that shows it).
    ``F`` is the drafter's restore and stays an alternative to the slack.
    ``A = 0`` is exactly the pre-#656 arithmetic, which is what a record
    written before the split reads back as.

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
    # The margin comes off the MEASURED POSITION, not off the floor. In the
    # floor-bound regime the two are algebraically identical; in the
    # slack-bound regime only this one has any effect, because ``F`` appears
    # there solely in the regime test. Taking it here therefore means the same
    # thing on both branches: "stand this far back from where we measured".
    have_m = max(0, int(reserve.have_bytes) - seam_margin_bytes(reserve))
    t_m = int(reserve.id_space)
    F = max(0, int(reserve.fixed_bytes))
    A = max(0, int(reserve.arena_fixed_bytes))
    a = max(0.0, float(reserve.per_row_bytes))

    t_floor = t_m + (have_m - A - F) // cell
    if a * max(0, t_floor) <= F:
        solved = int(max(0, t_floor))
    else:
        solved = int(max(0, (have_m + t_m * cell - A) // (cell + a)))
    if not abandon_is_survivable:
        # An abandoned seam is FATAL here, so the pool must pay for the
        # guarantee. This is boot E's lesson and it is not being unlearned.
        return solved
    return _clamped_to_the_law(solved, t_m, have_m, cell)


def _clamped_to_the_law(solved: int, t_m: int, have_m: int, cell: int) -> int:
    """The reserve may SPEND slack above the corridor law. It may not MAKE it.

    ``have(T) = have_m + (t_m - T) * cell`` is the whole reason this solver
    can answer at all, and it says plainly where the bytes come from: every
    token cut off the id space turns into free VRAM. So a solve that returns
    ``T < t_m`` is not finding memory, it is MANUFACTURING free VRAM by
    making the KV pool permanently smaller.

    That is legitimate only while it is spending slack the instance already
    had. Once ``have_m`` is exhausted -- i.e. once the measured position sits
    ON the law -- any further cut pushes free VRAM ABOVE the law and holds it
    there for the life of the instance.

    THE CORRIDOR LAW HAS TWO HALVES AND THIS BROKE THE SECOND ONE. The law is
    "~1024 MiB free per card, and best-filled": 2-3 GiB too much free is the
    same defect as too little, because those bytes buy no tokens and serve no
    request. Measured on this rig 2026-08-15, one boot, one config:

        t_m = 590000, have_m = 0 MiB spendable  ->  solved 486403
        at-rest free per card: 1139 / 1462 / 2279  ->  3153 / 4604 / 3977

    The instance was ON the law and the reserve moved it ~2 GiB per card above
    it, permanently, to guarantee a seam could be funded at an occupancy the
    pool now could not even reach. The guarantee is not worth a pool: an
    unfundable seam is a runtime ABANDON, which is free, unanimous and leaves
    every request intact, while the bytes it was bought with are gone for the
    whole boot.

    So the cut stops at the measured position. When that is not enough to
    fund the seam the caller is told, loudly, that flips may abandon at this
    pool size -- which is a true statement about a serving instance, not a
    reason to shrink one.
    """
    if solved >= t_m or t_m <= 0:
        return solved
    if have_m <= 0:
        logger.warning(
            "%s the seam wants a SMALLER POOL than the corridor law already "
            "pays for: solved %d tokens against a measured position of %d, "
            "with 0 MiB spendable above the law. Cutting there would hold "
            "roughly %.0f MiB per card free ABOVE the law for the life of "
            "the instance, and free VRAM above the law buys no tokens. The "
            "id space STAYS at %d. A seam that cannot be funded at this size "
            "abandons at runtime -- free, unanimous, every request intact.",
            LOG_PREFIX,
            solved,
            t_m,
            (t_m - solved) * cell / (1 << 20),
            t_m,
        )
        return int(t_m)
    # Slack above the law exists: spend it, and no more than it.
    spendable_tokens = int(have_m // cell)
    floor_at_law = max(0, t_m - spendable_tokens)
    if solved >= floor_at_law:
        return solved
    logger.warning(
        "%s the seam's cut is bounded by the corridor law: solved %d tokens, "
        "but only %d MiB sits above the law, which is %d tokens' worth. The "
        "id space stops at %d instead. Cutting further would manufacture free "
        "VRAM above the law rather than spend slack the instance had.",
        LOG_PREFIX,
        solved,
        have_m // (1 << 20),
        spendable_tokens,
        floor_at_law,
    )
    return int(floor_at_law)


def seam_adjusted_budget_bytes(
    budget_bytes: int,
    cell_bytes: int,
    reserve: "SeamReserve",
    *,
    abandon_is_survivable: bool = False,
) -> Tuple[int, int]:
    """(new_budget_bytes, allowed_tokens). Never GROWS the budget.

    Unchanged when there is nothing to charge -- a cold record, a disabled
    term, or a configurator with no single per-token cell -- so every
    non-flip boot and every first boot is byte-identical.
    """
    if not reserve.active or int(cell_bytes) <= 0:
        return int(budget_bytes), 0
    allowed = seam_allowed_tokens(
        cell_bytes, reserve, abandon_is_survivable=abandon_is_survivable
    )
    return min(int(budget_bytes), allowed * int(cell_bytes)), allowed


def _worst_case_fixed_bytes(runtime, direction: str) -> Tuple[int, int]:
    """(arena_tail, draft_restore) the seam WILL have to commit, not the
    amount pending right now.

    MEASURED IN THE WRONG PHASE IS MEASURED WRONG (boot F, 2026-08-12). Both
    of the runtime's live accessors are state readings:
    ``pending_tail_bytes`` is ``want - committed``, and
    ``pending_restore_bytes`` returns 0 unless the drafter is CURRENTLY
    spilled. At the first round the instance sits in its boot phase (PP) with
    the arena fully committed and nothing spilled, so both read ZERO -- and a
    reserve of zero is exactly the sizing that wedged boot E, whose refusal
    named 464 MiB. That 464 MiB is the tail rung 3 releases on ENTERING TP,
    which is a state this measuring point never sees.

    So the sizing term is the commit the seam would face from the OTHER
    phase: the arena span above the TP layout (what rung 3 releases and the
    tp->pp leg must take back) and the drafter's whole payload (what rung 2
    releases and the pp->tp leg must take back). Both are static layout
    quantities, which is what makes them safe to read at any time.
    """
    scheduler = getattr(runtime, "_census_scheduler", None)
    stacks = getattr(scheduler, "phase_flip_stacks", None) if scheduler else None
    if stacks is None:
        return 0, 0

    arena_tail = 0
    if direction == "tp_to_pp":
        try:
            # What rung 3 releases in TP = the arena span the PP layout needs
            # above what the TP layout does. Zero on a rank whose TP layout is
            # the larger one, which is where the "PP is always bigger"
            # assumption killed three ranks at the first flip.
            high_water = int(stacks.refill_high_water_bytes())
            arena_tail = max(0, high_water - int(stacks.layout_tp.total_bytes))
        except Exception:
            arena_tail = 0

    draft_restore = 0
    if direction == "pp_to_tp":
        try:
            from sglang.srt.managers.phase_flip_spill import carrier_of

            carrier = carrier_of(getattr(stacks, "draft_worker", None))
            if carrier is not None:
                draft_restore = int(carrier.payload_bytes)
        except Exception:
            draft_restore = 0
    return arena_tail, draft_restore


def measure_at_rest(runtime) -> Tuple[int, int, float, str]:
    """(arena_fixed_bytes, fixed_bytes, per_row_bytes, detail) for this boot.

    FOUR VALUES SINCE #656 MERGE-R9 12.4, not three. The arena tail and the
    drafter's restore used to be folded into one ``max`` here, matching the
    runtime's own; they are returned separately now because they combine
    differently -- see :class:`SeamReserve` for which is which and why.

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
    # The id space the sizer solves for: the PP allocator's capacity, which is
    # what every "T tokens" in this module means.
    scheduler = getattr(runtime, "_census_scheduler", None)
    id_space = max(1, int(getattr(scheduler, "max_total_num_tokens", 0) or 0))
    fixed = 0
    arena_fixed = 0
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
        arena, draft = _worst_case_fixed_bytes(runtime, direction)
        # SPLIT, NOT MAXED (#656, MERGE-R9 12.4). ``d_fixed = max(arena,
        # draft)`` was the sizer's mirror of the runtime's flat max(), and it
        # inherited the same defect: whichever of the two was smaller vanished
        # from the floor entirely. The arena tail is committed at the
        # PRE-cutover seam and is therefore additive against everything else
        # the seam holds; the drafter's restore is not. Kept as two numbers so
        # the solver can combine them the way the walk says.
        d_fixed = draft
        # NORMALISED BY THE ID SPACE, NOT BY src.num_rows (boot F).
        #
        # The sizer's T is the GLOBAL pool -- the id space every rank shares.
        # ``src.num_rows`` is this rank's PHYSICAL row count, and under the TP
        # layout that is its token SHARE of the id space, so tp_to_pp divided
        # by roughly T/3 and reported a coefficient ~3x too large (measured on
        # boot F: 5393.8 B/row against a pp_to_tp reading of 2360.7 for the
        # same 1396 MiB of slack). Dividing both directions by the id space
        # makes the two comparable and makes the number mean what the sizer
        # multiplies it by.
        d_per_row = float(slack) / float(id_space)
        fixed = max(fixed, d_fixed)
        arena_fixed = max(arena_fixed, arena)
        per_row = max(per_row, d_per_row)
        mib = 1 << 20
        parts.append(
            f"{direction}: staging {total / mib:.0f} MiB now, seam commit "
            f"{(arena + d_fixed) / mib:.0f} MiB (arena tail {arena / mib:.0f} "
            f"ADDITIVE + draft restore {draft / mib:.0f} vs slack), wave "
            f"slack {slack / mib:.0f} MiB over an id space of {id_space} = "
            f"{d_per_row:.1f} B/token [this rank holds {int(src.num_rows)} rows]"
        )
    return arena_fixed, fixed, per_row, "; ".join(parts)


def _rung_fundable_for_seam(scheduler, arena_fixed: int, fixed: int) -> int:
    """Bytes the KV rung can put toward the seam, bounded by the FIXED floor.

    THE BOUND IS THE POINT. The rung's honest capacity is "everything above
    the live high-water mark", which at rest is nearly the whole pool -- and
    sizing against that would grow the id space without limit and then fail
    under a live set the measurement never saw. So it may cover at most the
    seam's FIXED floor: the weights-arena tail plus the drafter's restore.

    Those two are one-shot commits at the cutover, which is exactly the moment
    the rung can release rows and exactly the moment it takes them back again;
    on this rig they are also the dominant term (815 MiB and 1456 MiB of arena
    tail on the two 3080 ranks). The PER-ROW slack is deliberately NOT covered:
    it is held across the whole wave walk and it scales with the very pool this
    would be growing, which is the shape that runs away.

    ASKED AS A PREDICATE, NOT OF AN OBJECT, and that is an ordering fact rather
    than a preference. The relief is installed at the first corridor gate,
    which happens AFTER both the pool sizing and this measurement -- measured
    on this rig, the MEASURED line is five lines above "KV backing relief is
    available" in the same millisecond. Asking the scheduler for the object
    here returns None every time, and a term that is silently zero is worse
    than one that is absent. So the question asked is the one that can be
    answered now: WILL there be a rung able to pay
    (:func:`kv_backing_relief.rung_can_pay`, the same disqualifiers the
    provider applies), and if so the fixed floor need not also be held in idle
    VRAM.

    Returns 0 whenever no such rung will exist -- the env switch off, no
    allocator, a chunkless arena, unreadable geometry. Zero reproduces the
    previous sizing exactly, which is the direction a term that GROWS a pool
    has to fail in, and it is the can-fail arm: with
    ``SGLANG_KV_BACKING_RELIEF=0`` the reserve charges what it always charged.
    """
    from sglang.srt.managers.kv_backing_relief import rung_can_pay

    ceiling = max(0, int(arena_fixed)) + max(0, int(fixed))
    if ceiling <= 0:
        return 0
    try:
        can_pay = bool(rung_can_pay(scheduler))
    except Exception as e:
        logger.warning(
            "%s could not decide whether a KV rung will be able to pay at the "
            "seam (%s); sizing as if it cannot",
            LOG_PREFIX,
            e,
        )
        return 0
    if not can_pay:
        logger.info(
            "%s no KV rung will be able to return bytes at the seam, so the "
            "%.0f MiB fixed floor stays charged to the pool exactly as before",
            LOG_PREFIX,
            ceiling / (1 << 20),
        )
        return 0
    logger.info(
        "%s a KV rung will be able to return backing at the seam, so the "
        "%.0f MiB fixed floor (%.0f arena tail + %.0f draft restore) is "
        "counted as spendable instead of held free. The per-row slack stays "
        "charged: it is held across the whole wave walk and scales with the "
        "pool this term would grow.",
        LOG_PREFIX,
        ceiling / (1 << 20),
        max(0, int(arena_fixed)) / (1 << 20),
        max(0, int(fixed)) / (1 << 20),
    )
    return ceiling


def measure_and_record(scheduler, runtime) -> None:
    """Measure this boot's seam and leave the record for the next one.

    Never raises: a bookkeeping write must not be able to take down an
    instance that is otherwise serving.
    """
    if not seam_reserve_enabled():
        return
    try:
        arena_fixed, fixed, per_row, detail = measure_at_rest(runtime)
    except Exception as e:
        logger.warning("%s could not measure the seam at rest: %s", LOG_PREFIX, e)
        return

    mib = 1 << 20
    try:
        import torch

        free_bytes = int(torch.cuda.mem_get_info()[0])
    except Exception:
        free_bytes = 0
    law = _corridor_law_bytes()
    rung_fund = _rung_fundable_for_seam(scheduler, arena_fixed, fixed)
    # THE MEASURED POSITION. Taken with every unnamed post already resident
    # -- activation reserve, capture peak, arena, TP stack, carve-out -- so
    # the next boot's correction needs no model of any of them.
    #
    # #662: AND FREE VRAM IS NO LONGER THE ONLY THING THAT CAN PAY. The KV
    # relief rung returns unoccupied backing at the seam and takes it back
    # after the cutover, which is bytes arriving exactly when the commit needs
    # them. Counting only the free column is what made the pool shrink until
    # enough VRAM sat idle to cover the seam -- the ~12.7 GiB this ticket
    # exists to remove. See _rung_fundable_for_seam for what it may cover and
    # why that bound is the fixed floor and not a token more.
    have = max(0, free_bytes - law) + rung_fund
    id_space = max(0, int(getattr(scheduler, "max_total_num_tokens", 0) or 0))
    path = write_seam_reserve(
        scheduler.server_args,
        int(runtime._rank),
        fixed,
        per_row,
        detail,
        have_bytes=have,
        id_space=id_space,
        arena_fixed_bytes=arena_fixed,
    )
    logger.info(
        "%s MEASURED (rank %d): floor %.0f MiB (%.0f arena tail ADDITIVE "
        "+ %.0f draft restore vs the wave slack), %.1f B/row. %s. Recorded "
        "in %s for the next boot with this configuration.",
        LOG_PREFIX,
        int(runtime._rank),
        (arena_fixed + fixed) / mib,
        arena_fixed / mib,
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
        spendable = have
        # THE WHOLE at-rest floor, both terms. Checking the alternative
        # term alone would let a boot whose arena tail is the binding
        # cost report that it can fund a flip it cannot.
        if spendable < arena_fixed + fixed:
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
