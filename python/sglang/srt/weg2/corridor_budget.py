# SPDX-License-Identifier: Apache-2.0
"""#1257 -- the corridor law as a HARD constraint on group D's budget solve.

WHAT THIS IS NOT: a second budget bookkeeping.  ``launcher.budgets_from_dc``
stays the one and only producer of ``--rank-gpu-memory-mib``; this module is a
pure post-pass it calls with its own numbers, and it can only ever LOWER a
budget or leave it byte-identical.  Nothing here reads NVML, spawns a process
or writes a file.

THE LAW (memory ``vram-korridor-regel.md``): 819-1229 MiB NVML-free per card
under the awake group's load, verdict constant 1024.  ``free`` is NVML free v2
-- never ``total - used``, which returns free PLUS the driver carve-out
(+424/+518 MiB on this rig, measured boot weg2rg6, #1250) and would score a
breaching card as satisfied.  ``Card.reserved_mib`` is that carve-out and is
subtracted here for exactly that reason.

THE ARITHMETIC, and why every term is a MEASUREMENT rather than a constant.
The brief's formula is

    predicted_free = total - (budget + dormant_other + carve + load_transient)

and on this rig it overpredicts the 5090 by 825 MiB: at boot weg2sb5f the
launcher's own budget line gives ``32607 - 29352 - 1334 - 518 = 1403`` while
the front's corridor sampler measured **578 MiB idle / 474 under load**.  The
gap is D's unbudgeted awake consumption (NEXTN draft + verify-tree transients
and the decode graphs sit outside the ``--rank-gpu-memory-mib`` fraction --
the same effect ``launcher.D_OVERSHOOT_MIB`` already charges 489 MiB of).
Shipping the bare formula would print SATISFIED on a card the metal measures
550 MiB BELOW the law, which is the indicator law's exact failure mode.  So a
fifth term is carried, derived from the SAME paired sample:

    awake_residue = (total_s - budget_s - dormant_s - carve_s) - free_idle_s
    predicted_free(b) = total - b - dormant - carve - awake_residue - transient

By construction this reproduces the measured load free at the sample's own
budget (578 - 104 = 474 on the 5090; 1101 - 6 = 1095 and 1111 - 8 = 1103 on
the 3080s), so the instrument is calibrated against the only three metal
points that exist rather than asserted.

THE ONE ASSUMPTION, stated because it is not measured on this form: that a MiB
removed from a rank's budget returns a MiB to that card's free column.
htsglang ``aeac561711`` (#631) measured **0.318** on the Weg-1 form (-1200 MiB
of budget -> +382 MiB of free), so 1.0 is an OPTIMISTIC bound and a cut
computed from it is a LOWER BOUND on the cut actually needed.  The sample may
carry a measured ``budget_return_ratio`` with its own provenance; where it does
not, the line says ``return_ratio=1.0:ASSUMED`` and names #631's scope.

THE DANGER DIRECTION, and the refusal that guards it (``kein-bindender-rang``:
per-rank capacity is a free variable and the WORLD POOL is the thing worth
protecting).  Under the weighted owner rule the served context is

    C = min_r(P_r // v_r) * sum(v)      (model_runner_kv_cache_mixin.py:5033)

so lowering the budget of the rank that achieves that minimum shrinks C for
every rank at once.  This module therefore never lowers a budget without first
recomputing C, and where C would shrink it REFUSES BY NAME with both numbers
and keeps the budget byte-identical.  Trading world pool for corridor margin is
the operator's decision, not a solver's.

MISSING INPUTS ARE A REFUSAL, NOT A DEFAULT.  Four separate ways this pass can
fail to know something all land on the same answer -- the launcher's budgets,
byte for byte, and a line saying which one it was: no paired sample; a sample
that does not cover every card of this boot; a sample whose rows do not pair
with this boot's card ordinals (below); and a world pool this boot cannot
price (below).  A boot must never silently get a budget that was priced off
numbers nobody measured.

THE ROWS ARE PAIRED BY ORDINAL, SO THE PAIRING IS CHECKED.  Capacities are
ordered by THIS boot's card ordinals (by uuid), but the token vector is a
POSITIONAL list, so ``vec[i]`` only belongs to ``order[i]`` while the sample's
own ``world_rank`` column agrees with the ordinal.  NVML enumeration order can
shift between boots on this rig -- the project's own standing warning -- and a
shifted pairing would not fail loudly, it would compute a plausible pool and a
plausible binder for the wrong cards.  Hence an explicit check, and a refusal
when it does not hold.

THE W-CODE IS W54 AND WAS ENUMERATED, NOT PICKED.  The first draft of this
module used W52, which #1290 already holds at the base commit: ``front.py:560``
builds ``NO_ROUTE_NAME`` by concatenating that code with ``NO_ROUTE_MARKER``,
and the counter keys ``W52_Weg2NoServiceableRoute`` sit at ``front.py:1980``
and ``:2602``.  The collision census that exists to stop exactly this
(``test_weg2_wcode_uniqueness_1263.py``) did not see it: its pattern was
``W<nn>`` + whitespace + ``Weg2<Name>``, and BOTH of #1290's forms hide that
whitespace -- a quote follows the code in the concatenation, an underscore in
the counter key.  The census is hardened in the same commit that renumbers
this, because the blind instrument is the durable half of the defect and the
label is only this slice's instance of it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: The two W-codes this pass owns, enumerated against the (hardened) census:
#: W53 and W54 were free, W52 was NOT -- see the module docstring.  Written as
#: plain literals rather than a bare code concatenated with a marker constant,
#: because that concatenated form is precisely what hid #1290's claim from the
#: census for a whole slice.
UNPRICED_NAME = "W54 Weg2CorridorBudgetUnpriced"
#: RENUMBERED 2026-09-09 (#1257c): W53 was claimed on the base commit
#: ``1f837b8e17`` by #1291's ``W53 Weg2StoreHandbackFailed``
#: (``front.py:569``, counter ``W53_Weg2StoreHandbackFailed``), which landed
#: between this module's enumeration and its merge.  Both claims were real and
#: the census in ``test_weg2_wcode_uniqueness_1263.py`` sees both, so on the
#: merged tree they contradicted each other.  W55 was enumerated free against
#: the same census -- as was W56 below -- and the older claim keeps its number.
WOULD_BIND_NAME = "W55 Weg2CorridorBudgetWouldBind"
#: #1257c.  The floor for this card is the named ``UNMEASURED-FALLBACK`` and no
#: user reserve was asked for, so NOTHING PRICED IT and the cut does not
#: happen.  A verdict, not an actuation -- see the module docstring.
UNMEASURED_FLOOR_NAME = "W56 Weg2CorridorFloorUnmeasured"

#: The stated law, MiB.  KEPT ONLY AS THE FALLBACK'S VALUE (#1257c): the
#: verdict constant is now DERIVED per card by
#: ``managers.corridor_guard.corridor_floor_mib`` as ``measured transient peak
#: of the awake group + user reserve``, and this literal is what that
#: derivation returns when nothing measured the transient.  Imported from the
#: guard rather than repeated, because the guard is THE ONE DECLARATION and a
#: private copy here is the fourth one its own comment forbids.
from sglang.srt.managers.corridor_guard import (  # noqa: E402
    CORRIDOR_LAW_MIB,
    CorridorFloor,
    corridor_floor_mib,
)

#: ``budgets_from_dc`` floors every budget to a multiple of 8 MiB; a cut that
#: is also a multiple of 8 keeps that invariant without re-flooring.
BUDGET_ALIGN_MIB = 8

BYTES_PER_MIB = 1 << 20

#: Where the paired corridor/budget sample lives by default.  Evidence tree,
#: not the repo: it is a measurement of THIS rig, produced by the front's
#: corridor sampler plus that boot's own launcher and D-group log lines.
DEFAULT_SAMPLE_PATH = "/spinning/gpu-arb/weg2/corridor_budget_sample.json"


@dataclass(frozen=True)
class SampleCard:
    """One card's paired measurement, all six figures from one boot."""

    card_uuid: str
    name: str
    world_rank: int
    nvml_total_mib: int
    budget_mib: int
    dormant_other_mib: int
    driver_reserved_mib: int
    free_idle_mib: int
    free_load_mib: int
    profiled_tokens: int

    @property
    def load_transient_mib(self) -> int:
        """MiB the awake group takes on top of idle, MEASURED (idle - load)."""
        return int(self.free_idle_mib) - int(self.free_load_mib)

    @property
    def awake_residue_mib(self) -> int:
        """Unbudgeted awake consumption: what the budget line does not book."""
        booked = (
            int(self.nvml_total_mib)
            - int(self.budget_mib)
            - int(self.dormant_other_mib)
            - int(self.driver_reserved_mib)
        )
        return booked - int(self.free_idle_mib)


@dataclass(frozen=True)
class CorridorSample:
    provenance: str
    cell_size: int
    token_vector: Tuple[int, ...]
    cards: Tuple[SampleCard, ...]
    budget_return_ratio: Optional[float]
    budget_return_provenance: str

    @property
    def by_uuid(self) -> Dict[str, SampleCard]:
        return {c.card_uuid: c for c in self.cards}

    @property
    def tokens_per_mib(self) -> int:
        """Tokens of a rank's own capacity that one MiB of its budget buys."""
        return BYTES_PER_MIB // int(self.cell_size)


@dataclass(frozen=True)
class CorridorSolve:
    """Result of the pass: the budgets that ship, and one line per card."""

    budgets: Tuple[int, ...]
    lines: Tuple[str, ...]
    unpriced_reason: Optional[str]
    world_pool_before: Optional[int]
    world_pool_after: Optional[int]

    @property
    def changed(self) -> bool:
        return self.unpriced_reason is None and any(
            l.count("->") and " verdict=APPLIED" in l for l in self.lines
        )


def load_sample(path: Optional[str]) -> Tuple[Optional[CorridorSample], Optional[str]]:
    """Read the paired sample.  Returns ``(sample, None)`` or ``(None, why)``.

    Every failure is a NAMED reason, because "no corridor line in the log" must
    never be readable as "the constraint was satisfied"."""
    p = path or DEFAULT_SAMPLE_PATH
    if not p or not os.path.exists(p):
        return None, f"no corridor/budget sample at {p!r}"
    try:
        with open(p, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:  # malformed evidence is a refusal, not a default
        return None, f"corridor/budget sample {p!r} is unreadable: {exc}"

    try:
        cell = int(raw["cell_size"])
        vec = tuple(int(v) for v in raw["token_vector"])
        cards = tuple(
            SampleCard(
                card_uuid=str(c["card_uuid"]),
                name=str(c.get("name", "")),
                world_rank=int(c["world_rank"]),
                nvml_total_mib=int(c["nvml_total_mib"]),
                budget_mib=int(c["budget_mib"]),
                dormant_other_mib=int(c["dormant_other_mib"]),
                driver_reserved_mib=int(c["driver_reserved_mib"]),
                free_idle_mib=int(c["free_idle_mib"]),
                free_load_mib=int(c["free_load_mib"]),
                profiled_tokens=int(c["profiled_tokens"]),
            )
            for c in raw["cards"]
        )
    except Exception as exc:
        return None, f"corridor/budget sample {p!r} is incomplete: {exc}"

    if cell <= 0:
        return None, f"corridor/budget sample {p!r} has cell_size={cell}"
    if not cards:
        return None, f"corridor/budget sample {p!r} has no cards"
    if len(vec) != len(cards) or any(v <= 0 for v in vec):
        return None, (
            f"corridor/budget sample {p!r} token_vector {list(vec)} does not "
            f"match its {len(cards)} cards"
        )
    for c in cards:
        if c.profiled_tokens <= 0:
            return None, (
                f"corridor/budget sample {p!r}: card {c.card_uuid} has no "
                f"profiled token capacity"
            )
        if c.load_transient_mib < 0:
            return None, (
                f"corridor/budget sample {p!r}: card {c.card_uuid} measured MORE "
                f"free under load ({c.free_load_mib}) than idle ({c.free_idle_mib}) "
                f"-- the pair is not one boot"
            )
    ratio = raw.get("budget_return_ratio", None)
    ratio = float(ratio) if ratio is not None else None
    if ratio is not None and not (0.0 < ratio <= 1.0):
        return None, (
            f"corridor/budget sample {p!r} budget_return_ratio={ratio} is not in (0, 1]"
        )
    return (
        CorridorSample(
            provenance=str(raw.get("provenance", "(undeclared)")),
            cell_size=cell,
            token_vector=vec,
            cards=cards,
            budget_return_ratio=ratio,
            budget_return_provenance=str(raw.get("budget_return_provenance", "")),
        ),
        None,
    )


def _predicted_free_mib(
    total_mib: int,
    budget_mib: int,
    dormant_mib: int,
    carve_mib: int,
    sc: SampleCard,
) -> int:
    """NVML free-v2 this card is predicted to show under D-awake load."""
    return (
        int(total_mib)
        - int(budget_mib)
        - int(dormant_mib)
        - int(carve_mib)
        - sc.awake_residue_mib
        - sc.load_transient_mib
    )


def _capacities(
    budgets: Sequence[int], order: Sequence[SampleCard], tokens_per_mib: int
) -> List[int]:
    """P_r at these budgets: the sample's measured P_r moved by the delta.

    Exact, not modelled: the KV cell is the same on every rank of a D group
    (``cell_size=32768`` on all three ranks of weg2sb5f), so one MiB of a
    rank's budget is exactly ``tokens_per_mib`` tokens of that rank's own
    physical capacity."""
    return [
        max(0, sc.profiled_tokens + (int(b) - sc.budget_mib) * tokens_per_mib)
        for b, sc in zip(budgets, order)
    ]


def _world_pool(caps: Sequence[int], vec: Sequence[int]) -> Tuple[int, int]:
    """``C = min_r(P_r // v_r) * sum(v)`` and the binding rank index.

    The runtime's own rule, quoted from where it is implemented:
    ``model_runner_kv_cache_mixin.py:5020-5037``."""
    units = [int(p) // int(v) for p, v in zip(caps, vec)]
    binder = min(range(len(units)), key=lambda i: units[i])
    return units[binder] * int(sum(vec)), binder


def _resolved_world_pool(
    caps: Sequence[int],
) -> Tuple[Optional[int], Optional[str]]:
    """C after the runtime re-solves the vector from these capacities.

    Group D ships NO token vector (``d_token_vector_decision`` default), so the
    runtime installs ``partition_units(64, [P_r...])`` gcd-reduced after
    profiling -- boot weg2sb5f installed ``[17, 7, 8]`` that way.  Under a
    re-solved vector EVERY rank is near-binding by construction, so this is the
    stricter of the two pool predictions and it is why the pass may refuse a
    cut on a rank the fixed vector calls non-binding.

    Returns ``(pool, None)`` or ``(None, why)``.  The reason is not decoration:
    when this cannot be priced the CALLER MUST REFUSE THE CUT, because the
    fixed-vector check alone is not the check.  The runtime re-solves the
    vector at boot from the profiled capacities, and under a re-solved vector
    every rank is near-binding by construction -- so a cut that the fixed
    vector calls free can still shrink the pool the boot actually serves.
    Falling back to the fixed check here would be the exact shape this module
    exists to refuse: a constraint satisfied on paper.  Never a locally
    reinvented partition either -- the runtime's own helper or nothing."""
    try:
        from sglang.srt.distributed.utils import partition_units
    except Exception as exc:
        return None, (
            f"the runtime's own vector solver (sglang.srt.distributed.utils."
            f"partition_units) is not importable here: {exc}"
        )
    if any(c <= 0 for c in caps):
        return None, f"a rank has no capacity to partition: {list(caps)}"
    try:
        units = list(partition_units(64, list(caps)))
    except Exception as exc:
        return None, f"partition_units refused these capacities: {exc}"
    if not units or any(u <= 0 for u in units):
        return None, f"partition_units returned {units} for {list(caps)}"
    g = math.gcd(*units) if len(units) > 1 else units[0]
    vec = [u // max(1, g) for u in units]
    pool, _ = _world_pool(caps, vec)
    return pool, None


def installed_cut_line(
    cf: CorridorFloor,
    *,
    cut_mib: int,
    pool_before,
    pool_after,
    vector,
    binder,
) -> str:
    """THE CUT LINE, emitted only where a cut was INSTALLED (#1257c).

    It names the actuating term, both pool figures and the binder, so the
    operator can read what the margin cost without reconstructing it from
    three other lines.  ``kein-bindender-rang``: a cut that made a rank the
    binder never reaches here -- it is refused above by name -- and the price
    of the one that did happen is printed rather than hidden.

    REFUTER FIX 4 (2026-09-09): ``measured_peak=`` may only ever carry a
    number somebody MEASURED.  It printed ``cf.transient_mib``
    unconditionally, so a cut installed for ``reason=user-reserve`` on an
    UNMEASURED-FALLBACK floor read ``measured_peak=1024`` for a number nobody
    measured -- flatly contradicting the ``provenance=`` field printed beside
    it.  ``n/a`` where the transient is a fallback or a hand-set law, and
    ``basis=`` says what the number IS wherever there is one.

    A FUNCTION, not an f-string inside the solve, so the acceptance can
    exercise the REAL producer instead of a restatement of it.
    """
    return (
        f"WEG2-BUDGET corridor-constrained card={cf.card_uuid} INSTALLED "
        f"cut_mib={int(cut_mib)} reason={cf.reason} "
        f"floor={cf.mib} source={cf.source} reserve={cf.reserve_mib} "
        f"pool_before={pool_before} pool_after={pool_after} vector={vector} "
        f"binder={binder} "
        f"measured_peak={cf.transient_mib if cf.measured else 'n/a'} "
        f"basis={cf.basis} provenance={cf.provenance}"
    )


def floors_for_cards(
    cards: Sequence,
    group: str,
    user_reserve_mib=None,
) -> Dict[str, CorridorFloor]:
    """The derived corridor floor per card, keyed by uuid. ONE reader.

    #1257c.  ``law_mib`` used to be a DEFAULT PARAMETER of the solve -- a
    scalar 1024 that every card was graded against and that no caller ever
    passed.  It is now a required per-card input with provenance, because the
    two halves of the old number are per-card quantities: the awake group's
    measured transient peak differs by card (1055/1097/858 MiB on this rig at
    the S3 P ingest) and so does the user reserve.
    """
    return {
        c.uuid: corridor_floor_mib(
            c.uuid,
            group=group,
            user_reserve_mib=(
                int((user_reserve_mib or {}).get(c.uuid, 0))
                if isinstance(user_reserve_mib, dict)
                else int(user_reserve_mib or 0)
            ),
        )
        for c in cards
    }


def solve_corridor_budgets(
    cards: Sequence,
    budgets: Sequence[int],
    dormant_mib: Dict[str, int],
    sample: Optional[CorridorSample],
    unpriced_reason: Optional[str] = None,
    floors: Optional[Dict[str, CorridorFloor]] = None,
    group: str = "D",
) -> CorridorSolve:
    """Apply the corridor law to ``budgets``; lower only where it is free to.

    ``cards`` are the launcher's ``Card`` objects in ordinal order (they carry
    ``uuid``, ``name``, ``total_mib`` and ``reserved_mib``); ``budgets`` is what
    ``budgets_from_dc`` just produced; ``dormant_mib`` is THIS boot's measured
    dormant image of the other group, keyed by uuid -- the same dict the budget
    line was computed from.  Only the residue and the load transient come from
    the sample; everything else is live."""
    budgets = [int(b) for b in budgets]
    if floors is None:
        floors = floors_for_cards(cards, group)
    # The span, for the header lines only. A verdict is NEVER taken against
    # it: every card is graded against its own floor below.
    law_span = (
        f"{min(f.mib for f in floors.values())}-{max(f.mib for f in floors.values())}"
        if floors
        else str(CORRIDOR_LAW_MIB)
    )
    law_mib = law_span
    if sample is None:
        why = unpriced_reason or "no corridor/budget sample was supplied"
        return CorridorSolve(
            budgets=tuple(budgets),
            lines=(
                f"WEG2-BUDGET corridor-constrained group={group} "
                f"{UNPRICED_NAME}: {why}. law={law_mib} MiB. "
                f"REFUSED TO PRICE -- every budget stands byte-identical "
                f"({','.join(str(b) for b in budgets)} MiB). The corridor is "
                f"still the law; this boot simply cannot say whether it holds, "
                f"and a boot must never get a budget priced off numbers nobody "
                f"measured.",
            ),
            unpriced_reason=why,
            world_pool_before=None,
            world_pool_after=None,
        )

    by_uuid = sample.by_uuid
    missing = [c.uuid for c in cards if c.uuid not in by_uuid]
    if missing or len(cards) != len(sample.cards):
        why = (
            f"the sample covers {len(sample.cards)} card(s) and this boot has "
            f"{len(cards)}"
            if not missing
            else f"the sample has no row for card(s) {missing}"
        )
        return solve_corridor_budgets(
            cards, budgets, dormant_mib, None, why, floors, group
        )

    order = [by_uuid[c.uuid] for c in cards]
    # F3. `order` is keyed by uuid, but `vec` is POSITIONAL: vec[i] belongs to
    # order[i] only while the sample's own world_rank column agrees with this
    # boot's ordinal. NVML enumeration order can shift between boots on this
    # rig, and a shifted pairing fails SILENTLY -- it computes a perfectly
    # plausible pool and binder for the wrong cards. So it is checked, and a
    # mismatch is unpriced rather than papered over by re-sorting: if the two
    # boots disagree about which card is rank 0, the residues and transients
    # measured on the other boot are not this boot's either.
    mispaired = [
        f"ordinal {i} ({cards[i].uuid}) carries world_rank "
        f"{int(order[i].world_rank)} in the sample"
        for i in range(len(cards))
        if int(order[i].world_rank) != i
    ]
    if mispaired:
        return solve_corridor_budgets(
            cards, budgets, dormant_mib, None,
            "the sample's world_rank column does not pair with this boot's "
            "card ordinals, so its positional token_vector cannot be trusted "
            "against these cards: " + "; ".join(mispaired),
            floors, group,
        )
    vec = list(sample.token_vector)
    tpm = sample.tokens_per_mib
    ratio = sample.budget_return_ratio
    ratio_note = (
        f"return_ratio={ratio}:MEASURED({sample.budget_return_provenance or 'undeclared'})"
        if ratio is not None
        else (
            "return_ratio=1.0:ASSUMED (no measured budget->free pair on the Weg-2 "
            "form; htsglang aeac561711 #631 measured 0.318 on the Weg-1 form, a "
            "different form and NOT carried, so a cut below is a LOWER BOUND)"
        )
    )
    eff_ratio = ratio if ratio is not None else 1.0

    caps0 = _capacities(budgets, order, tpm)
    pool0, binder0 = _world_pool(caps0, vec)
    res0, res0_why = _resolved_world_pool(caps0)

    working = list(budgets)
    verdicts: List[Tuple[int, str, str]] = []  # (ordinal, verdict, extra)

    for i, card in enumerate(cards):
        sc = order[i]
        cf = floors[card.uuid]
        card_law = int(cf.verdict_floor_mib)
        free = _predicted_free_mib(
            card.total_mib, working[i], int(dormant_mib.get(card.uuid, 0)),
            int(getattr(card, "reserved_mib", 0)), sc,
        )
        if free >= card_law:
            verdicts.append((i, "SATISFIED", f"margin_mib={free - card_law}"))
            continue
        # #1257c THE ACTUATION GATE, and it is the whole user decision in one
        # branch.  A floor whose transient nobody measured, on a card whose
        # operator asked for no reserve, is a VERDICT ONLY: it prints, it says
        # the card is below it, and it does not spend a single MiB of KV pool
        # to fix that.  Cutting a budget against 1024 MiB that nobody priced is
        # the same defect this module already refuses four other ways -- "a
        # boot must never silently get a budget that was priced off numbers
        # nobody measured" -- applied to the floor itself.
        if not cf.actuates:
            verdicts.append(
                (i, "REFUSED-UNMEASURED-FLOOR",
                 f"{UNMEASURED_FLOOR_NAME} shortfall_mib={card_law - free} "
                 f"floor={cf.mib} source={cf.source} reserve={cf.reserve_mib} "
                 f"({cf.provenance}); nothing priced this floor, so it is a "
                 f"verdict and not a cut -- the budget stands byte-identical. "
                 f"Measure the awake group's transient "
                 f"(scripts/vram_ledger/probe_activation.py ingest) or pass "
                 f"--rank-user-reserve-mib to make it actuate")
            )
            continue
        need = card_law - free
        cut = int(math.ceil(need / eff_ratio))
        cut = int(math.ceil(cut / BUDGET_ALIGN_MIB)) * BUDGET_ALIGN_MIB
        trial = list(working)
        trial[i] = working[i] - cut
        if trial[i] <= 0:
            verdicts.append(
                (i, "REFUSED-IMPOSSIBLE",
                 f"shortfall_mib={need} cut_mib={cut} would take the budget to "
                 f"{trial[i]} MiB")
            )
            continue
        caps_t = _capacities(trial, order, tpm)
        pool_t, binder_t = _world_pool(caps_t, vec)
        res_t, res_t_why = _resolved_world_pool(caps_t)
        if res0 is None or res_t is None:
            # F2. The danger direction is UNPRICEABLE, so the cut does not
            # happen. Degrading to the fixed-vector check here would apply a
            # cut whose effect on the pool the boot actually serves nobody
            # computed -- the module's own law (missing inputs are a refusal)
            # applied to the one input that is derived rather than read.
            verdicts.append(
                (i, "REFUSED-UNPRICED-POOL",
                 f"{UNPRICED_NAME} shortfall_mib={need} cut_mib={cut} "
                 f"re-solved world pool unpriceable "
                 f"({res0_why or res_t_why}); the fixed-vector pool alone is "
                 f"NOT the check -- the runtime re-solves the vector at boot "
                 f"and under a re-solved vector every rank is near-binding, so "
                 f"the budget stands byte-identical rather than take a cut "
                 f"whose world-pool cost nobody priced")
            )
            continue
        shrinks_fixed = pool_t < pool0
        shrinks_resolved = res_t < res0
        if shrinks_fixed or shrinks_resolved:
            which = []
            if shrinks_fixed:
                which.append(f"fixed-vector {pool0}->{pool_t}")
            if shrinks_resolved:
                which.append(f"re-solved-vector {res0}->{res_t}")
            verdicts.append(
                (i, "REFUSED-WOULD-BIND",
                 f"{WOULD_BIND_NAME} shortfall_mib={need} "
                 f"cut_mib={cut} would_be_free_mib={free + int(cut * eff_ratio)} "
                 f"would_shrink_world_pool[{'; '.join(which)}] "
                 f"binder_after={binder_t}. The corridor margin on this card is "
                 f"only buyable with world context; kein-bindender-rang leaves "
                 f"that trade to the operator, so the budget stands unchanged")
            )
            continue
        working[i] = trial[i]
        verdicts.append(
            (i, "APPLIED",
             f"shortfall_mib={need} cut_mib={cut} world pool unchanged "
             f"(this rank does not bind)")
        )

    caps1 = _capacities(working, order, tpm)
    pool1, binder1 = _world_pool(caps1, vec)
    res1, _res1_why = _resolved_world_pool(caps1)

    lines: List[str] = []
    for i, card in enumerate(cards):
        sc = order[i]
        free_after = _predicted_free_mib(
            card.total_mib, working[i], int(dormant_mib.get(card.uuid, 0)),
            int(getattr(card, "reserved_mib", 0)), sc,
        )
        verdict, extra = next((v, e) for j, v, e in verdicts if j == i)
        cf = floors[card.uuid]
        lines.append(
            f"WEG2-BUDGET corridor-constrained card={card.uuid} "
            f"budget_mib={budgets[i]}->{working[i]} "
            f"predicted_free_mib={free_after} law={cf.verdict_floor_mib} "
            f"floor={cf.mib} source={cf.source} reserve={cf.reserve_mib} "
            f"world_pool={pool0}->{pool1} binder={binder1} "
            f"verdict={verdict} ordinal={i} nvml_idx={card.nvml_index} "
            f"{card.name} terms[total={card.total_mib} "
            f"dormant_other={int(dormant_mib.get(card.uuid, 0))} "
            f"carve={int(getattr(card, 'reserved_mib', 0))} "
            f"awake_residue={sc.awake_residue_mib} "
            f"load_transient={sc.load_transient_mib}] {extra}"
        )
        # THE CUT LINE, emitted only where a cut was INSTALLED (#1257c).  It
        # names the actuating term, both pool figures and the binder, so the
        # operator can read what the margin cost without reconstructing it
        # from three other lines.  ``kein-bindender-rang``: a cut that made a
        # rank the binder never gets here -- it is refused above by name --
        # and the price of the one that did happen is printed rather than
        # hidden.
        if verdict == "APPLIED":
            lines.append(
                installed_cut_line(
                    cf,
                    cut_mib=budgets[i] - working[i],
                    pool_before=pool0,
                    pool_after=pool1,
                    vector=vec,
                    binder=binder1,
                )
            )
    lines.append(
        f"WEG2-BUDGET corridor-constrained group={group} SOURCE={sample.provenance} "
        f"cell_size={sample.cell_size} token_vector={vec} "
        f"world_pool fixed-vector {pool0}->{pool1} "
        f"re-solved-vector {res0 if res0 is not None else 'UNPRICED'}"
        f"->{res1 if res1 is not None else 'UNPRICED'} "
        f"binder {binder0}->{binder1} {ratio_note}"
    )
    return CorridorSolve(
        budgets=tuple(working),
        lines=tuple(lines),
        unpriced_reason=None,
        world_pool_before=pool0,
        world_pool_after=pool1,
    )
