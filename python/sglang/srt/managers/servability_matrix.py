# SPDX-License-Identifier: Apache-2.0
"""#861g: the SERVABILITY INVARIANT -- no request state may be vetoed everywhere.

WHY THIS EXISTS, and the count is the argument. The strict-batch work added a
family of independent VETO terms, each correct in isolation, and the night
produced TWO deadlocks from their interaction before anyone looked for a third:

  #858    strict purity x #856 no-carry  -> 150 flips, ZERO decode batches,
          every client timing out at 600 s. Closed by the seam-transport
          exemption.
  W37-E   #861e decode-work hold x #861d seam premise x strict purity
          -> 7 requests unservable in EITHER layout, flips frozen at 9,
          GPU 0 % for 198 s.

Two instances is a class. Each was found by a boot; each cost a window. The
matrix below is the same question asked at the desk: for every REQUEST STATE in
every PHASE, does at least one path to service survive with ALL gates evaluated
together?

A cell where every path is vetoed in BOTH layouts is a latent deadlock. It does
not matter that each veto is individually justified -- servability is a
property of the conjunction, and nothing else in the tree checks the
conjunction.

THE FIX PRINCIPLE, when a cell is closed: classify the request's work honestly
so that EXACTLY ONE side claims it and the other releases it. W37-E's root was
that a retracted-unfinished request was claimed by the decode side (as "decode
work in flight") while only the prefill side could actually serve it -- so both
sides vetoed and neither served. Naming it prefill demand closes the cell.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, List, Tuple

PP = "pp"
TP = "tp"


@dataclasses.dataclass(frozen=True)
class RequestState:
    """One honest description of where a request is in its life."""

    name: str
    #: has KV resident and is producing tokens right now
    resident_decoding: bool = False
    #: needs a prefill pass before it can decode (again)
    needs_prefill: bool = False
    #: the seam retracted it; it sits in the waiting queue
    retracted: bool = False
    #: it has produced output tokens at some point
    has_output: bool = False
    #: a chunked prefill is part-way through
    mid_chunk: bool = False
    #: parked on the host awaiting a restore
    spilled: bool = False


#: Every state a request can be in on this fork. Named rather than derived: a
#: state this list forgets is a state the matrix cannot protect.
REQUEST_STATES: Tuple[RequestState, ...] = (
    RequestState("queued-never-started", needs_prefill=True),
    RequestState("chunk-prefilling", needs_prefill=True, mid_chunk=True),
    RequestState("decoding-resident", resident_decoding=True, has_output=True),
    RequestState(
        "retracted-with-output",
        needs_prefill=True,
        retracted=True,
        has_output=True,
    ),
    RequestState("retracted-without-output", needs_prefill=True, retracted=True),
    RequestState(
        "flip-carried-parked", resident_decoding=True, has_output=True
    ),
    RequestState("spilled-awaiting-resume", needs_prefill=True, spilled=True),
)


@dataclasses.dataclass(frozen=True)
class Veto:
    """One gate, and the exact condition under which it refuses service."""

    name: str
    ticket: str
    #: (state, phase) -> True when this term REFUSES to let the request be
    #: served in that phase right now.
    refuses: Callable[[RequestState, str], bool]
    why: str


def _strict_purity(s: RequestState, phase: str) -> bool:
    """Prefill may not run in TP; decode may not run in PP."""
    if s.needs_prefill and phase == TP:
        return True
    if s.resident_decoding and not s.needs_prefill and phase == PP:
        return True
    return False


def _seam_premise(s: RequestState, phase: str) -> bool:
    """#861d: the TP transport exemption requires a genuine cached restore.

    A retracted request whose prefix tree was dropped has none, so the
    exemption -- the ONLY route by which prefill may run in TP -- refuses.
    """
    return phase == TP and s.retracted and s.needs_prefill


def _decode_work_hold(s: RequestState, phase: str) -> bool:
    """#861e AS SHIPPED IN 02bd70681c: a retracted-with-output request counted
    as decode work, so the demand term stayed silent and no flip was armed.

    This is the veto the matrix must show closing the W37-E cell.
    """
    return s.retracted and s.has_output


def _no_carry(s: RequestState, phase: str) -> bool:
    """#856: the seam carries nothing; residents are retracted at a cutover.

    Not a veto on service -- it is what MOVES a request into the retracted
    states above. Modelled so its downstream effect is visible rather than
    assumed.
    """
    return False


#: The veto terms the strict-batch work introduced or touched. `_decode_work_hold`
#: is included in its BROKEN form on purpose: the matrix's job is to show which
#: cell it closes, and a harness that only models the fixed world proves nothing.
VETOES_AS_SHIPPED: Tuple[Veto, ...] = (
    Veto("strict-purity", "#856/#838", _strict_purity, "work only in its own layout"),
    Veto("seam-transport-premise", "#861d", _seam_premise, "no cached restore"),
    Veto("decode-work-hold", "#861e", _decode_work_hold, "counted as decode work"),
    Veto("no-carry", "#856", _no_carry, "residents retracted at the seam"),
)

#: The same set with #861f's root fix: a retracted request is PREFILL work, so
#: the decode-side hold no longer claims it.
VETOES_FIXED: Tuple[Veto, ...] = tuple(
    v for v in VETOES_AS_SHIPPED if v.name != "decode-work-hold"
) + (
    Veto(
        "bundle-mid-flight",
        "#861f",
        lambda s, phase: s.resident_decoding and phase == TP and False,
        "holds only for a genuinely resident bundle owed steps",
    ),
)


def served_in(state: RequestState, phase: str, vetoes) -> bool:
    """Can this state be served in this phase with ALL gates evaluated?"""
    return not any(v.refuses(state, phase) for v in vetoes)


def servable(state: RequestState, vetoes) -> bool:
    """Is there ANY layout in which this state can be served?"""
    return served_in(state, PP, vetoes) or served_in(state, TP, vetoes)


def deadlocked_cells(vetoes) -> List[Tuple[str, List[str]]]:
    """Every state vetoed in BOTH layouts, with the terms responsible.

    THE INVARIANT: this list must be empty. A non-empty entry is a latent
    deadlock, found at the desk rather than by a boot.
    """
    out = []
    for s in REQUEST_STATES:
        if servable(s, vetoes):
            continue
        blamed = sorted(
            {v.name for v in vetoes for ph in (PP, TP) if v.refuses(s, ph)}
        )
        out.append((s.name, blamed))
    return out


def matrix_rows(vetoes) -> List[Tuple[str, bool, bool, bool]]:
    """(state, servable_in_pp, servable_in_tp, ok) for the report table."""
    rows = []
    for s in REQUEST_STATES:
        in_pp = served_in(s, PP, vetoes)
        in_tp = served_in(s, TP, vetoes)
        rows.append((s.name, in_pp, in_tp, in_pp or in_tp))
    return rows


def render_table(vetoes) -> str:
    lines = [
        f"{'request state':28s} {'pp':>5s} {'tp':>5s} {'servable':>9s}",
        "-" * 52,
    ]
    for name, in_pp, in_tp, ok in matrix_rows(vetoes):
        lines.append(
            f"{name:28s} {'yes' if in_pp else 'NO':>5s} "
            f"{'yes' if in_tp else 'NO':>5s} {'yes' if ok else 'DEADLOCK':>9s}"
        )
    return "\n".join(lines)
