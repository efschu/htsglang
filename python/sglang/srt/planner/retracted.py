"""Retracted investigations, and the config values they emitted (#797).

The rejected register next door (``planner/rejected.py``) answers "was this
CONFIGURATION tried and settled". This one answers a different question that
had no home: "is the number I am about to run *traceable to work that was
later withdrawn*".

The two are not the same question. A rejected entry says a combination loses.
A retraction says an investigation's OUTPUTS lost their warrant -- the run was
mismeasured, the attribution was wrong, the conclusion was pulled -- so any
value still carrying that lineage is riding on evidence that no longer exists.
Nothing noticed the difference before, and the shipped uneven-DCP token vector
is the proof: ``29,19,16`` traces to the retracted #602 investigation, it has
been the active vector on every boot since, and every one of those boots
printed a better vector it then discarded.

The rule this module enforces:

    AN ACTIVE VECTOR MUST NEVER ORIGINATE FROM A RETRACTED INVESTIGATION.

"Active" is the load-bearing word and it is why this is a runtime gate rather
than a comment. A retracted value that is merely PRESENT is harmless -- as a
sizing seed it is superseded in-process by the measured optimum (#797) before
anything runs on it. A retracted value that is PINNED, or that is seeded and
then fails to be superseded, becomes the number the server actually serves,
and that is refused. So the check is applied where a vector becomes active,
not where it is parsed, and the seed path is re-checked AFTER the install
attempt rather than waved through.

Two matching modes, layered, because a rule that only fires when the operator
volunteers the incriminating fact is not a rule:

1.  DECLARED provenance. ``--uneven-token-vector-provenance '#602'`` names the
    lineage; if that names a retracted investigation the vector is refused.
    This is the durable, general rule and it is value-independent.
2.  UNDECLARED provenance. A retraction records the concrete values the
    investigation emitted, and a vector matching one of them is refused with
    the same verdict. Without this, the rule would be toothless on exactly the
    vector that motivated it, because the launch that ships ``29,19,16``
    naturally declares no provenance at all.

Mode 2 is deliberately narrow: it matches the gcd-reduced vector, so it cannot
be dodged by writing ``58,38,32``, and it names what to do instead rather than
just refusing. A value that is genuinely re-derived by measurement is not
matched by it, because a measured install stamps ``PROVENANCE_MEASURED`` and
mode 1 short-circuits before any value comparison happens.

Adding a retraction here is the whole cost of enforcing one. The register is
data; the gate reads it.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Sequence, Tuple

__all__ = [
    "PROVENANCE_MEASURED",
    "RetractedInvestigation",
    "REGISTER",
    "by_investigation",
    "reduce_vector",
    "find_retracted_token_vector",
    "RetractedProvenanceError",
]

#: The provenance a vector carries when the runtime derived it from this
#: boot's own measured per-rank capacity. Never refused: it has no lineage to
#: an investigation at all, it has the measurement.
PROVENANCE_MEASURED = "measured"


class RetractedProvenanceError(ValueError):
    """An active config value traces to a retracted investigation."""


@dataclasses.dataclass(frozen=True)
class RetractedInvestigation:
    """One withdrawn investigation and the values it emitted."""

    #: Canonical id, as written in the docs and in --*-provenance flags.
    #: Matching is case-insensitive and tolerant of a missing '#'.
    investigation: str
    #: What the investigation was about, one line.
    what: str
    #: Why it was withdrawn. This text is quoted verbatim into the refusal,
    #: because a refusal that does not say what went wrong just gets
    #: overridden by the next person.
    retracted_because: str
    #: Where to read the record.
    evidence: str
    #: Token vectors this investigation emitted, gcd-reduced. A vector equal
    #: to one of these is refused even when no provenance was declared.
    token_vectors: Tuple[Tuple[int, ...], ...] = ()
    #: What to do instead. Actionable, one click, same contract as the
    #: rejected register's ``unlock``.
    instead: str = ""


REGISTER: Tuple[RetractedInvestigation, ...] = (
    RetractedInvestigation(
        investigation="#602",
        what=(
            "fill-side per-card attribution for the uneven-DCP KV split; "
            "source of the shipped token vector 29,19,16"
        ),
        retracted_because=(
            "the attribution was withdrawn and re-derived more than once "
            "(docs/dev/NOTE_602_fill_side_attribution.md:67, 'already had to "
            "retract twice'), so the per-card slack figures the vector was "
            "proportioned from no longer stand. The vector outlived the "
            "analysis that produced it: every boot since has measured its own "
            "per-rank capacity, computed a better vector, printed it, and "
            "discarded it (#797)."
        ),
        evidence=(
            "docs/dev/NOTE_602_fill_side_attribution.md; "
            "docs/dev/DESIGN_795_kv_page_federation.md:180-195"
        ),
        token_vectors=((29, 19, 16),),
        instead=(
            "let the runtime measure it: pass --uneven-token-vector-role seed "
            "so this boot's own profiled per-rank capacity supersedes the "
            "estimate in-process (#797), or declare a real lineage with "
            "--uneven-token-vector-provenance if the vector came from "
            "somewhere else"
        ),
    ),
)


def _normalise(investigation: Optional[str]) -> str:
    """'#602', '602', ' #602 ' and 'NOTE_602' all name the same thing."""
    if not investigation:
        return ""
    return str(investigation).strip().lstrip("#").strip().lower()


def by_investigation(investigation: Optional[str]) -> Optional[RetractedInvestigation]:
    """The retraction naming `investigation`, or None."""
    wanted = _normalise(investigation)
    if not wanted:
        return None
    for entry in REGISTER:
        if _normalise(entry.investigation) == wanted:
            return entry
    return None


def reduce_vector(vector: Sequence[int]) -> Tuple[int, ...]:
    """gcd-reduce, so 58,38,32 and 29,19,16 are recognised as one vector.

    Mirrors the reduction ``resolve_cp_token_ratios`` and the measured
    optimiser both apply, so the register stores exactly the form the runtime
    would compare against.
    """
    values = [int(v) for v in vector]
    if not values or any(v <= 0 for v in values):
        return tuple(values)
    g = math.gcd(*values) if len(values) > 1 else values[0]
    return tuple(v // g for v in values) if g > 1 else tuple(values)


def find_retracted_token_vector(
    vector: Optional[Sequence[int]],
    provenance: Optional[str] = None,
) -> Optional[RetractedInvestigation]:
    """The retraction this token vector traces to, or None.

    Mode 1 (declared): a provenance naming a retracted investigation matches
    regardless of the vector's value. ``PROVENANCE_MEASURED`` short-circuits
    to None -- a vector this boot measured has no lineage to withdraw.

    Mode 2 (undeclared): with no provenance given, the gcd-reduced vector is
    matched against the values each retraction emitted.

    A DECLARED, non-retracted provenance also short-circuits to None: the
    operator has stated where the number came from, and mode 2's value match
    is a fallback for an unstated lineage, not an override of a stated one.
    """
    declared = _normalise(provenance)
    if declared:
        if declared == _normalise(PROVENANCE_MEASURED):
            return None
        return by_investigation(declared)
    if not vector:
        return None
    reduced = reduce_vector(vector)
    for entry in REGISTER:
        if reduced in entry.token_vectors:
            return entry
    return None


def token_vector_refusal_text(
    entry: RetractedInvestigation,
    vector: Optional[Sequence[int]],
    how: str,
) -> str:
    """The verdict, assembled once so every call site refuses identically."""
    shown = ",".join(str(v) for v in vector) if vector else "(none)"
    return (
        f"Uneven-DCP token vector {shown} traces to RETRACTED investigation "
        f"{entry.investigation} ({entry.what}), and {how}. An active token "
        f"vector must never originate from a retracted investigation.\n"
        f"  Why {entry.investigation} was retracted: {entry.retracted_because}\n"
        f"  Record: {entry.evidence}\n"
        f"  Instead: {entry.instead}"
    )
