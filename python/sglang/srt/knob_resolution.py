"""One authority for HOW a knob's value is resolved and HOW that is reported.

WHY THIS MODULE EXISTS
======================
On 2026-08-26 four independent reporting mechanisms for the same question --
"which source actually supplied this knob's value, and what was lost?" --
were built in one day, in four modules, by four tickets:

* ``managers/phase_policy.py``: ``_flag_or_env`` + ``_env_source``, a
  provenance RECORDER that prints one ``knob=<v> from <source>`` line for
  fifteen phase-policy knobs (#896).
* ``managers/min_free_slots_delayer.py``: a ``(value, reason)`` verdict pair
  plus a NARROWED-knob warner (#894 S4).
* ``layers/quantization/gguf.py``: ``_announce_mmq_env_override``, a latched
  one-shot SUPERSEDED-knob warner (#894 S5).
* ``distributed/utils.py``: ``announce_superseded_rank_kv_ratio``, the same
  shape again for a five-level VECTOR precedence with gcd reduction, a #797
  retraction refusal and seed arming (#897).

Each was correct. Four of them are not a fix, they are a class: the fifth
knob that goes silent will go silent because its module had no reporter, and
nothing in the tree makes the omission visible. #901 is the structural answer
the operator asked for -- one resolution authority, four call sites.

WHAT IS UNIFIED AND WHAT IS DELIBERATELY NOT
============================================
UNIFIED: the resolution walk (an ordered ladder of candidate sources), the
equivalence test that decides whether a losing source actually LOST anything,
the verdict vocabulary, and the printed forms -- the provenance field
``knob=<value> from <source>``, the SUPERSEDED line, the NARROWED line, and
the one remedy that is always correct for an environment override:
**REMOVE the variable, never blank it** (``server_args.py:5607`` records the
day an empty append silently switched uneven token sharding off).

NOT UNIFIED -- and this is not an omission:

* **The precedence ORDER stays a per-site parameter.** Env-over-flag is
  deliberate DESIGN in this fork, not an accident: the server logs
  ``restart with SGLANG_...=`` after a calibration run and the environment
  re-applies the measured value without re-parsing ServerArgs (SKILL.md
  Rule 6, rig-runbook section 2). Flag-over-env is equally deliberate where
  #781 promoted a knob. Both orders are correct where they are; this module
  resolves and REPORTS a ladder, it never reorders one.
* **WHEN to announce stays the site's decision.** ``resolve_knob`` is a pure
  function: it logs nothing, latches nothing and touches no global. That is
  what lets it be called from ``resolve_cp_token_ratios``-shaped resolvers
  whose contract is "same input, same output, no side effects" and whose
  zero-logger property is pinned by a registered test. The announcement is a
  separate call the boot-time site makes once.

MIGRATION DEBT, STATED HONESTLY
===============================
Measured 2026-08-26 at ``b5f3dcbd46`` (merge/train-826 tip), by
``grep -rn 'os\.environ' --include=*.py python/sglang/srt``: **648 reads in
209 files**. This module is used by FOUR of them. The other 205 files are
unmigrated and each remains a possible private precedent island.
(The ticket was raised against a count of 646/208; the tree measures 648/209
at this commit, and the measurement is what is recorded here -- a debt figure
carried forward from a briefing rather than re-taken is how a number outlives
the tree it described.) The ratchet in
``test/registered/unit/server_args/test_knob_resolution_authority_901.py``
covers exactly the migrated scope and no more -- a 209-file big-bang gate
would have to be written as an allowlist of 648 entries, which is a second
document rather than a second guard. Widening the scope list is a deliberate
act with its own review; the debt is named here and in
``/spinning/gpu-arb/REGISTER_OPEN_876.txt`` so it cannot be mistaken for
finished work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "KIND_DEFAULT",
    "KIND_DERIVED",
    "KIND_ENV",
    "KIND_FLAG",
    "KnobSource",
    "PROVENANCE_DEFAULT",
    "Resolution",
    "VERDICT_CAPPED",
    "VERDICT_DISCARDED",
    "VERDICT_HONOURED",
    "VERDICT_SOLE",
    "VERDICT_SUPERSEDED",
    "Announcer",
    "env_present",
    "env_value",
    "env_present_nonempty",
    "env_provenance",
    "env_source",
    "flag_source",
    "loss_clause",
    "narrowed_head",
    "provenance_field",
    "provenance_line",
    "removal_remedy",
    "resolve_knob",
    "supersession_line",
    "unreadable_loss_clause",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
#: The word ``_env_source`` and ``_flag_or_env`` have printed since #896 for a
#: knob nobody set. Kept verbatim so the boot log's grammar does not move.
PROVENANCE_DEFAULT = "default"

KIND_FLAG = "flag"
KIND_ENV = "env"
KIND_DERIVED = "derived"
KIND_DEFAULT = "default"

#: What happened to the value on the way from the source to the consumer.
#: Two families, because the four sites needed two questions answered:
#:
#: WHO supplied it -- ``VERDICT_SOLE`` (nothing else was present) vs
#: ``VERDICT_SUPERSEDED`` (a lower-precedence source was present, carried a
#: DIFFERENT value, and was never consulted).
#:
#: WHAT SURVIVED of it -- ``VERDICT_HONOURED`` / ``VERDICT_CAPPED`` /
#: ``VERDICT_DISCARDED``, for a value a downstream constraint narrowed or
#: dropped. #894 S4's silent cap-and-discard is this family; a knob can be
#: SOLE and DISCARDED at the same time, which is exactly the pair that made
#: ``--min-free-slots-delay 6`` on a small pool indistinguishable from a
#: request that was honoured.
VERDICT_SOLE = "sole"
VERDICT_SUPERSEDED = "superseded"
VERDICT_HONOURED = "honoured"
VERDICT_CAPPED = "capped"
VERDICT_DISCARDED = "discarded"


def flag_source(field_name: str) -> str:
    """``"flag --rank-kv-ratio"`` for the ServerArgs field ``rank_kv_ratio``."""
    return "flag --" + field_name.replace("_", "-")


def env_source(env_name: str) -> str:
    """``"env SGLANG_X"``."""
    return f"env {env_name}"


# ---------------------------------------------------------------------------
# Presence rules -- a per-site parameter, because the sites genuinely differ
# ---------------------------------------------------------------------------
def env_present_nonempty(env_name: str, environ=None) -> bool:
    """``FOO=`` is NOT a source (#896's rule, and the safe default).

    ``_env_float`` and friends fall through to the default for an empty
    string, so reporting "env" there would name a source that did not supply
    the value.
    """
    environ = os.environ if environ is None else environ
    return environ.get(env_name) not in (None, "")


def env_present(env_name: str, environ=None) -> bool:
    """``FOO=`` IS a source (#894 S5's rule, and the more dangerous one).

    ``gguf._mmq_decode_threshold_enabled`` short-circuits on
    ``os.environ.get(name) is not None`` and then tests ``== "1"``, so an
    empty override is PRESENT and reads as OFF. A site whose reader keys on
    ``is not None`` must declare that here, or the report would claim the
    default supplied a value the env actually decided.
    """
    environ = os.environ if environ is None else environ
    return environ.get(env_name) is not None


def env_value(env_name: str, environ=None) -> Optional[str]:
    """The raw string, for a site that has already asked a presence rule.

    Exists so a migrated site never has to spell ``os.environ`` again: the
    presence question and the value question are asked of the same authority,
    which is what keeps the two from drifting apart in one module the way
    they drifted apart across four.
    """
    environ = os.environ if environ is None else environ
    return environ.get(env_name)


def env_provenance(env_name: str, environ=None) -> str:
    """The provenance string for a knob with no CLI flag: the env, or default."""
    return (
        env_source(env_name)
        if env_present_nonempty(env_name, environ)
        else PROVENANCE_DEFAULT
    )


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KnobSource:
    """One rung of a knob's precedence ladder, highest precedence first.

    ``value`` vs ``reader``: a rung may carry its value directly, or a
    zero-argument callable evaluated ONLY if that rung wins. The lazy form
    exists because a losing rung must not be parsed -- ``_env_float`` on a
    malformed value raises, and a rung that lost has no business raising.

    ``cost`` is the site's own sentence about what is lost when this rung is
    present and beaten. It is prose, not a template: #894 S5's loss ("the
    flag was never consulted") and #897's ("the mode is INERT in both phases,
    and a role='pin' env vector also suppresses the post-profiling install")
    are different facts, and flattening them into one generated sentence
    would report neither.
    """

    source: str
    value: Any = None
    present: bool = False
    kind: str = KIND_DERIVED
    reader: Optional[Callable[[], Any]] = None
    cost: Optional[str] = None
    #: Human name of the rung for a loss clause, e.g. "--rank-kv-ratio 3,7".
    #: Defaults to ``source`` when the site has nothing shorter to say.
    label: Optional[str] = None
    #: Is losing at this rung something to TELL somebody about?
    #:
    #: Present and reportable are different questions, and conflating them is
    #: a real defect rather than a nicety: #897's ladder ends in two rungs the
    #: resolver derives internally -- the per-rank free-KV budget estimate and
    #: the gcd-reduced weights fallback -- which are always "available" and
    #: which nobody ever chose. Reporting them as superseded produced a line
    #: on a configuration where nothing at all was lost, which is exactly the
    #: noise that teaches operators to skip the line. The rungs stay in the
    #: ladder because the ladder in the code should be the ladder in the
    #: docstring; they are marked unreportable because the operator did not
    #: write them.
    reportable: bool = True

    def resolved_value(self) -> Any:
        return self.reader() if self.reader is not None else self.value

    def loss_label(self) -> str:
        return self.label if self.label is not None else self.source


@dataclass(frozen=True)
class Resolution:
    """Who supplied the value, what was superseded, what survived."""

    value: Any
    source: str
    winner: KnobSource
    ladder: Tuple[KnobSource, ...]
    superseded: Tuple[KnobSource, ...] = ()
    verdict: str = VERDICT_SOLE
    constraint_verdict: str = VERDICT_HONOURED
    requested: Any = None

    @property
    def lost_anything(self) -> bool:
        return bool(self.superseded)

    @property
    def top_loser(self) -> Optional[KnobSource]:
        """The highest-precedence source that was present and lost.

        That is the one the operator most likely wrote by hand, so it is the
        one a single-line announcement names.
        """
        return self.superseded[0] if self.superseded else None


def _identity(value: Any) -> Any:
    return value


def resolve_knob(
    ladder: Sequence[KnobSource],
    *,
    normalize: Callable[[Any], Any] = _identity,
    constraint: Optional[Callable[[Any, KnobSource], Tuple[Any, str]]] = None,
) -> Resolution:
    """Walk the ladder, name the winner, and say who lost something.

    PURE. No logging, no latch, no global. The site decides when -- and
    whether -- any of this is announced; see the module docstring for why
    that separation is load-bearing rather than stylistic.

    ``normalize`` is the equivalence test between a winner and a present
    loser. It is a parameter because "the same value" is site-specific: #897
    compares gcd-reduced vectors (``6,2`` and ``3,1`` are the same ownership
    split), #894 S5 compares booleans across an env string and a flag. A
    loser whose normalized value equals the winner's lost NOTHING and is not
    reported -- a line that also fires when nothing was lost is a line
    readers learn to skip.

    ``constraint`` is the second question: given the resolved value and the
    rung that supplied it, what survived? It returns ``(final_value,
    verdict)``. Sites without a downstream cap or floor omit it.

    Scalar, vector and multi-stage are all the same walk: the ladder length
    is arbitrary and the values are opaque to this function. The measuring
    stick was #897 -- five rungs, list values, gcd equivalence -- and it is
    expressed here without a special case.
    """
    rungs = tuple(ladder)
    if not rungs:
        raise ValueError("a knob ladder needs at least one rung")

    winner_index = None
    for index, rung in enumerate(rungs):
        if rung.present:
            winner_index = index
            break
    if winner_index is None:
        # No rung declared itself present. The last rung is the default by
        # construction; treating it as the winner keeps the walk total rather
        # than making every caller append a present=True tail.
        winner_index = len(rungs) - 1

    winner = rungs[winner_index]
    value = winner.resolved_value()
    winner_key = normalize(value)

    superseded: List[KnobSource] = []
    for rung in rungs[winner_index + 1 :]:
        if not rung.present or not rung.reportable:
            continue
        try:
            same = normalize(rung.resolved_value()) == winner_key
        except Exception:
            # A loser that cannot even be normalized cannot be proven equal,
            # so it is reported. Swallowing it would be the silence this
            # module exists to end.
            same = False
        if not same:
            superseded.append(rung)

    verdict = VERDICT_SUPERSEDED if superseded else VERDICT_SOLE
    constraint_verdict = VERDICT_HONOURED
    requested = value
    if constraint is not None:
        value, constraint_verdict = constraint(value, winner)

    return Resolution(
        value=value,
        source=winner.source,
        winner=winner,
        ladder=rungs,
        superseded=tuple(superseded),
        verdict=verdict,
        constraint_verdict=constraint_verdict,
        requested=requested,
    )


# ---------------------------------------------------------------------------
# The printed forms
# ---------------------------------------------------------------------------
def provenance_field(name: str, value: Any, source: str, fmt: str = "") -> str:
    """``knob=<value> from <source>`` -- #896's field, verbatim.

    ``fmt`` is a format spec for the value (``"g"`` for the float knobs,
    ``"d"`` for counts, ``""`` for strings), so the one field builder covers
    every knob on the provenance line instead of the line growing a second
    spelling for each new type.
    """
    rendered = format(value, fmt) if fmt else f"{value}"
    return f"{name}={rendered} from {source}"


def provenance_line(prefix: str, fields: Iterable[str]) -> str:
    """``<PREFIX> knob provenance: a=1 from flag --a | b=2 from default``.

    ``" | "`` rather than ``", "``: a source may itself contain a comma (the
    phase-policy seam reports seed AND estimator state), and a separator a
    field can also produce is not a separator.
    """
    return f"{prefix} knob provenance: " + " | ".join(fields)


#: The sentence every superseding environment override deserves, and the one
#: mistake none of them may advise.
#:
#: The cited incident is named in EVERY site's remedy, not only in the module
#: it happened in, and that is deliberate: it is this fork's own recorded
#: evidence for the rule, and a rule stated without its evidence is advice
#: readers weigh against convenience. ``SGLANG_UNEVEN_TOKEN_VECTOR`` was set,
#: then blanked by a later empty append, and uneven token sharding was off for
#: a day with nobody aware (server_args.py:5607). Any presence-keyed override
#: can be switched off the same way; the GGUF threshold's presence rule is
#: ``is not None``, so it can be switched off that way TODAY.
_EMPTY_STRING_TRAP = (
    "not by setting it to an empty string, which is how uneven token sharding "
    "was silently switched off for a day once already (server_args.py:5607)"
)


def removal_remedy(env_name: str, *, govern: str = "the flag govern") -> str:
    """REMOVE, never blank -- the one remedy shared by every env supersession.

    #894 S5 said "unset SGLANG_GGUF_MMQ_DECODE_THRESHOLD to let the flag
    govern", which is right but does not close the trap: that site's presence
    rule is ``is not None``, so ``SGLANG_GGUF_MMQ_DECODE_THRESHOLD=`` is
    still PRESENT and still reads as OFF. An operator following the shorter
    advice with ``export FOO=`` would change nothing and see the same
    warning. One remedy, spelled out once.
    """
    return (
        f"to let {govern}, REMOVE {env_name} from the environment -- "
        f"{_EMPTY_STRING_TRAP}."
    )


def loss_clause(loser: str, cost: str) -> str:
    """``and --rank-kv-ratio 3,7 did not decide it: <cost>``."""
    return f"and {loser} did not decide it: {cost}"


def unreadable_loss_clause(loser: str, why: str) -> str:
    """The clause for a loser that could not even be READ at this call site.

    #894 S5 reaches the GGUF dispatch before ModelRunner publishes ServerArgs
    on some paths. The comparison is impossible there; the supersession is
    not, and the callers who hit it are the ones most likely to be surprised.
    """
    return f"and {loser} {why}"


def supersession_line(
    ticket: str,
    *,
    winner: str,
    subject: str,
    effective: str,
    loss: str,
    presence_rule: str,
    remedy: str,
    note: str = "",
) -> str:
    """THE line. One skeleton for every "another source decided this" report.

    ``#<ticket> SUPERSEDED KNOB: <winner> decided <subject> -- <effective> --
    <loss>.<note> <presence_rule> Documented precedence, announced rather
    than refused: <remedy>``

    Byte-compatible with #897's hand-built line, which is where the skeleton
    comes from. The parts stay per-site prose on purpose: what is lost, and
    what it costs, is a fact about that knob, and a generated sentence that
    fit all four would have said something true of none of them.

    "Documented precedence, announced rather than refused" is fixed text, not
    a parameter. Every site that reaches this line has already made the same
    call on the same grounds -- the danger direction. Refusing kills a boot on
    exactly the population carrying the stale variable; the defect's blast
    radius is a wrong belief about which value served. A site that ever wants
    to REFUSE instead does not want this line.
    """
    return (
        f"#{ticket} SUPERSEDED KNOB: {winner} decided {subject} -- "
        f"{effective} -- {loss}.{note} {presence_rule} "
        f"Documented precedence, announced rather than refused: {remedy}"
    )


def narrowed_head(ticket: str, prefix: str, flag_field: str, value: Any) -> str:
    """``<PREFIX> #894 NARROWED KNOB: --min-free-slots-delay=6``.

    The other half of the vocabulary: nothing superseded the value, a
    downstream constraint changed or dropped it. Same reason it is one
    builder -- the next capped knob should not have to invent a spelling.
    """
    return f"{prefix} #{ticket} NARROWED KNOB: --{flag_field.replace('_', '-')}={value}"


# ---------------------------------------------------------------------------
# Say it once
# ---------------------------------------------------------------------------
@dataclass
class Announcer:
    """A one-shot latch per knob, resettable for tests.

    Three of the four migrated sites hand-rolled a module-level
    ``_..._announced`` boolean plus a reset hook, and #894's own suite pins
    that a reset hook which clears the decision but not the latch is a defect
    (it leaves the second decision silent again). One implementation, one
    reset contract.

    ``said`` is public because the sites' tests assert on it.
    """

    key: str
    said: bool = field(default=False)

    def say(self, logger, message: str, level: str = "warning") -> bool:
        """Emit ``message`` once. Returns True if this call emitted it.

        Emitted as ``logger.warning("%s", message)`` so the record's
        ``getMessage()`` is exactly ``message`` -- the migrated sites' tests
        read that, and a pre-formatted line keeps them byte-comparable with
        the hand-built ones they replace.
        """
        if self.said:
            return False
        self.said = True
        getattr(logger, level)("%s", message)
        return True

    def reset(self) -> None:
        self.said = False
