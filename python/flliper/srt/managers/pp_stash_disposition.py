# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#800: what the phase flip does with a message parked in the tensor-dict inbox.

THE DEFECT THIS CLOSES -- a circular wait, measured twice on 2026-08-22, both
times on PP1 (specimens ``/spinning/evidence-665-f1/wedge_1208_120909/boot.log``
and ``/spinning/evidence-665-f1/boot_r8_0822_1210.log``):

    PP1] #757 armed drain took a tensor dict off the wire and STASHED it:
         kind=admission_decision stamp=None
    PP1] PHASE-FLIP epoch 4 round 0: WITHHOLDING presence (57922 rounds so far)
         -- tensor-dict inbox holds 1 stashed message(s).

The cycle has four links and every one of them is individually correct:

  1. An armed rank spins in the presence gate and services the wire each round
     (``phase_flip_runtime``'s ``_service_fn`` -> ``pp_flip_service`` ->
     ``pp_flip_drain_tensor_dicts``; ``_drain_fn`` is unwired in production and
     the service turn does that job). It must: the upstream's blocking send
     commit is satisfied by the message LEAVING THE WIRE, so a rank that stops
     consuming blocks its upstream.
  2. #757's classifier stashes anything that is not a provably void proxy --
     "an unidentifiable message is not evidence of a void pass". Correct: the
     alternative is corpse S, which ate an owed output.
  3. #791/#795 put a THIRD kind on that wire, ``admission_decision``, sent
     every pass by every non-last rank. Correct on its own terms.
  4. ``pp_flip_channels_empty`` counts every stashed message as "channels not
     empty", so the rank withholds presence. Correct for an output: a token
     that crosses the cutover is a token the client never sees (#631).

THE SEAM. The ONLY consumer of ``admission_decision`` is
``_pp_recv_admission_decision``, issued at the top of a PP pass -- which cannot
run while the rank is in the presence gate. So link 4 waits for a consumer that
link 4 itself is preventing from running. It is deterministic rather than racy:
PP0 sends its decision at the top of pass N+1 while PP1 is still finishing pass
N and arming, so an armed PP1 essentially always has one in flight. That is why
both specimens name rank 1 and no other rank.

WHAT IT LOOKS LIKE FROM OUTSIDE, and this is one defect, not two. The gate's
withhold branch and the group's abandonment describe the SAME state from two
sides: a rank appears in ``_abandon_no_quorum``'s ``missing`` list
(phase_flip_runtime.py) exactly when it did not call ``announce``, and the only
branch that skips ``announce`` for a rank that HAS reached the gate is the
withhold branch. The abandonment then advises "a rank that never reaches the
entry is blocked upstream of it: look there, not at the flip", which the
withhold line contradicts in its own text ("AT the entry and declining to
announce; it is not blocked upstream of it").

WHY NOT SIMPLY DISCARD THE DECISION WHILE ARMED. Because the flip usually
ABANDONS rather than commits, and an abandon resets nothing: the resumed PP body
issues exactly one decision receive per pass and the upstream posts exactly one
send per pass. Dropping one in the armed window makes every later receive off by
one, permanently -- the mispair class this file's neighbours exist to prevent.
The message must survive an abandon. It must simply stop blocking presence.

THE CONTRACT. A kind stashed on this channel has exactly one of three
dispositions at the flip gate, and a kind with no declared disposition is a
NAMED state rather than silence:

  ``BLOCKS_FLIP``     A real consumer looks for it AFTER the cutover, or its
                      loss is a lost payload. It must block presence. This is
                      the status quo for ``output``, ``proxy`` and
                      ``crossing``.
  ``PP_LOOP_ONLY``    Its only consumer is the PP event-loop body, which by
                      construction cannot run while the presence gate holds the
                      rank. It must NOT block presence -- doing so is the
                      circular wait above. It survives an abandon untouched
                      (the resumed loop pops it) and is RETIRED, loudly, at a
                      cutover, where the pass numbering it belongs to ceases to
                      exist.
  ``UNDECLARED``      A kind nobody has classified. It blocks presence -- the
                      conservative reading, because it might be an owed payload
                      -- but it says so BY NAME from the first round and is
                      retired loudly once the escape deadline expires, so a
                      future fourth kind costs a bounded, named abandonment
                      instead of an unbounded silence. This is the whole point:
                      extending a table for one more message type would merely
                      prepare the next seam.

Pure and module-level, so every decision here is testable without a scheduler,
a process group or a boot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "BLOCKS_FLIP",
    "PP_LOOP_ONLY",
    "UNDECLARED",
    "StashCensus",
    "declared_stash_kinds",
    "stash_flip_disposition",
    "census_stash",
]

#: A real consumer looks for this after the cutover; losing it loses a payload.
BLOCKS_FLIP = "blocks_flip"

#: Its only consumer is the PP loop body, which cannot run while the gate holds.
PP_LOOP_ONLY = "pp_loop_only"

#: Nobody declared a disposition for this kind. Blocks, loudly, under a deadline.
UNDECLARED = "undeclared"


#: The disposition of every kind that travels on the PP tensor-dict channel.
#:
#: ``output``  -- the sampled tokens ringed from the last rank. A discarded one
#:                is a token that reaches no ``output_ids`` (#631).
#: ``proxy``   -- stage-boundary hidden states. Paired to a batch purely by slot
#:                position, so a stranded one puts every later receive off by
#:                one (#631 defect Q).
#: ``crossing``-- the gapped wire's mid-loop activations (#753). Same pairing
#:                argument as ``proxy``.
#: ``admission_decision`` -- #791's per-pass admission verdict. Consumed only by
#:                ``_pp_recv_admission_decision`` at the top of a PP pass, which
#:                the presence gate structurally excludes. See this module's
#:                docstring for the measurement.
#: ``default`` -- an untyped message. Deliberately NOT declared: it is the
#:                catch-all ``pp_typed_channel`` assigns when a sender omitted a
#:                kind, so it is exactly the case the UNDECLARED escape exists
#:                for.
_DISPOSITIONS: Dict[str, str] = {
    "output": BLOCKS_FLIP,
    "proxy": BLOCKS_FLIP,
    "crossing": BLOCKS_FLIP,
    "admission_decision": PP_LOOP_ONLY,
}


def declared_stash_kinds() -> Tuple[str, ...]:
    """Every kind with a declared disposition, for tests and for diagnostics."""
    return tuple(sorted(_DISPOSITIONS))


def stash_flip_disposition(kind: Any) -> str:
    """This kind's disposition at the flip gate; ``UNDECLARED`` if none.

    Never raises and never returns None. A ``None`` that means both "no
    disposition applies" and "the lookup failed" is the error shape this whole
    change exists to remove, so the unknown case has its own name.
    """
    return _DISPOSITIONS.get(str(kind), UNDECLARED)


@dataclass(frozen=True)
class StashCensus:
    """What the inbox holds, split by what the flip may do with it.

    ``blocking`` and ``undeclared`` are DISJOINT: an undeclared kind blocks the
    flip too, but it is reported separately because it is the only group the
    escape hatch may retire, and because it needs a different sentence in the
    log -- one that says nobody classified it, rather than one that says a
    consumer is owed it.
    """

    #: (kind, count) whose consumer looks for it after the cutover.
    blocking: Tuple[Tuple[str, int], ...] = ()
    #: (kind, count) whose only consumer is the PP loop body.
    gate_blind: Tuple[Tuple[str, int], ...] = ()
    #: (kind, count) with no declared disposition.
    undeclared: Tuple[Tuple[str, int], ...] = ()

    @property
    def blocking_total(self) -> int:
        return sum(n for _, n in self.blocking)

    @property
    def gate_blind_total(self) -> int:
        return sum(n for _, n in self.gate_blind)

    @property
    def undeclared_total(self) -> int:
        return sum(n for _, n in self.undeclared)

    def block_reason(self) -> Optional[str]:
        """The presence gate's reason to withhold, or None if it has none.

        Names the KINDS, not just a total. The specimen this module documents
        cost a log-dig precisely because "inbox holds 1 stashed message(s)" did
        not say which kind, and the kind is the whole diagnosis.
        """
        parts: List[str] = []
        if self.blocking:
            parts.append(
                "tensor-dict inbox holds "
                + ", ".join(f"{n} stashed {kind}" for kind, n in self.blocking)
                + " owed to a consumer that looks for it after the cutover"
            )
        if self.undeclared:
            parts.append(
                "tensor-dict inbox holds "
                + ", ".join(f"{n} stashed {kind}" for kind, n in self.undeclared)
                + " of UNDECLARED disposition -- no consumer is registered for "
                "this kind at the flip gate, so it is held conservatively and "
                "retired once the escape deadline expires (#800)"
            )
        if not parts:
            return None
        return "; ".join(parts)


def census_stash(inbox: Optional[Dict[Tuple[int, str], Deque[Any]]]) -> StashCensus:
    """Split a ``(src, kind) -> deque`` inbox by flip disposition.

    Tolerant of anything dict-like, including the ``{}`` a stand-in that never
    ran ``init_pp_loop_state`` presents -- this file's neighbours already rely
    on that convention (#787).
    """
    if not inbox:
        return StashCensus()
    counts: Dict[str, Dict[str, int]] = {
        BLOCKS_FLIP: {},
        PP_LOOP_ONLY: {},
        UNDECLARED: {},
    }
    for key, queue in inbox.items():
        depth = len(queue or ())
        if depth <= 0:
            continue
        kind = str(key[1]) if isinstance(key, tuple) and len(key) >= 2 else str(key)
        bucket = counts[stash_flip_disposition(kind)]
        bucket[kind] = bucket.get(kind, 0) + depth
    return StashCensus(
        blocking=_as_sorted_pairs(counts[BLOCKS_FLIP]),
        gate_blind=_as_sorted_pairs(counts[PP_LOOP_ONLY]),
        undeclared=_as_sorted_pairs(counts[UNDECLARED]),
    )


def stash_keys_with_disposition(
    inbox: Optional[Dict[Tuple[int, str], Deque[Any]]],
    dispositions: Iterable[str],
) -> List[Tuple[int, str]]:
    """Every non-empty inbox key whose kind has one of these dispositions.

    Keys rather than counts, because the two actuators (the escape hatch and
    the cutover retirement) have to REMOVE entries, and removing by kind alone
    would cross the ``(src, kind)`` identity the inbox is keyed by (#753: the
    source is part of the message's identity, so it is part of the key).
    """
    wanted = set(dispositions)
    if not inbox:
        return []
    out: List[Tuple[int, str]] = []
    for key, queue in inbox.items():
        if not queue:
            continue
        kind = str(key[1]) if isinstance(key, tuple) and len(key) >= 2 else str(key)
        if stash_flip_disposition(kind) in wanted:
            out.append(key)
    return sorted(out, key=lambda k: (str(k[1]), str(k[0])))


def _as_sorted_pairs(counts: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted(counts.items()))
