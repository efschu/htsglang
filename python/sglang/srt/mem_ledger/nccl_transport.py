# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Who owns each communicator group's transport -- for the ledger's NCCL term.

The NCCL buffer term (:data:`~sglang.srt.mem_ledger.engine.TERM_NCCL_BUFFERS`)
has three states, and this module decides which one applies:

``UNBOUNDED``
    An NCCL communicator IS built and nobody has measured its buffers. A
    refusal, never a zero -- see the term's comment in ``engine.py``.

``priced``
    A measurement exists for the communicator set named by ``nccl_signature``.
    A measured ZERO is a legitimate value of this state and is NOT the same
    statement as the one below.

``NOT_APPLICABLE``
    No NCCL communicator is CONSTRUCTED for this launch at all, so there is
    nothing to allocate and nothing to measure. This is a statement about the
    configuration, not about a measurement, which is why it may not be
    collapsed into a measured zero: a measured zero says "we looked and found
    none", NOT_APPLICABLE says "there is no communicator to look at". The two
    invalidate differently -- a measured zero is void as soon as the
    communicator set changes, while NOT_APPLICABLE is void as soon as the
    TRANSPORT changes.

PER GROUP, NOT PER BOOT. One launch builds several groups (world, tp, pp, dcp,
the MoE groups) and they do not have to agree: barlink attaches to groups with
more than one rank, while a single-rank group builds no device communicator at
all, and a group constructed with ``use_pynccl=False`` never had one. So the
verdict is computed per group and composed: any group that builds NCCL puts the
whole card back into the UNBOUNDED/priced pair, and only a launch where NO
group builds one is NOT_APPLICABLE. An unresolvable group is UNBOUNDED and is
named, because "we could not tell" must not read like "there is none".

WHY THE PREDICATE IS IMPORTED AND NOT RESTATED. ``should_build_barlink`` and
``should_build_pynccl`` in ``parallel_state`` are the conditions the
GroupCoordinator itself branches on when it skips PyNccl construction. If this
module re-implemented them, a future change to the real condition would leave
the ledger pricing 0 for a communicator that is once again being built -- an
under-charge, i.e. an OOM at the far end. The import is also what makes the
call-through testable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Iterable, Optional, Sequence, Tuple

__all__ = [
    "CommunicatorGroup",
    "GroupVerdict",
    "classify_communicator_groups",
    "nccl_signature",
]


@dataclasses.dataclass(frozen=True)
class CommunicatorGroup:
    """One process group a launch builds, described the way the constructor
    sees it: a name, how many ranks it spans, and whether the caller asked for
    a PyNccl communicator on it.

    Deliberately a plain description rather than a live ``GroupCoordinator``:
    the ledger runs during argument parsing, when no group exists yet.
    """

    #: The group's name as ``GroupCoordinator.unique_name`` would render it
    #: ("tp", "world", "dcp", ...). Used only to name a refusal.
    name: str
    #: Number of ranks in the group.
    world_size: int
    #: What the GroupCoordinator is constructed with. False for groups that
    #: never had a device communicator to begin with.
    use_pynccl: bool = True


@dataclasses.dataclass(frozen=True)
class GroupVerdict:
    """Whether *one* group allocates NCCL communicator buffers.

    ``builds_nccl`` is tri-state on purpose: ``None`` is "not resolvable
    here", which the ledger turns into an UNBOUNDED entry naming this group.
    """

    name: str
    builds_nccl: Optional[bool]
    reason: str

    @property
    def resolved(self) -> bool:
        return self.builds_nccl is not None


def classify_communicator_groups(
    groups: Iterable[CommunicatorGroup],
) -> Tuple[GroupVerdict, ...]:
    """One :class:`GroupVerdict` per group, via the REAL construction
    predicates.

    The import is lazy and its failure is not fatal but UNRESOLVED: a tree
    where ``parallel_state`` cannot be imported is a tree where this module
    cannot answer, and an unanswerable term refuses rather than prices 0.
    """
    described: Sequence[CommunicatorGroup] = tuple(groups)
    try:
        from sglang.srt.distributed.parallel_state import (
            should_build_barlink,
            should_build_pynccl,
        )
    except Exception as e:  # pragma: no cover - import-environment differences
        return tuple(
            GroupVerdict(
                name=g.name,
                builds_nccl=None,
                reason=(
                    "the construction predicates in "
                    "sglang.srt.distributed.parallel_state could not be "
                    f"imported ({e}), so which transport owns this group is "
                    "not decidable here"
                ),
            )
            for g in described
        )

    out = []
    for g in described:
        barlink_owns = should_build_barlink(g.world_size)
        builds = should_build_pynccl(g.use_pynccl, g.world_size, barlink_owns)
        if builds:
            reason = (
                f"should_build_pynccl(use_pynccl={g.use_pynccl}, "
                f"world_size={g.world_size}, barlink_active={barlink_owns}) is "
                "True: this group constructs a PyNccl communicator"
            )
        elif barlink_owns:
            reason = (
                f"barlink owns group {g.name!r} (SGLANG_BARLINK is set, "
                f"world_size={g.world_size}), so GroupCoordinator skips PyNccl "
                "communicator construction for it -- the boot log line "
                "\"barlink is active for group '...': skipping PyNccl "
                'communicator construction"'
            )
        elif g.world_size <= 1:
            reason = (
                f"group {g.name!r} spans a single rank, which builds no device "
                "communicator"
            )
        else:
            reason = (
                f"group {g.name!r} is constructed with use_pynccl=False and "
                "therefore never builds a PyNccl communicator"
            )
        out.append(GroupVerdict(name=g.name, builds_nccl=bool(builds), reason=reason))
    return tuple(out)


def nccl_signature(groups: Iterable[CommunicatorGroup]) -> str:
    """What a measured NCCL figure is valid FOR: this launch's communicator set.

    Keyed on the groups that actually BUILD a communicator, plus each one's
    rank count -- the two things libnccl sizes its buffers from. A group that
    barlink owns, or that spans one rank, or that was constructed with
    ``use_pynccl=False``, allocates nothing and therefore may not move the
    signature: if it did, switching barlink on for an unrelated group would
    invalidate a measurement that is still exactly correct.

    The inverse matters more and is the reason this is a digest of the verdicts
    rather than of the ServerArgs: adding a TP rank, or turning barlink off for
    the TP group, DOES change what libnccl allocates, and both change the
    verdict set, so both invalidate the measurement automatically. A stale
    figure cannot survive a change to the thing it measured.

    An UNRESOLVED group (the predicates could not be imported) poisons the
    signature deliberately -- ``unresolved`` sorts into the digest and no
    measurement taken under a resolvable tree will match it, so the term stays
    UNBOUNDED instead of being priced from a launch we could not classify.
    """
    # Materialise once: ``groups`` is an Iterable, and classifying it would
    # otherwise exhaust a generator before the zip could read it again.
    described: Sequence[CommunicatorGroup] = tuple(groups)
    parts = []
    for verdict, group in zip(classify_communicator_groups(described), described):
        if verdict.builds_nccl is None:
            parts.append(f"{group.name}:unresolved")
        elif verdict.builds_nccl:
            parts.append(f"{group.name}:nccl:{group.world_size}")
    if not parts:
        # No group builds a communicator. NOT_APPLICABLE is decided by the
        # term, not here; this is just the stable name of that empty set.
        return "no-nccl"
    blob = json.dumps(sorted(parts), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]
