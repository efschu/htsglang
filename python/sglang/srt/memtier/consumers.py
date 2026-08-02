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
"""One read-only shim, so "consumable" is a test and not a claim (#407).

No consumer is migrated in this slice. That is the cut plan, and it is also
where #421 F6 came from: a package with a complete API, complete tests and
zero callers, whose consumability nobody had ever demonstrated. A design
document asserting that expert offload *could* call the registry is exactly
the kind of claim that turns out to be false on the day somebody tries.

So this module contains one function, shaped to the question the largest byte
mover in the fork actually asks -- expert offload's five ``device="cpu"`` plus
``.pin_memory()`` literals (#77/#123, DESIGN_407 §2.1) -- returning the answer
in the registry's own vocabulary. It is exercised only by tests. Nothing in
``layers/moe/`` imports it, and this docstring is where that stops being true
when cut 5 lands.

What the shim proves, concretely:

*   a caller with a *payload class* and a *byte count* and nothing else can
    reach a target list. It does not need to know a tier id, a host name, a
    profile or a probe;
*   the answer is ordered, every rejection is named, and the caller can print
    the whole thing into its own error;
*   the refusal path works: a rig where nothing is admissible produces a
    sentence naming every tier and why, rather than an empty list the caller
    has to interpret.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import msgspec

from sglang.srt.memtier.registry import TierQuery, TierRegistry, TierSelection
from sglang.srt.memtier.tiers import PayloadClass, TierId, TierKind

__all__ = ["HostTierAnswer", "expert_offload_host_targets"]


class HostTierAnswer(msgspec.Struct, frozen=True, kw_only=True):
    """Where an expert shard may rest, in the caller's terms.

    ``tier_id`` is ``None`` when nothing is admissible; ``refusal`` is then the
    itemised sentence, never an empty string. The two are exclusive by
    construction so a caller cannot read a target off a failed answer.
    """

    tier_id: Optional[TierId] = None
    #: The rest of the ordered list, so a caller that dislikes the head can
    #: walk down it without re-querying.
    alternatives: Tuple[TierId, ...] = ()
    refusal: str = ""
    selection: Optional[TierSelection] = None

    @property
    def ok(self) -> bool:
        return self.tier_id is not None

    def to_json(self) -> Dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "alternatives": list(self.alternatives),
            "refusal": self.refusal,
            "selection": None if self.selection is None else self.selection.to_json(),
        }


def expert_offload_host_targets(
    registry: TierRegistry,
    *,
    bytes_needed: int,
    origin: Optional[TierId] = None,
    require_measured_bandwidth: bool = False,
) -> HostTierAnswer:
    """Where could a parked expert shard live? The #77/#123 question.

    The three arguments are everything expert offload knows at the point it
    currently writes ``device="cpu"``: how many bytes, which card they came
    off, and whether the caller is about to act on the bandwidth number or
    only to place bytes somewhere.

    Every other constraint is derived here rather than asked for, and each one
    is a fact about the payload rather than about this rig:

    * the payload class is ``EXPENSIVE_RECONSTRUCTABLE``. An expert shard can
      be rebuilt -- it is on disk in the checkpoint -- but rebuilding it means
      re-reading and re-quantising weights mid-decode, so it is evacuated and
      never dropped;
    * the object class is ``"experts"``, the ``OFFLOAD_CLASSES`` member the
      offload register already uses, so a tier that does not admit experts is
      refused BY NAME instead of being silently ranked last;
    * device tiers are excluded. Not because peer VRAM is a bad idea -- #286's
      ladder puts it first -- but because *this* question is the one the five
      ``device="cpu"`` literals ask, and widening it to peer VRAM is cut 4's
      job, after the capacity ledger is one ledger.
    """
    query = TierQuery(
        payload=PayloadClass.EXPENSIVE_RECONSTRUCTABLE,
        bytes_needed=int(bytes_needed),
        object_class="experts",
        origin=origin,
        kinds=(TierKind.HOST, TierKind.FILESYSTEM),
        require_measured_bandwidth=require_measured_bandwidth,
        allow_unmeasured_bandwidth=not require_measured_bandwidth,
    )
    selection = registry.select(query)
    if not selection.candidates:
        return HostTierAnswer(
            refusal=(
                f"no tier can hold {bytes_needed} bytes of parked expert "
                f"weights on profile {registry.profile_id!r}:\n" + selection.render()
            ),
            selection=selection,
        )
    head, *rest = selection.candidates
    return HostTierAnswer(
        tier_id=head.tier.id,
        alternatives=tuple(c.tier.id for c in rest),
        selection=selection,
    )
