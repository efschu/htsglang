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
"""Consumer-shaped entry points into the registry (#407).

Cut 1 shipped one READ-ONLY shim, so "consumable" was a test and not a claim
-- that is where #421 F6 came from: a package with a complete API, complete
tests and zero callers, whose consumability nobody had ever demonstrated.
:func:`expert_offload_host_targets` is that shim; it is exercised only by
tests, nothing in ``layers/moe/`` imports it, and that stops being true when
cut 5 lands.

:func:`checkpoint_tier_targets` is the WRITE-PATH selection helper cut 3
owes, and #410 (server-side session checkpoints) is its first PRODUCTION
caller -- the first production caller the package has ever had. The #421 pin
test that asserted "zero consumers" is retired in the same merge and replaced
by a positive pin on this call site, mirroring what #394's cold tier did.

What the shims prove, concretely:

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

__all__ = [
    "CheckpointTierPolicy",
    "HostTierAnswer",
    "checkpoint_tier_targets",
    "expert_offload_host_targets",
]


class HostTierAnswer(msgspec.Struct, frozen=True, kw_only=True):
    """Where a payload may rest, in the caller's terms.

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


class CheckpointTierPolicy(msgspec.Struct, frozen=True, kw_only=True):
    """The VRAM -> RAM -> Disk demotion ladder for a session checkpoint.

    Age is the only dial, and it narrows the KINDS the query will consider
    rather than re-ranking them: a fresh checkpoint may rest on the card that
    produced it, an older one may not occupy VRAM a live session could use,
    and an old one has to be on something that survives the process. The
    registry still does all the picking within whatever kinds survive.

    The two thresholds are POLICY, not measurements -- they carry no
    provenance because there is nothing to measure about them. Every number
    the answer is ordered by (bandwidth, headroom) comes from the registry
    and keeps its own provenance, which is what
    :attr:`HostTierAnswer.selection` carries back to the caller.
    """

    #: Beyond this age a checkpoint stops competing for device VRAM.
    vram_max_age_s: float = 60.0
    #: Beyond this age it must be on a tier that survives process exit.
    host_max_age_s: float = 900.0

    def kinds_for(self, age_s: float) -> Tuple[TierKind, ...]:
        if age_s <= self.vram_max_age_s:
            return (TierKind.DEVICE, TierKind.HOST, TierKind.FILESYSTEM)
        if age_s <= self.host_max_age_s:
            return (TierKind.HOST, TierKind.FILESYSTEM)
        return (TierKind.FILESYSTEM, TierKind.BLOB)

    def payload_for(self, age_s: float, *, durable: bool) -> PayloadClass:
        if durable or age_s > self.host_max_age_s:
            return PayloadClass.PERSISTENCE_REQUIRED
        return PayloadClass.EXPENSIVE_RECONSTRUCTABLE


def checkpoint_tier_targets(
    registry: TierRegistry,
    *,
    bytes_needed: int,
    age_s: float = 0.0,
    durable: bool = False,
    origin: Optional[TierId] = None,
    policy: Optional[CheckpointTierPolicy] = None,
) -> HostTierAnswer:
    """Where may a session checkpoint rest? The #410 question.

    A checkpoint is a session's KV pages plus its GDN blob, serialised by the
    #261 snapshot. Three properties of that payload decide the query, and each
    is a fact about the payload rather than about this rig:

    * with ``durable=False`` the payload class is
      ``EXPENSIVE_RECONSTRUCTABLE``. A checkpoint can be rebuilt -- by
      re-prefilling the conversation -- but rebuilding it means redoing
      user-visible work, which is #224's whole point, so it is evacuated down
      the ladder and never silently dropped. With ``durable=True`` (or past
      ``host_max_age_s``) it becomes ``PERSISTENCE_REQUIRED``: a checkpoint
      the user expects to branch from tomorrow has to outlive the process,
      and only a ``PERSISTENT`` tier admits that class;
    * no ``object_class`` is passed. ``OFFLOAD_CLASSES`` has no member for KV
      pages, and ``TierQuery.object_class`` documents that case exactly --
      "Hibernate images and HiCache pages do not [have one], and are gated by
      volatility alone". Inventing a member would make every existing tier
      record refuse checkpoints by name until each ``admits`` set was edited;
    * the GDN blob does NOT make this ``DEVICE_BOUND``. That class exists
      because a LOSSY or REORDERED round trip of recurrent state is a
      correctness failure (DESIGN_407 X2), and it would pin every checkpoint
      to the origin card, which is the opposite of a checkpoint. The #261
      route is neither lossy nor reordered: the GDN state travels as its own
      explicitly named ``.mamba`` blob through the content-addressed store
      (the #212 lesson), and that path was gate-proven byte-identical against
      a never-moved reference. The gate that keeps this honest is
      ``validate_manifest_completeness``: a hybrid-GDN manifest without its
      mamba blob is refused, loudly, before anything is written.

    Absent tiers refuse rather than rank last, and a rig where nothing is
    admissible produces the itemised sentence, never an empty list.
    """
    policy = policy if policy is not None else CheckpointTierPolicy()
    query = TierQuery(
        payload=policy.payload_for(age_s, durable=durable),
        bytes_needed=int(bytes_needed),
        origin=origin,
        kinds=policy.kinds_for(age_s),
        # A checkpoint write is not on the decode path, so an unmeasured link
        # is usable -- but it is still reported as unmeasured in the
        # candidate's notes rather than being quietly treated as fast.
        allow_unmeasured_bandwidth=True,
    )
    selection = registry.select(query)
    if not selection.candidates:
        return HostTierAnswer(
            refusal=(
                f"no tier can hold a {bytes_needed}-byte session checkpoint "
                f"(age {age_s:.0f}s, durable={durable}) on profile "
                f"{registry.profile_id!r}:\n" + selection.render()
            ),
            selection=selection,
        )
    head, *rest = selection.candidates
    return HostTierAnswer(
        tier_id=head.tier.id,
        alternatives=tuple(c.tier.id for c in rest),
        selection=selection,
    )
