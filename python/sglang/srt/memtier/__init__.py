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
"""One memory-tier registry, many spill consumers (#407).

The charter, as stated: *every memory you have access to must be a level
cache, a spill target or an offload target depending on volatility -- disk,
RAM, VRAM, local as well as remote.* Today the tiers exist, fragmented: expert
offload owns VRAM plus pinned host RAM, HiCache owns L1/L2/L3, hibernate owns
a directory, #224 owns a destination chain, #286 owns a park-target ladder,
#389 designs an NVMe rung and #305 declares a residency ladder with no
mechanism. Four disjoint tier vocabularies, and the largest byte mover in the
fork -- expert offload -- is in none of them.

This package is the NODE layer they can share: what memories exist, how big
each is, what each may hold, what each costs, and whether it is reachable.
The EDGE layer (what a byte costs between two places) is not re-implemented
here; ``planner.cost_model`` owns it.

The node layer
--------------

*   :mod:`~sglang.srt.memtier.tiers` -- identity, the record, the volatility
    law that makes admission a refusal rather than a ranking, and the link
    identity #423's striping gate needs.
*   :mod:`~sglang.srt.memtier.profile` -- measured numbers as a JSON document,
    plus the live facts that make capacity current.
*   :mod:`~sglang.srt.memtier.registry` -- enumerate, filter, and a named
    refusal for every tier that did not make the list.
*   :mod:`~sglang.srt.memtier.reservations` -- a reservation is a NAMED post
    in the ledger that already owns the bytes; there is no second accounting.
*   :mod:`~sglang.srt.memtier.probe` -- the measurement catalogue, every arm
    declared, so an absence is a work item rather than a blank.

Generality (directive #434)
---------------------------

No number in this package is a constant about any particular machine, and the
enforcement is structural rather than editorial:

*   :mod:`~sglang.srt.memtier.fingerprint` -- what hardware is this, computed
    purely from facts handed in, keyed on NVML UUID (#397 canon);
*   :mod:`~sglang.srt.memtier.profile_store` -- a stored profile is applied
    only to hardware it matches, and a MODEL-only match licenses card
    templates and no host, filesystem or remote tier;
*   :mod:`~sglang.srt.memtier.bootstrap` -- what an unknown machine gets:
    sizes measured live, every cost ABSENT and naming the probe that fills it;
*   :mod:`~sglang.srt.memtier.adapters` -- the existing #213 / #271 / #278
    artifacts, read into tier records. #407 adds no probe of its own.

:meth:`TierRegistry.for_machine` is the entry point that composes all four.
``TierRegistry.from_profile`` no longer defaults to the bundled rig record --
that default was #421 F6's quiet half, and it is now a required argument.

No consumer reads any of this yet; :mod:`~sglang.srt.memtier.consumers` holds
one read-only shim, exercised by tests, so that "consumable" is demonstrated
rather than asserted. The migrations are cut 2 and later, in the order
``DESIGN_407_memtier_registry.md`` §5 gives.

Import weight: stdlib plus ``msgspec`` at module scope. NVML, torch and
``socket`` are imported lazily, inside the functions that need them.
"""

from sglang.srt.memtier.adapters import (
    ARTIFACT_ROUTES,
    ArtifactRoute,
    IngestReport,
    apply_outcomes,
    from_capability_matrix,
    from_card_probe,
    from_rig_artifact,
)
from sglang.srt.memtier.bootstrap import (
    BOOTSTRAP_PROFILE_ID,
    bootstrap_profile,
    bootstrap_tiers,
    collect_fs_types,
    default_host_name,
    filesystem_kind_properties,
)
from sglang.srt.memtier.consumers import (
    HostTierAnswer,
    expert_offload_host_targets,
)
from sglang.srt.memtier.fingerprint import (
    HardwareFingerprint,
    MatchScope,
    ProfileMatch,
    fingerprint_from_facts,
    licensed_document,
    match_profile,
)
from sglang.srt.memtier.probe import (
    PROBES,
    ProbeOutcome,
    ProbeSpec,
    ProbeTarget,
    ProvenanceUpgradeRefused,
    UnimplementedProbe,
    apply_outcome,
    missing_measurements,
    probes_for,
    require_measured,
    run_probe,
)
from sglang.srt.memtier.profile import (
    BUNDLED_PROFILE_PATH,
    CardFact,
    FilesystemFact,
    LocalFacts,
    ProfileError,
    RigProfile,
    apply_local_facts,
    bind_device_tiers,
    bundled_profile,
    collect_local_facts,
    load_profile,
)
from sglang.srt.memtier.profile_store import (
    ProfileSelection,
    candidate_paths,
    profile_store_dir,
    save_profile,
    select_profile,
)
from sglang.srt.memtier.registry import (
    Refusal,
    RefusalRule,
    TierCandidate,
    TierQuery,
    TierRegistry,
    TierSelection,
    UnknownTier,
)
from sglang.srt.memtier.reservations import (
    InMemoryTierLedger,
    TierLedger,
    TierPost,
    TierReservation,
    TierReservationRejected,
    UnnamedPost,
    VramLedgerHook,
)
from sglang.srt.memtier.tiers import (
    LinkDisjointness,
    LinkVerdict,
    PayloadClass,
    TierCapacity,
    TierCaps,
    TierDescriptor,
    TierHealth,
    TierId,
    TierIdError,
    TierKind,
    TierTransport,
    Volatility,
    admission_refusal,
    blob_tier_id,
    device_tier_id,
    filesystem_tier_id,
    host_tier_id,
    link_disjointness,
    parse_tier_id,
)

__all__ = [
    "ARTIFACT_ROUTES",
    "BOOTSTRAP_PROFILE_ID",
    "BUNDLED_PROFILE_PATH",
    "ArtifactRoute",
    "CardFact",
    "FilesystemFact",
    "HardwareFingerprint",
    "HostTierAnswer",
    "InMemoryTierLedger",
    "IngestReport",
    "LinkDisjointness",
    "LinkVerdict",
    "LocalFacts",
    "MatchScope",
    "PROBES",
    "PayloadClass",
    "ProbeOutcome",
    "ProbeSpec",
    "ProbeTarget",
    "ProfileError",
    "ProfileMatch",
    "ProfileSelection",
    "ProvenanceUpgradeRefused",
    "Refusal",
    "RefusalRule",
    "RigProfile",
    "TierCandidate",
    "TierCapacity",
    "TierCaps",
    "TierDescriptor",
    "TierHealth",
    "TierId",
    "TierIdError",
    "TierKind",
    "TierLedger",
    "TierPost",
    "TierQuery",
    "TierRegistry",
    "TierReservation",
    "TierReservationRejected",
    "TierSelection",
    "TierTransport",
    "UnimplementedProbe",
    "UnknownTier",
    "UnnamedPost",
    "VramLedgerHook",
    "Volatility",
    "admission_refusal",
    "apply_local_facts",
    "apply_outcome",
    "apply_outcomes",
    "bind_device_tiers",
    "blob_tier_id",
    "bootstrap_profile",
    "bootstrap_tiers",
    "bundled_profile",
    "candidate_paths",
    "collect_fs_types",
    "collect_local_facts",
    "default_host_name",
    "device_tier_id",
    "expert_offload_host_targets",
    "filesystem_kind_properties",
    "filesystem_tier_id",
    "fingerprint_from_facts",
    "from_capability_matrix",
    "from_card_probe",
    "from_rig_artifact",
    "host_tier_id",
    "licensed_document",
    "link_disjointness",
    "load_profile",
    "match_profile",
    "missing_measurements",
    "parse_tier_id",
    "probes_for",
    "profile_store_dir",
    "require_measured",
    "run_probe",
    "save_profile",
    "select_profile",
]
