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

Cut 1, which is what exists today
---------------------------------

*   :mod:`~sglang.srt.memtier.tiers` -- identity, the record, and the
    volatility law that makes admission a refusal rather than a ranking.
*   :mod:`~sglang.srt.memtier.profile` -- the measured numbers, as a JSON
    document for ONE rig, loadable and overridable, plus the live facts that
    make capacity current.
*   :mod:`~sglang.srt.memtier.registry` -- enumerate, filter, and a named
    refusal for every tier that did not make the list.
*   :mod:`~sglang.srt.memtier.reservations` -- a reservation is a NAMED post
    in the ledger that already owns the bytes; there is no second accounting.
*   :mod:`~sglang.srt.memtier.probe` -- the measurement catalogue, every arm
    declared and every arm refusing, so an absence is a work item rather than
    a blank.

No consumer reads any of this yet. The migrations are cut 2 and later, in the
order DESIGN_407 §5 gives.

Import weight: stdlib plus ``msgspec`` at module scope. NVML and torch are
imported lazily, inside the two functions that need a driver.
"""

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
    parse_tier_id,
)

__all__ = [
    "BUNDLED_PROFILE_PATH",
    "CardFact",
    "FilesystemFact",
    "InMemoryTierLedger",
    "LocalFacts",
    "PROBES",
    "PayloadClass",
    "ProbeOutcome",
    "ProbeSpec",
    "ProbeTarget",
    "ProfileError",
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
    "bind_device_tiers",
    "blob_tier_id",
    "bundled_profile",
    "collect_local_facts",
    "device_tier_id",
    "filesystem_tier_id",
    "host_tier_id",
    "load_profile",
    "missing_measurements",
    "parse_tier_id",
    "probes_for",
    "require_measured",
    "run_probe",
]
