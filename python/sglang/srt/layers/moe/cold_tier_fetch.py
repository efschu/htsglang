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
"""Routing for the shared cold expert tier (#394 slice 2).

Slice 1 (:mod:`sglang.srt.layers.moe.cold_tier_shm`) built the STORAGE: named
POSIX segments in host DRAM, identity-keyed by card UUID and PCI BDF, sealed
headers, ``PROT_READ`` peer views, and a manifest published as a file so no
rank needs a collective to find a peer's rows. Nothing imported it -- audit
#421 recorded the chain as INERT (declared), finding F4.

This module is the ROUTING half, and it answers exactly two questions:

1. **Who owns this cold expert?** The owner is read off a plan that is a pure
   function of ``(cold pool ids, link-proportional weights)`` -- see
   :class:`ColdTierAssignment`. Every rank computes the identical map from
   identical inputs, so the answer is rank-local to compute and rank-uniform by
   construction. That matters beyond tidiness: the fetch feeds a wave whose
   result is combined across the TP group, so a branch taken from LOCAL state
   would put two ranks on different code paths inside a collective family. The
   plan carries a :meth:`ColdTierAssignment.digest` precisely so a test can pin
   that uniformity rather than assert it in prose.

2. **Where are its bytes?** In the owner's segment, at the row its manifest
   names. :class:`ColdTierResolver` maps the segment once per process and hands
   back a zero-copy ``torch`` view over the peer's DRAM
   (:func:`cold_tier_shm.peer_row_tensor`). The existing fetch path then issues
   the same ``copy_`` it always did -- over THIS rank's own PCIe link, which is
   the only link a rank can pull over.

What this changes and what it does not
--------------------------------------

It removes a failure, not a byte. Before slice 2 a delegated cold expert was
absent: ``partition_cold_experts`` dropped it from the owner's spill pool and
nothing else held it, so the first token routed to it died inside
``ExpertResidencyPlanner.resolve`` (measured 2026-08-02, V4-Flash TP=3, all
three ranks). With a shared tier the same expert is REACHABLE: whichever rank
holds the bytes, any rank can DMA the row.

Be precise about the consequence for PCIe seconds, because the #394 header
comment invites a stronger reading than the mechanism supports. Moving BYTE
OWNERSHIP does not move H2D traffic: a rank that must compute expert ``e``
still pulls ``e`` across its own link, whether the bytes came from its own
pinned pool or from a peer's segment. What link-proportional ownership buys on
its own is host-DRAM placement and the reachability above. Path A' of
ANALYSE_393 §7.3 -- 2.79 GB over the group's aggregate 32.4 GB/s instead of
0.93 GB over one 6.4 GB/s link -- additionally requires the COMPUTE assignment
to move, i.e. which rank runs which expert, which is the #82 expert-range
question and is not this slice. The proof-window arm in
``scripts/dev/394_s2_proof/`` is written to measure per-rank H2D so the window
can falsify that distinction instead of assuming either side of it.

Enablement
----------

Off unless ``SGLANG_MOE_COLD_TIER_SHM=1``. With it off, every entry point here
returns ``None`` and the caller keeps its previous allocation and its previous
refusal, field for field. The flag is also what makes delegation admissible on
a disjoint expert shard at all: the slice-1 boot refusal stays exactly as it
was for a launch that has not asked for a shared tier, because without one the
unsoundness it names is still real.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.layers.moe.cold_tier_shm import (
    HEADER_BYTES,
    ColdTierError,
    ColdTierLayout,
    create_owned_segment,
    manifest_path,
    peer_row_tensor,
    publish_manifest,
    read_peer_manifest,
    register_for_dma,
    seal_owned_segment,
)

logger = logging.getLogger(__name__)

__all__ = [
    "COLD_TIER_INSTANCE_ENV",
    "ColdTierAssignment",
    "ColdTierOwner",
    "ColdTierResolver",
    "ColdTierUnavailable",
    "assign_cold_experts",
    "cold_tier_enabled",
    "instance_id",
    "publish_cold_tier_instance",
    "reset_for_tests",
    "resolver_for_layer",
]

#: Launch-unique id shared by every rank of one server, published by the
#: launcher into the environment (the same collective-free channel
#: ``registry.rank_cards`` uses). It is what makes a leftover segment from a
#: previous run refuse instead of read.
COLD_TIER_INSTANCE_ENV = "SGLANG_MOE_COLD_TIER_INSTANCE"


class ColdTierUnavailable(ColdTierError):
    """The shared tier cannot serve this layer, with the reason.

    Distinct from :class:`~sglang.srt.layers.moe.cold_tier_shm.SegmentMismatch`
    on purpose: a mismatch means somebody is about to read wrong bytes, this
    means nobody can read at all. The first must never be swallowed; the second
    is a legitimate "stay on the previous path" for a caller that has one.
    """


# ---------------------------------------------------------------------------
# enablement + launch identity
# ---------------------------------------------------------------------------


def cold_tier_enabled() -> bool:
    """Is the shared cold tier switched on for this launch?

    Read through ``environ.envs`` so the flag is inventoried with the rest, but
    tolerant of an import-time-less desk context.
    """
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_MOE_COLD_TIER_SHM.get())
    except Exception:  # noqa: BLE001 - a desk import must not decide policy
        return os.environ.get("SGLANG_MOE_COLD_TIER_SHM", "") in ("1", "true", "True")


def publish_cold_tier_instance() -> str:
    """Mint the launch id in the LAUNCHER, before any worker is spawned.

    Called from ``_launch_subprocesses`` next to the rank->card publication and
    for the same reason: the channel is the environment, so it only reaches a
    spawned scheduler if it is set before the spawn loop. A hand-set value is
    never overwritten, which is what lets an operator re-attach to a segment
    set on purpose.
    """
    existing = os.environ.get(COLD_TIER_INSTANCE_ENV, "").strip()
    if existing:
        return existing
    import uuid as _uuid

    minted = _uuid.uuid4().hex[:16]
    os.environ[COLD_TIER_INSTANCE_ENV] = minted
    return minted


def instance_id() -> str:
    """This launch's id, or a named refusal.

    Deliberately NOT minted here. A worker that minted its own id would build a
    segment set no peer can name, and the failure would surface as an
    unreadable manifest at the first fetch -- late, and one process at a time.
    """
    value = os.environ.get(COLD_TIER_INSTANCE_ENV, "").strip()
    if not value:
        raise ColdTierUnavailable(
            f"{COLD_TIER_INSTANCE_ENV} is not set. The launcher publishes it "
            "before spawning schedulers (entrypoints/engine.py); a worker must "
            "not mint its own, because the peers would then name different "
            "segments and only discover it at the first fetch"
        )
    return value


# ---------------------------------------------------------------------------
# 1. the rank-uniform plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColdTierAssignment:
    """Which rank owns the bytes of each expert in one layer's cold pool.

    ``cold_ids`` is the WHOLE pool -- this rank's cold experts before the
    link-proportional split -- ascending. ``owners[i]`` is the rank that holds
    ``cold_ids[i]``. Both are pure functions of ``(cold_ids, weights)``, so two
    ranks handed the same pool and the same ratio produce byte-identical maps
    without exchanging a message.
    """

    rank: int
    world_size: int
    cold_ids: Tuple[int, ...]
    owners: Tuple[int, ...]
    ratio_source: str
    ratio_provenance: str
    weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.cold_ids) != len(self.owners):
            raise ValueError(
                f"assignment names {len(self.cold_ids)} cold experts but "
                f"{len(self.owners)} owners"
            )
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank {self.rank} outside [0,{self.world_size})")
        if self.ratio_provenance == "absent":
            raise ValueError(
                "a cold-tier assignment may not be weighted by an absent "
                "provenance -- an equal split is the default path and needs no "
                "shared tier (#394 provenance chain: env > card probe > NVML "
                "negotiated width > refusal)"
            )

    @property
    def owned_ids(self) -> Tuple[int, ...]:
        return tuple(e for e, o in zip(self.cold_ids, self.owners) if o == self.rank)

    @property
    def remote_ids(self) -> Tuple[int, ...]:
        return tuple(e for e, o in zip(self.cold_ids, self.owners) if o != self.rank)

    def owner_of(self, expert_id: int) -> int:
        """Owning rank, or a named error. Never a guess and never a fallback.

        A missing id here means the caller built the assignment from a
        different pool than the one it is now routing -- exactly the drift that
        would otherwise show up as a plausible-looking wrong row.
        """
        expert_id = int(expert_id)
        try:
            return self.owners[self.cold_ids.index(expert_id)]
        except ValueError:
            raise ColdTierUnavailable(
                f"expert {expert_id} is not in this layer's cold pool "
                f"{list(self.cold_ids)[:8]}{'...' if len(self.cold_ids) > 8 else ''}; "
                "the assignment and the staging plan describe different pools"
            ) from None

    def digest(self) -> str:
        """Stable fingerprint of the PLAN, identical on every rank.

        ``rank`` is excluded on purpose: this is the group-wide decision, and
        the point of the digest is that a test (or the #431 recorder) can
        compare it across simulated ranks and see one value. A per-rank field
        in it would make every comparison trivially pass.
        """
        payload = "|".join(
            (
                str(self.world_size),
                ",".join(str(e) for e in self.cold_ids),
                ",".join(str(o) for o in self.owners),
                self.ratio_source,
                ",".join(f"{w:.12g}" for w in self.weights),
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def assign_cold_experts(
    cold_ids: Sequence[int], ratio, rank: int, world_size: int
) -> ColdTierAssignment:
    """Build the owner map from the link-proportional partition.

    Uses the SAME ``partition_cold_experts`` the staging plan uses, so the set
    a rank stages and the set the router believes it owns cannot drift: they
    are two reads of one function.
    """
    from sglang.srt.layers.moe.expert_offload import partition_cold_experts

    ids = tuple(int(e) for e in cold_ids)
    shares = partition_cold_experts(ids, ratio.weights)
    owner_of: Dict[int, int] = {}
    for owner, share in enumerate(shares):
        for expert_id in share:
            owner_of[int(expert_id)] = owner
    return ColdTierAssignment(
        rank=int(rank),
        world_size=int(world_size),
        cold_ids=ids,
        owners=tuple(owner_of[e] for e in ids),
        ratio_source=ratio.source,
        ratio_provenance=ratio.provenance,
        weights=tuple(float(w) for w in ratio.weights),
    )


# ---------------------------------------------------------------------------
# 2. owner side -- the spill pool IS the segment
# ---------------------------------------------------------------------------

#: layouts this process owns, across all layers, so one manifest names them
#: all. A dict rather than a list: re-staging a layer replaces its entry rather
#: than publishing two layouts for one segment.
_OWNED: Dict[Tuple[str, str], ColdTierLayout] = {}
_OWNED_LOCK = threading.RLock()
_OWNED_MAPS: List[object] = []


def _row_bytes(row_shape: Sequence[int], itemsize: int) -> int:
    n = 1
    for dim in row_shape:
        n *= int(dim)
    return n * int(itemsize)


class ColdTierOwner:
    """Allocates one layer's cold rows IN the shared segment, not beside it.

    The important word is IN. A staging path that allocated a private pinned
    pool and then copied it into a segment would double the host footprint of
    the cold tier, which on the reference rig is the resource that ran out
    first (76 -> 108 GB was needed for the 122B run). So
    :meth:`allocate_spill_pool` hands back a tensor whose storage IS the
    segment payload: the owner writes its experts exactly where a peer will
    later read them, one copy of the bytes, as slice 1's docstring promised.

    ``seal`` is called once per layer, after every row is written, and it is
    what turns "mapped" into "readable" -- before it, the header magic is zero
    and every peer refuses.
    """

    def __init__(
        self,
        instance: str,
        rank: int,
        layer_key: str,
        card_uuid: str,
        pci_bdf: str = "",
    ):
        if not card_uuid:
            raise ColdTierUnavailable(
                "the shared cold tier needs this rank's card UUID (#397): a "
                "segment named by rank index alone cannot be identified across "
                "process boundaries. Publish the rank->card vector "
                "(--rank-gpu-id, or SGLANG_RANK_CARD_PROBE_CUDA=1) or leave the "
                "tier off"
            )
        self.instance = instance
        self.rank = int(rank)
        self.layer_key = str(layer_key)
        self.card_uuid = card_uuid
        self.pci_bdf = pci_bdf or ""
        self._maps: Dict[str, object] = {}
        self._layouts: Dict[str, ColdTierLayout] = {}

    def allocate_spill_pool(
        self, param_attr: str, spill_ids: Sequence[int], row_shape, dtype
    ):
        """A ``[rows, *row_shape]`` tensor backed by this layer's segment.

        ``None`` when the rank owns no cold rows for this tensor -- a legal
        state under a lopsided ratio, and the caller already handles a ``None``
        spill pool because a fully resident layer produces one.
        """
        import torch

        ids = tuple(int(e) for e in spill_ids)
        if not ids:
            return None
        shape = tuple(int(d) for d in row_shape)
        dtype_name = str(dtype).rsplit(".", 1)[-1]
        layout = ColdTierLayout(
            instance_id=self.instance,
            owner_rank=self.rank,
            owner_card_uuid=self.card_uuid,
            owner_pci_bdf=self.pci_bdf,
            layer_key=self.layer_key,
            param_attr=param_attr,
            rows=len(ids),
            row_bytes=_row_bytes(shape, torch.empty((), dtype=dtype).element_size()),
            dtype=dtype_name,
            row_shape=shape,
            expert_ids=ids,
        )
        mm = create_owned_segment(layout)
        # Pin the owner's own mapping too: without it this rank's ordinary
        # spill fetch would become a pageable copy, i.e. the shared tier would
        # silently cost the local path its async H2D.
        register_for_dma(mm, f"owned:{layout.segment_name}")
        elems = len(ids)
        for dim in shape:
            elems *= int(dim)
        # The view starts at HEADER_BYTES: the header is written LAST, by
        # seal(), and must never be inside the tensor a caller writes rows
        # through.
        pool = torch.frombuffer(mm, dtype=dtype, count=elems, offset=HEADER_BYTES).view(
            (len(ids),) + shape
        )
        self._maps[param_attr] = mm
        self._layouts[param_attr] = layout
        return pool

    def seal(self) -> None:
        """Stamp every header of this layer and (re)publish the manifest.

        Republishing per layer rather than once at the end is deliberate: the
        write is an atomic rename of a few kB, a peer therefore always sees a
        complete manifest describing a prefix of the layers, and there is no
        end-of-load barrier anywhere in the design to hang on.
        """
        with _OWNED_LOCK:
            for attr, layout in self._layouts.items():
                seal_owned_segment(self._maps[attr], layout)
                _OWNED[(layout.layer_key, attr)] = layout
                _OWNED_MAPS.append(self._maps[attr])
            layouts = list(_OWNED.values())
            rank = self.rank
            instance = self.instance
        publish_manifest(instance, rank, layouts)


def owner_for_layer(
    layer_key: str, rank: int, world_size: int, card_uuids: Optional[Sequence[str]]
) -> Optional[ColdTierOwner]:
    """An owner handle for this layer, or ``None`` when the tier is off.

    ``None`` rather than a disabled handle: the caller's allocation then stays
    the ``torch.empty(...).pin_memory()`` it always was, with no branch taken
    inside the load loop on the default path.
    """
    if not cold_tier_enabled():
        return None
    if not card_uuids or len(card_uuids) != int(world_size):
        raise ColdTierUnavailable(
            f"the shared cold tier needs a rank->card vector of length "
            f"{world_size}; got {0 if not card_uuids else len(card_uuids)}. "
            "Publish it with --rank-gpu-id or leave SGLANG_MOE_COLD_TIER_SHM off"
        )
    return ColdTierOwner(
        instance=instance_id(),
        rank=int(rank),
        layer_key=str(layer_key),
        card_uuid=card_uuids[int(rank)],
    )


# ---------------------------------------------------------------------------
# 3. peer side -- resolve an expert to a zero-copy view of a peer's row
# ---------------------------------------------------------------------------

#: (instance, owner_rank) -> layouts, so a 43-layer model reads each peer's
#: manifest once per process instead of once per layer per fetch.
_MANIFESTS: Dict[Tuple[str, int], Dict[Tuple[str, str], ColdTierLayout]] = {}
_MANIFEST_LOCK = threading.RLock()


def _peer_layouts(instance: str, owner_rank: int, timeout_s: float):
    key = (instance, int(owner_rank))
    with _MANIFEST_LOCK:
        cached = _MANIFESTS.get(key)
    if cached is not None:
        return cached
    layouts = read_peer_manifest(instance, int(owner_rank), timeout_s=timeout_s)
    with _MANIFEST_LOCK:
        _MANIFESTS[key] = layouts
    return layouts


class ColdTierResolver:
    """One layer's view of the group's cold rows.

    Holds no bytes and no copies: :meth:`row` returns a ``torch`` view over the
    owner's ``PROT_READ`` mapping (slice-1 canon -- the mapping stays read-only
    and the view is built with ``torch.frombuffer``, because
    ``ctypes.from_buffer`` refuses a read-only map and
    ``UntypedStorage.from_buffer`` copies). The caller issues its own
    ``copy_`` into the scratch slot, over its own link.
    """

    def __init__(
        self,
        instance: str,
        assignment: ColdTierAssignment,
        layer_key: str,
        manifest_timeout_s: float = 30.0,
    ):
        self.instance = instance
        self.assignment = assignment
        self.layer_key = str(layer_key)
        self.manifest_timeout_s = float(manifest_timeout_s)

    @property
    def remote_ids(self) -> frozenset:
        return frozenset(self.assignment.remote_ids)

    def layout_for(self, param_attr: str, expert_id: int) -> ColdTierLayout:
        owner = self.assignment.owner_of(expert_id)
        if owner == self.assignment.rank:
            raise ColdTierUnavailable(
                f"expert {expert_id} is owned by this rank ({owner}); it must "
                "be served from the local spill pool, not through the peer "
                "path. Routing it here would read the rank's own segment "
                "through a second mapping for no reason"
            )
        layouts = _peer_layouts(self.instance, owner, self.manifest_timeout_s)
        layout = layouts.get((self.layer_key, param_attr))
        if layout is None:
            raise ColdTierUnavailable(
                f"rank {owner}'s manifest has no layout for "
                f"{self.layer_key}/{param_attr}; it names "
                f"{sorted(layouts)[:4]}. The owner staged a different set of "
                "tensors than this rank expects to fetch"
            )
        return layout

    def row(self, param_attr: str, expert_id: int):
        """Zero-copy host view of one delegated expert's row."""
        return peer_row_tensor(self.layout_for(param_attr, expert_id), expert_id)


def resolver_for_layer(layer) -> Optional[ColdTierResolver]:
    """Build the resolver a :class:`MoEExpertOffloadCache` needs, or ``None``.

    Reads what the load path stashed on the layer and nothing else, so the
    cache does not have to know how the plan was made. ``None`` means "no
    shared tier here", and the cache then keeps the slice-1 behaviour: a
    delegated expert that reaches the router is a named refusal.
    """
    if not cold_tier_enabled():
        return None
    assignment = getattr(layer, "_moe_cold_tier_assignment", None)
    layer_key = getattr(layer, "_moe_cold_tier_layer_key", None)
    if assignment is None or layer_key is None:
        return None
    timeout = 30.0
    try:
        from sglang.srt.environ import envs

        timeout = float(envs.SGLANG_MOE_COLD_TIER_MANIFEST_TIMEOUT_S.get())
    except Exception:  # noqa: BLE001 - the default is the documented one
        pass
    return ColdTierResolver(
        instance=instance_id(),
        assignment=assignment,
        layer_key=layer_key,
        manifest_timeout_s=timeout,
    )


def layer_key_for(layer) -> str:
    """Segment key for one MoE layer. Stable across ranks, unique per layer."""
    layer_id = getattr(layer, "layer_id", None)
    if layer_id is None:
        raise ColdTierUnavailable(
            "a cold-tier layer must have a layer_id; without one two layers "
            "would name the same segment and the second would overwrite the "
            "first's rows"
        )
    return f"L{int(layer_id)}"


def reset_for_tests() -> None:
    """Drop every process-level memo (owned layouts, peer manifests)."""
    from sglang.srt.layers.moe.cold_tier_shm import detach_all

    with _OWNED_LOCK:
        _OWNED.clear()
        _OWNED_MAPS.clear()
    with _MANIFEST_LOCK:
        _MANIFESTS.clear()
    detach_all()


def manifest_location(rank: int) -> str:
    """Where this rank's manifest lands. For logs and the preflight script."""
    return manifest_path(instance_id(), int(rank))
