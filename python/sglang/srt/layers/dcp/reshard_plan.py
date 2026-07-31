# Copyright 2023-2026 SGLang Team
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

"""Pure planning arithmetic for phase-boundary KV resharding (#297).

Given the set of LIVE global KV slot ids and an (old vector, new vector)
pair, derive -- per rank, from the weighted owner rule stated once in
``owner.py`` -- exactly which physical pool rows stay local (with a row
remap), which are sent to which peer, and which arrive from which peer.

Everything in this module is a pure function of its arguments: no process
group, no device requirement, no global state. That is deliberate and it is
the uniformity argument of the runtime built on top (managers/kv_reshard.py):
every rank computes the SAME transition from the SAME replicated inputs (the
radix tree's slot ids and the two vectors), so the pairwise transfer lists
agree across ranks by construction, without exchanging any plan metadata.
The consensus round only has to agree on (armed, ready, epoch, target
vector) -- never on the plan itself.

Determinism contract: ``build_transition`` requires the slot ids sorted
ascending and unique. Sender and receiver of a pair both enumerate the
pair's slots in this order, which is what makes the packed payload layout a
shared convention instead of transmitted metadata (a per-pair checksum
travels with each payload to keep this contract falsifiable at runtime).
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows


class KvReshardError(RuntimeError):
    """Loud, rank-uniform failure of the KV reshard family (#297)."""


def parse_reshard_vectors(spec: str) -> List[Tuple[int, ...]]:
    """Parse ``--kv-reshard-vectors`` (``"7,3,3;2,11,10"``) into vectors.

    Semicolon-separated list of comma-separated positive integer vectors, all
    of the same length. The length must equal ``dcp_size`` -- checked by the
    callers that know it (``reshard_vector_set``), not here.
    """
    vectors: List[Tuple[int, ...]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            vec = tuple(int(x) for x in part.split(","))
        except ValueError:
            raise KvReshardError(
                f"--kv-reshard-vectors: {part!r} is not a comma-separated "
                f"integer vector (full spec: {spec!r})"
            ) from None
        if not vec or any(v < 1 for v in vec):
            raise KvReshardError(
                f"--kv-reshard-vectors: every vector entry must be >= 1, "
                f"got {part!r}"
            )
        vectors.append(vec)
    if not vectors:
        raise KvReshardError(f"--kv-reshard-vectors: empty spec {spec!r}")
    lengths = {len(v) for v in vectors}
    if len(lengths) != 1:
        raise KvReshardError(
            f"--kv-reshard-vectors: vectors differ in length "
            f"({sorted(lengths)}); every vector must have one entry per DCP "
            f"rank (full spec: {spec!r})"
        )
    return vectors


def reshard_vector_set(
    spec: str, dcp_size: int, boot_vector: Sequence[int]
) -> List[Tuple[int, ...]]:
    """The declared ceiling set: boot vector first, then the reshard targets.

    This is what the pool sizing reserves rows for and what ``arm`` accepts
    targets from. The boot vector is always a member (resharding back is
    always legal), deduplicated.
    """
    boot = tuple(int(x) for x in boot_vector)
    if len(boot) != dcp_size:
        raise KvReshardError(
            f"boot KV vector {boot} has {len(boot)} entries but dcp_size is "
            f"{dcp_size}"
        )
    vectors = parse_reshard_vectors(spec)
    for vec in vectors:
        if len(vec) != dcp_size:
            raise KvReshardError(
                f"--kv-reshard-vectors: vector {vec} has {len(vec)} entries "
                f"but dcp_size is {dcp_size}"
            )
    out: List[Tuple[int, ...]] = [boot]
    for vec in vectors:
        if vec not in out:
            out.append(vec)
    return out


def reshard_ceiling_rows(
    global_tokens: int, vectors: Sequence[Sequence[int]], rank: int
) -> int:
    """Fitted-ceiling physical row count for one rank over the vector set.

    The #297 capacity decision (DESIGN_297 section 4): the pool is sized once
    at boot to the row MAXIMUM this rank needs under ANY declared vector, so
    a later reshard never has to grow the pool -- addresses and sizes stay
    stable and the decode CUDA graphs never notice. #93 VMM append-growth is
    the named follow-up for memory-tight configurations.
    """
    rows = 0
    for vec in vectors:
        rows = max(
            rows,
            dcp_compact_pool_rows(global_tokens, sum(vec), int(vec[rank])),
        )
    return rows


def _owner_prefix(vector: Sequence[int]) -> List[int]:
    prefix = [0]
    for v in vector:
        prefix.append(prefix[-1] + int(v))
    return prefix


def owner_of(slots: torch.Tensor, vector: Sequence[int]) -> torch.Tensor:
    """DCP owner rank of each global slot id under ``vector`` (pure).

    The owner rule of ``owner.py``, vectorized over an int64 slot tensor:
    rank ``r`` owns ``L`` iff ``L % S`` falls in ``[prefix[r], prefix[r+1])``.
    """
    prefix = _owner_prefix(vector)
    bounds = torch.tensor(prefix[1:], dtype=torch.int64, device=slots.device)
    res = slots.to(torch.int64) % prefix[-1]
    return torch.searchsorted(bounds, res, right=True)


def rows_of(slots: torch.Tensor, vector: Sequence[int], rank: int) -> torch.Tensor:
    """Compact physical pool row of each slot on ``rank`` under ``vector``.

    Same expression as ``dcp_weighted_read_slots``/``write_slots`` --
    ``(L // S) * ratio + (L % S - lo)`` -- restated on int64 for the mover.
    Only meaningful for slots ``rank`` owns under ``vector``.
    """
    prefix = _owner_prefix(vector)
    s = prefix[-1]
    lo = prefix[rank]
    ratio = prefix[rank + 1] - lo
    loc = slots.to(torch.int64)
    return (loc // s) * ratio + (loc % s - lo)


@dataclass
class ReshardTransition:
    """One rank's share of an (old vector -> new vector) KV move.

    All row tensors are int64 on CPU. ``outgoing_rows[p]`` / ``incoming_rows``
    enumerate the pair's slots in ascending global-slot order on BOTH ends --
    the shared payload-layout convention (module docstring).
    """

    rank: int
    old_vector: Tuple[int, ...]
    new_vector: Tuple[int, ...]
    #: rows this rank keeps, in the OLD layout (gather source).
    retained_old_rows: torch.Tensor
    #: the same slots' rows in the NEW layout (scatter destination).
    retained_new_rows: torch.Tensor
    #: peer rank -> OLD-layout local rows to send there.
    outgoing_rows: Dict[int, torch.Tensor]
    #: peer rank -> NEW-layout local rows their payload fills.
    incoming_rows: Dict[int, torch.Tensor]
    #: total live slots the transition covers (group-wide).
    total_slots: int

    @property
    def retained_moving(self) -> int:
        return int(self.retained_old_rows.numel())

    @property
    def outgoing_slots(self) -> int:
        return sum(int(r.numel()) for r in self.outgoing_rows.values())

    @property
    def incoming_slots(self) -> int:
        return sum(int(r.numel()) for r in self.incoming_rows.values())

    def max_new_row(self) -> int:
        """Largest NEW-layout row this rank will write (bounds check input)."""
        mx = -1
        if self.retained_new_rows.numel():
            mx = max(mx, int(self.retained_new_rows.max().item()))
        for r in self.incoming_rows.values():
            if r.numel():
                mx = max(mx, int(r.max().item()))
        return mx


def build_transition(
    slots: torch.Tensor,
    old_vector: Sequence[int],
    new_vector: Sequence[int],
    rank: int,
) -> ReshardTransition:
    """Derive this rank's transition from the replicated live-slot set.

    ``slots``: int64, sorted ascending, unique (the caller's obligation --
    checked here because the payload-layout convention depends on it).
    Retained slots whose row does not change are dropped from the move.
    """
    old_vec = tuple(int(x) for x in old_vector)
    new_vec = tuple(int(x) for x in new_vector)
    if len(old_vec) != len(new_vec):
        raise KvReshardError(f"vector length mismatch: old {old_vec} vs new {new_vec}")
    n_ranks = len(old_vec)
    if not (0 <= rank < n_ranks):
        raise KvReshardError(f"rank {rank} out of range for {n_ranks} ranks")
    slots = slots.detach().to("cpu", torch.int64)
    if slots.numel():
        if not bool((slots[1:] > slots[:-1]).all()):
            raise KvReshardError(
                "live slot ids must be sorted ascending and unique -- the "
                "pairwise payload layout is derived from this order on both "
                "ends"
            )
        if int(slots[0].item()) < 0:
            raise KvReshardError("negative KV slot id in live set")

    old_owner = owner_of(slots, old_vec)
    new_owner = owner_of(slots, new_vec)
    mine_old = old_owner == rank
    mine_new = new_owner == rank

    retained_mask = mine_old & mine_new
    retained_slots = slots[retained_mask]
    retained_old = rows_of(retained_slots, old_vec, rank)
    retained_new = rows_of(retained_slots, new_vec, rank)
    moving = retained_old != retained_new
    retained_old = retained_old[moving]
    retained_new = retained_new[moving]

    outgoing: Dict[int, torch.Tensor] = {}
    incoming: Dict[int, torch.Tensor] = {}
    for peer in range(n_ranks):
        if peer == rank:
            continue
        out_slots = slots[mine_old & (new_owner == peer)]
        if out_slots.numel():
            outgoing[peer] = rows_of(out_slots, old_vec, rank)
        in_slots = slots[(old_owner == peer) & mine_new]
        if in_slots.numel():
            incoming[peer] = rows_of(in_slots, new_vec, rank)

    return ReshardTransition(
        rank=rank,
        old_vector=old_vec,
        new_vector=new_vec,
        retained_old_rows=retained_old,
        retained_new_rows=retained_new,
        outgoing_rows=outgoing,
        incoming_rows=incoming,
        total_slots=int(slots.numel()),
    )
