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

"""Pure index math for decode context parallel (DCP): per-rank lengths and
the owner-rule local-index filter."""

from typing import Optional, Sequence, Union

import torch

from sglang.srt.runtime_context import get_parallel

# A host-side length mirror: either a python sequence of ints or a CPU tensor.
HostLens = Union[Sequence[int], torch.Tensor]


def get_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-rank visible KV length under the owner rule pos % dcp_size == dcp_rank.

    Superset implementation (PR #25090): supports both start=None and a per-request
    `start` offset. update_local_kv_lens_for_dcp is the start=None special case.
    """
    if dcp_size == 1:
        return lens
    if start is None:
        return lens // dcp_size + (dcp_rank < lens % dcp_size)

    first = start + torch.remainder(dcp_rank - start, dcp_size)
    remaining = start + lens - first
    return torch.clamp((remaining + dcp_size - 1) // dcp_size, min=0)


# ---------------------------------------------------------------------------
# Host-side length math (#616c / #616h / #618 / #623).
#
# The DCP index builders size their output from a SUM of the length vector.
# Taken off the device -- `int(full_indptr[bs].item())` in
# build_dcp_weighted_kv_indices, `int(dcp_lens.sum().item())` in the even
# sibling branches -- that sum is an UNBOUNDED blocking device-to-host read
# sitting inside the collective window. The 2026-08-07 02:00 wedge died there,
# with all three ranks stopped on the same line while their device queues each
# held a BAR1 spin kernel: a host parked in a CUDA sync can neither poll nor
# time out nor enqueue, so no rank could release the others. Of the 239
# recorded wedge events, the 195 that park in barlink's BOUNDED poll recover;
# the unbounded ones are the fatal-capable minority. Converting a site from an
# unbounded sync to none is what these helpers are for. They do NOT claim to
# cure the graph-replay family root -- they remove one class of fatal pin.
#
# Every length vector these sums are taken over already exists on the host as a
# scheduler-side mirror. These helpers turn such a mirror into the exact same
# integer, with no device access at all.
# ---------------------------------------------------------------------------


def dcp_host_lens(
    lens_cpu: Optional[HostLens],
    expected_sum: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """The host length vector as a CPU int64 tensor, or None when unusable.

    Returns None -- meaning "caller keeps its existing device read" -- when:

    * no mirror was supplied (``None``);
    * the mirror is a CUDA tensor, since reading it is the very sync being
      removed;
    * ``expected_sum`` is given and the mirror does not sum to it.

    That last case is a free host-side STALENESS check, not a formality: a
    ``forward_batch.seq_lens_cpu`` is a non-None but STALE slice on gpu_only
    batches (see triton_backend.py, ``have_cpu_mirror``), and sizing an index
    buffer from a stale vector would be a silent mis-size rather than a crash.
    Callers that hold an independently computed host sum of the same vector
    (``paged_kernel_lens_sum``, ``seq_lens_sum``) pass it here and get the
    mirror only if the two agree.
    """
    if lens_cpu is None:
        return None
    if isinstance(lens_cpu, torch.Tensor):
        if lens_cpu.is_cuda:
            return None
        lens = lens_cpu.to(torch.int64)
    else:
        lens = torch.tensor(list(lens_cpu), dtype=torch.int64)
    if expected_sum is not None and int(lens.sum()) != int(expected_sum):
        return None
    return lens


def dcp_fresh_host_lens(
    lens_cpu: Optional[HostLens],
    expected_sum: Optional[int],
) -> Optional[HostLens]:
    """The mirror, or None when its freshness signal is absent (#629).

    :func:`dcp_host_lens` refuses a mirror that DISAGREES with a known sum, but
    ACCEPTS one unchecked when ``expected_sum`` is None -- correct for a vector
    whose sum nobody independently knows, wrong for ``seq_lens_cpu``. That one
    is a non-None but STALE slice exactly when ``seq_lens_sum`` is None
    (gpu_only batches; see ``have_cpu_mirror`` in triton_backend.py and the
    same note on the eager decode site). Handing the stale slice on with no sum
    to check it against turns the #616h fix into a silent MIS-SIZE of the index
    buffer -- worse than the blocking read it replaces, because a mis-sized
    buffer is wrong output rather than a stall.

    Callers holding such a (vector, its-sum-or-None) pair route the vector
    through here, so "no sum" degrades to "no mirror" -- the old device read --
    instead of to "unchecked mirror".
    """
    return None if expected_sum is None else lens_cpu


def dcp_host_total_tokens(
    lens_cpu: Optional[HostLens],
    expected_sum: Optional[int] = None,
) -> Optional[int]:
    """``sum(lens)`` computed on the host, or None when no mirror is usable.

    This is the ``total_tokens`` channel of
    :func:`~sglang.srt.layers.dcp.owner.build_dcp_weighted_kv_indices`: the
    weighted index build derives the same number as
    ``int(full_indptr[bs].item())`` when it is not supplied. The mirror is the
    same vector, so this is the identical integer -- not an estimate.
    """
    lens = dcp_host_lens(lens_cpu, expected_sum)
    return None if lens is None else int(lens.sum())


def dcp_host_even_total(
    lens_cpu: Optional[HostLens],
    dcp_size: int,
    dcp_rank: int,
    start: Optional[HostLens] = None,
    expected_sum: Optional[int] = None,
) -> Optional[int]:
    """``sum(get_dcp_lens(lens, ...))`` on the host, or None (#618).

    The even (non-weighted) owner-rule branches size their kv_indices with
    ``int(dcp_lens.sum().item())`` -- the same unbounded-sync class as the
    weighted branch's read, reachable whenever ``uneven_dcp_weighted`` is off.
    ``get_dcp_lens`` is pure integer math, so running it on the host mirror
    yields exactly the vector the device holds and therefore exactly the same
    sum.

    ``start`` (the per-request ``kv_start_idx``) is part of the formula; when it
    is present but only exists on the device, this returns None rather than
    guessing, and the caller keeps its device read.
    """
    lens = dcp_host_lens(lens_cpu, expected_sum)
    if lens is None:
        return None
    if start is None:
        start_host = None
    else:
        start_host = dcp_host_lens(start)
        if start_host is None:
            return None
    return int(get_dcp_lens(lens, dcp_size, dcp_rank, start_host).sum())


def filter_dcp_local_kv_indices(kv_indices: torch.Tensor):
    parallel = get_parallel()
    if parallel.dcp_enabled:
        kv_indices = (
            kv_indices[kv_indices % parallel.dcp_size == parallel.dcp_rank]
            // parallel.dcp_size
        )
    return kv_indices


def update_local_kv_lens_for_dcp(kv_len_arr):
    """In-place per-rank KV length: the start=0 case of get_dcp_lens.

    floor((len - rank - 1) / N) + 1  ==  len // N + (rank < len % N)  for len >= 0
    (bit-identical; see test/registered/cp/test_dcp_layout_unit.py). Kept as an
    in-place mutation because callers (plan_dcp_decode_metadata, the FlashInfer-MLA
    cuda-graph replay path) rely on it.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return
    kv_len_arr.copy_(get_dcp_lens(kv_len_arr, parallel.dcp_size, parallel.dcp_rank))
