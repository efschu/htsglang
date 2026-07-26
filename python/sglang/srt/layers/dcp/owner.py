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

"""Weighted uneven-DCP owner rule (backend-agnostic).

Relocated VERBATIM from ``layers/attention/flashinfer_backend.py`` — the
mechanism never had a flashinfer dependency, it only lived there. It is pure
torch plus the shared Triton kernel ``create_flashinfer_kv_indices_triton``,
so it compiles and runs on any backend Triton supports (CUDA incl. sm75, and
ROCm/HIP incl. gfx900). ``flashinfer_backend.py`` imports it back, so the
CUDA path is byte-identical to before the move.

The owner rule, stated once:
  rank owns cache slot ``L``  <=>  ``(L % cp_S)`` in ``[cp_lo, cp_hi)``
  compact physical slot       ==  ``(L // cp_S) * cp_ratio + (L % cp_S - cp_lo)``
Ownership derives from ``out_cache_loc`` itself rather than the sequence
position, which keeps the compact slot an injective, collision-free function
of ``L`` across concurrent requests — exactly like the even ``L // dcp_size``.
"""

from typing import Optional

import torch

from sglang.kernels.ops.kvcache.kv_indices import create_flashinfer_kv_indices_triton

__all__ = [
    "build_dcp_weighted_kv_indices",
    "dcp_weighted_owned_lengths",
    "dcp_weighted_owner_bounds",
    "dcp_weighted_read_slots",
    "dcp_weighted_write_slots",
]


def dcp_weighted_owner_bounds(dcp_size: int, dcp_rank: int):
    """``(cp_S, cp_lo, cp_hi, cp_ratio)`` for this rank's weighted owner range.

    Single derivation shared by every backend, so a rank cannot end up reading
    with one set of bounds and writing with another.
    """
    from sglang.srt.distributed.utils import cp_token_prefix

    prefix = cp_token_prefix(dcp_size)
    cp_S = prefix[-1]
    cp_lo = prefix[dcp_rank]
    cp_hi = prefix[dcp_rank + 1]
    return cp_S, cp_lo, cp_hi, cp_hi - cp_lo


def dcp_weighted_write_slots(
    cache_loc: torch.Tensor,
    cp_S: int,
    cp_lo: int,
    cp_hi: int,
    cp_ratio: int,
):
    """WRITE side of the weighted owner rule: ``(loc, mask)`` for ``cache_loc``.

    ``mask[i]`` is True iff this rank owns slot ``cache_loc[i]``; ``loc[i]`` is
    the compact physical slot to write it to (0 where not owned, matching the
    established convention that unowned entries are masked out anyway).

    This is the EXACT INVERSE of the read-side mapping in
    ``build_dcp_weighted_kv_indices``:
        read:  compact = (L // cp_S) * cp_ratio + (L % cp_S - cp_lo)
        write: loc     = (L // cp_S) * cp_ratio + (L % cp_S - cp_lo)
    They are deliberately the same expression, defined ONCE here. Read and
    write drifting apart is the failure mode that makes a token be fetched
    from a slot it was never stored in -- silently wrong output rather than a
    crash -- so the two sides must not be maintained as separate copies in the
    flashinfer and Triton backends.
    """
    off = cache_loc % cp_S
    block = cache_loc // cp_S
    mask = (off >= cp_lo) & (off < cp_hi)
    loc = torch.where(
        mask,
        block * cp_ratio + (off - cp_lo),
        torch.zeros_like(cache_loc),
    )
    return loc, mask


def dcp_weighted_read_slots(
    cache_loc: torch.Tensor,
    cp_S: int,
    cp_lo: int,
    cp_hi: int,
    cp_ratio: int,
):
    """READ side of the weighted owner rule: ``(compact, owned)``.

    The exact counterpart of ``dcp_weighted_write_slots`` -- SAME expression,
    stated once -- for a flat list of GLOBAL cache slots (typically the
    ``req_to_token`` rows of a paged read). ``owned[i]`` is True iff this rank
    stores slot ``cache_loc[i]``; ``compact[i]`` is the physical row it stores
    it in. ``compact`` is computed for EVERY entry, not only the owned ones,
    so the caller can mask without a second pass; unowned entries carry a
    meaningless (possibly negative) value and must not be dereferenced.

    Extracted out of ``build_dcp_weighted_kv_indices`` so that the index math
    -- the part that decides whether uneven DCP is CORRECT -- is a pure tensor
    function with no Triton kernel in it, and can therefore be pinned against
    an independent reference on CPU, with no device and no collectives.
    """
    loc64 = cache_loc.to(torch.int64)
    off = loc64 % cp_S
    owned = (off >= cp_lo) & (off < cp_hi)
    compact = ((loc64 // cp_S) * cp_ratio + (off - cp_lo)).to(torch.int32)
    return compact, owned


def dcp_weighted_owned_lengths(
    owned: torch.Tensor,
    lens: torch.Tensor,
) -> torch.Tensor:
    """Per-request count of owned slots, given the flat ``owned`` mask.

    ``owned`` is the concatenation of the per-request slot lists in request
    order and ``lens`` their (global) lengths, so this is a segmented sum.
    Returned as int64, one entry per request. Pure tensor math -- CPU-safe,
    tested against numpy.
    """
    lens64 = lens.to(torch.int64)
    bs = lens64.numel()
    out = torch.zeros(bs, dtype=torch.int64, device=lens64.device)
    if owned.numel() == 0:
        return out
    seg = torch.repeat_interleave(
        torch.arange(bs, device=lens64.device, dtype=torch.int64), lens64
    )
    out.scatter_add_(0, seg, owned.to(torch.int64))
    return out


def build_dcp_weighted_kv_indices(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    paged_kernel_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_start_idx: Optional[torch.Tensor],
    cp_S: int,
    cp_lo: int,
    cp_hi: int,
    cp_ratio: int,
    pad: int = 0,
    req_to_token_stride: Optional[int] = None,
):
    """Weighted-DCP paged kv_indices: this rank's OWNED cache slots, compacted.

    Builds the full per-request out_cache_loc list for [kv_start, kv_start+len),
    keeps the slots this rank owns under the WEIGHTED owner rule
    (loc % cp_S in [cp_lo, cp_hi)) and maps each to its compact physical slot
    (loc // cp_S) * cp_ratio + (loc % cp_S - cp_lo) -- the exact inverse of the
    _dcp_masked_write packing, so a token is read from the slot it was written
    to. Writes the per-request owned counts into ``kv_indptr`` and returns
    (kv_indptr[:bs+1], kv_indices).

    ``req_to_token_stride`` defaults to ``req_to_token.shape[1]``, which is the
    row stride of the (contiguous) req_to_token table. Callers that hold a
    stride from the pool pass it explicitly rather than re-deriving it.

    Only the ``create_flashinfer_kv_indices_triton`` launch is device-bound;
    the actual owner-rule arithmetic is ``dcp_weighted_read_slots`` /
    ``dcp_weighted_owned_lengths``, which are pure tensor functions pinned on
    CPU against an independent reference.
    """
    bs = len(req_pool_indices)
    device = req_pool_indices.device
    lens64 = paged_kernel_lens.to(torch.int64)
    full_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    full_indptr[1:] = torch.cumsum(lens64, dim=0).to(torch.int32)
    total = int(full_indptr[bs].item())
    full_kv = torch.empty(max(total, 1), dtype=torch.int32, device=device)
    if total > 0:
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            full_indptr,
            kv_start_idx,
            full_kv,
            (
                req_to_token.shape[1]
                if req_to_token_stride is None
                else req_to_token_stride
            ),
        )
    full_kv = full_kv[:total]
    compact, owned = dcp_weighted_read_slots(full_kv, cp_S, cp_lo, cp_hi, cp_ratio)
    kv_indices = compact[owned].contiguous()
    if pad > 0:
        kv_indices = torch.cat([kv_indices, kv_indices.new_zeros(pad)])
    owned_per_req = dcp_weighted_owned_lengths(owned, lens64)
    kv_indptr[1 : bs + 1] = torch.cumsum(owned_per_req, dim=0)
    return kv_indptr[: bs + 1], kv_indices
