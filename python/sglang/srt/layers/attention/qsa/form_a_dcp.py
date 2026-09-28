# SPDX-License-Identifier: Apache-2.0
"""#239 S3b -- the QSA sparse attention under Form A x the token cut.

Form A puts every q head, every kv projection and the QSA indexer on ONE rank
(the attention host); the token cut (#239 S3a) spreads the full-attention KV
over all ranks by the weighted owner rule. The existing uneven-DCP machinery
(``_attend_rows``: q-head gather, owned-rows partial, LSE merge) already moves
data correctly for a head split of ``[H, 0, 0]`` -- a zero-head rank takes part
with a ``[T, 0, D]`` slice, exactly the 27B weightless worker's shape. What it
cannot do alone is the two things only the host has:

  A  the new k/v of this forward (the host projects them; every owner writes
     its rows) -- one uneven all-gather of ``cat((k, v), 0)`` with kv counts
     ``[kv, 0, 0]``, the weightless worker's fused write gather;
  T  the top-k of this forward (the indexer runs on the host only, #239 S0)
     -- one uneven all-gather of ``[T, 1, K]`` with counts ``[1, 0, 0]``.

Per full-attention layer every rank of the group therefore issues, in this
order and only this order:

  A                       always (the write),
  T, Q, M                 when the layer attends through the rows path
                          (decode, verify, draft-extend, extend with prefix);

an extend without prefix attends locally on the host (its own k/v) and stops
after A. The host issues them from the QSA backend (``_set_kv_buffer`` ->
``_rows_and_counts`` -> ``_attend_rows``); a worker issues the same sequence
from :func:`worker_attention_step`. Every branch is decided by rank-uniform
facts (forward mode, extend prefix lengths), never by this rank's share.

All collectives are the uneven head gathers and merges of ``layers/dcp/comm.py``
-- no new transport primitive, so barlink and NCCL carry it as they carry the
27B weightless path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FormADcpGeometry:
    """Who holds which heads in the Form A DCP group (rank order = DCP rank)."""

    world: int
    host_rank: int
    rank: int
    q_heads: int
    kv_heads: int

    def _host_only(self, n: int) -> list:
        return [int(n) if r == self.host_rank else 0 for r in range(self.world)]

    @property
    def q_counts(self) -> list:
        return self._host_only(self.q_heads)

    @property
    def kv_counts(self) -> list:
        return self._host_only(self.kv_heads)

    @property
    def topk_counts(self) -> list:
        return self._host_only(1)

    @property
    def is_host(self) -> bool:
        return self.rank == self.host_rank


def form_a_dcp_geometry(
    q_heads: int, kv_heads: int, dcp_size: int, dcp_rank: int
) -> Optional[FormADcpGeometry]:
    """The geometry when THIS process runs Form A with DCP over the group,
    else None (every classic boot, and Form A without a token cut)."""
    from sglang.srt.rank_role import installed_role_plan

    plan = installed_role_plan()
    if plan is None or int(dcp_size) <= 1:
        return None
    if len(plan.roles) != int(dcp_size):
        raise ValueError(
            f"#239 S3b: Form A roles {list(plan.roles)} but DCP spans "
            f"{dcp_size} ranks -- the token cut must span the whole Form A group."
        )
    return FormADcpGeometry(
        world=int(dcp_size),
        host_rank=int(plan.host_rank),
        rank=int(dcp_rank),
        q_heads=int(q_heads),
        kv_heads=int(kv_heads),
    )


def share_kv(
    k: torch.Tensor, v: torch.Tensor, group, geo: FormADcpGeometry
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collective A: the host's new k/v on every rank ([T, kv, D] each).

    A worker hands ``[T, 0, D]`` tensors; one gather of ``cat((k, v), 0)``
    (the weightless worker's fused write gather), split at T."""
    from sglang.srt.layers.dcp.comm import cp_all_gather_heads_uneven

    n = int(k.shape[0])
    kv_full = cp_all_gather_heads_uneven(
        torch.cat((k, v), dim=0).contiguous(), group, geo.kv_counts
    )
    return kv_full[:n], kv_full[n:]


def share_topk(
    topk: Optional[torch.Tensor],
    rows: int,
    width: int,
    group,
    geo: FormADcpGeometry,
    device,
) -> torch.Tensor:
    """Collective T: the host's top-k ([rows, K] int32) on every rank.

    The host passes its indices, a worker passes None and receives them."""
    from sglang.srt.layers.dcp.comm import cp_all_gather_heads_uneven

    if geo.is_host:
        if topk is None:
            raise ValueError("#239 S3b: the Form A host has no top-k to share")
        x = topk.to(torch.int32).reshape(int(rows), 1, int(width)).contiguous()
    else:
        x = torch.empty((int(rows), 0, int(width)), dtype=torch.int32, device=device)
    return cp_all_gather_heads_uneven(x, group, geo.topk_counts).reshape(
        int(rows), int(width)
    )


def gather_q(q: torch.Tensor, group, geo: FormADcpGeometry) -> torch.Tensor:
    """Collective Q: all q heads on every rank ([T, H, D])."""
    from sglang.srt.layers.dcp.comm import cp_all_gather_heads_uneven

    return cp_all_gather_heads_uneven(q.contiguous(), group, geo.q_counts)


def merge(out: torch.Tensor, lse: torch.Tensor, group, geo: FormADcpGeometry) -> torch.Tensor:
    """Collective M: the LSE merge to the host (a worker gets [T, 0, D])."""
    from sglang.srt.layers.dcp.comm import (
        cp_lse_ag_out_a2a_mha_uneven,
        cp_lse_ag_out_ar_mha_uneven,
        lse_merge_mode,
    )

    fn = cp_lse_ag_out_a2a_mha_uneven if lse_merge_mode() == "a2a" else cp_lse_ag_out_ar_mha_uneven
    return fn(out, lse, group, geo.q_counts)


def worker_attention_step(
    *,
    rows: int,
    head_dim: int,
    topk_width: int,
    dtype: torch.dtype,
    device,
    group,
    geo: FormADcpGeometry,
    write: Callable[[torch.Tensor, torch.Tensor], None],
    attends: bool,
    resolve_rows: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
    attend: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> None:
    """One full-attention layer on a Form A worker: A [, T, Q, M].

    ``write(k_full, v_full)`` stores this rank's owned rows; ``resolve_rows``
    maps the shared top-k to this rank's pool rows (and optional counts);
    ``attend(q_full, rows, counts)`` returns the owned partial ``(out, lse)``.
    The merged output belongs to the host; the worker's slice is empty."""
    if geo.is_host:
        raise ValueError("#239 S3b: worker_attention_step on the Form A host")
    empty = torch.empty((int(rows), 0, int(head_dim)), dtype=dtype, device=device)
    k_full, v_full = share_kv(empty, empty, group, geo)
    write(k_full, v_full)
    if not attends:
        return
    if resolve_rows is None or attend is None:
        raise ValueError("#239 S3b: an attending worker step needs resolve_rows and attend")
    topk = share_topk(None, rows, topk_width, group, geo, device)
    owned_rows, counts = resolve_rows(topk)
    q_full = gather_q(empty, group, geo)
    out, lse = attend(q_full, owned_rows, counts)
    merge(out, lse, group, geo)
