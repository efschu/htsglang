"""L15-10 S1: P adopts a delivered hot prefix as device nodes.

The hot D->P handover (L15-PLAN-0930 sec 2.0; design L15-10-REAL-HANDOVER-
DESIGN-1002.md) lands a prefix's KV on P rows and its END GDN anchor on one P
mamba row. Before any admission, P makes that delivery its own:

1. reserve the rows in the KV allocator and the anchor row in the mamba
   allocator (l15_retain.reserve_slots / reserve_mamba_slots: a row that is
   not free is a refusal -- adopting over a live row would double-book it);
2. insert the prefix into the radix tree as DEVICE nodes, the anchor on the
   end node (InsertParams key/value/mamba_value), so the follow-up's match is
   a device hit -- no store read, no re-prefill;
3. nothing else: no copy happens here (the transport is S3/S5), no
   collective (every P stage adopts its own rows of the same prefix).

Any refusal raises L15AdoptRefused BEFORE the tree is touched; the caller
falls back to today's store read (the hot_handover state machine's named
``fallen_back``). Pure of CUDA: tensors live on the allocator's device.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Sequence

import torch


class L15AdoptRefused(RuntimeError):
    """The delivery cannot be adopted (named reason); nothing was changed."""


@dataclass(frozen=True)
class AdoptResult:
    adopted: int
    last_node: object


def adopt(tree_cache, allocator, mamba_allocator, *, token_ids: Sequence[int],
          rows: Sequence[int], anchor_row: int, extra_key=None) -> AdoptResult:
    """Adopt ``token_ids`` living at P ``rows`` (same order) with the end
    anchor at mamba row ``anchor_row``. See the module docstring."""
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.weg2 import l15_retain

    toks = [int(t) for t in token_ids]
    rws = [int(r) for r in rows]
    if not toks or len(toks) != len(rws):
        raise L15AdoptRefused(
            "adopt: %d token(s) vs %d row(s)" % (len(toks), len(rws)))
    if int(anchor_row) < 0:
        raise L15AdoptRefused("adopt: no anchor row (anchor_row=%d)" % anchor_row)
    free_kv = set(int(x) for x in allocator.free_pages.tolist())
    busy = [r for r in rws if r not in free_kv]
    if busy:
        raise L15AdoptRefused(
            "adopt: row(s) %s not free in the KV allocator" % (busy[:6],))
    free_mb = set(int(x) for x in mamba_allocator.free_slots.tolist())
    if int(anchor_row) not in free_mb:
        raise L15AdoptRefused(
            "adopt: anchor row %d not free in the mamba allocator"
            % int(anchor_row))
    try:
        l15_retain.reserve_slots(allocator, rws)
        l15_retain.reserve_mamba_slots(mamba_allocator, [int(anchor_row)])
    except ValueError as exc:
        raise L15AdoptRefused("adopt: %s" % (exc,)) from exc
    dev = allocator.free_pages.device
    value = torch.tensor(rws, dtype=torch.int64, device=dev)
    mamba_value = torch.tensor([int(anchor_row)], dtype=torch.int64,
                               device=mamba_allocator.free_slots.device)
    res = tree_cache.insert(InsertParams(
        key=RadixKey(token_ids=array("q", toks), extra_key=extra_key,
                     is_bigram=getattr(tree_cache, "is_eagle", False)),
        value=value, mamba_value=mamba_value))
    return AdoptResult(adopted=len(toks),
                       last_node=getattr(res, "last_device_node", None))
