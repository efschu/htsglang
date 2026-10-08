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
    from flliper.srt.mem_cache.base_prefix_cache import InsertParams
    from flliper.srt.mem_cache.radix_cache import RadixKey
    from flliper.srt.pdflip import l15_retain

    toks = [int(t) for t in token_ids]
    rws = [int(r) for r in rows]
    if not toks or len(toks) != len(rws):
        raise L15AdoptRefused(
            "adopt: %d token(s) vs %d row(s)" % (len(toks), len(rws)))
    if int(anchor_row) < 0:
        raise L15AdoptRefused("adopt: no anchor row (anchor_row=%d)" % anchor_row)
    # L15-ADOPT-TAIL: insert() page-aligns the key and drops the value's
    # tail; reserved tail rows would then be owned by nobody (a leak per
    # adopt) and the anchor would sit deeper than its key -- refuse instead
    page = int(getattr(tree_cache, "page_size", 1) or 1)
    if page > 1 and len(toks) % page:
        raise L15AdoptRefused("adopt: %d token(s) not a multiple of page %d"
                              % (len(toks), page))
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
    # L15-ADOPT-TAIL: a bigram (EAGLE/MTP) tree files n tokens as n-1 keys
    # and drops the value's last row -- reserved, in no node: give it back
    kept = len(toks) - 1 if getattr(tree_cache, "is_eagle", False) else len(toks)
    if kept < len(rws):
        allocator.free(value[kept:])
    if getattr(res, "mamba_exist", False):
        # L15-ADOPT-TAIL: the tree kept another state at this node (or refused
        # the anchor off-grid) -- the reserved anchor row is nobody's, give
        # it back (the tree freed its own duplicate KV rows itself)
        mamba_allocator.free(mamba_value)
    return AdoptResult(adopted=len(toks),
                       last_node=getattr(res, "last_device_node", None))
