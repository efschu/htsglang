"""DP-NACHLAUF 02.10.: one host mamba state, loaded ONCE per start_loading.

A host load-back of a mamba node builds two transfers from the SAME host slot
(``mamba_component.build_hicache_transfers`` LOAD_BACK): the node's own device
anchor (``nodes_to_load``) and the request's copy-on-write slot
(``req.mamba_pool_idx``). Both device rows are needed -- the request mutates
its own, the node's stays intact -- but the second H2D re-read the same
bytes: N5p (b6a6a5c08d) printed two ``PDFLIP-ARENA-STATE-LOAD slots=1
own_bytes=53932032`` per load on PP0 (2 x 54 MB on the load stream before
P's first prefill forward, 14 copies on PP2, 16 on PP1).

``split_duplicates`` keeps the first transfer of each (name, host slots) and
returns the later ones as ``(dup, src)`` pairs; the controller loads only the
kept ones and, per layer, right after that layer's load and before the
layer's producer event, copies the rows device-to-device on the load stream
(``copy_layer``) -- so the forward's per-layer wait covers the copy.

Not deduplicated (kept as two loads): host indices on the card (a compare
would be a host wait), different row counts, a transfer without device rows,
the PLE side-state form (its side states ride the arena load into the loaded
rows only). Switch ``FLLIPER_PDFLIP_MAMBA_LOAD_DEDUP`` (unset = on; 0/false/no/
off = both loads, as before).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_MAMBA_LOAD_DEDUP"
_N = [0]


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _ple_on() -> bool:
    try:
        from flliper.srt.mem_cache.pool_host import arena_mamba_pool as _amp

        return bool(_amp.ple_state.enabled())
    except Exception:  # noqa: BLE001 - no PLE module: no side states
        return False


def split_duplicates(transfers) -> Tuple[Optional[list], List[tuple]]:
    """``(kept, [(dup, src), ...])``; ``kept`` is the list itself when there
    is nothing to drop."""
    if not transfers or len(transfers) < 2 or not enabled() or _ple_on():
        return transfers, []
    kept, dups = [], []
    for t in transfers:
        src = None
        h, d = getattr(t, "host_indices", None), getattr(t, "device_indices", None)
        if (torch.is_tensor(h) and torch.is_tensor(d) and not h.is_cuda
                and int(h.numel()) > 0 and int(d.numel()) == int(h.numel())):
            for k in kept:
                kh, kd = getattr(k, "host_indices", None), getattr(k, "device_indices", None)
                if (k.name == t.name and torch.is_tensor(kh) and torch.is_tensor(kd)
                        and not kh.is_cuda and kh.numel() == h.numel() and kd.numel() == d.numel()
                        and torch.equal(kh.to(torch.int64), h.to(torch.int64))):
                    src = k
                    break
        if src is None:
            kept.append(t)
        else:
            dups.append((t, src))
    if not dups:
        return transfers, []
    return kept, dups


def split_for(host_group, transfers):
    """The controller's entry: no dedup on a host pool without entries."""
    if getattr(host_group, "entry_map", None) is None:
        return transfers, []
    return split_duplicates(transfers)


def copy_dups(host_group, dups, layer_id: int) -> int:
    """Per GLOBAL layer: copy every duplicate's rows from its source on the
    card (right after that layer's load). Returns the layers copied."""
    n = 0
    for dup, src in dups:
        ent = host_group.entry_map.get(dup.name)
        ll = ent.local_layer(layer_id) if ent is not None else None
        if ll is not None:
            copy_layer(ent.device_pool, ll, src.device_indices, dup.device_indices)
            n += 1
    return n


def _rows(idx, dev):
    idx = idx.to(torch.int64) if idx.dtype != torch.int64 else idx
    return idx if idx.device == dev else idx.to(dev, non_blocking=True)


def copy_layer(device_pool, local_layer: int, src_rows, dst_rows) -> None:
    """Layer ``local_layer``'s temporal and conv rows ``src -> dst`` on the
    current (load) stream."""
    mc = device_pool.mamba_cache
    t = mc.temporal[local_layer]
    s, d = _rows(src_rows, t.device), _rows(dst_rows, t.device)
    t.index_copy_(0, d, t.index_select(0, s))
    for conv in mc.conv:
        c = conv[local_layer]
        c.index_copy_(0, d, c.index_select(0, s))


def note(dups, n_layers_copied: int) -> None:
    _N[0] += 1
    n = _N[0]
    if n <= 8 or n % 64 == 0:
        logger.info(
            "PDFLIP-MAMBA-LOAD-DEDUP n=%d pools=%s rows=%d layers=%d: the same host state "
            "loaded once, the second device row copied on the card (DP-NACHLAUF; %s=0 = "
            "both H2D loads)", n, sorted({str(t.name) for t, _ in dups}),
            sum(int(t.host_indices.numel()) for t, _ in dups), n_layers_copied, ENV,
        )
