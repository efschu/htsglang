"""L15-W2B: the wake sample-check SCRATCH pool
(design /spinning/gpu-arb/docs/L15-WIRE2-NOTES.md sec 2).

The wake's sample check (l15_sample.load_into_scratch / read_rows, folded
into the F11 refusal by l15_check.sample_check) must load a few sampled
held rows off the host into DEVICE rows.  It may NOT touch the live
token_to_kv_pool (its rows are the thing under test), and it has no
permanent device buffer: the scratch is allocated at wake, used once,
then freed.

ScratchKV is the tiny pool-shaped object both l15_sample functions
already speak to: it exposes ``k_buffer`` / ``v_buffer`` -- one tensor per
layer, shape ``(rows, *live_row_shape)`` -- exactly the attributes
ArenaMHAHostPool._load_pages_all_layers (arena_pool.py) reads
(``device_pool.k_buffer[0].device``, then per-layer
``.index_copy_(0, dst, ...)``), and l15_sample.read_rows unwraps the
optional ``.full_kv_pool`` hybrid wrapper the same way.  One-shot: the
caller frees it after the check, it is never resident.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


class ScratchKV:
    """A per-layer K/V buffer pair shaped like a live KV pool.

    Not a real pool: no allocator, no slots, no lifecycle.  The loader
    writes into rows 0..n-1 and the checker reads them back; free()
    releases the memory."""

    def __init__(self, k_buffer: List[torch.Tensor],
                 v_buffer: List[torch.Tensor]):
        self.k_buffer = k_buffer
        self.v_buffer = v_buffer
        self.rows = int(k_buffer[0].shape[0]) if k_buffer else 0

    def nbytes(self) -> int:
        total = 0
        for buf in (self.k_buffer or []) + (self.v_buffer or []):
            if buf is not None:
                total += buf.numel() * buf.element_size()
        return total

    def free(self) -> None:
        self.k_buffer = None
        self.v_buffer = None


def make_scratch_pool(live_pool, rows: int) -> ScratchKV:
    """One-shot device scratch for the wake sample check.

    live_pool: the live token_to_kv_pool (MHA or hybrid wrapper; the
        latter is unwrapped via .full_kv_pool, exactly like
        l15_sample.read_rows does).  Per-layer row shape, dtype and
        device are taken from the live pool itself.
    rows: number of sample rows (l15_restore.sample_rows's k).
    """
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError(f"scratch rows must be a positive int, got {rows!r}")
    pool = getattr(live_pool, "full_kv_pool", live_pool)
    k_src = getattr(pool, "k_buffer", None)
    v_src = getattr(pool, "v_buffer", None)
    if (not isinstance(k_src, (list, tuple))
            or not isinstance(v_src, (list, tuple))
            or not k_src or len(k_src) != len(v_src)):
        raise ValueError(
            "live pool lacks usable k_buffer/v_buffer layer lists: "
            f"k={type(k_src).__name__}, v={type(v_src).__name__}")
    k_bufs, v_bufs = [], []
    for l, (kb, vb) in enumerate(zip(k_src, v_src)):
        if kb.dim() < 2 or vb.dim() < 2 or kb.shape[0] != vb.shape[0]:
            raise ValueError(
                f"scratch: layer {l} row shape mismatch "
                f"({tuple(kb.shape)} vs {tuple(vb.shape)})")
        k_bufs.append(
            torch.empty((rows,) + tuple(kb.shape[1:]),
                        dtype=kb.dtype, device=kb.device))
        v_bufs.append(
            torch.empty((rows,) + tuple(vb.shape[1:]),
                        dtype=vb.dtype, device=vb.device))
    s = ScratchKV(k_bufs, v_bufs)
    logger.info("L15 scratch: %d rows, %d layers, %d MiB one-shot",
                rows, len(k_bufs), s.nbytes() >> 20)
    return s
