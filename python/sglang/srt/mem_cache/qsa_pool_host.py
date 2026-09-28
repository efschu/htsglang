"""Page-aligned HiCache host tier for the compressed QSA index keys (23.09.).

Why this exists (fnFL2x54, Task #106): the QSA block selector of
Qwen3.8-Flash-Next scores every index query against ONE compressed key per
``compress_ratio`` tokens, and that key is a projection of the hidden states
(index-K, pooled and normalised) -- it cannot be recomputed from the K/V page
alone. The indexer writes it only inside a forward
(``qsa_indexer.py`` ``set_qsa_compressed_k_buffer``). A prefix restored from
the carrier therefore had its KV pages and its GDN anchor, but an EMPTY index
for those pages: the mid-prompt needle was answered with a filler number
(x54 MISS) while the same prompt prefilled on the decode group itself matched
(mid control). This pool mirrors the compressed buffer alongside the KV page,
addressed by the KV page's own indices (``SidecarPoolSpec`` with
``indices_from_pool=KV``): one KV page of ``page_size`` tokens is
``page_size // ratio`` consecutive compressed slots, one byte row per layer.

Adapted from kanadaj/sglang PR #9 (ktsaou, patch 0022-hicache-qsa-sidecar,
merged 2026-09-14), which carries the same state through RAM and files under
TP. Two departures: no draft pools (the NEXTN draft here shares the target's
index, ``SGLANG_QSA_MTP_INDEX_SHARE``), and a CANONICAL page window
(``build_qsa_index_window``) because group P is a PP=3 pipeline whose stages
hold 7/3/2 of the 12 full-attention layers -- without the window every stage
would write the whole ``{hash}.qsa_indexer`` key with only its own layers and
the last writer would win.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from sglang.srt.mem_cache.canonical_page_store import (
    CanonicalExtentWindow,
    CanonicalPageError,
)
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool

logger = logging.getLogger(__name__)


def qsa_index_bytes_per_token(device_pools, page_size: int) -> int:
    """Host bytes per KV token over every pool: ``slot_bytes // ratio`` per layer."""
    total = 0
    for pool in device_pools:
        ratio = int(pool.qsa_compress_ratio)
        if ratio <= 0 or page_size <= 1 or page_size % ratio:
            raise ValueError(
                "QSA HiCache requires complete compression groups per page "
                f"(page_size={page_size}, compress_ratio={ratio})"
            )
        buffers = pool.qsa_compressed_k_buffer_pool
        if not buffers:
            raise ValueError("QSA HiCache requires compressed index buffers")
        for buffer in buffers:
            if buffer.ndim != 3:
                raise ValueError("QSA HiCache requires [slot, head, dim] index buffers")
            slot_bytes = buffer[0].numel() * buffer.element_size()
            if slot_bytes % ratio:
                raise ValueError("QSA compressed index byte size must divide the ratio")
            total += slot_bytes // ratio
    return total


def qsa_index_layer_block_bytes(device_pool, page_size: int) -> int:
    """Bytes of ONE layer's compressed keys for ONE KV page (the canonical
    page's per-layer block): ``page_size // ratio`` slots x slot bytes."""
    ratio = int(device_pool.qsa_compress_ratio)
    buffer = device_pool.qsa_compressed_k_buffer_pool[0]
    return page_size // ratio * buffer[0].numel() * buffer.element_size()


class QSAPagedHostPool(DeepSeekV4PagedHostPool):
    """Mirror of the compressed QSA index in the KV page address space."""

    def __init__(
        self,
        device_pools,
        num_host_tokens: int,
        page_size: int,
        layout: str,
        *,
        allocator_type: str = "default",
        pin_memory: bool = True,
    ):
        device_pools = tuple(device_pools)
        if (
            not device_pools
            or page_size <= 1
            or num_host_tokens <= 0
            or num_host_tokens % page_size
        ):
            raise ValueError("QSA HiCache requires pools and a page-aligned host size")
        if layout not in ("layer_first", "page_first", "page_first_direct"):
            raise ValueError(f"Unsupported QSA HiCache layout: {layout}")
        bytes_per_token = qsa_index_bytes_per_token(device_pools, page_size)
        buffers = []
        item_bytes = None
        index_shape = None
        for pool in device_pools:
            ratio = int(pool.qsa_compress_ratio)
            for buffer in pool.qsa_compressed_k_buffer_pool:
                shape = (ratio, *buffer.shape[1:])
                if index_shape is not None and shape != index_shape:
                    raise ValueError("QSA index shapes must match across pools")
                index_shape = shape
                page_bytes = page_size // ratio * buffer[0].numel() * buffer.element_size()
                if item_bytes is not None and page_bytes != item_bytes:
                    raise ValueError("QSA index page shapes must match across pools")
                if (
                    not buffer.is_contiguous()
                    or buffer.numel() * buffer.element_size() % page_bytes
                ):
                    raise ValueError("QSA index buffers must contain contiguous complete pages")
                item_bytes = page_bytes
                # The reused transport copies byte rows, independent of index dtype.
                buffers.append(buffer.view(torch.uint8).reshape(-1, page_bytes))
        super().__init__(
            pool_name="qsa_indexer",
            device_buffers=buffers,
            item_bytes=item_bytes,
            num_host_pages=num_host_tokens // page_size,
            slot_page_size=page_size,
            layout=layout,
            allocator_type=allocator_type,
            pin_memory=pin_memory,
        )
        self.size_per_token = bytes_per_token

    def get_size_per_token(self):
        return self.layer_num * self.item_bytes // self.slot_page_size

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def _has_transfer_indices(self, host_indices, device_indices):
        present = super()._has_transfer_indices(host_indices, device_indices)
        if present and host_indices.numel() % self.slot_page_size:
            # Partial groups would need ring state; restored prefixes end on pages.
            raise ValueError("QSA HiCache transfers must contain complete KV pages")
        if present and not self._host_rows_in_range(host_indices):
            return False
        return present

    #: H106 (rc12z15): transfers a Form A worker skipped, and their pages
    _formA_skip_n = 0
    _formA_skip_pages = 0

    #: H106b: the transfer the skip lines are folded into (one merged load or
    #: backup -- ``CacheOperation.merge_ops`` joins the requests, so no rid is
    #: visible here; the ``#988 LOADBACK rid=`` line of the same pass names it)
    _h106_open = None
    _h106_suppressed = 0
    _h106_transfers = 0
    #: a periodic line after this many folded calls
    H106_PERIODIC = 256

    def load_to_device_per_layer(self, *args, **kwargs):
        self._h106_dir = "load"
        return super().load_to_device_per_layer(*args, **kwargs)

    def backup_from_device_all_layer(self, *args, **kwargs):
        self._h106_dir = "backup"
        return super().backup_from_device_all_layer(*args, **kwargs)

    def backup_from_device_indices(self, *args, **kwargs):
        self._h106_dir = "backup"
        return super().backup_from_device_indices(*args, **kwargs)

    def _h106_note_skip(self, pages: int, hi: int, lo: int) -> None:
        """H106b (rc12z17, 10:50:22Z): one line per skipped TRANSFER and
        direction, not per layer call -- the load path calls this pool once per
        layer with the same ids (dozens of lines in one second). A call with
        the same (direction, first id, pages) as the open transfer is folded
        into it; a new one prints its line with the previous transfer's sums
        (calls, pages_total); every H106_PERIODIC folded calls a counter line
        with suppressed_since_last_print."""
        cls = QSAPagedHostPool
        d = getattr(self, "_h106_dir", "?")
        key = (d, int(lo), int(pages))
        cur = cls._h106_open
        if cur is not None and cur["key"] == key:
            cur["calls"] += 1
            cur["pages_total"] += int(pages)
            cls._h106_suppressed += 1
            if cls._h106_suppressed % cls.H106_PERIODIC == 0:
                logger.warning(
                    "H106 FORM-A SIDECAR SKIP (periodic) pool=%s dir=%s transfers=%d "
                    "skipped_calls=%d pages_total=%d suppressed_since_last_print=%d",
                    self.pool_name, d, cls._h106_transfers, cls._formA_skip_n,
                    cls._formA_skip_pages, cls.H106_PERIODIC,
                )
            return
        prev = ""
        if cur is not None:
            prev = " prev(dir=%s calls=%d pages_total=%d)" % (
                cur["key"][0], cur["calls"], cur["pages_total"])
        cls._h106_open = {"key": key, "calls": 1, "pages_total": int(pages)}
        cls._h106_transfers += 1
        logger.warning(
            "H106 FORM-A SIDECAR SKIP pool=%s dir=%s pages=%d host_page_max=%d "
            "host_pages=%d min_id=%d transfer=%d%s (skipped_calls=%d "
            "pages_total=%d; further calls of this transfer are folded): the "
            "byteless KV anchor grew its ids (#249), this pool did not; a Form A "
            "worker never reads its QSA rows, so nothing is transferred.",
            self.pool_name, d, int(pages), int(hi), int(self.num_host_pages), int(lo),
            cls._h106_transfers, prev, cls._formA_skip_n, cls._formA_skip_pages,
        )

    def _host_rows_in_range(self, host_indices) -> bool:
        """H106 (rc12z15 f49f7bddd2, D 10:22:39, TP1+TP2 at once): the KV
        anchor of a Form A worker is BYTELESS and, under #249 (9996780367),
        grows its id space past the synced size instead of refusing -- rc12z15
        ``#249 BYTELESS-GROW pool=MHATokenToKVPoolHost rows 353600 -> 373504``
        at the wake that read six held prompts (63360 + 17536 + 105664 +
        17664 + 63488 + 105792 = 373504 ids). This pool is addressed by the
        KV's ids (``indices_from_pool=KV``) but was sized ONCE from the KV's
        size at assembly (5525 pages = 353600 ids) with real 4 KiB rows and
        does not grow: the tail of weg2-0-4 named page >= 5525, the host slice
        came back EMPTY and ``transfer_kv_direct`` died on ``output with shape
        [1, 4096] doesn't match the broadcast shape [0, 4096]`` (the kernel
        branches would have written past the pinned buffer instead).

        A Form A worker runs no dense chain (``form_a_worker_forward``: the
        host alone holds attention, the QSA indexer and KV; the worker's KV
        pool is 0 B and its storage tier the null backend), so its QSA rows
        are never read: an out-of-range transfer there is skipped by name.
        On any other rank the same shape would be a wrong index for real
        bytes -- a named stop, never a silent skip. Per-rank local copy, no
        collective on this path, so skipping changes no group sequence."""
        try:
            if host_indices.numel() == 0:
                return True
            hi = int(host_indices.max()) // int(self.slot_page_size)
            lo = int(host_indices.min())
        except Exception:  # noqa: BLE001 - an index we cannot read is left to the kernel
            return True
        if lo >= 0 and hi < int(self.num_host_pages):
            return True
        from sglang.srt.rank_role import this_rank_is_form_a_worker

        pages = int(host_indices.numel()) // int(self.slot_page_size)
        if this_rank_is_form_a_worker():
            QSAPagedHostPool._formA_skip_n += 1
            QSAPagedHostPool._formA_skip_pages += pages
            self._h106_note_skip(pages, hi, lo)
            return False
        raise RuntimeError(
            f"H106 SIDECAR HOST INDEX OUT OF RANGE pool={self.pool_name} "
            f"host_page_max={hi} host_pages={int(self.num_host_pages)} min_id={lo} "
            f"pages={pages}: this rank holds real QSA bytes and a KV id names a "
            "row this pool does not have -- copying it would be a wrong index for "
            "the attention; stopping by name instead."
        )


def build_qsa_index_window(
    attn_layer_ids: Sequence[int], device_pool, host_pool
) -> CanonicalExtentWindow:
    """This rank's window in the canonical ``{hash}.qsa_indexer`` page.

    The page is layer-major: one block of ``host_pool.item_bytes`` per
    full-attention layer of the MODEL, in ``attn_layer_ids`` order. A PP
    stage's full layers are a contiguous run of that list, so its window is
    ONE extent; a stage whose layers are not contiguous, or not all in the
    model's list, is refused -- a wrong offset would deposit the right
    number of bytes under the wrong layer and nobody would notice until the
    selector picked the wrong blocks.
    """
    ids = [int(i) for i in attn_layer_ids]
    local = sorted(int(g) for g in device_pool.full_attention_layer_id_mapping.keys())
    if not local:
        raise CanonicalPageError("this rank holds no full-attention layer; no QSA index window")
    try:
        positions = [ids.index(g) for g in local]
    except ValueError as e:
        raise CanonicalPageError(
            f"this rank's full-attention layers {local} are not all in the "
            f"model's attention layer list {ids}."
        ) from e
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise CanonicalPageError(
            f"this rank's full-attention layers map to positions {positions}, "
            "which is not one contiguous run of the canonical QSA index page."
        )
    block = int(host_pool.item_bytes)
    if int(host_pool.layer_num) != len(local):
        raise CanonicalPageError(
            f"the QSA host pool mirrors {host_pool.layer_num} layer(s) but this "
            f"rank's device pool holds {len(local)} full-attention layer(s)."
        )
    total = len(ids) * block
    return CanonicalExtentWindow(
        total_bytes=total,
        extents=((positions[0] * block, len(local) * block),),
        label="qsa index page",
    )
