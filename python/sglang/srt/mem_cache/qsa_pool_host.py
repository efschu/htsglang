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

from typing import Sequence

import torch

from sglang.srt.mem_cache.canonical_page_store import (
    CanonicalExtentWindow,
    CanonicalPageError,
)
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool


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
        return present


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
