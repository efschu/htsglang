# SPDX-License-Identifier: Apache-2.0
"""D-MM-ITEM-CACHE-1007: a repeated image costs group D no pixel work.

Group D under ``--pdflip-vision transient`` tokenizes images and never encodes
them (``vision_d_guard``): an image a D leg serves lies inside the cached
prefix, so of all the processor computes D uses the image's token count, its
grid (mrope positions) and its content hash (the ``pad_value`` that keys the
prefix). MEASURED on NF (Voranalyse 07.10.): an agent turn of ~150k tokens
with 5 images spends 1.7-1.9 s of its 2.1-2.4 s lead on D in the tokenizer
process -- every image decoded and preprocessed again on every turn, the
prompt tokenized again by the HF processor -- and the scheduler hashes the
pixels once more.

This cache keeps, per inline image (its bytes plus the processor's
settings), the three values. A request whose images are ALL known is built
from them (``QwenVLImageProcessor.build_output_for_known_images``): the
compact ids the tokenizer manager already holds, each placeholder expanded,
items with ``hash`` set and no ``feature``. Anything else takes the
processor exactly as before; an eligible request then also leaves the
tokenizer with ``item.hash = hash_feature(item.feature)`` -- the very function
the scheduler would apply to the very CPU tensor (``set_pad_value``), so the
value is the same and the scheduler skips it -- and its values are kept.

WHERE IT IS ARMED: ``FLLIPER_ENABLE_D_MM_ITEM_CACHE`` on, group D with
multimodal tokenization and no tower (``vision_d_guard.d_guard_armed``), the
Qwen-VL processor, and a hash that is the content hash computed the same way
in every process (no ``FLLIPER_MM_SKIP_COMPUTE_HASH``, no GPU hash buffer, no
CUDA-IPC transport, features kept on the host). Off, or anywhere else, the
tokenizer manager never builds it and every request runs as before.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import time
from typing import Any, List, Optional

import msgspec
import torch

from flliper.srt.environ import envs
from flliper.srt.managers.io_struct import GenerateReqInput
from flliper.srt.managers.mm_utils import hash_feature
from flliper.srt.managers.schedule_batch import MultimodalProcessorOutput
from flliper.srt.utils import ImageData
from flliper.srt.pdflip.vision_d_guard import d_guard_armed

logger = logging.getLogger(__name__)

TAG = "PDFLIP D-MM-ITEM-CACHE"


class KnownImage(msgspec.Struct, frozen=True, kw_only=True):
    """What D needs of one image it tokenized before, without its pixels."""

    hash: int
    grid_thw: torch.Tensor
    num_tokens: int


def image_source_bytes(item: Any) -> Optional[bytes]:
    """The bytes that ARE this image, or ``None`` when the request carries a
    reference whose content may change (http(s), a file path) or a form whose
    bytes are not at hand (a decoded image, a preprocessed dict)."""
    if isinstance(item, bytes):
        return b"bytes:" + item
    if isinstance(item, str):
        return b"data:" + item.encode("utf-8") if item.startswith("data:") else None
    if isinstance(item, ImageData) and item.preprocess_kwargs is None:
        inner = image_source_bytes(item.url)
        if inner is None:
            return None
        knobs = json.dumps([item.detail, item.max_dynamic_patch], default=str)
        return inner + b"|" + knobs.encode("utf-8")
    return None


def image_key(*, fingerprint: bytes, source: bytes) -> str:
    return hashlib.sha256(fingerprint + b"\0" + source).hexdigest()


def compact_ids_fit(
    *, ids: List[int], n_images: int, image_token_id: int, start_id: int,
    end_id: int, video_token_id: Optional[int]
) -> bool:
    """One bracketed placeholder per image and no video placeholder -- the
    shape the processor's fast path aligns 1:1 with ``image_data``."""
    if video_token_id is not None and video_token_id in ids:
        return False
    positions = [i for i, tok in enumerate(ids) if tok == image_token_id]
    if len(positions) != n_images:
        return False
    return all(
        0 < i < len(ids) - 1 and ids[i - 1] == start_id and ids[i + 1] == end_id
        for i in positions
    )


def armed_reason(*, server_args: Any, model_config: Any, mm_processor: Any) -> str:
    """Empty when the cache may run in this process, else why not."""
    from flliper.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    if not isinstance(mm_processor, QwenVLImageProcessor):
        return "no Qwen-VL processor"
    if not d_guard_armed(model_config):
        return "not a tower-less multimodal group D"
    if server_args.language_only or server_args.skip_tokenizer_init:
        return "language_only / skip_tokenizer_init"
    if server_args.keep_mm_feature_on_device:
        return "keep_mm_feature_on_device (the hash would be the GPU hash)"
    if envs.FLLIPER_MM_SKIP_COMPUTE_HASH.get():
        return "FLLIPER_MM_SKIP_COMPUTE_HASH (no content hash)"
    if envs.FLLIPER_MM_BUFFER_SIZE_MB.get() > 0:
        return "FLLIPER_MM_BUFFER_SIZE_MB > 0 (the scheduler hashes on the GPU)"
    if envs.FLLIPER_USE_CUDA_IPC_TRANSPORT.get():
        return "FLLIPER_USE_CUDA_IPC_TRANSPORT"
    return ""


def build_d_mm_item_cache(
    *, server_args: Any, model_config: Any, mm_processor: Any
) -> Optional["DMmItemCache"]:
    """The cache for this tokenizer process, or ``None`` (switch off, or not
    group D's tokenizer -- then nothing below ever runs)."""
    if not envs.FLLIPER_ENABLE_D_MM_ITEM_CACHE.get():
        return None
    why = armed_reason(
        server_args=server_args, model_config=model_config, mm_processor=mm_processor
    )
    if why:
        logger.info("%s not armed: %s -- the processor runs for every image", TAG, why)
        return None
    entries = envs.FLLIPER_D_MM_ITEM_CACHE_ENTRIES.get()
    logger.info("%s armed: entries=%d", TAG, entries)
    return DMmItemCache(processor=mm_processor, max_entries=entries)


class DMmItemCache:
    """Per tokenizer process; touched only on its event loop."""

    def __init__(self, *, processor: Any, max_entries: int):
        self._processor = processor
        self._max_entries = max(1, int(max_entries))
        self._fingerprint = processor.image_cache_fingerprint().encode("utf-8")
        self._entries: "collections.OrderedDict[str, KnownImage]" = (
            collections.OrderedDict()
        )
        self.hits = 0
        self.misses = 0

    async def process(
        self, *, obj: Any, prompt: Any, compact_ids: Any, max_req_input_len: Any
    ) -> Optional[MultimodalProcessorOutput]:
        """``process_mm_data_async``'s output for this request; ``prompt`` is
        exactly what the tokenizer manager hands the processor."""
        t0 = time.perf_counter()
        keys = self._request_keys(obj=obj, prompt=prompt, compact_ids=compact_ids)
        if keys is not None:
            out = self._try_hit(keys=keys, compact_ids=compact_ids)
            if out is not None:
                self.hits += 1
                self._log(obj=obj, verdict="hit", n=len(keys), t0=t0)
                return out
        out = await self._processor.process_mm_data_async(
            image_data=obj.image_data,
            audio_data=obj.audio_data,
            input_text=prompt,
            request_obj=obj,
            max_req_input_len=max_req_input_len,
        )
        if keys is not None:
            self.misses += 1
            self._remember(keys=keys, output=out)
            self._log(obj=obj, verdict="miss", n=len(keys), t0=t0)
        return out

    def _request_keys(
        self, *, obj: Any, prompt: Any, compact_ids: Any
    ) -> Optional[List[str]]:
        """One key per image, in prompt order, or ``None`` when the request is
        not a plain image request this cache can stand behind."""
        if not isinstance(obj, GenerateReqInput) or obj.mm_hashes:
            return None
        if obj.video_data or obj.audio_data or not obj.image_data:
            return None
        if not isinstance(prompt, str) or not prompt or not isinstance(compact_ids, list):
            return None
        sources = [image_source_bytes(item) for item in obj.image_data]
        if any(source is None for source in sources):
            return None
        p = self._processor
        if not compact_ids_fit(
            ids=compact_ids,
            n_images=len(sources),
            image_token_id=p.mm_tokens.image_token_id,
            start_id=p.vision_start_token_id,
            end_id=p.vision_end_token_id,
            video_token_id=p.mm_tokens.video_token_id,
        ):
            return None
        return [image_key(fingerprint=self._fingerprint, source=s) for s in sources]

    def _try_hit(
        self, *, keys: List[str], compact_ids: List[int]
    ) -> Optional[MultimodalProcessorOutput]:
        known = [self._entries.get(key) for key in keys]
        if any(entry is None for entry in known):
            return None
        for key in keys:
            self._entries.move_to_end(key)
        return self._processor.build_output_for_known_images(
            compact_ids=compact_ids,
            hashes=[entry.hash for entry in known],
            grids=[entry.grid_thw for entry in known],
            token_counts=[entry.num_tokens for entry in known],
        )

    def _remember(
        self, *, keys: List[str], output: Optional[MultimodalProcessorOutput]
    ) -> None:
        """Hash the items here and keep their values -- only when the output
        has the per-image shape the hit path rebuilds."""
        items = [] if output is None else output.mm_items
        if len(items) != len(keys) or not all(_rememberable(item) for item in items):
            return
        for key, item in zip(keys, items):
            item.hash = hash_feature(item.feature)
            start, end = item.offsets[0]
            self._entries[key] = KnownImage(
                hash=item.hash,
                grid_thw=item.model_specific_data["image_grid_thw"].clone(),
                num_tokens=end - start + 1,
            )
            self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def _log(self, *, obj: Any, verdict: str, n: int, t0: float) -> None:
        logger.info(
            "%s rid=%s verdict=%s images=%d ms=%.1f hits=%d misses=%d entries=%d",
            TAG, obj.rid, verdict, n, (time.perf_counter() - t0) * 1000.0,
            self.hits, self.misses, len(self._entries),
        )


def _rememberable(item: Any) -> bool:
    grid = item.model_specific_data.get("image_grid_thw")
    return (
        item.is_image()
        and item.hash is None
        and item.precomputed_embeddings is None
        and isinstance(item.feature, torch.Tensor)
        and not item.feature.is_cuda
        and item.offsets is not None
        and len(item.offsets) == 1
        and isinstance(grid, torch.Tensor)
        and grid.shape[0] == 1
    )
