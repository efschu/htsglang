# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#760: refuse a mis-shaped KV transfer at the seam, not inside the kernel.

DEFENSE IN DEPTH. F4-r5 owns the root fix (torn binding on the storage write
path); this layer is worth having whatever form that takes, because it converts
the whole class from a CUDA context death -- which takes the scheduler with it
and leaves a stack 30 layers deep in a GDN kernel -- into a named Python
refusal that fails one request.

WHY A SECOND GUARD. F4-r5's ``_host_binding_is_stale`` asks "is this binding
stale?". This asks "do these two pools even have the same shape?". Different
invariants, and the callsite census says the second one has nowhere to live yet:

    file                            transfer_kv_* callsites   guarded
    hisparse_memory_pool.py                     2                0
    memory_pool_host.py                        32                2
    pool_host/mha.py   <-- LIVE                28                0
    pool_host/mla.py                           12                0
    TOTAL                                      74                2

The live boot instantiates ``MHATokenToKVPoolHost``
(``model_runner_kv_cache_mixin.py:2766``), in ``pool_host/mha.py``, which has
ZERO guard callsites -- which is why the existing guard emitted no refusals
across both crashes.

WHAT IT COSTS. O(layers) integer comparisons on METADATA only: vector lengths,
one stride equality, and two index-bound reductions. No tensor element is read,
nothing is copied, no device synchronisation. On a 16-layer model that is ~35
integer comparisons plus two min/max over the index vectors per transfer,
against a transfer that moves megabytes. The matched-shape path returns None
and changes nothing.

FAIL OPEN ON ABSENT METADATA, deliberately. A caller that cannot supply a
capacity is not refused on that ground. A defense-in-depth layer that fails
closed on paths it does not understand becomes the outage it was added to
prevent -- and absence is not a mismatch.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

__all__ = ["KvTransferShapeMismatch", "validate_kv_transfer"]


logger = logging.getLogger(__name__)
_ARMED_SEAMS: set = set()


class KvTransferShapeMismatch(RuntimeError):
    """A KV transfer was refused because the two pools disagree in shape.

    Carries both shapes and, when the caller has them, the #719 generation
    stamps -- so the reader starts from an identity rather than from a kernel
    address.
    """


def _len_or_none(v) -> Optional[int]:
    try:
        return len(v)
    except TypeError:
        return None


def _bounds(indices) -> Optional[tuple]:
    """``(min, max, count)`` without reading device memory.

    Torch tensors expose ``tolist``/``min``/``max``; a list works directly.
    Anything else is unknown, and unknown is not a mismatch.
    """
    n = _len_or_none(indices)
    if n is None:
        return None
    if n == 0:
        return (0, -1, 0)
    try:
        lo = int(min(indices))
        hi = int(max(indices))
    except (TypeError, ValueError):
        return None
    return (lo, hi, n)


def validate_kv_transfer(
    *,
    src_ptr_vectors: Sequence[Any],
    dst_ptr_vectors: Sequence[Any],
    src_indices: Any,
    dst_indices: Any,
    src_capacity: Optional[int] = None,
    dst_capacity: Optional[int] = None,
    src_item_size: Optional[int] = None,
    dst_item_size: Optional[int] = None,
    src_generation: Optional[int] = None,
    dst_generation: Optional[int] = None,
    where: str = "",
) -> None:
    """Raise :class:`KvTransferShapeMismatch`, or return None.

    ``*_ptr_vectors`` is the per-pool list of per-layer pointer vectors (k and
    v are separate vectors), which is exactly what the kernel indexes by layer.
    """

    # SAY, ONCE PER SEAM, THAT THIS GUARD IS ON THE PATH.
    #
    # A guard that is silent when shapes match cannot be distinguished from a
    # guard that was never reached -- and that is not hypothetical here: the
    # previous #760 guard sat in DeepSeekV4PagedHostPool, covered 2 of 74
    # callsites and ZERO on the live class, and its silence was read for hours
    # as "the binding is sound". The callsite census is what settled it, after
    # two boots had already been spent.
    #
    # So the first call at each seam logs the shapes it actually saw. That turns
    # "the guard is armed on the live path" from an assumption into a line an
    # operator can grep for BEFORE trusting a clean run -- which is exactly how
    # this boot is being judged.
    global _ARMED_SEAMS
    if where not in _ARMED_SEAMS:
        _ARMED_SEAMS.add(where)
        logger.info(
            "KV-TRANSFER-GUARD ARMED at %s: src layer-vectors=%s dst layer-vectors=%s "
            "src_capacity=%s dst_capacity=%s%s",
            where or "a transfer seam",
            [_len_or_none(v) for v in src_ptr_vectors],
            [_len_or_none(v) for v in dst_ptr_vectors],
            src_capacity,
            dst_capacity,
            (
                f" | #719 generation src={src_generation} dst={dst_generation}"
                if (src_generation is not None or dst_generation is not None)
                else ""
            ),
        )

    def _stamps() -> str:
        if src_generation is None and dst_generation is None:
            return ""
        return f" | #719 generation src={src_generation} dst={dst_generation}"

    def _fail(what: str) -> None:
        raise KvTransferShapeMismatch(
            f"#760 KV transfer REFUSED at {where or 'a transfer seam'}: {what}. "
            f"src layer-vector lengths="
            f"{[_len_or_none(v) for v in src_ptr_vectors]}, "
            f"dst layer-vector lengths="
            f"{[_len_or_none(v) for v in dst_ptr_vectors]}, "
            f"src item_size={src_item_size}, dst item_size={dst_item_size}"
            f"{_stamps()}. Handing these to the kernel indexes one pool by the "
            f"other's layer count and corrupts memory rather than failing, so "
            f"this request is refused and the scheduler kept alive."
        )

    # 1. the pointer vectors must agree WITHIN each pool (k vs v) ...
    for name, vecs in (("src", src_ptr_vectors), ("dst", dst_ptr_vectors)):
        lens = {_len_or_none(v) for v in vecs}
        lens.discard(None)
        if len(lens) > 1:
            _fail(f"{name} pool's own layer vectors disagree: {sorted(lens)}")

    # 2. ... and ACROSS the two pools. This is the specimen: a PP-shaped host
    #    pool (7 layers on this stage) against a TP-shaped device pool (16).
    src_len = next((n for n in (_len_or_none(v) for v in src_ptr_vectors) if n), None)
    dst_len = next((n for n in (_len_or_none(v) for v in dst_ptr_vectors) if n), None)
    if src_len is not None and dst_len is not None and src_len != dst_len:
        _fail(f"layer counts differ: src has {src_len}, dst has {dst_len}")

    # 3. per-layer extent. Same layer COUNT with different bytes per token is
    #    the quieter half: it does not crash, it writes wrong output.
    if (
        src_item_size is not None
        and dst_item_size is not None
        and int(src_item_size) != int(dst_item_size)
    ):
        _fail(
            f"per-layer item_size differs: src={src_item_size}, "
            f"dst={dst_item_size}"
        )

    # 4. indices: one row per index on both sides, and in bounds for BOTH pools.
    sb, db = _bounds(src_indices), _bounds(dst_indices)
    if sb is not None and db is not None and sb[2] != db[2]:
        _fail(f"index counts differ: src has {sb[2]}, dst has {db[2]}")
    for name, b, cap in (("src", sb, src_capacity), ("dst", db, dst_capacity)):
        if b is None or b[2] == 0:
            continue
        if b[0] < 0:
            _fail(f"{name} indices out of bounds: min={b[0]} < 0")
        if cap is not None and b[1] >= int(cap):
            _fail(
                f"{name} indices out of bounds: max={b[1]} >= capacity {int(cap)}"
            )
    return None
