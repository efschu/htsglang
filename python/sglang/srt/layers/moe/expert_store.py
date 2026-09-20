"""One shared host store per (layer, expert tensor), rows = GLOBAL expert ids
(Task #47 build step 1, 19.09.).

Today every rank pins a PRIVATE spill pool per layer and tensor whose row j
is ``plan.spill_ids[j]`` (a LOCAL id of the uneven expert-dim shard). Two
rank groups that must both reach the whole expert set -- the TP3 decode group
(disjoint expert ranges per rank) and the PP3 prefill group (every expert of
its layers) -- would need two host copies, 98 GB against the 88 GB mark. With
the store there is ONE tmpfs file per (layer, tensor) of shape
``(num_experts, *row)``: every rank writes the rows it loaded (residents and
spill alike) at ``lo + local - 1``, overlapping writers write identical
bytes, and a reader from any group indexes the same file by global id. The
device pool's row copy (``expert_pool_device.copy_rows``) needs exactly one
contiguous host source per tensor, which is why this is one file per tensor
and not the cold tier's one segment per owner (#394 slice 2).

Off unless ``SGLANG_MOE_EXPERT_STORE_DIR`` names a directory. Pure file and
tensor arithmetic here; ``shared_pinned`` does the mmap + cudaHostRegister.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Sequence, Tuple

import torch

from sglang.srt.layers.moe.shared_pinned import shared_pinned_empty

STORE_DIR_ENV = "SGLANG_MOE_EXPERT_STORE_DIR"

__all__ = [
    "STORE_DIR_ENV",
    "store_dir",
    "store_enabled",
    "store_path",
    "open_store",
    "global_rows",
    "write_rows",
    "mark_rows_written",
    "rows_written",
]


def store_dir() -> str:
    return os.environ.get(STORE_DIR_ENV, "").strip()


def store_enabled() -> bool:
    return bool(store_dir())


def store_path(directory: str, layer_key: str, attr: str) -> str:
    safe = str(layer_key).replace("/", "_").replace(".", "_")
    return os.path.join(directory, f"{safe}-{attr}.bin")


def open_store(
    directory: str,
    layer_key: str,
    attr: str,
    num_experts: int,
    row_shape: Sequence[int],
    dtype: torch.dtype,
    register=None,
) -> Tuple[torch.Tensor, bool]:
    """The ``(num_experts, *row_shape)`` tensor backed by the store file.
    Returns (tensor, created): the first opener creates the file (zero pages),
    every later one maps the same bytes."""
    os.makedirs(directory, exist_ok=True)
    path = store_path(directory, layer_key, attr)
    return shared_pinned_empty(
        path, (int(num_experts),) + tuple(int(d) for d in row_shape), dtype, register
    )


def global_rows(local_ids: Iterable[int], lo: int, pad: bool = True) -> Dict[int, int]:
    """Local id -> store row. ``pad=True`` is the generic expert-dim shard:
    local 0 is the zero pad expert (no row), local i >= 1 is global
    ``lo + i - 1``. ``pad=False`` is an unsharded layer (a PP stage holding
    every expert): local i is global ``lo + i``."""
    out: Dict[int, int] = {}
    for e in local_ids:
        e = int(e)
        if pad:
            if e >= 1:
                out[e] = int(lo) + e - 1
        else:
            out[e] = int(lo) + e
    return out


def write_rows(
    store: torch.Tensor, src: torch.Tensor, local_ids: Sequence[int], lo: int, pad: bool = True
) -> Dict[int, int]:
    """Copy ``src[local]`` into its store row (see :func:`global_rows`) for
    every local id (``src`` is expert-major over the rank's LOCAL ids, on any
    device). Returns the local -> row map that was written."""
    rows = global_rows(local_ids, lo, pad)
    if not rows:
        return rows
    locals_ = list(rows.keys())
    dst = torch.as_tensor([rows[e] for e in locals_], dtype=torch.long)
    # NO device-side gather: fn8m (20.09.) showed the index_select copies of
    # every layer's expert tensors staying RESERVED in the caching allocator
    # (+1.9 GiB on the x4 3080), which the KV sizer then read as used and
    # the first 8k chunk OOMed on. Move the rows through the host instead.
    if locals_ == list(range(len(locals_))) and len(locals_) == int(src.shape[0]):
        picked = src.to("cpu")  # every local row, in order: one D2H, no device alloc
    elif src.device.type == "cpu":
        picked = src[torch.as_tensor(locals_, dtype=torch.long)]
    else:
        picked = torch.empty((len(locals_),) + tuple(src.shape[1:]), dtype=src.dtype)
        for j, e in enumerate(locals_):
            picked[j].copy_(src[e])
    store.index_copy_(0, dst, picked.to(store.dtype))
    return rows


def _sentinel(directory: str, layer_key: str, attr: str, rank: int) -> str:
    return store_path(directory, layer_key, attr) + f".r{int(rank)}.written.json"


def mark_rows_written(directory: str, layer_key: str, attr: str, rank: int, rows: Iterable[int]) -> str:
    """Publish which store rows ``rank`` finished writing (atomic rename)."""
    path = _sentinel(directory, layer_key, attr, rank)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"rank": int(rank), "rows": sorted(int(r) for r in rows)}, fh)
    os.replace(tmp, path)
    return path


def rows_written(directory: str, layer_key: str, attr: str, world: int) -> Dict[int, int]:
    """Row -> writer rank over every published sentinel of the group; a row
    two ranks both claim keeps the lower rank (identical bytes by contract)."""
    out: Dict[int, int] = {}
    for rank in range(int(world) - 1, -1, -1):
        path = _sentinel(directory, layer_key, attr, rank)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            data = json.load(fh)
        for r in data.get("rows", []):
            out[int(r)] = int(rank)
    return out
