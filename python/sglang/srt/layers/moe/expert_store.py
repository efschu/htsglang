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
    "written_rows_cached",
    "forget_written_rows",
    "store_has_row",
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
    if src.device.type == "cpu":
        store.index_copy_(0, dst, src[torch.as_tensor(locals_, dtype=torch.long)].to(store.dtype))
        return rows
    # fn8m4 (20.09.): a host intermediate (src.to("cpu")) per tensor left the
    # load's anon footprint at 60 GiB while the store filled (shmem 26 GiB)
    # -> 90 GiB against the 88 mark. Copy each row straight from the device
    # into the registered store mapping (D2H DMA), no host buffer at all.
    assert src.dtype == store.dtype, (src.dtype, store.dtype)
    for e in locals_:
        store[rows[e]].copy_(src[e], non_blocking=False)
    return rows


#: #75: der LESER des Sentinels, den es bisher nicht gab.
#:
#: `mark_rows_written` publiziert seit dem ersten Tag, welche Zeilen ein Rang
#: geschrieben hat -- und `rows_written` hatte NULL AUFRUFER (devindex, 21.09.).
#: Die Folge, gemessen an fnFL2w1: Gruppe D liest den ganzen Checkpoint ein
#: zweites Mal (237 s, ~14 GiB Page-Cache, der die Container-Decke bricht),
#: obwohl P dieselben Bytes im geteilten tmpfs liegen hat.
#:
#: Ein Cache je (dir, layer_key, attr): die Sentinels sind Dateien, und der
#: Ladepfad fragt je Experte einmal -- 512 stat()+json.load() je Tensor waere
#: die Ersparnis wieder aufgefressen. Der Cache lebt fuer die Dauer des Ladens
#: und wird von `forget_written_rows` verworfen, wenn jemand schreibt.
_ROWS_CACHE: Dict[Tuple[str, str, str, int], Dict[int, int]] = {}


def written_rows_cached(directory: str, layer_key: str, attr: str,
                        world: int) -> Dict[int, int]:
    """``rows_written`` mit Cache je (dir, layer_key, attr, world)."""
    key = (str(directory), str(layer_key), str(attr), int(world))
    hit = _ROWS_CACHE.get(key)
    if hit is None:
        hit = rows_written(directory, layer_key, attr, world)
        _ROWS_CACHE[key] = hit
    return hit


def forget_written_rows() -> None:
    """Den Cache verwerfen -- nach jedem Schreiben, damit ein Leser nie eine
    Sentinel-Lage von vor dem Schreiben sieht."""
    _ROWS_CACHE.clear()


def store_has_row(layer_key: str, attr: str, global_row: int,
                  world: int) -> bool:
    """Liegt diese GLOBALE Zeile schon im geteilten Store?

    KONSERVATIV PER KONSTRUKTION: jede Unsicherheit -- Store aus, Verzeichnis
    unlesbar, Sentinel fehlt, Zeile nicht darin -- antwortet False, also
    "lies sie vom Checkpoint". Ein Irrtum in diese Richtung kostet Ladezeit;
    der Irrtum in die andere Richtung laedt ein Modell mit einer Luecke und
    rechnet still falsch.
    """
    if not store_enabled():
        return False
    try:
        rows = written_rows_cached(store_dir(), layer_key, attr, int(world))
    except Exception:  # noqa: BLE001 -- ein unlesbarer Sentinel ist ein Lader,
        return False   # kein Absturz: der Checkpoint ist immer noch da.
    return int(global_row) in rows


def _sentinel(directory: str, layer_key: str, attr: str, rank: int) -> str:
    return store_path(directory, layer_key, attr) + f".r{int(rank)}.written.json"


def mark_rows_written(directory: str, layer_key: str, attr: str, rank: int, rows: Iterable[int]) -> str:
    """Publish which store rows ``rank`` finished writing (atomic rename)."""
    path = _sentinel(directory, layer_key, attr, rank)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"rank": int(rank), "rows": sorted(int(r) for r in rows)}, fh)
    os.replace(tmp, path)
    # #75: der Cache des Lesers darf eine gerade publizierte Zeile nicht
    # verpassen -- er wird hier verworfen, nicht per Zeitstempel geraten.
    forget_written_rows()
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
