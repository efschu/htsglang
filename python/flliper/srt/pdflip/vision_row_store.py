# SPDX-License-Identifier: Apache-2.0
"""VISION-ROWS (NF rc30, 21.10.): an image P's tower encoded once can be computed by group D.

THE FINDING (boot ..._1d822bd150_1021_063252, session a4c8b097ac, front + P + D log):
  07:23:52 pdflip-8-511   covered=127232 prompt=127831 uncached=599 image_in_tail=1 (image of 506 tokens NEW)
                          -> flip to P (SOLL R9: D has no tower -- correct)
  07:24:22 pdflip-10-516  the next turn, common=127827 with its predecessor, covered=127808 (1997 full 64-token
                          pages), uncached=179, image_in_tail=1, D-MM-ITEM-CACHE hit -> a SECOND flip to P for
                          179 tokens (07:24:27, R4 T_halb run-out, ~4.7 s wait + the flip)
  pdflip-12-518           covered=128512, image_in_tail=0 -> D, no flip from here on.
The image of 8-511 had been encoded on P, but it ends behind the last FULL page (127808 < ~127827): the KV and
the recurrent state are kept per 64-token page/anchor (radix, host tier, store), so the last image positions
fall into the partial page D recomputes -- and D, without a tower, cannot compute an image position
(``vision_d_guard``: W123). Every image at the end of a message (screenshot, tool result) cost two flips.

THE ROOT. What D lacks is not the KV but the INPUT of those positions: the tower's rows. D's model takes rows
exactly where P's does -- upstream's per-image path ``mm_utils._get_chunked_embedding_by_item`` looks a row
tensor up by the item's content hash before it would call the ViT, and only the positions of the current extend
are scattered. P already keeps those rows per image (S66c ``vision_rank_runner.emb_cache``, upstream
``MultiModalStaticCache``) -- in P's PP0 process, invisible to D's.

THE FIX. This module is the cross-process half of that cache:
  * P (PP0, after every attach -- ``vision_rank_runner.emb_cache_store``) PUBLISHES each image's rows as one file
    ``<hash>-<grid>.rows`` in a host tmpfs directory named by the served model (``/dev/shm/pdflip-vrows-<id>``):
    a header (hash, grid, rows, width, dtype) plus the raw bytes, written atomically (tmp + rename), bounded in
    bytes (LRU by mtime, readers refresh it). The identity of an entry is (content hash, image_grid_thw) -- the
    S66c-r B1 rule: equal pixel bytes under another grid are other rows.
  * D's TOKENIZER answers the front's probe with the image spans whose rows are published (``mm_rows``); the
    FRONT (R11: it decides alone) counts an image as ``image_in_tail`` only when its span ends past ``covered``
    AND D cannot get its rows (``d_match_probe.exact_price``). An already encoded image is no p_only (R9 holds
    for an image D really cannot compute); its tail is priced as text (R2: 179 tokens are kurz, D at once).
  * D's SCHEDULER, at the vision verdict (``Scheduler._pdflip_vision_d_verdict``), loads the rows of exactly the
    images that reach past the group's covered prefix: TP rank 0 reads the files, ONE broadcast hands the same
    tensors to every rank (rank-uniform by construction, no rank reads the directory on its own -- an eviction
    between two ranks' reads can never split the group), and each rank attaches them to its items
    (``ITEM_ATTR``). An image with rows no longer bounds the verdict; one without still refuses by name (W123 +
    the reroute through P, unchanged). All or nothing per request.
  * ``mm_utils._get_chunked_embedding_by_item`` takes an attached row tensor where the per-image cache misses,
    before it would call the tower.

Not chosen: a partial last page in the store (KV and the recurrent state are page/anchor-granular; S66c named
this a store format change) -- and it would not help an image whose rows sit further back after an eviction.
Upstream's global embedding cache (``--enable-mm-global-cache``, Mooncake store) is the same idea across hosts
but needs the Mooncake transfer engine and its master service, which this image does not carry; the upstream
EPD path (encoder server -> language server) ships rows per request from a dedicated encoder instance, while
here the encoder (P's PP0) sleeps whenever D serves. The in-process ``MultiModalStaticCache`` is reused as the
P-side keeper (S66c) and its lookup path is the D-side consumer.

``FLLIPER_PDFLIP_VISION_ROWS_MB`` (default 2048; 0 = off), ``FLLIPER_PDFLIP_VISION_ROWS_DIR`` (directory
override; default under ``/dev/shm``). Only where the item hash is the content hash computed alike in every
process (``d_mm_item_cache.hash_reason``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

TAG = "PDFLIP VISION-ROWS"
ENV_MB = "FLLIPER_PDFLIP_VISION_ROWS_MB"
ENV_DIR = "FLLIPER_PDFLIP_VISION_ROWS_DIR"
DEFAULT_MB = 2048
ROOT = "/dev/shm"
PREFIX = "pdflip-vrows-"
SUFFIX = ".rows"
MAGIC = b"PDVROWS1\n"
#: the attribute D's scheduler attaches a row tensor under (read by mm_utils._get_chunked_embedding_by_item)
ITEM_ATTR = "pdflip_tail_rows"

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------- configuration
def cap_bytes(env: Optional[Dict[str, str]] = None) -> int:
    e = os.environ if env is None else env
    try:
        mb = int(str(e.get(ENV_MB, str(DEFAULT_MB)) or str(DEFAULT_MB)).strip())
    except ValueError:
        mb = DEFAULT_MB
    return max(0, mb) << 20


def model_id(model_name: Any) -> str:
    """The directory's model key: the SERVED model name -- the name the shared KV store keys its pages with, the
    one P and D share. NOT the checkpoint path: rc30 boots P from ...-INT4-Mixed-AutoRound-... and D from
    ...-MTP-INT4-g32-... (boot 1d822bd150_1021_063252), both served as one model, D reading P's KV pages."""
    p = str(model_name or "")
    return hashlib.sha1(p.encode("utf-8", "replace")).hexdigest()[:16]


def store_dir(model_name: Any, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The shared directory of this model's rows, or None when the store is off / no model is named."""
    e = os.environ if env is None else env
    if cap_bytes(e) <= 0:
        return None
    override = str(e.get(ENV_DIR, "") or "").strip()
    if override:
        return override
    if not model_name:
        return None
    return os.path.join(ROOT, PREFIX + model_id(model_name))


def hash_ok(server_args: Any) -> str:
    """Empty when the item hash is the content hash computed alike in every process, else why not
    (``d_mm_item_cache.hash_reason``; a position- or GPU-hash would key other rows)."""
    if server_args is None:
        return "no server_args"
    try:
        from flliper.srt.pdflip.d_mm_item_cache import hash_reason

        return hash_reason(server_args)
    except Exception as exc:  # noqa: BLE001 -- no proof of a content hash is no store
        return f"hash_reason unavailable ({type(exc).__name__})"


def store_dir_for(server_args: Any, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The directory for a process with these server args, or None (off, no content hash)."""
    if hash_ok(server_args):
        return None
    name = getattr(server_args, "served_model_name", None) or getattr(server_args, "model_path", None)
    return store_dir(name, env)


# ---------------------------------------------------------------------------- items
def item_grid(it: Any) -> Optional[Tuple[int, ...]]:
    """The item's ``image_grid_thw`` as an int tuple, or None when it cannot be read (no identity)."""
    msd = getattr(it, "__dict__", {}).get("model_specific_data")
    g = msd.get("image_grid_thw") if isinstance(msd, dict) else None
    if g is None:
        try:
            g = getattr(it, "image_grid_thw", None)
        except Exception:  # noqa: BLE001 -- MultimodalDataItem.__getattr__ raises for a missing key
            g = None
    if g is None:
        return None
    try:
        flat = torch.as_tensor(g).reshape(-1).tolist()
    except Exception:  # noqa: BLE001
        return None
    return tuple(int(x) for x in flat) if flat else None


def item_hash(it: Any) -> Optional[int]:
    h = getattr(it, "hash", None)
    try:
        return None if h is None else int(h)
    except (TypeError, ValueError):
        return None


def item_offsets(it: Any) -> List[Tuple[int, int]]:
    offs = getattr(it, "offsets", None)
    out: List[Tuple[int, int]] = []
    for off in (() if offs is None else offs):
        try:
            out.append((int(off[0]), int(off[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def item_rows(it: Any) -> int:
    """The item's placeholder count (offsets are inclusive)."""
    return sum(b - a + 1 for a, b in item_offsets(it))


def per_image(it: Any) -> bool:
    """One offset per item: the per-image path (``_get_chunked_embedding_by_item``) serves it."""
    return len(item_offsets(it)) == 1


def rows_of(it: Any) -> Optional[torch.Tensor]:
    d = getattr(it, "__dict__", None)
    return None if not isinstance(d, dict) else d.get(ITEM_ATTR)


def has_rows(it: Any) -> bool:
    return rows_of(it) is not None


def attach(items: Sequence[Any], rows: Sequence[torch.Tensor]) -> int:
    """Attach one row tensor per item (all or nothing: the caller checked count and shape)."""
    if len(items) != len(rows):
        raise ValueError(f"{len(rows)} row tensors for {len(items)} items")
    for it, r in zip(items, rows):
        if int(r.shape[0]) != item_rows(it):
            raise ValueError(f"rows {tuple(r.shape)} for an item of {item_rows(it)} placeholders")
    n = 0
    for it, r in zip(items, rows):
        it.__dict__[ITEM_ATTR] = r
        n += int(r.shape[0])
    return n


# ---------------------------------------------------------------------------- the files
def key_name(h: int, grid: Sequence[int]) -> str:
    return "%016x-%s%s" % (int(h) & ((1 << 64) - 1), "x".join(str(int(x)) for x in grid), SUFFIX)


def _path(d: str, h: int, grid: Sequence[int]) -> str:
    return os.path.join(d, key_name(h, grid))


def has(d: Optional[str], h: Optional[int], grid: Optional[Sequence[int]]) -> bool:
    if not d or h is None or not grid:
        return False
    return os.path.isfile(_path(d, h, grid))


#: a writer's temporary file older than this is a crashed write: removed at the next publish
STALE_TMP_S = 300.0


def _dir_entries(d: str) -> List[Tuple[float, int, str]]:
    out = []
    now = time.time()
    try:
        with os.scandir(d) as it:
            for e in it:
                if ".tmp." in e.name:
                    try:
                        if now - e.stat().st_mtime > STALE_TMP_S:
                            os.unlink(e.path)
                    except OSError:
                        pass
                    continue
                if not e.name.endswith(SUFFIX):
                    continue
                try:
                    st = e.stat()
                except OSError:
                    continue
                out.append((st.st_mtime, st.st_size, e.path))
    except OSError:
        return []
    return out


def _evict_to(d: str, cap: int, incoming: int) -> int:
    """Drop the least recently used files until ``incoming`` more bytes fit under ``cap``. Returns bytes kept."""
    entries = sorted(_dir_entries(d))
    total = sum(s for _t, s, _p in entries)
    for _t, s, p in entries:
        if total + incoming <= cap:
            break
        try:
            os.unlink(p)
            total -= s
        except OSError:
            continue
    return total


def put(d: Optional[str], h: Optional[int], grid: Optional[Sequence[int]], rows: Any,
        cap: Optional[int] = None) -> bool:
    """Publish one image's rows (host tensor, 2-D). True when the file exists afterwards."""
    if not d or h is None or not grid or not isinstance(rows, torch.Tensor):
        return False
    if rows.is_cuda or rows.dim() != 2:
        return False
    dt = {v: k for k, v in _DTYPES.items()}.get(rows.dtype)
    if dt is None:
        return False
    cap = cap_bytes() if cap is None else int(cap)
    p = _path(d, h, grid)
    if os.path.isfile(p):
        try:
            os.utime(p)
        except OSError:
            pass
        return True
    t = rows.detach().contiguous()
    payload = t.view(torch.uint8).numpy().tobytes() if t.numel() else b""
    head = json.dumps({"hash": "%016x" % (int(h) & ((1 << 64) - 1)), "grid": [int(x) for x in grid],
                       "rows": int(t.shape[0]), "width": int(t.shape[1]), "dtype": dt},
                      separators=(",", ":")).encode() + b"\n"
    size = len(MAGIC) + len(head) + len(payload)
    if size > cap:
        return False
    try:
        os.makedirs(d, exist_ok=True)
        _evict_to(d, cap, size)
        tmp = "%s.tmp.%d.%d" % (p, os.getpid(), time.monotonic_ns())
        with open(tmp, "wb") as f:
            f.write(MAGIC)
            f.write(head)
            f.write(payload)
        os.replace(tmp, p)
    except OSError as exc:
        logger.warning("%s PUT-FAILED key=%s %s: %s", TAG, key_name(h, grid), type(exc).__name__, exc)
        return False
    return True


def get(d: Optional[str], h: Optional[int], grid: Optional[Sequence[int]],
        width: Optional[int] = None) -> Optional[torch.Tensor]:
    """One image's rows (a host tensor), or None: missing, or a header that is not exactly this identity."""
    if not d or h is None or not grid:
        return None
    p = _path(d, h, grid)
    try:
        with open(p, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if not data.startswith(MAGIC):
        return None
    nl = data.find(b"\n", len(MAGIC))
    if nl < 0:
        return None
    try:
        head = json.loads(data[len(MAGIC):nl].decode())
        dtype = _DTYPES[str(head["dtype"])]
        nrows, w = int(head["rows"]), int(head["width"])
        same = (str(head["hash"]) == "%016x" % (int(h) & ((1 << 64) - 1))
                and tuple(int(x) for x in head["grid"]) == tuple(int(x) for x in grid))
    except (ValueError, KeyError, TypeError):
        return None
    if not same or (width is not None and w != int(width)):
        return None
    raw = data[nl + 1:]
    elsize = torch.tensor([], dtype=dtype).element_size()
    if len(raw) != nrows * w * elsize:
        return None
    try:
        os.utime(p)
    except OSError:
        pass
    if not raw:
        return torch.empty((nrows, w), dtype=dtype)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(nrows, w)


# ---------------------------------------------------------------------------- the three callers
def publish(items: Iterable[Any], d: Optional[str], cap: Optional[int] = None) -> int:
    """P: publish every item's attached rows (``precomputed_embeddings``, host) under (hash, grid)."""
    if not d:
        return 0
    n = 0
    for it in items:
        rows = getattr(it, "precomputed_embeddings", None)
        if not isinstance(rows, torch.Tensor) or rows.dim() != 2 or int(rows.shape[0]) != item_rows(it):
            continue
        h, grid = item_hash(it), item_grid(it)
        if h is None or grid is None:
            continue  # B1: no grid, no identity -- never published
        if put(d, h, grid, rows, cap):
            n += 1
    return n


def available_spans(items: Iterable[Any], d: Optional[str]) -> List[List[int]]:
    """D's tokenizer, for the probe answer: the ``[start, end)`` spans of the per-image items whose rows are
    published (the same span form as ``d_match_probe.mm_spans``)."""
    out: List[List[int]] = []
    items = list(items or ())
    if not d or not all(per_image(it) for it in items):
        return out  # the scheduler attaches only to an all-per-image request (mm_utils' per-image path)
    for it in items:
        if has(d, item_hash(it), item_grid(it)):
            a, b = item_offsets(it)[0]
            out.append([a, b + 1])
    out.sort()
    return out


def load(items: Sequence[Any], d: Optional[str], width: Optional[int] = None) -> Optional[List[torch.Tensor]]:
    """D's rank 0: the rows of EVERY item, or None (any one missing / not per-image / a count or width that does
    not fit). All or nothing -- a half attached request would reach the tower for the rest."""
    if not d or not items:
        return None
    out: List[torch.Tensor] = []
    for it in items:
        if not per_image(it):
            return None
        r = get(d, item_hash(it), item_grid(it), width)
        if r is None or int(r.shape[0]) != item_rows(it):
            return None
        out.append(r)
    return out
