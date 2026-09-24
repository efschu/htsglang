"""fnFL2 H29: what P saw at the END of a prompt, handed to D for the decode.

TWO SMALL FILES BESIDE THE TAIL HAND-OFF (``<SGLANG_HICACHE_ARENA_DIR>/handoff``),
written by P after every prefill forward (the last write before the flip is the
last chunk of the flipped prompt) and read by D once after the wake:

``lru_route.<stage>.json`` (H29b, SGLANG_WEG2_LRU_WARM_FROM_HANDOFF)
    per MoE layer of this P stage: the GLOBAL expert ids routed by the last
    ``SGLANG_WEG2_LRU_WARM_TOKENS`` tokens of the forward, with their counts,
    most-routed first. D fills the LRU rows its rearm left free with them.

``ple_rows.npy`` (H29a, SGLANG_WEG2_PLE_DECODE_PREFETCH)
    the PLE table rows of P's last prefill gather, repeated rows first (by
    multiplicity), then the rows of the last tokens. D faults their pages into
    its own mapping before its decode gathers read them.

Nothing here is needed for correctness: a missing, stale or unreadable file
means "no warm", never a refusal. A file older than
``SGLANG_WEG2_DECODE_WARM_MAX_AGE_S`` is another prompt's and is ignored.
Writes are atomic (tmp + ``os.replace``), so D never reads a torn file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

ROUTE_PREFIX = "lru_route."
PLE_ROWS_FILE = "ple_rows.npy"
#: rows P publishes for the PLE warm at most (8 B each; D caps by pages)
PLE_PUBLISH_MAX_ROWS = 1 << 16
#: rows of the last tokens that always follow the repeated rows (16 per token)
PLE_TAIL_ROWS = 4096 * 16


def warm_dir() -> str:
    base = os.environ.get("SGLANG_HICACHE_ARENA_DIR", "").strip()
    return os.path.join(base, "handoff") if base else ""


def _atomic_write(path: str, write) -> bool:
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.debug("decode-warm publish %s failed: %s", path, exc)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def _fresh(path: str, max_age_s: float, now: Optional[float] = None) -> Optional[float]:
    """The file's age in seconds, or None if it is missing or too old."""
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return None
    age = (time.time() if now is None else now) - mtime
    return age if age <= max_age_s else None


# ---- H29b: P's last routing ------------------------------------------------
def route_counts(rows: np.ndarray, num_experts: int) -> List[Tuple[int, int]]:
    """``rows`` [T, k] expert ids (padding < 0) -> [(expert, count)], most
    routed first, ties by expert id (deterministic)."""
    flat = np.asarray(rows, dtype=np.int64).reshape(-1)
    flat = flat[(flat >= 0) & (flat < int(num_experts))]
    if flat.size == 0:
        return []
    ids, counts = np.unique(flat, return_counts=True)
    order = np.lexsort((ids, -counts))
    return [(int(ids[i]), int(counts[i])) for i in order]


class RouteRecorder:
    """P side: the last forward's tail routing per layer of this process.

    ``note`` is called by every MoE layer's eager forward with the host ids it
    already has (no extra device sync); the file is written when the LAST
    layer this process registered has noted, i.e. once per forward.
    """

    def __init__(self) -> None:
        self._layers: set = set()
        self._routes: Dict[int, List[Tuple[int, int]]] = {}

    def register(self, layer_id: int) -> None:
        self._layers.add(int(layer_id))

    @property
    def last_layer(self) -> Optional[int]:
        return max(self._layers) if self._layers else None

    def stage_tag(self) -> str:
        return f"L{min(self._layers)}-{max(self._layers)}" if self._layers else "L"

    def note(self, layer_id: int, rows: np.ndarray, num_experts: int,
             tokens: int) -> Optional[str]:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 2:
            raise ValueError("RouteRecorder.note wants [T, k] ids")
        tail = rows[-int(tokens):] if tokens > 0 else rows
        self._routes[int(layer_id)] = route_counts(tail, num_experts)
        if int(layer_id) != self.last_layer:
            return None
        return self.publish()

    def publish(self, directory: Optional[str] = None) -> Optional[str]:
        d = warm_dir() if directory is None else directory
        if not d or not self._routes:
            return None
        path = os.path.join(d, f"{ROUTE_PREFIX}{self.stage_tag()}.json")
        body = json.dumps({
            "t": time.time(),
            "pid": os.getpid(),
            "layers": {str(k): v for k, v in sorted(self._routes.items())},
        }).encode()
        return path if _atomic_write(path, lambda f: f.write(body)) else None


def load_routes(directory: Optional[str] = None, max_age_s: Optional[float] = None
                ) -> Tuple[Dict[int, List[Tuple[int, int]]], int]:
    """D side: every fresh stage file merged -> ({layer_id: [(expert, count)]},
    number of files read). A layer in two files takes the newer one."""
    d = warm_dir() if directory is None else directory
    if max_age_s is None:
        max_age_s = float(envs.SGLANG_WEG2_DECODE_WARM_MAX_AGE_S.get())
    out: Dict[int, List[Tuple[int, int]]] = {}
    stamp: Dict[int, float] = {}
    files = 0
    if not d or not os.path.isdir(d):
        return out, 0
    for name in sorted(os.listdir(d)):
        if not (name.startswith(ROUTE_PREFIX) and name.endswith(".json")):
            continue
        path = os.path.join(d, name)
        if _fresh(path, max_age_s) is None:
            continue
        try:
            with open(path, "rb") as f:
                doc = json.loads(f.read())
            t = float(doc.get("t", 0.0))
            layers = doc.get("layers", {})
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        files += 1
        for k, v in layers.items():
            lid = int(k)
            if lid in stamp and stamp[lid] >= t:
                continue
            stamp[lid] = t
            out[lid] = [(int(e), int(c)) for e, c in v]
    return out, files


# ---- H29a: P's last PLE rows ------------------------------------------------
def select_ple_rows(ids: np.ndarray, max_rows: int = PLE_PUBLISH_MAX_ROWS,
                    tail_rows: int = PLE_TAIL_ROWS) -> np.ndarray:
    """``ids`` = one gather's row ids in token order -> the rows worth warming:
    rows that occur more than once (most frequent first), then the distinct
    rows of the last ``tail_rows`` ids (latest first), capped at ``max_rows``."""
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    ids = ids[ids >= 0]
    if ids.size == 0:
        return ids
    u, counts = np.unique(ids, return_counts=True)
    rep = counts > 1
    ru, rc = u[rep], counts[rep]
    first = ru[np.lexsort((ru, -rc))]
    tail = ids[-int(tail_rows):][::-1]
    _, idx = np.unique(tail, return_index=True)
    tail = tail[np.sort(idx)]
    tail = tail[~np.isin(tail, first)]
    return np.concatenate([first, tail])[: int(max_rows)]


def publish_ple_rows(ids: np.ndarray, directory: Optional[str] = None) -> Optional[str]:
    d = warm_dir() if directory is None else directory
    if not d:
        return None
    rows = select_ple_rows(ids)
    if rows.size == 0:
        return None
    path = os.path.join(d, PLE_ROWS_FILE)
    return path if _atomic_write(path, lambda f: np.save(f, rows)) else None


def load_ple_rows(directory: Optional[str] = None, max_age_s: Optional[float] = None
                  ) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """D side: (rows, mtime) of the fresh file, or (None, None)."""
    d = warm_dir() if directory is None else directory
    if max_age_s is None:
        max_age_s = float(envs.SGLANG_WEG2_DECODE_WARM_MAX_AGE_S.get())
    if not d:
        return None, None
    path = os.path.join(d, PLE_ROWS_FILE)
    if _fresh(path, max_age_s) is None:
        return None, None
    try:
        mtime = os.stat(path).st_mtime
        rows = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None, None
    return np.asarray(rows, dtype=np.int64).reshape(-1), mtime


__all__: Sequence[str] = (
    "RouteRecorder", "route_counts", "load_routes", "select_ple_rows",
    "publish_ple_rows", "load_ple_rows", "warm_dir",
)
