"""fnFL2 H29: what P saw at the END of a prompt, handed to D for the decode.

SMALL FILES BESIDE THE TAIL HAND-OFF (``<SGLANG_HICACHE_ARENA_DIR>/handoff``),
written by P after every prefill forward (the last write before the flip is the
last chunk of the flipped prompt) and read by D once after the wake:

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

import logging
import os
import time
from typing import Optional, Sequence, Tuple

import numpy as np

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

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
    "select_ple_rows", "publish_ple_rows", "load_ple_rows", "warm_dir",
)
