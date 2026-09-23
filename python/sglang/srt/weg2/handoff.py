"""#1442: the P -> D hand-off of a finished leg-1 request, by rid, through the
shared arena directory (/dev/shm, visible to every rank of both groups).

What P knows when its prefill finished -- the token ids the tokenizer produced
and the page-key chain of the radix path it inserted -- D re-derived from the
text: it re-tokenised the 100k prompt and re-hashed the chain before it could
even ask the store. Both are handed over instead. The file is small (ids as
JSON, keys as hex), written atomically, read once by D's tokenizer manager
(ids) and once by D's scheduler (keys), and removed after the keys were read.

The rid is the front's (payload["rid"], #1442 sets it), so both legs of one
request share it. Absent file = the old path, byte for byte.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_HANDOFF"


def _dir() -> str:
    if os.environ.get(ENV, "1") != "1":
        return ""
    base = os.environ.get("SGLANG_HICACHE_ARENA_DIR", "").strip()
    return os.path.join(base, "handoff") if base else ""


def path(rid: str) -> str:
    d = _dir()
    return os.path.join(d, f"{rid}.json") if d and rid else ""


def write(rid: str, input_ids: Sequence[int], page_keys: Sequence[str]) -> bool:
    p = path(rid)
    if not p:
        return False
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"input_ids": list(int(t) for t in input_ids), "page_keys": list(page_keys)}, f)
        os.replace(tmp, p)
        return True
    except Exception:  # noqa: BLE001 - a hand-off that fails is the old path
        logger.warning("#1442 hand-off write failed for rid=%s", rid, exc_info=True)
        return False


def read(rid: str) -> Optional[dict]:
    p = path(rid)
    if not p or not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        logger.warning("#1442 hand-off read failed for rid=%s", rid, exc_info=True)
        return None


def read_ids(rid: str) -> Optional[list]:
    d = read(rid)
    ids = d.get("input_ids") if d else None
    return list(ids) if ids else None


def read_keys(rid: str, remove: bool = True) -> Optional[list]:
    d = read(rid)
    keys = d.get("page_keys") if d else None
    if remove:
        try:
            os.remove(path(rid))
        except OSError:
            pass
    return list(keys) if keys else None
