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

ENV = "FLLIPER_PDFLIP_HANDOFF"


def _dir() -> str:
    if os.environ.get(ENV, "1") != "1":
        return ""
    base = os.environ.get("FLLIPER_HICACHE_ARENA_DIR", "").strip()
    return os.path.join(base, "handoff") if base else ""


#: DP-NACHLAUF 02.10. (N5t epoch 11, P>D quiesce 443 ms): every P stage wrote
#: the SAME hand-off file at its request finish (json.dump of 98880 ids +
#: 98875 hex keys, 7.5 MB, ~42 ms streamed through the pure-Python encoder),
#: PP2's being the last work before P reports idle. Only PP0 writes it now
#: (its finish lies before PP1/PP2's drain of the last chunk), in one shot
#: (C encoder, ~18 ms, byte-identical file). Unset = on; 0/false/no/off =
#: every stage writes, streamed, as before.
ONE_WRITER_ENV = "FLLIPER_PDFLIP_HANDOFF_ONE_WRITER"


def one_writer_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ONE_WRITER_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def writes_on_stage(pp_rank) -> bool:
    """Whether a stage writes the hand-off file (all stages with the switch off)."""
    try:
        return (not one_writer_on()) or int(pp_rank or 0) == 0
    except (TypeError, ValueError):
        return True


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
        obj = {"input_ids": list(int(t) for t in input_ids), "page_keys": list(page_keys)}
        with open(tmp, "w") as f:
            if one_writer_on():
                f.write(json.dumps(obj))   # one shot: the C encoder, same bytes
            else:
                json.dump(obj, f)
        os.replace(tmp, p)
        return True
    except Exception:  # noqa: BLE001 - a hand-off that fails is the old path
        logger.warning("#1442 hand-off write failed for rid=%s", rid, exc_info=True)
        return False


def exists(rid: Optional[str]) -> bool:
    """A hand-off for this rid is waiting (REUSE-TEXT-IDS-1007: D's chat serving
    leaves such a request on the text path, whose tokenizer manager takes P's
    ids). No file read, one stat."""
    p = path(rid) if isinstance(rid, str) else ""
    return bool(p) and os.path.exists(p)


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


#: S58 (19.10., NF rc15 1019_071429): the hand-off file outlived its rid. The
#: only reader that removed it (``read_keys(remove=True)``) lost its caller
#: with xsn330 (``remove=False``: the dormant hold re-reads the keys every
#: 2 s), and the scheduler's hold release removes it only for rids that sat
#: in the dormant hold. Every other P-prefilled rid left its file in the
#: shared arena directory on /dev/shm for the rest of the boot: measured
#: 08:40Z, 85 min into the boot, 565 files / 318 MiB in ``handoff/`` against
#: ONE pending and ONE park mark (63-84 files and 17-80 MiB per 10 min, i.e.
#: ~0.25 GiB/h of shmem under memory.max, unbounded, on a 16 GiB /dev/shm
#: that the arena files share). The front's rid-end drop
#: (``handoff_pending.drop`` via ``handoff_seam.wrap_handler``, every return,
#: exception and cancel of the generate handler, plus the W35/reroute drops)
#: is the one site that knows the rid is over; it removes the file there.
#: Unset = on; 0/false/no/off = the file stays, as before.
REMOVE_AT_RID_END_ENV = "FLLIPER_PDFLIP_HANDOFF_REMOVE_AT_RID_END"


def remove_at_rid_end_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(REMOVE_AT_RID_END_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def remove(rid: Optional[str]) -> bool:
    """Remove ``rid``'s hand-off file (and a leftover ``*.tmp`` of a writer
    that died mid-write is not touched: it carries a pid suffix and is
    replaced by the next write). True when a file was removed; never raises.
    Absent file = nothing to do (D then takes the old path, byte for byte)."""
    p = path(rid) if isinstance(rid, str) else ""
    if not p:
        return False
    try:
        os.remove(p)
        return True
    except OSError:
        return False


def read_keys(rid: str, remove: bool = True) -> Optional[list]:
    d = read(rid)
    keys = d.get("page_keys") if d else None
    if remove:
        try:
            os.remove(path(rid))
        except OSError:
            pass
    return list(keys) if keys else None
