"""#243 HANDOFF-PENDING: a P hand-off waiting for its D seat is evicted LAST.

THE LOSS (rc12r 09271632, weg2-12-39; tmp/r989/befund_243_weg2-12-39.md). P
published the whole hand-off at 16:53:40 (END-ANCHOR ok on PP0/1/2,
TAIL-PUBLISH 76544 + 58 rows). Its KV pages lost their last holder at the
flip, 5 s later (#1427 RESET-RELEASE released=3940, FULL sum=0). Its mamba end
anchors lost theirs at P's next wake (H81 CARRIER-HOLD released=6 at=wake,
"the D phase that read them is over"). The request was still waiting for a
D seat (6/6 taken, D-ADMIT oldest_wait_s=255.3). By the time D admitted it
at 16:57:10, the claims of other rids had taken every page:

* ``#1028B FETCH CAP kv=1196 claimed=0 lost=1196``, anchors 0;
* then X-GATE W31, a second P prefill, and the client reset.

A hand-off's life was bound to PHASES (P's reset, one flip of carrier hold)
when it should be bound to CONSUMPTION (D takes the rid).

THE HOLDER IS AN EVICTION ORDER, NOT A REFERENCE.

* A reader reference cannot be evicted. A reference held across several
  flips would need a budget, and without one the arena fills and every
  writer gets ARENA-CLAIM REFUSED (the fnNV4f2 class).
* An order needs no budget. It is bounded by the arena itself, and what
  still has to go is named.

The mechanics:

* **Mark.** P marks the rid when it publishes the hand-off:
  ``<arena>/handoff/pending/<rid>`` (:func:`mark`, group P only).
* **Keep.** Every claim that has to make room (``_evict_for_claim``, KV and
  mamba arena, any group, any rank) first evicts everything else. It passes
  the keys of the marked rids as the arena's keep list
  (:func:`keep_for`):
  * KV: every page of the hand-off chain;
  * mamba: the end anchor, which is the last two chain keys (the N-1 anchor
    sits on the last page, or one page short when N is page-aligned).
* **Last.** Only if the claim still lacks room are kept slots evicted. Each
  such rid is named with a ``HANDOFF-LOST rid= pool= first_lost_page=`` line
  and a status file ``<arena>/handoff/lost/<rid>.json``
  (:func:`note_evicted`).
* **Consume.** The mark goes when D takes the rid: at the group-uniform
  prefetch termination, or at the rid's end on D. The front also drops it
  (:func:`drop`) at every rid end it sees, served or aborted. A mark nobody
  ended leaves the order after ``SGLANG_WEG2_HANDOFF_PENDING_EXPIRE_S``
  (EXPIRED, 900 s) and reads LOST (reason=expired, first_lost_page=0), never
  none -- a rid still waiting is re-routed fresh, never mispriced. A garbage
  bound, never a capacity. (Without a clock: the front reconciling marks
  against its live rids -- PA follow-up.)

Any rank may consume first. Every rank's pages are referenced from the moment
its prefetch resolved them, and the termination is a group vote that only
passes once every rank has resolved.

THE FRONT SEAM (27B-PA, shared code). The two front entry points are
:func:`status` and :func:`drop`. Neither raises; when in doubt,
``state == "none"``.

A park PINS its pages by reference, and a pending hand-off is only KEPT by
order, so a park always wins. That is intended. The HOLDERS census line shows
both side by side: ``arena_pinned`` (every process's referenced complete
slots) next to ``handoff_kept`` (kept keys still complete).

Host bookkeeping only: small files in the shared arena directory, no device
work, no collective. Absent directory (no arena, hand-off off) = the old path,
byte for byte.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

PENDING = "pending"
LOST = "lost"
#: the mamba end anchor lies on the last chain page, or one page short when the
#: prompt length is page-aligned (N-1 floors to the previous page)
ANCHOR_TAIL_KEYS = 2
#: class attribute of the arena host pools: which keys of a chain they keep
ROLE_ATTR = "_weg2_handoff_keep"
#: (rid, pool role) -> pages this process already named lost (log rate)
_LOST_SEEN: dict = {}


def _base() -> str:
    try:
        from sglang.srt.weg2 import handoff as _ho

        return _ho._dir()
    except Exception:  # noqa: BLE001
        return ""


def _sub(name: str) -> str:
    b = _base()
    return os.path.join(b, name) if b else ""


def _expire_s() -> float:
    try:
        from sglang.srt.environ import envs

        return float(envs.SGLANG_WEG2_HANDOFF_PENDING_EXPIRE_S.get())
    except Exception:  # noqa: BLE001
        return 900.0


def _group() -> str:
    return (os.environ.get("SGLANG_WEG2_GROUP", "") or "").strip().upper()


def _write_atomic(path: str, rec: dict) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, path)
        return True
    except Exception:  # noqa: BLE001 - a mark that fails is the old path
        logger.warning("#243 HANDOFF-PENDING write failed: %s", path, exc_info=True)
        return False


def _read(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _unlink(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


# -- the rid's state ---------------------------------------------------------
def mark(rid: str, pages: int, page_size: int = 0) -> bool:
    """P published rid's hand-off (``pages`` chain pages): keep it until D
    takes it. Group P only; a later publish of the same rid (a re-prefill)
    renews the mark and clears an earlier loss."""
    if _group() != "P" or not rid:
        return False
    d = _sub(PENDING)
    if not d:
        return False
    _unlink(os.path.join(_sub(LOST), f"{rid}.json"))
    return _write_atomic(os.path.join(d, rid), {
        "t": time.time(), "pages": int(pages), "page_size": int(page_size), "pid": os.getpid()})


def _end(rid: str, where: str, verb: str, **kv) -> bool:
    d = _sub(PENDING)
    if not d or not rid:
        return False
    rec = _read(os.path.join(d, str(rid)))
    gone = _unlink(os.path.join(d, str(rid)))
    lost = _unlink(os.path.join(_sub(LOST), f"{rid}.json"))
    if gone:
        held = (time.time() - float(rec.get("t", time.time()))) if rec else -1.0
        logger.info("#243 HANDOFF-PENDING %s rid=%s at=%s held_s=%.1f lost=%s%s", verb, rid, where, held,
                    lost, "".join(f" {k}={v}" for k, v in kv.items()))
    return gone


def _expire(rid: str, age: float) -> None:
    """A mark nobody ended within SGLANG_WEG2_HANDOFF_PENDING_EXPIRE_S leaves
    the order -- and is reported LOST (first_lost_page=0, reason=expired),
    never none: a rid still waiting for its seat is then routed fresh via P
    by the front instead of priced on pages nobody protects any more."""
    rec = _read(os.path.join(_sub(PENDING), rid)) or {}
    ldir = _sub(LOST)
    if ldir:
        _write_atomic(os.path.join(ldir, f"{rid}.json"), {
            "rid": rid, "state": "lost", "reason": "expired", "first_lost_page": 0,
            "pages": rec.get("pages"), "page_size": rec.get("page_size"), "t": time.time(),
            "group": _group(), "pid": os.getpid()})
    if _unlink(os.path.join(_sub(PENDING), rid)):
        logger.warning("#243 HANDOFF-PENDING EXPIRED rid=%s age_s=%.0f -> status lost (reason=expired, "
                       "first_lost_page=0): no D take and no front drop within the bound", rid, age)


def consume(rid: str, where: str, **kv) -> bool:
    """D took rid (prefetch terminated, or the rid ended on D): the order
    protection ends. Every rank may call it; the first one removes the mark."""
    try:
        return _end(str(rid), where, "CONSUMED", **kv)
    except Exception:  # noqa: BLE001 - bookkeeping never breaks the caller
        return False


def drop(rid: str, reason: str = "front") -> bool:
    """FRONT SEAM: rid ended where D never took it (served via P, aborted,
    disconnected, re-routed fresh). Never raises."""
    try:
        return _end(str(rid), str(reason), "DROPPED")
    except Exception:  # noqa: BLE001
        return False


def status(rid: str) -> dict:
    """FRONT SEAM: ``{state: pending|lost|none, first_lost_page, pages,
    page_size, reason}``. ``lost`` wins over ``pending`` (the mark stays to
    keep the surviving prefix); an expired mark is ``lost`` with
    ``reason=expired``, ``first_lost_page=0``. Never raises; in doubt
    ``none``."""
    out = {"state": "none", "first_lost_page": None, "pages": None, "page_size": None, "reason": None}
    try:
        rid = str(rid)
        rec = _read(os.path.join(_sub(PENDING), rid)) if _sub(PENDING) else None
        if rec is not None:
            out.update(state="pending", pages=rec.get("pages"), page_size=rec.get("page_size") or None)
        lost = _read(os.path.join(_sub(LOST), f"{rid}.json")) if _sub(LOST) else None
        if lost is not None:
            out.update(state="lost", first_lost_page=lost.get("first_lost_page"),
                       pages=lost.get("pages", out["pages"]),
                       page_size=lost.get("page_size") or out["page_size"],
                       reason=lost.get("reason") or "evicted")
    except Exception:  # noqa: BLE001
        return {"state": "none", "first_lost_page": None, "pages": None, "page_size": None, "reason": None}
    return out


# -- the keep list -----------------------------------------------------------
class Keep:
    """The kept keys of one arena pool: ``keys`` sorted uint64 (the arena's
    key_lo), ``rid_ix``/``page`` aligned with them, ``rids`` the names."""

    __slots__ = ("keys", "rid_ix", "page", "rids", "stems")

    def __init__(self, keys, rid_ix, page, rids, stems):
        self.keys, self.rid_ix, self.page, self.rids, self.stems = keys, rid_ix, page, rids, stems

    def __len__(self) -> int:
        return int(self.keys.shape[0])


def _empty_keep() -> Keep:
    import numpy as np

    z = np.zeros(0, dtype=np.uint64)
    return Keep(z, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), [], [])


def _rid_keys(pool, rid: str):
    """(key_lo uint64[], page idx int64[], stems) this pool keeps for rid."""
    import numpy as np

    from sglang.srt.mem_cache.storage.file.hicache_arena import stem_keys_lo
    from sglang.srt.weg2 import handoff as _ho

    rec = _ho.read(rid)
    chain = list((rec or {}).get("page_keys") or ())
    if not chain:
        return None
    role = getattr(pool, ROLE_ATTR, "kv")
    first = max(0, len(chain) - ANCHOR_TAIL_KEYS) if role == "anchor" else 0
    stems = pool._stems(chain[first:])
    lo = stem_keys_lo(stems)
    return lo, np.arange(first, len(chain), dtype=np.int64), stems


def keep_for(pool) -> Keep:
    """The kept keys for ``pool`` now. Cached on the pool; re-read only when
    the pending directory changed (its mtime), and then only the new rids."""
    import numpy as np

    d = _sub(PENDING)
    if not d:
        return _empty_keep()
    try:
        stamp = os.stat(d).st_mtime_ns
    except OSError:
        return _empty_keep()
    cache = pool.__dict__.get("_weg2_hp_keep")
    if cache is not None and cache[0] == stamp:
        return cache[1]
    per = pool.__dict__.setdefault("_weg2_hp_rid_keys", {})
    try:
        entries = [e for e in os.scandir(d) if not e.name.endswith(".tmp")]
    except OSError:
        entries = []
    now, rids = time.time(), []
    ttl = _expire_s()
    for e in entries:
        try:
            age = now - e.stat().st_mtime
        except OSError:
            continue
        if ttl > 0 and age > ttl:
            _expire(e.name, age)
            continue
        rids.append((e.name, e.stat().st_mtime_ns))
    rids.sort()
    live = dict(rids)
    for gone in [r for r in per if r not in live]:
        per.pop(gone, None)
    for rid, mtime in rids:
        # a renewed mark (P re-prefilled the rid, e.g. RESUME-VIA-P with the
        # context grown) re-reads the chain; an unchanged one is not re-hashed
        if rid not in per or per[rid][0] != mtime:
            try:
                per[rid] = (mtime, _rid_keys(pool, rid))
            except Exception:  # noqa: BLE001 - a rid we cannot read keeps nothing
                logger.warning("#243 HANDOFF-PENDING keys of rid=%s unreadable", rid, exc_info=True)
                per[rid] = (mtime, None)
    names = [r for r, _ in rids if per.get(r, (0, None))[1] is not None]
    if not names:
        keep = _empty_keep()
    else:
        parts = [per[r][1] for r in names]
        keys = np.concatenate([p[0] for p in parts])
        rid_ix = np.concatenate([np.full(p[0].shape[0], i, dtype=np.int64) for i, p in enumerate(parts)])
        page = np.concatenate([p[1] for p in parts])
        order = np.argsort(keys, kind="stable")
        keep = Keep(keys[order], rid_ix[order], page[order], names,
                    [s for p in parts for s in p[2]])
    pool.__dict__["_weg2_hp_keep"] = (stamp, keep)
    return keep


def note_evicted(pool, cands: Iterable, keep: Keep, need: int = 0) -> int:
    """Name the kept slots the claim had to take after all (the second, keep-
    less pass): one HANDOFF-LOST line and status file per rid. ``cands`` are
    the arena's (slot, key_lo, key_hi, total). Returns the rids named."""
    import numpy as np

    if not len(keep):
        return 0
    lo = np.fromiter((int(c[1]) & 0xFFFFFFFFFFFFFFFF for c in cands), dtype=np.uint64)
    if lo.shape[0] == 0:
        return 0
    pos = np.searchsorted(keep.keys, lo)
    pos = np.minimum(pos, keep.keys.shape[0] - 1)
    hit = keep.keys[pos] == lo
    if not bool(hit.any()):
        return 0
    per: dict = {}
    for p in pos[hit].tolist():
        # the same key may belong to several rids (a shared prefix): all lose it
        k = keep.keys[p]
        j = p
        while j >= 0 and keep.keys[j] == k:
            j -= 1
        j += 1
        while j < keep.keys.shape[0] and keep.keys[j] == k:
            r = int(keep.rid_ix[j])
            per.setdefault(r, []).append(int(keep.page[j]))
            j += 1
    role = getattr(pool, ROLE_ATTR, "kv")
    ldir = _sub(LOST)
    for r, pages in per.items():
        rid = keep.rids[r]
        pend = _read(os.path.join(_sub(PENDING), rid)) or {}
        first = min(pages)
        path = os.path.join(ldir, f"{rid}.json") if ldir else ""
        prev = _read(path) if path else None
        if prev is not None and prev.get("first_lost_page") is not None:
            first = min(first, int(prev["first_lost_page"]))
        held = time.time() - float(pend.get("t", time.time()))
        if path:
            _write_atomic(path, {"rid": rid, "state": "lost", "reason": "evicted", "first_lost_page": int(first),
                                 "pages": pend.get("pages"), "page_size": pend.get("page_size"),
                                 "pool": role, "t": time.time(), "group": _group(), "pid": os.getpid()})
        # one line per rid and pool the first time, then at every 256 pages
        # more (a full arena takes a waiting chain page by page, claim by claim)
        seen = _LOST_SEEN.get((rid, role), 0)
        _LOST_SEEN[(rid, role)] = seen + len(pages)
        if seen == 0 or (seen // 256) != ((seen + len(pages)) // 256):
            logger.warning(
                "#243 HANDOFF-LOST rid=%s pool=%s pages_lost=%d (cumulative %d) first_lost_page=%d of %s "
                "held_s=%.1f need=%d (a claim found no other slot: the hand-off waiting for its D seat "
                "was evicted LAST; D's claim stops at the first lost page, the front re-prices)",
                rid, role, len(pages), seen + len(pages), first, pend.get("pages"), held, int(need))
    while len(_LOST_SEEN) > 4096:
        _LOST_SEEN.pop(next(iter(_LOST_SEEN)))
    return len(per)


def census(pool) -> str:
    """HOLDERS-line fields for ``pool``: this pool's kept keys still COMPLETE
    in the arena, and every process's reference-pinned complete slots."""
    try:
        cache = pool.__dict__.get("_weg2_hp_keep")
        keep = cache[1] if cache is not None else keep_for(pool)
        arena = getattr(pool, "arena", None)
        present = 0
        if len(keep) and arena is not None:
            present = sum(1 for _, st in arena.find_slots(keep.stems) if int(st) == 2)
        pinned = complete = "-"
        if arena is not None and hasattr(arena, "ref_census"):
            pinned, _refs, complete = arena.ref_census()
        return (f"handoff_kept={present}/{len(keep)} handoff_rids={len(keep.rids)} "
                f"arena_pinned={pinned} arena_complete={complete}")
    except Exception as exc:  # noqa: BLE001 - an instrument never breaks a census
        return f"handoff_kept=failed:{type(exc).__name__}"
