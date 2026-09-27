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

A reference always wins over the order. The HOLDERS census line shows both
side by side: ``arena_pinned`` (every process's referenced complete slots)
next to ``handoff_kept`` (kept keys still complete).

#248 (rc12s 17:32:40: a sleeping D held 5213 of 5461 KV slots by reference)
adds the second role. A D request that does not run -- parked by the flip,
held in the dormant hold -- no longer pins by reference; D marks it
``<arena>/handoff/park/<rid>`` with the chain of its retained span
(:func:`mark_park`, group D), and the same order keeps it. The claim then has
three stages: (i) kept by nobody, (ii) kept WITH an L3 copy (the D demoter,
``weg2.park_demote``, copies every kept page to disk without freeing it --
:func:`copied_mask`), freed without I/O, read back from disk; (iii) kept
without a copy, named (HANDOFF-LOST / PARK-LOST). The census splits
``handoff_kept`` and ``park_kept``.

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
#: #248: the marks of D's parked / held rids (the chain inside the mark)
PARK = "park"
#: keep roles: a P hand-off waiting for its seat, a D request that does not run
ROLE_HANDOFF = "handoff"
ROLE_PARK = "park"
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


def mark_park(rid: str, page_keys, page_size: int = 0, page_range=None) -> bool:
    """#248: D keeps a request that does not run (parked by the flip, held in
    the dormant hold) by ORDER, not by reference: ``page_keys`` is the chain
    of the span it retained (the tree's page keys, what the write-through
    stored). Group D only; a renewed park replaces the mark and clears an
    earlier loss. The chain rides in the mark -- the hand-off file of a rid
    admitted earlier is gone by then.

    ``page_range=(a, b)`` (PA's partial park, :func:`keep_role`): the span is
    ``[0, b)``, and ``[a, b)`` goes FIRST -- the demoter copies it to L3 from
    the back before anything else, so a claim frees it first (stage ii)."""
    if _group() != "D" or not rid:
        return False
    keys = [str(k) for k in (page_keys or ())]
    d = _sub(PARK)
    if not d or not keys:
        return False
    rng = _norm_range(page_range, len(keys))
    if rng is not None:
        keys = keys[:rng[1]]
    _unlink(os.path.join(_sub(LOST), f"{rid}.json"))
    rec = {"t": time.time(), "pages": len(keys), "page_size": int(page_size), "page_keys": keys,
           "pid": os.getpid(), "role": ROLE_PARK}
    if rng is not None:
        rec["range"] = [int(rng[0]), int(rng[1])]
    return _write_atomic(os.path.join(d, rid), rec)


def _norm_range(page_range, n: int):
    """(a, b) clipped to ``[0, n]`` with a <= b; None = the whole span."""
    if page_range is None:
        return None
    try:
        a, b = int(page_range[0]), int(page_range[1])
    except Exception:  # noqa: BLE001 - a malformed range is the whole span
        return None
    b = max(0, min(b, n))
    a = max(0, min(a, b))
    return a, b


def _end(rid: str, where: str, verb: str, **kv) -> bool:
    d = _sub(PENDING)
    if not d or not rid:
        return False
    rec = _read(os.path.join(d, str(rid)))
    gone = _unlink(os.path.join(d, str(rid)))
    pd = _sub(PARK)
    prec = _read(os.path.join(pd, str(rid))) if pd else None
    pgone = _unlink(os.path.join(pd, str(rid))) if pd else False
    lost = _unlink(os.path.join(_sub(LOST), f"{rid}.json"))
    if gone or pgone:
        r = rec or prec
        held = (time.time() - float(r.get("t", time.time()))) if r else -1.0
        logger.info("#243 HANDOFF-PENDING %s rid=%s at=%s held_s=%.1f lost=%s roles=%s%s", verb, rid, where, held,
                    lost, "+".join(n for n, g in ((ROLE_HANDOFF, gone), (ROLE_PARK, pgone)) if g),
                    "".join(f" {k}={v}" for k, v in kv.items()))
    return gone or pgone


def _expire(rid: str, age: float, where: str = PENDING) -> None:
    """A mark nobody ended within SGLANG_WEG2_HANDOFF_PENDING_EXPIRE_S leaves
    the order -- and is reported LOST (first_lost_page=0, reason=expired),
    never none: a rid still waiting for its seat is then routed fresh via P
    by the front instead of priced on pages nobody protects any more."""
    rec = _read(os.path.join(_sub(where), rid)) or {}
    ldir = _sub(LOST)
    if ldir:
        _write_atomic(os.path.join(ldir, f"{rid}.json"), {
            "rid": rid, "state": "lost", "reason": "expired", "first_lost_page": 0,
            "pages": rec.get("pages"), "page_size": rec.get("page_size"), "t": time.time(),
            "role": ROLE_PARK if where == PARK else ROLE_HANDOFF,
            "group": _group(), "pid": os.getpid()})
    if _unlink(os.path.join(_sub(where), rid)):
        logger.warning("#243 HANDOFF-PENDING EXPIRED rid=%s role=%s age_s=%.0f -> status lost (reason=expired, "
                       "first_lost_page=0): no D take and no front drop within the bound", rid,
                       ROLE_PARK if where == PARK else ROLE_HANDOFF, age)


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
        if rec is None and _sub(PARK):  # #248: a D park is kept by the same order
            rec = _read(os.path.join(_sub(PARK), rid))
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
    key_lo), ``rid_ix``/``page`` aligned with them, ``rids`` the names and
    ``roles`` their roles (handoff | park), ``stems`` every rid's stems in
    rid order."""

    __slots__ = ("keys", "rid_ix", "page", "rids", "stems", "roles", "_order", "_sorted")

    def __init__(self, keys, rid_ix, page, rids, stems, roles=None, order=None):
        self.keys, self.rid_ix, self.page, self.rids, self.stems = keys, rid_ix, page, rids, stems
        self.roles = list(roles) if roles is not None else [ROLE_HANDOFF] * len(rids)
        self._order, self._sorted = order, None

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def sorted_stems(self) -> list:
        """The stems aligned with ``keys`` (#248: the L3-copy test of stage
        ii asks the disk index by stem). Built on first use."""
        if self._sorted is None:
            order = self._order
            self._sorted = (list(self.stems) if order is None
                            else [self.stems[int(i)] for i in order])
        return self._sorted


def _empty_keep() -> Keep:
    import numpy as np

    z = np.zeros(0, dtype=np.uint64)
    return Keep(z, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), [], [])


def _chain_of(rid: str, where: str) -> list:
    if where == PARK:
        return list((_read(os.path.join(_sub(PARK), rid)) or {}).get("page_keys") or ())
    from sglang.srt.weg2 import handoff as _ho

    return list((_ho.read(rid) or {}).get("page_keys") or ())


def _rid_keys(pool, rid: str, where: str = PENDING):
    """(key_lo uint64[], page idx int64[], stems, demote order) this pool
    keeps for rid. The demote order is the stems in the order the L3 copy
    runs: a partial park's ``[a, b)`` from the back first, then the rest."""
    import numpy as np

    from sglang.srt.mem_cache.storage.file.hicache_arena import stem_keys_lo

    chain = _chain_of(rid, where)
    if not chain:
        return None
    role = getattr(pool, ROLE_ATTR, "kv")
    first = max(0, len(chain) - ANCHOR_TAIL_KEYS) if role == "anchor" else 0
    stems = pool._stems(chain[first:])
    lo = stem_keys_lo(stems)
    rng = _range_of(rid, where)
    if rng is not None and role != "anchor":
        a, b = rng
        order = list(reversed(stems[a:b])) + stems[:a]
    else:
        order = list(stems)
    return lo, np.arange(first, len(chain), dtype=np.int64), stems, order


def _range_of(rid: str, where: str):
    if where != PARK:
        return None
    rec = _read(os.path.join(_sub(PARK), rid)) or {}
    r = rec.get("range")
    return _norm_range(r, len(rec.get("page_keys") or ())) if r else None


def _marks(where: str, now: float, ttl: float) -> list:
    """[(rid, mtime_ns, age_s)] of one mark directory; an expired mark is
    named and leaves (``_expire``)."""
    d = _sub(where)
    try:
        entries = [e for e in os.scandir(d) if not e.name.endswith(".tmp")] if d else []
    except OSError:
        entries = []
    out = []
    for e in entries:
        try:
            st = e.stat()
        except OSError:
            continue
        age = now - st.st_mtime
        if ttl > 0 and age > ttl:
            _expire(e.name, age, where)
            continue
        out.append((e.name, st.st_mtime_ns, age))
    return out


def _stamp():
    out = []
    for where in (PENDING, PARK):
        d = _sub(where)
        try:
            out.append(os.stat(d).st_mtime_ns if d else None)
        except OSError:
            out.append(None)
    return tuple(out)


def keep_for(pool) -> Keep:
    """The kept keys for ``pool`` now: the P hand-offs waiting for a D seat
    (``pending``, chain from the hand-off file) and -- #248 -- the D requests
    that do not run (``park``, chain inside the mark). Cached on the pool;
    re-read only when a mark directory changed (its mtime), and then only
    the new rids. Parks first (they already ran), each role oldest first."""
    import numpy as np

    if not _sub(PENDING):
        return _empty_keep()
    stamp = _stamp()
    if stamp == (None, None):
        return _empty_keep()
    cache = pool.__dict__.get("_weg2_hp_keep")
    if cache is not None and cache[0] == stamp:
        return cache[1]
    per = pool.__dict__.setdefault("_weg2_hp_rid_keys", {})
    now, ttl = time.time(), _expire_s()
    marks = []
    for where, role in ((PARK, ROLE_PARK), (PENDING, ROLE_HANDOFF)):
        for rid, mtime, age in sorted(_marks(where, now, ttl), key=lambda m: -m[2]):
            marks.append((where, role, rid, mtime))
    live = {(w, r) for w, _, r, _ in marks}
    for gone in [k for k in per if k not in live]:
        per.pop(gone, None)
    for where, role, rid, mtime in marks:
        # a renewed mark (P re-prefilled the rid, e.g. RESUME-VIA-P with the
        # context grown; D parked it again further on) re-reads the chain;
        # an unchanged one is not re-hashed
        k = (where, rid)
        if k not in per or per[k][0] != mtime:
            try:
                per[k] = (mtime, _rid_keys(pool, rid, where))
            except Exception:  # noqa: BLE001 - a rid we cannot read keeps nothing
                logger.warning("#243 HANDOFF-PENDING keys of rid=%s (%s) unreadable", rid, role, exc_info=True)
                per[k] = (mtime, None)
    kept = [(where, role, rid) for where, role, rid, _ in marks if per.get((where, rid), (0, None))[1] is not None]
    if not kept:
        keep = _empty_keep()
    else:
        parts = [per[(w, r)][1] for w, _, r in kept]
        keys = np.concatenate([p[0] for p in parts])
        rid_ix = np.concatenate([np.full(p[0].shape[0], i, dtype=np.int64) for i, p in enumerate(parts)])
        page = np.concatenate([p[1] for p in parts])
        order = np.argsort(keys, kind="stable")
        keep = Keep(keys[order], rid_ix[order], page[order], [r for _, _, r in kept],
                    [s for p in parts for s in p[2]], roles=[role for _, role, _ in kept], order=order)
    pool.__dict__["_weg2_hp_keep"] = (stamp, keep)
    return keep


def rid_spans(pool) -> list:
    """#248 (the demoter): ``[(role, rid, stems)]`` this pool keeps, in keep
    order (parks first, oldest first)."""
    keep = keep_for(pool)
    per = pool.__dict__.get("_weg2_hp_rid_keys") or {}
    out = []
    for rid, role in zip(keep.rids, keep.roles):
        rec = per.get((PARK if role == ROLE_PARK else PENDING, rid), (0, None))[1]
        if rec is not None:
            out.append((role, rid, list(rec[3] if len(rec) > 3 else rec[2])))
    return out


def copied_mask(pool, keep: Keep):
    """#248 stage ii: which kept keys have an L3 (disk) copy -- bool array
    aligned with ``keep.keys``. The disk index answers (HiCacheFile's
    ``_stat_stems``: the shared L3 stem index first); no index, no copy."""
    import numpy as np

    n = len(keep)
    if not n:
        return np.zeros(0, dtype=bool)
    stat = getattr(getattr(pool, "_backend", None), "_stat_stems", None)
    if stat is None:
        return np.zeros(n, dtype=bool)
    stems = keep.sorted_stems()
    try:
        on_disk = stat(list(dict.fromkeys(stems)))
    except Exception:  # noqa: BLE001 - no answer = no copy (stage iii names the rest)
        return np.zeros(n, dtype=bool)
    return np.fromiter((s in on_disk for s in stems), dtype=bool, count=n)


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
        krole = keep.roles[r] if r < len(keep.roles) else ROLE_HANDOFF
        pend = _read(os.path.join(_sub(PARK if krole == ROLE_PARK else PENDING), rid)) or {}
        first = min(pages)
        path = os.path.join(ldir, f"{rid}.json") if ldir else ""
        prev = _read(path) if path else None
        if prev is not None and prev.get("first_lost_page") is not None:
            first = min(first, int(prev["first_lost_page"]))
        held = time.time() - float(pend.get("t", time.time()))
        if path:
            _write_atomic(path, {"rid": rid, "state": "lost", "reason": "evicted", "first_lost_page": int(first),
                                 "pages": pend.get("pages"), "page_size": pend.get("page_size"),
                                 "pool": role, "role": krole, "t": time.time(), "group": _group(),
                                 "pid": os.getpid()})
        # one line per rid and pool the first time, then at every 256 pages
        # more (a full arena takes a waiting chain page by page, claim by claim)
        seen = _LOST_SEEN.get((rid, role), 0)
        _LOST_SEEN[(rid, role)] = seen + len(pages)
        if seen == 0 or (seen // 256) != ((seen + len(pages)) // 256):
            if krole == ROLE_PARK:
                logger.warning(
                    "#248 PARK-LOST rid=%s pool=%s pages_lost=%d (cumulative %d) first_lost_page=%d of %s "
                    "held_s=%.1f need=%d (a claim found no other slot: the parked span had no L3 copy yet "
                    "and was evicted LAST; its wake read stops at the first lost page)",
                    rid, role, len(pages), seen + len(pages), first, pend.get("pages"), held, int(need))
            else:
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
        # #248: per role -- handoff_kept (P hand-offs) and park_kept (D
        # requests that do not run), each "complete in the arena / kept"
        counts = {ROLE_HANDOFF: [0, 0, 0], ROLE_PARK: [0, 0, 0]}
        per = pool.__dict__.get("_weg2_hp_rid_keys") or {}
        for rid, role in zip(keep.rids, keep.roles):
            rec = per.get((PARK if role == ROLE_PARK else PENDING, rid), (0, None))[1]
            if rec is None:
                continue
            c = counts[role]
            c[1] += len(rec[2])
            c[2] += 1
            if arena is not None and rec[2]:
                c[0] += sum(1 for _, st in arena.find_slots(rec[2]) if int(st) == 2)
        pinned = complete = "-"
        if arena is not None and hasattr(arena, "ref_census"):
            pinned, _refs, complete = arena.ref_census()
        h, k = counts[ROLE_HANDOFF], counts[ROLE_PARK]
        return (f"handoff_kept={h[0]}/{h[1]} handoff_rids={h[2]} park_kept={k[0]}/{k[1]} park_rids={k[2]} "
                f"arena_pinned={pinned} arena_complete={complete}")
    except Exception as exc:  # noqa: BLE001 - an instrument never breaks a census
        return f"handoff_kept=failed:{type(exc).__name__}"


# -- PA's partial park (27B, TEILPARKEN_notiz_0927.md §5) --------------------
#: the arena host pools of this process (``register_pool``), weak -- keep_state
#: reads the arena through them; a Form A worker has none
_POOLS: list = []


def register_pool(pool) -> None:
    """The tree registers its arena host pools once (D; idempotent)."""
    import weakref

    try:
        if not any(r() is pool for r in _POOLS):
            _POOLS.append(weakref.ref(pool))
    except Exception:  # noqa: BLE001
        pass


def _kv_pool():
    for r in list(_POOLS):
        p = r()
        if p is not None and getattr(p, ROLE_ATTR, "kv") != "anchor" and getattr(p, "arena", None) is not None:
            return p
    return None


def _park_chain(rid: str) -> list:
    rec = _read(os.path.join(_sub(PARK), rid)) if _sub(PARK) else None
    chain = list((rec or {}).get("page_keys") or ())
    if chain:
        return chain
    from sglang.srt.weg2 import handoff as _ho

    return list((_ho.read(rid) or {}).get("page_keys") or ())


def keep_role(rid, role, page_range=None, *, page_keys=None, page_size: int = 0) -> int:
    """PA's partial park seam. Never raises; any failure is 0 (= today's
    retain, the LRU spill).

    * ``role="park"``: keep rid's span by ORDER (no reference). ``page_keys``
      is the chain (default: the rid's park mark, else its P hand-off).
      ``page_range=(a, b)``: the span is ``[0, b)`` and ``[a, b)`` may go
      FIRST -- the demoter copies it to L3 from the back before anything
      else, so a claim frees exactly those pages first (stage ii) while
      ``[0, a)`` stays. ``a == b`` (shortfall 0) is a pure pause: the order
      only.
    * ``role=None``: the rid's marks go (idempotent; D also ends them itself
      at the rid's end -- ``consume`` -- so this is only the front's net).

    Returns the number of pages of ``[a, b)`` put FIRST in the demotion order
    (0 for ``role=None`` or an empty window). It is the same on every rank --
    the inputs are replicated and the call is bookkeeping only; what is on
    L2/L3 right now answers :func:`keep_state` (the arena lives on the
    attention rank 0)."""
    try:
        rid = str(rid)
        if role is None:
            _end(rid, "keep_role", "DROPPED")
            return 0
        if role != ROLE_PARK:
            return 0
        chain = [str(k) for k in (page_keys if page_keys is not None else _park_chain(rid))]
        if not chain:
            return 0
        rng = (_norm_range(page_range, len(chain)) if page_range is not None
               else (len(chain), len(chain)))
        if rng is None:
            return 0
        # every rank writes the same bytes (atomic replace); group D only
        if not mark_park(rid, chain, page_size, page_range=rng):
            return 0
        n = int(rng[1] - rng[0])
        logger.info("#248 KEEP-ROLE rid=%s role=park span=%d first=[%d,%d) pages_first=%d",
                    rid, rng[1], rng[0], rng[1], n)
        return n
    except Exception:  # noqa: BLE001
        return 0


def keep_state(rid, tree=None) -> dict:
    """``{device, l2, l3, pages}`` page counts of rid's kept span (its park
    mark, else its P hand-off): ``l2`` COMPLETE in the arena, ``l3`` with a
    disk copy, ``device`` the pages whose tree node still has its device
    value (None without a ``tree``). Needs the arena (the attention rank 0);
    elsewhere ``l2``/``l3`` are None. Never raises."""
    out = {"device": None, "l2": None, "l3": None, "pages": None}
    try:
        rid = str(rid)
        chain = _park_chain(rid)
        out["pages"] = len(chain)
        pool = _kv_pool()
        if pool is not None and chain:
            stems = pool._stems(chain)
            out["l2"] = sum(1 for _, st in pool.arena.find_slots(stems) if int(st) == 2)
            stat = getattr(getattr(pool, "_backend", None), "_stat_stems", None)
            out["l3"] = len(stat(stems)) if stat is not None else 0
        if tree is not None and chain:
            out["device"] = _device_pages(tree, chain)
    except Exception:  # noqa: BLE001
        return {"device": None, "l2": None, "l3": None, "pages": None}
    return out


def _device_pages(tree, chain) -> int:
    """Pages of ``chain`` whose tree node still has its device value (the
    nodes' ``hash_value`` pages are the chain's own keys)."""
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    want = {str(k) for k in chain}
    n = 0
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        hv = getattr(node, "hash_value", None) or ()
        hit = sum(1 for h in hv if str(h) in want)
        if hit:
            try:
                on_dev = node.component_data[BASE_COMPONENT_TYPE].value is not None
            except Exception:  # noqa: BLE001
                on_dev = False
            if on_dev:
                n += hit
        stack.extend(getattr(node, "children", {}).values())
    return n
