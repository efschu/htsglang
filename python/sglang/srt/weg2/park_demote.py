"""#248 PARK-DEMOTE: the kept spans get an L3 copy in the background.

rc12s 17:32:40 (tmp/r989/befund_248_park_l2_pinnt.md): D held 5213 of the
5461 KV arena slots by reference while it slept -- the sleep prefetch of two
parked and three held requests -- and P's claims found no free slot. #248
takes those references away (the hold read runs at the wake), so the spans
are kept by ORDER only (``handoff_pending``: role ``park`` for D's parked and
held rids, role ``handoff`` for P's hand-offs waiting for a seat). An order
alone still loses the page when a claim has nothing else to take.

The L3 copy closes that: this thread copies every kept page from the arena to
the HiCacheFile disk store WITHOUT freeing it (``HiCacheFile.
arena_copy_to_disk``). A claim then frees a kept page that has a copy without
any I/O (stage ii of ``_evict_for_claim``), and the read at the wake takes it
back (``arena_fill_from_disk``, already on the ARENA-GET path). Only a kept
page WITHOUT a copy is lost (stage iii, PARK-LOST / HANDOFF-LOST).

Where it runs: ONE thread, in group D's attention rank 0 (the process with a
real store backend; Form A workers have the null backend), never a scheduler
thread (H81: no copy in the compute path). Parks first, oldest first; a rid
whose chain is on disk is not asked again until its mark changes.

Line: ``#248 PARK-DEMOTE rid= role= pool= pages= written= l3_pages= absent=
bytes= ms=`` once per rid (and when a later tick completes it).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

THREAD_NAME = "weg2-park-demote"


def _env():
    from sglang.srt.environ import envs

    return envs


def enabled() -> bool:
    try:
        e = _env()
        return bool(e.SGLANG_WEG2_ENABLE_PARK_L3.get()) and float(e.SGLANG_WEG2_PARK_DEMOTE_S.get()) > 0
    except Exception:  # noqa: BLE001
        return False


def _batch() -> int:
    try:
        return max(1, int(_env().SGLANG_WEG2_PARK_DEMOTE_BATCH.get()))
    except Exception:  # noqa: BLE001
        return 256


def demote_once(backend, pools: Iterable, *, state: Optional[dict] = None, batch: Optional[int] = None) -> list:
    """One pass over every kept span of ``pools``: copy what has no disk copy
    yet. ``state`` remembers the rids already fully on disk (keyed by pool,
    role, rid and the chain length). Returns one record per rid touched."""
    from sglang.srt.weg2 import handoff_pending as _hp

    copy = getattr(backend, "arena_copy_to_disk", None)
    if copy is None:
        return []
    state = {} if state is None else state
    batch = batch or _batch()
    done_now = []
    live = set()
    for pool in pools:
        arena = getattr(pool, "arena", None)
        if arena is None:
            continue
        prole = getattr(pool, _hp.ROLE_ATTR, "kv")
        for role, rid, stems in _hp.rid_spans(pool):
            key = (id(pool), role, rid, len(stems))
            live.add(key)
            if state.get(key):
                continue
            t0 = time.perf_counter()
            tot = {"written": 0, "on_disk": 0, "absent": 0, "bytes": 0}
            for i in range(0, len(stems), batch):
                r = copy(arena, stems[i:i + batch])
                for k in tot:
                    tot[k] += int(r.get(k, 0))
            l3 = tot["written"] + tot["on_disk"]
            full = l3 >= len(stems)
            if full:
                state[key] = True
            if tot["written"] or full:
                rec = dict(rid=rid, role=role, pool=prole, pages=len(stems), l3_pages=l3,
                           ms=(time.perf_counter() - t0) * 1e3, **tot)
                done_now.append(rec)
                if tot["written"]:
                    logger.info(
                        "#248 PARK-DEMOTE rid=%s role=%s pool=%s pages=%d written=%d l3_pages=%d absent=%d "
                        "bytes=%d ms=%.0f (kept span copied to L3 without a free: a claim may now take it "
                        "without I/O, the wake reads it back)", rid, role, prole, len(stems), tot["written"],
                        l3, tot["absent"], tot["bytes"], rec["ms"])
    for k in [k for k in state if k not in live]:
        state.pop(k, None)
    return done_now


class _Demoter:
    def __init__(self, backend, pools_fn, every: float):
        self.backend, self.pools_fn, self.every = backend, pools_fn, float(every)
        self.stop = threading.Event()
        self.state: dict = {}
        self.thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)

    def _run(self):
        n = 0
        while not self.stop.wait(self.every):
            try:
                demote_once(self.backend, list(self.pools_fn()), state=self.state)
            except Exception as exc:  # noqa: BLE001 - the demoter never takes the process down
                n += 1
                if n <= 8 or n % 256 == 0:
                    logger.warning("#248 PARK-DEMOTE pass failed (n=%d): %r", n, exc)


_DEMOTERS: dict = {}


def start(backend, pools_fn, *, key: str = "default") -> bool:
    """Start the one demoter of this process (idempotent per ``key``)."""
    if not enabled() or getattr(backend, "arena_copy_to_disk", None) is None:
        return False
    if key in _DEMOTERS:
        return False
    d = _Demoter(backend, pools_fn, float(_env().SGLANG_WEG2_PARK_DEMOTE_S.get()))
    _DEMOTERS[key] = d
    d.thread.start()
    logger.info("#248 PARK-DEMOTE thread started (every %.1f s, batch %d): kept spans get an L3 copy",
                d.every, _batch())
    return True
