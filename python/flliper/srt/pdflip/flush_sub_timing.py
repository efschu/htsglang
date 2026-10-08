"""PDFLIP-L (02.10.2026): where a sleeping group's flush and kv release spend
their time -- one line per sleep and rank.

THE QUESTION (27B N3u 1002_072908 against NF dauer10020814). The P->D layer is
~0.45 s longer on 27B than on NF, and the extra sits on group P before the
weight legs (FLIP-TIMELINE p50: quiesce 302 ms against ~120, sleep-kv 339 ms
against ~70):

* ``#1476 DISPATCH kind=FlushCacheReqInput`` on PP0 p50 164 ms (NF: every one
  under the 20 ms print floor);
* PP1 ``PP-CHAIN-RECV #1460 CHAIN-RECV blocked 152 ms`` (the z30j PP0 verdict:
  a follower flushes one PP0 pass later), then its own flush ~160 ms;
* PP1 ``PDFLIP-SLEEP-CHUNK ... paused in 148 ms`` (NF p50 20 ms) -- that clock
  starts at the release RPC, so it holds the release's own flush too.

The existing lines carry totals only. This module stamps the segments of
``Scheduler.flush_cache`` (sweep, verdict, store join, L15 as ONE segment,
lost anchors, tree reset, pool clears, zeroing, grammar/metrics, draft pool,
empty_cache, scrub) and of the release's kv block (flush, kv pause, graph
pause, sync), keeps the last ``CHAIN-RECV blocked`` wait, and prints

  PDFLIP-SLEEP-SUB group=P pp=1 tp=0 rpc_flush[...] release_flush[...] kv_release[...] chain_blocked_ms=...

once at the end of the kv release (the sleep's SLEEP-CHUNK point). Instrument
only: no branch reads anything recorded here. Process-global (one scheduler
per rank process); every entry point swallows its own errors.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: a chain wait older than this when the flush starts belongs to another pass
CHAIN_FRESH_S = 5.0
#: refused flushes (quiesce polls) cheaper than this are not kept
REFUSED_KEEP_MS = 20.0

_LOCK = threading.Lock()
_STATE: Dict[str, object] = {"rpc": None, "release": None, "chain": None, "refused": []}


class FlushClock:
    """Ordered marks of one flush; ``segments()`` = ms between adjacent marks."""

    def __init__(self, site: str, now=time.perf_counter):
        self.site = site
        self._now = now
        self.t0 = now()
        self.marks: List[Tuple[str, float]] = []
        self.path = "?"

    def mark(self, name: str) -> None:
        try:
            self.marks.append((name, self._now()))
        except Exception:  # noqa: BLE001 -- an instrument never breaks the flush
            pass

    def segments(self) -> List[Tuple[str, float]]:
        out, prev = [], self.t0
        for name, t in self.marks:
            out.append((name, (t - prev) * 1000.0))
            prev = t
        return out

    def total_ms(self) -> float:
        end = self.marks[-1][1] if self.marks else self.t0
        return (end - self.t0) * 1000.0

    def text(self) -> str:
        segs = " ".join("%s=%.0f" % (n, ms) for n, ms in self.segments())
        return "%s total=%.0f %s" % (self.path, self.total_ms(), segs)


def begin_flush(zero_kv) -> FlushClock:
    """``zero_kv is False`` is the release's flush (weight_updater passes it);
    anything else is the /flush_cache RPC (the front's quiesce poll)."""
    return FlushClock("release" if zero_kv is False else "rpc")


def end_flush(clock: Optional[FlushClock], path: str) -> None:
    if clock is None:
        return
    try:
        clock.path = path
        with _LOCK:
            if path == "refused":
                if clock.total_ms() >= REFUSED_KEEP_MS:
                    _STATE["refused"].append(clock.total_ms())  # type: ignore[union-attr]
                return
            _STATE[clock.site] = clock
            if clock.site == "rpc":
                ch = _STATE.get("chain")
                if ch is not None and clock.t0 - ch[1] <= CHAIN_FRESH_S:  # type: ignore[index]
                    _STATE["chain_at_flush"] = ch[0]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        pass


def note_chain_blocked(ms: float, now=time.perf_counter) -> None:
    """``#1460 CHAIN-RECV blocked``: the follower's wait for PP0's pass."""
    try:
        with _LOCK:
            _STATE["chain"] = (float(ms), now())
    except Exception:  # noqa: BLE001
        pass


def sleep_line(group: str, pp_rank: int, tp_rank: int, kv: Optional[FlushClock]) -> str:
    """The one line of this sleep; resets the state for the next one."""
    with _LOCK:
        rpc = _STATE.get("rpc")
        rel = _STATE.get("release")
        chain = _STATE.get("chain_at_flush")
        refused = list(_STATE.get("refused") or [])  # type: ignore[arg-type]
        _STATE.update({"rpc": None, "release": None, "chain_at_flush": None, "refused": []})
    return (
        "PDFLIP-SLEEP-SUB group=%s pp=%d tp=%d rpc_flush[%s] release_flush[%s] kv_release[%s] "
        "chain_blocked_ms=%s refused_polls=%d refused_ms=%.0f (ms per segment, in order; "
        "rpc = the front's /flush_cache that passed, release = the flush inside the kv "
        "release; l15 = every L15 block of the flush as one segment; chain_blocked = this "
        "rank's last #1460 CHAIN-RECV wait before that flush)" % (
            group or "-", int(pp_rank), int(tp_rank),
            rpc.text() if isinstance(rpc, FlushClock) else "none",
            rel.text() if isinstance(rel, FlushClock) else "none",
            kv.text() if isinstance(kv, FlushClock) else "none",
            "%.0f" % chain if chain is not None else "none",
            len(refused), sum(refused),
        )
    )


def emit_sleep_line(group: str, pp_rank: int, tp_rank: int, kv: Optional[FlushClock]) -> None:
    try:
        logger.info("%s", sleep_line(group, pp_rank, tp_rank, kv))
    except Exception:  # noqa: BLE001
        pass


def _reset_for_tests() -> None:
    with _LOCK:
        _STATE.clear()
        _STATE.update({"rpc": None, "release": None, "chain": None, "refused": []})
