"""fnFL2 H111b: the sleeper's pair lanes run one tag ahead of the pause.

THE MEASUREMENT (x177/x178/h91v1, all 15 P->D flips, H111 legsplit.py /
lockstep_gain.py). The legs end with PP0's deposit chain (1.55-1.76 s). Its
floor is PHYSICS, and not the per-lane copy the lane lines suggest: both
BAR1 deposit lanes of the 5090 (p0 -> TP1 x4, p1 -> TP2 x8) share the card's
ONE D2H copy engine (H22 probe: together 9.34 GB/s), so 11.69 GB of expert
rows cost >= 1.25 s. What sits ABOVE that floor on the chain, per flip:

* tag lockstep: a tag ends when its SLOWEST lane ends, the other lane idles
  until then (sum over tags of max(p0, p1) minus max(sum p0, sum p1) =
  42-156 ms, median ~65; x177 97k: w1 p0 181 ms / p1 63 ms);
* the pause, the credit publish and the loop step of every tag, in series
  between two tags' copies (~5-7 ms x 10 tags, ~55 ms);
* the per-call lane setup (plan filter, lane map, rendezvous) of the next tag.

THE FORM (switch SGLANG_WEG2_DEPOSIT_LANE_LOOKAHEAD, default off = the
per-tag lockstep byte for byte). The sleep loop keeps its law per tag --
DEPOSIT, then PAUSE, then CREDIT (xchg-lane-ordnung) -- and only the pair
lanes (cross-card, BAR1 or host) leave the lockstep: each runs in its own
worker over the tag order and may start tag j once the loop has paused tag
j-1-AHEAD (AHEAD=1: at most one tag beyond the one being paused). The
on-card (diagonal) lane stays on the loop thread, in the old place, AFTER the
previous tag's pause: its IPC staging is a cudaMalloc on the sleeper's own
card, and that card's free memory is the waker's credit -- moving it ahead of
a pause would be a reserve, which the law forbids. The pair lanes allocate
nothing on the sleeper's card (they write through the peer's BAR1 window or
the host lane), so running them early changes WHEN bytes cross, never HOW
MANY BYTES ARE HELD where. No new VRAM, no host bytes, no leftover.

NO NEW WAIT ON THE CYCLE PATH. A lane that runs ahead can only block on its
own collector (the peer's credit / drained handshake), and the peer's
collector for that tag depends on the peer card's pauses (another P stage),
never on this rank's next pause. The loop waits for every lane of tag j
before pausing tag j, exactly as the lockstep did; a lane's refusal is
re-raised there, by name, on the loop thread (the group fence sees the same
exception type as before).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LINE = "WEG2-H111B"


def lookahead_on(group: Optional[str] = None) -> bool:
    """Armed for this sleeping ``group`` (None = the switch alone)."""
    from sglang.srt.environ import envs

    if not bool(envs.SGLANG_WEG2_DEPOSIT_LANE_LOOKAHEAD.get()):
        return False
    if group is None:
        return True
    groups = {g.strip() for g in str(envs.SGLANG_WEG2_DEPOSIT_LANE_LOOKAHEAD_GROUPS.get() or "").split(",")}
    return str(group) in groups


class LaneLookaheadError(RuntimeError):
    """A lane worker did not finish a tag inside the budget (named, never a hang)."""


class LaneLookahead:
    """One worker per pair lane over ``tags``; the loop joins tag by tag.

    ``deposit(tag, lane)`` is the one call a worker makes per tag (the
    sleeper's own deposit, restricted to that lane). ``ahead`` bounds how far
    a lane may run beyond the tag the loop is about to pause.
    """

    def __init__(
        self,
        tags: Sequence[str],
        lanes: Sequence[object],
        deposit: Callable[[str, object], None],
        *,
        ahead: int = 1,
        budget_s: float = 600.0,
        log: Optional[Callable[..., None]] = None,
        thread_init: Optional[Callable[[], None]] = None,
        clock: Callable[[], float] = time.perf_counter,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.tags = [str(t) for t in tags]
        self.lanes = list(lanes)
        self._deposit = deposit
        self.ahead = max(0, int(ahead))
        self.budget_s = float(budget_s)
        self._log = log or logger.info
        self._thread_init = thread_init
        self._clock = clock
        self._wall = wall
        self._cv = threading.Condition()
        self._allowed = -1
        self._abort = False
        # (lane, tag index) -> (exception or None, ms, wall t0, wall t1)
        self._done: Dict[Tuple[object, int], Tuple[Optional[BaseException], float, float, float]] = {}
        self._threads: List[threading.Thread] = []
        self._t0 = None

    # -- loop side -----------------------------------------------------------
    def start(self) -> None:
        with self._cv:
            self._allowed = min(self.ahead, len(self.tags) - 1)
        self._t0 = self._clock()
        for lane in self.lanes:
            th = threading.Thread(
                target=self._run, args=(lane,), name=f"weg2-h111b-p{lane}", daemon=True
            )
            self._threads.append(th)
            th.start()

    def join_tag(self, index: int) -> float:
        """Wait until every lane has finished tag ``index``; re-raise the first
        lane refusal of this or an earlier tag. Returns the ms waited."""
        t0 = self._clock()
        deadline = t0 + self.budget_s
        with self._cv:
            while True:
                failed = self._first_failure(upto=index)
                if failed is not None:
                    lane, j, exc = failed
                    raise exc
                if all((lane, index) in self._done for lane in self.lanes):
                    break
                left = deadline - self._clock()
                if left <= 0:
                    missing = [f"p{lane}" for lane in self.lanes if (lane, index) not in self._done]
                    raise LaneLookaheadError(
                        f"H111b lane lookahead: lane(s) {','.join(missing)} did not "
                        f"finish tag {self.tags[index]} within {self.budget_s:.0f} s "
                        f"(each lane's own rendezvous budget; the loop refuses by "
                        f"name instead of pausing a tag whose bytes are still in flight)"
                    )
                self._cv.wait(timeout=min(left, 1.0))
        return (self._clock() - t0) * 1000.0

    def advance(self, paused_index: int) -> None:
        """The loop paused tag ``paused_index``: lanes may now start up to
        ``paused_index + 1 + ahead``."""
        with self._cv:
            self._allowed = max(
                self._allowed, min(paused_index + 1 + self.ahead, len(self.tags) - 1)
            )
            self._cv.notify_all()

    def close(self, join_s: float = 5.0) -> None:
        with self._cv:
            self._abort = True
            self._cv.notify_all()
        for th in self._threads:
            th.join(timeout=join_s)

    def chain_ms(self) -> float:
        return 0.0 if self._t0 is None else (self._clock() - self._t0) * 1000.0

    # -- worker side ---------------------------------------------------------
    def _first_failure(self, upto: int):
        for (lane, j), rec in sorted(self._done.items(), key=lambda kv: kv[0][1]):
            if j <= upto and rec[0] is not None:
                return lane, j, rec[0]
        return None

    def _run(self, lane) -> None:
        if self._thread_init is not None:
            try:
                self._thread_init()
            except BaseException as exc:  # noqa: BLE001 -- reported at the join
                with self._cv:
                    self._done[(lane, 0)] = (exc, 0.0, self._wall(), self._wall())
                    self._cv.notify_all()
                return
        for j, tag in enumerate(self.tags):
            with self._cv:
                while self._allowed < j and not self._abort:
                    self._cv.wait(timeout=1.0)
                if self._abort:
                    return
            w0, c0 = self._wall(), self._clock()
            exc: Optional[BaseException] = None
            try:
                self._deposit(tag, lane)
            except BaseException as e:  # noqa: BLE001 -- carried to the loop thread
                exc = e
            ms = (self._clock() - c0) * 1000.0
            with self._cv:
                self._done[(lane, j)] = (exc, ms, w0, self._wall())
                self._cv.notify_all()
            if exc is not None:
                return
            self._log(
                "%s lane=p%s tag=%s ms=%.0f ahead=%d t0=%.3f t=%.3f",
                LINE, lane, tag, ms, self.ahead, w0, w0 + ms / 1000.0,
            )
