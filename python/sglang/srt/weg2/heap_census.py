"""The sleep's host-heap census, off the flip's critical path (fnFL2 H16).

``WEG2-SLEEP-HOST-HEAP`` (weg2xsn297) trims the glibc heap of a sleeping rank
(``malloc_trim(0)``) and, on sleeps 1, 2, 4, 8, ..., walks every gc-tracked
object to name what the heap is made of. The walk runs INSIDE the sleep RPC,
and the front starts the wake only when that RPC answers -- so it sits on the
flip's critical path. Measured on fnFL2x127 (a1dab24409), last deposit tag
(``WEG2-SLEEP-TAG-TIME tag=weights t=``) to ``WEG2-GROUP-FENCE release t=``:

  group  sleep  census  tail
  P PP0    2     yes    538 ms   (57.911 -> 58.449, the 97k P->D flip)
  P PP0    3     no      18 ms
  P PP0    4     yes    505 ms
  P PP0    5     no      23 ms
  D TP0    1     yes    616 ms   (the D->P flips: the same walk on D's sleep)
  D TP0    2     yes    639 ms
  D TP0    3     no      57 ms
  D TP0    4     yes    558 ms

The trim stays where it is (it is part of the 18-57 ms tails). The census
keeps its line and its cadence, but by default it runs on a daemon timer
AFTER the sleep has answered -- the process is dormant then -- and its line
carries its own cost (``census_ms``) so the next boot proves it off the path.
"""

from __future__ import annotations

import gc
import sys
import threading
import time
from typing import Callable, Dict, Iterable, Tuple

from sglang.srt.environ import Weg2HeapCensus

#: Seconds between the sleep's trim and the deferred census. The P->D legs of
#: fnFL2x127 end 18-57 ms before the sleep RPC answers; 2 s puts the walk
#: behind the answer, the wake RPC and the waker's first extend.
DEFER_S = 2.0
TOP_N = 8


def census_due(*, sleep_count: int) -> bool:
    """Sleeps 1, 2, 4, 8, ... -- the cadence of weg2xsn297, unchanged."""
    n = int(sleep_count)
    return n >= 1 and (n <= 2 or (n & (n - 1)) == 0)


def census_action(*, mode: int, sleep_count: int) -> str:
    """'off' | 'inline' | 'defer' for this sleep."""
    if not census_due(sleep_count=sleep_count) or int(mode) == Weg2HeapCensus.OFF:
        return "off"
    return "inline" if int(mode) == Weg2HeapCensus.INLINE else "defer"


def census_text(objects: Iterable[object], *, top: int = TOP_N) -> str:
    """``type:countxMiB`` for the ``top`` types by shallow size (sys.getsizeof)."""
    counts: Dict[str, int] = {}
    sizes: Dict[str, int] = {}
    for o in objects:
        k = type(o).__name__
        counts[k] = counts.get(k, 0) + 1
        try:
            sizes[k] = sizes.get(k, 0) + sys.getsizeof(o)
        except Exception:  # noqa: BLE001 -- an object without a size is skipped
            pass
    ranked = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[: int(top)]
    return ",".join(f"{k}:{counts[k]}x{v >> 20}MiB" for k, v in ranked)


def timed_census(
    *, objects: Callable[[], Iterable[object]] = gc.get_objects
) -> Tuple[str, float]:
    """The census and what it cost (ms, wall)."""
    t0 = time.perf_counter()
    text = census_text(objects())
    return text, (time.perf_counter() - t0) * 1000.0


def deferred_line(
    *, sleep_count: int, text: str, census_ms: float, delay_s: float
) -> str:
    return (
        f"WEG2-SLEEP-HOST-HEAP-CENSUS sleep={int(sleep_count)} deferred_s={delay_s:.1f} "
        f"census_ms={census_ms:.0f} census={text} (H16: the gc walk runs after the sleep "
        f"answered, off the flip's critical path; census = sys.getsizeof over gc-tracked "
        f"objects by type, shallow sizes, top {TOP_N})"
    )


def defer_census(
    *,
    sleep_count: int,
    log: Callable[[str], None],
    delay_s: float = DEFER_S,
    timer_factory: Callable[..., threading.Timer] = threading.Timer,
    objects: Callable[[], Iterable[object]] = gc.get_objects,
) -> threading.Timer:
    """Start a daemon timer that runs the census and logs it. Never raises
    into the sleep: a failing walk logs ``census=n/a``."""

    def _run() -> None:
        try:
            text, ms = timed_census(objects=objects)
        except Exception as exc:  # noqa: BLE001 -- an instrument never kills a rank
            text, ms = f"n/a({type(exc).__name__})", 0.0
        log(
            deferred_line(
                sleep_count=sleep_count, text=text, census_ms=ms, delay_s=delay_s
            )
        )

    timer = timer_factory(float(delay_s), _run)
    timer.daemon = True
    timer.start()
    return timer
