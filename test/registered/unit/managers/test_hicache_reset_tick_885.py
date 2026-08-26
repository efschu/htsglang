"""#885: the seam paid a 1 Hz poll to stop threads that had nothing to do.

`hicache_quiesce -> resident_release` measured 461 ms MEDIAN over 1959 flip legs
while releasing ZERO residents, flat across [0, 1009 ms] on all three ranks in
both directions. The segment is `_release_residents_for_cutover`, whose only
blocking call in the empty case is `tree.reset()` ->
`UnifiedRadixCache._reset_full` -> `HiCacheController.reset()`, which set a stop
event and then joined threads that read that event only between
`queue.get(block=True, timeout=1)` calls.

Two properties are pinned here:

  1. THE STOP IS PROMPT. `reset()` wakes the threads instead of waiting out
     their poll. The can-fail is direct: delete the `_wake_storage_threads()`
     call and this test measures ~500 ms instead of ~0.
  2. THE RENDEZVOUS SURVIVES. The threads are still STOPPED before `reset()`
     returns -- `_reset_full` clears the host pool on the next line, and a live
     prefetch/backup thread there is the #760 use-after-release class. A fix
     that made the wait cheap by removing the wait would pass (1) and fail (2).
"""

import threading
import time
from queue import Queue

import pytest

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.managers.phase_flip_seam_census import (
    _TICK_MIN_MEDIAN_MS,
    SeamCensus,
    observe_segments_for_tick,
    reset_tick_history,
    tick_suspect,
)

# The poll the threads are parked in. If this ever changes in
# cache_controller.py the numbers below stop meaning what they say, so it is
# named once here rather than spelled 1.0 in four assertions.
POLL_TIMEOUT_S = 1.0


def _controller_with_storage_threads():
    """A controller carrying ONLY what the stop path touches.

    Deliberately not a mock of `reset`: the REAL `HiCacheController.reset`, the
    REAL `prefetch_thread_func` and the REAL `backup_thread_func` run here.
    Everything they reach past the empty `Queue.get` is unreachable in this
    test by construction, which is the same state the seam is in -- the queues
    are empty because there is nothing resident to prefetch or back up.
    """
    c = object.__new__(HiCacheController)
    c.storage_stop_event = threading.Event()
    c.write_queue = []
    c.load_queue = []
    c.ack_write_queue = []
    c.ack_load_queue = []
    c.enable_storage = True
    c.storage_backend = None
    c.backup_skip = True
    c.prefetch_queue = Queue()
    c.backup_queue = Queue()
    c.prefetch_revoke_queue = Queue()
    c.ack_backup_queue = Queue()
    c.host_mem_release_queue = Queue()
    c.prefetch_tokens_occupied = 0
    c.prefetch_thread = threading.Thread(target=c.prefetch_thread_func, daemon=True)
    c.backup_thread = threading.Thread(target=c.backup_thread_func, daemon=True)
    c.prefetch_thread.start()
    c.backup_thread.start()
    # Let both threads reach their first blocking get, so the stop below is
    # measured against a thread that is genuinely parked -- which is the state
    # the seam finds them in and the only state where the defect exists.
    time.sleep(0.25)
    return c


def _shutdown(c):
    c.enable_storage = False
    c.storage_stop_event.set()
    c._wake_storage_threads()
    for t in (
        getattr(c, "prefetch_thread", None),
        getattr(c, "backup_thread", None),
        getattr(c, "prefetch_io_aux_thread", None),
    ):
        if t is not None:
            t.join(timeout=5)


def test_reset_does_not_wait_out_the_one_second_poll():
    """The #885 measurement itself, at desk scale.

    Ten resets entered at phases spread across the poll period. Under the
    defect every one of them costs the REMAINDER of that period, so the worst
    of ten lands near 1000 ms and the mean near 500. The bound is deliberately
    generous (a fifth of the poll) so this pins the CLASS -- "does not wait for
    the tick" -- and not a machine's scheduling jitter.
    """
    c = _controller_with_storage_threads()
    try:
        costs = []
        for i in range(10):
            # Spread the entry phase across the period; the defect's cost is a
            # pure function of this phase, so a fixed gap would sample one
            # point of the distribution and could pass by luck.
            time.sleep(POLL_TIMEOUT_S * (i % 7) / 7.0 + 0.02)
            t0 = time.perf_counter()
            c.reset()
            costs.append(time.perf_counter() - t0)

        worst = max(costs)
        mean = sum(costs) / len(costs)
        assert worst < 0.2 * POLL_TIMEOUT_S, (
            f"reset() took {worst * 1000:.0f} ms at its worst over {len(costs)} "
            f"entries (mean {mean * 1000:.0f} ms). That is the #885 defect: the "
            f"stop event does not wake a thread parked in Queue.get, so the "
            f"joins wait out the remainder of the {POLL_TIMEOUT_S:.0f} s poll. "
            f"costs_ms={[round(x * 1000) for x in costs]}"
        )
    finally:
        _shutdown(c)


def test_reset_still_stops_the_threads_before_returning():
    """THE RENDEZVOUS IS THE POINT OF THE JOIN AND IT MUST NOT GET CHEAPER.

    `UnifiedRadixCache._reset_full` calls `mem_pool_host.clear()` immediately
    after this returns. A prefetch or backup thread still alive at that moment
    touches host pages as they are cleared, which is #760 exactly. So "fast"
    is only correct while "stopped" still holds.
    """
    c = _controller_with_storage_threads()
    try:
        old_prefetch = c.prefetch_thread
        old_backup = c.backup_thread
        old_aux = getattr(c, "prefetch_io_aux_thread", None)
        assert old_aux is not None, (
            "prefetch_thread_func should have started an aux thread"
        )

        c.reset()

        assert not old_prefetch.is_alive(), "prefetch thread outlived reset()"
        assert not old_backup.is_alive(), "backup thread outlived reset()"
        assert not old_aux.is_alive(), "the prefetch IO aux thread outlived reset()"
        # And the replacements are up, because reset() is a restart.
        assert c.prefetch_thread is not old_prefetch and c.prefetch_thread.is_alive()
        assert c.backup_thread is not old_backup and c.backup_thread.is_alive()
    finally:
        _shutdown(c)


def test_reset_joins_an_aux_thread_that_ignores_the_sentinel():
    """THE AUX JOIN, TESTED SO THAT REMOVING IT ACTUALLY FAILS.

    The assertion in the test above passes with the aux join deleted -- measured,
    not assumed. The sentinel wakes the aux thread too, so by the time the other
    two joins return it has already exited on its own and "not alive" holds for a
    reason that is not the join. That is a green light for the wrong reason, and
    it would have left the aux join untested while looking covered.

    So this drives the case the join exists FOR: an aux thread that does not
    observe the sentinel -- because it was started after the wake, or because
    `prefetch_thread_func` rebound `prefetch_buffer` and the sentinel went into a
    queue nobody reads any more. It stops only on the event, and `reset()` CLEARS
    that event on its way out. Without the join, reset returns while the thread
    is still parked, the clear lands before it looks, and it never stops at all:
    one leaked worker per flip, holding a host-pool reference across the
    `mem_pool_host.clear()` that follows.
    """
    c = _controller_with_storage_threads()
    stray_saw_stop = threading.Event()

    def _deaf_aux():
        # Polls the event directly and ignores the queue entirely, so no
        # sentinel can reach it. 0.3 s is long enough that an unjoined reset
        # would demonstrably return first.
        while not c.storage_stop_event.is_set():
            time.sleep(0.3)
        stray_saw_stop.set()

    try:
        real_aux = c.prefetch_io_aux_thread
        c.storage_stop_event.set()
        c._wake_storage_threads()
        real_aux.join(timeout=5)
        c.storage_stop_event.clear()

        stray = threading.Thread(target=_deaf_aux, daemon=True)
        stray.start()
        c.prefetch_io_aux_thread = stray
        time.sleep(0.05)

        c.reset()

        assert not stray.is_alive(), (
            "reset() returned while an aux thread that cannot hear the sentinel "
            "was still parked. reset() clears storage_stop_event immediately "
            "after, so that thread never stops -- one leaked worker per flip. "
            "The join is what closes this, and deleting it must fail here"
        )
        assert stray_saw_stop.is_set()
    finally:
        c.storage_stop_event.set()
        stray.join(timeout=5)
        _shutdown(c)


def test_reset_does_not_leak_threads_across_many_flips():
    """A flip happens hundreds of times per boot; the thread count may not track it."""
    c = _controller_with_storage_threads()
    try:
        c.reset()
        baseline = threading.active_count()
        for _ in range(12):
            c.reset()
        assert threading.active_count() <= baseline, (
            f"thread count grew from {baseline} to {threading.active_count()} "
            f"over 12 resets; a stop path that leaves a worker behind leaks one "
            f"per flip"
        )
    finally:
        _shutdown(c)


# --------------------------------------------------------------------------
# The CLASS check: the shape, not the site.
# --------------------------------------------------------------------------


def _uniform(n, period_ms):
    """Deterministic uniform samples: entering a 1/P cycle at even phases."""
    return [period_ms * (i + 0.5) / n for i in range(n)]


def _work_shaped(n, peak_ms):
    """A mode near the low end with a long tail -- what real work looks like."""
    out = []
    for i in range(n):
        # Deterministic pseudo-random in [0,1) with no seed dependence.
        u = ((i * 2654435761) % 10007) / 10007.0
        # Inverse-CDF of a heavy-tailed shape, scaled so the max lands near peak.
        out.append(peak_ms * (u**3))
    return out


@pytest.mark.parametrize("period", [250.0, 500.0, 1000.0])
def test_tick_suspect_fires_on_a_uniform_segment(period):
    reset_tick_history()
    verdict = tick_suspect(_uniform(64, period))
    assert verdict is not None, f"a flat [0,{period}] segment must be flagged"
    assert f"{period:.0f} ms" in verdict


def test_tick_suspect_is_silent_on_work():
    """The FALSE-POSITIVE half. A check that flags everything names nothing."""
    assert tick_suspect(_work_shaped(64, 1000.0)) is None


def test_tick_suspect_is_silent_on_a_timeout_being_hit():
    """Piling up AT the cap is a timeout expiring, which is the OPPOSITE finding.

    Uniform-under-a-ceiling says "entered a cycle at a random phase". A spike at
    the ceiling says "waited for something that never came". Reporting the second
    as the first would send the reader looking for a poll loop that is not the
    problem.
    """
    samples = [995.0 + (i % 5) for i in range(60)]
    assert tick_suspect(samples) is None


def test_tick_suspect_abstains_below_the_sample_floor():
    """Few samples look flat by accident; the check must not speak yet."""
    assert tick_suspect(_uniform(8, 1000.0)) is None


def test_tick_suspect_ignores_cheap_segments():
    """A 3 ms segment that happens to be flat is not worth a line.

    Sampling noise on a sub-millisecond segment is flat by nature, so without
    a floor this check would flag the cheapest segments in the census -- the
    ones whose cost nobody would act on -- and be ignored for it.
    """
    cheap = [x * 0.06 for x in _uniform(64, 50.0)]  # uniform on [0, 3 ms]
    assert max(cheap) < _TICK_MIN_MEDIAN_MS
    assert tick_suspect(cheap) is None


def test_tick_suspect_would_flag_the_same_shape_once_it_is_expensive():
    """The floor is the ONLY reason the case above is silent, not the shape.

    Without this, a floor set too high would make the previous test pass for
    the wrong reason and the check could be dead entirely.
    """
    expensive = [x * 20.0 for x in _uniform(64, 50.0)]  # uniform on [0, 1000 ms]
    assert tick_suspect(expensive) is not None


def test_the_accumulator_reaches_the_verdict_and_reports_it_once():
    """THE WIRING, not just the statistic.

    `tick_suspect` is a pure function and passes its tests whether or not
    anything ever calls it. This drives the accumulator the census actually
    feeds -- one flip's segments at a time, both directions interleaved as they
    really arrive -- and pins the two properties the log depends on: the verdict
    appears once it has the samples, and it appears exactly ONCE. A standing
    defect that reprints on every flip becomes furniture.
    """
    reset_tick_history()
    # ARRIVAL ORDER MATTERS AND MUST NOT BE SORTED. Flips arrive at arbitrary
    # phases, so any prefix of the history already spans the whole period. Feed
    # `_uniform` in its natural ascending order and the first 30 samples span
    # only [0, 460] -- which is honestly uniform under a 500 ms ceiling, and the
    # check would correctly say "tick" while naming the wrong period. That is a
    # property of monotone test data, not of the check, so the data is permuted
    # (37 is coprime with 64) to match how the samples really arrive.
    ticky = [_uniform(64, 1000.0)[(i * 37) % 64] for i in range(64)]
    work = [_work_shaped(64, 1000.0)[(i * 37) % 64] for i in range(64)]
    verdicts = []
    for i in range(64):
        verdicts += observe_segments_for_tick(
            [
                ("hicache_quiesce->resident_release", ticky[i]),
                ("plan->wave0", work[i]),
            ]
        )
    assert len(verdicts) == 1, f"expected exactly one verdict, got {verdicts}"
    assert "hicache_quiesce->resident_release" in verdicts[0]
    assert "1000 ms" in verdicts[0]
    # The work-shaped segment is never named -- if it were, the check would be
    # flagging the whole census and saying nothing.
    assert "plan->wave0" not in verdicts[0]


def test_segment_names_match_between_the_timing_line_and_the_shape_test():
    """One authority for segment naming.

    The shape test keys its history by segment name across flips. If the timing
    line and the accumulator ever derived that name separately, one spelling
    drift would split the history in two and the check would silently never
    accumulate enough samples to speak.
    """
    census = SeamCensus("pp_to_tp", 0, probe=lambda: None)
    census.mark("a")
    census.mark("b")
    census.mark("c")
    names = [name for name, _ms in census.segments_ms()]
    assert names == ["a->b", "b->c"]
    line = census.format_timing_line()
    for name in names:
        assert name in line
