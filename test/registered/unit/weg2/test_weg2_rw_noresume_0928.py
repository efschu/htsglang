"""RW instrument: the wake-to-first-decode window opens only when the wake has
parked work. NF rc12z26 18:32: all requests aborted before the wake, the first
decode (a health check) came 114 s later and the line read 116342 ms."""
import inspect


def test_the_window_opens_only_with_parked_work():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu)
    i = src.index("scheduler._rw_first_token_open = ")
    seg = src[i - 900:i + 500]
    assert "weg2_dormant_hold" in seg and "waiting_queue" in seg
    assert "scheduler._rw_first_token_open = _rw_work > 0" in seg
    assert "RW WAKE-FIRST-TOKEN kein Resume" in seg
    assert "scheduler._rw_first_token_open = True" not in src
