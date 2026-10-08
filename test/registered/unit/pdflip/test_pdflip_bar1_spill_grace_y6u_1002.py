"""W109b spill latency (NF y6u, 01.10. 23:47:31Z and 23:47:47Z, D->P epochs 2->3
and 4->5): the blocked depositor answers a credit cycle within ~0.7 s, not 1.5-2 s.

Measured (D log, PDFLIP-SLEEP-TAG-TIME / PDFLIP-BAR1 lane-time / CYCLE-SPILL):
  D TP1 weights_15 lane p3 deposit_ms=1906 credit_ms=1819,
  D TP0 weights_15 lane p1 deposit_ms=1759 credit_ms=1724,
  D TP2 weights_11 lane p5 deposit_ms=1597 -> CYCLE-SPILL from_batch=4/33
  (sleeper1 blocked depositing weights_15 to waker2 -> sleeper2 blocked
  depositing weights_11 to waker1); gathered legs 3396 / 3455 ms against
  1314-1447 ms in the four uncycled D->P flips of the same boot.

Why so late: the `blocked` flag goes up after SLOW_WAIT_S (0.5 s), the
depositor demanded edges a third of the waker's W109 grace old (1.0 s), and on
the metal path (credits=sock) asked only once per 0.5-s socket slice. The fix:
its own grace (FLLIPER_PDFLIP_BAR1_SPILL_GRACE_S, 0.1 s) and the tick cadence on
the socket path too. The waker's W109 grace (3 s) is untouched."""
from __future__ import annotations

import os
import socket
import threading
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import bar1_lanes as b1  # noqa: E402

from test_pdflip_bar1_cycle_spill_w109b import (  # noqa: E402
    NONCE,
    SLOT,
    _collect,
    _HostOps,
    _lanes,
    _pair,
    _tag,
)


def _y6u_cycle_fresh(tmp_path, since):
    """The y6u chain with REAL ages: both P wakers entered their credit wait and
    sleeper1 blocked on p3 (1->2) at ``since``; sleeper2's p5 edge is posted by
    its own wait."""
    root = str(tmp_path)
    cd = os.path.join(root, f"pdflip-bar1-{NONCE}", "credit")
    b1.post_flag(cd, "wait", "P", 1, {"tag": "weights_11", "since": since, "submitted": []})
    b1.post_flag(cd, "wait", "P", 2, {"tag": "weights_15", "since": since, "submitted": []})
    b1.post_flag(b1.flag_dir(NONCE, "p3", "P", root), "blocked", "2-weights_15", 0,
                 {"since": since, "wait": "free", "lane": "p3", "rank": 1})


def test_y6u_cycle_spills_within_a_slow_wait_plus_the_spill_grace(tmp_path):
    _y6u_cycle_fresh(tmp_path, time.time())
    dep, col, _w = _pair(tmp_path)
    src, dst, descs = _tag(SLOT * 5 + 777, "model.layers.45.mlp.experts.w13")
    want = bytes(src)
    t0 = time.perf_counter()
    why = b1.run_bar1_units(descs, _HostOps(), lanes=dep, lane_key="p5", role="src",
                            seq="2-weights_11", phase="deposit", budget_s=8.0,
                            log=lambda *_a: None)
    took = time.perf_counter() - t0
    assert why == "", why
    # base 8d438079aa: 1.5 s (own flag at 0.5 s + 1.0 s grace, 0.5-s tick)
    assert took < 1.0, took
    # bytes still arrive whole (the spill path, unchanged)
    src[:] = bytes(len(src))
    out = {}
    th = threading.Thread(target=_collect, args=(col, descs, "2-weights_11", out, "c"))
    th.start()
    th.join(15)
    assert out.get("c") == ""
    assert bytes(dst) == want
    assert dep.join_backlog(10.0) == ""


def test_a_fresh_chain_is_not_yet_a_cycle(tmp_path):
    """The grace still stands between flag-read skew and a spill."""
    now = time.time()
    _y6u_cycle_fresh(tmp_path, now)
    b1.post_flag(b1.flag_dir(NONCE, "p5", "P", str(tmp_path)), "blocked", "2-weights_11", 3,
                 {"since": now, "wait": "free", "lane": "p5", "rank": 2})
    s2 = _lanes(tmp_path, "D", 2)
    early = s2.deposit_cycle()
    # (a slow runner may already be past the grace -- then the chain is due)
    assert early is None or time.time() - now >= b1.spill_grace_s(), early
    time.sleep(b1.spill_grace_s() + 0.05)
    chain = s2.deposit_cycle()
    assert chain is not None and {(s, d) for s, d, _t in chain} == {(2, 1), (1, 2)}


def test_socket_wait_asks_the_cycle_question_every_spill_tick():
    """credits=sock (the metal path): the tick question is asked at the
    SPILL_TICK_S cadence, not once per 0.5-s socket slice."""
    a, b = socket.socketpair()
    try:
        cred = b1.SocketCredits(a)
        t0 = time.monotonic()
        cred.on_tick = lambda: time.monotonic() - t0 >= 0.15
        with pytest.raises(b1._CycleSpill):
            cred.recv("free", 0, 5.0)
        took = time.monotonic() - t0
        # base 8d438079aa: 0.5 s (the first socket timeout)
        assert took < 0.4, took
    finally:
        a.close()
        b.close()


def test_socket_wait_posts_blocked_after_slow_wait_not_after_the_tick():
    a, b = socket.socketpair()
    try:
        cred = b1.SocketCredits(a)
        t0 = time.monotonic()
        slow = []
        cred.on_slow = lambda: slow.append(time.monotonic() - t0)
        cred.on_tick = lambda: False
        assert cred.recv("free", 0, 0.9) is None
        assert len(slow) == 1 and b1.SLOW_WAIT_S - 0.01 <= slow[0] < b1.SLOW_WAIT_S + 0.3, slow
    finally:
        a.close()
        b.close()


def test_spill_grace_is_its_own_knob_bounded_by_the_old_third():
    with envs.FLLIPER_PDFLIP_BAR1_SPILL_GRACE_S.override(0.1):
        assert b1.spill_grace_s() == pytest.approx(0.1)
    with envs.FLLIPER_PDFLIP_BAR1_SPILL_GRACE_S.override(5.0):
        # never slower than before (a third of the waker's 3-s W109 grace)
        assert b1.spill_grace_s() == pytest.approx(1.0)
    with envs.FLLIPER_PDFLIP_BAR1_SPILL_GRACE_S.override(0.0):
        assert b1.spill_grace_s() == pytest.approx(0.05)
    # the waker's W109 grace is a different number and stays 3 s
    assert b1.cycle_grace_s({}) == pytest.approx(3.0)
