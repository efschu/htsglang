"""y7z (02.10., A/B 7cwk87): the blocked depositor counts its OWN live credit
wait as a cycle edge from the wait's START.

Measured (same image y7v2 f47931648d, only FLLIPER_OPT_PDFLIP_DRAFT_PARK_SKIP_UNCHANGED
1 vs 0): with the skip, D TP0 sleeps ~120 ms earlier; D's serial tag deposits
(TP2 weights_0 838-868 ms vs 206-270) block on P collectors that sit in their own
VRAM credit wait for their first claim (PDFLIP-LEG-ORDER first claims 787-1788 ms
vs 143-562, via=card-overdraw); D>P layer 3.52-3.76 s vs 2.06-2.26 s.

The wait edge: `_recv_marked` (the depositor's 'free'/'done' wait) posts its
`blocked` flag only after SLOW_WAIT_S (0.5 s), and `deposit_cycle` ages edges
from that flag -- so the depositor's own half of the cycle could not be seen
before ~0.6 s although the depositor KNOWS it is blocked from the first
microsecond. With FLLIPER_PDFLIP_BAR1_SPILL_OWN_EDGE the tick hands its own wait
(dst, seq, wait start) to `deposit_cycle` and the spill comes ~spill_grace
after the block."""
from __future__ import annotations

import os
import threading
import time

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
from test_pdflip_bar1_spill_grace_y6u_1002 import _y6u_cycle_fresh  # noqa: E402


def _deposit_p5(tmp_path):
    # the ring window must outlive this helper: its address is in both lane objects
    dep, col, window = _pair(tmp_path)
    src, dst, descs = _tag(SLOT * 5 + 777, "model.layers.45.mlp.experts.w13")
    want = bytes(src)
    t0 = time.perf_counter()
    why = b1.run_bar1_units(descs, _HostOps(), lanes=dep, lane_key="p5", role="src",
                            seq="2-weights_11", phase="deposit", budget_s=8.0,
                            log=lambda *_a: None)
    took = time.perf_counter() - t0
    return why, took, (dep, col, src, dst, descs, want, window)


def _bytes_arrive(ctx):
    dep, col, src, dst, descs, want, _window = ctx
    src[:] = bytes(len(src))
    out = {}
    th = threading.Thread(target=_collect, args=(col, descs, "2-weights_11", out, "c"))
    th.start()
    th.join(15)
    assert out.get("c") == ""
    assert bytes(dst) == want
    assert dep.join_backlog(10.0) == ""


def test_own_edge_spills_without_waiting_for_its_own_blocked_flag(tmp_path):
    # the other half of the chain is old: both P wakers wait, sleeper1 is blocked
    _y6u_cycle_fresh(tmp_path, time.time() - 1.0)
    with envs.FLLIPER_PDFLIP_BAR1_SPILL_OWN_EDGE.override(True):
        why, took, ctx = _deposit_p5(tmp_path)
    assert why == "", why
    # base 999b64781a: >= SLOW_WAIT_S (0.5 s) -- the own flag must go up first
    assert took < 0.35, took
    _bytes_arrive(ctx)


def test_switch_off_keeps_the_flag_path(tmp_path):
    _y6u_cycle_fresh(tmp_path, time.time() - 1.0)
    with envs.FLLIPER_PDFLIP_BAR1_SPILL_OWN_EDGE.override(False):
        why, took, ctx = _deposit_p5(tmp_path)
    assert why == "", why
    assert took >= b1.SLOW_WAIT_S - 0.05, took
    _bytes_arrive(ctx)


def test_own_edge_alone_is_no_cycle(tmp_path):
    """No waker in a credit wait: the own edge closes nothing."""
    s2 = _lanes(tmp_path, "D", 2)
    assert s2.deposit_cycle(own=(1, "2-weights_11", time.time() - 5.0)) is None


def test_own_edge_still_needs_the_grace(tmp_path):
    """A wait younger than the spill grace is flag-read skew, not a cycle."""
    _y6u_cycle_fresh(tmp_path, time.time() - 1.0)
    s2 = _lanes(tmp_path, "D", 2)
    assert s2.deposit_cycle(own=(1, "2-weights_11", time.time())) is None
    chain = s2.deposit_cycle(own=(1, "2-weights_11", time.time() - b1.spill_grace_s() - 0.05))
    assert chain is not None and {(s, d) for s, d, _t in chain} == {(2, 1), (1, 2)}


def test_an_older_posted_flag_wins_over_the_own_edge(tmp_path):
    now = time.time()
    _y6u_cycle_fresh(tmp_path, now - 1.0)
    b1.post_flag(b1.flag_dir(NONCE, "p5", "P", str(tmp_path)), "blocked", "2-weights_11", 3,
                 {"since": now - 2.0, "wait": "free", "lane": "p5", "rank": 2})
    s2 = _lanes(tmp_path, "D", 2)
    # own edge brand new, the posted one is old -> the posted one counts
    assert s2.deposit_cycle(own=(1, "2-weights_11", now)) is not None


def test_switch_defaults_off():
    assert b1.own_edge_on() is False
