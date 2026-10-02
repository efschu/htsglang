"""D-SEAT-REWAKE, the front's half (NF-Operator 30.09.): a request waiting for a
D seat must be SEEN by D when the seats are full, so the GROW can take it.

The admitter's seats are ``--d-bs`` (the semaphore; ``_arrival_seat_taken``
already counts D's cap under D-SEAT-REWAKE): a hand-off in ``_ready_for_d`` is
posted to D up to the cap, and past the wake's n it lands in D's waiting queue,
which is what the GROW reads. The one gap: H91c3-3's ``_d_phase_seats_full``
(the path without ARRIVAL-SEAT) still counted the WAKE's n and sent a SHORT
past it to the P queue -- D never saw it. With the rewake on, full = the cap.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402


def _front(running):
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0)
    f.d_bs = 6
    f.d_wait_bound_s = 60.0
    f._d_phase_n = 2
    for i in range(running):
        f.groups["D"].outstanding["r%d" % i] = 1.0
    return f


def test_with_the_rewake_a_short_past_the_wakes_n_goes_to_d():
    with envs.SGLANG_WEG2_D_SEAT_REWAKE.override(True):
        assert _front(2)._d_phase_seats_full("s") is False     # 2 of the cap 6: D grows
        assert _front(6)._d_phase_seats_full("s") is True      # the cap: no GROW possible


def test_without_the_rewake_the_wakes_n_holds():
    with envs.SGLANG_WEG2_D_SEAT_REWAKE.override(False):
        assert _front(2)._d_phase_seats_full("s") is True


def test_the_admitters_seats_are_the_cap():
    """The admitter posts up to --d-bs, never the wake's n: a hand-off past n reaches D."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0)
    f._d_phase_n = 1
    assert f._d_seat._value == f.d_bs
