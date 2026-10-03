# SPDX-License-Identifier: Apache-2.0
"""Y8P-PPFWD-ARM-RACE, port to the 27B line (deskq item 490; NF 044316dd1a).

NF y8p 03.10. (08:52:05 / 08:53:41 / 08:57:00): P's first forward after a D->P flip can start
0-70 ms BEFORE the front logs ``done`` (the wake RPC returns, P admits at once). The beacon
snapshot taken at ``done`` then already contains that forward; ``DpFlipClock`` waited for the
NEXT forward_ct rise and reported the start of the SECOND chunk -- a false Nachlauf of one whole
chunk (2.5-2.9 s) in 3 of 7 flips (D->P total 5.2 s where the flip was 2.2 s). P sleeps for the
whole D phase, so a rank whose ``t_start`` lies at or after the flip's begin ran its first forward
of the woken group. Hermetic, CPU. Tests named red_* are red on edeb022aee.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

from sglang.srt.weg2 import front_state_ipc as fsi  # noqa: E402

NS = 1_000_000_000


def _clock(snap, begin=100.0, done=102.3):
    c = fsi.DpFlipClock()
    c.begin(8, begin, begin - 0.1)
    c.done(done, beacon_snap=snap)
    return c


def test_red_a_forward_that_began_before_done_but_after_begin_is_the_first_forward():
    # PP0 started at 102.25 (after begin 100.0, 50 ms before done 102.3): it is IN the snapshot.
    snap = {11: (41, int(102.25 * NS), 0), 12: (40, 90 * NS, 91 * NS), 13: (39, 90 * NS, 91 * NS)}
    c = _clock(snap)
    assert c.first_forward_ts() == 102.25
    assert c.waits_for_beacon()                       # the other stages are still to come
    # the second chunk (4.9 s later) must NOT replace the first one
    assert c.note_beacon({11: (42, int(105.0 * NS), 0), 12: (40, 90 * NS, 91 * NS),
                          13: (39, 90 * NS, 91 * NS)}) == 102.25
    ev = c.first_prefill("r", 102.4, 108.0, 2.0)
    assert ev["prefill_start_source"] == "pp_first_forward"
    assert ev["prefill_start_ts"] == 102.3            # clamped to done: Nachlauf 0, not one chunk
    assert ev["parts"]["first_chunk_ms"] == 0
    assert ev["flip_user_ms"] == 2300                 # begin 100.0 -> done 102.3 (Nachlauf 0)


def test_red_forward_of_the_previous_phase_does_not_count():
    # every rank's last forward is from before the flip's begin (the P phase of the epoch before)
    snap = {11: (40, int(95.0 * NS), int(95.5 * NS)), 12: (40, int(95.0 * NS), int(95.5 * NS))}
    c = _clock(snap)
    assert c.first_forward_ts() is None
    assert c.note_beacon({11: (41, int(103.0 * NS), 0), 12: (40, int(95.0 * NS), 0)}) == 103.0


def test_red_all_stages_already_inside_the_snapshot_close_the_clock_at_done():
    snap = {11: (41, int(102.20 * NS), 0), 12: (41, int(102.26 * NS), 0)}
    c = _clock(snap)
    assert c.first_forward_ts() == 102.20
    assert not c.waits_for_beacon()
    ev = c.first_prefill("r", 102.5, 106.0, 2.0)
    assert ev["pp_last_start_ts"] == 102.26


def test_empty_baseline_and_no_beacon_stay_as_before():
    assert _clock({}).first_forward_ts() is None
    c = _clock(None)
    assert not c.waits_for_beacon()
    assert c.first_prefill("r", 102.4, 105.0, 2.0)["prefill_start_source"] == "missing"


def test_red_the_front_fires_a_first_forward_known_at_done():
    from sglang.srt.weg2 import front as F

    src = inspect.getsource(F.Front._watch_pp_last_forward)
    assert "first_forward_ts()" in src and 'Front._ipc_first_work_at(self, "P", "p_first_stage_forward"' in src
