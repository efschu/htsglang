# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-E (user order 02.10.2026, both lines): FLIPZEIT D->P ends at the
first forward of the LAST P stage (all PP stages have begun), never at the
leg-1 dispatch. "NUR DIESE ZAHL ZAEHLT. DIE FALSCHE, ZU KLEINE ZAHL MUSS
UEBERALL WEG."

N5a 1002_114540 ep9: D->P done 11:49:06.227, the front dispatched leg 1 at once
(flip_user_time nachlauf 0.01 s, total 1.91 s) -- PP0 then read the store
(LOAD-DEVICE ms=860), waited the PF TOLD window (1.06 s), PP2 began its first
forward only at 11:49:08.297: the real D->P ended ~2.07 s after done (total ~4 s).

Variant (a), agreed with NF: flip_user_time.prefill_start_ts =
the last P rank's first forward_ct rise after flip_done in the progress beacon
(t_start_ns), prefill_start_source="pp_last_forward"; no beacon ->
prefill_start_ts None, source "missing", flip_user_ms None. Hermetic, CPU.
Each test named red_* is red on 4e7f1bcd8e.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

from sglang.srt.weg2 import front_state_ipc as fsi  # noqa: E402

NS = 1_000_000_000


def _clock(snap):
    c = fsi.DpFlipClock()
    c.begin(8, 100.0, 99.9)            # flip begin 100.0, the waiter arrived 99.9
    c.done(101.9, beacon_snap=snap)    # done 1.9 s later
    return c


SNAP = {11: (40, 90 * NS, 91 * NS), 12: (40, 90 * NS, 91 * NS), 13: (39, 90 * NS, 91 * NS)}


def test_red_dp_ends_at_the_last_stage_first_forward_not_at_the_dispatch():
    c = _clock(SNAP)
    assert c.waits_for_beacon()
    # PP0 starts at 102.8 (store read), PP1 103.6, PP2 104.0
    assert c.note_beacon({11: (41, int(102.8 * NS), 0), 12: (40, 90 * NS, 91 * NS), 13: (39, 90 * NS, 0)}) is None
    assert c.note_beacon({11: (42, int(103.3 * NS), 0), 12: (41, int(103.6 * NS), 0), 13: (39, 90 * NS, 0)}) is None
    assert c.note_beacon({11: (43, int(103.9 * NS), 0), 12: (42, int(104.2 * NS), 0),
                          13: (40, int(104.0 * NS), 0)}) == 104.0
    ev = c.first_prefill("weg2-7-9", 101.91, 106.0, 2.0)
    assert ev["prefill_start_source"] == "pp_last_forward"
    assert ev["prefill_start_ts"] == 104.0 and ev["leg1_dispatch_ts"] == 101.91
    assert ev["flip_user_ms"] == 4000                      # flip begin 100.0 -> 104.0, not 1910 (dispatch)
    assert ev["parts"]["first_chunk_ms"] == 2100           # done -> last stage's first forward


def test_red_a_wider_rise_between_readings_is_named_approximate():
    c = _clock({11: (40, 0, 0)})
    assert c.note_beacon({11: (43, int(103.0 * NS), 0)}) == 103.0
    assert c.first_prefill("r", 102.0, 105.0, None)["prefill_start_source"] == "pp_last_forward_approx"


def test_red_no_beacon_is_missing_never_the_dispatch():
    c = _clock(None)
    assert not c.waits_for_beacon()
    ev = c.first_prefill("r", 101.95, 105.0, 2.5)
    assert ev["prefill_start_ts"] is None and ev["prefill_start_source"] == "missing"
    assert ev["flip_user_ms"] is None and ev["parts"]["first_chunk_ms"] is None


def test_red_a_rank_that_never_rose_leaves_the_end_missing():
    c = _clock(SNAP)
    c.note_beacon({11: (41, int(102.8 * NS), 0), 12: (41, int(103.0 * NS), 0), 13: (39, 90 * NS, 0)})
    ev = c.first_prefill("r", 101.95, 105.0, 2.5)
    assert ev["prefill_start_source"] == "missing" and ev["flip_user_ms"] is None


def test_red_the_front_watches_the_beacon_and_fires_no_dispatch_first_work():
    from sglang.srt.weg2 import front as F

    src = inspect.getsource(F.Front)
    assert '_ipc_first_work_seen("P", "p_leg1_dispatch"' not in src
    assert 'Front._ipc_first_work_at(self, "P", "p_last_stage_forward"' in src
    assert "beacon_snap=_bsnap" in src and "Front._watch_pp_last_forward(self)" in src


# ---- P->D end (NF rule, 02.10.): D's first decode token may precede done, never its own wake --

def _fw():
    c = fsi.FirstWorkClock()
    c.arm(4, "P", "D", 100.0)
    return c


def test_red_d_content_after_the_kv_wake_counts_before_done_even_for_an_earlier_leg2():
    """N4p/N4q first P->D: weg2-0-1's leg 2 was dispatched 1.5 s BEFORE the flip began and
    its first token came before done -- the old rule skipped it (front 7.59 s vs 5.22 s)."""
    c = _fw()
    c.note_awake(102.4)                                  # D's kv wake answered
    ev = c.seen("D", "decode_token", "weg2-0-1", 102.5, leg2_dispatch_ts=98.5)
    assert ev is not None and ev["first_work_ts"] == 102.5 and ev["before_done"] is True


def test_red_d_content_before_its_wake_is_still_refused():
    """NF y6d class: a chunk 0.02-0.99 s after begin, D not awake -- refused."""
    c = _fw()
    assert c.seen("D", "decode_token", "r", 100.3, leg2_dispatch_ts=95.0) is None    # an earlier phase's stream
    c.note_awake(102.4)
    assert c.seen("D", "decode_token", "r", 102.0, leg2_dispatch_ts=100.1) is None   # > 0.3 s early
    assert c.seen("D", "decode_token", "r", 102.2, leg2_dispatch_ts=100.1) is not None
