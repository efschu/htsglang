"""H91c2 / 27B DPWAIT port: K7's min-dwell must not price the previous drain -- on
the NF front ALSO inside the H34b warm median (default on here, absent on 27B).

27B measured (dkr27bnvfp4bar1agent09252328): a D->P flip that drained a running
decode for 29.7 s recorded flip_ms=31115 and held the next D->P 39.2 s. On the NF
front the same record enters H34b's median of the last 5 same-direction warm
flips; once drained flips are the majority of the window (agent load, fallback
drains of the wait bound) the median IS a drain.

SGLANG_WEG2_MIN_DWELL_EXCLUDE_DRAIN (same name and rule as 27B 30d33262dd,
default off = byte-identical); on, every priced record is flip_ms minus its
drain. Hermetic: no GPU, no model, no server.
"""

import os
import time
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402

FIRST = {"sleep": "P", "wake": "D", "flip_ms": 24600, "drain_quiesce_ms": 5}
DRAINED = {"sleep": "D", "wake": "P", "flip_ms": 31115, "drain_quiesce_ms": 29670}
CLEAN = {"sleep": "D", "wake": "P", "flip_ms": 1445, "drain_quiesce_ms": 7}
ON = {"SGLANG_WEG2_MIN_DWELL_EXCLUDE_DRAIN": "1"}


def _front(*recs):
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0)
    f.flip_log.extend(dict(r) for r in recs)
    return f


def test_warm_median_of_drained_flips_holds_the_next_d_to_p_unless_excluded():
    """Red on c8bf488fe7 (no flag, no exclusion): three drained of five warm
    D->P flips make the median 31115 ms and D, awake 8.8 s with its work done,
    is held although the flip itself costs ~1.4 s."""
    recs = [FIRST, CLEAN, DRAINED, CLEAN, DRAINED, DRAINED]
    with mock.patch.dict(os.environ, ON):
        f = _front(*recs)
        ms, prov = f._derived_min_dwell_ms("D", "P")
        assert ms == 1445.0, (ms, prov)
        assert prov.startswith("median-warm-D->P:n=5:first-flip-P->D-24600ms-excluded")
        assert prov.endswith(":drain-max-29670ms-excluded") and " " not in prov
        f.t_awake = time.time() - 8.758
        assert f._dwell_ok("D", "P", fairness_fired=False, work_exhausted=True,
                           oldest_wait_s=6.0) is True


def test_flag_off_is_byte_identical_on_both_branches():
    assert front_mod.envs.SGLANG_WEG2_MIN_DWELL_EXCLUDE_DRAIN.get() is False
    f = _front(FIRST, CLEAN, DRAINED, CLEAN, DRAINED, DRAINED)
    assert f._derived_min_dwell_ms("D", "P") == (
        31115.0, "median-warm-D->P:n=5:first-flip-P->D-24600ms-excluded")
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_ENABLE_WARM_MIN_DWELL": "0"}):
        g = _front(FIRST, DRAINED)
        assert g._derived_min_dwell_ms("D", "P") == (31115.0, "last-flip-D->P")


def test_flag_on_legacy_branch_matches_27b():
    with mock.patch.dict(os.environ, dict(ON, SGLANG_WEG2_ENABLE_WARM_MIN_DWELL="0")):
        f = _front(FIRST, DRAINED)
        assert f._derived_min_dwell_ms("D", "P") == (1445.0, "last-flip-D->P:drain-29670ms-excluded")
        g = _front({"sleep": "D", "wake": "P", "flip_ms": 14000})       # no drain field
        assert g._derived_min_dwell_ms("D", "P") == (14000.0, "last-flip-D->P")


def test_flag_on_keeps_the_hold_for_a_fresh_group_and_the_any_direction_fallback():
    with mock.patch.dict(os.environ, ON):
        f = _front(FIRST, DRAINED)
        f.t_awake = time.time() - 0.2
        assert f._dwell_ok("D", "P", fairness_fired=False, work_exhausted=False,
                           oldest_wait_s=0.0) is False
        g = _front(FIRST, dict(DRAINED))
        ms, prov = g._derived_min_dwell_ms("P", "D")          # no warm P->D yet
        assert ms == 1445.0 and prov.startswith("median-warm-any-direction:n=1")
        assert prov.endswith(":drain-max-29670ms-excluded")
        assert _front(FIRST)._derived_min_dwell_ms("D", "P")[1].startswith("none-after-first-flip")


def test_flip_price_ms_is_the_27b_reader():
    fp = front_mod.flip_price_ms
    assert fp(DRAINED, exclude_drain=False) == (31115.0, 0.0)
    assert fp(DRAINED, exclude_drain=True) == (1445.0, 29670.0)
    assert fp({"flip_ms": 100, "drain_quiesce_ms": 500}, exclude_drain=True) == (0.0, 100.0)
    assert fp({"flip_ms": None}, exclude_drain=True) == (0.0, 0.0)
    assert fp({"flip_ms": 900, "drain_quiesce_ms": -3}, exclude_drain=True) == (900.0, 0.0)
