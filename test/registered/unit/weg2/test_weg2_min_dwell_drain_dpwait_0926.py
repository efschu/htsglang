"""27B DPWAIT (release table row 28): K7's min-dwell must not price the previous drain.

Measured on dkr27bnvfp4bar1agent09252328 (front log 23:48:13-23:49:53): the D->P
flip of epoch 12 drained a running 10197-token decode for 29.7 s, so its record
carried flip_ms=31115 (drain_quiesce_ms=29670). The next D->P decision at
23:49:14 read ``MIN-DWELL ... awake_ms=8758 derived_from_flip_ms=31115 ...
verdict=hold`` and held a queued batch for 39.2 s, until FAIRNESS fired and a
short admitted during the hold had to be drained too (11.3 s).

Hermetic: no GPU, no model, no server. The flag SGLANG_WEG2_MIN_DWELL_EXCLUDE_DRAIN
defaults to off (behaviour byte-identical to before); on, the price is the flip
without its drain.
"""

import os
import time
from unittest import mock

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2.front import Front, flip_price_ms

# The epoch-12 record of dkr27bnvfp4bar1agent09252328, reduced to the fields K7 reads.
REC_DRAINED = {"sleep": "D", "wake": "P", "flip_ms": 31115, "drain_quiesce_ms": 29670}
REC_CLEAN = {"sleep": "D", "wake": "P", "flip_ms": 1445, "drain_quiesce_ms": 7}


def _front():
    return Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _qwen27b_form(monkeypatch):
    """UNIFY S7: K7's last-flip price is the qwen27b profile's form (NF H34b
    warm min-dwell is the nextflash default); published as the launcher does."""
    from sglang.srt.weg2 import form as _F

    monkeypatch.setenv(_F.FORM_ENV, _F.Weg2Form(
        arch="dense", experts="none", draft="dflash", p_draft="none", kv="paged_dcp",
        flip="family", vision="off", profile="qwen27b", model="m").env_value())


def test_flag_default_off_keeps_the_old_price_and_provenance():
    assert front_mod.envs.SGLANG_WEG2_MIN_DWELL_EXCLUDE_DRAIN.get() is False
    f = _front()
    f.flip_log.append(dict(REC_DRAINED))
    assert f._derived_min_dwell_ms("D", "P") == (31115.0, "last-flip-D->P")


def test_flag_on_prices_the_flip_without_its_drain():
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_MIN_DWELL_EXCLUDE_DRAIN": "1"}):
        f = _front()
        f.flip_log.append(dict(REC_DRAINED))
        assert f._derived_min_dwell_ms("D", "P") == (1445.0, "last-flip-D->P:drain-29670ms-excluded")
        # the measured situation: D awake 8758 ms -> the flip may go (it was held 39.2 s)
        f.t_awake = time.time() - 8.758
        assert f._dwell_ok("D", "P", fairness_fired=False, work_exhausted=False, oldest_wait_s=6.0) is True


def test_flag_on_without_drain_changes_nothing_but_the_value():
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_MIN_DWELL_EXCLUDE_DRAIN": "1"}):
        f = _front()
        f.flip_log.append({"sleep": "D", "wake": "P", "flip_ms": 14000})       # old record, no drain field
        assert f._derived_min_dwell_ms("D", "P") == (14000.0, "last-flip-D->P")
        f.flip_log.append(dict(REC_CLEAN))
        ms, prov = f._derived_min_dwell_ms("D", "P")
        assert ms == 1438.0 and prov == "last-flip-D->P:drain-7ms-excluded"
        assert " " not in prov                        # the L12 line is split on spaces


def test_flag_on_keeps_the_hold_for_a_genuinely_fresh_group():
    """The drain exclusion must not open the thrash K7 exists for (R-5): a group
    that woke 200 ms ago is still held by the flip price itself."""
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_MIN_DWELL_EXCLUDE_DRAIN": "1"}):
        f = _front()
        f.flip_log.append(dict(REC_DRAINED))
        f.t_awake = time.time() - 0.2
        assert f._dwell_ok("D", "P", fairness_fired=False, work_exhausted=False, oldest_wait_s=0.0) is False


def test_flip_price_ms_edges():
    assert flip_price_ms(REC_DRAINED, exclude_drain=False) == (31115.0, 0.0)
    assert flip_price_ms(REC_DRAINED, exclude_drain=True) == (1445.0, 29670.0)
    # a drain longer than the record (never measured, but no negative price)
    assert flip_price_ms({"flip_ms": 100, "drain_quiesce_ms": 500}, exclude_drain=True) == (0.0, 100.0)
    assert flip_price_ms({"flip_ms": None}, exclude_drain=True) == (0.0, 0.0)
    assert flip_price_ms({"flip_ms": 900, "drain_quiesce_ms": -3}, exclude_drain=True) == (900.0, 0.0)


def test_other_direction_and_flag_pin_are_untouched():
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_MIN_DWELL_EXCLUDE_DRAIN": "1"}):
        f = _front()
        f.flip_log.append(dict(REC_DRAINED))
        assert f._derived_min_dwell_ms("P", "D") == (0.0, "none-first-flip")
        g = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, min_dwell_ms=5000.0)
        g.flip_log.append(dict(REC_DRAINED))
        assert g._derived_min_dwell_ms("D", "P") == (5000.0, "flag")
