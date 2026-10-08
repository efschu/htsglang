# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-B (02.10.2026): the X-SOLO band floor and FLIP-ECONOMICS follow the
excursion-priced live X -- no flip pair for a turn D prices cheaper itself.

N5d 1002_124821: 7 of 7 D->P flips by 12:56 were PARK-IMMEDIATE cause=over-x at
est_uncached 7668 / 1942 / 18825 / 8139 / 18422 / 6382 / 6324. pdflip-6-12:
``X-SOLO uncached=7668 X_live=10674 verdict=p reason=d_outstanding=2 (band
floor = X_busy 4096)`` -> x_deferred -> needs_p -> immediate park; FLIP-ECONOMICS
``threshold=4096`` beside X=10674; MIN-DWELL ``overridden_by=fairness`` with
oldest_wait_s=0.0 (admission closed by the park itself). Helper
``Front._x_excursion_band_off``, switch ``FLLIPER_PDFLIP_X_BAND_FOLLOWS_PRICE``.
Hermetic, CPU. Each test named red_* is red on e90d7fcf88.
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402


def _front(**kw):
    ns = types.SimpleNamespace(x_start_tokens=4096, x_busy_tokens=None, tp_prefill_max_tokens=10674,
                               flip_min_work_tokens=4096, epoch=6, _park_attempt_epoch=6)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_red_the_band_floor_is_the_live_x_under_the_excursion_price():
    ns = _front()
    with envs.FLLIPER_PDFLIP_X_EXCURSION_PRICE.override(True), envs.FLLIPER_PDFLIP_ENABLE_X_COST_LINE.override(True):
        assert F.Front._x_band_floor(ns) == 10674
        # pdflip-6-12: 7668 uncached is no band request any more -> it never reaches the X-SOLO
        # singleton rule, is routed SHORT on the live X, and never needs P
        assert not pp.needs_p(7668, F.Front._x_band_floor(ns))


def test_switch_off_restores_the_band_floor():
    ns = _front()
    with envs.FLLIPER_PDFLIP_X_BAND_FOLLOWS_PRICE.override(False):
        assert F.Front._x_band_floor(ns) == 4096
    with envs.FLLIPER_PDFLIP_X_EXCURSION_PRICE.override(False):
        assert F.Front._x_band_floor(ns) == 4096


def test_red_flip_economics_compares_the_backlog_with_the_live_x(caplog):
    import logging

    ns = _front(queue=[types.SimpleNamespace(est_uncached=7668, est_prompt=26868, p_only=False)],
                admit_d=True, groups={"D": types.SimpleNamespace(outstanding={})}, vision_flip_urgent=False)
    with envs.FLLIPER_PDFLIP_X_EXCURSION_PRICE.override(True), caplog.at_level(logging.INFO):
        ok = F.Front._flip_economics_ok(ns, False)
    assert ok is False                                       # 7668 < X 10674: hold
    assert "threshold=10674" in caplog.text


def test_red_a_park_closed_admission_is_named_park_not_fairness(caplog):
    import logging
    import time

    ns = _front(t_awake=time.time() - 6.65, counters={"min_dwell_holds": 0}, w_s=45.0)
    ns._derived_min_dwell_ms = lambda src, dst: (12610.0, "last-flip-D->P")
    with caplog.at_level(logging.INFO):
        F.Front._dwell_ok(ns, "D", "P", True, work_exhausted=False, oldest_wait_s=0.0)
    assert "overridden_by=park" in caplog.text and "overridden_by=fairness" not in caplog.text
