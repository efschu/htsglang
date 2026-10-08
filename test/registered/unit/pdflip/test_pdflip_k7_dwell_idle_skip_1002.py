# SPDX-License-Identifier: Apache-2.0
"""K7-DWELL idle skip (02.10.2026): the D->P min-dwell holds only a D with work.

N5j 1002_134624 (582d0fca0d) 13:50:12.317 P->D done; 13:50:12.798-14.805 eleven
``MIN-DWELL src=D dst=P awake_ms=482..2489 derived_from_flip_ms=2586 verdict=hold``
with LONG pdflip-2-6 (32676 uncached) queued and D running nothing
(outstanding=0); the flip to P came at 15.006 -- 2.2 s of an idle D before the
D->P flip. Same at 13:50:43-44.9 (1890 ms). ``Front._k7_dwell_idle_skip``,
marker ``PDFLIP ARRIVAL-SEAT K7-DWELL skip ... d_running=0`` and counter
``arrival_seat_k7_dwell_skip_idle`` (NF cd12370e30 names), switch
``FLLIPER_PDFLIP_K7_DWELL_IDLE_SKIP``. Hermetic, CPU; red_* red on a0d03e9321.
"""

from __future__ import annotations

import logging
import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402


def _front(awake_s=0.482, outstanding=None, handoff=0, ready=()):
    D = types.SimpleNamespace(outstanding=dict(outstanding or {}))
    ns = types.SimpleNamespace(t_awake=time.time() - awake_s, counters={"min_dwell_holds": 0}, w_s=45.0,
                               epoch=4, _park_attempt_epoch=-1, groups={"D": D}, queue=[object()],
                               _ready_for_d=list(ready))
    ns._derived_min_dwell_ms = lambda src, dst: (2586.0, "last-flip-D->P")
    ns._handoff_in_flight = lambda: handoff
    ns._flip_ledger = lambda g: list(g.outstanding)
    return ns


def test_red_an_idle_d_flips_at_once(caplog):
    ns = _front()
    with caplog.at_level(logging.INFO):
        ok = F.Front._dwell_ok(ns, "D", "P", False, work_exhausted=True, oldest_wait_s=0.0)
    assert ok is True
    assert "PDFLIP ARRIVAL-SEAT K7-DWELL skip" in caplog.text and "d_running=0" in caplog.text
    assert "overridden_by=d_idle" in caplog.text and "verdict=flip" in caplog.text
    assert ns.counters["min_dwell_holds"] == 0 and ns.counters["arrival_seat_k7_dwell_skip_idle"] == 1


def test_a_parked_decode_is_no_running_decode():
    ns = _front(outstanding={"a": object()})
    ns._flip_ledger = lambda g: []                     # D parked it: the ledger drops it (NF)
    assert F.Front._dwell_ok(ns, "D", "P", False, True, 0.0) is True


def test_a_d_with_running_decodes_still_holds():
    ns = _front(outstanding={"pdflip-2-5": object()})
    assert F.Front._dwell_ok(ns, "D", "P", False, work_exhausted=False, oldest_wait_s=0.0) is False
    assert ns.counters["min_dwell_holds"] == 1


def test_a_handoff_or_a_ready_request_is_work():
    assert F.Front._dwell_ok(_front(handoff=1), "D", "P", False, True, 0.0) is False
    assert F.Front._dwell_ok(_front(ready=[object()]), "D", "P", False, True, 0.0) is False


def test_the_p_side_dwell_is_untouched():
    ns = _front()
    assert F.Front._dwell_ok(ns, "P", "D", False, True, 0.0) is False


def test_switch_off_holds_the_idle_d_as_before(caplog):
    ns = _front()
    with envs.FLLIPER_PDFLIP_K7_DWELL_IDLE_SKIP.override(False), caplog.at_level(logging.INFO):
        assert F.Front._dwell_ok(ns, "D", "P", False, True, 0.0) is False
    assert "K7-DWELL skip" not in caplog.text


def test_past_the_dwell_no_skip_marker(caplog):
    ns = _front(awake_s=3.0)
    with caplog.at_level(logging.INFO):
        assert F.Front._dwell_ok(ns, "D", "P", False, True, 0.0) is True
    assert "K7-DWELL skip" not in caplog.text and "overridden_by=none" in caplog.text
