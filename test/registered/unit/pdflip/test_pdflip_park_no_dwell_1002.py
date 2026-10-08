# SPDX-License-Identifier: Apache-2.0
"""PARK-NO-DWELL (02.10.): a request over X that arrives while D decodes is
prefilled at once -- D's decodes park now, no dwell of any kind.

User rule (02.10.): an arriving request is prefilled AT ONCE; grace for running
decodes is forbidden. N6d (..._ec4d492f58_1002_170955) epoch 4: pdflip-4-6 (LONG,
uncached 25215 > X 8359) verdict 17:13:24.972, then 'PDFLIP PARK-IMMEDIATE-DWELL
epoch=4 rid=pdflip-4-6 awake_ms=310 min_dwell_ms=1974 (last-flip-D->P)
floor_ms=2000' -- the park waited for the 2-token SHORT pdflip-4-5 on D to finish
(SERVED 25.109), the flip began at 25.114: 142 ms Verdikt -> begin. With a
longer decode the hold runs to the dwell (~2 s), and without a free seat the
PARK-COLLECT-WINDOW holds for the ski price (round trip x streams).

The 27B front runs this through Front._immediate_park_due (ARRIVAL-SEAT is
off on 27B; its own FLLIPER_PDFLIP_ARRIVAL_MIN_DWELL is inert there). Switch
FLLIPER_PDFLIP_PARK_NO_DWELL (default on): no K7 min-dwell, no park-cycle /
decode dwell, no fairness floor, no seat-free min-dwell, no collect window --
and the D->P MIN-DWELL after that park does not hold. ACCEPTED CONSEQUENCE
(user rule): a decode D resumed in this phase can be parked again at once
(the y5c ping-pong protection is off with the switch; 0 restores it).
"""
from __future__ import annotations

import importlib.util
import logging
import time
import types
from pathlib import Path

from flliper.srt.environ import envs
from flliper.srt.pdflip import front as F

_spec = importlib.util.spec_from_file_location(
    "_flipwait_harness_nodwell", Path(__file__).with_name("test_27b_flipwait_1002.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def test_red_n6d_epoch4_the_long_parks_d_at_once(caplog):
    # D woke 310 ms ago, one running decode, a free seat -- the N6d shape
    a, b = H._on()
    with a, b, caplog.at_level(logging.INFO, logger="pdflip.front"):
        fn, ns = H._front(running=1, waits_s=[0.0], awake_s=0.31, uncached=25215)
        got = fn(ns, None, time.time())
    assert got is not None and got.rid == "pdflip-16-39"
    assert ns.counters["park_immediate_dwell_holds"] == 0
    assert any("PDFLIP PARK-NO-DWELL epoch=16 rid=pdflip-16-39" in m and "would_hold_ms=2000" in m
               for m in caplog.messages), caplog.messages


def test_no_free_seat_no_collect_window():
    a, b = H._on()
    with a, b:
        fn, ns = H._front(running=6, waits_s=[0.5])
        assert fn(ns, None, time.time()) is not None
        assert ns.counters["park_collect_holds"] == 0


def test_a_resumed_decode_is_no_hold_either():
    a, b = H._on()
    with a, b:
        fn, ns = H._front(running=2, waits_s=[0.5], awake_s=5.0, resumed=("pdflip-15-0",))
        assert fn(ns, None, time.time()) is not None
        assert ns.counters["park_seat_free_min_dwell_hold"] == 0


def test_the_park_latch_and_the_trigger_stay():
    with envs.FLLIPER_PDFLIP_PARK_NO_DWELL.override(True):
        fn, ns = H._front(running=2, waits_s=[0.5])
        ns._park_attempt_epoch = ns.epoch          # this phase already parked
        assert fn(ns, None, time.time()) is None
        fn, ns = H._front(running=2, waits_s=[0.5], uncached=100)  # nothing over X
        assert fn(ns, None, time.time()) is None
        fn, ns = H._front(running=0, waits_s=[0.5])                 # idle D: its own path
        assert fn(ns, None, time.time()) is None


def test_switch_off_is_the_dwell_as_before():
    a, b = H._on()
    with a, b, envs.FLLIPER_PDFLIP_PARK_NO_DWELL.override(False):
        fn, ns = H._front(running=1, waits_s=[0.0], awake_s=0.31, uncached=25215)
        assert fn(ns, None, time.time()) is None
        assert ns.counters["park_immediate_dwell_holds"] == 1


def _dwell_front(parked_this_phase: bool, awake_s: float = 0.31):
    import collections

    ns = types.SimpleNamespace(
        epoch=4, t_awake=time.time() - awake_s, w_s=45.0, min_dwell_ms=None,
        _park_attempt_epoch=4 if parked_this_phase else -1, counters=collections.Counter(),
        groups={"D": types.SimpleNamespace(outstanding={})},
        _ready_for_d=[object()],                   # K7's idle skip does not apply
        _flip_ledger=lambda g: [], _handoff_in_flight=lambda: 0,
        tp_prefill_max_tokens=8359)
    ns._derived_min_dwell_ms = lambda s, d: (1974.0, "last-flip-D->P")
    return ns


def test_the_min_dwell_after_the_park_does_not_hold(caplog):
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        ok = F.Front._dwell_ok(_dwell_front(True), "D", "P", True, True, 0.0)
    assert ok is True
    assert any("overridden_by=park_no_dwell" in m and "verdict=flip" in m for m in caplog.messages)
    with envs.FLLIPER_PDFLIP_PARK_NO_DWELL.override(False):
        assert F.Front._dwell_ok(_dwell_front(True), "D", "P", True, True, 0.0) is False
    # no park in this phase: the floor stands (fairness alone is no arrival rule)
    assert F.Front._dwell_ok(_dwell_front(False), "D", "P", True, True, 0.0) is False

