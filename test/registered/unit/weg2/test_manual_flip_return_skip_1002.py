# SPDX-License-Identifier: Apache-2.0
"""MANUAL-FLIP RETURN-SKIP (02.10.2026): the probe's round trip leaves P awake
when P-bound work arrived during its first half.

L15 boot dac8b62b8c (...10021440_dac8b62b8c_1002_144019.front.log): POST
/weg2/flip ran D->P; 14:43:03.793 LONG weg2-2-4 (133401 uncached) arrived
during it; 03.935 the handler still ran the return P->D with queue=1; the LONG
waited 2.2 s and got its own D->P at 06.006. ``Front._manual_return_skip_reason``,
marker ``WEG2 MANUAL-FLIP RETURN-SKIP reason=...``, switch
``SGLANG_WEG2_MANUAL_FLIP_RETURN_SKIP``. Hermetic, CPU; red_* red on 043f3c3d12.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402


def _front(arrive=None, p_out=None):
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0)
    f.state = "serving"
    f.tp_prefill_max_tokens = 6400
    f._manual_flip_refusal = lambda: None
    flips = []

    async def flip(src, dst):
        flips.append((src, dst))
        f.awake = dst
        if (src, dst) == ("D", "P"):
            if arrive is not None:
                f.queue.append(arrive)          # arrives while the first half runs
            if p_out:
                f.groups["P"].outstanding.update(p_out)

    f.flip = flip
    f.state_dict = lambda: {"awake": f.awake}
    f._kick_controller = lambda why: None
    return f, flips


def _long(uncached=133401):
    return types.SimpleNamespace(rid="weg2-2-4", est_uncached=uncached, t_arrive=time.time())


def test_red_a_long_arriving_in_the_first_half_keeps_p_awake(caplog):
    f, flips = _front(arrive=_long())
    with caplog.at_level(logging.INFO):
        r = asyncio.run(f.handle_manual_flip(None))
    assert r.status == 200
    assert flips == [("D", "P")] and f.awake == "P"
    assert "WEG2 MANUAL-FLIP RETURN-SKIP reason=queue_p rid=weg2-2-4" in caplog.text
    assert f.counters["manual_flip_return_skip"] == 1


def test_red_p_outstanding_keeps_p_awake(caplog):
    f, flips = _front(p_out={"weg2-2-9": time.time()})
    with caplog.at_level(logging.INFO):
        asyncio.run(f.handle_manual_flip(None))
    assert flips == [("D", "P")]
    assert "reason=p_outstanding n=1" in caplog.text


def test_no_p_work_the_round_trip_returns_as_before():
    f, flips = _front()
    asyncio.run(f.handle_manual_flip(None))
    assert flips == [("D", "P"), ("P", "D")] and f.awake == "D"


def test_a_short_request_for_d_is_no_p_work():
    f, flips = _front(arrive=_long(uncached=200))
    asyncio.run(f.handle_manual_flip(None))
    assert flips == [("D", "P"), ("P", "D")]


def test_switch_off_returns_as_before():
    f, flips = _front(arrive=_long())
    with envs.SGLANG_WEG2_MANUAL_FLIP_RETURN_SKIP.override(False):
        asyncio.run(f.handle_manual_flip(None))
    assert flips == [("D", "P"), ("P", "D")]
