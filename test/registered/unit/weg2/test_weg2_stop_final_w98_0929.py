# SPDX-License-Identifier: Apache-2.0
"""W98-STOP-FINAL (boot z30u, 2026-09-29 07:14:10Z): a front STOP is final.

MEASURED (front log boot_weg2_dkrnfh91dprsavisadoptstcutvsyncodx2bswre2cutbar1dauer09290654_
4f23714e2d_0929_065429.front.log:5564-5565): the host rate latch fired
``W98 Weg2HostRateLatched`` while a flip was open; ``do_stop`` set STOP and
failed the queue -- and the flip, finishing its legs, wrote ``state = "serving"``
over it. The boot served 136 more requests with ``/weg2/state`` reading
``state=serving stop=W98``: a named, controlled teardown (user order
2026-09-08: controlled teardown, never a kernel kill) was lost without a word.

Guarded here, both directions:
  * a flip asked for WHILE in STOP is refused by name before any effect
    (no clock unlock, no RPC, no state change);
  * the 07:14 race: the flip begins BEFORE the STOP, the STOP falls into its
    legs, the flip ends in STOP -- the legs' facts (awake/epoch) still recorded;
  * the STOP reaches the boot's state directory as event + A5 stop request
    (IPC, never over log lines), so the host writer ends the boot as ``dead``
    with this cause and rc 24; a planned end (``stopping``) is not converted.

Hermetic: no GPU, no HTTP (the group RPCs are stubbed).
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import state_file
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

MIB = 1024 * 1024
W98 = "W98 Weg2HostRateLatched"
W98_DETAIL = ("cushion=1.33 GiB BELOW the floor 1.50 while shmem is STILL RISING "
              "(now=79.81 GiB, shmem=53.96, headroom=16.09 GiB <= relevance 32.00, gaps_seen=0)")


def _front(on_rpc=None):
    f = front_mod.Front(
        prefill="http://p", decode="http://d", awake="D", tag="w98stop",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, flip_min_work_tokens=1,
    )
    f.rpc_calls = []

    async def rpc(g, path, body, timeout):
        f.rpc_calls.append((g.name, path))
        if on_rpc is not None:
            on_rpc(f, g, path, body)
        if path == "/flush_cache":
            return 200, "{}"
        await asyncio.sleep(0.001)
        tags = tuple((body or {}).get("tags", ()))
        return 200, json.dumps({"per_tag": {t: [MIB, 1.0] for t in tags},
                                "critical_path": "rank=0 card=GPU-x ms=1"})

    f.rpc = rpc
    return f


class _Clock:
    """Idle-clock stand-in: records the unlock the flip must NOT reach in STOP."""

    def __init__(self):
        self.calls = []

    async def before_flip(self):
        self.calls.append("unlock")


class _NoStateDir:
    def setUp(self):
        self._saved_sd = os.environ.pop("WEG2_STATE_DIR", None)

    def tearDown(self):
        os.environ.pop("WEG2_STATE_DIR", None)
        if self._saved_sd is not None:
            os.environ["WEG2_STATE_DIR"] = self._saved_sd


class StopIsFinal(_NoStateDir, CustomTestCase):
    def test_a_flip_in_stop_is_refused_before_any_effect(self):
        """RED before: the flip unlocked the clocks, set 'flipping', ran both
        legs and ended 'serving' -- a STOP was undone by the next flip."""
        f = _front()
        clock = _Clock()
        f._idle_clock = clock
        f.do_stop(W98, W98_DETAIL)
        asyncio.run(f.flip("D", "P"))
        self.assertEqual(f.state, "STOP")
        self.assertEqual(f.rpc_calls, [], "no group RPC after a STOP")
        self.assertEqual(clock.calls, [], "the clock unlock is an effect too")
        self.assertEqual((f.awake, f.epoch), ("D", 0))
        self.assertEqual(f.counters["flip_refused_stop"], 1)

    def test_a_stop_inside_the_legs_is_the_flips_end_state(self):
        """RED before (z30u 07:14:10): W98 fired while the legs ran; the flip
        then wrote 'serving' over STOP. The legs themselves did happen, so
        awake/epoch move -- only the return to serving is gone."""

        def latch_fires_in_the_legs(f, g, path, body):
            tags = (body or {}).get("tags") or []
            if path == "/resume_memory_occupation" and any(t != front_mod.KV_TAG for t in tags):
                f.do_stop(W98, W98_DETAIL)   # the rate latch's own call, mid-flip

        f = _front(on_rpc=latch_fires_in_the_legs)
        asyncio.run(f.flip("D", "P"))
        self.assertEqual(f.state, "STOP", "the flip wrote over the STOP")
        self.assertEqual(f.stop.name, W98)
        self.assertEqual((f.awake, f.epoch), ("P", 1))
        self.assertEqual(f.counters["flip_ended_in_stop"], 1)
        # ...and no further flip runs.
        n = len(f.rpc_calls)
        asyncio.run(f.flip("P", "D"))
        self.assertEqual(len(f.rpc_calls), n)
        self.assertEqual(f.state, "STOP")

    def test_no_writer_of_state_but_do_stop_and_the_gate(self):
        """Bookkeeping: every assignment to ``self.state`` in front.py is the
        constructor's, do_stop's, or inside ``_enter_state`` -- a new raw
        ``self.state = ...`` would reopen the 07:14 path."""
        import inspect
        import re

        src = inspect.getsource(front_mod)
        writes = [m.group(0) for m in re.finditer(r"self\.state = [^\n]+", src)]
        self.assertEqual(sorted(writes), sorted([
            'self.state = "serving"  # serving | flipping | STOP',
            'self.state = "STOP"',
            "self.state = new",
        ]), writes)


class StopReachesTheStateDir(_NoStateDir, CustomTestCase):
    def _boot_dir(self, lifecycle):
        root = tempfile.mkdtemp(prefix="w98stop-")
        d = state_file.init(root, "nfw98-boot-20260929T071410Z-abcd", "boot", {})
        for s in ("launching", "loading", "serving"):
            state_file.transition(d, s)
        if lifecycle == "stopping":
            state_file.transition(d, "stopping", cause=state_file.make_cause(
                "stop_file", "operator", "geplantes Ende"))
        os.environ["WEG2_STATE_DIR"] = d
        return d

    def _events(self, d, typ):
        return [e for e in state_file.events(d) if e["type"] == typ]

    def test_the_stop_becomes_the_boots_cause_without_a_log_line(self):
        d = self._boot_dir("serving")
        f = _front()
        f.do_stop(W98, W98_DETAIL)
        ev = self._events(d, "front_stop")
        self.assertEqual(len(ev), 1, state_file.events(d))
        self.assertEqual(ev[0]["code"], "W98_Weg2HostRateLatched")
        self.assertEqual(ev[0]["data"]["front_state_before"], "serving")
        req = state_file.stop_request(d)
        self.assertEqual((req["code"], req["origin"], req["rc"]),
                         ("W98_Weg2HostRateLatched", "front", state_file.RC_STOPPED_BY_WATCHER))
        # the host writer's end of the boot (A5): stop request -> dead, rc 24
        state_file.transition(d, "stopping", if_state=["serving", "flipping"],
                              cause=state_file.make_cause("stop_file", "operator", "Hold-Ende"))
        self.assertEqual(state_file.main(["finish", "--dir", d]), 0)
        st = state_file.read(d)
        self.assertEqual(st["lifecycle"]["state"], "dead")
        self.assertEqual((st["cause"]["code"], st["cause"]["origin"], st["cause"]["rc"]),
                         ("W98_Weg2HostRateLatched", "front", 24))

    def test_a_planned_end_is_not_turned_into_a_death(self):
        """Negative branch: while the boot is already stopping (planned end),
        a STOP from the front's own teardown writes the event, not a request
        -- else ``finish`` would convert a clean stop into dead rc 24 (the
        z30j lesson of the deadman's Phase-2 writer)."""
        d = self._boot_dir("stopping")
        f = _front()
        f.do_stop(W98, W98_DETAIL)
        self.assertEqual(len(self._events(d, "front_stop")), 1)
        self.assertIsNone(state_file.stop_request(d))

    def test_an_earlier_stop_request_keeps_its_cause(self):
        d = self._boot_dir("serving")
        state_file.write_json_atomic(os.path.join(d, "stop_request.json"), {
            "code": "DEADMAN_SPIN", "origin": "deadman", "group": "D", "rank": None,
            "detail_full": "first"})
        f = _front()
        f.do_stop(W98, W98_DETAIL)
        self.assertEqual(state_file.stop_request(d)["code"], "DEADMAN_SPIN")


if __name__ == "__main__":
    unittest.main()
