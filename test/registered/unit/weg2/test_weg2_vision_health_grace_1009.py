# SPDX-License-Identifier: Apache-2.0
"""VISION-WEIGHTS (metal 09.10., M12 dual, boot ...dualvwweights...155005): a 4096x4096 image request kept P's
/health silent for >14 s (decode + patchify in the tokenizer process) BEFORE the vision stage ran; the front's health
poller counted two failures and stopped a healthy group (W17), although the stage then answered W105b by name.

The front knows it routed an image to P (like it knows ``flipping``): a silent /health of P is not a failure for
VISION_HEALTH_GRACE_S behind that request. Past the grace, without an image request, with a dead process or a held
rank, the verdict is the old one."""

from __future__ import annotations

import asyncio
import collections
import os
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import front_health as FH  # noqa: E402

P_SID, D_SID = 4711, 4712


def _front():
    f = object.__new__(front_mod.Front)
    f.groups = {"P": front_mod.Group("P", "http://p", P_SID), "D": front_mod.Group("D", "http://d", D_SID)}
    f.state, f.awake, f.epoch, f.stop, f.tag = "serving", "D", 3, None, "t"
    f.t0 = time.time() - 600
    f.counters = collections.Counter()
    f.queue = collections.deque()
    f._ready_for_d = collections.deque()
    f._batch_gate = asyncio.Event()
    return f


def _poll(f, http, alive=None):
    alive = alive or {P_SID: True, D_SID: True}

    async def probe(self, g, timeout_s):
        return http[g.name]

    with mock.patch.object(front_mod.Front, "_probe_group_health", probe), \
         mock.patch.object(front_mod, "_sid_alive", lambda sid: alive.get(sid, True)):
        asyncio.run(f.health_poll_once([]))


class VisionHealthGrace(unittest.TestCase):
    def test_the_oldest_image_request_age(self):
        self.assertIsNone(front_mod.vision_inflight_age({}, 100.0))
        self.assertEqual(front_mod.vision_inflight_age({1: 90.0, 2: 70.0}, 100.0), 30.0)

    def test_silent_p_health_behind_an_image_request_does_not_stop_the_group(self):
        """The metal case: two silent polls while an image request is on P -> serving, streak 0."""
        with mock.patch.dict(os.environ, {FH.ENV: "1"}):
            f = _front()
            f._vision_inflight = {1: time.monotonic() - 14.0}
            for _ in range(3):
                _poll(f, {"P": False, "D": True})
        self.assertEqual(f.state, "serving")
        self.assertEqual(f.groups["P"].health_fail_streak, 0)
        self.assertGreaterEqual(f.counters["health_busy"], 3)

    def test_without_an_image_request_the_old_verdict_stands(self):
        with mock.patch.dict(os.environ, {FH.ENV: "1"}):
            f = _front()
            _poll(f, {"P": False, "D": True})
            _poll(f, {"P": False, "D": True})
        self.assertEqual(f.state, "STOP")

    def test_past_the_grace_a_silent_p_counts_again(self):
        """A stage that hangs is still a dead group: the grace is a bound, not an exemption."""
        with mock.patch.dict(os.environ, {FH.ENV: "1"}):
            f = _front()
            f._vision_inflight = {1: time.monotonic() - front_mod.VISION_HEALTH_GRACE_S - 5.0}
            _poll(f, {"P": False, "D": True})
            _poll(f, {"P": False, "D": True})
        self.assertEqual(f.state, "STOP")

    def test_a_dead_p_process_is_fatal_despite_an_image_request(self):
        with mock.patch.dict(os.environ, {FH.ENV: "1"}):
            f = _front()
            f._vision_inflight = {1: time.monotonic() - 5.0}
            _poll(f, {"P": False, "D": True}, alive={P_SID: False, D_SID: True})
            _poll(f, {"P": False, "D": True}, alive={P_SID: False, D_SID: True})
        self.assertEqual(f.state, "STOP")

    def test_the_grace_is_for_group_p_only(self):
        with mock.patch.dict(os.environ, {FH.ENV: "1"}):
            f = _front()
            f._vision_inflight = {1: time.monotonic() - 5.0}
            _poll(f, {"P": True, "D": False})
            _poll(f, {"P": True, "D": False})
        self.assertEqual(f.state, "STOP")


if __name__ == "__main__":
    unittest.main()
