"""D-HEALTH (10.10.): P keeps a healthy /health behind a big image WITHOUT the front's vision grace.

HISTORY. Metal 09.10. (M12 dual): a 4096x4096 image request held P's tokenizer
loop -- the loop that serves /health -- while the image was preprocessed; two
/health probes failed and W17 Weg2GroupDead stopped a healthy group. The first
answer was a front-side grace behind an image request on P (4495c6a44a); the
root fix was moving the preprocessing off the loop (mm_tokenize_offload,
ba9ee39476). Metal 10.10. (window jzmxnp, image = the branch with the grace at
0 s): no WEG2-HEALTH/WEG2-HEALTH-BUSY/STOP line, ``WEG2-MM-TOKENIZE ...
offloaded=1`` on P with process_ms=32555 -- the grace was reverted.

THE GUARDED PROPERTY. The front's own W17 verdict (``health_poll_once``, no
grace) stays ``serving`` with streak 0 for a whole preprocessing of the
metal's length, because P's loop answers every probe in < 1 s while the image
is in the offload worker. A diff that puts the preprocessing back on the loop
(or lets the offload stop applying) turns ``test_offloaded_*`` red: then the
second case is what happens -- P's probes time out and W17 stops the group,
the metal STOP of 09.10.

TIME. P's tokenizer loop runs in its own thread (in reality its own process).
The preprocessing is a worker that blocks on a gate the test releases only
after the front polled across ``PREPROCESS_S`` of front time (one poll per
``front_health.POLL_S``, virtual: the polls run back to back). Each probe is
a real round trip into P's loop, bounded by ``PROBE_S`` (the front's real
probe timeout is 8 s; the claim here is the stricter < 1 s).
"""

import asyncio
import collections
import math
import os
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_mm_tokenize_offload_dhealth_1010 import _make_proc, _req, _StallingQwenVL

from sglang.srt.managers import mm_tokenize_offload as mto
from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: F401
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import front_health as FH
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

#: ~24 s of preprocessing (metal 10.10.: process_ms=32555 on P; 09.10.: P silent > 14 s).
PREPROCESS_S = 24.0
PROBE_S = 1.0
#: safety bound for the gate; never reached in a passing or failing run.
GATE_CAP_S = 30.0
P_SID, D_SID = 4711, 4712


class _GatedQwenVL(_StallingQwenVL):
    """The preprocessing lasts until the test opens the gate (deterministic, no wall-clock stall)."""

    def process_and_combine_mm_data(self, base_output, mm_tokens, **kwargs):
        self.entered.set()
        if not self.gate.wait(GATE_CAP_S):
            raise TimeoutError("gate never opened")
        return super().process_and_combine_mm_data(base_output, mm_tokens, **kwargs)


def _front():
    f = object.__new__(front_mod.Front)
    f.groups = {
        "P": front_mod.Group("P", "http://p", P_SID),
        "D": front_mod.Group("D", "http://d", D_SID),
    }
    f.state, f.awake, f.epoch, f.stop, f.tag = "serving", "D", 3, None, "t"
    f.t0 = time.time() - 600
    f.counters = collections.Counter()
    f.queue = collections.deque()
    f._ready_for_d = collections.deque()
    f._batch_gate = asyncio.Event()
    return f


class _PTokenizerLoop:
    """P's tokenizer event loop in a thread: /health and the image request share it."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


def _run_scenario():
    """One image on P, the front polls across PREPROCESS_S; returns what the front saw."""
    proc = _make_proc(0.0)
    proc.__class__ = _GatedQwenVL
    proc.entered, proc.gate = threading.Event(), threading.Event()
    p = _PTokenizerLoop()
    sent, latencies = [], []

    async def image_request():
        async with mto.MmDispatchOrder().request(rid="img-1", is_mm=True) as turn:
            out = await proc.process_mm_data_async(
                image_data=["x"], input_text="<img>", request_obj=_req()
            )
            await turn.dispatch(sent.append, out)
        return turn.offloaded

    async def p_health():
        return True  # the tokenizer's /health handler: answers when the loop runs

    async def probe(self, g, timeout_s):
        if g.name != "P":
            return True
        t0 = time.monotonic()
        try:
            ok = await asyncio.wait_for(asyncio.wrap_future(p.submit(p_health())), PROBE_S)
        except asyncio.TimeoutError:
            ok = False
        latencies.append(time.monotonic() - t0)
        return ok

    f = _front()
    n_polls = math.ceil(PREPROCESS_S / FH.POLL_S) + 1
    still_preprocessing = []
    try:
        image = p.submit(image_request())
        assert proc.entered.wait(10), "the image never reached its preprocessing"
        with mock.patch.dict(os.environ, {FH.ENV: "1"}), mock.patch.object(
            front_mod.Front, "_probe_group_health", probe
        ), mock.patch.object(front_mod, "_sid_alive", lambda sid: True):
            for _ in range(n_polls):
                asyncio.run(f.health_poll_once([]))
                still_preprocessing.append(not image.done())
                if f.state == "STOP":
                    break
        proc.gate.set()
        offloaded = image.result(10)
    finally:
        proc.gate.set()
        p.close()
    return dict(front=f, latencies=latencies, offloaded=offloaded, sent=sent,
                n_polls=n_polls, still_preprocessing=still_preprocessing)


class TestPHealthWithoutGrace(CustomTestCase):
    def _run(self, offload_permitted):
        with mock.patch(
            "sglang.srt.managers.mm_utils.wrap_shm_features", side_effect=lambda o: o
        ), mock.patch.object(mto, "offload_permitted", return_value=offload_permitted):
            return _run_scenario()

    def test_offloaded_preprocessing_keeps_p_serving_without_grace(self):
        r = self._run(True)
        f = r["front"]
        self.assertTrue(r["offloaded"])
        self.assertEqual(len(r["latencies"]), r["n_polls"])
        self.assertTrue(all(r["still_preprocessing"]), "the image finished before the polls ended")
        self.assertLess(max(r["latencies"]), PROBE_S, f"P's /health took {max(r['latencies']):.2f} s")
        self.assertEqual(f.state, "serving")
        self.assertIsNone(f.stop)
        self.assertEqual(f.groups["P"].health_fail_streak, 0)
        self.assertEqual(f.counters["health_busy"], 0)
        self.assertEqual(len(r["sent"]), 1)  # the image reached the scheduler after the gate

    def test_preprocessing_on_the_loop_is_the_w17_stop(self):
        """The old path: P's loop stands, the probes time out, W17 stops the group at streak 2
        (proves the scenario reaches the failure the first case guards against)."""
        r = self._run(False)
        f = r["front"]
        self.assertFalse(r["offloaded"])
        self.assertEqual(f.state, "STOP")
        self.assertIn("W17 Weg2GroupDead", str(f.stop))
        self.assertEqual(f.groups["P"].health_fail_streak, 2)
        self.assertTrue(all(r["still_preprocessing"]))


if __name__ == "__main__":
    unittest.main()
