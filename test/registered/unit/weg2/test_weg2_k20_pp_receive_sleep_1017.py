"""K20 (17.10.): no PP receive stays posted into the sleep.

Beleg: boot 1016_172204, P log: the last told-ack of rid weg2-178-2512 re-armed PP0's standing receive at 22:36:54Z, P
slept 22:36:57Z, and at 00:36:54Z (exactly 7200 s) PP0 died on ``PF told-ack receive[pp1] ... Timed out waiting
7200000ms``. Gloo's clock on a posted receive is the group timeout; the standing frame is never cancelled.

The network is faked in memory (posted receives, a virtual gloo clock that raises at 7200 s after the post, as the real
one does). SOLL R11 / Rang-Einigkeit: PP0 decides, followers only report (bye).
"""
import ast
import unittest.mock
import os
import threading
import time
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.distributed as dist

from sglang.srt.managers import weg2_told_fallback as wtf
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-weg2-unit")

GLOO_TIMEOUT_S = 7200.0


class FakeNet:
    def __init__(self):
        self.now = 0.0  # virtual gloo clock
        self.sender = 1
        self.posted = {}  # (src, tag) -> [Work]
        self.pending = {}  # (src, tag) -> [tensor] sent before the receive was posted
        self.lock = threading.Lock()

    def irecv(self, tensor, src, group=None, tag=0):
        w = _Work(self, tensor, self.now)
        with self.lock:
            pend = self.pending.get((src, tag))
            if pend:
                w.tensor.copy_(pend.pop(0).reshape(w.tensor.shape))
                w.done.set()
            else:
                self.posted.setdefault((src, tag), []).append(w)
        return w

    def isend(self, tensor, dst, group=None, tag=0):
        key = (self.sender, tag)
        with self.lock:
            q = self.posted.get(key) or []
            w = q.pop(0) if q else None
            if w is None:
                self.pending.setdefault(key, []).append(tensor.clone())
        if w is not None:
            w.tensor.copy_(tensor.reshape(w.tensor.shape))
            w.done.set()
        sw = _Work(self, tensor, self.now)
        sw.done.set()
        return sw


class _Work:
    def __init__(self, net, tensor, t_post):
        self.net, self.tensor, self.t_post = net, tensor, t_post
        self.done = threading.Event()

    def wait(self, *a, **k):
        while not self.done.wait(0.002):
            if self.net.now - self.t_post >= GLOO_TIMEOUT_S:
                raise RuntimeError("Timed out waiting 7200000ms for recv operation to complete")
        return True


def _sched(rank, size=3):
    ps = types.SimpleNamespace(pp_rank=rank, pp_size=size)
    return types.SimpleNamespace(
        ps=ps, world_group=types.SimpleNamespace(cpu_group=object()), __dict__={}
    )


class K20Case(unittest.TestCase):
    def setUp(self):
        self.net = FakeNet()
        self._p = [
            unittest.mock.patch.object(dist, "irecv", self.net.irecv),
            unittest.mock.patch.object(dist, "isend", self.net.isend),
        ]
        for p in self._p:
            p.start()
        os.environ["SGLANG_WEG2_PF_TOLD_ACK_TIMEOUT_S"] = "2"
        self.world = [0, 1, 2]
        self.s0, self.s1, self.s2 = _sched(0), _sched(1), _sched(2)
        for s in (self.s0, self.s1, self.s2):
            s.__dict__["_weg2_fb_channel"] = wtf.GlooAckChannel(None, s.ps.pp_rank, self.world)
            s.__dict__["_weg2_pr_channel"] = wtf.GlooAckChannel(
                None, s.ps.pp_rank, self.world, tag=wtf.WEG2_TOLD_ACK_TAG + 1, label="pp-room"
            )

    def tearDown(self):
        for p in self._p:
            p.stop()
        os.environ.pop("SGLANG_WEG2_PF_TOLD_ACK_TIMEOUT_S", None)
        os.environ.pop("SGLANG_WEG2_PF_ACK_HEARTBEAT_S", None)

    def _ack(self, s, rid):
        self.net.sender = s.ps.pp_rank
        ch = s.__dict__["_weg2_fb_channel"]
        ack = wtf.Weg2ToldReadAck(rank=s.ps.pp_rank, seq=1, reads=[(rid, 100)])
        assert ch.send_nowait(ack)

    def _drain(self, ch, n=1):
        got = []
        t_end = time.time() + 3
        while len(got) < n and time.time() < t_end:
            got += ch.harvest()
            time.sleep(0.005)
        return got

    def _quiesce_all(self):
        out = {}

        def run(s):
            self.net.sender = s.ps.pp_rank
            out[s.ps.pp_rank] = wtf.sleep_quiesce(s)

        # followers first (each sends its byes with the sender id set), PP0 collects
        for s in (self.s1, self.s2):
            run(s)
        run(self.s0)
        return out

    def test_open_receive_at_sleep_is_zero_and_survives_gloo_timeout(self):
        """K20: last ack re-arms the standing receive; sleep; 7201 s later PP0 must NOT die."""
        ch0 = self.s0.__dict__["_weg2_fb_channel"]
        self._ack(self.s1, "weg2-178-2512")
        self._ack(self.s2, "weg2-178-2512")
        self.assertEqual(len(self._drain(ch0, 2)), 2)
        ch0.harvest()  # re-arms the standing frames (what ACK-DRAIN does every pass)
        self.assertEqual(ch0.open_receives(), 2)
        out = self._quiesce_all()
        self.assertEqual(out[0], 0, "PP-RECEIVE-CENSUS at=sleep open must be 0")
        for _ in range(3):  # passes that still run while asleep must not re-post the receives
            ch0.harvest()
            self.s0.__dict__["_weg2_pr_channel"].harvest()
        self.assertEqual(ch0.open_receives(), 0)
        self.net.now += GLOO_TIMEOUT_S + 1
        ch0.harvest()
        self.assertEqual(ch0.errors, 0, "PP0's standing receive hit gloo's 7200 s timeout across the sleep")
        self.assertEqual(self.s0.__dict__["_weg2_pr_channel"].errors, 0)

    def test_without_the_fix_the_receive_dies(self):
        """The negative control: no quiesce -> exactly the 00:36:54Z death."""
        ch0 = self.s0.__dict__["_weg2_fb_channel"]
        ch0.harvest()
        self.net.now += GLOO_TIMEOUT_S + 1
        time.sleep(0.05)
        ch0.harvest()
        self.assertGreater(ch0.errors, 0)

    def test_wake_rearms_and_acks_flow_again(self):
        ch0 = self.s0.__dict__["_weg2_fb_channel"]
        ch0.harvest()
        self._quiesce_all()
        wtf.wake(self.s0)
        ch0.harvest()
        self.assertEqual(ch0.open_receives(), 2)
        self._ack(self.s1, "rid-after-wake")
        got = self._drain(ch0, 1)
        self.assertEqual(got[0].reads[0][0], "rid-after-wake")

    def test_heartbeat_restarts_the_gloo_clock_while_awake(self):
        os.environ["SGLANG_WEG2_PF_ACK_HEARTBEAT_S"] = "0.05"
        ch0 = self.s0.__dict__["_weg2_fb_channel"]
        ch1 = self.s1.__dict__["_weg2_fb_channel"]
        ch0.harvest()
        self.net.now = 7000.0
        time.sleep(0.1)
        self.net.sender = 1
        ch1.pump()  # idle follower: due -> beat completes PP0's standing receive
        self.net.sender = 2
        self.s2.__dict__["_weg2_fb_channel"].pump()
        t_end = time.time() + 3
        while ch0.beats_taken < 2 and time.time() < t_end:
            ch0.harvest()
            time.sleep(0.005)
        self.assertEqual(ch0.beats_taken, 2)
        ch0.harvest()  # re-armed at virtual 7000
        self.net.now = 7000.0 + GLOO_TIMEOUT_S - 1
        time.sleep(0.05)
        ch0.harvest()
        self.assertEqual(ch0.errors, 0)

    def test_call_edges_in_the_sleep_and_wake_legs(self):
        """Aufrufkante: the sleep leg calls sleep_quiesce, the wake leg calls wake (a helper nobody calls protects nothing)."""
        here = os.path.dirname(__file__)
        root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
        src = open(os.path.join(root, "python/sglang/srt/managers/scheduler_components/weight_updater.py")).read()
        tree = ast.parse(src)
        calls = {}
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.name in ("release_memory_occupation", "resume_memory_occupation"):
                calls[fn.name] = {n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertIn("sleep_quiesce", calls["release_memory_occupation"])
        self.assertIn("wake", calls["resume_memory_occupation"])
        pp = open(os.path.join(root, "python/sglang/srt/managers/scheduler_pp_mixin.py")).read()
        self.assertIn("channel.pump()", pp)  # follower heartbeat edge of the pp-room stream


if __name__ == "__main__":
    unittest.main()
