"""P-RECV-STREAM (--p-recv-stream): the proxy frame's receive off the fence.

Measured reason (#PGAP, docker-acceptance 27b 26.09., i8B/i8drt/n4B): PP1/PP2
idle between two chunks follows the FRAME SIZE, not the host work after the
wait (i8drt PP2: 6.5/6.8 ms while plan alternates 2/15-21 ms), and PP0 -- which
receives no frame -- idles 0.1 ms. The receive of chunk k is issued on the
schedule stream, which carries the device fence on forward k-1, so the transfer
starts only after forward k-1 has ended. See managers/weg2_p_overlap.py.

These tests pin: off = byte-identical launcher output and the stock receive
call; on = the receive runs on a side stream that does NOT depend on forward
k-1, forward k still depends on the receive, every other schedule-stream op
keeps the fence, and every received tensor is recorded on its reader streams.
The dependency tests run on a fake stream model (ops + wait edges), no GPU.
"""
import inspect
import os
import types
import unittest
from contextlib import contextmanager
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler_pp_mixin as ppm
from sglang.srt.managers import weg2_p_overlap as pov
from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")


# --------------------------------------------------------------- fake streams
class _Op:
    def __init__(self, name, deps):
        self.name = name
        self.deps = [d for d in deps if d is not None]


class _Stream:
    def __init__(self, name):
        self.name = name
        self.ops = []

    def last(self):
        return self.ops[-1] if self.ops else None

    def enqueue(self, name, extra=()):
        op = _Op(name, [self.last(), *extra])
        self.ops.append(op)
        return op

    def wait_stream(self, other):
        self.enqueue("wait:" + other.name, [other.last()])

    def wait_event(self, ev):
        self.enqueue("wait:event", [ev.op])


class _Event:
    def __init__(self, dm):
        self.dm = dm
        self.op = None

    def record(self, stream=None):
        self.op = (stream or self.dm.current_stream()).last()


class _DM:
    """torch.cuda's surface as the scheduler uses it: Stream(), stream(s),
    current_stream(), Event(). ``nccl`` stands for ProcessGroupNCCL's internal
    stream: an irecv first waits for the CURRENT stream, and ``work.wait()``
    makes the current stream wait for the NCCL op."""

    def __init__(self):
        self.schedule = _Stream("schedule")
        self.cur = self.schedule
        self.nccl = _Stream("nccl")
        self.created = []

    def Stream(self):
        s = _Stream("side%d" % len(self.created))
        self.created.append(s)
        return s

    def current_stream(self):
        return self.cur

    @contextmanager
    def stream(self, s):
        prev, self.cur = self.cur, s
        try:
            yield
        finally:
            self.cur = prev

    def Event(self):
        return _Event(self)

    def irecv_wait(self, name):
        op = self.nccl.enqueue(name, [self.cur.last()])
        self.cur.enqueue("wait:nccl", [op])
        return op


def _depends(op, target, seen=None):
    seen = set() if seen is None else seen
    if op is target:
        return True
    if id(op) in seen:
        return False
    seen.add(id(op))
    return any(_depends(d, target, seen) for d in op.deps)


class _T:
    """A received CUDA tensor: remembers the streams it was recorded on."""

    is_cuda = True

    def __init__(self):
        self.recorded = []

    def record_stream(self, s):
        self.recorded.append(s)


def _tensors(msg):
    return [v for v in msg.values() if isinstance(v, _T)] if isinstance(msg, dict) else []


def _pipeline(dm, on):
    """forward k-1 launched, fence on the schedule stream (the proxy-send fence
    of the previous pass), receive of frame k, a plan op, launch of forward k."""
    holder = types.SimpleNamespace(
        forward_stream=_Stream("forward"), copy_stream=_Stream("copy")
    )
    fwd = holder.forward_stream
    fwd_prev = fwd.enqueue("fwd k-1")
    launch_event = dm.Event()
    launch_event.record(fwd)
    dm.schedule.wait_event(launch_event)
    frame = {"hidden_states": _T(), "residual": _T(), "__msg_type__": "proxy"}
    ops = {}

    def recv(on_wire=None):
        ops["recv"] = dm.irecv_wait("recv frame k")
        if on_wire is not None:
            on_wire(frame)
        return frame

    if on:
        out = pov.recv_off_fence(holder, recv, dm, cuda_tensors=_tensors)
    else:
        out = recv()
    plan_op = dm.schedule.enqueue("plan op")
    fwd.wait_stream(dm.schedule)
    fwd_k = fwd.enqueue("fwd k")
    return holder, out, frame, fwd_prev, ops["recv"], plan_op, fwd_k


class TestDependency(unittest.TestCase):
    def test_off_the_receive_is_queued_behind_the_previous_forward(self):
        # today's code: the measured idle -- the frame moves after fwd k-1
        dm = _DM()
        _, _, _, fwd_prev, recv, _, fwd_k = _pipeline(dm, on=False)
        self.assertTrue(_depends(recv, fwd_prev))
        self.assertTrue(_depends(fwd_k, recv))
        self.assertEqual(dm.created, [])

    def test_on_the_receive_does_not_wait_for_the_previous_forward(self):
        dm = _DM()
        _, out, frame, fwd_prev, recv, plan_op, fwd_k = _pipeline(dm, on=True)
        self.assertIs(out, frame)
        self.assertFalse(_depends(recv, fwd_prev))
        # correctness: forward k still reads the frame only after it landed ...
        self.assertTrue(_depends(fwd_k, recv))
        self.assertTrue(_depends(fwd_k, fwd_prev))
        # ... and every other schedule-stream op keeps the fence on fwd k-1
        self.assertTrue(_depends(plan_op, fwd_prev))
        self.assertIs(dm.current_stream(), dm.schedule)

    def test_every_wire_tensor_is_recorded_on_its_reader_streams(self):
        dm = _DM()
        holder, _, frame, *_ = _pipeline(dm, on=True)
        want = [dm.schedule, holder.forward_stream, holder.copy_stream]
        for key in ("hidden_states", "residual"):
            self.assertEqual(frame[key].recorded, want)

    def test_a_stashed_message_is_recorded_too(self):
        # a message of another kind that came off the wire in the same call is
        # stashed and read later -- its block is in the side pool as well
        dm = _DM()
        holder = types.SimpleNamespace(forward_stream=_Stream("forward"))
        stashed = {"next_token_ids": _T(), "__msg_type__": "output"}
        frame = {"hidden_states": _T(), "__msg_type__": "proxy"}

        def recv(on_wire):
            on_wire(stashed)
            on_wire(frame)
            return frame

        pov.recv_off_fence(holder, recv, dm, cuda_tensors=_tensors)
        self.assertEqual(stashed["next_token_ids"].recorded,
                         [dm.schedule, holder.forward_stream])

    def test_side_stream_made_once_and_restored_on_error(self):
        dm = _DM()
        holder = types.SimpleNamespace()
        pov.recv_off_fence(holder, lambda w: {}, dm, cuda_tensors=_tensors)
        pov.recv_off_fence(holder, lambda w: {}, dm, cuda_tensors=_tensors)
        self.assertEqual(len(dm.created), 1)
        self.assertEqual(holder._weg2_recv_stream_n, 2)

        def boom(w):
            raise RuntimeError("wire")

        with self.assertRaises(RuntimeError):
            pov.recv_off_fence(holder, boom, dm, cuda_tensors=_tensors)
        self.assertIs(dm.current_stream(), dm.schedule)

    def test_recv_runs_with_the_side_stream_current(self):
        dm = _DM()
        seen = []
        pov.recv_off_fence(types.SimpleNamespace(),
                           lambda w: seen.append(dm.current_stream()) or {},
                           dm, cuda_tensors=_tensors)
        self.assertEqual(seen, dm.created)


# ------------------------------------------------------------- the wiring
class _Env:
    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self.old = os.environ.get(pov.P_RECV_STREAM_ENV)
        os.environ.pop(pov.P_RECV_STREAM_ENV, None)
        if self.on:
            os.environ[pov.P_RECV_STREAM_ENV] = "1"

    def __exit__(self, *a):
        os.environ.pop(pov.P_RECV_STREAM_ENV, None)
        if self.old is not None:
            os.environ[pov.P_RECV_STREAM_ENV] = self.old


def _stub(allgather=False):
    calls = []
    frame = {"hidden_states": object()}

    def typed(**kw):
        calls.append(kw)
        if kw.get("on_wire") is not None:
            kw["on_wire"](frame)
        return dict(frame)

    s = types.SimpleNamespace(
        pp_group=types.SimpleNamespace(is_first_rank=False),
        _pp_gapped_wire=False,
        require_attn_tp_allgather=allgather,
        attn_tp_group="TPG",
        device_module=_DM(),
        forward_ct=7,
        _pp_wait_for_proxy_readiness=lambda mb_id: None,
        _pp_recv_typed_dict=typed,
    )
    return s, calls


class TestProxyRecvWiring(unittest.TestCase):
    def _call(self, s):
        with mock.patch.object(ppm, "pp_pass_retraction_reason_of", return_value=None), \
                mock.patch.object(ppm, "pp_flip_epoch_of", return_value=0), \
                mock.patch.object(ppm, "_999_geom", return_value=None):
            return ppm.SchedulerPPMixin._pp_recv_proxy_tensors(s, -1)

    def test_off_is_the_stock_call(self):
        s, calls = _stub()
        with _Env(False), mock.patch.object(pov, "recv_off_fence") as rof:
            out = self._call(s)
        rof.assert_not_called()
        self.assertEqual(calls, [{"expected_kind": "proxy", "all_gather_group": None}])
        self.assertIsInstance(out, ppm.PPProxyTensors)
        self.assertEqual(s.device_module.created, [])

    def test_on_goes_through_the_side_stream(self):
        s, calls = _stub()
        with _Env(True):
            out = self._call(s)
        self.assertIsInstance(out, ppm.PPProxyTensors)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["expected_kind"], "proxy")
        self.assertIsNone(calls[0]["all_gather_group"])
        self.assertTrue(callable(calls[0]["on_wire"]))
        self.assertEqual(len(s.device_module.created), 1)
        self.assertEqual(s._weg2_recv_stream_n, 1)

    def test_on_with_attention_allgather_keeps_the_stock_path(self):
        s, calls = _stub(allgather=True)
        with _Env(True), mock.patch.object(pov, "recv_off_fence") as rof:
            self._call(s)
        rof.assert_not_called()
        self.assertEqual(calls, [{"expected_kind": "proxy", "all_gather_group": "TPG"}])

    def test_only_the_proxy_receive_is_moved(self):
        # the output receive, the row-probe drain and the disagg/admission
        # receives keep the schedule stream: only _pp_recv_proxy_tensors calls
        # recv_off_fence
        src = inspect.getsource(ppm)
        self.assertEqual(src.count("_pov.recv_off_fence("), 1)
        body = inspect.getsource(ppm.SchedulerPPMixin._pp_recv_proxy_tensors)
        self.assertIn("_pov.recv_off_fence(", body)
        self.assertIn("_pov.p_recv_stream_on()", body)


class TestTypedDictOnWire(unittest.TestCase):
    def _self(self):
        s = types.SimpleNamespace(pp_group="G", _pp_boundary_stats=lambda: None,
                                  _pp_flip_bump_consumed=lambda ch: None)
        return s

    def test_on_wire_sees_each_wire_message_and_default_none_is_inert(self):
        msgs = [{"__msg_type__": "output", "a": 1}, {"__msg_type__": "proxy", "b": 2}]

        def fake_recv(group, kind, src=None, all_gather_group=None, on_message=None,
                      accept=None, on_reject=None):
            for m in msgs:
                on_message(m)
            return msgs[-1]

        seen = []
        with mock.patch.object(ppm, "recv_typed_tensor_dict", fake_recv), \
                mock.patch.object(ppm, "_pp_flip_bump_kind", lambda *a, **k: None):
            s = self._self()
            ppm.SchedulerPPMixin._pp_recv_typed_dict(
                s, expected_kind="output", on_wire=seen.append)
            self.assertEqual(seen, msgs)
            # default: no hook, same result
            out = ppm.SchedulerPPMixin._pp_recv_typed_dict(s, expected_kind="output")
            self.assertIs(out, msgs[-1])


class TestLauncherSwitch(unittest.TestCase):
    def test_off_adds_nothing(self):
        self.assertEqual(launcher.p_recv_stream_env(False), {})
        self.assertEqual(launcher.p_recv_stream_lines(False), [])

    def test_flag_exists_and_default_off(self):
        base = ["--tree", "/t", "--tag", "t"]
        self.assertFalse(launcher.build_parser().parse_args(base).p_recv_stream)
        self.assertTrue(
            launcher.build_parser().parse_args(base + ["--p-recv-stream"]).p_recv_stream)

    def test_on_sets_the_variable_the_runtime_reads(self):
        self.assertEqual(launcher.p_recv_stream_env(True),
                         {"SGLANG_WEG2_P_RECV_STREAM": "1"})
        self.assertEqual(len(launcher.p_recv_stream_lines(True)), 1)
        with _Env(True):
            self.assertTrue(pov.p_recv_stream_on())
        with _Env(False):
            self.assertFalse(pov.p_recv_stream_on())

    def test_env_p_is_the_only_group_that_gets_it(self):
        src = inspect.getsource(launcher.main)
        i = src.index("env_p.update(p_recv_stream_env(")
        self.assertNotIn("env_d.update(p_recv_stream_env(", src)
        self.assertLess(src.index('group="P"'), i)

    def test_overlap_env_is_unchanged(self):
        # --p-host-overlap does not switch it on implicitly
        self.assertNotIn(pov.P_RECV_STREAM_ENV, launcher.p_host_overlap_env(True, True))


if __name__ == "__main__":
    unittest.main()
