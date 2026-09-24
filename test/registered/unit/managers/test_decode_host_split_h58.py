"""fnFL2 H58: DECODE-HOST-SPLIT -- H49's ``host_other_ms`` split into the
device waits inside ``run_batch`` and the named host spans, plus the GPU idle
the host caused, read from six stream events per round with ``query()``.

Hermetic, CPU only. The counters and the round boundary run directly and
through the real ``DecodeRoundLog``; the device half runs on scripted events
whose ``synchronize()`` fails the test; the wiring into the forward path is
driven through the real functions where a CPU stub can carry them
(``resolve_seq_lens_cpu``, ``finish_ple_verify_stage``, the barlink
broadcast, the BAR1 forced wait) and pinned structurally where it cannot
(the scheduler loop, ``EAGLEWorkerV2``). Numbers are x162's D-TP0 steady
state (24.09.): round 26.0 ms, verify gpu-ms 21.2, PLE sync 1.3 / stage 1.2.
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
import time
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import decode_host_split as dhs
from sglang.srt.managers.scheduler_components.decode_host_split import (
    MARK_DEXT_BEGIN,
    MARK_DEXT_END,
    MARK_DRAFT_BEGIN,
    MARK_DRAFT_END,
    MARK_VERIFY_END,
    MARK_VERIFY_LAUNCH,
    DecodeHostSplit,
    DeviceGapProbe,
    SplitCounters,
)
from sglang.srt.managers.scheduler_components.host_round_cost import (
    HostCostCounters,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MOD = "sglang.srt.managers.scheduler_components.decode_host_split"
#: the ``python/`` root the module under test was imported from
PY = os.path.normpath(os.path.join(os.path.dirname(dhs.__file__), "..", "..", "..", ".."))


def _src(rel: str) -> str:
    with open(os.path.join(PY, rel)) as f:
        return f.read()


def _between(text: str, a: str, b: str) -> str:
    i = text.index(a)
    return textwrap.dedent(text[i : text.index(b, i + len(a))])


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _capture(test: unittest.TestCase, name: str = MOD) -> _Capture:
    cap = _Capture()
    lg = logging.getLogger(name)
    lg.addHandler(cap)
    lg.setLevel(logging.INFO)
    test.addCleanup(lg.removeHandler, cap)
    return cap


class _Clock:
    """A scripted perf_counter: ``advance(ms)`` moves it."""

    def __init__(self) -> None:
        self.t = 1000.0

    def advance(self, ms: float) -> None:
        self.t += ms / 1000.0

    def perf_counter(self) -> float:
        return self.t


# -- the device half on scripted events -----------------------------------


class _Dev:
    """The device clock: an event completes once the device reached it."""

    def __init__(self) -> None:
        self.now = 0.0  # ms
        self.at = 0.0  # ms, where the next record() lands
        self.syncs = 0
        self.capturing = False


class _Ev:
    def __init__(self, dev: _Dev) -> None:
        self._dev = dev
        self.t = None

    def record(self) -> None:
        self.t = self._dev.at

    def query(self) -> bool:
        return self.t is not None and self._dev.now >= self.t

    def elapsed_time(self, other: "_Ev") -> float:
        assert self.query() and other.query(), "elapsed_time on an unfinished event"
        return other.t - self.t

    def synchronize(self) -> None:  # pragma: no cover - must never run
        self._dev.syncs += 1
        raise AssertionError("the instrument synchronized the device")


class _Backend:
    def __init__(self, dev: _Dev) -> None:
        self.dev = dev

    def event(self) -> _Ev:
        return _Ev(self.dev)

    def is_capturing(self) -> bool:
        return self.dev.capturing


#: one x162-like round on TP0's stream, ms between the six marks, then the
#: gap to the next round's draft
ROUND = dict(draft=1.2, gap_ple=1.5, verify=21.2, accept=0.3, dext=0.7, gap_round=1.1)


def _run_round(probe: DeviceGapProbe, dev: _Dev, rid: int, r=ROUND, skip=()) -> None:
    probe.begin(rid)
    steps = (
        (MARK_DRAFT_BEGIN, 0.0),
        (MARK_DRAFT_END, r["draft"]),
        (MARK_VERIFY_LAUNCH, r["gap_ple"]),
        (MARK_VERIFY_END, r["verify"]),
        (MARK_DEXT_BEGIN, r["accept"]),
        (MARK_DEXT_END, r["dext"]),
    )
    for point, dt in steps:
        dev.at += dt
        if point not in skip:
            probe.mark(point)
    dev.at += r["gap_round"]


class DeviceGapProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dev = _Dev()
        self.probe = DeviceGapProbe(backend=_Backend(self.dev))

    def test_rounds_are_read_late_in_order_and_never_synchronized(self):
        for rid in (10, 11, 12):
            _run_round(self.probe, self.dev, rid)
        self.probe.end()
        # the device has not reached round 10's end yet: nothing is read
        self.dev.now = 20.0
        self.assertEqual(self.probe.harvest(), [])
        # it has finished everything
        self.dev.now = 1e9
        got = self.probe.harvest()
        self.assertEqual([g.round_id for g in got], [10, 11, 12])
        g = got[1]
        self.assertAlmostEqual(g.draft_ms, 1.2, places=6)
        self.assertAlmostEqual(g.gap_ple_ms, 1.5, places=6)
        self.assertAlmostEqual(g.verify_ms, 21.2, places=6)
        self.assertAlmostEqual(g.accept_ms, 0.3, places=6)
        self.assertAlmostEqual(g.dext_ms, 0.7, places=6)
        # previous round's dext end -> this draft begin
        self.assertAlmostEqual(g.gap_round_ms, 1.1, places=6)
        self.assertAlmostEqual(g.idle_ms, 2.6, places=6)
        # the first read round has no predecessor to measure against
        self.assertIsNone(got[0].gap_round_ms)
        self.assertIsNone(got[0].idle_ms)
        self.assertEqual(self.dev.syncs, 0)
        self.assertEqual(self.probe.unread, 0)

    def test_the_round_still_taking_marks_is_never_read(self):
        _run_round(self.probe, self.dev, 20)
        self.probe.begin(21)  # round 21 open, no marks yet
        self.probe.mark(MARK_DRAFT_BEGIN)
        self.dev.now = 1e9
        got = self.probe.harvest()
        self.assertEqual([g.round_id for g in got], [20])
        self.assertEqual(self.probe.unread, 0)

    def test_a_round_missing_a_mark_is_dropped_and_counted(self):
        _run_round(self.probe, self.dev, 30)
        _run_round(self.probe, self.dev, 31, skip=(MARK_VERIFY_END,))
        _run_round(self.probe, self.dev, 32)
        self.probe.end()
        self.dev.now = 1e9
        got = self.probe.harvest()
        self.assertEqual([g.round_id for g in got], [30, 32])
        self.assertEqual(self.probe.unread, 1)
        # 32's predecessor was not read: no gap is invented across it
        self.assertIsNone(got[1].gap_round_ms)

    def test_non_consecutive_rounds_get_no_round_gap(self):
        _run_round(self.probe, self.dev, 40)
        _run_round(self.probe, self.dev, 45)  # a prefill batch in between
        self.probe.end()
        self.dev.now = 1e9
        got = self.probe.harvest()
        self.assertEqual([g.round_id for g in got], [40, 45])
        self.assertIsNone(got[1].gap_round_ms)

    def test_ring_overwrite_counts_unread_and_capture_records_nothing(self):
        for rid in range(DeviceGapProbe.DEPTH + 3):
            _run_round(self.probe, self.dev, 100 + rid)
        self.assertEqual(self.probe.unread, 3)
        self.dev.capturing = True
        self.probe.begin(500)
        self.probe.mark(MARK_DRAFT_BEGIN)  # inert
        self.dev.capturing = False
        self.probe.end()
        self.dev.now = 1e9
        got = self.probe.harvest()
        self.assertEqual(len(got), DeviceGapProbe.DEPTH)
        self.assertNotIn(500, [g.round_id for g in got])


# -- the host half ----------------------------------------------------------


def _x162_round(c: SplitCounters, h49: HostCostCounters, *, ctl_wait=16.5) -> None:
    """The spans one x162-like NF TP0 round adds (ms): the forced BAR1 wait
    sits inside the draft span (the draft-token broadcast's abort check),
    seq_wait is 0 (NF's backends opt out of the CPU seq_lens mirror)."""
    c.recv_ms += 0.1
    c.sched_ms += 1.2
    h49.hc_ms += 0.4  # inside sched
    c.seq_wait_ms += 0.0
    c.draft_ms += 0.9 + ctl_wait
    c.ctl_wait_ms += ctl_wait  # inside draft
    c.ctl_wait_n += 1
    c.bcast_ms += 0.2  # inside draft + accept
    c.bcast_n += 2
    c.vprep_ms += 0.8
    c.ple_sync_ms += 1.3
    c.ple_stage_ms += 1.2
    c.launch_ms += 0.3
    c.accept_ms += 0.4
    c.verify_ms += 4.0  # = vprep + ple_sync + ple_stage + launch + accept
    c.dext_ms += 0.8
    c.result_ms += 0.6
    h49.result_wait_ms += 0.0  # inside result


class DecodeHostSplitTest(unittest.TestCase):
    def _split(self, period=0, probe=None):
        c, h49 = SplitCounters(), HostCostCounters()
        return DecodeHostSplit(rank=0, period=period, counters=c, h49=h49, probe=probe), c, h49

    def test_round_terms_sync_host_work_and_other(self):
        s, c, h49 = self._split()
        s.on_round_open(round_id=10, mono=1.000)
        _x162_round(c, h49)
        s.on_round_open(round_id=11, mono=1.026)
        r = s.last
        self.assertEqual(r.round_id, 10)
        self.assertAlmostEqual(r.wall_ms, 26.0, places=6)
        # the host was blocked on the device for ctl_wait + ple_sync
        self.assertAlmostEqual(r.sync_ms, 17.8, places=6)
        self.assertAlmostEqual(r.host_work_ms, 8.2, places=6)
        # wall - recv - sched - seq_wait - draft - verify - dext - result
        self.assertAlmostEqual(r.other_ms, 1.9, places=6)
        self.assertAlmostEqual(r.hicache_ms, 0.4, places=6)
        self.assertEqual(r.bcast_n, 2)
        self.assertEqual(r.ctl_wait_n, 1)
        s.on_round_end(mono=1.050)
        self.assertEqual(s.last.round_id, 11)
        self.assertAlmostEqual(s.last.wall_ms, 24.0, places=6)
        self.assertAlmostEqual(s.last.sync_ms, 0.0)

    def test_spans_outside_a_round_are_not_charged(self):
        s, c, h49 = self._split()
        s.on_round_end(mono=0.5)  # no round open: inert
        c.sched_ms += 400.0  # a prefill phase's scheduling
        s.on_round_open(round_id=1, mono=1.0)
        s.on_round_open(round_id=2, mono=1.02)
        self.assertEqual(s.last.sched_ms, 0.0)

    def test_period_line_and_off(self):
        cap = _capture(self)
        dev = _Dev()
        probe = DeviceGapProbe(backend=_Backend(dev))
        s, c, h49 = self._split(period=4, probe=probe)
        t = 1.0
        for rid in range(9):
            s.on_round_open(round_id=rid, mono=t)
            _x162_round(c, h49)
            s.mark(MARK_DRAFT_BEGIN)
            for point, dt in ((MARK_DRAFT_END, 1.2), (MARK_VERIFY_LAUNCH, 1.5),
                              (MARK_VERIFY_END, 21.2), (MARK_DEXT_BEGIN, 0.3),
                              (MARK_DEXT_END, 0.7)):
                dev.at += dt
                s.mark(point)
            dev.at += 1.1
            dev.now = dev.at - 30.0  # the device runs ~one round behind
            t += 0.026
        lines = [l for l in cap.lines if l.startswith("DECODE-HOST-SPLIT")]
        self.assertEqual(len(lines), 2, lines)
        self.assertRegex(
            lines[0],
            r"^DECODE-HOST-SPLIT rank=0 n=4 last_round=3 wall_ms=26\.0/26\.0 "
            r"sync_ms=17\.8/17\.8 host_work_ms=8\.2/8\.2 seq_wait_ms=0\.0/0\.0 "
            r"ple_sync_ms=1\.3/1\.3 result_wait_ms=0\.0/0\.0 ctl_wait_ms=16\.5/16\.5 "
            r"ctl_wait_n=4 recv_ms=0\.1/0\.1 sched_ms=1\.2/1\.2 hicache_ms=0\.4/0\.4 "
            r"draft_ms=17\.4/17\.4 verify_ms=4\.0/4\.0 vprep_ms=0\.8/0\.8 "
            r"ple_stage_ms=1\.2/1\.2 launch_ms=0\.3/0\.3 accept_ms=0\.4/0\.4 "
            r"dext_ms=0\.8/0\.8 result_ms=0\.6/0\.6 bcast_ms=0\.2/0\.2 bcast_n=8 "
            r"fetch_plan_ms=0\.0/0\.0 fetch_plan_n=0 other_ms=1\.9/1\.9 gpu_n=\d+ "
            r"gpu_unread=0 gpu_draft_ms=1\.2/1\.2 gpu_gap_ple_ms=1\.5/1\.5 "
            r"gpu_verify_ms=21\.2/21\.2 gpu_accept_ms=0\.3/0\.3 gpu_dext_ms=0\.7/0\.7 "
            r"gpu_gap_round_ms=1\.1/1\.1 gpu_idle_ms=2\.6/2\.6 \(",
        )
        n = int(re.search(r"gpu_n=(\d+)", lines[0]).group(1))
        self.assertGreaterEqual(n, 2)
        self.assertEqual(dev.syncs, 0)
        # off: no probe, no line
        cap.lines.clear()
        off, c2, h2 = self._split(period=0)
        self.assertIsNone(off.probe)
        self.assertFalse(off.on)
        for rid in range(200):
            off.on_round_open(round_id=rid, mono=rid * 0.026)
            off.mark(MARK_DRAFT_BEGIN)
        self.assertEqual([l for l in cap.lines if "DECODE-HOST-SPLIT" in l], [])

    def test_gpu_fields_print_dash_without_reads(self):
        line = DecodeHostSplit.period_line([], [], rank=2)
        self.assertIn("gpu_n=0 gpu_unread=0 gpu_draft_ms=- ", line)
        self.assertIn("gpu_idle_ms=- (", line)

    def test_a_failing_device_half_turns_itself_off(self):
        class _Broken:
            def event(self):
                raise RuntimeError("no CUDA")

            def is_capturing(self):
                return False

        cap = _capture(self)
        s, c, h49 = self._split(period=2, probe=DeviceGapProbe(backend=_Broken()))
        s.on_round_open(round_id=1, mono=1.0)
        self.assertIsNone(s.probe)
        s.mark(MARK_DRAFT_BEGIN)  # inert now
        s.on_round_open(round_id=2, mono=1.02)
        s.on_round_open(round_id=3, mono=1.04)
        self.assertTrue(any("device marks off" in l for l in cap.lines))
        self.assertTrue(any(l.startswith("DECODE-HOST-SPLIT rank=0 n=2") for l in cap.lines))

    def test_env_drives_the_period(self):
        self.assertEqual(envs.SGLANG_DEBUG_DECODE_HOST_SPLIT.get(), 64)
        with envs.SGLANG_DEBUG_DECODE_HOST_SPLIT.override(0):
            s = DecodeHostSplit.from_env(rank=0)
            self.assertFalse(s.on)
            self.assertIsNone(s.probe)

    def test_note_span_and_timed_add_to_the_singleton(self):
        clk = _Clock()
        before = (dhs.SPLIT.seq_wait_ms, dhs.SPLIT.fetch_plan_ms, dhs.SPLIT.fetch_plan_n)

        @dhs.timed("fetch_plan_ms", "fetch_plan_n")
        def plan(fail):
            clk.advance(0.25)
            if fail:
                raise ValueError("x")
            return 7

        with mock.patch.object(time, "perf_counter", clk.perf_counter):
            t0 = time.perf_counter()
            clk.advance(17.0)
            dhs.note_span("seq_wait_ms", t0)
            self.assertEqual(plan(False), 7)
            with self.assertRaises(ValueError):
                plan(True)
        self.assertAlmostEqual(dhs.SPLIT.seq_wait_ms - before[0], 17.0, places=6)
        self.assertAlmostEqual(dhs.SPLIT.fetch_plan_ms - before[1], 0.5, places=6)
        self.assertEqual(dhs.SPLIT.fetch_plan_n - before[2], 2)


class DecodeRoundLogCarriesTheSplitTest(unittest.TestCase):
    def test_begin_and_end_round_drive_the_split_and_register_it(self):
        from sglang.srt.managers.scheduler_components.decode_round_log import (
            DecodeRoundLog,
        )

        with envs.SGLANG_DEBUG_DECODE_HOST_SPLIT.override(0):
            log = DecodeRoundLog(clock=None, rank=1)
        self.assertIs(dhs.active_split(), log.host_split)
        c = SplitCounters()
        log.host_split = DecodeHostSplit(rank=1, period=0, counters=c,
                                         h49=HostCostCounters())
        log.begin_round(round_id=7, bs=1, rows=4)
        c.seq_wait_ms += 12.5
        log.end_round()
        self.assertEqual(log.host_split.last.round_id, 7)
        self.assertAlmostEqual(log.host_split.last.seq_wait_ms, 12.5)


# -- wiring through the real functions --------------------------------------


class WiringTest(unittest.TestCase):
    def test_resolve_seq_lens_cpu_times_its_existing_sync_as_seq_wait(self):
        import torch

        from sglang.srt.managers import overlap_utils

        clk = _Clock()

        class _Stream:
            def wait_event(self, ev):
                pass

            def synchronize(self):
                clk.advance(16.5)  # the previous round's verify

        class _Publish:
            def wait(self):
                pass

        fm = types.SimpleNamespace(
            publish_ready=_Publish(),
            _publish_fresh=True,
            new_seq_lens_buf=torch.tensor([0, 700, 0], dtype=torch.int32),
            needs_cpu_seq_lens=True,
            fwd_prepare_d2h_stream=_Stream(),
            new_seq_lens_cpu_pinned=torch.zeros(3, dtype=torch.int32),
            device="cpu",
            req_pool_size=3,
            max_context_len=262144,
        )
        batch = types.SimpleNamespace(
            spec_info=types.SimpleNamespace(future_indices=torch.tensor([1])),
            req_pool_indices_cpu=torch.tensor([1]),
        )
        before = dhs.SPLIT.seq_wait_ms
        with mock.patch.object(time, "perf_counter", clk.perf_counter), \
                mock.patch.object(overlap_utils.torch, "get_device_module",
                                  lambda dev: types.SimpleNamespace(
                                      stream=lambda s: mock.MagicMock())):
            overlap_utils.FutureMap.resolve_seq_lens_cpu(fm, batch)
        self.assertAlmostEqual(dhs.SPLIT.seq_wait_ms - before, 16.5, places=6)
        # the pre-existing mirror still lands: row 1 of the pinned copy
        self.assertEqual(int(batch.seq_lens_sum), 700)

    def test_finish_ple_verify_stage_splits_sync_and_stage(self):
        from sglang.srt.models import qwen4_exp_ple_decode_pread as ple

        clk = _Clock()
        staged = []

        class _Event:
            def synchronize(self):
                clk.advance(1.3)  # the draft

        class _Stager:
            def stage(self, rows, *, sync_s, t_ready):
                clk.advance(1.2)  # hash + pread
                staged.append(rows)

        stage = ple.PleVerifyStage(
            stagers=(_Stager(),),
            ctx_host=types.SimpleNamespace(tolist=lambda: [[1, 2, 3, 4, 5, 6]]),
            event=_Event(),
        )
        b = (dhs.SPLIT.ple_sync_ms, dhs.SPLIT.ple_stage_ms)
        with mock.patch.object(time, "perf_counter", clk.perf_counter):
            ple.finish_ple_verify_stage(stage)
            ple.finish_ple_verify_stage(None)  # a worker rank: nothing
        self.assertEqual(staged, [[[1, 2, 3, 4, 5, 6]]])
        self.assertAlmostEqual(dhs.SPLIT.ple_sync_ms - b[0], 1.3, places=6)
        self.assertAlmostEqual(dhs.SPLIT.ple_stage_ms - b[1], 1.2, places=6)

    def test_barlink_broadcast_counts_every_host_path_broadcast(self):
        from sglang.srt.distributed.device_communicators.barlink import (
            BarlinkCommunicator,
        )

        clk = _Clock()
        seen = []

        class _T:
            def barlink_broadcast(self, comm, tensor, src):
                clk.advance(0.1)
                seen.append(src)
                return tensor

        stub = types.SimpleNamespace(
            disabled=False,
            _select=lambda op, nbytes: _T(),
            _after_transport=lambda t, op: clk.advance(0.02),
        )
        tensor = types.SimpleNamespace(numel=lambda: 12, element_size=lambda: 4)
        b = (dhs.SPLIT.bcast_ms, dhs.SPLIT.bcast_n)
        with mock.patch.object(time, "perf_counter", clk.perf_counter):
            self.assertIs(BarlinkCommunicator.broadcast(stub, tensor, 0), tensor)
        self.assertEqual(seen, [0])
        self.assertAlmostEqual(dhs.SPLIT.bcast_ms - b[0], 0.12, places=6)
        self.assertEqual(dhs.SPLIT.bcast_n - b[1], 1)

    def test_bar1_forced_wait_is_ctl_wait_and_only_that(self):
        import torch

        from sglang.srt.distributed.device_communicators import barlink_abort_gate
        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            BarlinkBar1Transport,
        )

        clk = _Clock()

        class _Ev:
            def __init__(self):
                self.records = 0

            def query(self):
                return False

            def record(self):
                self.records += 1

        def _wait():
            clk.advance(19.0)  # to the verify end
            return True

        stub = types.SimpleNamespace(
            _ctl_defer=True,
            _abort_code_seen=0,
            _ctl_inflight=True,
            _ctl_event=_Ev(),
            _ctl_lag=0,
            _ctl_stall_run=0,
            _ctl_build_deferred_s=0.0,
            _ctl_stage=torch.zeros(1, dtype=torch.int32),
            _ctl_src=torch.zeros(1, dtype=torch.int32),
            _wait_ctl_event=_wait,
        )
        b = (dhs.SPLIT.ctl_wait_ms, dhs.SPLIT.ctl_wait_n)
        with mock.patch.object(time, "perf_counter", clk.perf_counter), \
                mock.patch.object(barlink_abort_gate, "max_lag", lambda: 4):
            # three unresolved checks: no wait, nothing counted
            for _ in range(3):
                self.assertIsNone(BarlinkBar1Transport._read_status_for_check(stub))
            self.assertEqual(dhs.SPLIT.ctl_wait_n - b[1], 0)
            # the fourth hits the bound: one forced wait
            self.assertEqual(BarlinkBar1Transport._read_status_for_check(stub), 0)
        self.assertEqual(dhs.SPLIT.ctl_wait_n - b[1], 1)
        self.assertAlmostEqual(dhs.SPLIT.ctl_wait_ms - b[0], 19.0, places=6)


_SYNCS = (".synchronize(", ".item(", ".tolist(", ".cpu(", "torch.cuda.synchronize")


class StructuralWiringTest(unittest.TestCase):
    """Where a CPU stub cannot carry the real method, pin the placement and
    that no instrument line adds a device read."""

    def _no_sync_on_h58_lines(self, src: str) -> None:
        for line in src.splitlines():
            if "_h58" in line:
                for s in _SYNCS:
                    self.assertNotIn(s, line, line)

    def test_scheduler_overlap_loop_spans(self):
        text = _src("sglang/srt/managers/scheduler.py")
        src = _between(text, "    def event_loop_overlap(self):",
                       "    def is_disable_overlap_for_batch(")
        for span in ("result_ms", "recv_ms", "sched_ms"):
            self.assertIn('_h58_span("%s", _h58_t0)' % span, src)
        # recv brackets recv + process_input; sched brackets get_next_batch_to_run
        r = src.index('_h58_span("recv_ms"')
        self.assertLess(src.index("self.process_input_requests(recv_reqs)"), r)
        s = src.index('_h58_span("sched_ms"')
        self.assertLess(src.index("plan = self.get_next_batch_to_run("), s)
        self._no_sync_on_h58_lines(src)

    def test_eagle_round_marks_and_spans_in_order(self):
        text = _src("sglang/srt/speculative/eagle_worker_v2.py")
        rnd = _between(text, "    def _forward_decode_round(", "    def _forward_spill_tick_spec(")
        order = [
            "_h58.mark(_h58.MARK_DRAFT_BEGIN)",
            "self.draft_worker.draft(batch)",
            "_h58.mark(_h58.MARK_DRAFT_END)",
            '_h58.note_span("draft_ms", _h58_t)',
            "batch_output = self.verify(batch)",
            '_h58.note_span("verify_ms", _h58_t)',
            "_h58.mark(_h58.MARK_DEXT_BEGIN)",
            "on_publish(batch_output.new_seq_lens)",
            "self.draft_worker._draft_extend_for_decode(",
            "_h58.mark(_h58.MARK_DEXT_END)",
            '_h58.note_span("dext_ms", _h58_t)',
        ]
        pos = [rnd.index(o) for o in order]
        self.assertEqual(pos, sorted(pos))
        self._no_sync_on_h58_lines(rnd)
        ver = _between(text, "    def verify(self, batch: ScheduleBatch):",
                       "    def _finalize_accept_tree_path(")
        order = [
            "_h58_t = time.perf_counter()",
            "eagle_prepare_for_verify(",
            '_h58.note_span("vprep_ms", _h58_t)',
            "finish_ple_verify_stage(ple_stage)",
            "_h58.mark(_h58.MARK_VERIFY_LAUNCH)",
            "self.target_worker.forward_batch_generation(",
            "_h58.mark(_h58.MARK_VERIFY_END)",
            '_h58.note_span("launch_ms", _h58_t)',
            "eagle_sample(",
            "commit_mamba_states_after_verify(",
            '_h58.note_span("accept_ms", _h58_t)',
        ]
        pos = [ver.index(o) for o in order]
        self.assertEqual(pos, sorted(pos))
        self._no_sync_on_h58_lines(ver)
        # the verify's pre-existing host reads are exactly the grammar path's
        self.assertEqual(ver.count(".cpu()"), 3)
        self.assertEqual(ver.count(".synchronize()"), 0)

    def test_expert_planner_resolve_is_timed_once(self):
        text = _src("sglang/srt/layers/moe/expert_offload.py")
        cls = _between(text, "class ExpertResidencyPlanner:", "    def resolve_sticky(")
        self.assertIn('@_h58_timed("fetch_plan_ms", "fetch_plan_n")\n    def resolve(', cls)
        sticky = _between(text, "    def resolve_sticky(", "\n    def ")
        self.assertNotIn("_h58", sticky)  # its fallback calls resolve(): once

    def test_nf_backends_opt_out_of_the_cpu_seq_lens_mirror(self):
        """Why seq_wait is 0 on NF and the forced BAR1 wait is where NF's host
        blocks: every backend NF's spec-v2 round touches opts out, so
        ``resolve_seq_lens_cpu`` takes the GPU-only branch (no host sync)."""
        qsa = _src("sglang/srt/layers/attention/qwen_sparse_attn_backend.py")
        for cls in ("class QwenSparseAttnBackend(", "class QwenSparseMultiStepDraftBackend"):
            head = qsa[qsa.index(cls):][:400]
            self.assertIn("needs_cpu_seq_lens: bool = False", head, cls)
        gdn = _src("sglang/srt/layers/attention/linear/gdn_backend.py")
        self.assertIn("needs_cpu_seq_lens: bool = False", gdn)
        hyb = _src("sglang/srt/layers/attention/hybrid_linear_attn_backend.py")
        self.assertRegex(
            hyb,
            r"self\.needs_cpu_seq_lens = \(\s*full_attn_backend\.needs_cpu_seq_lens\s*"
            r"or linear_attn_backend\.needs_cpu_seq_lens",
        )
        from sglang.srt.managers.overlap_utils import decide_needs_cpu_seq_lens

        args = types.SimpleNamespace(enable_two_batch_overlap=False,
                                     speculative_algorithm="EAGLE")
        off = types.SimpleNamespace(needs_cpu_seq_lens=False)
        self.assertFalse(decide_needs_cpu_seq_lens(args, [off, off, off]))
        self.assertTrue(decide_needs_cpu_seq_lens(args, [off, object()]))

    def test_seq_wait_brackets_the_existing_sync(self):
        text = _src("sglang/srt/managers/overlap_utils.py")
        src = _between(text, "    def resolve_seq_lens_cpu(", "    def publish(")
        self.assertEqual(src.count(".synchronize()"), 2)  # HIP publish + d2h stream
        a = src.index("_h58_t0 = time.perf_counter()")
        z = src.index("self.fwd_prepare_d2h_stream.synchronize()")
        e = src.index('_h58_span("seq_wait_ms", _h58_t0)', z)
        self.assertLess(a, z)
        self.assertLess(z, e)


if __name__ == "__main__":
    unittest.main()
