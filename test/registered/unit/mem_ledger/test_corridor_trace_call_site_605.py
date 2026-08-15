# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The corridor sampler's PRODUCTION call site (#605 R2 open item 1).

RED-FIRST. ``corridor_trace`` was tested and ready and had no caller, which
made it a module the tree carried and never ran -- the desk-written-never-
executed shape. The natural home is the scheduler tick, and the property that
matters most is the one this file spends the most tests on: with the flag
unset the tick must leave ZERO trace, no thread, no NVML call, no attribute.

INVERTED 2026-08-15. That last property was the bug. "Off unless asked" meant
a default boot could not see the corridor law, and the whole self-correcting
chain hangs off this call site: the audit reports a breach only if the trace
armed, and ``record_corridor_shortfall`` only ever writes a number the audit
produced. Measured over two boots on this rig, an external 100 ms NVML sampler
saw 57 and 15 breaches (minima 895 and 935 MiB) while the serving logs
contained ZERO "CORRIDOR LAW BREACHED" lines and the seam records kept
``corridor_shortfall_bytes: 0``. The pins now say the tick ARMS by default and
that an explicit off still disarms it.
"""

import os
import types
import unittest
from unittest import mock

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_ledger import corridor_trace


class _Stub:
    """The smallest object the tick can run against.

    Deliberately not a mock of the method under test: the method is taken off
    the REAL Scheduler class and bound here, so a rename or a signature change
    on the production side fails this file rather than passing it.
    """

    def __init__(self):
        self._corridor_trace = None
        self._corridor_trace_armed = False


def _tick(stub):
    return types.MethodType(Scheduler._corridor_trace_tick, stub)()


class TestArmedByDefault(unittest.TestCase):
    """INVERTED 2026-08-15 together with the module default.

    The tick is where the law becomes observable, so with the flag unset it
    must ARM rather than stay silent. The old pins asserted the opposite and
    were therefore certifying the bug.
    """

    def setUp(self):
        os.environ.pop(corridor_trace.TRACE_ENV, None)

    def test_the_tick_arms_the_trace_with_no_flag_set(self):
        stub = _Stub()
        started = mock.MagicMock()
        with mock.patch.object(corridor_trace, "start", return_value=started):
            _tick(stub)
        self.assertIs(stub._corridor_trace, started)

    def test_the_sampler_is_constructed_with_no_flag_set(self):
        stub = _Stub()
        with mock.patch.object(corridor_trace, "CorridorTrace") as ctor:
            _tick(stub)
        ctor.assert_called_once()

    def test_it_still_arms_only_once_however_many_ticks_run(self):
        stub = _Stub()
        with mock.patch.object(corridor_trace, "CorridorTrace") as ctor:
            for _ in range(50):
                _tick(stub)
        ctor.assert_called_once()


class TestTheOperatorCanStillDisarmIt(unittest.TestCase):
    def setUp(self):
        os.environ[corridor_trace.TRACE_ENV] = "0"
        self.addCleanup(os.environ.pop, corridor_trace.TRACE_ENV, None)

    def test_an_explicit_off_leaves_no_trace(self):
        stub = _Stub()
        with mock.patch.object(corridor_trace, "CorridorTrace") as ctor:
            _tick(stub)
        ctor.assert_not_called()
        self.assertIsNone(stub._corridor_trace)


class TestTheFlagArmsIt(unittest.TestCase):
    def setUp(self):
        os.environ[corridor_trace.TRACE_ENV] = "100"
        self.addCleanup(os.environ.pop, corridor_trace.TRACE_ENV, None)

    def test_the_flag_arms_the_sampler_on_the_tick(self):
        stub = _Stub()
        started = mock.MagicMock()
        with mock.patch.object(corridor_trace, "start", return_value=started) as st:
            _tick(stub)
        st.assert_called_once()
        self.assertIs(stub._corridor_trace, started)

    def test_it_is_armed_ONCE_however_many_ticks_run(self):
        """The tick runs every scheduler iteration; a sampler per iteration
        would be thousands of NVML threads."""
        stub = _Stub()
        with mock.patch.object(
            corridor_trace, "start", return_value=mock.MagicMock()
        ) as st:
            for _ in range(2000):
                _tick(stub)
        self.assertEqual(st.call_count, 1)


class TestTheInstrumentCannotBreakServing(unittest.TestCase):
    def setUp(self):
        os.environ[corridor_trace.TRACE_ENV] = "100"
        self.addCleanup(os.environ.pop, corridor_trace.TRACE_ENV, None)

    def test_a_raising_sampler_does_not_escape_the_tick(self):
        stub = _Stub()
        with mock.patch.object(
            corridor_trace, "start", side_effect=RuntimeError("nvml")
        ):
            _tick(stub)  # must not raise
        self.assertIsNone(stub._corridor_trace)

    def test_a_failed_arming_is_not_retried_every_iteration(self):
        stub = _Stub()
        with mock.patch.object(
            corridor_trace, "start", side_effect=RuntimeError("nvml")
        ) as st:
            for _ in range(100):
                _tick(stub)
        self.assertEqual(st.call_count, 1)


class TestItIsWiredIntoTheTick(unittest.TestCase):
    def test_the_scheduler_tick_calls_it(self):
        import inspect

        source = inspect.getsource(Scheduler)
        self.assertIn("self._corridor_trace_tick()", source)

    def test_it_sits_beside_the_census_tick(self):
        """R2 named the scheduler tick as the natural home; keeping the two
        instruments adjacent is what makes that reviewable."""
        import inspect

        source = inspect.getsource(Scheduler)
        census = source.index("self._census_tick()")
        corridor = source.index("self._corridor_trace_tick()")
        self.assertLess(abs(corridor - census), 800)


if __name__ == "__main__":
    unittest.main()


class TestTheWholeChainReachesTheRecord(unittest.TestCase):
    """The bug was never the env alone -- it was that NOTHING downstream ran.

    A breach has to travel: trace -> summary(breach) -> the audit's report ->
    record_corridor_shortfall -> next boot's margin. Pinning only the arming
    would leave three of those four links uncovered, which is how this stayed
    dead while its own unit tests were green.
    """

    def setUp(self):
        os.environ.pop(corridor_trace.TRACE_ENV, None)

    def _breaching_stub(self, free_min_mib: int, law_mib: int = 1024):
        stub = _Stub()
        stub._corridor_trace_armed = True
        trace = mock.MagicMock()
        trace.summary.return_value = {
            "breach": True,
            "free_min_mib": free_min_mib,
            "corridor_mib": law_mib,
            "n": 600,
            "span_s": 60.0,
            "period_ms": 100,
        }
        stub._corridor_trace = trace
        stub.server_args = object()
        stub.phase_flip_runtime = types.SimpleNamespace(_rank=1)
        return stub

    def test_a_breach_is_written_down_for_the_next_boot(self):
        stub = self._breaching_stub(free_min_mib=935)
        with mock.patch(
            "sglang.srt.managers.phase_flip_seam_reserve.record_corridor_shortfall"
        ) as rec:
            _tick(stub)
        rec.assert_called_once()
        _args, _kw = rec.call_args
        # 1024 - 935 = 89 MiB of depth, in bytes.
        self.assertEqual(_args[2], 89 << 20)

    def test_a_lawful_run_writes_nothing(self):
        stub = self._breaching_stub(free_min_mib=1200)
        stub._corridor_trace.summary.return_value["breach"] = False
        with mock.patch(
            "sglang.srt.managers.phase_flip_seam_reserve.record_corridor_shortfall"
        ) as rec:
            _tick(stub)
        rec.assert_not_called()

    def test_only_a_DEEPER_breach_is_reported_again(self):
        stub = self._breaching_stub(free_min_mib=935)
        with mock.patch(
            "sglang.srt.managers.phase_flip_seam_reserve.record_corridor_shortfall"
        ) as rec:
            _tick(stub)
            stub._corridor_trace_next_check = 0.0
            _tick(stub)  # same depth: nothing new to say
            self.assertEqual(rec.call_count, 1)
            stub._corridor_trace.summary.return_value["free_min_mib"] = 800
            stub._corridor_trace_next_check = 0.0
            _tick(stub)
        self.assertEqual(rec.call_count, 2)
        self.assertEqual(rec.call_args[0][2], 224 << 20)
