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


class TestOffByDefault(unittest.TestCase):
    def setUp(self):
        os.environ.pop(corridor_trace.TRACE_ENV, None)

    def test_the_tick_leaves_no_trace_when_the_flag_is_unset(self):
        stub = _Stub()
        _tick(stub)
        self.assertIsNone(stub._corridor_trace)

    def test_the_sampler_is_never_constructed_when_the_flag_is_unset(self):
        stub = _Stub()
        with mock.patch.object(corridor_trace, "CorridorTrace") as ctor:
            _tick(stub)
        ctor.assert_not_called()

    def test_repeated_ticks_stay_silent(self):
        stub = _Stub()
        with mock.patch.object(corridor_trace, "CorridorTrace") as ctor:
            for _ in range(50):
                _tick(stub)
        ctor.assert_not_called()


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
