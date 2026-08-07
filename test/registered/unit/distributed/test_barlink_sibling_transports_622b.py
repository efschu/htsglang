# Copyright 2023-2026 SGLang Team
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
"""#622b: an abort must also report the transports the raise skips.

THE GAP THESE TESTS PIN
-----------------------
A serving process on this fork brings up THREE independent BAR1 transports --
``world:0``, ``tp:0`` and ``dcp:0``. Each owns a separate flag region and a
separate round counter. ``check_aborts`` raises at the FIRST transport that
reports, and every instrument on the abort path (capture census, own-flag
snapshot, peer-flag snapshot) is emitted by that transport alone.

The 2026-08-07 06:12 specimen is what that costs. It dumped ``tp:0`` and
nothing else, and the dump contradicts itself: every kernel writes its flag
BEFORE entering its deadline-bearing spin, yet ``tp:0``'s snapshot shows every
cell already at the round the spin was waiting for. ``dcp:0``, which carried
40544 all-gathers and 12768 all-reduces inside the same replayed decode graph,
was never read at all.

The dump added here is host-only on purpose. In that same specimen the raising
transport's own ``_abort_flag_snapshot`` -- a ``cuMemcpy`` -- took 55 s to
return, because a device read on the abort path can still queue behind a spin
that has not finished. So the sibling dump reads mapped BAR windows and the
pinned staged status word, and nothing else. Two of the tests below exist only
to pin that property, because it is the one that decides whether this
instrument is safe to leave on a production path.
"""

import logging
import unittest
from unittest import mock

import torch

from sglang.srt.distributed.device_communicators import barlink_abort_gate
from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    Bar1CollectiveAborted,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_517_harness():
    """Reuse the #517 test's transport builder rather than re-implementing it.

    Same reason #622's tests give: a re-implemented ``_transport`` is a second
    definition of the object under test, and it keeps passing after the real
    one changes. #624a already had to repair that builder once for exactly
    this drift.
    """
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_barlink_bar1_abort_deferred_517.py",
    )
    spec = importlib.util.spec_from_file_location("_abort_deferred_517", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_H = _load_517_harness()


class _Hostile:
    """A registered object whose every attribute access raises.

    Not a strawman: the abort path runs on a process that is already broken,
    and a transport mid-``close()`` has had its tensors dropped. An instrument
    that dies there replaces the real error with its own.
    """

    def __getattr__(self, name):
        raise RuntimeError(f"hostile attribute {name}")


class TestSiblingTransportDump(CustomTestCase):
    def setUp(self):
        barlink_abort_gate.reset_for_test()
        self.addCleanup(barlink_abort_gate.reset_for_test)

    # -- the gap itself ----------------------------------------------------

    def test_sibling_group_and_staged_word_are_reported(self):
        """The transport that never gets checked still appears in the log."""
        raiser = _H._transport(aborted=1, group="tp:0", defer=False)
        raiser._captured_launches = True
        sibling = _H._transport(aborted=1, group="dcp:0", defer=False)
        sibling._ctl_stage = torch.tensor([1], dtype=torch.int32)
        barlink_abort_gate.register(raiser)
        barlink_abort_gate.register(sibling)

        with self.assertLogs(barlink_abort_gate.logger, level=logging.ERROR) as cm:
            with self.assertRaises(Bar1CollectiveAborted):
                barlink_abort_gate.check_aborts("unit")

        line = "\n".join(cm.output)
        self.assertIn("SIBLING TRANSPORT state at abort (#622b)", line)
        self.assertIn("group dcp:0", line)
        self.assertIn("staged status word 1", line)
        # The raising transport must NOT be repeated in the sibling line; its
        # own dump already carries it with the full rank/op/rounds context.
        self.assertNotIn("group tp:0", line)

    def test_registration_order_does_not_hide_the_later_transport(self):
        """dcp:0 registers last in production; that must not cost its dump."""
        world = _H._transport(aborted=0, group="world:0", defer=False)
        raiser = _H._transport(aborted=1, group="tp:0", defer=False)
        raiser._captured_launches = True
        dcp = _H._transport(aborted=1, group="dcp:0", defer=False)
        for t in (world, raiser, dcp):
            barlink_abort_gate.register(t)

        with self.assertLogs(barlink_abort_gate.logger, level=logging.ERROR) as cm:
            with self.assertRaises(Bar1CollectiveAborted):
                barlink_abort_gate.check_aborts("unit")

        line = "\n".join(cm.output)
        self.assertIn("group world:0", line)
        self.assertIn("group dcp:0", line)

    # -- host-only, which is what makes it safe here -----------------------

    def test_no_device_read_on_siblings(self):
        """The sibling dump must never call the cuMemcpy-based snapshot.

        In the 06:12 specimen that call took 55 s on the raising transport.
        Paying it once per sibling on an abort path is how a diagnostic
        becomes the reason the evidence is late.
        """
        raiser = _H._transport(aborted=1, group="tp:0", defer=False)
        raiser._captured_launches = True
        sibling = _H._transport(aborted=0, group="dcp:0", defer=False)
        barlink_abort_gate.register(raiser)
        barlink_abort_gate.register(sibling)

        # Patched on the INSTANCE, not the class: the raising transport is of
        # the same class and legitimately calls both, so a class-level patch
        # would disarm the very raise this test needs.
        with (
            mock.patch.object(sibling, "_abort_flag_snapshot") as own_snap,
            mock.patch.object(sibling, "_read_status_for_check") as status_read,
            mock.patch.object(sibling, "status") as status,
        ):
            with self.assertLogs(barlink_abort_gate.logger, level=logging.ERROR):
                with self.assertRaises(Bar1CollectiveAborted):
                    barlink_abort_gate.check_aborts("unit")
            own_snap.assert_not_called()
            status_read.assert_not_called()
            status.assert_not_called()

    def test_peer_snapshot_is_the_only_flag_source_used(self):
        raiser = _H._transport(aborted=1, group="tp:0", defer=False)
        raiser._captured_launches = True
        sibling = _H._transport(aborted=0, group="dcp:0", defer=False)
        barlink_abort_gate.register(raiser)
        barlink_abort_gate.register(sibling)

        with mock.patch.object(
            sibling, "_abort_peer_flag_snapshot", return_value="peer 0 [1 lines]: 0:7"
        ):
            with self.assertLogs(barlink_abort_gate.logger, level=logging.ERROR) as cm:
                with self.assertRaises(Bar1CollectiveAborted):
                    barlink_abort_gate.check_aborts("unit")
        self.assertIn("peer 0 [1 lines]: 0:7", "\n".join(cm.output))

    # -- warn-never-raise --------------------------------------------------

    def test_the_original_abort_still_propagates_unchanged(self):
        raiser = _H._transport(aborted=1, group="tp:0", defer=False)
        raiser._captured_launches = True
        sibling = _H._transport(aborted=0, group="dcp:0", defer=False)
        barlink_abort_gate.register(raiser)
        barlink_abort_gate.register(sibling)

        with self.assertLogs(barlink_abort_gate.logger, level=logging.ERROR):
            with self.assertRaises(Bar1CollectiveAborted) as ctx:
                barlink_abort_gate.check_aborts("unit")
        self.assertEqual(ctx.exception.group, "tp:0")
        self.assertIn("observed at unit", str(ctx.exception))

    def test_hostile_sibling_does_not_mask_the_abort(self):
        raiser = _H._transport(aborted=1, group="tp:0", defer=False)
        raiser._captured_launches = True
        barlink_abort_gate.register(raiser)
        barlink_abort_gate.register(_Hostile())

        with self.assertRaises(Bar1CollectiveAborted):
            barlink_abort_gate.check_aborts("unit")

    def test_non_abort_exception_also_carries_the_dump(self):
        """A transport that fails for any other reason is the same evidence case."""

        class _Boom:
            group = "tp:0"

            def check_aborted(self, where):
                raise ValueError("something else")

        barlink_abort_gate.register(_Boom())
        sibling = _H._transport(aborted=1, group="dcp:0", defer=False)
        barlink_abort_gate.register(sibling)

        with self.assertLogs(barlink_abort_gate.logger, level=logging.ERROR) as cm:
            with self.assertRaises(ValueError):
                barlink_abort_gate.check_aborts("unit")
        self.assertIn("group dcp:0", "\n".join(cm.output))

    # -- default path ------------------------------------------------------

    def test_single_transport_emits_no_sibling_line(self):
        raiser = _H._transport(aborted=1, group="tp:0", defer=False)
        raiser._captured_launches = True
        barlink_abort_gate.register(raiser)
        self.assertIsNone(barlink_abort_gate.format_sibling_transports(raiser))

    def test_clean_run_logs_nothing(self):
        for group in ("world:0", "tp:0", "dcp:0"):
            barlink_abort_gate.register(
                _H._transport(aborted=0, group=group, defer=False)
            )
        with mock.patch.object(barlink_abort_gate.logger, "error") as err:
            barlink_abort_gate.check_aborts("unit")
        err.assert_not_called()

    def test_disabled_gate_still_short_circuits(self):
        raiser = _H._transport(aborted=1, group="tp:0", defer=False)
        raiser._captured_launches = True
        barlink_abort_gate.register(raiser)
        barlink_abort_gate.register(
            _H._transport(aborted=1, group="dcp:0", defer=False)
        )
        with mock.patch.object(
            barlink_abort_gate, "abort_check_enabled", return_value=False
        ):
            with mock.patch.object(barlink_abort_gate.logger, "error") as err:
                barlink_abort_gate.check_aborts("unit")
            err.assert_not_called()

    def test_unreadable_stage_is_reported_not_raised(self):
        raiser = _H._transport(aborted=1, group="tp:0", defer=False)
        raiser._captured_launches = True
        sibling = _H._transport(aborted=0, group="dcp:0", defer=False)
        sibling._ctl_stage = object()  # not indexable
        barlink_abort_gate.register(raiser)
        barlink_abort_gate.register(sibling)
        line = barlink_abort_gate.format_sibling_transports(raiser)
        self.assertIn("<unreadable>", line)

    def test_missing_stage_is_reported_as_never_staged(self):
        raiser = _H._transport(aborted=1, group="tp:0", defer=False)
        raiser._captured_launches = True
        sibling = _H._transport(aborted=0, group="dcp:0", defer=False)
        sibling._ctl_stage = None
        barlink_abort_gate.register(raiser)
        barlink_abort_gate.register(sibling)
        line = barlink_abort_gate.format_sibling_transports(raiser)
        self.assertIn("<never staged>", line)


if __name__ == "__main__":
    unittest.main()
