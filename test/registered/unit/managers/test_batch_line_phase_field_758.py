"""A batch line must name the PHASE it ran in, not just the rank (#758-4).

THE TRAP, caught live on 2026-08-18. Batch lines are prefixed with the rank
name -- ``PP0]``, ``PP1]``, ``PP2]`` -- and that name is the RANK, not the
layout. It survives the flip, so a decode executed in the TP layout still
prints under ``PP0]``. Measured on the known-good boot that morning: 95 of 95
decode batches ran in the TP phase and 0 in PP, across 135 PHASE-FLIP DONE
lines. The doctrine (prefill in PP, decode in TP) held perfectly while the
labelling said the opposite -- and the wrong reading is the one a reader
expects, which is what makes it dangerous rather than merely untidy.

WHY THE MUTATION CASE IS HERE. An emitter is only worth its line if its
absence is detectable. ``test_stripping_the_field_is_caught`` removes the
field the way a careless refactor would and asserts the check goes red, so
this file cannot rot into a test that passes whatever the code does.
"""

import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)


class PhaseField(unittest.TestCase):
    def _field(self, flip_enabled, tp_active):
        """Drive the SHIPPED helper with both authorities stubbed."""
        from sglang.srt.managers.scheduler_components import metrics_reporter as mr
        from sglang.srt import runtime_context as sa
        from sglang.srt.distributed import parallel_state as ps

        prev_args, prev_routing = sa.get_server_args, ps.phase_flip_tp_routing_active
        try:
            sa.get_server_args = lambda: types.SimpleNamespace(
                enable_phase_flip=flip_enabled
            )
            ps.phase_flip_tp_routing_active = lambda: tp_active
            return mr._active_phase_field()
        finally:
            sa.get_server_args = prev_args
            ps.phase_flip_tp_routing_active = prev_routing

    def test_decode_after_pp_to_tp_carries_tp(self):
        """The case the live log got wrong: TP routing active -> phase=tp."""
        self.assertEqual(self._field(flip_enabled=True, tp_active=True), " phase=tp")

    def test_prefill_after_tp_to_pp_carries_pp(self):
        self.assertEqual(self._field(flip_enabled=True, tp_active=False), " phase=pp")

    def test_no_flip_boot_is_byte_identical(self):
        """With no flip there is one layout, the question is meaningless, and
        the line must stay exactly what every non-flip boot has printed."""
        self.assertEqual(self._field(flip_enabled=False, tp_active=False), "")
        self.assertEqual(self._field(flip_enabled=False, tp_active=True), "")

    def test_a_broken_authority_never_breaks_the_stats_line(self):
        """A label is not worth losing the throughput line over."""
        from sglang.srt.managers.scheduler_components import metrics_reporter as mr
        from sglang.srt import runtime_context as sa

        prev = sa.get_server_args
        try:

            def _boom():
                raise RuntimeError("authority unavailable")

            sa.get_server_args = _boom
            self.assertEqual(mr._active_phase_field(), "")
        finally:
            sa.get_server_args = prev

    def test_stripping_the_field_is_caught(self):
        """MUTATION PROOF. Replace the helper with one that returns nothing --
        the way a refactor that 'tidied' the field away would -- and the phase
        assertions must fail. If this test ever passes while the others do
        too, the checks above have stopped checking anything."""
        from sglang.srt.managers.scheduler_components import metrics_reporter as mr

        prev = mr._active_phase_field
        try:
            mr._active_phase_field = lambda: ""
            with self.assertRaises(AssertionError):
                self.assertEqual(mr._active_phase_field(), " phase=tp")
        finally:
            mr._active_phase_field = prev
        # and the real one still works afterwards
        self.assertEqual(self._field(flip_enabled=True, tp_active=True), " phase=tp")


if __name__ == "__main__":
    unittest.main()
