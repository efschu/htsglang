"""#917: the cutover census must BRACKET a poisoned context, not swallow it.

THE SPECIMENS THESE ARE WRITTEN AGAINST. Two boots of the 0826 acceptance
window died of one CUDA illegal memory access apiece, and in neither could the
desk name where the fault was born:

  * ``boot_accept0826r7fix_0826_1817.log`` 18:27:14 (tp_to_pp) -- all three
    ranks' #760 device-tier quiesce SUCCEEDED, no stream failure anywhere in
    the log, and then PP2's barlink watchdog reported the fault.
  * ``boot_rerun0826_0826_2149.log`` 21:53:36 (pp_to_tp) -- the same watchdog
    report on PP1/PP2 at lines 2162/2171, THEN "#760 quiesce ... synchronizing
    load_stream failed" at 2189/2195, then a traceback in ``get_cpu_copy``.

Three reporting sites, one sticky fault, and the ordering cannot discriminate:
the watchdog runs on a timer, so it wins the race to observe an
already-poisoned context regardless of who poisoned it. That is NOTE_867's
open question 2, and it stayed open because nothing on the cutover path
carried a POSITION in the walk.

The census already stops at ~19 named boundaries and already calls the driver
at each one (``mem_get_info``). It was throwing that answer away under a bare
``except Exception`` -- #867's class, second instance, in the one instrument
positioned to answer the question.

RED-FIRST. Every assertion below fails on the parent commit: ``probe_poison``,
``last_clean_label`` and ``probe_failures`` do not exist there, and
``format_line`` emits no clause.
"""

import unittest

from sglang.srt.distributed.device_communicators import barlink_abort_gate
from sglang.srt.managers import phase_flip_seam_census as census

MIB = 1024 * 1024

#: The driver's wording, verbatim from the specimens. Matched by MESSAGE and
#: not by class, because torch raises ``AcceleratorError`` for a recoverable
#: OOM too -- the distinction the gate's own classifier exists to keep.
IMA = "CUDA error: an illegal memory access was encountered\nSearch for ..."


class _Ima(RuntimeError):
    pass


def _probe_then_poison(clean_stages: int):
    """A probe that answers ``clean_stages`` times, then raises the sticky IMA.

    Sticky in the test as it is on metal: once the context is gone it never
    answers again. A probe that recovered would let a pin pass for a reason
    the hardware cannot supply.
    """
    state = {"n": 0}

    def probe():
        state["n"] += 1
        if state["n"] > clean_stages:
            raise _Ima(IMA)
        return (5000 * MIB, 4000 * MIB, 3000 * MIB)

    return probe


class TestThePoisonBracket(unittest.TestCase):
    def setUp(self):
        barlink_abort_gate.clear_poison_record()
        census.reset()

    def tearDown(self):
        barlink_abort_gate.clear_poison_record()
        census.reset()

    def test_the_bracket_names_the_segment_the_fault_was_born_in(self):
        c = census.SeamCensus("pp_to_tp", 1, probe=_probe_then_poison(2))
        c.mark("flip_writeback")
        c.mark("hicache_quiesce")
        c.mark("resident_release")
        self.assertEqual(c.last_clean_label, "hicache_quiesce")
        self.assertIsNotNone(c.probe_poison)
        self.assertEqual(c.probe_poison[0], "resident_release")
        line = c.format_line()
        self.assertIn("#917 CONTEXT POISONED", line)
        self.assertIn("'hicache_quiesce'", line)
        self.assertIn("'resident_release'", line)

    def test_the_first_poisoned_boundary_wins_not_the_last(self):
        """The ORIGIN, not the newest report -- the rule ``record_poison`` states.

        A sticky fault raises at every later boundary too. A bracket that
        tracked the newest would name the last stage of the walk on every
        specimen, which is the misattribution this exists to end.
        """
        c = census.SeamCensus("pp_to_tp", 0, probe=_probe_then_poison(1))
        c.mark("kv_pack")
        c.mark("kv_local_read")
        c.mark("kv_write")
        c.mark("cutover")
        self.assertEqual(c.probe_poison[0], "kv_local_read")

    def test_it_registers_with_the_process_wide_first_wins_record(self):
        c = census.SeamCensus("tp_to_pp", 2, probe=_probe_then_poison(0))
        c.mark("hicache_quiesce")
        record = barlink_abort_gate.poison_record()
        self.assertIsNotNone(record)
        self.assertIn("hicache_quiesce", record["source"])

    def test_a_watchdog_that_reported_first_still_gets_a_coordinate(self):
        """The common case on metal, and the whole point of the second shape.

        The barlink poll wins the race in both specimens. Its record carries no
        position in the walk; the census supplies the last boundary this rank
        passed cleanly, which turns "somebody saw poison" into "poison appeared
        after <stage>" without silencing the fastest detector on the path.
        """
        c = census.SeamCensus("pp_to_tp", 1, probe=_probe_then_poison(99))
        c.mark("flip_writeback")
        c.mark("hicache_quiesce")
        barlink_abort_gate.record_poison("barlink poll_status_word", _Ima(IMA))
        line = c.format_line()
        self.assertIn("#917 CONTEXT POISONED", line)
        self.assertIn("barlink poll_status_word", line)
        self.assertIn("'hicache_quiesce'", line)

    def test_a_healthy_flip_gains_no_clause(self):
        """A byte-identical line when nothing was poisoned.

        An instrument that decorates the healthy case makes every consumer of
        this corpus re-learn the format for nothing.
        """
        c = census.SeamCensus("pp_to_tp", 0, probe=_probe_then_poison(99))
        c.mark("kv_pack")
        c.mark("cutover")
        self.assertNotIn("#917", c.format_line())
        self.assertIsNone(c.probe_poison)


class TestAHiccupIsNotAContextKill(unittest.TestCase):
    """#867's distinction, held in the module that held its second instance."""

    def setUp(self):
        barlink_abort_gate.clear_poison_record()

    def tearDown(self):
        barlink_abort_gate.clear_poison_record()

    def test_a_transient_probe_failure_is_counted_and_not_recorded(self):
        def probe():
            raise RuntimeError("nvmlDeviceGetMemoryInfo: transient failure")

        c = census.SeamCensus("pp_to_tp", 0, probe=probe)
        c.mark("kv_pack")
        self.assertEqual(c.probe_failures, 1)
        self.assertIsNone(c.probe_poison)
        self.assertIsNone(barlink_abort_gate.poison_record())
        self.assertNotIn("#917", c.format_line())

    def test_an_out_of_memory_is_a_hiccup_not_poison(self):
        """The direction that must NOT over-refuse.

        torch raises the same exception class for a recoverable OOM. Reading
        the class instead of the message would turn every memory-pressed flip
        into a reported context kill -- and the cutover is where memory
        pressure lives.
        """

        def probe():
            raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")

        c = census.SeamCensus("tp_to_pp", 1, probe=probe)
        c.mark("hicache_quiesce")
        self.assertIsNone(c.probe_poison)
        self.assertIsNone(barlink_abort_gate.poison_record())

    def test_a_probe_returning_none_is_still_just_a_gap(self):
        """The pre-existing degrade path is untouched.

        ``None`` means the probe declined, not that it faulted. It keeps
        producing a ``probe-failed`` row and nothing else.
        """
        c = census.SeamCensus("pp_to_tp", 0, probe=lambda: None)
        c.mark("kv_pack")
        self.assertEqual(c.probe_failures, 0)
        self.assertIsNone(c.probe_poison)
        self.assertIn("kv_pack=probe-failed", c.format_line())


class TestTheInstrumentStillCannotKillAFlip(unittest.TestCase):
    """The no-return-path contract this module has held since #631.

    The classifier runs INSIDE the cutover's no-return region. Every way it can
    go wrong must degrade to a missing log line.
    """

    def setUp(self):
        barlink_abort_gate.clear_poison_record()

    def tearDown(self):
        barlink_abort_gate.clear_poison_record()

    def test_a_classifier_that_cannot_reach_the_gate_does_not_escape(self):
        c = census.SeamCensus("pp_to_tp", 0, probe=_probe_then_poison(0))
        original = barlink_abort_gate.is_poison_error
        barlink_abort_gate.is_poison_error = lambda exc: (_ for _ in ()).throw(
            RuntimeError("gate unavailable")
        )
        try:
            c.mark("hicache_quiesce")  # must not raise
        finally:
            barlink_abort_gate.is_poison_error = original
        self.assertEqual(c.probe_failures, 1)
        self.assertEqual(c.stages[-1], ("hicache_quiesce", -1, -1, -1))

    def test_format_line_survives_a_gate_that_cannot_be_read(self):
        c = census.SeamCensus("pp_to_tp", 0, probe=_probe_then_poison(99))
        c.mark("kv_pack")
        original = barlink_abort_gate.poison_record
        barlink_abort_gate.poison_record = lambda: (_ for _ in ()).throw(
            RuntimeError("gate unavailable")
        )
        try:
            line = c.format_line()
        finally:
            barlink_abort_gate.poison_record = original
        self.assertIn("kv_pack", line)
        self.assertNotIn("#917", line)

    def test_the_module_level_mark_still_swallows_everything(self):
        """``census.mark`` on a poisoned probe is still a no-op for the caller."""
        census.reset()
        c = census.begin("pp_to_tp", 0, probe=_probe_then_poison(0))
        self.assertIsNotNone(c)
        census.mark("hicache_quiesce")  # must not raise
        census.reset()


if __name__ == "__main__":
    unittest.main()
