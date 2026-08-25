"""#867: the watchdog swallowed a fault the process cannot survive.

SPECIMEN (W40, boot_w40_857strict_0825_2113.log, CUDA_LAUNCH_BLOCKING=1 arm).
The FIRST CUDA fault in the whole log, at 21:17:13, immediately after
`SEAM DRAIN tp_to_pp: device-tier streams quiesced at the no-return point`:

    barlink_abort_gate.py:351 poll_status_words -> barlink_bar1.py:4937
      self._abort_poll_dst.copy_(self._ctl_dev[0:1], non_blocking=True)
    torch.AcceleratorError: CUDA error: an illegal memory access was encountered

`poll_status_words` caught it, logged "barlink-BAR1 status poll failed", and
CONTINUED. The scheduler then died in `get_cpu_copy` inside the seam CAPTURE;
the boot before it died in `load_cpu_copy` inside the seam RESTORE. Three sites,
one fault, two of them innocent -- and this shift chased two wrong roots because
of it.

THE CLASS. An exception handler that assumes its failure is survivable. A CUDA
illegal access poisons the context: every later CUDA call in the process raises
the same error at whatever site runs next. Swallowing it does not keep serving
alive; it only decides that the crash will be reported somewhere innocent. The
handler is right for a poll that hiccups (an OOM, a transient) and wrong here,
and it could not tell the two apart because it caught bare `Exception`.

THIS MODULE ALREADY OWNS THE CLASS. Its own docstring opens on #431: a run that
tripped the cap on every collective produced a file with ZERO matching lines, so
"nothing tripped" and "everything tripped" were indistinguishable. The handler
had made "a poll hiccuped" and "the CUDA context is dead" indistinguishable in
exactly the same way, inside the module written to end that.

WHAT IS FIXED HERE is the ATTRIBUTION, not the fault: the origin is recorded and
named, device polling stops for the round, and the watchdog still does not die.
WHICH POINTER IS FREED UNDERNEATH IS NOT ESTABLISHED and nothing here assumes
it -- see docs/dev/NOTE_867_poison_attribution.md.

Hermetic: no CUDA. The poison is a real `torch.AcceleratorError` carrying the
specimen's message, raised by a stub transport.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.distributed.device_communicators import barlink_abort_gate as gate
from sglang.test.test_utils import CustomTestCase

IMA = "CUDA error: an illegal memory access was encountered"


class _Transport:
    def __init__(self, name, behaviour):
        self._name = name
        self._behaviour = behaviour
        self.polls = 0

    def poll_status_word(self):
        self.polls += 1
        if self._behaviour == "ok":
            return False
        if self._behaviour == "trip":
            return True
        if self._behaviour == "benign":
            raise RuntimeError("transient poll failure")
        if self._behaviour == "poison":
            raise torch.AcceleratorError(IMA)
        if self._behaviour == "oom":
            # An AcceleratorError that is RECOVERABLE. The class alone cannot
            # separate this from the poison above; only the message can.
            raise torch.AcceleratorError("CUDA out of memory. Tried to allocate")
        raise AssertionError(self._behaviour)


def _clear():
    """Absent before the fix. Guarded so the CONTROLS below genuinely run on
    unpatched code -- an all-red file proves nothing about which behaviour the
    change is responsible for."""
    fn = getattr(gate, "clear_poison_record", None)
    if fn is not None:
        fn()


def _is_poison(exc):
    fn = getattr(gate, "is_poison_error", None)
    return fn(exc) if fn is not None else False


def _poison():
    fn = getattr(gate, "poison_record", None)
    return fn() if fn is not None else None


class _GateFixture(CustomTestCase):
    def setUp(self):
        gate.reset_for_test()
        _clear()
        self._registered = []

    def tearDown(self):
        for t in self._registered:
            try:
                gate.unregister(t)
            except Exception:
                pass
        _clear()
        gate.reset_for_test()

    def _register(self, *transports):
        for t in transports:
            gate.register(t)
            self._registered.append(t)
        return transports


class TestPoisonIsDistinguishedFromAHiccup(_GateFixture):
    def test_a_poison_fault_is_recorded_as_the_origin(self):
        (bad,) = self._register(_Transport("bad", "poison"))
        gate.poll_status_words()
        rec = _poison()
        self.assertIsNotNone(
            rec,
            "an unsurvivable CUDA fault must leave a record naming itself as "
            "the origin; without it the crash that lands later is attributed to "
            "an innocent site",
        )
        self.assertIn(IMA, rec["error"])
        self.assertIn("poll_status_word", rec["source"])

    def test_a_benign_poll_failure_is_still_only_swallowed(self):
        """CONTROL. The handler exists for this case and must not change."""
        (bad,) = self._register(_Transport("bad", "benign"))
        gate.poll_status_words()
        self.assertIsNone(_poison())

    def test_a_recoverable_oom_is_not_treated_as_poison(self):
        """CONTROL, and the reason the check reads the MESSAGE not the class:
        torch raises AcceleratorError for OOM too, and an OOM is survivable."""
        (bad,) = self._register(_Transport("bad", "oom"))
        gate.poll_status_words()
        self.assertIsNone(
            _poison(),
            "matching on the exception class alone would over-refuse here",
        )

    def test_polling_stops_after_a_poison_but_continues_after_a_hiccup(self):
        bad, later = self._register(
            _Transport("bad", "poison"), _Transport("later", "ok")
        )
        gate.poll_status_words()
        self.assertEqual(bad.polls, 1)
        self.assertEqual(
            later.polls,
            0,
            "after the context is poisoned every further device read returns "
            "the same error and multiplies the misattribution",
        )

    def test_a_hiccup_does_not_stop_the_round(self):
        """CONTROL: the can-fail direction of the line above."""
        bad, later = self._register(
            _Transport("bad", "benign"), _Transport("later", "trip")
        )
        tripped = gate.poll_status_words()
        self.assertEqual(later.polls, 1)
        self.assertEqual(tripped, 1)

    def test_the_first_poison_wins_not_the_latest(self):
        """The record is the ORIGIN. A record that tracked the newest fault
        would name the innocent site, which is the whole defect."""
        (bad,) = self._register(_Transport("bad", "poison"))
        gate.poll_status_words()
        first = _poison()
        gate.poll_status_words()
        self.assertIsNotNone(first)
        self.assertEqual(_poison()["monotonic"], first["monotonic"])

    def test_a_clean_round_records_nothing(self):
        """CONTROL."""
        (ok,) = self._register(_Transport("ok", "ok"))
        gate.poll_status_words()
        self.assertIsNone(_poison())
        self.assertEqual(ok.polls, 1)


class TestTheClassifierItself(_GateFixture):
    def test_it_matches_the_specimen_text(self):
        self.assertTrue(_is_poison(torch.AcceleratorError(IMA)))

    def test_it_matches_the_other_context_killers(self):
        for msg in (
            "CUDA error: unspecified launch failure",
            "CUDA error: misaligned address",
            "device-side assert triggered",
        ):
            self.assertTrue(_is_poison(RuntimeError(msg)), msg)

    def test_it_does_not_match_recoverable_faults(self):
        """CONTROL, both directions of the classifier can fail."""
        for msg in (
            "CUDA out of memory. Tried to allocate 2.00 GiB",
            "transient poll failure",
            "NCCL timeout",
        ):
            self.assertFalse(_is_poison(RuntimeError(msg)), msg)


if __name__ == "__main__":
    unittest.main()
