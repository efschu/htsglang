"""#721: host free-headroom guard on the flip staging transient.

Two host-OOM-shaped kills on this box landed 7 s and 11 s after a COMPLETED
flip (03:22:17 and 03:31:17), and a third kill at 21:24 is ledger-confirmed as
the kernel OOM killer. The attribution needs host journal access this LXC does
not have (dmesg_restrict=1, no /dev/kmsg, container journal carries no kernel
ring), so the guard ships as the DISCRIMINATOR as well as the defense: every
firing produces the terms the journal would have given us.

  * fires, and no kill follows      -> the flip-transient candidate gains
  * kill happens despite ample headroom reported here -> that candidate dies
    and lane RSS spikes gain

Failure-if-wrong is benign (one deferred flip, logged with every term).
Failure-if-right is an unrecoverable group kill. That asymmetry is why it is
default-on ahead of full attribution.
"""

import unittest

from sglang.test.test_utils import CustomTestCase

GB = 1024**3


def _v(avail, transient=20 * GB, defers=0, **kw):
    from sglang.srt.managers.phase_flip_runtime import flip_host_headroom_verdict

    return flip_host_headroom_verdict(avail, transient, defers, **kw)


class TestFlipHostHeadroomGuard721(CustomTestCase):
    def test_healthy_headroom_allows_and_does_not_name_a_defer(self):
        """CAN-FAIL: a guard that deferred on healthy state would pass every
        firing test below and still be useless -- it must stay silent here."""
        allow, escalated, detail = _v(40 * GB)
        self.assertTrue(allow)
        self.assertFalse(escalated)
        self.assertNotIn("DEFERRED-HOST-RAM", detail, detail)
        self.assertIn("OK", detail)

    def test_low_headroom_defers(self):
        allow, escalated, detail = _v(10 * GB)
        self.assertFalse(allow, detail)
        self.assertFalse(escalated)
        self.assertIn("DEFERRED-HOST-RAM", detail)

    def test_the_boundary_is_transient_plus_floor(self):
        """Exactly enough passes; one byte short defers. The floor is not
        decoration -- it is the whole margin."""
        from sglang.srt.managers.phase_flip_runtime import FLIP_HOST_RAM_FLOOR_BYTES

        need = 20 * GB + FLIP_HOST_RAM_FLOOR_BYTES
        self.assertTrue(_v(need)[0])
        self.assertFalse(_v(need - 1)[0])

    def test_unreadable_host_ram_stands_the_guard_DOWN(self):
        """None means no honest number. Refusing a flip on a fabricated figure
        is worse than not checking, because the refusal is what costs service."""
        allow, escalated, detail = _v(None)
        self.assertTrue(allow)
        self.assertFalse(escalated)
        self.assertIn("stood down", detail)

    def test_defer_is_bounded_then_escalates_and_proceeds(self):
        """A permanent hold is worse than the hazard: it converts a POSSIBLE
        kill into a CERTAIN half-service outage, and the kill is recoverable."""
        from sglang.srt.managers.phase_flip_runtime import FLIP_HOST_RAM_MAX_DEFERS

        for n in range(FLIP_HOST_RAM_MAX_DEFERS):
            with self.subTest(defers=n):
                allow, escalated, _ = _v(1 * GB, defers=n)
                self.assertFalse(allow, f"defer {n} must still hold")
                self.assertFalse(escalated)
        allow, escalated, detail = _v(1 * GB, defers=FLIP_HOST_RAM_MAX_DEFERS)
        self.assertTrue(allow, "must proceed rather than hold forever")
        self.assertTrue(escalated)
        self.assertIn("ESCALATED", detail)
        self.assertIn("PROCEEDING WITH EYES OPEN", detail)

    def test_every_term_is_quoted_so_the_claim_is_checkable(self):
        """The guard is the discriminator; a verdict without its numbers
        collects nothing."""
        _, _, detail = _v(10 * GB, transient=20 * GB)
        for token in ("available", "needed", "transient", "floor"):
            self.assertIn(token, detail, f"{token} missing: {detail}")

    def test_defer_reason_is_the_shared_constant(self):
        """#696 INTERACTION, named not absorbed: a host-RAM defer must be
        visible to the SLO / ARM-UNFUNDED accounting under its OWN reason. A
        flip that did not arm because the HOST was tight is a different fact
        from one that could not fund its VRAM seam, and merging them hides the
        signal #721 exists to collect."""
        from sglang.srt.managers.phase_flip_runtime import DEFERRED_HOST_RAM

        self.assertEqual(DEFERRED_HOST_RAM, "DEFERRED-HOST-RAM")
        self.assertIn(DEFERRED_HOST_RAM, _v(1 * GB)[2])

    def test_zero_and_negative_inputs_do_not_crash(self):
        for avail, tr in ((0, 0), (0, -5), (-1, 10 * GB)):
            with self.subTest(avail=avail, transient=tr):
                allow, _, detail = _v(avail, transient=tr)
                self.assertIsInstance(allow, bool)
                self.assertTrue(detail)


if __name__ == "__main__":
    unittest.main()
