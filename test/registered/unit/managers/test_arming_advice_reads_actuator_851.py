"""#851 F5: the advice must read the value the actuator adopted.

W22 armed the #826 solver (`SGLANG_ARMING_FLOOR_SOLVED=1`) and it WORKED --
boot_w22_0824_0656.log:396/411/412 (06:56:43, all three ranks):

    CORRIDOR-GUARD #826 arming floor 1037 MiB, solver-derived,
    corridor ceiling 1229 MiB

1037 fits under 1229. The gate armed. Yet EVERY abandon line in that window
(4864, 4884, 8885 ...) still appended:

    UNSATISFIABLE ARMING FLOOR: the gate would arm at 1331 ... the flip cannot
    arm from a correctly filled card

1331 is the SHIPPED-constant floor (band floor 819 + the shipped 512 reserve),
not the one this boot adopted. `_arming_floor_advice` re-solves from
`cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB` while the actuator reads
`cg.seam_entry_reserve_mib_resolved()`. So on a boot whose gate armed at 1037,
the advice diagnosed an already-solved contradiction 14 times and sent every
reader away from the real defect.

An INDIKATOR-GESETZ instance sitting INSIDE the #770 Defect-A fix: a diagnostic
that describes a state the process is not in. The advice itself is sound -- it
exists to withdraw "just wait for occupancy to drop" when waiting provably
cannot help (18f measured that draining the load did not lift the lock). It was
simply reading the wrong bookkeeper, which is this whole family's signature.

BOTH DIRECTIONS ARE THE POINT, and they are why this is not a one-line change
with a one-line test: with the solver armed the retraction must go SILENT, and
with the solver off -- the shipped default -- it must still FIRE, because on
the shipped constants the contradiction is real (819 + 512 + 192 = 1523 against
a 1229 ceiling).

Hermetic: the advice function is bound to a bare object, no scheduler, no CUDA.
The env flag is set and restored around each case, which is also what makes the
two directions independent rather than order-dependent.
"""

import os
import unittest

from sglang.srt.managers import corridor_guard as cg
from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

SOLVED_ENV = cg.SOLVED_FLOOR_ENV


class _Bare:
    """`_arming_floor_advice` uses no instance state; give it the cheapest self."""


def _advice() -> str:
    return PhaseFlipRuntime._arming_floor_advice(_Bare())


class _EnvGuard:
    def __init__(self, value):
        self._value = value
        self._old = None

    def __enter__(self):
        self._old = os.environ.get(SOLVED_ENV)
        if self._value is None:
            os.environ.pop(SOLVED_ENV, None)
        else:
            os.environ[SOLVED_ENV] = self._value
        return self

    def __exit__(self, *exc):
        if self._old is None:
            os.environ.pop(SOLVED_ENV, None)
        else:
            os.environ[SOLVED_ENV] = self._old
        return False


class TestTheAdviceFollowsTheActuator(unittest.TestCase):
    def test_with_the_solver_armed_the_retraction_is_SILENT(self):
        """RED before F5. W22's exact configuration: gate armed at 1037."""
        with _EnvGuard("1"):
            resolved = cg.seam_entry_reserve_mib_resolved()
            floor = cg.corridor_band_floor_mib() + resolved
            # Precondition, so a change in the shipped band cannot make this
            # test vacuously green: the adopted floor really does fit.
            self.assertLessEqual(floor, cg.corridor_band_ceiling_mib())
            self.assertEqual(_advice(), "")

    def test_with_the_solver_off_the_retraction_still_FIRES(self):
        """CAN-FAIL TWIN. The shipped default IS contradictory; say so.

        Silencing the advice unconditionally would pass the test above and
        destroy the #770 Defect-A fix. This is the assertion that forbids it.
        """
        with _EnvGuard(None):
            reserve = cg.seam_entry_reserve_mib_resolved()
            self.assertEqual(reserve, int(cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB))
            advice = _advice()
            self.assertIn("RETRACTION OF THE RETRY ADVICE", advice)

    def test_the_two_directions_really_differ(self):
        """Guards against both branches collapsing onto one answer."""
        with _EnvGuard("1"):
            armed = _advice()
        with _EnvGuard(None):
            off = _advice()
        self.assertNotEqual(armed, off)

    def test_the_advice_never_raises(self):
        """Its own contract: 'advice that cannot be computed is simply not
        given'. F5 adds a call that CAN raise (ArmingFloorUnsatisfiable), so
        the swallow has to keep holding."""
        with _EnvGuard("definitely-not-a-bool"):
            self.assertIsInstance(_advice(), str)


if __name__ == "__main__":
    unittest.main()
