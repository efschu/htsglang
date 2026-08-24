"""#853(i) -- the EXPOSURE law must not stop exactly when flips stop.

THE W24 SPECIMEN, and it is the reason observability alone does not close this
ticket. From the window result::

    "#851 EXPOSURE ENFORCED"   0   UNDETERMINABLE -- grep finds a SINGLE call
    site (:9989, at cutover). So "EXPOSURE ENFORCED"=0 provably means "never
    ran" for the entire stuck phase (zero cutovers).

The stuck phase is 23.6 minutes long, carries 153 arm attempts, and contains
ZERO cutovers. An enforcement wired only to the cutover therefore has a duty
cycle equal to the thing it is supposed to police: it runs when flips are
healthy and goes quiet for exactly the window where the pool is worth checking.
A gate that is unreachable in the failure mode is not a gate.

WHY THE ARM IS A LAWFUL PLACE TO RUN IT. Established against the tree before
wiring, because this is a distributed actuator:

  * NO COLLECTIVE. `kv_backing_relief.py` contains no collective anywhere --
    stated as a checkable claim in `phase_flip_spill.py`. `_exposure_ceiling`
    reads `_group_backed_floor`, a CACHED already-agreed value written by
    `note_group_backing_floor`; it never triggers a reduction. So ranks may
    call it independently, at different instants, without divergent
    participation -- the property an arm-time call needs and a cutover-time
    one gets for free.
  * NO QUIESCENCE NEEDED. `KvRowCap` is "non-destructive by construction: live
    allocations are not enumerated, not moved and not touched. Only unallocated
    ids are held back", so engaging a cap cannot invalidate a row a request is
    using. The arm runs with requests live; this is safe there.
  * NOTHING TO RACE. `exposed > backed` is never a sanctioned mid-grow state --
    a grow "adds BACKED rows which stay UNEXPOSED until a later collective
    raises the level". It is the #816/#833 defect state, so an arm-time clamp
    can only ever be correcting a defect.
  * PRECEDENT. The same actuator already has four call sites on non-quiescent
    paths (after a failed shrink, after a successful shrink, inside `recover()`
    on the ordinary round path, and after the cap agreement).

WHAT THIS FILE PINS. Reachability as a property of the FIX LAYER (NOTE_851's
instrument-correction rule): the enforcement is invoked from the arm, not only
from the cutover. It is asserted against the shipped call graph rather than by
counting log lines, because the log count is precisely the reading W24 showed
to be multi-valued.

Hermetic: no CUDA, no NVML, no pool. CUDA_VISIBLE_DEVICES="".
"""

import inspect
import unittest

from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

ENFORCE = "_enforce_exposure_at_seam"


def _callers_of(name):
    """Methods of PhaseFlipRuntime whose source calls ``name``."""
    out = set()
    for attr, value in vars(PhaseFlipRuntime).items():
        if not callable(value):
            continue
        try:
            src = inspect.getsource(value)
        except (OSError, TypeError):  # pragma: no cover - builtins/slots
            continue
        if attr == name:
            continue
        if f"self.{name}(" in src:
            out.add(attr)
    return out


class TheExposureLawIsReachableWhenNoFlipCompletes(unittest.TestCase):
    def test_the_arm_enforces_exposure(self):
        """THE W24 GAP. 153 arms, 0 cutovers, 0 enforcement runs."""
        self.assertIn(ENFORCE, inspect.getsource(PhaseFlipRuntime.arm))

    def test_enforcement_is_not_reachable_from_the_cutover_alone(self):
        """The property stated as W24 stated it: a SINGLE call site was the
        defect. Two call sites that are both on the cutover path would satisfy
        a naive count while leaving the duty cycle unchanged, so this asserts
        that a caller exists which is NOT the cutover."""
        callers = _callers_of(ENFORCE)
        self.assertIn("arm", callers)
        self.assertTrue(
            callers - {"_cutover", "arm"} or "arm" in callers,
            f"enforcement reachable only from {sorted(callers)}",
        )

    def test_the_arm_names_its_own_seam_event(self):
        """A marker shared with the cutover would make the two runs
        indistinguishable in the log, which is #853(i)'s other half undone."""
        src = inspect.getsource(PhaseFlipRuntime.arm)
        idx = src.index(ENFORCE)
        call = src[idx : idx + 200]
        self.assertIn("arm", call)


class TheEnforcementStaysHarmlessWhereItIsNowReached(unittest.TestCase):
    """The danger direction. The arm runs with requests LIVE and must never be
    taken down by a capacity instrument -- `arm` returns a (bool, str) contract
    its callers rely on."""

    class _Exploding:
        def clamp_exposure_to_backing(self, why):
            raise RuntimeError("the clamp exploded")

    def test_a_raising_clamp_does_not_escape_the_seam(self):
        import types

        from sglang.srt.managers.phase_flip_spill import KV_BACKING_RELIEF_ATTR

        r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
        sched = types.SimpleNamespace()
        setattr(sched, KV_BACKING_RELIEF_ATTR, self._Exploding())
        r._census_scheduler = sched
        self.assertEqual(r._enforce_exposure_at_seam("tp_to_pp arm"), 0)


if __name__ == "__main__":
    unittest.main()
