"""#656: the kvso-under-PP refusal has to say what it COSTS, not just that it refuses.

WHAT THE REFUSAL WAS
--------------------
    "--enable-kv-session-offload (S1) supports single-node pure TP/DCP only
     (pp_size=3, dp_size=1)."

Accurate and useless. Under ``pp_size>1`` there is no argv that produces a
vacate line, so the flip-setup capacity spec's axis -- "bs2-4 reserves,
INCLUDING unused mamba states, are spilled during bs1 time" -- is
STRUCTURALLY UNREACHABLE on a flip boot (MERGE-R9 12.5). An operator reading
the old message learns that a flag is unavailable; what they need to learn is
that a named acceptance axis cannot be satisfied on this topology at all.

THE SUBSTITUTION THIS PREVENTS
------------------------------
The obvious thing to reach for next is ``--phase-flip-spill-depth``, and it
IS real spilling that works under PP. It is not the same thing: the ladder
gives up the INACTIVE layout's cold memory at the flip seam (cached allocator
segments -> draft weights -> weights-arena tail), which is flip-seam spilling,
not the idle-session mamba vacate the spec axis asks for. Reporting its rung
count as satisfying that axis is exactly the substitution the contradictions
register exists to prevent -- so the message names the alternative AND names
what the alternative is not, in the same breath.

The refusal itself is NOT lifted here and this file does not ask for that.
Its stated reason -- the host pool's rows are sized from the boot vector -- is
real, and the route out is a PP-safe idle-session SOURCE (``GdnStateStore``
is an interface, so the dependency is a sourcing one), not a wider refusal.

CAN-FAIL PROOF: revert the message to the one-sentence form above and every
test in TestTheRefusalCarriesTheSpecNote goes red.

Run:
  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python python -m pytest \
      test/registered/unit/server_args/test_kvso_pp_spec_note_656.py -q
"""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _refusal(**over) -> str:
    """The kvso validation's message for a PP launch, or '' if it allowed it."""
    kw = dict(model_path="dummy", enable_kv_session_offload=True, pp_size=3)
    kw.update(over)
    args = ServerArgs(**kw)
    try:
        args._handle_kv_session_offload()
    except ValueError as e:
        return str(e)
    return ""


class TestTheRefusalStillRefuses(unittest.TestCase):
    def test_pp_is_refused(self):
        self.assertIn("single-node pure", _refusal())

    def test_dp_is_refused_on_the_same_check(self):
        self.assertIn("single-node pure", _refusal(pp_size=1, dp_size=2))

    def test_pure_tp_is_not_refused_by_this_check(self):
        """The guard may not widen. A message change that also changed who is
        refused would be a behaviour change wearing a documentation commit."""
        self.assertEqual("", _refusal(pp_size=1, dp_size=1))


class TestTheRefusalCarriesTheSpecNote(unittest.TestCase):
    def test_it_names_the_spec_axis_it_makes_unreachable(self):
        msg = _refusal()
        self.assertIn("SPEC NOTE", msg)
        self.assertIn("bs2-4", msg)
        self.assertIn("mamba", msg)
        self.assertIn("bs1", msg)

    def test_it_says_unreachable_rather_than_merely_unsupported(self):
        """ "Unsupported" reads as "not wired yet". The distinction is the
        whole finding: no argv reaches it on this topology."""
        self.assertIn("STRUCTURALLY UNREACHABLE", _refusal())

    def test_it_names_the_pressure_ladder_as_the_alternative(self):
        msg = _refusal()
        self.assertIn("--phase-flip-spill-depth", msg)
        for rung in ("cache", "draft", "arena"):
            self.assertIn(f"'{rung}'", msg)

    def test_it_says_what_the_ladder_is_not(self):
        """The alternative must arrive with its own disclaimer attached.

        Naming a substitute without naming the axis it does not cover is how
        a rung count ends up quoted as an acceptance result.
        """
        msg = _refusal()
        self.assertIn("NOT the idle-session mamba", msg)
        self.assertIn("must not be reported as", msg)

    def test_it_still_states_the_reason_the_refusal_is_real(self):
        self.assertIn("boot vector", _refusal())


class TestTheLadderRungsQuotedAreTheRealOnes(unittest.TestCase):
    def test_the_rung_names_come_from_the_ladder_module(self):
        """A message that names rungs the ladder does not accept sends the
        operator to a flag value that is refused at parse time."""
        from sglang.srt.managers.phase_flip_spill import (
            DEPTH_NAMES,
            IMPLEMENTED_DEPTH,
        )

        msg = _refusal()
        for name, value in DEPTH_NAMES.items():
            if 0 < value <= IMPLEMENTED_DEPTH:
                self.assertIn(
                    f"'{name}'",
                    msg,
                    f"the ladder implements rung {name!r} and the refusal "
                    "does not offer it",
                )
            if value > IMPLEMENTED_DEPTH:
                self.assertNotIn(
                    f"'{name}'",
                    msg,
                    f"rung {name!r} is defined but not wired; offering it "
                    "sends the operator to a value resolve_spill_depth refuses",
                )


if __name__ == "__main__":
    unittest.main()
