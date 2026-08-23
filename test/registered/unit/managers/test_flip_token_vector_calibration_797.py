"""#797 -- the flip's TP stack consumes the MEASURED token vector, not the seed.

THE DEFECT, stated once. ``build_phase_flip_tp_stack`` parses a SEED vector
(``parse_flip_token_vector``: --phase-flip-tp-vector, or
SGLANG_UNEVEN_TOKEN_VECTOR), installs it process-globally, and then builds the
TP decode worker. That construction reaches the install-capable calibration
site (``_resolve_memory_pool_config`` -> ``_maybe_suggest_dcp_token_vector``
with ``allow_install=True``) and may replace the global vector with the
MEASURED optimum -- the tree records that this site is really reached on this
rig, with all three ranks logging ``allow_install=True`` at ``dcp_size=3``.
The TP pools, allocator, backends and graphs are then built under whatever
vector is active at that point.

``PhaseFlipStacks.token_vector`` was nevertheless frozen from the pre-
construction seed, and ``phase_flip_runtime._cutover`` reinstalls that value at
every flip. Two consequences:

  1. the measured vector never reached the decode phase, so the flip served an
     estimate the calibration had already superseded;
  2. worse, the owner rule was pointed at a DIFFERENT vector than the pools
     were sized under -- the out-of-bounds slot id the cutover's own comment
     warns about, reached by a stale TOKEN vector rather than by the weight
     vector it anticipated.

WHAT THIS IS WORTH TODAY: no additional pool capacity. The phase-flip boot caps
every TP rank at the PP id space, which makes rank 0 binding under both the
seed and the measured vector; the vector cannot pay until that cap moves
(threshold cap0 > 605056). This closes a mechanism-present/actuator-missing gap
and removes a latent owner-rule mismatch. It does not raise a pool.

THE EDGE INSTALLS NOTHING. ``allow_install`` inside the calibration remains the
single authority for when a vector goes live; this code only reads back what
that authority decided. That is also why it carries no collective: the install
decision is already rank-uniform, so every rank reads the identical value from
its own process-global state. The absence of a collective is pinned below
rather than asserted in prose, because a rank-local verdict would hang the
group.

Covered here:
  1. the three verdict states, apart, each naming its own cause;
  2. the read-back seam against real process-global state (the regression);
  3. the provenance rule in BOTH directions, including why a measured vector
     is exempt from a value-match against the retracted register;
  4. rank-uniformity without a collective;
  5. the call site: the stack must carry the verdict, not the seed.
"""

import inspect
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.distributed.utils import set_cp_token_ratios
from sglang.srt.managers import phase_flip_boot as pfb
from sglang.srt.planner.retracted import REGISTER, RetractedProvenanceError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROLE_ENV = "SGLANG_UNEVEN_TOKEN_VECTOR_ROLE"
_PROV_ENV = "SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE"

# A vector the retracted register really carries, taken from the register
# itself rather than hardcoded -- a literal here would silently stop testing
# the rule the day the register is edited.
_RETRACTED_ENTRY = next(e for e in REGISTER if e.token_vectors)
_RETRACTED_VEC = list(_RETRACTED_ENTRY.token_vectors[0])
_CLEAN_VEC = [7, 39, 18]


def _server_args(role="pin", provenance=None):
    return SimpleNamespace(
        uneven_token_vector_role=role,
        uneven_token_vector_provenance=provenance,
    )


class _EnvClean(CustomTestCase):
    """Both provenance envs are process-global and read at call time."""

    def setUp(self):
        super().setUp()
        self._saved = {k: os.environ.get(k) for k in (_ROLE_ENV, _PROV_ENV)}
        for k in (_ROLE_ENV, _PROV_ENV):
            os.environ.pop(k, None)
        self.addCleanup(self._restore)
        set_cp_token_ratios(None)
        self.addCleanup(set_cp_token_ratios, None)

    def _restore(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestVerdictStates(CustomTestCase):
    """Pin 1: three states, each with its own cause. Pure function, no state."""

    def test_holds_when_the_calibration_left_the_seed_alone(self):
        v = pfb.resolve_effective_flip_token_vector([29, 19, 16], [29, 19, 16])
        self.assertEqual(v.state, pfb.FLIP_VECTOR_HOLDS)
        self.assertEqual(v.vector, (29, 19, 16))
        self.assertEqual(v.installed, (29, 19, 16))
        self.assertIn("left", v.reason)

    def test_recalibrated_carries_the_measured_vector_not_the_seed(self):
        v = pfb.resolve_effective_flip_token_vector([29, 19, 16], [7, 39, 18])
        self.assertEqual(v.state, pfb.FLIP_VECTOR_RECALIBRATED)
        # THE REGRESSION: the carried vector is the one the pools were built
        # under. Returning the seed here is the pre-#797 behaviour.
        self.assertEqual(v.vector, (7, 39, 18))
        self.assertEqual(v.seed, (29, 19, 16))

    def test_undecided_on_nothing_installed_keeps_the_seed(self):
        v = pfb.resolve_effective_flip_token_vector([29, 19, 16], None)
        self.assertEqual(v.state, pfb.FLIP_VECTOR_UNDECIDED)
        self.assertEqual(v.vector, (29, 19, 16))
        self.assertIsNone(v.installed)

    def test_undecided_on_length_disagreement_keeps_the_seed(self):
        # Adopting a wrong-length vector would convert an unclear read-back
        # into a certain out-of-bounds slot id.
        v = pfb.resolve_effective_flip_token_vector([29, 19, 16], [1, 1])
        self.assertEqual(v.state, pfb.FLIP_VECTOR_UNDECIDED)
        self.assertEqual(v.vector, (29, 19, 16))
        self.assertEqual(v.installed, (1, 1))

    def test_the_three_states_are_distinguishable(self):
        # The point of the split: no two of them collapse onto one value, and
        # every one of them explains itself.
        seen = {
            pfb.resolve_effective_flip_token_vector([1, 2, 3], [1, 2, 3]).state,
            pfb.resolve_effective_flip_token_vector([1, 2, 3], [3, 2, 1]).state,
            pfb.resolve_effective_flip_token_vector([1, 2, 3], None).state,
        }
        self.assertEqual(len(seen), 3)
        for installed in ([1, 2, 3], [3, 2, 1], None, [9]):
            v = pfb.resolve_effective_flip_token_vector([1, 2, 3], installed)
            self.assertTrue(v.reason.strip(), f"no cause given for {v.state}")


class TestReadBackSeam(_EnvClean):
    """Pin 2: the seam against real process-global state."""

    def test_reads_back_the_installed_vector(self):
        set_cp_token_ratios(_CLEAN_VEC)
        v = pfb.effective_flip_token_vector(_server_args(), [29, 19, 16])
        self.assertEqual(v.state, pfb.FLIP_VECTOR_RECALIBRATED)
        self.assertEqual(v.vector, tuple(_CLEAN_VEC))

    def test_seed_stands_when_nothing_was_installed(self):
        set_cp_token_ratios(None)
        v = pfb.effective_flip_token_vector(_server_args(), _CLEAN_VEC)
        self.assertEqual(v.state, pfb.FLIP_VECTOR_UNDECIDED)
        self.assertEqual(v.vector, tuple(_CLEAN_VEC))

    def test_installs_nothing_itself(self):
        # The edge must not become a second installation authority: the
        # process-global vector is exactly as it was before the call.
        set_cp_token_ratios(_CLEAN_VEC)
        with mock.patch.object(
            pfb,
            "resolve_effective_flip_token_vector",
            wraps=pfb.resolve_effective_flip_token_vector,
        ):
            pfb.effective_flip_token_vector(_server_args(), [29, 19, 16])
        from sglang.srt.distributed.utils import get_cp_token_ratios

        self.assertEqual(get_cp_token_ratios(), _CLEAN_VEC)


class TestProvenanceBothDirections(_EnvClean):
    """Pin 3: a retracted lineage is refused, a clean one passes."""

    def test_retracted_seed_is_refused(self):
        set_cp_token_ratios(_RETRACTED_VEC)  # HOLDS -> the seed is carried
        with self.assertRaises(RetractedProvenanceError) as cm:
            pfb.effective_flip_token_vector(_server_args(), _RETRACTED_VEC)
        self.assertIn(_RETRACTED_ENTRY.investigation, str(cm.exception))

    def test_clean_seed_passes(self):
        set_cp_token_ratios(_CLEAN_VEC)
        v = pfb.effective_flip_token_vector(_server_args(), _CLEAN_VEC)
        self.assertEqual(v.state, pfb.FLIP_VECTOR_HOLDS)
        self.assertEqual(v.vector, tuple(_CLEAN_VEC))

    def test_measured_vector_is_exempt_from_the_value_match(self):
        # A recalibrated vector came from THIS boot's profiling, which is what
        # 'measured' provenance means. Asking the register about it would match
        # it by value and refuse a measurement for resembling a withdrawn
        # estimate -- so the exemption is substantive, not a convenience.
        set_cp_token_ratios(_RETRACTED_VEC)
        v = pfb.effective_flip_token_vector(_server_args(), _CLEAN_VEC)
        self.assertEqual(v.state, pfb.FLIP_VECTOR_RECALIBRATED)
        self.assertEqual(v.vector, tuple(_RETRACTED_VEC))

    def test_retracted_seed_under_role_seed_warns_instead_of_refusing(self):
        # The established split: a seed is superseded in-process, so it is
        # permitted past this gate and enforced at the install site instead.
        os.environ[_ROLE_ENV] = "seed"
        set_cp_token_ratios(_RETRACTED_VEC)
        v = pfb.effective_flip_token_vector(_server_args(role="seed"), _RETRACTED_VEC)
        self.assertEqual(v.state, pfb.FLIP_VECTOR_HOLDS)


class TestRankUniformityWithoutCollective(_EnvClean):
    """Pin 4: same answer on every rank, and no collective to hang in."""

    def test_no_collective_is_entered(self):
        import torch

        def _explode(*a, **k):
            raise AssertionError(
                "the calibration edge entered a collective; a rank-local "
                "verdict around one would hang the group"
            )

        set_cp_token_ratios(_CLEAN_VEC)
        with (
            mock.patch.object(torch.distributed, "all_gather_object", _explode),
            mock.patch.object(torch.distributed, "all_reduce", _explode),
        ):
            pfb.effective_flip_token_vector(_server_args(), [29, 19, 16])

    def test_every_rank_reaches_the_same_verdict(self):
        # The install decision is rank-uniform, so the read-back is too. Three
        # simulated ranks reading the same global state must agree on all of
        # state, vector and cause.
        set_cp_token_ratios(_CLEAN_VEC)
        verdicts = [
            pfb.effective_flip_token_vector(_server_args(), [29, 19, 16])
            for _ in range(3)
        ]
        self.assertEqual(len({v.state for v in verdicts}), 1)
        self.assertEqual(len({v.vector for v in verdicts}), 1)
        self.assertEqual(len({v.reason for v in verdicts}), 1)


class TestCallSiteCarriesTheVerdict(CustomTestCase):
    """Pin 5: the stack is constructed from the verdict, not from the seed.

    A source pin, because the surrounding construction needs real weights and
    devices. It is narrow on purpose: it fails if the seed is ever wired back
    into PhaseFlipStacks, which is the exact regression.
    """

    def test_stacks_token_vector_comes_from_the_verdict(self):
        src = inspect.getsource(pfb.build_phase_flip_tp_stack)
        self.assertIn("effective_flip_token_vector(server_args, tok_vec)", src)
        self.assertIn("token_vector=token_verdict.vector", src)
        self.assertNotIn("token_vector=tuple(tok_vec)", src)


if __name__ == "__main__":
    unittest.main()
