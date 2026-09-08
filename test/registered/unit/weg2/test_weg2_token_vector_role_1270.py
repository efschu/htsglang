# SPDX-License-Identifier: Apache-2.0
"""#1270: a launcher ESTIMATE is not a pin, and it arms its own supersession.

THE DEFECT, measured on boot weg2sb1 (tip 7b60280f57, D.log:355-382). Group D
shipped no token vector, no env and no --rank-kv-ratio, so
``resolve_cp_token_ratios`` fell to its budget-estimate rung and derived
``[30,17,17]``. ``_token_vector_role`` then returned a flat ``or "pin"``, the
install gate reads ``not pinned_vector and (derived_mode or seed_role)``, and
BOTH disjuncts were False -- so the measured optimum ``[17,7,8]`` was printed
as a restart hint and the boot served 574,336 tokens where rg6, on the same
rig and the same code path, installed the same optimum and served 681,856
(-15.8 %).

rg6's ONLY arming key was ``role='seed'``, supplied by the
``--uneven-token-vector 29,19,16`` that #1032 deleted for a good reason (a
retracted #602 lineage). The seed was never the right way to arm the install
-- it was the way that happened to work.

WHAT IS PINNED HERE

* THE ROLE OF AN UNDECLARED VECTOR IS ``estimate``, never ``pin``. A default
  is not an assertion.
* ``estimate`` arms the measured install exactly as ``seed`` does, so the sb1
  shape -- no env, no seed, ``derived_mode`` False -- reaches the install.
* A DECLARED vector still defaults to ``pin``: the operator's assertion is
  untouched, which is the property that makes this safe.
* ONE reader. The default lived in three places and the mixin's copy is where
  the estimate lost its install.
* The hint must not offer a bare ``SGLANG_UNEVEN_TOKEN_VECTOR=`` restart --
  that freezes the numbers into the mechanism that stops recomputing them.

MUTANT (stated so the record can be checked): reverting
``_token_vector_role``'s default to a flat ``or "pin"`` turns
``test_an_undeclared_vector_is_an_estimate`` and
``test_the_sb1_shape_arms_the_measured_install`` red -- they are the two that
carry the fix.

Hermetic: no GPU, no server, no NVML; the env is saved and restored per test.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.distributed import utils as dist_utils
from sglang.test.test_utils import CustomTestCase

_ENV = ("SGLANG_UNEVEN_TOKEN_VECTOR", "SGLANG_UNEVEN_TOKEN_VECTOR_ROLE")

def _mixin_method_source(name: str) -> str:
    """The source of one method of the KV-cache mixin, by name.

    Read from the module FILE via ast rather than through an attribute: the
    module defines a mixin class, not ``ModelRunner`` (the ``self: ModelRunner``
    annotation is a type hint, not where the method lives), and hard-coding the
    class name would make this guard break on a rename rather than on the
    property it is about.
    """
    import ast
    import inspect

    from sglang.srt.model_executor import model_runner_kv_cache_mixin as mixin

    src = inspect.getsource(mixin)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"{name} not found in the mixin module")




class _Args:
    """Exactly the ServerArgs surface the role resolver reads."""

    def __init__(self, rank_kv_ratio=None, role="", provenance=""):
        self.rank_kv_ratio = rank_kv_ratio
        self.uneven_token_vector_role = role
        self.uneven_token_vector_provenance = provenance


class _EnvScope(CustomTestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _ENV}
        for k in _ENV:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestTheRoleOfAnUndeclaredVector(_EnvScope):

    def test_an_undeclared_vector_is_an_estimate(self):
        """THE FIX. The sb1 shape: nothing in the env, no --rank-kv-ratio."""
        self.assertEqual(
            dist_utils.token_vector_role(_Args()), dist_utils.ROLE_ESTIMATE
        )

    def test_a_mode_string_is_not_a_declaration(self):
        # --rank-kv-ratio capacity ASKS for a derivation; it does not supply a
        # vector, so it cannot make the derived vector an assertion.
        for mode in ("capacity", "coupled", "speed"):
            self.assertFalse(dist_utils.token_vector_is_declared(_Args(mode)))
            self.assertEqual(
                dist_utils.token_vector_role(_Args(mode)), dist_utils.ROLE_ESTIMATE
            )

    def test_a_declared_vector_still_defaults_to_pin(self):
        """THE SAFETY HALF: an operator's assertion is untouched. Both spellings
        of a declaration -- the env value and a --rank-kv-ratio LIST."""
        os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "29,19,16"
        self.assertTrue(dist_utils.token_vector_is_declared(_Args()))
        self.assertEqual(dist_utils.token_vector_role(_Args()), dist_utils.ROLE_PIN)
        os.environ.pop("SGLANG_UNEVEN_TOKEN_VECTOR")
        self.assertEqual(
            dist_utils.token_vector_role(_Args([29, 19, 16])), dist_utils.ROLE_PIN
        )

    def test_an_explicit_role_always_wins(self):
        for role in ("seed", "pin", "estimate"):
            os.environ["SGLANG_UNEVEN_TOKEN_VECTOR_ROLE"] = role
            self.assertEqual(dist_utils.token_vector_role(_Args()), role)
        os.environ.pop("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE")
        self.assertEqual(
            dist_utils.token_vector_role(_Args(role="seed")), dist_utils.ROLE_SEED
        )

    def test_an_empty_env_role_is_unstated_not_pin(self):
        # The pre-existing rule this fix must not break: a blank override reads
        # as "not stated" and defers, rather than silently meaning pin.
        os.environ["SGLANG_UNEVEN_TOKEN_VECTOR_ROLE"] = "   "
        self.assertEqual(
            dist_utils.token_vector_role(_Args(role="seed")), dist_utils.ROLE_SEED
        )
        self.assertEqual(dist_utils.token_vector_role(_Args()), dist_utils.ROLE_ESTIMATE)


class TestTheInstallArming(_EnvScope):

    def test_the_sb1_shape_arms_the_measured_install(self):
        """THE POINT OF THE FIX, as the boot's own inputs: no env, no seed, no
        --rank-kv-ratio list. derived_mode is False on that argv, so this
        predicate is the ONLY thing that can arm the install -- and the answer
        to 'is --rank-kv-ratio capacity also needed' is therefore NO."""
        self.assertTrue(dist_utils.token_vector_arms_measured_install(_Args()))

    def test_a_seed_still_arms_it(self):
        # rg6's path, unchanged.
        os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "29,19,16"
        os.environ["SGLANG_UNEVEN_TOKEN_VECTOR_ROLE"] = "seed"
        self.assertTrue(dist_utils.token_vector_arms_measured_install(_Args()))

    def test_a_pin_still_suppresses_it(self):
        os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "29,19,16"
        self.assertFalse(dist_utils.token_vector_arms_measured_install(_Args()))

    def test_the_install_gate_asks_this_predicate_and_not_its_own_default(self):
        """ONE READER. The mixin used to spell `or "pin"` itself, and that copy
        is where the estimate lost its install -- so the absence of a second
        spelling is the assertion, not a style preference."""
        src = _mixin_method_source("_maybe_suggest_dcp_token_vector")
        self.assertIn("token_vector_arms_measured_install", src)
        # The defect shape precisely: this method must not READ the role env
        # itself, because reading it means defaulting it, and defaulting it
        # here is what disagreed with utils. Asserted on the executable lines
        # only -- prose that NAMES the removed spelling is the record of the
        # fix, not a second copy of it.
        code = "\n".join(
            ln for ln in src.split("\n")
            if not ln.lstrip().startswith("#")
        )
        self.assertNotIn("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", code)


class TestTheAdvisoryDoesNotTrapTheReader(_EnvScope):

    def test_the_hint_never_offers_a_bare_env_restart(self):
        """#1270 (d). Following the old advice was a one-way door: an env
        vector is DECLARED, a declared vector defaults to pin, and a pin
        suppresses the calibration that produced the advice."""
        src = _mixin_method_source("_maybe_suggest_dcp_token_vector")
        self.assertNotIn("restart with SGLANG_UNEVEN_TOKEN_VECTOR=%s", src)
        # ... and it must name the two routes that DO arm the install.
        self.assertIn("--rank-kv-ratio capacity", src)
        self.assertIn("--uneven-token-vector-role seed", src)

    def test_the_loss_warning_is_not_gated_on_a_pin(self):
        """#1270 (c). weg2sb1 is precisely the case the old `if pinned_vector`
        gate excluded: nothing pinned, 15.8 % short, and the line that says so
        did not print."""
        src = _mixin_method_source("_maybe_suggest_dcp_token_vector")
        self.assertNotIn("if pinned_vector and c_active > 0", src)
        self.assertIn("COSTING", src)


class TestTheOtherRoleReadersAreUnreachableFromAnEstimate(_EnvScope):
    """The safety argument, asserted rather than reasoned about.

    Three existing sites branch on ``role == 'seed'`` -- the retracted-vector
    refusal, the precedence reporter and the seed-supersession arming. All
    three run ONLY on a path where an explicit vector exists, so the new
    default cannot reach them. If that ever stops being true, this fails.
    """

    def test_every_seed_branch_sits_behind_an_explicit_vector(self):
        import inspect

        src = inspect.getsource(dist_utils)
        # The estimate rung returns before any of them could apply: it is the
        # LAST rung, below env, flag-list and planner seed.
        self.assertIn("def token_vector_is_declared", src)
        # An undeclared vector is never handed to the retracted-provenance
        # refusal, because that is called with an explicit source string.
        for caller in ("--rank-kv-ratio", "SGLANG_UNEVEN_TOKEN_VECTOR"):
            self.assertIn(caller, src)

    def test_a_retracted_vector_is_still_refused_as_a_pin(self):
        # The estimate role must not become a way to smuggle a retracted
        # vector past the refusal: retracted checks run on DECLARED vectors,
        # which still default to pin.
        os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "29,19,16"
        self.assertEqual(dist_utils.token_vector_role(_Args()), dist_utils.ROLE_PIN)


if __name__ == "__main__":
    unittest.main()
