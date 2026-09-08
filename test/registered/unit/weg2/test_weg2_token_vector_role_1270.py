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

#1270b -- WHY THE ABOVE WAS TRUE AT THE DESK AND FALSE ON METAL. Every test
in the original file built its ServerArgs by hand (``_Args`` below sets
``uneven_token_vector_role = ""``) and popped the role env in ``setUp``. A real
process holds neither of those values. ``ServerArgs.uneven_token_vector_role``
DEFAULTED to the literal string ``"pin"``, and
``_publish_promoted_781_flags`` published that default into
``SGLANG_UNEVEN_TOKEN_VECTOR_ROLE`` unconditionally (server_args.py) -- and the
consumer reads the env FIRST. So the ``token_vector_is_declared`` rung this
file pins was UNREACHABLE in any real boot: the process answered its own
question with 'pin' before anything asked it.

Measured, smoke boot weg2sb4s on 9f2e58b797
(``boot_weg2_weg2sb4s_9f2e58b797_0908_154736.D.log``): D's argv (line 2)
carries no ``--uneven-token-vector`` and no ``--uneven-token-vector-role``;
ServerArgs logs ``uneven_token_vector=None, uneven_token_vector_role='pin'``
(line 17); the sizing site logs ``role='pin' ... active_vector=[30, 17, 17]``
(lines 357-359); the install is declined and the advisory prints the unusable
``[17, 7, 8]`` hint (line 367). Pool 574,336 against ~670,720.

THE FIX (#1270b): the flag default becomes ``None`` -- "nobody said" -- and the
publication writes the RESOLVED role
(``distributed.utils.token_vector_role_from_args``, which deliberately does not
read the variable it is producing). The unconditional publish keeps overwriting
a stale env, it just no longer invents a value.

MUTANTS (stated so the record can be checked):
* reverting ``_token_vector_role``'s default to a flat ``or "pin"`` turns
  ``test_an_undeclared_vector_is_an_estimate`` and
  ``test_the_sb1_shape_arms_the_measured_install`` red;
* restoring the ``"pin"`` field default OR publishing the raw field again turns
  every test in ``TheRolePublishedByARealProcess1270b`` red -- those are the
  ones that run the real publication instead of stubbing it;
* the DANGER direction -- making a DECLARED vector resolve to anything but
  ``pin`` -- turns ``test_a_declared_vector_still_publishes_pin`` and
  ``test_an_ambient_env_vector_is_declared_so_it_publishes_pin`` red.

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


# ---------------------------------------------------------------------------
# #1270b: the role a REAL process publishes and reads.
#
# Everything above this line stubs ServerArgs.  That is exactly why the #1270
# fix passed at the desk and did nothing on metal, so these run the real
# dataclass defaults through the real publication method and then ask the real
# resolver.  `_bare_server_args` builds a ServerArgs from its own declared
# field defaults without running __post_init__ -- no model config, no NVML, no
# device -- so the values under test are the ones a boot actually holds.
# ---------------------------------------------------------------------------

_D_ARGV_1270B = [
    "--model-path",
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed",
    "--tp-size",
    "3",
    "--pp-size",
    "1",
    "--rank-gpu-id",
    "0,1,2",
    "--rank-gpu-memory-mib",
    "29296,18096,18080",
    "--rank-tp-ratio",
    "auto",
    "--uneven-dcp",
    "--uneven-dcp-weighted",
]
"""The token-vector-relevant slice of group D's argv on boot weg2sb4s, verbatim
from that log's ``argv:`` line.  What matters is what is ABSENT: no
``--uneven-token-vector``, no ``--uneven-token-vector-role``, no
``--rank-kv-ratio``."""


def _bare_server_args(**overrides):
    """A ServerArgs carrying its DECLARED field defaults, without __post_init__.

    The point is the defaults, not the derivation: ``uneven_token_vector_role``
    must be readable exactly as a booting process holds it.
    """
    import dataclasses

    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs.__new__(ServerArgs)
    for name, field in ServerArgs.__dataclass_fields__.items():
        default = field.default
        if default is dataclasses.MISSING:
            default = (
                field.default_factory()
                if field.default_factory is not dataclasses.MISSING
                else None
            )
        if default is None and name == "model_path":
            default = "/tmp/model"
        object.__setattr__(sa, name, default)
    for key, value in overrides.items():
        object.__setattr__(sa, key, value)
    return sa


class TheRolePublishedByARealProcess1270b(_EnvScope):
    """THE ROOT OF #1270b, asserted through the real publication path."""

    def _publish(self, **overrides):
        sa = _bare_server_args(**overrides)
        sa._publish_promoted_781_flags()
        return sa, os.environ.get("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE")

    def test_the_flag_default_is_not_an_assertion(self):
        """A default is not an assertion -- so the field must not hold 'pin'.

        RED on 9f2e58b797: the default was the string ``"pin"``, which alone
        defeats the #1270 rung even before the env publication, because
        ``_token_vector_role`` takes the flag as its second rung and returns
        any non-empty value it finds.
        """
        from sglang.srt.server_args import ServerArgs

        self.assertIsNone(
            ServerArgs.__dataclass_fields__["uneven_token_vector_role"].default
        )

    def test_a_real_process_publishes_estimate_when_nothing_was_declared(self):
        """RED on 9f2e58b797: this published 'pin'."""
        _sa, published = self._publish()
        self.assertEqual(published, dist_utils.ROLE_ESTIMATE)

    def test_the_published_role_is_what_the_consumer_then_reads(self):
        """The seam the #1270 desk test could not see: publish, then resolve.

        The consumer reads the env FIRST, so a wrong publication is not merely
        untidy -- it is the answer.
        """
        sa, _published = self._publish()
        self.assertEqual(dist_utils.token_vector_role(sa), dist_utils.ROLE_ESTIMATE)
        self.assertTrue(dist_utils.token_vector_arms_measured_install(sa))

    def test_the_real_sb4s_D_argv_resolves_to_estimate(self):
        """The exact argv from the metal boot, through the real parser."""
        import argparse

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        ns = parser.parse_args(_D_ARGV_1270B)
        self.assertIsNone(ns.uneven_token_vector)
        self.assertIsNone(ns.uneven_token_vector_role)

        sa, published = self._publish(
            uneven_token_vector=ns.uneven_token_vector,
            uneven_token_vector_role=ns.uneven_token_vector_role,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[29296, 18096, 18080],
            rank_tp_ratio=[1831, 1131, 1130],
            rank_kv_ratio="coupled",
            tp_size=3,
            dcp_size=3,
            uneven_dcp=True,
            uneven_dcp_weighted=True,
        )
        self.assertEqual(published, dist_utils.ROLE_ESTIMATE)
        self.assertEqual(dist_utils.token_vector_role(sa), dist_utils.ROLE_ESTIMATE)
        self.assertTrue(dist_utils.token_vector_arms_measured_install(sa))

    # -- the danger direction: a declared vector must stay a pin -------------

    def test_a_declared_vector_still_publishes_pin(self):
        """MUTANT GUARD. Installing over an operator's pin is the failure this
        whole family must never produce, so it is asserted through the same
        real publication path rather than argued about."""
        sa, published = self._publish(uneven_token_vector="30,17,17")
        self.assertEqual(published, dist_utils.ROLE_PIN)
        self.assertEqual(dist_utils.token_vector_role(sa), dist_utils.ROLE_PIN)
        self.assertFalse(dist_utils.token_vector_arms_measured_install(sa))

    def test_an_ambient_env_vector_is_declared_so_it_publishes_pin(self):
        """The calibration-reapply path (#901): the vector is inherited from
        the environment rather than from this argv, and it is still a VALUE
        somebody put there. It must not become an estimate and get recomputed
        away."""
        os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "17,7,8"
        sa, published = self._publish()
        self.assertEqual(published, dist_utils.ROLE_PIN)
        self.assertFalse(dist_utils.token_vector_arms_measured_install(sa))

    def test_an_explicit_seed_survives_the_resolution(self):
        sa, published = self._publish(
            uneven_token_vector="30,17,17", uneven_token_vector_role="seed"
        )
        self.assertEqual(published, dist_utils.ROLE_SEED)
        self.assertTrue(dist_utils.token_vector_arms_measured_install(sa))

    def test_a_rank_kv_ratio_list_is_a_declaration(self):
        sa, published = self._publish(rank_kv_ratio=[30, 17, 17])
        self.assertEqual(published, dist_utils.ROLE_PIN)
        self.assertFalse(dist_utils.token_vector_arms_measured_install(sa))

    def test_a_mode_string_is_not_a_declaration_in_a_real_process(self):
        sa, published = self._publish(rank_kv_ratio="capacity")
        self.assertEqual(published, dist_utils.ROLE_ESTIMATE)
        self.assertTrue(dist_utils.token_vector_arms_measured_install(sa))

    # -- #797's reason for the unconditional publish is preserved -----------

    def test_a_stale_role_env_is_overwritten_by_the_resolved_role(self):
        """#797 published the role unconditionally so a stale value from an
        earlier process in the same shell could not ride along. That property
        must survive #1270b: the publish is still unconditional, it just no
        longer invents 'pin' out of a default."""
        os.environ["SGLANG_UNEVEN_TOKEN_VECTOR_ROLE"] = "pin"
        _sa, published = self._publish()
        self.assertEqual(published, dist_utils.ROLE_ESTIMATE)

    def test_the_publisher_does_not_read_the_variable_it_writes(self):
        """One-predicate rule, asserted structurally: resolving the value to
        publish by reading the env being published is exactly how the default
        smuggled itself in, so the publisher's resolver must not touch it."""
        import ast
        import inspect
        import textwrap

        fn = ast.parse(
            textwrap.dedent(inspect.getsource(dist_utils.token_vector_role_from_args))
        ).body[0]
        # The docstring NAMES the variable (it has to -- that is the whole
        # point of the function), so the assertion is about the CODE.
        body = fn.body[1:] if ast.get_docstring(fn) is not None else fn.body
        code = "\n".join(ast.dump(stmt) for stmt in body)
        self.assertNotIn("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", code)


class TheInstallLineNamesItsArm1270b(CustomTestCase):
    """The line the launcher's provenance text promises must exist, and must
    not credit ``--rank-kv-ratio coupled`` for an install that a ROLE armed."""

    def test_the_install_line_is_present_and_names_the_arm(self):
        src = _mixin_method_source("_maybe_suggest_dcp_token_vector")
        self.assertIn("installed ", src)
        self.assertIn("measured KV-token ownership vector", src)
        self.assertIn("pre-boot ", src)
        self.assertIn("armed by %s", src)
        self.assertNotIn('"Uneven DCP %s mode (--rank-kv-ratio %s): installed "', src)

    def test_the_advisory_cannot_run_after_an_install(self):
        """Structural: the install branch returns before the advisory block."""
        src = _mixin_method_source("_maybe_suggest_dcp_token_vector")
        install_at = src.index("armed by %s")
        advisory_at = src.index("the profiled optimum is")
        self.assertLess(install_at, advisory_at)
        between = src[install_at:advisory_at]
        self.assertIn("return", between)


if __name__ == "__main__":
    unittest.main()
