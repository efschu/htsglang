"""#837: the round-4 seam knobs are FLAGS; their SGLANG_* keys are a bridge.

These six knobs arrived with #830 and #834 as environment variables and never
reached a flag. That was not merely untidy. The #539 boot gate refuses any
governed SGLANG_* key that the ship capture does not carry, and it carries none
of these -- so the seam shrink was UNREACHABLE from route_a_631_prod_boot.sh,
and the W13b window that is supposed to attribute it could not have been run
without editing the boot script. argv is not policed by that gate, which is the
whole point of the promotion.

What has to hold, and why each arm is here rather than implied by the others:

  1. THE DEFAULT PATH IS BYTE-IDENTICAL. With all six flags unset, publication
     writes none of the six keys and overwrites none that already exist. Every
     deployment that has not moved must be unaffected, and this is the arm that
     says so.
  2. A set flag reaches the key its consumer reads, with the exact string.
  3. ``seam_shrink=False`` publishes "0" and is not skipped as falsy -- the
     trap the #781 suite calls out by name.

     READ THE SPELLING CAREFULLY. There is no ``--seam-shrink false``.
     ``Optional[bool]`` compiles to ``action="store_true", default=None``
     (arg_utils.py:276-282), so argv can only say ON or say nothing, and
     ``False`` is a state this process reaches from an explicit
     ``seam_shrink=False`` (a config file, a planner setting, a test), never
     from a command line. That is the shipped shape of all fourteen
     ``Optional[bool]`` promoted flags, not something this ticket introduced.
     The consequence an operator has to know: to run a W13b gate-OFF segment
     against a host whose environment already sets SGLANG_SEAM_SHRINK=1, the
     command line is ``--seam-shrink-prearm-quiesce 0 --seam-shrink-defer-grow
     0`` -- the per-half overrides DO take an explicit 0, and they reach both
     halves. Unsetting the env is the other way.
  4. THE TRI-STATE SURVIVES. -1 (follow master) and 1 (force on) are different
     answers, and a boolean-ish coercion collapses them.
  5. The validation refuses a bad value AT ARGV, by CLI name -- because the
     consumer's rule is "override >= 1 means force ON", so a typo'd 2 arms the
     half a window believed it had pinned to an off master.
  6. The flag WINS over a conflicting env var. This is the arm that would have
     caught the original #781 defect ("last one wins" between a heredoc and
     EXTRA_ENV) in this family.
  7. The env still resolves when the flag is unset -- the deprecation bridge.
  8. END TO END: master on, one half forced off, and the runtime's own readers
     report exactly that. This is the per-half attribution capability the
     overrides exist for, and it is the single most valuable assertion here.
  9. The deprecation notice fires when a human set the key, and stays silent
     when nobody did.

Hermetic: attribute-to-environ plumbing plus the runtime's pure env readers.
No model, no device, no server, no collective.
"""

import argparse
import dataclasses
import inspect
import os
import unittest
import warnings
from unittest import mock

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


# The six promoted knobs: (ServerArgs field, published key).
SIX = [
    ("seam_shrink", "SGLANG_SEAM_SHRINK"),
    ("seam_shrink_prearm_quiesce", "SGLANG_SEAM_SHRINK_PREARM_QUIESCE"),
    ("seam_shrink_defer_grow", "SGLANG_SEAM_SHRINK_DEFER_GROW"),
    ("seam_shrink_grow_debt_rounds", "SGLANG_SEAM_SHRINK_GROW_DEBT_ROUNDS"),
    ("flip_seam_drain_budget_ms", "SGLANG_FLIP_SEAM_DRAIN_BUDGET_MS"),
    ("hicache_read_buffers", "SGLANG_HICACHE_READ_BUFFERS"),
]

# The two per-half attribution overrides, which are tri-states and not bools.
PER_HALF = ["seam_shrink_prearm_quiesce", "seam_shrink_defer_grow"]

# The three that are counts or durations and may not be negative.
COUNTS = [
    "seam_shrink_grow_debt_rounds",
    "flip_seam_drain_budget_ms",
    "hicache_read_buffers",
]


def _bare(**kw):
    """A ServerArgs carrying only these six fields, without constructing one.

    ``__new__`` skips ``__post_init__``; the dataclass defaults remain readable
    as class attributes, so every field not named here resolves to its default.
    The six are set explicitly anyway, so a test states its own inputs rather
    than inheriting them.
    """
    sa = ServerArgs.__new__(ServerArgs)
    for field, _env in SIX:
        setattr(sa, field, None)
    for k, v in kw.items():
        setattr(sa, k, v)
    return sa


def _publish(**kw):
    """Run only the publication step against such an instance."""
    sa = _bare(**kw)
    sa._publish_promoted_781_flags()
    return sa


def _validate(**kw):
    """Run only the #837 validation step."""
    _bare(**kw)._handle_seam_shrink_flags_837()


def _clean_env():
    """os.environ isolation with all six keys removed.

    These tests write REAL environment keys that the runtime readers consult,
    so a leak between them is not cosmetic: it would let one test's value
    decide another test's verdict.
    """
    ctx = mock.patch.dict(os.environ, {}, clear=False)
    ctx.start()
    for _field, env in SIX:
        os.environ.pop(env, None)
    return ctx


class TestTheDefaultPathIsUntouched837(unittest.TestCase):
    """The compatibility guarantee, stated as its own arm."""

    def test_no_flag_set_publishes_none_of_the_six_keys(self):
        """Byte-identity. An unset flag must neither invent a key nor
        overwrite one, or every deployment that has not moved moves."""
        ctx = _clean_env()
        try:
            os.environ["SGLANG_SEAM_SHRINK"] = "preexisting"
            _publish()  # every one of the six is None
            self.assertEqual(
                os.environ.get("SGLANG_SEAM_SHRINK"),
                "preexisting",
                "an unset --seam-shrink overwrote an existing env value",
            )
            for field, env in SIX[1:]:
                self.assertIsNone(
                    os.environ.get(env),
                    f"unset --{field.replace('_', '-')} invented {env}",
                )
        finally:
            ctx.stop()

    def test_the_six_are_real_server_args_fields(self):
        """A flag argparse merely tolerates is not a promoted flag."""
        names = {f.name for f in dataclasses.fields(ServerArgs)}
        for field, _env in SIX:
            self.assertIn(field, names, f"{field} is not a ServerArgs field")

    def test_validation_and_publication_are_wired_in_that_order(self):
        """A validator nothing calls validates nothing -- and it has to run
        BEFORE the publish, or a refused value has already been written to the
        environment of the process that is about to abort."""
        src = inspect.getsource(ServerArgs.__post_init__)
        self.assertIn(
            "_handle_seam_shrink_flags_837()",
            src,
            "the #837 validator is not called from __post_init__",
        )
        self.assertIn("_handle_environment_variables()", src)
        self.assertLess(
            src.index("_handle_seam_shrink_flags_837()"),
            src.index("_handle_environment_variables()"),
            "the #837 knobs are published before they are validated",
        )


class TestSetFlagsReachTheEnvironment837(unittest.TestCase):
    def test_each_set_flag_publishes_its_key(self):
        cases = [
            (dict(seam_shrink=True), "SGLANG_SEAM_SHRINK", "1"),
            (dict(seam_shrink_prearm_quiesce=1),
             "SGLANG_SEAM_SHRINK_PREARM_QUIESCE", "1"),
            (dict(seam_shrink_defer_grow=0),
             "SGLANG_SEAM_SHRINK_DEFER_GROW", "0"),
            (dict(seam_shrink_grow_debt_rounds=7),
             "SGLANG_SEAM_SHRINK_GROW_DEBT_ROUNDS", "7"),
            (dict(flip_seam_drain_budget_ms=250),
             "SGLANG_FLIP_SEAM_DRAIN_BUDGET_MS", "250"),
            (dict(hicache_read_buffers=4), "SGLANG_HICACHE_READ_BUFFERS", "4"),
        ]
        for kw, env, expected in cases:
            with self.subTest(env=env):
                ctx = _clean_env()
                try:
                    _publish(**kw)
                    self.assertEqual(
                        os.environ.get(env),
                        expected,
                        f"{list(kw)[0]} did not publish {env}={expected}",
                    )
                finally:
                    ctx.stop()

    def test_seam_shrink_false_publishes_zero_not_nothing(self):
        """``--seam-shrink false`` must be able to turn the shrink OFF.

        ``if self.seam_shrink:`` instead of ``is not None`` skips False and
        leaves whatever the environment already said. The operator asks for
        off, the flip runs the shrink anyway, and the next window's numbers are
        attributed to a move that was never disabled.
        """
        ctx = _clean_env()
        try:
            os.environ["SGLANG_SEAM_SHRINK"] = "1"
            _publish(seam_shrink=False)
            self.assertEqual(
                os.environ.get("SGLANG_SEAM_SHRINK"),
                "0",
                "--seam-shrink false did not publish '0'",
            )
        finally:
            ctx.stop()

        import sglang.srt.managers.phase_flip_runtime as rt

        ctx = _clean_env()
        try:
            os.environ["SGLANG_SEAM_SHRINK"] = "1"
            _publish(seam_shrink=False)
            self.assertFalse(
                rt._seam_shrink_master(),
                "the runtime still read the shrink as armed after "
                "--seam-shrink false",
            )
        finally:
            ctx.stop()

    def test_the_tri_state_survives_publication(self):
        """-1 and 1 are DIFFERENT answers and must not collapse.

        -1 means "follow --seam-shrink" and 1 means "force this half on". A
        boolean coercion -- the ``_b()`` helper the neighbouring clauses use --
        maps both onto "1". A W13b attribution run that asked for
        ``--seam-shrink-prearm-quiesce -1`` under a master that is off would
        then be running that half ON, measure both halves, and report the
        result as an attribution to one of them. This family has already paid
        twice for reading adjacency as attribution.
        """
        for field in PER_HALF:
            env = dict(SIX)[field]
            for value in (-1, 0, 1):
                with self.subTest(field=field, value=value):
                    ctx = _clean_env()
                    try:
                        _publish(**{field: value})
                        self.assertEqual(
                            os.environ.get(env),
                            str(value),
                            f"--{field.replace('_', '-')} {value} published "
                            f"{os.environ.get(env)!r}; the tri-state was "
                            f"coerced",
                        )
                    finally:
                        ctx.stop()


class TestValidationRefusesByName837(unittest.TestCase):
    """The refusal belongs at argv.

    ``_seam_shrink_half`` resolves the override as ``0 -> False``,
    ``>= 1 -> True``, anything else -> the master. So a typo'd ``2`` is NOT
    swallowed into follow-the-master: it is FORCE ON. ``EnvInt.parse`` accepts
    "2" cleanly, so the surrounding ``try/except Exception`` -- whose contract
    is that a gate lookup must never break a flip -- never fires for a numeric
    typo either. A window that believed it had pinned this half to a master
    that was off would have run it ON and attributed the number to the other
    move. That is why the refusal belongs at argv.
    """

    def test_a_bad_tri_state_is_refused_naming_the_cli_flag(self):
        for field in PER_HALF:
            for value in (2, -2):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError) as cm:
                        _validate(**{field: value})
                    self.assertIn(
                        f"--{field.replace('_', '-')}",
                        str(cm.exception),
                        "the refusal must name the flag in its CLI spelling; "
                        "an operator greps the message, not the field name",
                    )

    def test_a_negative_count_is_refused_naming_the_cli_flag(self):
        for field in COUNTS:
            with self.subTest(field=field):
                with self.assertRaises(ValueError) as cm:
                    _validate(**{field: -1})
                self.assertIn(f"--{field.replace('_', '-')}", str(cm.exception))

    def test_legal_values_are_not_refused(self):
        """A validator that refuses everything is not a validator.

        Zero is legal for all three counts and is meaningful for two of them:
        ``--flip-seam-drain-budget-ms 0`` stands the #830 guard down (the
        projection is still logged) and ``--hicache-read-buffers 0`` is the
        shipped default that allocates per read.
        """
        for field in PER_HALF:
            for value in (-1, 0, 1):
                with self.subTest(field=field, value=value):
                    _validate(**{field: value})
        for field in COUNTS:
            for value in (0, 1, 4096):
                with self.subTest(field=field, value=value):
                    _validate(**{field: value})
        _validate()  # nothing set at all

    def test_a_per_half_override_without_the_master_is_allowed(self):
        """Deliberately legal, and it is how a window turns ONE half on.

        Refusing it would forbid exactly the attribution run the overrides
        exist for: master off, one half forced on, the other untouched.
        """
        _validate(seam_shrink_prearm_quiesce=1)
        _validate(seam_shrink_defer_grow=1)
        _validate(seam_shrink=False, seam_shrink_prearm_quiesce=1)
        _validate(seam_shrink=True, seam_shrink_defer_grow=0)


class TestTheFlagBeatsTheEnv837(unittest.TestCase):
    def test_flag_wins_over_env(self):
        """The defect the whole promotion exists for.

        The boot env used to be a captured shell environment plus a heredoc
        plus EXTRA_ENV, and a key written twice resolved as "last one wins" --
        SGLANG_PHASE_POLICY_TP_DECODE_FLOOR_S was 10 in one half and 8 in the
        other and the 10 had been dead the whole time. With the flag
        authoritative, the flag's value is the one that survives.
        """
        cases = [
            (dict(seam_shrink=False), "SGLANG_SEAM_SHRINK", "1", "0"),
            (dict(seam_shrink_prearm_quiesce=-1),
             "SGLANG_SEAM_SHRINK_PREARM_QUIESCE", "1", "-1"),
            (dict(seam_shrink_defer_grow=0),
             "SGLANG_SEAM_SHRINK_DEFER_GROW", "1", "0"),
            (dict(seam_shrink_grow_debt_rounds=7),
             "SGLANG_SEAM_SHRINK_GROW_DEBT_ROUNDS", "32", "7"),
            (dict(flip_seam_drain_budget_ms=250),
             "SGLANG_FLIP_SEAM_DRAIN_BUDGET_MS", "1094", "250"),
            (dict(hicache_read_buffers=4),
             "SGLANG_HICACHE_READ_BUFFERS", "0", "4"),
        ]
        for kw, env, stale, expected in cases:
            with self.subTest(env=env):
                ctx = _clean_env()
                try:
                    os.environ[env] = stale
                    _publish(**kw)
                    self.assertEqual(
                        os.environ.get(env),
                        expected,
                        f"env {env}={stale} survived the flag -- last-wins is "
                        f"back",
                    )
                finally:
                    ctx.stop()


class TestTheEnvBridgeStillWorks837(unittest.TestCase):
    """Nobody's running deployment changes today."""

    def test_env_still_resolves_when_the_flag_is_unset(self):
        import sglang.srt.managers.phase_flip_runtime as rt
        from sglang.srt.environ import envs

        ctx = _clean_env()
        try:
            os.environ["SGLANG_SEAM_SHRINK"] = "1"
            os.environ["SGLANG_SEAM_SHRINK_PREARM_QUIESCE"] = "0"
            os.environ["SGLANG_SEAM_SHRINK_DEFER_GROW"] = "1"
            os.environ["SGLANG_SEAM_SHRINK_GROW_DEBT_ROUNDS"] = "7"
            os.environ["SGLANG_FLIP_SEAM_DRAIN_BUDGET_MS"] = "250"
            os.environ["SGLANG_HICACHE_READ_BUFFERS"] = "4"
            _publish()  # every flag unset; publication must not disturb these
            self.assertTrue(
                rt._seam_shrink_master(), "SGLANG_SEAM_SHRINK stopped resolving"
            )
            self.assertFalse(
                rt.seam_shrink_prearm_quiesce_enabled(),
                "an env-set force-off half stopped being honoured",
            )
            self.assertTrue(
                rt.seam_shrink_defer_grow_enabled(),
                "an env-set force-on half stopped being honoured",
            )
            self.assertEqual(7, rt.seam_shrink_grow_debt_rounds())
            self.assertEqual(250, rt.flip_seam_drain_budget_ms())
            self.assertEqual(4, envs.SGLANG_HICACHE_READ_BUFFERS.get())
        finally:
            ctx.stop()

    def test_the_in_code_defaults_are_still_the_documented_ones(self):
        """The promotion must not have re-seeded a default while moving it.

        1094 ms is the largest cutover observed across the 1014-flip
        HiCache-off corpus (ANALYSE_830 2.1) -- a 0 here would ship the #830
        guard permanently stood down. 32 rounds is the #834 ratchet patience.
        0 read buffers is the shipped per-read allocation.
        """
        import sglang.srt.managers.phase_flip_runtime as rt
        from sglang.srt.environ import envs

        ctx = _clean_env()
        try:
            _publish()
            self.assertFalse(rt._seam_shrink_master())
            self.assertFalse(rt.seam_shrink_prearm_quiesce_enabled())
            self.assertFalse(rt.seam_shrink_defer_grow_enabled())
            self.assertEqual(1094, rt.flip_seam_drain_budget_ms())
            self.assertEqual(32, rt.seam_shrink_grow_debt_rounds())
            self.assertEqual(0, envs.SGLANG_HICACHE_READ_BUFFERS.get())
        finally:
            ctx.stop()


class TestPerHalfAttributionEndToEnd837(unittest.TestCase):
    """From argv to the runtime's own readers, which is the capability."""

    def test_master_on_with_one_half_forced_off(self):
        """The W13b run: arm the shrink, hold ONE half down, and have the
        runtime agree. Without this the overrides are a published string that
        nothing was ever observed to obey.
        """
        import sglang.srt.managers.phase_flip_runtime as rt

        ctx = _clean_env()
        try:
            _publish(seam_shrink=True, seam_shrink_defer_grow=0)
            self.assertTrue(
                rt._seam_shrink_master(), "--seam-shrink did not arm the master"
            )
            self.assertTrue(
                rt.seam_shrink_prearm_quiesce_enabled(),
                "the untouched half stopped following the master",
            )
            self.assertFalse(
                rt.seam_shrink_defer_grow_enabled(),
                "--seam-shrink-defer-grow 0 did not force that half off; the "
                "window would measure both halves and attribute to one",
            )
        finally:
            ctx.stop()

    def test_master_off_with_one_half_forced_on(self):
        """The other direction, which is the cheaper half of an attribution
        pair: nothing armed except the half under test."""
        import sglang.srt.managers.phase_flip_runtime as rt

        ctx = _clean_env()
        try:
            _publish(seam_shrink=False, seam_shrink_prearm_quiesce=1)
            self.assertFalse(rt._seam_shrink_master())
            self.assertTrue(
                rt.seam_shrink_prearm_quiesce_enabled(),
                "--seam-shrink-prearm-quiesce 1 did not force that half on",
            )
            self.assertFalse(
                rt.seam_shrink_defer_grow_enabled(),
                "one half on must NOT turn the other on",
            )
        finally:
            ctx.stop()


class TestDeprecationNotices837(unittest.TestCase):
    """The env is a bridge and must say so, once, to whoever set it.

    environ.py is imported long before ServerArgs runs ``__post_init__``, so a
    key found by the notice was written by a human or a boot script -- the
    value a flag publishes arrives later and is never warned about. That
    asymmetry is intended, and pinning the SILENT direction is what keeps the
    notice from firing at every promoted boot and training operators to ignore
    it.
    """

    def _warn_for(self, env, preset):
        from sglang.srt.environ import _warn_deprecated_env_to_cli_flag

        ctx = _clean_env()
        try:
            if preset is not None:
                os.environ[env] = preset
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _warn_deprecated_env_to_cli_flag(
                    env, f"Please use '--{env[7:].lower().replace('_', '-')}' "
                    f"instead."
                )
            return [str(w.message) for w in caught]
        finally:
            ctx.stop()

    def test_setting_the_env_warns_and_names_the_flag(self):
        for field, env in SIX:
            with self.subTest(env=env):
                messages = self._warn_for(env, "1")
                self.assertEqual(
                    1, len(messages), f"{env} produced {len(messages)} warnings"
                )
                self.assertIn(env, messages[0])
                self.assertIn(
                    f"--{field.replace('_', '-')}",
                    messages[0],
                    "the notice must name the flag that replaces the env",
                )

    def test_an_unset_env_is_silent(self):
        for _field, env in SIX:
            with self.subTest(env=env):
                self.assertEqual(
                    [],
                    self._warn_for(env, None),
                    f"{env} warned while nobody had set it",
                )

    def test_all_six_notices_are_registered_in_environ(self):
        """The helper is only reached if environ.py calls it for this key."""
        import sglang.srt.environ as environ_mod

        src = inspect.getsource(environ_mod)
        for field, env in SIX:
            with self.subTest(env=env):
                self.assertIn(
                    f'_warn_deprecated_env_to_cli_flag(\n    "{env}",',
                    src,
                    f"{env} has no deprecation notice in environ.py",
                )
                marker = src.index(f'_warn_deprecated_env_to_cli_flag(\n    "{env}",')
                self.assertIn(
                    f"--{field.replace('_', '-')}",
                    src[marker : marker + 400],
                    f"{env}'s notice does not name its CLI flag",
                )


class TestTheSixAreActuallyOnTheCommandLine837(unittest.TestCase):
    """The ticket's own subject: these are FLAGS, not just dataclass fields.

    Every other test in this file reaches the fields by attribute, and
    ``_bare`` even bypasses ``__init__``. That is deliberate -- it keeps the
    publication arms hermetic -- but it leaves one hole big enough to drive
    the whole ticket through: the suite would stay entirely green if all six
    carried ``Arg(no_cli=True)``, or if a CLI name collided and never
    registered. Then #837 would have promoted nothing and every assertion
    about publication would still pass.

    So this class is the only one that goes through the real argparse parser.
    """

    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        return parser.parse_args(["--model-path", "/m"] + argv)

    def test_every_promoted_knob_registers_a_cli_flag(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        registered = {
            opt for action in parser._actions for opt in action.option_strings
        }
        for field, _env in SIX:
            with self.subTest(field=field):
                self.assertIn(
                    f"--{field.replace('_', '-')}",
                    registered,
                    f"{field} is a dataclass field with no command line; the "
                    f"promotion did not happen for it",
                )

    def test_the_master_is_a_bare_store_true(self):
        self.assertIs(self._parse(["--seam-shrink"]).seam_shrink, True)
        # Absent means "not specified", which is what keeps the default path
        # byte-identical -- not False.
        self.assertIsNone(self._parse([]).seam_shrink)

    def test_a_negative_tri_state_survives_argparse(self):
        """-1 must arrive as a value, not be eaten as an option string.

        argparse treats a leading '-' as an option prefix, so this is a real
        hazard for exactly the value that means "follow the master" and for no
        other value in the family.
        """
        for field in PER_HALF:
            with self.subTest(field=field):
                ns = self._parse([f"--{field.replace('_', '-')}", "-1"])
                self.assertEqual(getattr(ns, field), -1)

    def test_the_counts_and_durations_parse_as_ints(self):
        ns = self._parse(
            [
                "--seam-shrink-grow-debt-rounds",
                "7",
                "--flip-seam-drain-budget-ms",
                "1500",
                "--hicache-read-buffers",
                "8",
            ]
        )
        self.assertEqual(ns.seam_shrink_grow_debt_rounds, 7)
        self.assertEqual(ns.flip_seam_drain_budget_ms, 1500)
        self.assertEqual(ns.hicache_read_buffers, 8)

    def test_the_w13b_gate_off_spelling_parses_and_reaches_both_halves(self):
        """The command line an operator needs and nothing else documents.

        There is no ``--no-seam-shrink``, so a gate-OFF segment against a host
        whose environment sets SGLANG_SEAM_SHRINK=1 is spelled with the two
        per-half zeros. If this ever stops parsing, the W13b control arm loses
        its only argv expression.
        """
        ns = self._parse(
            ["--seam-shrink-prearm-quiesce", "0", "--seam-shrink-defer-grow", "0"]
        )
        self.assertEqual(ns.seam_shrink_prearm_quiesce, 0)
        self.assertEqual(ns.seam_shrink_defer_grow, 0)
        with mock.patch.dict(os.environ, {"SGLANG_SEAM_SHRINK": "1"}, clear=False):
            _publish(
                seam_shrink_prearm_quiesce=ns.seam_shrink_prearm_quiesce,
                seam_shrink_defer_grow=ns.seam_shrink_defer_grow,
            )
            from sglang.srt.managers import phase_flip_runtime as rt

            self.assertFalse(rt.seam_shrink_prearm_quiesce_enabled())
            self.assertFalse(rt.seam_shrink_defer_grow_enabled())


if __name__ == "__main__":
    unittest.main()
