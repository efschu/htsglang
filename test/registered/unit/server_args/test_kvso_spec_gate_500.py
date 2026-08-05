"""#500-B10: the kvso x speculation gate is DELIBERATE, and must read as it.

THE QUESTION THE AUDIT ASKED
----------------------------
``FEATURE_CATALOG.md`` §3 used to describe kv-session-offload as "decoupled
from speculation" while the runtime refused the pair outright::

    if self.speculative_algorithm is not None and (
        os.environ.get("KVSO_ALLOW_SPEC", "0") != "1"
    ):
        raise ValueError("--enable-kv-session-offload does not yet support
                          speculative decoding ...")

(``server_args.py:6580``). Two readings were possible: the refusal is stale
(the decoupling landed and nobody removed the gate), or the decoupling is real
but incompletely validated (the gate is the opt-in, and the catalog line was
the wrong one).

THE ANSWER, FROM THE TREE
-------------------------
The second. The mechanism EXISTS and is documented as an opt-in route, not as
an unimplemented one -- ``FEATURES_VS_UPSTREAM.md`` (Speculation row of the
kv-session-offload table): "a spilled session decodes under MTP/NEXTN: the
draft-KV share spills and restores with the session, with on-device resume and
draft backfill; the drafter can run inside the spill tick
(``--kv-session-offload-spec-in-tick``); every spill+spec boot rides the
``KVSO_ALLOW_SPEC=1`` opt-in gate". And the same table's "Bounds that hold" row
names what is still unobserved: "a spill landing in the same round as a
drafter-in-tick step has not been observed in validation".

So the refusal stays, and what was wrong is that it read as "not implemented"
and hid its own way out: ``KVSO_ALLOW_SPEC`` appeared nowhere in
``--enable-kv-session-offload``'s help, whose S1 scope line still said "no
speculative decoding" flatly. An operator reading the CLI could not discover
the supported route at all.

WHAT IS PINNED HERE
-------------------
1. the gate still fires by default (nothing was loosened);
2. the opt-in still works;
3. the refusal NAMES the mechanism's state -- that the pair is built and
   opt-in, not missing -- rather than reading as a not-implemented stub;
4. the flag help surfaces the env, so the route is discoverable from ``--help``.

CAN-FAIL PROOF: drop ``KVSO_ALLOW_SPEC`` from the help string and
``test_the_flag_help_surfaces_the_opt_in`` goes red; delete the reason
sentences and ``test_the_refusal_states_why_the_gate_is_there`` goes red, both
while the behaviour tests stay green -- which is the point, since this posten
changes no behaviour.
"""

import argparse
import os
import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def make_args(**kwargs):
    return ServerArgs(model_path="dummy", **kwargs)


def _flag_help(option: str) -> str:
    """The help argparse would print for ``option`` -- read off the built
    parser, not off the dataclass annotation, so this asserts what an operator
    actually sees from ``--help``."""
    parser = argparse.ArgumentParser(prog="sglang.launch_server")
    ServerArgs.add_cli_args(parser)
    for action in parser._actions:
        if option in action.option_strings:
            return action.help or ""
    raise AssertionError(f"no CLI action for {option}")


def _no_env():
    return patch.dict(os.environ, {"KVSO_ALLOW_SPEC": "0"})


class TestKvsoSpecGateStillHolds(unittest.TestCase):
    def test_the_gate_fires_by_default(self):
        with _no_env():
            for algo in ("EAGLE", "NEXTN", "EAGLE3", "DFLASH"):
                with self.subTest(algo=algo):
                    args = make_args(
                        enable_kv_session_offload=True, speculative_algorithm=algo
                    )
                    with self.assertRaises(ValueError) as cm:
                        args._handle_kv_session_offload()
                    self.assertIn("KVSO_ALLOW_SPEC=1", str(cm.exception))

    def test_the_opt_in_admits_the_pair(self):
        with patch.dict(os.environ, {"KVSO_ALLOW_SPEC": "1"}):
            make_args(
                enable_kv_session_offload=True, speculative_algorithm="EAGLE"
            )._handle_kv_session_offload()

    def test_without_speculation_the_gate_is_not_reached(self):
        with _no_env():
            make_args(enable_kv_session_offload=True)._handle_kv_session_offload()


class TestKvsoSpecGateIsLegible(unittest.TestCase):
    """The posten itself: the gate must say WHY it is there."""

    def _message(self):
        with _no_env():
            args = make_args(
                enable_kv_session_offload=True, speculative_algorithm="NEXTN"
            )
            with self.assertRaises(ValueError) as cm:
                args._handle_kv_session_offload()
            return str(cm.exception)

    def test_the_refusal_states_why_the_gate_is_there(self):
        msg = self._message()
        # it must not read as "not implemented"
        self.assertNotIn("does not yet support", msg)
        # it must name the mechanism that DOES exist
        self.assertIn("draft-KV share", msg)
        # and the concrete unobserved case that keeps the gate on
        self.assertIn("drafter-in-tick", msg)

    def test_the_refusal_still_names_the_way_out_and_the_algorithm(self):
        msg = self._message()
        self.assertIn("KVSO_ALLOW_SPEC=1", msg)
        self.assertIn("NEXTN", msg)

    def test_the_flag_help_surfaces_the_opt_in(self):
        """An operator must be able to find the supported route from --help."""
        self.assertIn("KVSO_ALLOW_SPEC", _flag_help("--enable-kv-session-offload"))

    def test_the_help_no_longer_claims_speculation_is_out_of_scope(self):
        self.assertNotIn(
            "no speculative decoding", _flag_help("--enable-kv-session-offload")
        )


class TestResumeUnderSpecIsAFirstClassSurface(unittest.TestCase):
    """#552: the on-device MTP resume path must be reachable from ``--help``.

    Before this posten the path existed and was gated on a bare ``KVSO_RESUME``
    env read inside ``kv_session_offload.resume_under_spec_enabled``. That
    string appeared NOWHERE in ``server_args.py`` -- not in a flag, not in a
    help text, not in a validation message -- so an operator reading ``--help``
    could not discover, arm, or even learn of the mechanism. A path nobody can
    find is not an opt-in; it is a path that only its author can run, which is
    precisely how a built mechanism decays into an unvalidated one.

    CAN-FAIL PROOF: delete the flag and every test here goes red; keep the flag
    but drop the env OR in ``_handle_kv_session_offload`` and
    ``test_the_env_twin_arms_the_flag`` / ``test_the_legacy_alias_still_arms_it``
    go red; drop either fail-fast and its test goes red.
    """

    def _resume_args(self, **over):
        kw = dict(
            enable_kv_session_offload=True,
            speculative_algorithm="NEXTN",
            kv_session_offload_resume_under_spec=True,
        )
        kw.update(over)
        return make_args(**kw)

    def _no_resume_env(self):
        return patch.dict(
            os.environ, {"KVSO_ALLOW_SPEC": "1", "SGLANG_KVSO_RESUME": "0"}
        )

    def test_the_flag_exists_and_is_discoverable(self):
        help_text = _flag_help("--kv-session-offload-resume-under-spec")
        self.assertTrue(help_text)
        # It must name its env twin, so the two surfaces are known to be one.
        self.assertIn("SGLANG_KVSO_RESUME", help_text)
        # ...and it must say the default is a decision, not an omission.
        self.assertIn("NAMED decision", help_text)

    def test_the_default_is_off(self):
        self.assertIs(make_args().kv_session_offload_resume_under_spec, False)
        with self._no_resume_env():
            args = make_args(
                enable_kv_session_offload=True, speculative_algorithm="NEXTN"
            )
            args._handle_kv_session_offload()
            self.assertIs(args.kv_session_offload_resume_under_spec, False)

    def test_the_env_twin_arms_the_flag(self):
        with patch.dict(
            os.environ, {"KVSO_ALLOW_SPEC": "1", "SGLANG_KVSO_RESUME": "1"}
        ):
            args = make_args(
                enable_kv_session_offload=True, speculative_algorithm="NEXTN"
            )
            args._handle_kv_session_offload()
            self.assertIs(args.kv_session_offload_resume_under_spec, True)

    def test_the_legacy_alias_still_arms_it(self):
        """Existing boot-matrix arms and tickets export the bare name."""
        env = dict(os.environ)
        env.pop("SGLANG_KVSO_RESUME", None)
        env["KVSO_ALLOW_SPEC"] = "1"
        env["KVSO_RESUME"] = "1"
        with patch.dict(os.environ, env, clear=True):
            args = make_args(
                enable_kv_session_offload=True, speculative_algorithm="NEXTN"
            )
            args._handle_kv_session_offload()
            self.assertIs(args.kv_session_offload_resume_under_spec, True)

    def test_arming_it_without_the_feature_is_rejected(self):
        with self._no_resume_env():
            args = make_args(
                enable_kv_session_offload=False,
                kv_session_offload_resume_under_spec=True,
            )
            with self.assertRaises(ValueError) as cm:
                args._handle_kv_session_offload()
            self.assertIn("sub-mode of --enable-kv-session-offload", str(cm.exception))

    def test_arming_it_without_speculation_is_rejected(self):
        with self._no_resume_env():
            args = self._resume_args(speculative_algorithm=None)
            with self.assertRaises(ValueError) as cm:
                args._handle_kv_session_offload()
            self.assertIn(
                "requires an active --speculative-algorithm", str(cm.exception)
            )

    def test_resume_and_ps2_deep_prefill_are_rejected_together(self):
        """A placement fact, not a policy: a born-spilled prompt never wrote
        the draft KV that the rejoined session's drafter attends."""
        with self._no_resume_env():
            args = self._resume_args(kv_session_offload_prefill=True)
            with self.assertRaises(ValueError) as cm:
                args._handle_kv_session_offload()
            self.assertIn("--kv-session-offload-prefill", str(cm.exception))

    def test_the_runtime_predicate_agrees_with_the_flag(self):
        """The gate the runtime reads and the flag the operator sets must be
        the same switch -- otherwise --help documents a lie."""
        from sglang.srt.managers.kv_session_offload import resume_under_spec_enabled

        env = dict(os.environ)
        env.pop("SGLANG_KVSO_RESUME", None)
        env.pop("KVSO_RESUME", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertIs(resume_under_spec_enabled(), False)
        with patch.dict(os.environ, {"SGLANG_KVSO_RESUME": "1"}):
            self.assertIs(resume_under_spec_enabled(), True)


if __name__ == "__main__":
    unittest.main()
