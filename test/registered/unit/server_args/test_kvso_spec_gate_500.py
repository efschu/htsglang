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


if __name__ == "__main__":
    unittest.main()
