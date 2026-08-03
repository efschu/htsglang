"""#500-B8: ``validate_breakable_boot`` must not have a total-bypass arm.

THE DEFECT
----------
The gate protects #462's breakable route from the one shape that cannot work:
the route splits the decode graph at the MoE fetch through ``eager_on_graph``,
which is a PASS-THROUGH under every backend other than ``breakable``. Off that
backend the fetch's host reads execute inside a real stream capture, where a
D2H sync is illegal rather than merely slow -- and #452 already measured what
happens when the fetch goes into the capture (6.60x, refuted).

Both of the gate's server-args preconditions -- decode backend ``breakable``,
prefill backend ``disabled`` -- sat behind

    backend = resolved_backend("decode")
    if backend is None:
        # Server args not wired (unit / test context): nothing to validate
        return

and ``resolved_backend`` answers ``None`` for FOUR different situations
(``offload_capture_gate.py:408-421``): the runtime context raising, no
``cuda_graph_config`` on the args, no config for the phase, and a phase whose
``backend`` attribute is ``None``. Only the first is the unit/test context the
comment describes. The other three are REAL boots whose backend could not be
resolved, and for them the gate returned silently -- so the exact failure the
gate exists to prevent was reachable through the gate's own ``None`` arm.

THE RULE
--------
``None`` must either resolve or refuse by name. The two situations are now
distinguished at the source: ``resolved_backend`` returns the ``NO_SERVER_ARGS``
sentinel when there are no server args to read at all, and ``None`` only when
args ARE present and the phase's backend could not be resolved from them. The
gate skips on the sentinel (and says so) and refuses on ``None``.

CAN-FAIL PROOF
--------------
Restore ``if backend is None: return`` in place of the split and
``test_present_args_with_an_unresolved_decode_backend_refuse`` goes red for all
three of its shapes, while ``test_absent_server_args_still_skip`` stays green --
which is the distinction the fix is about.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.moe.offload_capture_gate import (
    NO_SERVER_ARGS,
    BreakableModeRefused,
    resolved_backend,
    validate_breakable_boot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


_GATE = "sglang.srt.layers.moe.offload_capture_gate"


def _args(decode=None, prefill=None, has_config=True, disable_cuda_graph=False):
    """A ServerArgs stand-in carrying only what ``resolved_backend`` reads."""
    config = (
        SimpleNamespace(
            decode=SimpleNamespace(backend=decode) if decode is not None else None,
            prefill=SimpleNamespace(backend=prefill) if prefill is not None else None,
        )
        if has_config
        else None
    )
    return SimpleNamespace(
        cuda_graph_config=config, disable_cuda_graph=disable_cuda_graph
    )


class _NoArgs:
    """``get_server_args`` raising is the genuine unit/test context."""

    def __call__(self):
        raise RuntimeError("no runtime context")


class TestResolvedBackendSeparatesAbsentFromUnresolved(unittest.TestCase):
    def test_absent_args_answer_the_sentinel(self):
        with patch("sglang.srt.runtime_context.get_server_args", _NoArgs()):
            self.assertIs(resolved_backend("decode"), NO_SERVER_ARGS)

    def test_present_args_without_a_graph_config_answer_none(self):
        with patch(
            "sglang.srt.runtime_context.get_server_args",
            lambda: _args(has_config=False),
        ):
            self.assertIsNone(resolved_backend("decode"))

    def test_present_args_without_a_phase_config_answer_none(self):
        with patch(
            "sglang.srt.runtime_context.get_server_args",
            lambda: _args(prefill="disabled"),
        ):
            self.assertIsNone(resolved_backend("decode"))

    def test_a_resolved_backend_is_returned_verbatim(self):
        with patch(
            "sglang.srt.runtime_context.get_server_args",
            lambda: _args(decode="breakable", prefill="disabled"),
        ):
            self.assertEqual(resolved_backend("decode"), "breakable")

    def test_disable_cuda_graph_still_wins(self):
        with patch(
            "sglang.srt.runtime_context.get_server_args",
            lambda: _args(decode="breakable", disable_cuda_graph=True),
        ):
            self.assertEqual(resolved_backend("decode"), "disabled")

    def test_the_sentinel_is_not_none_and_not_a_backend_name(self):
        self.assertIsNotNone(NO_SERVER_ARGS)
        self.assertNotIn(NO_SERVER_ARGS, ("breakable", "disabled", "full"))


class TestValidateBreakableBootNoneArm(unittest.TestCase):
    def test_absent_server_args_still_skip(self):
        """The one situation the bypass was written for keeps working: a unit
        or test context has nothing to validate against."""
        with patch("sglang.srt.runtime_context.get_server_args", _NoArgs()):
            validate_breakable_boot(0.5, layer_id=3)

    def test_present_args_with_an_unresolved_decode_backend_refuse(self):
        """THE falsifier. Every one of these is a real boot -- server args
        exist -- whose decode backend could not be resolved. Passing silently
        admits the illegal host read inside a real capture."""
        shapes = {
            "no cuda_graph_config": _args(has_config=False),
            "no decode phase config": _args(prefill="disabled"),
            "decode backend is None": SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend=None),
                    prefill=SimpleNamespace(backend="disabled"),
                ),
                disable_cuda_graph=False,
            ),
        }
        for name, args in shapes.items():
            with self.subTest(shape=name):
                with patch(
                    "sglang.srt.runtime_context.get_server_args", lambda a=args: a
                ):
                    with self.assertRaises(BreakableModeRefused) as cm:
                        validate_breakable_boot(0.5, layer_id=7)
                msg = str(cm.exception)
                # names the phase, the layer and the way out
                self.assertIn("decode", msg)
                self.assertIn("7", msg)
                self.assertIn("--cuda-graph-backend-decode", msg)

    def test_the_covered_shape_still_passes(self):
        with patch(
            "sglang.srt.runtime_context.get_server_args",
            lambda: _args(decode="breakable", prefill="disabled"),
        ):
            validate_breakable_boot(0.5, layer_id=1)

    def test_a_wrong_decode_backend_is_still_refused_by_name(self):
        with patch(
            "sglang.srt.runtime_context.get_server_args",
            lambda: _args(decode="full", prefill="disabled"),
        ):
            with self.assertRaises(BreakableModeRefused) as cm:
                validate_breakable_boot(0.5, layer_id=1)
            self.assertIn("'full'", str(cm.exception))

    def test_a_captured_prefill_is_still_refused(self):
        with patch(
            "sglang.srt.runtime_context.get_server_args",
            lambda: _args(decode="breakable", prefill="full"),
        ):
            with self.assertRaises(BreakableModeRefused) as cm:
                validate_breakable_boot(0.5, layer_id=1)
            self.assertIn("prefill", str(cm.exception))

    def test_no_offload_is_still_refused_first(self):
        """Precondition 1 fires before anything reads the server args, so it
        must be reachable with no runtime context at all."""
        with patch("sglang.srt.runtime_context.get_server_args", _NoArgs()):
            with self.assertRaises(BreakableModeRefused) as cm:
                validate_breakable_boot(1.0, layer_id=2)
            self.assertIn("resident fraction", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
