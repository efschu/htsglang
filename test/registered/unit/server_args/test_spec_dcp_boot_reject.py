"""NGRAM / FROZEN_KV_MTP x DCP are refused at argument resolution (#229).

THE DEFECT THIS PINS
--------------------
``NGRAM_VERIFY`` and ``FROZEN_KV_MTP_VERIFY`` are outside both attention
backends' ``_DCP_VERIFY_SPEC_INPUT_TYPES``, so any ``--dcp-size > 1`` boot
with either algorithm is refused -- but the refusal used to fire in the spec
worker init (ngram) or the first target-verify metadata build (frozen KV MTP).
Everything before that succeeds: weights load, pools are sized, graphs are
captured, the server reports ready. The user paid a full model load to learn
about a configuration error that is decidable from ``server_args`` alone.

``ServerArgs._handle_dcp_validation`` now calls the two gates
(``reject_ngram_verify_under_dcp`` / ``reject_frozen_kv_mtp_verify_under_dcp``
in ``spec_info``) during ``__post_init__``, i.e. at argument resolution in the
launcher process. Nothing in this module touches a model, a GPU or a worker:
a raise out of ``_handle_dcp_validation`` on a bare ``ServerArgs`` IS the
proof that the reject lands before any load.
"""

import inspect
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def make_args(**kwargs):
    """``model_path='dummy'`` short-circuits ``__post_init__`` (no resolution,
    no strict mutation guard), so the DCP handler can be driven in isolation
    with exactly the fields under test."""
    return ServerArgs(model_path="dummy", **kwargs)


class TestNgramDcpResolutionReject(unittest.TestCase):
    def test_ngram_with_dcp_is_rejected_at_resolution(self):
        for dcp_size in (2, 3, 8):
            with self.subTest(dcp_size=dcp_size):
                args = make_args(
                    speculative_algorithm="NGRAM", dcp_size=dcp_size
                )
                with self.assertRaises(ValueError) as cm:
                    args._handle_dcp_validation()
                msg = str(cm.exception)
                # The message names the condition and the way out, not just
                # "unsupported".
                self.assertIn("NGRAM_VERIFY", msg)
                self.assertIn(f"--dcp-size {dcp_size}", msg)
                self.assertIn("--dcp-size 1", msg)

    def test_frozen_kv_mtp_with_dcp_is_rejected_at_resolution(self):
        for dcp_size in (2, 3, 8):
            with self.subTest(dcp_size=dcp_size):
                args = make_args(
                    speculative_algorithm="FROZEN_KV_MTP", dcp_size=dcp_size
                )
                with self.assertRaises(ValueError) as cm:
                    args._handle_dcp_validation()
                msg = str(cm.exception)
                self.assertIn("FROZEN_KV_MTP_VERIFY", msg)
                self.assertIn(f"--dcp-size {dcp_size}", msg)
                self.assertIn("--dcp-size 1", msg)

    def test_dcp_size_one_is_inert(self):
        """The default path must not move: dcp_size <= 1 is every stock boot."""
        for algo in ("NGRAM", "FROZEN_KV_MTP", None):
            with self.subTest(algo=algo):
                args = make_args(speculative_algorithm=algo, dcp_size=1)
                args._handle_dcp_validation()  # must not raise


class TestGatePlacement(unittest.TestCase):
    def test_resolution_pipeline_reaches_the_handler(self):
        """``__post_init__`` must call ``_handle_dcp_validation``; that call is
        what makes the gate a construction-time refusal rather than a helper
        nobody runs (#182's lesson, one layer up)."""
        src = inspect.getsource(ServerArgs.__post_init__)
        self.assertIn("_handle_dcp_validation()", src)

    def test_gates_precede_the_platform_branches(self):
        """The two gates must sit before the ``is_hip()`` early return: the
        late refusals they replace fire on every platform (the Triton verify
        arm raises for any spec type outside the split's set whenever
        ``dcp_size > 1``), so the boot gate must too."""
        src = inspect.getsource(ServerArgs._handle_dcp_validation)
        for gate in (
            "reject_ngram_verify_under_dcp(",
            "reject_frozen_kv_mtp_verify_under_dcp(",
        ):
            self.assertIn(gate, src)
            self.assertLess(src.index(gate), src.index("is_hip()"), gate)


if __name__ == "__main__":
    unittest.main()
