"""--dcp-size on flashinfer must not be a silent no-op.

Every DCP branch in ``FlashInferAttnBackend`` is gated on ``self.uneven_dcp``
(a --rank-tp-ratio plan with dcp_size == tp_size, or the weightless-KV fast
lane). Upstream flashinfer has no DCP path at all, so with that predicate
false the backend does not fall back to a slower DCP -- it runs stock full-KV
attention. Measured (Qwen3.5-2B, TP=2/DCP=2, SGLANG_UNEVEN_TOKEN_VECTOR=2,1,
no plan): boots green, output token-identical to TP=1, zero uneven-machinery
log lines. Correct answers, and the flag did nothing.

``test_the_forwards_have_no_even_dcp_path_to_fall_back_on`` pins the mechanism
claim, so the guard cannot outlive its own premise: if someone later teaches
this backend a real even-DCP path, that test fails and the rejection must be
revisited rather than silently keeping a now-wrong refusal.

CPU only: the rule is a pure function of the decision inputs.
"""

import inspect
import unittest

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    reject_silently_inert_dcp,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFlashInferDcpNoOpGuard(CustomTestCase):
    def test_plain_even_dcp_is_rejected_not_ignored(self):
        """THE falsifier: the measured config (dcp_size 2, no plan, no fast
        lane, target worker) used to boot and serve plain TP."""
        with self.assertRaises(ValueError) as ctx:
            reject_silently_inert_dcp(2, uneven_dcp=False, is_draft_worker=False)
        msg = str(ctx.exception)
        self.assertIn("SILENTLY IGNORED", msg)
        # names all three ways out, so the message is actionable
        self.assertIn("--rank-tp-ratio", msg)
        self.assertIn("triton", msg)
        self.assertIn("drop --dcp-size", msg)

    def test_the_validated_uneven_arm_is_untouched(self):
        for dcp in (2, 3, 8):
            reject_silently_inert_dcp(dcp, uneven_dcp=True, is_draft_worker=False)

    def test_the_draft_worker_is_exempt_by_design(self):
        """M4: dcp_size lives in the parallel context, so an EAGLE/NEXTN draft
        runner sees dcp_size > 1 as well, and deliberately does NOT token-shard
        its 1-layer full-context KV pool -- its uneven_dcp is forced False for
        that reason. Rejecting it would refuse the validated MTP + uneven-DCP
        arm, so this direction is asserted explicitly and not left to luck.
        """
        for dcp in (2, 3):
            reject_silently_inert_dcp(dcp, uneven_dcp=False, is_draft_worker=True)

    def test_dcp_off_is_inert(self):
        """The default path: no DCP, nothing to reject, whatever else is set."""
        for dcp in (0, 1):
            for uneven in (False, True):
                for draft in (False, True):
                    with self.subTest(dcp=dcp, uneven=uneven, draft=draft):
                        reject_silently_inert_dcp(
                            dcp, uneven_dcp=uneven, is_draft_worker=draft
                        )

    def test_the_constructor_calls_the_rule(self):
        src = inspect.getsource(FlashInferAttnBackend.__init__)
        self.assertIn("reject_silently_inert_dcp(", src)
        self.assertIn("uneven_dcp=self.uneven_dcp", src)

    def test_the_forwards_have_no_even_dcp_path_to_fall_back_on(self):
        """Pins the premise of the rejection: both DCP forwards are entered
        only through ``self.uneven_dcp``."""
        for name in ("forward_extend", "forward_decode"):
            src = inspect.getsource(getattr(FlashInferAttnBackend, name))
            self.assertIn(
                "if self.uneven_dcp:",
                src,
                f"{name} no longer gates its DCP work on self.uneven_dcp; the "
                f"no-op rejection's premise must be re-checked",
            )


if __name__ == "__main__":
    unittest.main()
