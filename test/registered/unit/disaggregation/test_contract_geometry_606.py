"""#606 blast-radius: PD transport identity requires total_kv_head_num and
head_dim on kv_args.

If either is missing, ``identity_from_args`` must raise rather than silently
producing a zero that breaks the destination's head layout reconstruction.
"""

import types
import unittest

from sglang.srt.disaggregation.nccl.contract import identity_from_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_server_args(**kw):
    return types.SimpleNamespace(
        model_path="/fake/model",
        revision=None,
        dtype="bfloat16",
        quantization=None,
        dcp_size=1,
        kv_cache_dtype="auto",
        page_size=1,
        tp_size=2,
        pp_size=1,
        **kw,
    )


def _make_kv_args(**kw):
    return types.SimpleNamespace(
        state_types=(),
        engine_rank=0,
        **kw,
    )


class TestContractGeometryRequiredFields(unittest.TestCase):
    """total_kv_head_num and head_dim are contractually required."""

    def test_missing_total_kv_head_num_raises(self):
        """No total_kv_head_num on kv_args -> ValueError."""
        sa = _make_server_args()
        ka = _make_kv_args(head_dim=128)  # head_dim present, total_kv_head_num absent

        with self.assertRaises(ValueError) as cm:
            identity_from_args(sa, ka)
        self.assertIn("total_kv_head_num", str(cm.exception))

    def test_missing_head_dim_raises(self):
        """No head_dim on kv_args -> ValueError."""
        sa = _make_server_args()
        ka = _make_kv_args(total_kv_head_num=16)  # total present, head_dim absent

        with self.assertRaises(ValueError) as cm:
            identity_from_args(sa, ka)
        self.assertIn("head_dim", str(cm.exception))

    def test_can_fail_proof_restoring_default_zero_would_silence_errors(self):
        """If someone restores ``getattr(kv_args, 'total_kv_head_num', 0)`` this
        test goes red because no exception is raised.

        Mechanical proof: the old code would silently produce
        total_kv_head_num=0, breaking the destination's head layout.
        """
        sa = _make_server_args()
        ka = _make_kv_args(head_dim=128)  # missing total_kv_head_num

        with self.assertRaises(ValueError):
            identity_from_args(sa, ka)

    def test_both_present_produces_valid_identity(self):
        """Both fields present -> TransportIdentity built with correct values."""
        sa = _make_server_args()
        ka = _make_kv_args(total_kv_head_num=16, head_dim=128)

        identity = identity_from_args(sa, ka)
        self.assertEqual(identity.total_kv_head_num, 16)
        self.assertEqual(identity.head_dim, 128)


if __name__ == "__main__":
    unittest.main()
