"""#470 Boot B: the draft ServerArgs copy must be PUBLISHED, not just handed on.

The 2026-08-04 DSV4F window fixed the draft copy itself -- under solo placement
``build_draft_tp_worker`` neutralises the target's per-rank vectors, which are
meaningless for an unsharded weight-TP=1 draft -- and the boot refused
identically anyway:

    ValueError: the resident-fraction vector has 3 entries (0.23,0.42,0.42)
    but tensor parallelism is 1

Mechanism (``TICKET_470_RESULT_first_boot.md`` §2, "NOT FIXED"): the copy only
reaches readers that are HANDED it. ``resident_fraction._from_flag()`` falls
back to the process-wide runtime context when no ServerArgs is passed, and the
context still held the TARGET's arguments for the whole draft build, because
``draft_server_args`` went into ``TpModelWorker`` and nowhere else.

This pins the CONTRACT, not the arithmetic: what the context resolves to inside
the build, that the neutralised fields are the ones it resolves, and that the
target's arguments come back afterwards on both the normal and the exception
path. No GPU, no model, no weights -- ``TpModelWorker`` is replaced by a probe
that reports what the context looked like from inside the build.
"""

from __future__ import annotations

import unittest
from unittest import mock

import sglang.srt.speculative.draft_worker_common as dwc
from sglang.srt.layers.moe.resident_fraction import resident_fraction_vector
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: The rig geometry the defect was found on: 5090 + 2x 3080, TP=3, the measured
#: cut from TICKET_470 Boot A.
TARGET_RESIDENT_VECTOR = "0.23,0.42,0.42"

#: Every per-rank vector the solo draft copy neutralises. rank_gpu_id is
#: deliberately absent: it is placement, not sharding, and the solo host is
#: resolved through it.
NEUTRALISED_FIELDS = (
    "rank_moe_resident_fraction",
    "rank_moe_ratio",
    "rank_kv_ratio",
    "rank_tp_ratio",
    "rank_auto_reserve_mib",
)


class _TargetModelConfig:
    context_len = 4096


def _target_args(placement="solo"):
    args = ServerArgs(model_path="dummy")
    args.override(
        "test.target",
        rank_moe_resident_fraction=TARGET_RESIDENT_VECTOR,
        rank_moe_ratio="1,1,1",
        rank_kv_ratio="1,1,1",
        rank_tp_ratio="1,1,1",
        rank_auto_reserve_mib="0,0,0",
        speculative_draft_placement=placement,
    )
    return args


class _ContextProbe:
    """Stands in for TpModelWorker and records what the context resolved to.

    ``during`` is invoked inside the build so an arm can run a real reader
    (the resident-fraction validator) at exactly the point the boot died.
    """

    instances: list = []

    def __init__(self, **kwargs):
        probe = _ContextProbe.instances[-1]
        probe["published"] = get_server_args()
        probe["worker_kwargs"] = kwargs
        during = probe.get("during")
        if during is not None:
            probe["during_result"] = during()
        self.model_runner = mock.MagicMock()


class DraftArgsContextPublicationTest(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(self._clear_context)

    @staticmethod
    def _clear_context():
        get_context()._server_args = None

    def _build(self, target, *, during=None, worker=_ContextProbe):
        get_context().set_server_args(target)
        probe: dict = {"during": during}
        _ContextProbe.instances.append(probe)
        try:
            with mock.patch.object(dwc, "TpModelWorker", worker):
                dwc.build_draft_tp_worker(
                    server_args=target,
                    gpu_id=0,
                    tp_rank=0,
                    dp_rank=None,
                    moe_ep_rank=0,
                    attn_cp_rank=0,
                    moe_dp_rank=0,
                    nccl_port=12345,
                    target_model_config=_TargetModelConfig(),
                    algo_label="TEST",
                )
        finally:
            _ContextProbe.instances.pop()
        return probe

    # ---------------------------------------------------------------- publish

    def test_the_context_resolves_to_the_draft_copy_during_the_build(self):
        target = _target_args()
        probe = self._build(target)
        published = probe["published"]
        self.assertIsNot(
            published,
            target,
            "the runtime context still resolved to the TARGET's arguments "
            "inside the draft build -- this is the #470 Boot B defect",
        )
        self.assertIs(
            published,
            probe["worker_kwargs"]["server_args"],
            "the published object must be the same copy TpModelWorker was "
            "handed, or two readers of the same build see two configurations",
        )

    def test_the_published_copy_carries_the_neutralised_vectors(self):
        probe = self._build(_target_args())
        for field in NEUTRALISED_FIELDS:
            self.assertIsNone(
                getattr(probe["published"], field),
                f"{field} is still target-shaped in the published draft args",
            )

    def test_the_published_copy_carries_the_draft_overrides(self):
        """Publication is only useful if it publishes the RESOLVED copy."""
        probe = self._build(_target_args())
        published = probe["published"]
        self.assertTrue(published.skip_tokenizer_init)
        self.assertEqual(published.context_length, _TargetModelConfig.context_len)
        self.assertIsNotNone(published.speculative_draft_attention_backend)
        self.assertEqual(
            published.attention_backend,
            published.speculative_draft_attention_backend,
        )
        self.assertIsNone(published.prefill_attention_backend)
        self.assertIsNone(published.decode_attention_backend)

    # ------------------------------------------------------- the real reader

    def test_the_validator_that_refused_the_boot_now_resolves(self):
        """The load-bearing arm: the exact call that killed Boot B.

        ``resident_fraction_vector(tp_size=1)`` is what the weight-TP=1 draft
        build reaches through ``get_server_args()``. Against the target's
        3-entry vector it raises; against the published draft copy it must
        resolve to the scalar default broadcast over one rank.
        """
        probe = self._build(
            _target_args(), during=lambda: resident_fraction_vector(tp_size=1)
        )
        self.assertEqual(len(probe["during_result"]), 1)

    def test_can_fail_the_target_vector_still_refuses_at_weight_tp_1(self):
        """Falsifier for the arm above: without the publication the same call
        raises, so a green result there is not an artefact of tp_size=1."""
        get_context().set_server_args(_target_args())
        with self.assertRaises(ValueError) as caught:
            resident_fraction_vector(tp_size=1)
        self.assertIn("3 entries", str(caught.exception))

    # ---------------------------------------------------------------- restore

    def test_the_target_arguments_come_back_after_the_build(self):
        target = _target_args()
        self._build(target)
        self.assertIs(get_server_args(), target)

    def test_the_target_arguments_come_back_when_the_build_raises(self):
        target = _target_args()

        class _Boom:
            def __init__(self, **kwargs):
                raise RuntimeError("draft build failed")

        get_context().set_server_args(target)
        with mock.patch.object(dwc, "TpModelWorker", _Boom):
            with self.assertRaises(RuntimeError):
                dwc.build_draft_tp_worker(
                    server_args=target,
                    gpu_id=0,
                    tp_rank=0,
                    dp_rank=None,
                    moe_ep_rank=0,
                    attn_cp_rank=0,
                    moe_dp_rank=0,
                    nccl_port=12345,
                    target_model_config=_TargetModelConfig(),
                    algo_label="TEST",
                )
        self.assertIs(
            get_server_args(),
            target,
            "a failed draft build must not leave the draft copy published -- "
            "the target's own scheduler reads this context for the rest of "
            "the process's life",
        )

    # ------------------------------------------------------------- neutrality

    def test_split_placement_publishes_the_copy_with_its_vectors_intact(self):
        """Non-solo boots keep every per-rank vector: the draft is sharded like
        the target there, so neutralising would be the wrong answer. Only the
        publication is new for them."""
        target = _target_args(placement="split")
        probe = self._build(target)
        self.assertIsNot(probe["published"], target)
        self.assertEqual(
            probe["published"].rank_moe_resident_fraction, TARGET_RESIDENT_VECTOR
        )


if __name__ == "__main__":
    unittest.main()
