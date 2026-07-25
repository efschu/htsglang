"""The spec kernel backend is ONE GROUP-WIDE decision -- CPU only.

The tree build and the greedy verify exist twice in this tree: as sgl_kernel
ops and as Triton kernels. Both dispatches used to be keyed on PLATFORM, so a
build where `_is_cuda` is true but sgl_kernel has no code for the arch (sm75,
gfx900) fell into the native branch and called a symbol that is None:

    eagle_utils.py build_tree_kernel_efficient
    TypeError: 'NoneType' object is not callable

measured on Nordstern L0 S3, ranks 3 (RTX 2080 Ti) and 4 (Vega 64).

Re-keying on LOCAL availability would boot and be wrong in the dangerous
direction. verify decides how many draft tokens are ACCEPTED, and the accepted
count changes the batch on every rank; ranks 0-2 on the CUDA kernel and ranks
3-4 on Triton would disagree about a candidate and diverge with no hang and no
error. Hence: one collective, MIN rules, all-or-nothing.

These tests pin the DECISION and the DISPATCH KEYING. They do not compare
kernel outputs -- that is byte-gate step 1 (native vs Triton on one card that
has both) and needs a GPU.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.speculative import eagle_utils as eu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeGroup:
    """Minimal stand-in for a GroupCoordinator: only world_size and cpu_group
    are touched by the decision."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.cpu_group = object()


def _fake_all_reduce(peer_has_native):
    """all_reduce(MIN) as the group would compute it with ONE peer whose
    capability is `peer_has_native`."""

    def _impl(tensor, op=None, group=None):
        tensor.fill_(min(int(tensor.item()), int(peer_has_native)))

    return _impl


class TestBackendSelector(CustomTestCase):
    def setUp(self):
        eu.reset_spec_kernel_backend()

    def tearDown(self):
        eu.reset_spec_kernel_backend()

    def test_asserts_before_defaulting(self):
        """The important property. A default would silently restore the
        per-rank answer on any path that forgot the collective."""
        with self.assertRaises(RuntimeError) as cm:
            eu.get_spec_kernel_backend()
        self.assertIn("ensure_spec_kernel_backend", str(cm.exception))

    def test_set_get_roundtrip_and_idempotence(self):
        eu.set_spec_kernel_backend(eu.SPEC_KERNEL_BACKEND_TRITON)
        self.assertEqual(eu.get_spec_kernel_backend(), "triton")
        eu.set_spec_kernel_backend(eu.SPEC_KERNEL_BACKEND_TRITON)  # same: fine

    def test_conflicting_decision_is_refused(self):
        """Two different answers in one process means the collective ran twice
        with different inputs -- the divergence this machinery prevents."""
        eu.set_spec_kernel_backend(eu.SPEC_KERNEL_BACKEND_NATIVE)
        with self.assertRaisesRegex(RuntimeError, "already decided"):
            eu.set_spec_kernel_backend(eu.SPEC_KERNEL_BACKEND_TRITON)

    def test_unknown_backend_refused(self):
        with self.assertRaises(ValueError):
            eu.set_spec_kernel_backend("cuda-graphs-please")


class TestGroupDecision(CustomTestCase):
    """The MIN semantics, with the collective mocked."""

    def setUp(self):
        eu.reset_spec_kernel_backend()

    def tearDown(self):
        eu.reset_spec_kernel_backend()

    def _decide(self, local_native, peer_native, topk=1, world=2):
        with mock.patch.object(
            eu, "_has_native_spec_kernels", local_native
        ), mock.patch.object(
            torch.distributed, "all_reduce", _fake_all_reduce(peer_native)
        ):
            return eu.decide_spec_kernel_backend(topk, tp_group=_FakeGroup(world))

    def test_all_native_stays_native(self):
        self.assertEqual(self._decide(True, True), "native")

    def test_one_rank_without_kernels_moves_EVERY_rank_to_triton(self):
        """THE case: this rank HAS the native kernels and still switches,
        because a peer does not. That is what makes the group uniform."""
        self.assertEqual(self._decide(True, False), "triton")

    def test_rank_without_kernels_also_gets_triton(self):
        self.assertEqual(self._decide(False, True), "triton")

    def test_single_rank_group_uses_its_own_capability(self):
        eu.reset_spec_kernel_backend()
        with mock.patch.object(eu, "_has_native_spec_kernels", False):
            self.assertEqual(
                eu.decide_spec_kernel_backend(1, tp_group=_FakeGroup(1)), "triton"
            )

    def test_decision_is_uniform_across_ranks(self):
        """Same group inputs -> same verdict on every rank, whatever each
        rank's own capability is."""
        verdicts = {self._decide(local, False) for local in (True, False)}
        self.assertEqual(verdicts, {"triton"})

    def test_topk_gt_1_refused_on_the_triton_path(self):
        """Trees are not validated on the Triton kernels: they have run on XPU
        only, never on sm75 or gfx900, and the Triton build does not implement
        QLEN_ONLY_BITPACKING. Refuse rather than let an unvalidated kernel
        decide accept counts. L0 runs topk=1 chains, so nothing is lost."""
        with self.assertRaisesRegex(ValueError, "topk"):
            self._decide(True, False, topk=4)

    def test_topk_gt_1_allowed_on_the_native_path(self):
        self.assertEqual(self._decide(True, True, topk=4), "native")

    def test_refusal_leaves_no_half_decision_behind(self):
        with self.assertRaises(ValueError):
            self._decide(True, False, topk=4)
        with self.assertRaises(RuntimeError):
            eu.get_spec_kernel_backend()


@unittest.skipUnless(eu._is_cuda or eu._is_hip, "dispatch keying is CUDA/HIP-specific")
class TestDispatchKeying(CustomTestCase):
    """The re-key itself: which implementation the dispatch reaches.

    No kernel runs -- both implementations are replaced by recorders, so this
    is about the BRANCH, on a CPU-only test host.
    """

    def setUp(self):
        eu.reset_spec_kernel_backend()

    def tearDown(self):
        eu.reset_spec_kernel_backend()

    def _build_args(self):
        bs, topk, steps, num_verify = 1, 1, 2, 3
        return dict(
            bonus_tokens=torch.zeros((bs,), dtype=torch.long),
            parent_list=[torch.zeros((bs, topk * steps + 1), dtype=torch.long)],
            top_scores_index=torch.zeros((bs, steps), dtype=torch.long),
            draft_tokens=torch.zeros((bs, num_verify - 1), dtype=torch.long),
            seq_lens=torch.ones((bs,), dtype=torch.long),
            seq_lens_sum=1,
            topk=topk,
            spec_steps=steps,
            num_verify_tokens=num_verify,
        )

    def test_build_reaches_triton_when_the_group_decided_triton(self):
        eu.set_spec_kernel_backend(eu.SPEC_KERNEL_BACKEND_TRITON)
        calls = []
        with mock.patch.object(
            eu, "sgl_build_tree_kernel_triton", lambda *a, **k: calls.append("triton")
        ), mock.patch.object(
            eu, "sgl_build_tree_kernel_efficient", lambda *a, **k: calls.append("native")
        ):
            eu.build_tree_kernel_efficient(**self._build_args())
        self.assertEqual(calls, ["triton"])

    def test_build_reaches_native_when_the_group_decided_native(self):
        eu.set_spec_kernel_backend(eu.SPEC_KERNEL_BACKEND_NATIVE)
        calls = []
        with mock.patch.object(
            eu, "sgl_build_tree_kernel_triton", lambda *a, **k: calls.append("triton")
        ), mock.patch.object(
            eu, "sgl_build_tree_kernel_efficient", lambda *a, **k: calls.append("native")
        ):
            eu.build_tree_kernel_efficient(**self._build_args())
        self.assertEqual(calls, ["native"])

    def test_build_without_a_decision_raises_instead_of_guessing(self):
        with self.assertRaises(RuntimeError):
            eu.build_tree_kernel_efficient(**self._build_args())

    def _verify_args(self):
        bs, n = 1, 3
        return dict(
            predicts=torch.zeros((bs * n,), dtype=torch.int32),
            accept_index=torch.full((bs, n), -1, dtype=torch.int32),
            accept_token_num=torch.zeros((bs,), dtype=torch.int32),
            candidates=torch.zeros((bs, n), dtype=torch.int32),
            retrieve_index=torch.zeros((bs, n), dtype=torch.int32),
            retrieve_next_token=torch.zeros((bs, n), dtype=torch.int32),
            retrieve_next_sibling=torch.zeros((bs, n), dtype=torch.int32),
            target_predict=torch.zeros((bs, n), dtype=torch.int32),
        )

    def test_verify_reaches_triton_when_the_group_decided_triton(self):
        eu.set_spec_kernel_backend(eu.SPEC_KERNEL_BACKEND_TRITON)
        calls = []
        with mock.patch.object(
            eu, "verify_tree_greedy_triton", lambda **k: calls.append("triton")
        ), mock.patch.object(
            eu, "sgl_verify_tree_greedy", lambda **k: calls.append("native")
        ):
            eu.verify_tree_greedy_func(**self._verify_args())
        self.assertEqual(calls, ["triton"])

    def test_verify_reaches_native_when_the_group_decided_native(self):
        eu.set_spec_kernel_backend(eu.SPEC_KERNEL_BACKEND_NATIVE)
        calls = []
        with mock.patch.object(
            eu, "verify_tree_greedy_triton", lambda **k: calls.append("triton")
        ), mock.patch.object(
            eu, "sgl_verify_tree_greedy", lambda **k: calls.append("native")
        ):
            eu.verify_tree_greedy_func(**self._verify_args())
        self.assertEqual(calls, ["native"])


class TestAvailabilityFlags(CustomTestCase):
    def test_verify_has_an_availability_flag_at_all(self):
        """It had none: the import lived inside verify_tree_greedy_func, so a
        build without sgl_kernel raised ImportError mid-verify. Both
        capabilities are probed at import time now."""
        self.assertIsInstance(eu._has_sgl_build_tree_kernel, bool)
        self.assertIsInstance(eu._has_sgl_verify_tree_greedy, bool)
        self.assertEqual(
            eu._has_native_spec_kernels,
            eu._has_sgl_build_tree_kernel and eu._has_sgl_verify_tree_greedy,
        )

    def test_no_inline_sgl_kernel_import_left_in_the_verify_dispatch(self):
        import inspect

        src = inspect.getsource(eu.verify_tree_greedy_func)
        self.assertNotIn(
            "from sgl_kernel import verify_tree_greedy",
            src,
            "the inline import is back -- a missing op must be a startup fact, "
            "not an ImportError in the middle of the verify step",
        )


if __name__ == "__main__":
    unittest.main()
