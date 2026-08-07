# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture shapes must be RANK-UNIFORM (#631).

``get_batch_sizes_to_capture`` both extends and clamps ``capture_bs`` by
``model_runner.req_to_token_pool.size``. That value is rank-LOCAL: it follows
each rank's own memory sizing, which under uneven TP or a heterogeneous group
differs by construction. Graph capture replays a collective per captured shape,
so a rank-local shape list makes the ranks run DIFFERENT numbers of collectives
-- the rank with the shorter list leaves the capture loop while its peer blocks
in the next all-reduce. No timeout, no error, no output.

This is not a hypothetical. Three decode-arm boots on the same rig, same code,
same afternoon:

    TP0 bs=[..,16,19]   TP1 bs=[..,16,24]     -> wedged 20 min
    TP0 bs=[1..8,10]    TP1 bs=[1..8,10,12]   -> wedged
    TP0 bs=[..,16,24]   TP1 bs=[..,16,24]     -> captured in 50 s, served

The third was not configured differently. Both of its ranks simply held pools
at least as large as the biggest configured bs, so neither clamped and the
lists coincided by luck. A property that holds by luck is what this test
converts into one that holds by construction.

The decisive test is :meth:`test_two_ranks_converge_on_one_list` -- it drives
the real function twice with the two ranks' OWN local sizes and asserts the
outputs are equal. Without the min-reduce that assertion fails, which is the
can-fail proof: it reproduces the exact 19-vs-24 divergence that hung the rig.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MOD = "sglang.srt.model_executor.runner.base_cuda_graph_runner"


def _model_runner(pool_size: int, configured_bs):
    """The narrow surface get_batch_sizes_to_capture actually reads."""
    return SimpleNamespace(
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=configured_bs)),
            enable_two_batch_overlap=False,
            torch_compile_max_bs=32,
        ),
        req_to_token_pool=SimpleNamespace(size=pool_size),
    )


class _Parallel(SimpleNamespace):
    pass


def _run(pool_size, configured_bs, tp_size, peer_min=None):
    """Call the real function with the group collapsed to a stub.

    ``peer_min`` is the smallest pool across the simulated group; the stubbed
    all_reduce(MIN) writes it back exactly as gloo would.
    """
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        get_batch_sizes_to_capture,
    )

    def fake_all_reduce(t, op=None, group=None):
        t.fill_(min(int(t.item()), int(peer_min)))

    with mock.patch(
        f"{_MOD}.get_parallel",
        return_value=_Parallel(tp_size=tp_size, attn_tp_size=1, attn_cp_size=1),
    ), mock.patch(
        f"{_MOD}.get_flags",
        return_value=SimpleNamespace(
            capture=SimpleNamespace(enable_torch_compile=False)
        ),
    ), mock.patch(
        f"{_MOD}.require_gathered_buffer", return_value=False
    ), mock.patch(
        "sglang.srt.distributed.get_tp_group",
        return_value=SimpleNamespace(cpu_group=object()),
    ), mock.patch(
        "torch.distributed.all_reduce", side_effect=fake_all_reduce
    ):
        return get_batch_sizes_to_capture(_model_runner(pool_size, configured_bs))[0]


class CaptureBsRankUniformityTest(CustomTestCase):
    CONFIGURED = [1, 2, 4, 8, 12, 16, 24]

    def test_two_ranks_converge_on_one_list(self):
        """The rig's actual failure, reproduced and fixed.

        Rank 0 sized a pool of 19, rank 1 a pool of 24, from the same boot.
        After the min-reduce both must build the SAME shapes.
        """
        rank0 = _run(19, self.CONFIGURED, tp_size=2, peer_min=19)
        rank1 = _run(24, self.CONFIGURED, tp_size=2, peer_min=19)
        self.assertEqual(
            rank0,
            rank1,
            "the two ranks captured different shape lists; graph capture "
            "replays a collective per shape, so this deadlocks the group",
        )
        self.assertEqual(max(rank0), 19, "both ranks must honour the SMALLEST pool")

    def test_the_divergence_is_real_without_the_reduce(self):
        """Can-fail proof: with tp_size == 1 no reduce happens, so the two
        local sizes genuinely produce different lists. If this assertion ever
        stops holding, the test above has stopped proving anything."""
        rank0 = _run(19, self.CONFIGURED, tp_size=1)
        rank1 = _run(24, self.CONFIGURED, tp_size=1)
        self.assertNotEqual(
            rank0, rank1, "expected the unreduced, rank-local lists to differ"
        )

    def test_single_rank_path_is_untouched(self):
        """tp_size == 1 must not reduce and must not change behaviour: there
        is no peer to diverge from, and the default path stays byte-identical."""
        with mock.patch("torch.distributed.all_reduce") as ar:
            got = _run(24, self.CONFIGURED, tp_size=1)
        ar.assert_not_called()
        self.assertEqual(got, [1, 2, 4, 8, 12, 16, 24])

    def test_uniform_group_is_unchanged_by_the_reduce(self):
        """When both ranks already agree the reduce is a no-op -- the case
        that was silently working before, and must keep working."""
        rank0 = _run(64, self.CONFIGURED, tp_size=2, peer_min=64)
        rank1 = _run(64, self.CONFIGURED, tp_size=2, peer_min=64)
        self.assertEqual(rank0, rank1)
        self.assertEqual(rank0, [1, 2, 4, 8, 12, 16, 24])


if __name__ == "__main__":
    unittest.main()
