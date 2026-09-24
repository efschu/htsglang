"""MRotaryEmbedding.axis_map covers every mrope layout (upstream #34446).

Ported from upstream ``test/registered/rotary/test_mrope_axis_map.py``. The
fused Qwen3.5 QK-norm+RoPE kernel takes each rotary lane's position from the
row the map names, so the map has to reproduce the layouts the reference
paths build (``apply_interleaved_rope`` and the contiguous section split).
Before the port only GLM built a map; Qwen3.8-27B ([11, 11, 10], interleaved)
had none and the fused kernel read the temporal row alone for image tokens.

Fork adaptation: the fork's constructor reads ``get_server_args()`` (not a
``_force_native`` attribute), so server args are installed in ``setUp``; the
27B's own section is added to the interleaved cases.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.rotary_embedding.mrope import (
    Ernie4_5_VLRotaryEmbedding,
    MRotaryEmbedding,
    apply_interleaved_rope,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def build_mrope(
    mrope_section: list[int],
    rotary_dim: int,
    interleaved: bool = False,
    glm: bool = False,
) -> MRotaryEmbedding:
    return MRotaryEmbedding(
        head_size=rotary_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=mrope_section,
        mrope_interleaved=interleaved,
        mrope_interleaved_glm=glm,
    )


def select_by_axis(table: torch.Tensor, axis_map: torch.Tensor) -> torch.Tensor:
    """Take every lane from the axis the map names, the way the kernel does."""
    lanes = torch.arange(table.shape[2])
    return table[axis_map, :, lanes].T


class TestMRopeAxisMap(CustomTestCase):
    def setUp(self):
        cpu_patch = patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        cpu_patch.start()
        self.addCleanup(cpu_patch.stop)
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(0)

    def test_interleaved_matches_apply_interleaved_rope(self):
        # [11, 11, 10] over 64 rotary dims is Qwen3.8-27B (head_dim 256,
        # partial_rotary_factor 0.25). Under [1, 1, 30] the reference places
        # only 10 of the 30 lanes it was asked for, so this pins that the map
        # reproduces that loss.
        for section, rotary_dim in (
            ([24, 20, 20], 128),
            ([11, 11, 10], 64),
            ([1, 1, 30], 64),
        ):
            with self.subTest(section=section):
                rope = build_mrope(section, rotary_dim, interleaved=True)
                table = torch.randn(3, 7, rotary_dim // 2)
                torch.testing.assert_close(
                    select_by_axis(table, rope.axis_map),
                    apply_interleaved_rope(table, section),
                    atol=0,
                    rtol=0,
                )

    def test_qwen38_27b_map_uses_all_three_axes(self):
        # A map that is all zeros would pass for text and still drop h/w for
        # images -- the defect itself -- so pin the lane counts per axis.
        rope = build_mrope([11, 11, 10], 64, interleaved=True)
        counts = torch.bincount(rope.axis_map, minlength=3).tolist()
        self.assertEqual(counts, [11, 11, 10])

    def test_contiguous_matches_section_split(self):
        section, rotary_dim = [24, 20, 20], 128
        rope = build_mrope(section, rotary_dim)
        table = torch.randn(3, 7, rotary_dim // 2)
        torch.testing.assert_close(
            select_by_axis(table, rope.axis_map),
            torch.cat(
                [m[i] for i, m in enumerate(table.split(section, dim=-1))], dim=-1
            ),
            atol=0,
            rtol=0,
        )

    def test_glm_map_keeps_its_round_robin_order(self):
        """GLM's kernel is out of tree, so the order is pinned rather than compared."""
        rope = build_mrope([8, 12, 12], 64, interleaved=True, glm=True)
        want = [0, 1, 2] * 8 + [1, 1, 2, 1, 1, 2, 2, 2]
        self.assertEqual(rope.axis_map.tolist(), want)

    def test_only_glm_reaches_the_older_kernels(self):
        self.assertIsNone(
            build_mrope([24, 20, 20], 128, interleaved=True)._legacy_axis_map
        )
        glm = build_mrope([8, 12, 12], 64, interleaved=True, glm=True)
        torch.testing.assert_close(glm._legacy_axis_map, glm.axis_map, atol=0, rtol=0)

    def test_ernie_has_no_map(self):
        ernie = Ernie4_5_VLRotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=64,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
            mrope_section=[16, 16, 32],
        )
        self.assertIsNone(ernie.axis_map)

    def test_axis_map_is_a_non_persistent_buffer(self):
        # It must travel with .to(device) like cos_sin_cache (the fused kernel
        # reads it on the GPU) but must never enter a checkpoint state_dict.
        rope = build_mrope([11, 11, 10], 64, interleaved=True)
        self.assertIn("axis_map", dict(rope.named_buffers()))
        self.assertNotIn("axis_map", rope.state_dict())


if __name__ == "__main__":
    unittest.main()
