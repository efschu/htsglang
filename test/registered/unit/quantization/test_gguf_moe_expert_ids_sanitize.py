"""Falsification + inertness proof for the GGUF-MoE expert_ids sanitizer
(task #109).

fused_moe_gguf (MMQ path, prefill / M>64) feeds expert_ids from
moe_align_block_size to the sgl-kernel `ggml_moe_a8`. moe_align_block_size
allocates expert_ids with torch.empty and only fills blocks up to
num_tokens_post_padded; the trailing entries are UNINITIALIZED garbage.
The MMQ device kernel (sgl-kernel moe.cuh moe_q) guards them with a
HARDCODED `exp_idx > 255 || exp_idx < 0`. That is exact for a full
256-row expert table, but under uneven-TP EXPERT-DIM sharding (#80/#81)
each rank's LOCAL weight table has only n_local+1 rows, so garbage ids in
[E_local, 255] pass the guard and the kernel reads
`vx + exp_idx * exp_stride` beyond the local weight tensor -> illegal
memory access (seen live: Qwen3.6-35B-A3B GGUF, TP=3, any prefill > 64
tokens).

Mitigation in gguf.py: `expert_ids = where(expert_ids < E, expert_ids,
-1)` before ggml_moe_a8, so the kernel's own `exp_idx < 0` guard rejects
trailing garbage. Every VALID block id is < E by construction, so only
out-of-range (garbage / trailing) blocks are touched. The durable
kernel-side bound (exp_idx < nrows(W_local) instead of 255) is backlog
task #112; this Python sanitizer is the correct mitigation now because
sgl_kernel ships prebuilt.

These tests exercise the EXACT sanitizer expression on CPU tensors — no
GPU, no kernel, no server.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _sanitize(expert_ids: torch.Tensor, E: int) -> torch.Tensor:
    """Mirrors the fused_moe_gguf sanitizer in gguf.py exactly.

    masked_fill with a scalar (not torch.where against a host scalar
    tensor) so the real op stays CUDA-graph-capture safe under MTP/NEXTN.
    """
    return expert_ids.masked_fill(expert_ids >= E, -1)


class TestGGUFMoEExpertIdsSanitizer(CustomTestCase):
    def test_sharded_garbage_is_sanitized(self):
        # Uneven-TP local table: n_local owned experts + 1 zero-pad expert
        # => E = n_local + 1 rows. Valid block ids live in [0, E). A
        # trailing garbage id like 200 (< 256, so the kernel's hardcoded
        # guard would WRONGLY accept it) must be mapped to -1.
        n_local = 114
        E = n_local + 1  # 115
        expert_ids = torch.tensor(
            [0, 5, n_local, 3, 200, 255, 130, -1], dtype=torch.int32
        )
        out = _sanitize(expert_ids, E)
        # every valid id (< 115) is preserved; 200/255/130 -> -1; existing
        # -1 stays -1.
        self.assertEqual(
            out.tolist(), [0, 5, n_local, 3, -1, -1, -1, -1]
        )
        # No surviving id can index beyond the local weight table.
        self.assertTrue(bool(((out < E) & (out >= -1)).all()))
        self.assertTrue(bool((out[out >= 0] < E).all()))

    def test_full_table_is_byte_identical_noop(self):
        # E == 256 (full expert table, e.g. TP=1 GGUF MoE): all valid ids
        # are < 256, so `where(id < 256, id, -1)` touches NOTHING. The
        # already-working full-table path must be provably unchanged.
        E = 256
        expert_ids = torch.arange(0, 256, dtype=torch.int32)
        out = _sanitize(expert_ids, E)
        self.assertTrue(torch.equal(out, expert_ids))

    def test_full_table_with_padding_sentinel_noop(self):
        # moe_align uses num_experts+1 padding; a full model gives E=256
        # and the align kernel's own padding sentinel is < 256 as well, so
        # a realistic mixed vector is still identity.
        E = 256
        expert_ids = torch.tensor(
            [0, 1, 255, 128, 42, 255, 3], dtype=torch.int32
        )
        out = _sanitize(expert_ids, E)
        self.assertTrue(torch.equal(out, expert_ids))

    def test_dtype_and_shape_preserved(self):
        E = 64
        expert_ids = torch.tensor([[0, 100], [63, 64]], dtype=torch.int32)
        out = _sanitize(expert_ids, E)
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.shape, expert_ids.shape)
        # 100 and 64 (== E) are out of [0, 64) -> -1; 63 (< 64) kept.
        self.assertEqual(out.tolist(), [[0, -1], [63, -1]])

    def test_boundary_E_is_excluded(self):
        # The zero-pad expert sits at local index n_local = E-1; index E
        # itself is out of range and must be rejected (strict `< E`).
        E = 10
        expert_ids = torch.tensor([9, 10, 11], dtype=torch.int32)
        out = _sanitize(expert_ids, E)
        self.assertEqual(out.tolist(), [9, -1, -1])


if __name__ == "__main__":
    unittest.main()
