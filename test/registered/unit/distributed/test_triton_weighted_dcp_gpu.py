"""#173, device half: the weighted-DCP kv-index build on a real GPU.

The CPU twin (``test_triton_weighted_dcp_wiring.py``) pins the owner rule, the
head geometry and the collective gate -- everything that is pure arithmetic.
What it cannot reach is the one device-bound step inside
``build_dcp_weighted_kv_indices``: the Triton kernel that materialises
``req_to_token[req, kv_start : kv_start+len]``. That step is what turns the
rule into an actual ``kv_indptr`` / ``kv_indices`` pair, and it is shared with
the FlashInfer backend, so this file checks the composed result against the
same prose reference the CPU tests use.

Also covered here, because both need device tensors:

  * ``TritonAttnBackend._dcp_weighted_kv_indices``' BUFFER CONTRACT -- eager
    gets a fresh exactly-sized tensor, a caller that hands in a capture-stable
    buffer gets that same buffer object back with the rows copied INTO it.
    Getting this wrong is a silent wrong-context decode under CUDA graphs, not
    a crash.
  * the write side's compact rows agreeing with the read side's, slot for slot,
    through the real ``dcp_weighted_write_slots``.

SKIPPED WITHOUT CUDA. Single-process, single-rank: no collectives, no model,
no server -- it fakes the per-rank bounds directly, so it runs in seconds in a
GPU window without competing for a whole card's worth of memory.

    python -m pytest test/registered/unit/distributed/test_triton_weighted_dcp_gpu.py -v
"""

import unittest

import numpy as np
import torch

from sglang.srt.distributed.utils import get_cp_token_ratios, set_cp_token_ratios
from sglang.srt.layers.dcp.owner import (
    build_dcp_weighted_kv_indices,
    dcp_weighted_owner_bounds,
    dcp_weighted_write_slots,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()

PLANS = ([2, 1, 1], [13, 30, 21], [1, 1, 1], [5, 3])


def _reference(table_np, lens, plan, rank, starts=None):
    """kv_indptr / kv_indices from the rule as prose (see the CPU twin)."""
    prefix = np.concatenate([[0], np.cumsum(plan)])
    S, lo = int(prefix[-1]), int(prefix[rank])
    ratio = int(prefix[rank + 1]) - lo
    idx, ptr = [], [0]
    for r, n in enumerate(lens):
        start = 0 if starts is None else int(starts[r])
        row = table_np[r, start : start + int(n)]
        own = row[((row % S) >= lo) & ((row % S) < lo + ratio)]
        idx.extend((own // S) * ratio + (own % S - lo))
        ptr.append(len(idx))
    return np.array(ptr, dtype=np.int64), np.array(idx, dtype=np.int64)


@unittest.skipUnless(_HAS_CUDA, "Triton kv-index kernels require a GPU")
class TestTritonWeightedDcpGpu(CustomTestCase):
    def setUp(self):
        self._saved = get_cp_token_ratios()
        self.dev = torch.device("cuda")
        rng = np.random.default_rng(173)
        self.n_req, self.width = 5, 128
        self.table_np = rng.integers(0, 4000, size=(self.n_req, self.width)).astype(
            np.int32
        )
        self.table = torch.from_numpy(self.table_np).to(self.dev)
        self.reqs = torch.arange(self.n_req, dtype=torch.int32, device=self.dev)

    def tearDown(self):
        set_cp_token_ratios(self._saved)

    def _bounds(self, plan, rank):
        set_cp_token_ratios(plan)
        return dcp_weighted_owner_bounds(len(plan), rank)

    def test_the_index_build_matches_the_prose_reference(self):
        lens_np = np.array([0, 1, 17, self.width, 63], dtype=np.int32)
        lens = torch.from_numpy(lens_np).to(self.dev)
        for plan in PLANS:
            for rank in range(len(plan)):
                with self.subTest(plan=plan, rank=rank):
                    S, lo, hi, ratio = self._bounds(plan, rank)
                    kv_indptr = torch.zeros(
                        self.n_req + 1, dtype=torch.int32, device=self.dev
                    )
                    ptr, idx = build_dcp_weighted_kv_indices(
                        self.table, self.reqs, lens, kv_indptr, None, S, lo, hi, ratio
                    )
                    ref_ptr, ref_idx = _reference(self.table_np, lens_np, plan, rank)
                    np.testing.assert_array_equal(ptr.cpu().numpy(), ref_ptr)
                    np.testing.assert_array_equal(
                        idx.cpu().numpy().astype(np.int64), ref_idx
                    )

    def test_the_index_build_honours_a_per_request_start_offset(self):
        """Chunked prefill reads [kv_start, kv_start+len); an ignored start
        offset silently attends the wrong window of the context."""
        starts_np = np.array([0, 3, 40, 0, 11], dtype=np.int32)
        lens_np = np.array([10, 10, 20, 5, 30], dtype=np.int32)
        starts = torch.from_numpy(starts_np).to(self.dev)
        lens = torch.from_numpy(lens_np).to(self.dev)
        for plan in PLANS:
            for rank in range(len(plan)):
                with self.subTest(plan=plan, rank=rank):
                    S, lo, hi, ratio = self._bounds(plan, rank)
                    kv_indptr = torch.zeros(
                        self.n_req + 1, dtype=torch.int32, device=self.dev
                    )
                    ptr, idx = build_dcp_weighted_kv_indices(
                        self.table, self.reqs, lens, kv_indptr, starts, S, lo, hi, ratio
                    )
                    ref_ptr, ref_idx = _reference(
                        self.table_np, lens_np, plan, rank, starts_np
                    )
                    np.testing.assert_array_equal(ptr.cpu().numpy(), ref_ptr)
                    np.testing.assert_array_equal(
                        idx.cpu().numpy().astype(np.int64), ref_idx
                    )

    def test_read_rows_are_exactly_the_rows_the_write_side_used(self):
        """The property the whole feature rests on. Write the batch's slots
        through dcp_weighted_write_slots, read them back through the index
        build, and require the two row sets to be identical."""
        lens_np = np.array([self.width] * self.n_req, dtype=np.int32)
        lens = torch.from_numpy(lens_np).to(self.dev)
        for plan in PLANS:
            for rank in range(len(plan)):
                with self.subTest(plan=plan, rank=rank):
                    S, lo, hi, ratio = self._bounds(plan, rank)
                    kv_indptr = torch.zeros(
                        self.n_req + 1, dtype=torch.int32, device=self.dev
                    )
                    _, read_rows = build_dcp_weighted_kv_indices(
                        self.table, self.reqs, lens, kv_indptr, None, S, lo, hi, ratio
                    )
                    flat = self.table.reshape(-1).to(torch.int64)
                    w_loc, w_mask = dcp_weighted_write_slots(flat, S, lo, hi, ratio)
                    write_rows = w_loc[w_mask]
                    self.assertEqual(
                        sorted(read_rows.cpu().tolist()),
                        sorted(write_rows.cpu().tolist()),
                    )

    def test_the_capture_stable_buffer_is_filled_in_place_and_returned(self):
        """CUDA-graph replay reads the buffer whose pointer was frozen at
        capture. Returning a fresh tensor instead would leave every replay
        decoding against the capture-time context -- fluent nonsense, no error.
        """
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        lens_np = np.array([20, 0, 33, 7, 64], dtype=np.int32)
        lens = torch.from_numpy(lens_np).to(self.dev)
        plan, rank = [13, 30, 21], 1
        S, lo, hi, ratio = self._bounds(plan, rank)

        # A stand-in with just the fields _dcp_weighted_kv_indices reads.
        be = TritonAttnBackend.__new__(TritonAttnBackend)
        be.req_to_token = self.table
        be.device = self.dev
        be.cp_S, be.cp_lo, be.cp_hi, be.cp_ratio = S, lo, hi, ratio

        buf = torch.full((4096,), -1, dtype=torch.int64, device=self.dev)
        ptr_buf = torch.zeros(self.n_req + 1, dtype=torch.int32, device=self.dev)
        ptr, out, owned = be._dcp_weighted_kv_indices(
            self.reqs, lens, ptr_buf, buf, None
        )
        self.assertIs(out, buf, "the capture-stable buffer must be returned as is")

        ptr_fresh = torch.zeros(self.n_req + 1, dtype=torch.int32, device=self.dev)
        ptr2, fresh, owned2 = be._dcp_weighted_kv_indices(
            self.reqs, lens, ptr_fresh, None, None
        )
        self.assertIsNot(fresh, buf)
        n = int(ptr[-1].item())
        np.testing.assert_array_equal(
            buf[:n].cpu().numpy(), fresh[:n].cpu().numpy()
        )
        ref_ptr, ref_idx = _reference(self.table_np, lens_np, plan, rank)
        np.testing.assert_array_equal(ptr.cpu().numpy(), ref_ptr)
        np.testing.assert_array_equal(buf[:n].cpu().numpy(), ref_idx)
        # owned lengths drive the split-KV schedule
        np.testing.assert_array_equal(
            owned.cpu().numpy(), np.diff(ref_ptr).astype(owned.cpu().numpy().dtype)
        )
        np.testing.assert_array_equal(owned.cpu().numpy(), owned2.cpu().numpy())

    def test_an_undersized_buffer_raises_instead_of_truncating(self):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        lens = torch.full((self.n_req,), self.width, dtype=torch.int32, device=self.dev)
        S, lo, hi, ratio = self._bounds([13, 30, 21], 1)
        be = TritonAttnBackend.__new__(TritonAttnBackend)
        be.req_to_token = self.table
        be.device = self.dev
        be.cp_S, be.cp_lo, be.cp_hi, be.cp_ratio = S, lo, hi, ratio
        tiny = torch.zeros(4, dtype=torch.int64, device=self.dev)
        ptr_buf = torch.zeros(self.n_req + 1, dtype=torch.int32, device=self.dev)
        with self.assertRaises(ValueError) as ctx:
            be._dcp_weighted_kv_indices(self.reqs, lens, ptr_buf, tiny, None)
        self.assertIn("overflow", str(ctx.exception))

    def test_a_rank_that_owns_nothing_still_gets_a_usable_index_tensor(self):
        """Short prefix + low token ratio -> zero owned rows. The kernels are
        then driven by an all-zero indptr and never dereference the index
        tensor, but a 0-element tensor has no storage to take a pointer from."""
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        table = torch.arange(64, dtype=torch.int32, device=self.dev).reshape(1, 64)
        S, lo, hi, ratio = self._bounds([13, 30, 21], 2)
        be = TritonAttnBackend.__new__(TritonAttnBackend)
        be.req_to_token = table
        be.device = self.dev
        be.cp_S, be.cp_lo, be.cp_hi, be.cp_ratio = S, lo, hi, ratio
        reqs = torch.zeros(1, dtype=torch.int32, device=self.dev)
        lens = torch.tensor([3], dtype=torch.int32, device=self.dev)
        ptr_buf = torch.zeros(2, dtype=torch.int32, device=self.dev)
        ptr, out, owned = be._dcp_weighted_kv_indices(reqs, lens, ptr_buf, None, None)
        self.assertEqual(ptr.cpu().tolist(), [0, 0])
        self.assertGreaterEqual(out.numel(), 1)
        self.assertEqual(owned.cpu().tolist(), [0])


if __name__ == "__main__":
    unittest.main()
