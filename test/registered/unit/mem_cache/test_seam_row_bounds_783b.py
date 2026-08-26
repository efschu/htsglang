"""#783b: the extent contract validated a LENGTH and admitted ROWS it never checked.

SPECIMEN (W40b, boot_w40_857strict_0825_1949.log, 19:52:52, PP2, whole instance
down 5 s later):

    prepare_for_extend -> restore_seam_state (schedule_batch.py:2748, :2091)
      -> Req.load_kv_cache (:1778) -> load_cpu_copy (memory_pool.py:3303)
      -> current_platform.synchronize()
    torch.AcceleratorError: CUDA error: an illegal memory access was encountered

THE CLASS. #783's contract (25e7849844) compares `kv_cache_cpu_extent` against
`_seam_extent_of(req)` -- ONE SCALAR, a row COUNT -- and then hands
`req_to_token[req_pool_idx, : seqlen - 1]` to the pool as destination ROWS
without ever asking whether those row ids address this pool. `load_cpu_copy`
writes `self.k_buffer[layer_id][chunk_indices] = k_chunk` with no bound of its
own; its only assert (:3298) compares the saved host chunk's shape against the
chunk LENGTH, a different axis entirely, which that commit already names as
"per-chunk, on the wrong axis" and "no backstop below". Equal length is not
equal validity.

TWO FAILURE MODES, AND THE QUIET ONE IS WORSE:

  * an index >= the buffer row count faults. On CPU that is an IndexError; on
    CUDA it is an ASYNCHRONOUS illegal memory access surfaced by whatever call
    synchronizes next -- which is why the W40b traceback points at
    `synchronize()` and not at the store that caused it.
  * a NEGATIVE index does not fault at all. `-1` is the classic stale-row
    sentinel and it is also valid torch indexing: it silently writes the LAST
    row of the pool. No crash, no log, wrong KV under a prefix the tree reports
    as restored. That is a wrong ANSWER, and today nothing anywhere refuses it.

Both directions carry the shape: `get_cpu_copy` READS the same unvalidated rows
into the host copy, so a stale row id there poisons the saved copy instead of
the live pool.

WHAT THIS TEST ASSERTS: the pool refuses an out-of-range or negative destination
row BEFORE any buffer is touched, by name, naming the offending value and the
bound. A refusal costs a recompute; the two alternatives cost a dead scheduler
or a wrong answer.

Hermetic: real pool methods bound to a stub with CPU buffers, no CUDA.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import MambaPool, MHATokenToKVPool
from sglang.test.test_utils import CustomTestCase

ROWS = 8
DIM = 2


class _StubPool:
    """The REAL `get_cpu_copy` / `load_cpu_copy`, bound to CPU buffers.

    Stubbing the pool's data and running the tree's own methods is the point:
    a test that reimplements the copy would encode the defect's own assumption,
    which is the failure mode 25e7849844 called out in its three corrected mocks.
    """

    use_hnd = False
    layer_num = 1
    cpu_offloading_chunk_size = 4

    def __init__(self):
        self.size = ROWS
        self.k_buffer = [torch.zeros((ROWS, DIM))]
        self.v_buffer = [torch.zeros((ROWS, DIM))]
        self.get_cpu_copy = types.MethodType(MHATokenToKVPool.get_cpu_copy, self)
        self.load_cpu_copy = types.MethodType(MHATokenToKVPool.load_cpu_copy, self)

    def _committed_row_bound(self):
        """#913: an eager stub pool has no VMM arena, so it cannot state a
        backing -- None, never 0. Production pools answer this from
        ``KVCache._committed_row_bound`` or the ``MHATokenToKVPool`` override;
        modelling it here keeps the double from diverging from the real pool,
        which is the W29 failure ("the suite's own double had the attribute and
        not the method, exactly backwards from production").
        """
        return None


SLOTS = 6
LAYERS = 2


class _StubMambaPool:
    """The REAL `MambaPool` copy methods over CPU buffers.

    THE AXIS IS THE POINT. `conv` and `temporal` are
    ``[num_layers, num_slots, ...]`` and the methods index ``[:, indices]`` --
    so the bound is dim **1**, not dim 0 as in the KV pools. A guard that read
    ``k_buffer[0].shape[0]`` would be perfectly inert here, which is why the
    real one takes its bound as an argument.
    """

    def __init__(self):
        self.mamba_cache = types.SimpleNamespace(
            conv=[torch.zeros((LAYERS, SLOTS, 3))],
            temporal=torch.zeros((LAYERS, SLOTS, 3)),
        )
        self.get_cpu_copy = types.MethodType(MambaPool.get_cpu_copy, self)
        self.load_cpu_copy = types.MethodType(MambaPool.load_cpu_copy, self)


def _pool():
    return _StubPool()


def _saved_for(pool, indices):
    """A host copy taken from VALID rows, so only the destination is under test."""
    with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
        return pool.get_cpu_copy(indices)


class TestTheStubReproducesTheRealFailure(CustomTestCase):
    """CONTROL. Passes WITHOUT the fix -- without it the reds below say nothing
    about the real pool."""

    def test_an_out_of_range_row_faults_in_the_unguarded_store(self):
        pool = _pool()
        good = torch.tensor([0, 1], dtype=torch.int64)
        saved = _saved_for(pool, good)
        bad = torch.tensor([0, ROWS + 3], dtype=torch.int64)
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            with self.assertRaises(IndexError):
                # On CUDA this same store is the W40b asynchronous IMA.
                pool.k_buffer[0][bad] = torch.zeros((2, DIM))
        del saved

    def test_a_negative_row_does_NOT_fault_and_hits_the_last_row(self):
        """The quiet mode, pinned as a fact about torch: -1 is a valid write."""
        pool = _pool()
        pool.k_buffer[0][torch.tensor([-1], dtype=torch.int64)] = torch.full(
            (1, DIM), 7.0
        )
        self.assertEqual(float(pool.k_buffer[0][ROWS - 1][0]), 7.0)


class TestLoadRefusesRowsItCannotAddress(CustomTestCase):
    """RED until the pool range-checks its destination rows."""

    def test_out_of_range_destination_is_refused_by_name(self):
        pool = _pool()
        good = torch.tensor([0, 1], dtype=torch.int64)
        saved = _saved_for(pool, good)
        bad = torch.tensor([0, ROWS + 3], dtype=torch.int64)
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            with self.assertRaises(ValueError) as ctx:
                pool.load_cpu_copy(saved, bad)
        msg = str(ctx.exception)
        self.assertIn(str(ROWS + 3), msg, "the refusal must name the offending row")
        self.assertIn(str(ROWS), msg, "and the bound it violated")

    def test_negative_destination_is_refused_rather_than_silently_wrapped(self):
        pool = _pool()
        good = torch.tensor([0, 1], dtype=torch.int64)
        saved = _saved_for(pool, good)
        stale = torch.tensor([0, -1], dtype=torch.int64)
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            with self.assertRaises(ValueError):
                pool.load_cpu_copy(saved, stale)
        self.assertEqual(
            float(pool.k_buffer[0][ROWS - 1][0]),
            0.0,
            "a -1 row must never reach the store: it writes the LAST row and "
            "returns a wrong answer with no crash and no log",
        )

    def test_refusal_happens_before_any_buffer_is_written(self):
        """No partial application. A copy that refuses must leave the pool
        byte-identical, or the refusal is just a louder corruption."""
        pool = _pool()
        good = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        saved = _saved_for(pool, good)
        pool.k_buffer[0].fill_(3.0)
        pool.v_buffer[0].fill_(4.0)
        before_k = pool.k_buffer[0].clone()
        before_v = pool.v_buffer[0].clone()
        # The bad row sits in the SECOND chunk (chunk_size 4), so an unguarded
        # implementation writes chunk 1 before it ever reaches the fault.
        bad = torch.tensor([0, 1, 2, 3, 4, ROWS + 1], dtype=torch.int64)
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            with self.assertRaises(ValueError):
                pool.load_cpu_copy(saved, bad)
        self.assertTrue(torch.equal(pool.k_buffer[0], before_k))
        self.assertTrue(torch.equal(pool.v_buffer[0], before_v))

    def test_valid_rows_still_load(self):
        """CONTROL: the guard must not over-refuse."""
        pool = _pool()
        idx = torch.tensor([0, 1, 2], dtype=torch.int64)
        pool.k_buffer[0][idx] = torch.full((3, DIM), 5.0)
        pool.v_buffer[0][idx] = torch.full((3, DIM), 6.0)
        saved = _saved_for(pool, idx)
        pool.k_buffer[0].zero_()
        pool.v_buffer[0].zero_()
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            pool.load_cpu_copy(saved, idx)
        self.assertEqual(float(pool.k_buffer[0][1][0]), 5.0)
        self.assertEqual(float(pool.v_buffer[0][2][0]), 6.0)

    def test_an_empty_index_set_is_not_an_error(self):
        """CONTROL: zero rows is a legitimate no-op, not a violation."""
        pool = _pool()
        empty = torch.empty((0,), dtype=torch.int64)
        saved = _saved_for(pool, empty)
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            pool.load_cpu_copy(saved, empty)


class TestOffloadRefusesTheSameRows(CustomTestCase):
    """THE MIRROR, swept rather than spot-fixed. `get_cpu_copy` READS the same
    unvalidated rows; a stale id there poisons the SAVED copy, so the corruption
    is carried forward to a restore that will look perfectly consistent."""

    def test_out_of_range_source_is_refused_by_name(self):
        pool = _pool()
        bad = torch.tensor([0, ROWS + 2], dtype=torch.int64)
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            with self.assertRaises(ValueError) as ctx:
                pool.get_cpu_copy(bad)
        self.assertIn(str(ROWS + 2), str(ctx.exception))

    def test_negative_source_is_refused(self):
        pool = _pool()
        stale = torch.tensor([-1], dtype=torch.int64)
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            with self.assertRaises(ValueError):
                pool.get_cpu_copy(stale)

    def test_valid_source_still_copies(self):
        pool = _pool()
        idx = torch.tensor([0, 1], dtype=torch.int64)
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            out = pool.get_cpu_copy(idx)
        self.assertEqual(len(out), 1)


class TestMambaSlotsAreBoundedOnTheirOwnAxis(CustomTestCase):
    """`MambaPool` :1180/:1192 -- same class, DIM 1, and not optional on this
    rig: `HybridLinearKVPool.load_cpu_copy` (:4399) forwards here through
    `_mamba_translate`, which is a second place an id can go stale."""

    def _saved(self, pool, indices):
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            return pool.get_cpu_copy(indices)

    def test_the_stub_reproduces_the_unguarded_slot_store(self):
        """CONTROL, passes WITHOUT the fix."""
        pool = _StubMambaPool()
        with self.assertRaises(IndexError):
            pool.mamba_cache.temporal[:, torch.tensor([SLOTS + 1])] = 1.0

    def test_a_negative_slot_does_NOT_fault_and_hits_the_last_slot(self):
        """CONTROL: the silent half, on the mamba axis. Passes today."""
        pool = _StubMambaPool()
        pool.mamba_cache.temporal[:, torch.tensor([-1], dtype=torch.int64)] = 9.0
        self.assertEqual(float(pool.mamba_cache.temporal[0, SLOTS - 1, 0]), 9.0)

    def test_out_of_range_slot_is_refused_by_name(self):
        pool = _StubMambaPool()
        saved = self._saved(pool, torch.tensor([0, 1], dtype=torch.int64))
        bad = torch.tensor([0, SLOTS + 2], dtype=torch.int64)
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            with self.assertRaises(ValueError) as ctx:
                pool.load_cpu_copy(saved, bad)
        msg = str(ctx.exception)
        self.assertIn(str(SLOTS + 2), msg)
        self.assertIn("slot", msg, "the refusal must name the axis it bounded")

    def test_negative_slot_is_refused_rather_than_silently_wrapped(self):
        pool = _StubMambaPool()
        saved = self._saved(pool, torch.tensor([0, 1], dtype=torch.int64))
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            with self.assertRaises(ValueError):
                pool.load_cpu_copy(saved, torch.tensor([0, -1], dtype=torch.int64))
        self.assertEqual(float(pool.mamba_cache.temporal[0, SLOTS - 1, 0]), 0.0)

    def test_refusal_leaves_the_mamba_state_byte_identical(self):
        pool = _StubMambaPool()
        saved = self._saved(pool, torch.tensor([0, 1], dtype=torch.int64))
        pool.mamba_cache.temporal.fill_(2.0)
        pool.mamba_cache.conv[0].fill_(3.0)
        before_t = pool.mamba_cache.temporal.clone()
        before_c = pool.mamba_cache.conv[0].clone()
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            with self.assertRaises(ValueError):
                pool.load_cpu_copy(
                    saved, torch.tensor([0, SLOTS], dtype=torch.int64)
                )
        self.assertTrue(torch.equal(pool.mamba_cache.temporal, before_t))
        self.assertTrue(torch.equal(pool.mamba_cache.conv[0], before_c))

    def test_offload_refuses_the_same_slots(self):
        pool = _StubMambaPool()
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            with self.assertRaises(ValueError):
                pool.get_cpu_copy(torch.tensor([SLOTS], dtype=torch.int64))
            with self.assertRaises(ValueError):
                pool.get_cpu_copy(torch.tensor([-1], dtype=torch.int64))

    def test_valid_slots_still_round_trip(self):
        """CONTROL: the guard must not over-refuse."""
        pool = _StubMambaPool()
        idx = torch.tensor([0, 2], dtype=torch.int64)
        pool.mamba_cache.temporal[:, idx] = 4.0
        saved = self._saved(pool, idx)
        pool.mamba_cache.temporal.zero_()
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            pool.load_cpu_copy(saved, idx)
        self.assertEqual(float(pool.mamba_cache.temporal[0, 2, 0]), 4.0)


if __name__ == "__main__":
    unittest.main()
