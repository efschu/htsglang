"""Task #260: the reserve-to-budget mapping must survive a co-resident process.

THE DEFECT
----------
``--rank-gpu-memory-mib`` is documented as an ABSOLUTE per-rank allowance, and
``--rank-auto-reserve-mib`` as ``NVML total - reserve``. The derivation in
``_resolve_auto_rank_tp_ratio`` silently capped that at ``free_mib - 1024``,
i.e. at the memory that happened to be free at parse time. With a process
sharing the card that reading is already reduced by the neighbour, and the
operator's reserve was sized to cover the same neighbour -- so it was charged
twice. Worse, the capped budgets feed the gcd reduction that derives
``--rank-tp-ratio``, so the SHARD PLAN of the model depended on who else was
on the card at launch.

The measured trigger (2026-07-27, 5090 32607 MiB, a PD instance holding
22227 MiB, TP=3 with ``--rank-auto-reserve-mib 25800,7000,7000``): the boot
was rejected with "the rank already uses 4.32 GiB for weights and runtime
state, which exhausts the per-rank budget", which reads as if 4.32 GiB had
eaten a 6.65 GiB budget. It had not. The other 2.50 GiB went to the mamba
state pool, the speculative intermediate state and the prefill activation
reserve, and the 0.73 GiB GGUF scratch tipped it over -- none of them named.
The misattribution is what produced the co-residence double-count theory.

WHAT IS PINNED HERE
-------------------
 1. The budget is ``total - reserve``, unchanged by a neighbour holding the
    card (server-args side, in test_uneven_tp_args.py).
 2. The KV profiler's per-rank arithmetic is byte-identical without
    co-residence -- the fix moves no bytes on the default path.
 3. When the budget cannot be realized because the DEVICE is full, the
    failure says so, names the occupancy, and states that the budget was not
    reduced by it.
 4. When the budget was handed out in full and the POSTS exhaust it, the
    failure itemizes those posts -- the measured case above, reproduced to
    the 0.01 GiB.

CPU only: ``torch.cuda.mem_get_info`` is the single stubbed primitive, and
every number below is arithmetic over it.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.model_executor import model_runner_kv_cache_mixin as M
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_GIB = 1 << 30

# The measured co-residence case, in the units the flags use.
GPU_TOTAL_MIB = 32607  # RTX 5090, NVML total
NEIGHBOUR_MIB = 22227  # co-resident PD instance
RESERVE_MIB = 25800  # --rank-auto-reserve-mib for this GPU
BUDGET_MIB = GPU_TOTAL_MIB - RESERVE_MIB  # 6807, the logged budget

# Rank 0 of that boot, from the boot log.
USED_BY_ME_GB = 4.3239  # weights + runtime state ("uses 4.32 GiB")
FREE_AFTER_WEIGHTS_GB = 4.51  # "Load weight end ... avail mem=4.51 GB"
MAMBA_POSTS_GB = 2.4975  # state pool 0.91 + spec intermediate 0.58 + 1.00
GGUF_SCRATCH_GB = 0.73  # "reserving 0.73 GiB of the KV budget"


class _FakeServerArgs:
    def __init__(self, budget_mib, rank_gpu_id):
        self.rank_gpu_memory_mib = budget_mib
        self.rank_gpu_id = list(rank_gpu_id)
        # Fixture drift, 2026-07-31: #307 (84e5d3e265) made the budget message
        # name the --max-running-requests-ceiling this budget WOULD carry, so
        # the field is read on the exhausted-budget path. This fixture pins
        # the budget directly and sets no ceiling, which is what ``None``
        # means here -- with it the message keeps exactly the shape the
        # assertions below were written against (no ceiling clause).
        self.max_running_requests_ceiling = None

    def uneven_memory_budgets_active(self):
        return self.rank_gpu_memory_mib is not None

    def derived_reserve_infeasible_note(self, rank_index, short_mib):
        """Fixture drift, 2026-08-03: #458 (b59e442a2c) appends a note when
        the budget was DERIVED by ``--rank-auto-reserve-mib auto``.

        The real method returns ``None`` for any pinned reserve, and this
        fixture hands ``rank_gpu_memory_mib`` in directly rather than deriving
        it from ``auto`` -- so ``None`` is the answer the production method
        would give for this configuration, not a suppression of the note.
        """
        return None


class _FakeRunner:
    """The collaborators ``_profile_available_bytes`` actually touches."""

    def __init__(
        self,
        budget_mib=BUDGET_MIB,
        rank_gpu_id=(0, 1, 2),
        tp_rank=0,
        mamba_posts_gb=MAMBA_POSTS_GB,
        gguf_scratch_gb=GGUF_SCRATCH_GB,
    ):
        self.server_args = _FakeServerArgs(budget_mib, rank_gpu_id)
        self.tp_rank = tp_rank
        self.gpu_id = 0
        self.device = "cuda"
        self.mambaish_config = object() if mamba_posts_gb else None
        self.post_capture_kv_active = False  # uneven DCP: not planned
        self._mamba_posts_gb = mamba_posts_gb
        self._gguf_scratch = gguf_scratch_gb
        self._mem_ckpt_post_weights = None
        self._measured_kv_budget_provenance = None
        self._measured_kv_budget_ts = None

    # --- collaborators that are out of scope for this test -------------
    def _expert_offload_lane_active(self):
        return False

    def _colocated_sibling_reserved_gb(self):
        return 0.0

    def _gguf_dequant_scratch_gb(self):
        return self._gguf_scratch

    def _measured_kv_budget_correction_bytes(self):
        return 0

    def handle_max_mamba_cache(self, rest_memory):
        return rest_memory - self._mamba_posts_gb

    # --- the code under test -------------------------------------------
    budget_physical_shortfall_gb = staticmethod(
        M.ModelRunnerKVCacheMixin.budget_physical_shortfall_gb
    )
    budget_exhausted_message = staticmethod(
        M.ModelRunnerKVCacheMixin.budget_exhausted_message
    )

    # Fixture drift, 2026-07-28: this file was written against a
    # ``_ranks_on_my_gpu`` that indexed the per-rank vectors with ``tp_rank``
    # directly. [PP] #201 slice 1 (358522e207) put ``_rank_vector_index``
    # between the two, because under a pipeline the vectors are laid out in
    # WORLD-rank order (pp_rank * tp_size + tp_rank) and every stage would
    # otherwise read stage 0's entry. The fake wires the mixin methods one by
    # one, so it has to carry the new helper as well; its ``world_rank``
    # lookup on ``server_args`` is a getattr with a ``tp_rank`` fallback, so
    # _FakeServerArgs needs no new field.
    def _rank_vector_index(self):
        return M.ModelRunnerKVCacheMixin._rank_vector_index(self)

    def _ranks_on_my_gpu(self):
        return M.ModelRunnerKVCacheMixin._ranks_on_my_gpu(self)

    def _device_occupancy_gb(self, device_free_gb):
        return M.ModelRunnerKVCacheMixin._device_occupancy_gb(self, device_free_gb)

    def _assert_budget_physically_available(self, *a):
        return M.ModelRunnerKVCacheMixin._assert_budget_physically_available(self, *a)

    def profile(self, pre_model_load_memory):
        return M.ModelRunnerKVCacheMixin._profile_available_bytes(
            self, pre_model_load_memory
        )


class _StubbedDevice:
    """``mem_get_info`` is the only device primitive the tests provide.

    ``get_available_gpu_memory`` is re-implemented on top of it exactly as
    the real one does (driver free bytes -> GiB), so the profiler under test
    reads its free memory through the stub and nothing else.
    """

    def __init__(self, free_gb, total_gb, own_reserved_gb=0.0):
        self.free_b = int(free_gb * _GIB)
        self.total_b = int(total_gb * _GIB)
        self.own_reserved_b = int(own_reserved_gb * _GIB)

    def __enter__(self):
        self._patches = [
            mock.patch.object(
                torch.cuda, "mem_get_info", lambda *a, **k: (self.free_b, self.total_b)
            ),
            mock.patch.object(
                torch.cuda, "memory_reserved", lambda *a, **k: self.own_reserved_b
            ),
            mock.patch.object(torch.cuda, "memory_allocated", lambda *a, **k: 0),
            mock.patch.object(
                M,
                "get_available_gpu_memory",
                lambda *a, **k: torch.cuda.mem_get_info(0)[0] / _GIB,
            ),
            mock.patch.object(
                M, "get_world_group", lambda: mock.Mock(world_size=1, cpu_group=None)
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False


class BudgetIsNotReducedByTheNeighbourTest(CustomTestCase):
    """The measured case: the budget WAS handed out in full."""

    def test_measured_case_reproduces_the_boot_arithmetic(self):
        runner = _FakeRunner()
        pre = USED_BY_ME_GB + FREE_AFTER_WEIGHTS_GB
        with _StubbedDevice(
            free_gb=FREE_AFTER_WEIGHTS_GB,
            total_gb=GPU_TOTAL_MIB / 1024,
            own_reserved_gb=USED_BY_ME_GB,
        ):
            with self.assertRaises(ValueError) as ctx:
                runner.profile(pre)
        msg = str(ctx.exception)
        # The full budget arrived: 6807 MiB, not the free reading.
        self.assertIn(f"{BUDGET_MIB} MiB (6.65 GiB) budget", msg)
        # ... and it is the POSTS that exhaust it, each one named.
        self.assertIn("weights + runtime state 4.32 GiB", msg)
        self.assertIn("prefill activation reserve 2.50 GiB", msg)
        self.assertIn("GGUF dequant scratch 0.73 GiB", msg)
        self.assertIn("7.55 GiB together", msg)
        self.assertIn("926 MiB more than the budget", msg)
        # ... and the co-residence question is answered in the message
        # instead of being left to a night of arithmetic.
        self.assertIn("a co-resident process does not reduce the budget", msg)
        self.assertNotIn("exhausts the per-rank budget", msg)

    def test_the_shortfall_is_the_posts_not_the_weights(self):
        """Weights alone fit; it takes every post together to overrun."""
        self.assertLess(USED_BY_ME_GB, BUDGET_MIB / 1024)
        self.assertLess(USED_BY_ME_GB + GGUF_SCRATCH_GB, BUDGET_MIB / 1024)
        self.assertGreater(
            USED_BY_ME_GB + MAMBA_POSTS_GB + GGUF_SCRATCH_GB, BUDGET_MIB / 1024
        )

    def test_a_budget_that_covers_every_post_carries(self):
        """Same rank, reserve lowered so the posts fit: no rejection.

        7.55 GiB of posts + a KV pool -> 9000 MiB of budget carries, which is
        reserve 23607 on this card. That is the number the itemized message
        hands the operator; the old message pointed at 4428 MiB.
        """
        runner = _FakeRunner(budget_mib=9000)
        free_gb = FREE_AFTER_WEIGHTS_GB + 3.0
        pre = USED_BY_ME_GB + free_gb
        with _StubbedDevice(
            free_gb=free_gb,
            total_gb=GPU_TOTAL_MIB / 1024,
            own_reserved_gb=USED_BY_ME_GB,
        ):
            got = runner.profile(pre)
        expected_gb = 9000 / 1024 - USED_BY_ME_GB - MAMBA_POSTS_GB - GGUF_SCRATCH_GB
        self.assertAlmostEqual(got / _GIB, expected_gb, places=5)


class PhysicalAvailabilityIsSaidInItsOwnWordsTest(CustomTestCase):
    """The other half: when the DEVICE really cannot hand out the budget."""

    def test_shortfall_helper(self):
        # Reachable = what the rank holds + its share of the free bytes.
        self.assertEqual(
            M.ModelRunnerKVCacheMixin.budget_physical_shortfall_gb(10.0, 4.0, 6.0, 1),
            0.0,
        )
        self.assertAlmostEqual(
            M.ModelRunnerKVCacheMixin.budget_physical_shortfall_gb(10.0, 4.0, 4.0, 1),
            2.0,
        )
        # Co-located ranks split the free bytes.
        self.assertAlmostEqual(
            M.ModelRunnerKVCacheMixin.budget_physical_shortfall_gb(10.0, 4.0, 8.0, 2),
            2.0,
        )

    def test_message_blames_the_occupancy_not_the_weights(self):
        """Budget 12 GiB on a card whose free memory cannot cover it."""
        runner = _FakeRunner(budget_mib=12 * 1024)
        with _StubbedDevice(
            free_gb=2.0,
            total_gb=GPU_TOTAL_MIB / 1024,
            own_reserved_gb=USED_BY_ME_GB,
        ):
            with self.assertRaises(ValueError) as ctx:
                runner.profile(USED_BY_ME_GB + 2.0)
        msg = str(ctx.exception)
        self.assertIn("is not physically available", msg)
        self.assertIn("2.00 GiB of the device is free to it", msg)
        self.assertIn("held outside this process", msg)
        self.assertIn("deliberately NOT reduced by that occupancy", msg)
        # Never the misdiagnosis this task started from.
        self.assertNotIn("exhausts the per-rank budget", msg)

    def test_occupancy_is_total_minus_free_minus_this_process(self):
        runner = _FakeRunner()
        with _StubbedDevice(free_gb=4.0, total_gb=32.0, own_reserved_gb=5.0):
            total_gb, outside_gb = runner._device_occupancy_gb(4.0)
        self.assertAlmostEqual(total_gb, 32.0)
        self.assertAlmostEqual(outside_gb, 23.0)


class WithoutCoResidenceNothingMovesTest(CustomTestCase):
    """The byte-identity claim for every boot that has the card to itself."""

    def test_budget_arithmetic_is_the_documented_chain(self):
        budget_mib = 13480  # the 3080 rank of the same boot
        runner = _FakeRunner(budget_mib=budget_mib, tp_rank=1, gguf_scratch_gb=0.72)
        used_gb, free_gb = 6.427, 12.79
        with _StubbedDevice(
            free_gb=free_gb, total_gb=20480 / 1024, own_reserved_gb=6.4
        ):
            got = runner.profile(used_gb + free_gb)
        expected_gb = budget_mib / 1024 - used_gb - MAMBA_POSTS_GB - 0.72
        self.assertAlmostEqual(got / _GIB, expected_gb, places=5)
        # No free-derived term anywhere in it: doubling the free memory
        # leaves the result bit-for-bit the same.
        with _StubbedDevice(
            free_gb=free_gb * 2, total_gb=20480 / 1024, own_reserved_gb=6.4
        ):
            again = _FakeRunner(
                budget_mib=budget_mib, tp_rank=1, gguf_scratch_gb=0.72
            ).profile(used_gb + free_gb * 2)
        self.assertEqual(got, again)

    def test_non_uneven_path_untouched(self):
        """The stock --mem-fraction-static branch keeps its own arithmetic."""
        runner = _FakeRunner(budget_mib=None, mamba_posts_gb=0.0, gguf_scratch_gb=0.0)
        runner.mem_fraction_static = 0.9
        with _StubbedDevice(free_gb=8.0, total_gb=32.0):
            got = runner.profile(20.0)
        # free - pre * (1 - fraction) = 8.0 - 20.0 * 0.1
        self.assertAlmostEqual(got / _GIB, 6.0, places=5)


if __name__ == "__main__":
    unittest.main()
