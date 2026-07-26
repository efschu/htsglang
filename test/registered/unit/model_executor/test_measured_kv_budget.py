"""Unit tests for the measured KV-budget path (#188) -- CPU only, no GPU.

SGLANG_MEASURED_KV_BUDGET=1 sizes this boot's KV pool from the PREVIOUS
boot's measured leftover. Two identical commands therefore produced
max_total_num_tokens 380289 and 447173 (measured, R3 validation window),
and nothing in the boot log said so. Three separable defects:

1. The leftover was read with a bare mem_get_info, WITHOUT releasing this
   process's own allocator cache first -- so the persisted correction
   carried run-to-run allocator/JIT slack instead of a stable quantity.
2. A FOREIGN consumer on the same card (a leftover server from the previous
   boot on a shared box) silently shrinks the same reading, and nothing
   distinguishes that from a genuinely full card.
3. The cross-boot dependency itself was silent: a cold budget record logs
   nothing at all, a warm one logs INFO only when the correction is nonzero.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.environ import envs
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

GIB = 1 << 30
MIB = 1 << 20


def _mem_get_info_seq(*readings):
    """Fake torch.cuda.mem_get_info returning one reading per call (the
    last one repeats)."""
    calls = {"n": 0}

    def _fn(_gpu_id=0):
        i = min(calls["n"], len(readings) - 1)
        calls["n"] += 1
        return readings[i]

    return _fn


class TestLeftoverMeasuredAfterOwnCleanup(CustomTestCase):
    """Defect 1: the measurement must not depend on how much the caching
    allocator happened to be holding at the post-capture point."""

    TOTAL = 32 * GIB
    RESIDENT = 20 * GIB  # weights + pools + graphs, physically resident

    def _measure(self, cached_b):
        """One boot whose allocator holds ``cached_b`` reserved-but-free."""
        released = {"n": 0}

        def _empty_cache():
            released["n"] += 1

        free_before = self.TOTAL - self.RESIDENT - cached_b
        free_after = self.TOTAL - self.RESIDENT  # cache handed back
        fn = _mem_get_info_seq((free_before, self.TOTAL), (free_after, self.TOTAL))
        free_b, total_b, released_b = (
            ModelRunnerKVCacheMixin.measure_free_after_own_cleanup(
                mem_get_info=fn, empty_cache=_empty_cache, gpu_id=0
            )
        )
        self.assertEqual(released["n"], 1, "own cleanup must run first")
        self.assertEqual(total_b, self.TOTAL)
        return free_b, released_b

    def test_fresh_and_fragmented_boots_measure_the_same_leftover(self):
        """The reproducibility gate. Same resident set, different allocator
        cache -> the SAME measured leftover, hence the same persisted
        correction and the same max_total_num_tokens next boot."""
        fresh, released_fresh = self._measure(cached_b=0)
        fragmented, released_frag = self._measure(cached_b=3 * GIB)
        self.assertEqual(fresh, fragmented)
        self.assertEqual(released_fresh, 0)
        self.assertEqual(released_frag, 3 * GIB)

    def test_the_old_single_read_would_have_diverged(self):
        """Pins WHY the cleanup is load-bearing: the pre-cleanup reading --
        what the old code persisted -- differs by the cache size."""
        stale_fresh = self.TOTAL - self.RESIDENT
        stale_fragmented = self.TOTAL - self.RESIDENT - 3 * GIB
        self.assertNotEqual(stale_fresh, stale_fragmented)


class TestForeignResidueIsLoud(CustomTestCase):
    """Defect 2: a co-resident foreign process eats into the same reading.
    After our own cleanup, every used byte on this rank's share must be
    attributable to this rank; a large unexplained remainder is reported
    LOUDLY, not folded silently into a smaller correction."""

    TOTAL = 32 * GIB

    def _unaccounted(self, free_b, reserved_b, ranks_on_gpu=1):
        """free_b / total are DEVICE-wide, as the driver reports them."""
        return ModelRunnerKVCacheMixin.unaccounted_used_bytes(
            total_b=self.TOTAL,
            free_b=free_b,
            ranks_on_gpu=ranks_on_gpu,
            reserved_b=reserved_b,
            ctx_allowance_b=1024 * MIB,
        )

    def test_clean_card_reports_nothing(self):
        # 20 GiB reserved by us + ~0.6 GiB CUDA context, 11.4 GiB free.
        reserved = 20 * GIB
        free = self.TOTAL - reserved - 600 * MIB
        self.assertEqual(self._unaccounted(free, reserved), 0)

    def test_leftover_process_is_flagged(self):
        # Same boot, but a 6 GiB zombie from the previous boot is resident.
        reserved = 20 * GIB
        free = self.TOTAL - reserved - 600 * MIB - 6 * GIB
        self.assertGreater(self._unaccounted(free, reserved), 5 * GIB)

    def test_colocated_ranks_only_own_their_share(self):
        # Two ranks on one card, 10 GiB reserved each: the SIBLING's 10 GiB
        # is not foreign, so nothing is reported.
        reserved = 10 * GIB
        free = self.TOTAL - 2 * reserved - 2 * 600 * MIB
        self.assertEqual(self._unaccounted(free, reserved, ranks_on_gpu=2), 0)

    def test_colocated_pair_still_sees_a_third_party(self):
        # Same pair, plus a 6 GiB zombie: reported despite co-location.
        reserved = 10 * GIB
        free = self.TOTAL - 2 * reserved - 2 * 600 * MIB - 6 * GIB
        self.assertGreater(self._unaccounted(free, reserved, ranks_on_gpu=2), 2 * GIB)

    def test_warning_text_names_the_bytes_and_the_cause(self):
        msg = ModelRunnerKVCacheMixin.foreign_residue_warning(
            tp_rank=1, unaccounted_b=6 * GIB, free_b=5 * GIB
        )
        self.assertIsNotNone(msg)
        self.assertIn("6.00", msg)
        self.assertIn("rank 1", msg)
        self.assertIsNone(
            ModelRunnerKVCacheMixin.foreign_residue_warning(
                tp_rank=1, unaccounted_b=0, free_b=5 * GIB
            )
        )


class _FakeRunner(ModelRunnerKVCacheMixin):
    """Minimal stand-in exposing exactly what the correction reader touches."""

    def __init__(self, path, tp_rank=0, tp_size=2):
        self._path = path
        self.tp_rank = tp_rank

        class _SA:
            pass

        self.server_args = _SA()
        self.server_args.tp_size = tp_size
        self.server_args.rank_tp_ratio = None
        self.server_args.rank_mlp_ratio = None

    def _measured_kv_budget_cache_path(self):
        return self._path


class TestCrossBootProvenanceIsAnnounced(CustomTestCase):
    """Defect 3: the capacity depends on an on-disk record from a previous
    boot. That must be stated on every boot, cold or warm -- otherwise a
    harness comparing two trees compares itself."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "budget.json")
        self._env = envs.SGLANG_MEASURED_KV_BUDGET.override(True)
        self._env.__enter__()

    def tearDown(self):
        self._env.__exit__(None, None, None)
        self._tmp.cleanup()

    def _write(self, correction_bytes, mlp_vector, safety_mib=400):
        with open(self.path, "w") as f:
            json.dump(
                {
                    "correction_bytes": correction_bytes,
                    "mlp_vector": mlp_vector,
                    "components": [
                        {"safety_mib": safety_mib} for _ in correction_bytes
                    ],
                    "ts": "2026-07-26 00:00:00",
                },
                f,
            )

    def test_cold_record_is_reported_as_cold(self):
        r = _FakeRunner(self.path)
        self.assertEqual(r._measured_kv_budget_correction_bytes(), 0)
        self.assertEqual(r._measured_kv_budget_provenance, "cold")

    def test_stored_record_is_reported_with_its_timestamp(self):
        self._write([7 * GIB, 5 * GIB], [1, 1])
        r = _FakeRunner(self.path)
        self.assertEqual(r._measured_kv_budget_correction_bytes(), 7 * GIB)
        self.assertEqual(r._measured_kv_budget_provenance, "stored")
        self.assertEqual(r._measured_kv_budget_ts, "2026-07-26 00:00:00")

    def test_vector_mismatch_is_reported_as_reset_not_as_cold(self):
        self._write([7 * GIB, 5 * GIB], [6, 1])
        r = _FakeRunner(self.path)
        self.assertEqual(r._measured_kv_budget_correction_bytes(), 0)
        self.assertEqual(r._measured_kv_budget_provenance, "vector-reset")

    def test_safety_mismatch_is_reported_as_reset(self):
        self._write([7 * GIB, 5 * GIB], [1, 1], safety_mib=1350)
        r = _FakeRunner(self.path)
        self.assertEqual(r._measured_kv_budget_correction_bytes(), 0)
        self.assertEqual(r._measured_kv_budget_provenance, "safety-reset")

    def test_cold_notice_warns_that_the_next_boot_will_differ(self):
        note = ModelRunnerKVCacheMixin.measured_budget_provenance_note(
            provenance="cold", path="/tmp/x.json", ts=None, correction_b=0
        )
        self.assertIn("differently", note)
        self.assertIn("/tmp/x.json", note)
        # The deterministic alternative must be named, not implied.
        self.assertIn("--max-total-tokens", note)

    def test_stored_notice_names_the_source_boot(self):
        note = ModelRunnerKVCacheMixin.measured_budget_provenance_note(
            provenance="stored",
            path="/tmp/x.json",
            ts="2026-07-26 00:00:00",
            correction_b=7 * GIB,
        )
        self.assertIn("2026-07-26 00:00:00", note)
        self.assertIn("7.00", note)
        self.assertIn("--max-total-tokens", note)


if __name__ == "__main__":
    unittest.main()
