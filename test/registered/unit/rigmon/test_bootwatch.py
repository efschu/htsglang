"""CPU unit tests for boot classification.

Log excerpts are shaped after real boot logs from this rig.
"""

import os
import tempfile
import unittest

from sglang.srt.rigmon.bootwatch import (
    SIGNATURES,
    STAGES,
    classify_boot,
    read_boot_log,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


HEAD = [
    "[2026-07-26 14:40:01] Launch server",
    "[2026-07-26 14:40:12] Load weight begin. avail mem=31.34 GB",
]
WEIGHTS_DONE = ["[2026-07-26 14:40:58] Load weight end. mem usage=18.4 GB"]
KV_DONE = ["[2026-07-26 14:41:02] KV Cache is allocated. #tokens: 98328"]
READY = ["[2026-07-26 14:41:44] The server is fired up and ready to roll!"]

OOM = [
    "  File \"python/sglang/srt/mem_cache/memory_pool.py\", line 1904",
    "    torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)",
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 514.00 MiB. "
    "GPU 0 has a total capacity of 31.34 GiB of which 63.19 MiB is free.",
    "[2026-07-26 14:41:12] Received sigquit from a child process.",
]
ARCH = [
    "  File \"flashinfer/norm/kernels/rmsnorm.py\", line 1323, in rmsnorm_cute",
    "RuntimeError: CUDA Error: cudaErrorNoKernelImageForDevice",
]
NCCL = ["RuntimeError: NCCL error: invalid usage (run with NCCL_DEBUG=WARN)"]
MAMBA = [
    "RuntimeError: Hybrid (mamba/linear-attention) state cache is too small "
    "to serve any requests. max_mamba_cache_size=0",
]
BENIGN_MAMBA = ["[2026-07-26 14:40:30] max_mamba_cache_size=512, page_size=64"]
GEOMETRY = ["AssertionError: 32 is not divisible by 3"]


class TestStages(CustomTestCase):
    def test_progress_is_tracked(self):
        d = classify_boot(HEAD + WEIGHTS_DONE, HEAD + WEIGHTS_DONE)
        self.assertEqual(d.stage, "weights_done")
        self.assertIn("weights_begin", d.reached)
        self.assertFalse(d.ready)

    def test_ready_boot_is_clean(self):
        lines = HEAD + WEIGHTS_DONE + KV_DONE + READY
        d = classify_boot(lines, lines)
        self.assertTrue(d.ready)
        self.assertFalse(d.failed)
        self.assertIsNone(d.diagnosis)

    def test_stages_are_ordered(self):
        self.assertEqual([s.order for s in STAGES], sorted(s.order for s in STAGES))


class TestClassification(CustomTestCase):
    def test_oom_after_weights_is_a_planning_failure(self):
        lines = HEAD + WEIGHTS_DONE + OOM
        d = classify_boot(lines, lines)
        self.assertTrue(d.failed)
        self.assertEqual(d.diagnosis["key"], "oom_pools")
        self.assertIn("planning failure", d.diagnosis["meaning"])

    def test_oom_before_weights_is_a_capacity_failure(self):
        """Same exception, different stage, different meaning."""
        lines = HEAD + OOM
        d = classify_boot(lines, lines)
        self.assertEqual(d.diagnosis["key"], "oom_weights")
        self.assertIn("does not fit", d.diagnosis["meaning"])

    def test_arch_mismatch(self):
        lines = HEAD + WEIGHTS_DONE + ARCH
        d = classify_boot(lines, lines)
        self.assertEqual(d.diagnosis["key"], "arch_mismatch")
        self.assertIn("arch", d.diagnosis["remedy"].lower())

    def test_specific_signature_beats_the_generic_child_death(self):
        lines = HEAD + WEIGHTS_DONE + OOM  # OOM block ends with the sigquit line
        d = classify_boot(lines, lines)
        self.assertEqual(d.diagnosis["key"], "oom_pools")
        self.assertIn("child_died", [m["key"] for m in d.other_matches])

    def test_child_death_alone_is_reported_as_a_symptom(self):
        lines = HEAD + ["[..] Rank 0 scheduler died during initialization (exit code: 1)"]
        d = classify_boot(lines, lines)
        self.assertEqual(d.diagnosis["key"], "child_died")
        self.assertEqual(d.diagnosis["severity"], "warning")
        self.assertIn("symptom", d.diagnosis["meaning"])

    def test_mamba_pool_error_is_matched(self):
        lines = HEAD + WEIGHTS_DONE + MAMBA
        d = classify_boot(lines, lines)
        self.assertEqual(d.diagnosis["key"], "mamba_pool_too_small")

    def test_benign_config_line_does_not_trigger_the_mamba_signature(self):
        """An earlier draft matched `max_mamba_`, which fired on 96 of 116 real
        logs including successful boots."""
        lines = HEAD + BENIGN_MAMBA + WEIGHTS_DONE + KV_DONE + READY
        d = classify_boot(lines, lines)
        self.assertIsNone(d.diagnosis)
        lines2 = HEAD + BENIGN_MAMBA + WEIGHTS_DONE
        d2 = classify_boot(lines2, lines2)
        self.assertIsNone(d2.diagnosis)

    def test_geometry_mismatch_points_at_uneven_tp(self):
        lines = HEAD + GEOMETRY
        d = classify_boot(lines, lines)
        self.assertEqual(d.diagnosis["key"], "geometry_mismatch")
        self.assertIn("uneven", d.diagnosis["remedy"])

    def test_nccl_error_names_the_rank_local_cause(self):
        lines = HEAD + WEIGHTS_DONE + NCCL
        d = classify_boot(lines, lines)
        self.assertEqual(d.diagnosis["key"], "nccl_error")
        self.assertIn("FIRST rank", d.diagnosis["remedy"])

    def test_every_signature_has_meaning_and_remedy(self):
        for s in SIGNATURES:
            self.assertTrue(s.meaning, s.key)
            self.assertTrue(s.remedy, s.key)

    def test_excerpt_is_bounded(self):
        lines = HEAD + WEIGHTS_DONE + OOM * 50
        d = classify_boot(lines, lines)
        self.assertLessEqual(len(d.excerpt), 12)


class TestReading(CustomTestCase):
    def test_large_log_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "boot.log")
            with open(path, "w") as f:
                f.write("\n".join(HEAD + WEIGHTS_DONE) + "\n")
                f.write(("filler line to grow the log\n") * 60000)
                f.write("\n".join(OOM) + "\n")
            markers, tail = read_boot_log(path, max_bytes=64 * 1024, tail_bytes=8 * 1024)
            d = classify_boot(markers, tail)
        self.assertEqual(d.stage, "weights_done")
        self.assertEqual(d.diagnosis["key"], "oom_pools")

    def test_markers_found_in_the_tail_when_head_is_truncated(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "boot.log")
            with open(path, "w") as f:
                f.write(("noise\n") * 20000)
                f.write("\n".join(WEIGHTS_DONE + KV_DONE + READY) + "\n")
            markers, tail = read_boot_log(path, max_bytes=4096, tail_bytes=8192)
            d = classify_boot(markers, tail)
        self.assertTrue(d.ready)


if __name__ == "__main__":
    unittest.main()
