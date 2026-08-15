"""A partial checkpoint must not be sized as if it were the whole model.

``_checkpoint_size_mib`` sums every ``*.safetensors`` present in a directory.
Nothing checks that those files ARE the checkpoint. The planner then anchors
every quantized family's bytes/param on that number, so a directory that is
still downloading is planned as a smaller model -- silently, with a confident
answer and no warning.

Measured 2026-08-14, planning Qwen3.8-27B-INT8 mid-download with 4 of 18
shards on disk: the planner reported the new model's weights as SMALLER than
the Qwen3.6-27B-INT8-W8A8 it replaces (25.1 GiB summed over ranks against
29.2), when the completed checkpoint measures 33.86 GiB -- 4.76 GiB LARGER.
Every downstream number inherited the error: per-rank budgets, "FITS: yes",
and a KV capacity that cannot exist. The 3.6 figure was right at the same
moment, because that checkpoint was complete, which is what made the
comparison look credible rather than broken.

This is the silent-falsity shape: not a crash, not a missing number, but a
plausible number computed from an input that was never checked. The index is
the checkpoint's own statement of what it consists of, so it is the thing to
check against.

Contract pinned here:

* complete (every shard the index names is present) -> measured on-disk size,
  unchanged from before;
* incomplete -> the index's declared ``total_size`` instead, and a WARNING
  naming the missing shards;
* incomplete with no declared total -> 0, i.e. "unknown", which the caller
  already treats as "fall back to the config-derived estimate";
* no index at all (single-file checkpoint) -> measured, as before: there is
  no manifest to be short of.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import json
import os
import tempfile
import unittest

from sglang.srt.distributed.utils import _checkpoint_size_mib
from sglang.test.test_utils import CustomTestCase

SHARD_BYTES = 4 * 2**20  # 4 MiB per stand-in shard
N_SHARDS = 18
DECLARED_TOTAL = N_SHARDS * SHARD_BYTES


def _shard(i):
    return f"model-{i:05d}-of-{N_SHARDS:05d}.safetensors"


def _write(directory, present, declared_total=DECLARED_TOTAL, with_index=True):
    if with_index:
        weight_map = {f"t{i}": _shard(i) for i in range(1, N_SHARDS + 1)}
        index = {"weight_map": weight_map}
        if declared_total is not None:
            index["metadata"] = {"total_size": declared_total}
        with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
    for i in present:
        with open(os.path.join(directory, _shard(i)), "wb") as f:
            f.write(b"\0" * SHARD_BYTES)
    return directory


class TestPartialCheckpointIsNotSizedAsWhole(CustomTestCase):
    def test_the_complete_checkpoint_is_measured(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, range(1, N_SHARDS + 1))
            self.assertEqual(_checkpoint_size_mib(d), DECLARED_TOTAL // 2**20)

    def test_a_half_download_does_not_report_partial_bytes(self):
        """The exact scenario that mis-planned Qwen3.8: 4 of 18 shards."""
        with tempfile.TemporaryDirectory() as d:
            _write(d, range(1, 5))
            got = _checkpoint_size_mib(d)
            self.assertNotEqual(
                got,
                4 * SHARD_BYTES // 2**20,
                "sized from the 4 shards that happen to be on disk; the "
                "planner then plans a model that does not exist",
            )
            self.assertEqual(got, DECLARED_TOTAL // 2**20)

    def test_a_single_missing_shard_is_enough(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, range(1, N_SHARDS))  # 17 of 18
            self.assertEqual(_checkpoint_size_mib(d), DECLARED_TOTAL // 2**20)

    def test_the_missing_shards_are_named_in_a_warning(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, range(1, 5))
            with self.assertLogs(
                "sglang.srt.distributed.utils", level="WARNING"
            ) as cm:
                _checkpoint_size_mib(d)
            blob = "\n".join(cm.output)
            self.assertIn(_shard(5), blob)
            self.assertIn("14", blob)  # the count of missing shards

    def test_incomplete_without_a_declared_total_is_unknown(self):
        """0 means "unknown"; callers already fall back to the config model."""
        with tempfile.TemporaryDirectory() as d:
            _write(d, range(1, 5), declared_total=None)
            self.assertEqual(_checkpoint_size_mib(d), 0)

    def test_a_checkpoint_without_an_index_is_still_measured(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "model.safetensors"), "wb") as f:
                f.write(b"\0" * SHARD_BYTES)
            self.assertEqual(_checkpoint_size_mib(d), SHARD_BYTES // 2**20)


if __name__ == "__main__":
    unittest.main()
