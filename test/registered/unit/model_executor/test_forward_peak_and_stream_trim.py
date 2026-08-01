"""Two probes added after #417 window 3, both default-off.

`ForwardPeakTracker` turns the prefill transient -- the thing that killed
window 3 and that every fixposten since has budgeted by taste -- into a
measured per-rank number.

`ProgressCoupledTrim` makes the load-time page-cache peak managed instead of
lucky: the previous trim was an external sampler on a wall-clock interval, and
window 3 saw memory.current move 88 -> 102 GiB inside one 15 s window.

The property that matters most for both is that they are INERT unless asked
for -- they sit on the forward path and the load path respectively.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.model_executor.forward_peak import (
    ForwardPeakTracker,
    maybe_create,
    peak_scope,
)
from sglang.srt.model_loader.gguf_shards import ProgressCoupledTrim
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _EnvMixin:
    def setUp(self):
        super().setUp()
        self._saved = {}
        self.addCleanup(self._restore)

    def _set(self, key, value):
        if key not in self._saved:
            self._saved[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)

    def _restore(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestForwardPeakTracker(_EnvMixin, CustomTestCase):
    def test_off_by_default(self):
        self._set("SGLANG_FORWARD_PEAK_PATH", None)
        self.assertIsNone(maybe_create("tp0"))

    def test_on_when_asked(self):
        with tempfile.TemporaryDirectory() as d:
            self._set("SGLANG_FORWARD_PEAK_PATH", os.path.join(d, "peak"))
            self.assertIsNotNone(maybe_create("tp0"))

    def test_records_peak_per_phase_and_bucket(self):
        with tempfile.TemporaryDirectory() as d:
            t = ForwardPeakTracker(os.path.join(d, "peak"), "tp1")
            fake = mock.MagicMock()
            fake.cuda.max_memory_allocated.side_effect = [1000, 5000, 2000]
            with mock.patch.dict("sys.modules", {"torch": fake}):
                t.begin("extend", 1500)
                t.end()
                t.begin("extend", 1500)
                t.end()
                t.begin("decode", 1)
                t.end()
            self.assertEqual(len(t.rows), 2)
            extend = t.rows["extend/2048"]
            self.assertEqual(extend["calls"], 2)
            self.assertEqual(extend["peak_bytes_max"], 5000)
            self.assertEqual(extend["tokens_max"], 1500)
            self.assertEqual(t.rows["decode/1"]["peak_bytes_max"], 2000)

    def test_dump_writes_one_file_per_rank(self):
        """A file, not a log line: worker logger.warning provably does not
        reach the server log on this rig, while per-rank JSON does."""
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "peak")
            t = ForwardPeakTracker(base, "tp2")
            fake = mock.MagicMock()
            fake.cuda.max_memory_allocated.return_value = 4242
            with mock.patch.dict("sys.modules", {"torch": fake}):
                t.begin("extend", 2048)
                t.end()
            t.dump()
            with open(f"{base}.tp2.json") as fh:
                payload = json.load(fh)
            self.assertEqual(payload["rank_tag"], "tp2")
            self.assertEqual(payload["peak_bytes_overall"], 4242)
            self.assertEqual(payload["rows"][0]["phase"], "extend")

    def test_scope_closes_the_bracket_even_when_the_forward_raises(self):
        """An OOM is exactly when the peak matters, so the bracket must close
        on the exception path too."""
        with tempfile.TemporaryDirectory() as d:
            t = ForwardPeakTracker(os.path.join(d, "peak"), "tp0")
            fake = mock.MagicMock()
            fake.cuda.max_memory_allocated.return_value = 777
            with mock.patch.dict("sys.modules", {"torch": fake}):
                t.begin("extend", 100)
                with self.assertRaises(RuntimeError):
                    with peak_scope(t):
                        raise RuntimeError("simulated OOM")
            self.assertEqual(t.rows["extend/128"]["peak_bytes_max"], 777)

    def test_scope_tolerates_no_tracker(self):
        with peak_scope(None):
            pass

    def test_can_fail(self):
        """Falsifier: a tracker that never reset would report a running max,
        not a per-forward peak, and this comparison would stop discriminating."""
        with tempfile.TemporaryDirectory() as d:
            t = ForwardPeakTracker(os.path.join(d, "peak"), "tp0")
            fake = mock.MagicMock()
            fake.cuda.max_memory_allocated.side_effect = [9000, 100]
            with mock.patch.dict("sys.modules", {"torch": fake}):
                t.begin("extend", 10)
                t.end()
                t.begin("extend", 10)
                t.end()
            self.assertEqual(fake.cuda.reset_peak_memory_stats.call_count, 2)
            with self.assertRaises(AssertionError):
                self.assertEqual(t.rows["extend/16"]["peak_bytes_last"], 9000)


class TestProgressCoupledTrim(_EnvMixin, CustomTestCase):
    def test_off_by_default(self):
        self._set("SGLANG_GGUF_STREAM_TRIM_SOFT_GIB", None)
        t = ProgressCoupledTrim()
        self.assertFalse(t.enabled)
        t.maybe_trim()
        self.assertEqual(t.trims, 0)

    def test_refuses_when_swap_is_configured(self):
        """The safety argument -- reclaim cannot touch anon on a swapless box --
        is a property of the HOST, so it is checked rather than assumed."""
        self._set("SGLANG_GGUF_STREAM_TRIM_SOFT_GIB", 90)
        with mock.patch.object(
            ProgressCoupledTrim, "_swapless", staticmethod(lambda: False)
        ):
            self.assertFalse(ProgressCoupledTrim().enabled)
        with mock.patch.object(
            ProgressCoupledTrim, "_swapless", staticmethod(lambda: True)
        ):
            self.assertTrue(ProgressCoupledTrim().enabled)

    def test_quiet_below_the_watermark_and_acts_above_it(self):
        self._set("SGLANG_GGUF_STREAM_TRIM_SOFT_GIB", 90)
        self._set("SGLANG_GGUF_STREAM_TRIM_TARGET_GIB", 80)
        gib = 1 << 30
        with mock.patch.object(
            ProgressCoupledTrim, "_swapless", staticmethod(lambda: True)
        ):
            t = ProgressCoupledTrim()
            writes = []
            with mock.patch.object(
                ProgressCoupledTrim, "_current", staticmethod(lambda: 50 * gib)
            ):
                t.maybe_trim()
            self.assertEqual(t.trims, 0, "must not reclaim below the watermark")

            m = mock.mock_open()
            with mock.patch.object(
                ProgressCoupledTrim, "_current", staticmethod(lambda: 95 * gib)
            ), mock.patch("builtins.open", m):
                t.maybe_trim()
            self.assertEqual(t.trims, 1)
            writes = [c.args[0] for c in m().write.call_args_list]
            self.assertEqual(writes, [str(15 * gib)], "ask = current - target")

    def test_target_defaults_below_soft(self):
        """A target at or above the soft mark would reclaim on every call."""
        self._set("SGLANG_GGUF_STREAM_TRIM_SOFT_GIB", 90)
        self._set("SGLANG_GGUF_STREAM_TRIM_TARGET_GIB", None)
        with mock.patch.object(
            ProgressCoupledTrim, "_swapless", staticmethod(lambda: True)
        ):
            t = ProgressCoupledTrim()
        self.assertLess(t.target_bytes, t.soft_bytes)

    def test_disables_itself_when_reclaim_is_unavailable(self):
        """A probe must never be the reason a load fails."""
        self._set("SGLANG_GGUF_STREAM_TRIM_SOFT_GIB", 90)
        gib = 1 << 30
        with mock.patch.object(
            ProgressCoupledTrim, "_swapless", staticmethod(lambda: True)
        ):
            t = ProgressCoupledTrim()
        with mock.patch.object(
            ProgressCoupledTrim, "_current", staticmethod(lambda: 95 * gib)
        ), mock.patch("builtins.open", side_effect=OSError("no memory.reclaim")):
            t.maybe_trim()
        self.assertFalse(t.enabled)
        self.assertEqual(t.trims, 0)


if __name__ == "__main__":
    unittest.main()
