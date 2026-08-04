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

import contextlib
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

GIB = 1 << 30


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


class _CgroupWithPinnedInTheFileBucket:
    """A cgroup whose ``file`` bucket holds page cache AND the pinned pool.

    That is the whole defect (#537, measured 2026-08-04): CUDA pinned host
    memory is charged to ``file``, so ``memory.current`` cannot tell the two
    apart, while ``memory.reclaim`` can only ever take the page-cache half.
    """

    def __init__(self, *, anon_gib, pinned_gib, cache_gib):
        self.anon = int(anon_gib * GIB)
        self.pinned = int(pinned_gib * GIB)
        self.cache = int(cache_gib * GIB)
        self.asks = []

    @property
    def current(self):
        return self.anon + self.pinned + self.cache

    @property
    def reclaimable(self):
        return self.cache

    def reclaim(self, ask):
        self.asks.append(int(ask))
        taken = min(int(ask), self.cache)
        self.cache -= taken
        return taken


@contextlib.contextmanager
def _driven_by(cgroup, *, pinned_visible=True, anon_visible=True):
    """Wire a ProgressCoupledTrim to ``cgroup``.

    ``pinned_visible=False`` reproduces the module's PRE-#537 world model --
    the pinned pool assumed anonymous and therefore invisible in the file
    bucket -- against otherwise identical inputs. That is the can-fail arm.
    ``anon_visible=False`` is a cgroup whose ``memory.stat`` cannot be read.
    """

    class _ReclaimFile:
        def write(self, payload):
            cgroup.reclaim(int(payload))
            return len(payload)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    with mock.patch.object(
        ProgressCoupledTrim, "_current", staticmethod(lambda: cgroup.current)
    ), mock.patch.object(
        ProgressCoupledTrim,
        "_anon",
        staticmethod(lambda: cgroup.anon if anon_visible else None),
    ), mock.patch.object(
        ProgressCoupledTrim,
        "_pinned",
        staticmethod(lambda: cgroup.pinned if pinned_visible else 0),
    ), mock.patch(
        "builtins.open", lambda *a, **k: _ReclaimFile()
    ):
        yield


class TestStreamTrimBudgetModel(_EnvMixin, CustomTestCase):
    """#537: the trim's target must sit above the bytes reclaim cannot take.

    Geometry taken from the UD-Q3_K_XL boot that died on it
    (``docs/dev/ANALYSE_478_RESULT_q3kxl_refused.md``): a ~104 GiB ceiling, a
    pinned expert pool far larger than the runtime's anon, and marks that the
    desk note had already raised ABOVE the predicted floor and that still lost.
    """

    def _armed_trim(self, *, soft=96, target=90, headroom=None):
        self._set("SGLANG_GGUF_STREAM_TRIM_SOFT_GIB", soft)
        self._set("SGLANG_GGUF_STREAM_TRIM_TARGET_GIB", target)
        self._set("SGLANG_GGUF_STREAM_TRIM_HEADROOM_GIB", headroom)
        with mock.patch.object(
            ProgressCoupledTrim, "_swapless", staticmethod(lambda: True)
        ):
            return ProgressCoupledTrim()

    def test_ask_is_capped_at_what_reclaim_can_actually_give_back(self):
        cg = _CgroupWithPinnedInTheFileBucket(
            anon_gib=15, pinned_gib=85, cache_gib=4
        )
        self.assertEqual(cg.current, 104 * GIB)
        t = self._armed_trim()
        # What the pre-#537 arithmetic would have asked for, stated so the
        # over-ask is visible rather than implied.
        unfixed_ask = cg.current - t.target_bytes
        self.assertEqual(unfixed_ask, 14 * GIB)
        self.assertGreater(unfixed_ask, cg.reclaimable)

        with _driven_by(cg):
            t.maybe_trim()

        self.assertEqual(cg.asks, [4 * GIB], "ask = current - unreclaimable floor")
        self.assertTrue(t.floor_overrode_target)
        self.assertEqual(t.floor_bytes, 100 * GIB)

    def test_it_converges_instead_of_asking_on_every_call(self):
        """The falsifier. Unfixed the target is unsatisfiable, so the trim
        never stops asking and keeps evicting the loader's read-ahead; fixed it
        drains the reclaimable part once and then goes quiet.
        """
        cg = _CgroupWithPinnedInTheFileBucket(
            anon_gib=15, pinned_gib=85, cache_gib=10
        )
        t = self._armed_trim()
        with _driven_by(cg):
            for _ in range(10):
                t.maybe_trim()

        self.assertEqual(cg.cache, 0, "the reclaimable part is given back once")
        self.assertEqual(t.trims, 1, "and then the trim stops asking")
        # The unfixed arithmetic at the SAME converged state still wants
        # 10 GiB that no longer exist -- which is why it never stopped.
        self.assertEqual(cg.current - t.target_bytes, 10 * GIB)
        self.assertEqual(cg.reclaimable, 0)

    def test_can_fail_when_the_pinned_pool_stays_invisible(self):
        """Same inputs, pinned pool assumed anonymous (the documented error).

        This must be RED behaviour: the trim reverts to permanent reclaim
        pressure. It proves the pinned term is load-bearing and that the
        convergence above is not an artefact of the harness.
        """
        cg = _CgroupWithPinnedInTheFileBucket(
            anon_gib=15, pinned_gib=85, cache_gib=10
        )
        t = self._armed_trim()
        with _driven_by(cg, pinned_visible=False):
            for _ in range(10):
                t.maybe_trim()

        self.assertEqual(cg.cache, 0)
        self.assertEqual(t.trims, 10, "asks on every call, forever")
        self.assertFalse(t.floor_overrode_target)
        self.assertEqual(cg.asks[-1], 10 * GIB, "still asking for what is gone")

    def test_neutral_when_the_floor_sits_below_the_configured_target(self):
        """Behavioural neutrality for every boot without a large pinned pool:
        the arithmetic is exactly ``current - target``, as before #537."""
        cg = _CgroupWithPinnedInTheFileBucket(anon_gib=15, pinned_gib=0, cache_gib=85)
        t = self._armed_trim()
        with _driven_by(cg):
            t.maybe_trim()
        self.assertEqual(cg.asks, [10 * GIB])
        self.assertFalse(t.floor_overrode_target)

    def test_neutral_when_the_cgroup_cannot_report_anon(self):
        """An unreadable memory.stat must not invent a floor."""
        cg = _CgroupWithPinnedInTheFileBucket(anon_gib=15, pinned_gib=85, cache_gib=4)
        t = self._armed_trim()
        with _driven_by(cg, anon_visible=False):
            t.maybe_trim()
        self.assertIsNone(t.floor_bytes)
        self.assertEqual(cg.asks, [14 * GIB], "pre-#537 arithmetic, unchanged")

    def test_headroom_is_added_to_the_floor_when_the_operator_asks_for_it(self):
        cg = _CgroupWithPinnedInTheFileBucket(anon_gib=15, pinned_gib=80, cache_gib=9)
        t = self._armed_trim(headroom=3)
        with _driven_by(cg):
            t.maybe_trim()
        self.assertEqual(t.floor_bytes, 98 * GIB)
        self.assertEqual(cg.asks, [6 * GIB])

    def test_headroom_defaults_to_zero_so_the_formula_adds_no_desk_number(self):
        t = self._armed_trim()
        self.assertEqual(t.headroom_bytes, 0)


if __name__ == "__main__":
    unittest.main()
