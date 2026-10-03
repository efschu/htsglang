"""#257 (iii): WEG2-LOAD-DEVICE rates the READ, not the scheduler pass.

Vision boot 0928 (P log): every slow line of the boot (0.01-0.2 GB/s) had
``ms`` equal to the rank's previous scheduler pass -- PP0 weg2-3-22 23232 tok
in 1849 ms after a 1823 ms pass (a 2.4 s forward of a 20-token prefill),
weg2-1-20/2-21/3-23/3-24 in the same 1840-1867 ms; PP1 weg2-0-7 6616 ms after
a 6566 ms pass; PP2 8183 after 8142. The same reads between short passes ran
at 0.9-2.3 GB/s (weg2-0-9 49152 tok in 267 ms), ARENA-GET resolved 1610 pages
in 2+1+1 ms. The clock ran from the operation's creation to its reap, and the
reap happens only between forwards: queue + read + harvest wait, never the
read -- so '0.09 GB/s' named no I/O, index, pin or gather cost at all.

The aux thread now stamps the read (``read_start_time`` / ``read_end_time``
around ``_page_transfer``); the line splits queue_ms / read_ms / harvest_ms
and rates GB/s over read_ms. Driven through the real
``check_prefetch_progress`` on the #937/#1157 harness."""

import importlib.util
import os
import re
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.hicache_phase_binding import binding_state  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_1157", os.path.join(os.path.dirname(__file__), "test_1157_reaper_prices_requested_span.py")
)
h1157 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h1157)

REQ = h1157.REAP_REQ


def _line(cm):
    lines = [x for x in cm.output if "WEG2-LOAD-DEVICE" in x]
    assert lines, "no WEG2-LOAD-DEVICE line"
    return lines[-1]


def _field(line, name):
    m = re.search(r"\b%s=(-?[\d.]+)" % re.escape(name), line)
    assert m, f"{name} missing in: {line}"
    return float(m.group(1))


class TheLoadClockIsTheRead(CustomTestCase):
    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_weg2_3_22_a_long_pass_is_harvest_wait_not_read_time(self):
        """RED on 1961f756ad: the line has no read_ms, and GB/s is bytes over
        the whole 1.85 s (queue + read + the 1.8 s forward it waited out).
        GREEN: read_ms ~50, harvest_ms ~1800, GB/s over the 50 ms read."""
        cache, op = h1157._reap_scenario(probed=True)
        now = time.monotonic()
        op.start_time = now - 1.85
        op.read_start_time = now - 1.83
        op.read_end_time = now - 1.78
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", "INFO") as cm:
            cache.check_prefetch_progress(REQ)
        line = _line(cm)
        read_ms = _field(line, "read_ms")
        harvest_ms = _field(line, "harvest_ms")
        queue_ms = _field(line, "queue_ms")
        total_ms = _field(line, "ms")
        self.assertAlmostEqual(read_ms, 50.0, delta=15.0)
        self.assertGreater(harvest_ms, 1700.0)
        self.assertAlmostEqual(queue_ms, 20.0, delta=15.0)
        self.assertAlmostEqual(queue_ms + read_ms + harvest_ms, total_ms, delta=5.0)
        bytes_ = _field(line, "bytes")
        gbs = _field(line, "GB/s")
        if bytes_ > 0:
            self.assertAlmostEqual(gbs, bytes_ / (read_ms / 1000.0) / 1e9, delta=0.02 + 0.05 * gbs)

    def test_no_read_stamp_says_so_instead_of_inventing_a_rate(self):
        """An operation the aux thread never read (a revoke, a full device
        hit) prints -1, never the old creation-to-reap rate."""
        cache, op = h1157._reap_scenario(probed=True)
        op.start_time = time.monotonic() - 1.0
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", "INFO") as cm:
            cache.check_prefetch_progress(REQ)
        line = _line(cm)
        self.assertEqual(_field(line, "read_ms"), -1.0)
        self.assertEqual(_field(line, "GB/s"), -1.0)
