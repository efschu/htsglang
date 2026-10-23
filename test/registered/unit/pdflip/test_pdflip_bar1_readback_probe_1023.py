"""barlink-switch-1023: the Fix-A arrival readback is measured before a lane
relies on it.

REPRO (boot dkr27bnvfp4duobar1duo10230546, 23.10. 05:46Z, ACS redirect on):
every bar1 lane in which the RTX 5090 deposited into a 3080's BAR1 window
cost ~33-90 ms per batch regardless of the slot size, the reverse direction
0.2-0.6 ms with the same code. The lane-time lines below are copied verbatim
from that boot's P/D logs; ``test_repro_*`` derives the per-batch cost from
them and shows it is a FIXED per-batch price (the one per-batch op whose cost
does not scale with the bytes is the 16-byte P2P readback), not a bandwidth.

FIX: ``Bar1Lanes._probe_readback`` times a few production-shaped reads at
connect time; a stalled read refuses the lane BY NAME (host path) instead of
paying the stall on every batch. Red on the base 7156ffa3bd: the helpers do
not exist there (ImportError).

CPU-only: the CUDA calls are faked.
"""

import inspect
import re
import time
import unittest
from unittest import mock

from flliper.srt.pdflip import bar1_lanes as bl
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

#: Verbatim (cut after the fields used) from
#: /spinning/docker-acceptance/27b/evidence/
#: boot_weg2_dkr27bnvfp4duobar1duo10230546_f244804b85_1023_054641.{P,D}.log
BOOT_LINES = (
    # 5090 (PP0) deposits into the 3080 (D TP1), 7-MiB slots
    "[2026-10-23 05:49:52 PP0] PDFLIP-BAR1 lane-time lane=p0 phase=deposit seq=1-weights_0 units=374 batches=87 bytes=633762236 total_ms=2957 wait_ms=113 copy_sync_ms=2792 rate_GBs=0.2 via=bar1 credits=sock credit_ms=113 issue_ms=47 sync_ms=2792 sync_cpu_ms=2762 max_sync_ms=67.1 send_ms=4 mode=serial write_order=readbackx87",
    "[2026-10-23 05:49:55 PP0] PDFLIP-BAR1 lane-time lane=p0 phase=deposit seq=1-weights_1 units=374 batches=87 bytes=633762236 total_ms=2946 wait_ms=3 copy_sync_ms=2931 rate_GBs=0.2 via=bar1 credits=sock credit_ms=3 issue_ms=7 sync_ms=2931 sync_cpu_ms=2929 max_sync_ms=67.1 send_ms=3 mode=serial write_order=readbackx87",
    # 5090 (D TP0) deposits into the 3080 (P PP1), 19-MiB slots
    "[2026-10-23 05:48:59 TP0] PDFLIP-BAR1 lane-time lane=p0 phase=deposit seq=0-weights_6 units=138 batches=47 bytes=922801746 total_ms=1649 wait_ms=2 copy_sync_ms=1561 rate_GBs=0.6 via=bar1 credits=sock credit_ms=2 issue_ms=84 sync_ms=1561 sync_cpu_ms=1559 max_sync_ms=67.1 send_ms=1 mode=serial write_order=readbackx47",
    "[2026-10-23 05:49:01 TP0] PDFLIP-BAR1 lane-time lane=p0 phase=deposit seq=0-weights_draft units=135 batches=53 bytes=1037967360 total_ms=1824 wait_ms=2 copy_sync_ms=1817 rate_GBs=0.6 via=bar1 credits=sock credit_ms=2 issue_ms=3 sync_ms=1817 sync_cpu_ms=1814 max_sync_ms=65.3 send_ms=1 mode=serial write_order=readbackx53",
    # 3080 (P PP1) deposits into the 5090 (D TP0), 7-MiB slots
    "[2026-10-23 05:49:52 PP1] PDFLIP-BAR1 lane-time lane=p1 phase=deposit seq=1-weights_7 units=489 batches=202 bytes=1478399044 total_ms=3017 wait_ms=2974 copy_sync_ms=36 rate_GBs=0.5 via=bar1 credits=sock credit_ms=2974 issue_ms=5 sync_ms=36 sync_cpu_ms=36 max_sync_ms=0.4 send_ms=1 mode=serial write_order=readbackx202",
    # 3080 (D TP1) deposits into the 5090 (P PP0), 19-MiB slots
    "[2026-10-23 05:49:02 TP1] PDFLIP-BAR1 lane-time lane=p1 phase=deposit seq=0-weights units=64 batches=64 bytes=1271398400 total_ms=53 wait_ms=1 copy_sync_ms=50 rate_GBs=23.8 via=bar1 credits=sock credit_ms=1 issue_ms=1 sync_ms=50 sync_cpu_ms=50 max_sync_ms=1.9 send_ms=0 mode=serial write_order=readbackx64",
    # epoch 8, the hang: one batch sync of the 5090's deposit held 75 s
    "[2026-10-23 05:55:12 PP0] PDFLIP-BAR1 lane-time lane=p0 phase=deposit seq=7-weights units=174 batches=174 bytes=1271398400 total_ms=171815 wait_ms=11 copy_sync_ms=171785 rate_GBs=0.0 via=bar1 credits=sock credit_ms=11 issue_ms=10 sync_ms=171785 sync_cpu_ms=97553 max_sync_ms=75040.1 send_ms=5 mode=serial write_order=readbackx174",
)
#: The depositor's card per rank in that boot (P = PP0 5090 / PP1 3080,
#: D = TP0 5090 / TP1 3080; --rank-gpu-id 0,1 in both groups).
CARD = {"PP0": "5090", "TP0": "5090", "PP1": "3080", "TP1": "3080"}


def _parse(line):
    f = dict(re.findall(r"(\w+)=([\w.\-]+)", line))
    rank = re.search(r" (PP\d|TP\d)\]", line).group(1)
    return rank, int(f["batches"]), int(f["bytes"]), int(f["copy_sync_ms"]), float(f["max_sync_ms"])


class TestReproTheBootSignature(CustomTestCase):
    def _per_batch(self):
        out = []
        for line in BOOT_LINES[:-1]:
            rank, b, by, cs, mx = _parse(line)
            out.append((CARD[rank], by / b / 2**20, cs / b, mx))
        return out

    def test_repro_5090_depositor_pays_a_fixed_price_per_batch(self):
        rows = [r for r in self._per_batch() if r[0] == "5090"]
        small = [ms for _, mib, ms, _ in rows if mib < 10]
        big = [ms for _, mib, ms, _ in rows if mib > 15]
        self.assertTrue(small and big)
        # ~2.7x the bytes per batch, the same time per batch: not a bandwidth
        for ms in small + big:
            self.assertGreater(ms, 25.0)
            self.assertLess(ms, 40.0)
        self.assertLess(abs(sum(small) / len(small) - sum(big) / len(big)), 5.0)

    def test_repro_3080_depositor_same_code_is_two_orders_faster(self):
        for card, mib, ms, mx in self._per_batch():
            if card == "3080":
                self.assertLess(ms, 1.0)
                self.assertLess(mx, 2.5)

    def test_repro_the_stalls_are_beyond_the_probe_threshold_healthy_is_below(self):
        stall = bl.READBACK_STALL_MS_DEFAULT
        for card, _mib, ms, mx in self._per_batch():
            if card == "5090":
                self.assertGreater(mx, stall)
                self.assertGreater(ms, stall)
            else:
                self.assertLess(mx, stall)

    def test_repro_the_hang_sat_in_a_5090_deposit_sync(self):
        rank, b, _by, cs, mx = _parse(BOOT_LINES[-1])
        self.assertEqual(CARD[rank], "5090")
        self.assertGreater(mx, 60000.0)        # 75 s, the GSP heartbeat gap
        self.assertIn("write_order=readbackx174", BOOT_LINES[-1])


class TestVerdict(CustomTestCase):
    def test_fast_reads_keep_the_proof(self):
        self.assertEqual(bl.readback_verdict([0.02, 0.03, 0.02, 0.04], 5.0), "")
        self.assertEqual(bl.readback_verdict([], 5.0), "")

    def test_a_stalled_read_is_named_with_its_numbers(self):
        why = bl.readback_verdict([33.4], 5.0, all_ones=1)
        self.assertIn("readback stalls", why)
        self.assertIn("33.4", why)
        self.assertIn("all-ones", why)
        self.assertIn("barlink-switch-1023", why)

    def test_all_ones_alone_decides_nothing(self):
        self.assertEqual(bl.readback_verdict([0.02], 5.0, all_ones=1), "")


class TestProbeLoop(CustomTestCase):
    def test_stops_at_the_first_stall(self):
        t = [0.0]
        calls = []

        def clock():
            return t[0]

        def read_once():
            calls.append(1)
            t[0] += 0.050 if len(calls) == 2 else 0.00002
            return b"\x00" * 16

        # confirm=1 = the 23.10. form; the default (READBACK-CONFIRM, 2) reads
        # on after a single stall -- test_pdflip_bar1_lanes3x80_1023.py
        lat, ones = bl.run_readback_probe(read_once, 8, 5.0, clock=clock, confirm=1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(lat), 2)
        self.assertGreater(lat[1], 5.0)
        self.assertEqual(ones, 0)

    def test_counts_all_ones(self):
        # 27b-lanes3x80-1023 review: all-ones counts only on STALLED reads (the
        # completion-timeout signature); fast reads of 0xff window bytes are data
        lat, ones = bl.run_readback_probe(lambda: b"\xff" * 16, 3, 5.0)
        self.assertEqual(ones, 0)
        self.assertEqual(len(lat), 3)
        t = [0.0]

        def slow():
            t[0] += 0.040
            return b"\xff" * 16
        lat, ones = bl.run_readback_probe(slow, 3, 5.0, clock=lambda: t[0])
        self.assertEqual((ones, len(lat)), (1, 1))

    def test_env_knobs(self):
        self.assertEqual(bl.readback_probe_n({}), bl.READBACK_PROBE_N_DEFAULT)
        self.assertEqual(bl.readback_probe_n({bl.ENV_READBACK_PROBE_N: "0"}), 0)
        self.assertEqual(bl.readback_probe_n({bl.ENV_READBACK_PROBE_N: "999"}), 64)
        self.assertEqual(bl.readback_probe_n({bl.ENV_READBACK_PROBE_N: "x"}),
                         bl.READBACK_PROBE_N_DEFAULT)
        self.assertEqual(bl.readback_stall_ms({}), bl.READBACK_STALL_MS_DEFAULT)
        self.assertEqual(bl.readback_stall_ms({bl.ENV_READBACK_STALL_MS: "-1"}),
                         bl.READBACK_STALL_MS_DEFAULT)
        self.assertEqual(bl.readback_stall_ms({bl.ENV_READBACK_STALL_MS: "12.5"}), 12.5)


class _FakeCu:
    """The two cudaMemcpy calls of the probe; the peer read optionally slow."""

    def __init__(self, peer_dev, read_s=0.0, data=b"\x00" * 16, fail=None):
        self.peer_dev, self.read_s, self.data, self.fail = peer_dev, read_s, data, fail
        self.freed, self.reads = [], 0

    def malloc(self, n):
        if self.fail:
            raise RuntimeError(self.fail)
        return 0x1000

    def free(self, p):
        self.freed.append(p)

    def memcpy(self, dst, src, n):
        if src == self.peer_dev:
            self.reads += 1
            if self.read_s:
                time.sleep(self.read_s)
        else:
            import ctypes
            ctypes.memmove(dst, self.data, n)


def _lanes():
    return bl.Bar1Lanes("nonce", "P", 0, 0, [(0, 1)], log=lambda m: None, root="/tmp")


class TestProbeOnTheMapping(CustomTestCase):
    def test_a_fast_peer_keeps_the_lane_and_notes_the_price(self):
        cu = _FakeCu(0xBEEF)
        with mock.patch.dict("os.environ", {bl.ENV_WRITE_ORDER: "1"}):
            why, note = _lanes()._probe_readback(cu, 0xBEEF)
        self.assertEqual(why, "")
        self.assertIn("readback_probe_ms=", note)
        self.assertEqual(cu.reads, bl.READBACK_PROBE_N_DEFAULT)
        self.assertEqual(cu.freed, [0x1000])

    def test_a_stalling_peer_refuses_the_lane_after_ONE_read(self):
        cu = _FakeCu(0xBEEF, read_s=0.008, data=b"\xff" * 16)
        with mock.patch.dict("os.environ", {bl.ENV_WRITE_ORDER: "1"}):
            why, note = _lanes()._probe_readback(cu, 0xBEEF)
        self.assertIn("readback stalls", why)
        self.assertIn("all-ones", why)
        self.assertEqual(cu.reads, 1)
        self.assertEqual(cu.freed, [0x1000])

    def test_write_order_off_means_no_readback_and_no_probe(self):
        cu = _FakeCu(0xBEEF, read_s=0.008)
        with mock.patch.dict("os.environ", {bl.ENV_WRITE_ORDER: "0"}):
            self.assertEqual(_lanes()._probe_readback(cu, 0xBEEF), ("", ""))
        self.assertEqual(cu.reads, 0)

    def test_probe_off_by_operator_word_is_named(self):
        cu = _FakeCu(0xBEEF, read_s=0.008)
        with mock.patch.dict("os.environ", {bl.ENV_WRITE_ORDER: "1",
                                            bl.ENV_READBACK_PROBE_N: "0"}):
            why, note = _lanes()._probe_readback(cu, 0xBEEF)
        self.assertEqual(why, "")
        self.assertIn("readback_probe=OFF", note)
        self.assertEqual(cu.reads, 0)

    def test_a_probe_that_cannot_run_refuses_nothing(self):
        cu = _FakeCu(0xBEEF, fail="no memory")
        with mock.patch.dict("os.environ", {bl.ENV_WRITE_ORDER: "1"}):
            why, note = _lanes()._probe_readback(cu, 0xBEEF)
        self.assertEqual(why, "")
        self.assertIn("readback_probe=FAILED", note)

    def test_connect_peer_refuses_before_the_lane_is_published(self):
        src = inspect.getsource(bl.Bar1Lanes.connect_peer)
        i_probe = src.index("self._probe_readback(")
        self.assertLess(i_probe, src.index("PeerWindow("))
        self.assertLess(i_probe, src.index("self._mark_mapped(lane_key)"))
        refuse = src[i_probe:src.index("PeerWindow(")]
        for needle in ("cu.unregister(", "mapped.close()", "self._holder.release(",
                       "self.refusals[lane_key] = rb_why", "return None"):
            self.assertIn(needle, refuse, needle)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# The lane-time line separates the collector's START from the transport
# (boot dkr27bnvfp4duorebarbar1duo10230932, ACS 0011: 25 of the 49 tags under
# 10 GB/s had a depositor span 0.1-0.27 s longer than the collector's whole
# collect -- the depositor waited for free(0) while its collector had not
# begun). ``first_free_ms`` is that wait, ``xfer_GBs`` the rate without it.
# ---------------------------------------------------------------------------

def _late_collector_run(delay_s, ring=3, n=7):
    import ctypes
    import os
    import tempfile
    import threading
    from types import SimpleNamespace

    from flliper.srt.pdflip import weight_exchange_transport as tp

    slot = 4096

    def addr(buf):
        return ctypes.addressof(ctypes.c_char.from_buffer(buf))

    class Ops:
        def __init__(self):
            self._n, self._keep = 0, []

        def create_stream(self, device):
            self._n += 1
            return 100 + self._n

        def synchronize(self, stream):
            pass

        def memcpy_async(self, dst, src, nb, stream):
            ctypes.memmove(dst, src, nb)

        def memcpy2d_async(self, *a):
            raise AssertionError("flat only")

        def ptr_attrs(self, a):
            return (0, 2, 0)

        def raw_malloc(self, device, nb):
            b = bytearray(nb)
            self._keep.append(b)
            return addr(b)

        def rdma_flush(self, device):
            pass

    bl._RB_SCRATCH.clear()
    root = tempfile.mkdtemp(prefix="ff1023")
    pairs = ((0, 1),)
    window = bytearray(slot * ring)
    dep = bl.Bar1Lanes("n1", "P", 0, 0, pairs, log=lambda *_a: None, root=root)
    col = bl.Bar1Lanes("n1", "D", 1, 0, pairs, log=lambda *_a: None, root=root)
    col.recv["p0"] = SimpleNamespace(dptr=addr(window), slot_bytes=slot, ring=ring)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=addr(window), slot_bytes=slot, ring=ring)
    srcs = [bytearray(os.urandom(slot)) for _ in range(n)]
    dsts = [bytearray(slot) for _ in range(n)]
    descs = [SimpleNamespace(kind=tp.FLAT, nbytes=slot, src_off=0, dst_off=0,
                             param_name=f"w{i}", tag="weights_9",
                             src_ptr=addr(srcs[i]), dst_ptr=addr(dsts[i]))
             for i in range(n)]
    out, dlog = {}, []

    def collect():
        time.sleep(delay_s)
        out["c"] = bl.run_bar1_units(descs, Ops(), lanes=col, lane_key="p0", role="dst",
                                     seq="5-weights_9", phase="collect", budget_s=10.0,
                                     log=lambda *_a: None)

    th = threading.Thread(target=collect)
    th.start()
    out["d"] = bl.run_bar1_units(descs, Ops(), lanes=dep, lane_key="p0", role="src",
                                 seq="5-weights_9", phase="deposit", budget_s=10.0,
                                 log=dlog.append)
    th.join(20)
    assert out == {"c": "", "d": ""}, out
    assert all(bytes(s) == bytes(d) for s, d in zip(srcs, dsts))
    line = [x for x in dlog if "lane-time" in x][0]
    f = dict(re.findall(r"(\w+)=([\d.]+)", line))
    return float(f["first_free_ms"]), float(f["rate_GBs"]), float(f["xfer_GBs"]), float(f["total_ms"])


class TestFirstFreeSeparatesTheCollectorStart(CustomTestCase):
    def test_a_late_collector_shows_as_first_free_not_as_transport(self):
        ff, rate, xfer, total = _late_collector_run(0.3)
        self.assertGreater(ff, 250.0)
        self.assertGreater(total - ff, 0.0)
        self.assertLess(total - ff, 0.5 * total)
        self.assertGreaterEqual(xfer, rate)

    def test_a_ready_collector_has_a_small_first_free(self):
        ff, _rate, _xfer, total = _late_collector_run(0.0)
        self.assertLess(ff, 200.0)
        self.assertLessEqual(ff, total)
