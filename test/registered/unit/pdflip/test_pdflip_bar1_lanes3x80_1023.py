"""27b-lanes3x80-1023: the two lane defects of the 2x RTX 3080 boot
``boot_weg2_dkr27bint4arbar1int4ar10231723_fe572f3224_1023_172423`` (P = PP2,
D = TP2, NVML 1 = 18:00.0, 2 = 19:00.0, BAR1 256 MiB each, PEX880xx switch,
ACS 0011), each with its repro:

1. READBACK-CONFIRM. ``P PP0 lane=p0 role=src REFUSED peer=0000:19:00.0
   write-order readback stalls: ... took 10.5 ms (> 5 ms; 1 read(s): 10.46
   ms)`` -- ONE cold read refused the P->D lane PP0 -> TP1 for the whole boot
   (all 37 P->D flips ran it on the host path). Healthy reads on the same
   rig: 0.02-1.9 ms; the sick class (root-port hairpin) stalled on EVERY read.

2. MULTI-REGION ring. D's ranks own ``{'world:0': 15, 'tp:0': 31, 'dcp:0':
   39} MiB`` and receive on one lane; the lane borrowed dcp:0 only (ring 4 x 9
   MiB) while P's lane got pp:0 (4 x 23 MiB). Per batch both directions ran at
   the same period (1.55 / 1.58 ms), so P->D p1 moved 5.3 GB/s p50 against
   D->P 12.5. The spare tp:0 region now joins the lane: 4 x 15 MiB.
"""
import ctypes
import os
import re
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from flliper.srt.pdflip import bar1_lanes as bl
from flliper.srt.pdflip import weight_exchange_transport as tp
from flliper.test.test_utils import CustomTestCase

MIB = 1 << 20

# the boot's own lines (verbatim, prefix trimmed)
REFUSED_1723 = ("[2026-10-23 17:28:03 PP0] PDFLIP-BAR1 lane=p0 role=src REFUSED peer=0000:19:00.0 "
                "write-order readback stalls: a 16-B P2P read of the peer window took 10.5 ms "
                "(> 5 ms; 1 read(s): 10.46 ms) -- every batch would pay it and it proves no "
                "arrival (barlink-switch-1023) -> host lane")
REFUSED_1824 = ("[2026-10-23 18:27:25 TP0] PDFLIP-BAR1 lane=p0 role=src REFUSED peer=0000:19:00.0 "
                "write-order readback stalls: a 16-B P2P read of the peer window took 224.9 ms "
                "(> 5 ms; 2 read(s): 0.75, 224.91 ms)")
WINDOWS_D = "[2026-10-23 17:28:03 TP0] PDFLIP-BAR1 group windows of this process: {'world:0': 15, 'tp:0': 31, 'dcp:0': 39} MiB; dst lanes=['p1']"
WINDOWS_P = "[2026-10-23 17:26:15 PP0] PDFLIP-BAR1 group windows of this process: {'world:0': 23, 'pp:0': 95} MiB; dst lanes=['p1']"
DST_D = ("[2026-10-23 17:28:03 TP0] PDFLIP-BAR1 lane=p1 role=dst group=D window=36 MiB slot=9 MiB "
         "ring=4 bdf=0000:18:00.0 export=NV_ESC_EXPORT_TO_DMABUF_FD borrowed=dcp:0 offset=0MiB")
DST_P = ("[2026-10-23 17:26:15 PP0] PDFLIP-BAR1 lane=p1 role=dst group=P window=92 MiB slot=23 MiB "
         "ring=4 bdf=0000:18:00.0 export=NV_ESC_EXPORT_TO_DMABUF_FD borrowed=pp:0 offset=0MiB")
PAIRS = [(0, 1), (1, 0)]       # p0: card 0 -> card 1, p1: card 1 -> card 0


def _windows(line, extra_kib=512):
    """The group windows of a boot line, in bytes; the log rounds down to MiB,
    the payload regions carry a sub-MiB remainder (kept here)."""
    got = re.search(r"process: (\{[^}]*\}) MiB", line).group(1)
    return {k: (v << 20) + extra_kib * 1024 for k, v in eval(got).items()}  # noqa: S307


def _lat_reader(lat_ms, data=b"\x00" * 16):
    """read_once + clock that replay a list of read latencies (ms)."""
    t, i = [0.0], [0]

    def clock():
        return t[0]

    def read_once():
        t[0] += lat_ms[min(i[0], len(lat_ms) - 1)] / 1000.0
        i[0] += 1
        return data
    return read_once, clock, i


class TestReadbackConfirm(CustomTestCase):
    def test_repro_1723_one_cold_read_no_longer_refuses_the_lane(self):
        self.assertIn("1 read(s): 10.46 ms", REFUSED_1723)
        read_once, clock, n = _lat_reader([10.46, 0.066, 0.043, 0.033])
        lat, ones = bl.run_readback_probe(read_once, 4, 5.0, clock=clock, confirm=2)
        self.assertEqual(n[0], 4)                       # the probe read on
        self.assertEqual(bl.readback_verdict(lat, 5.0, ones, confirm=2), "")
        # the 23.10. form (confirm=1) refused exactly this lane
        read_once, clock, n = _lat_reader([10.46, 0.066, 0.043, 0.033])
        lat1, ones1 = bl.run_readback_probe(read_once, 4, 5.0, clock=clock, confirm=1)
        self.assertEqual(n[0], 1)
        self.assertIn("readback stalls", bl.readback_verdict(lat1, 5.0, ones1, confirm=1))

    def test_repro_1824_a_second_read_outlier_no_longer_refuses(self):
        self.assertIn("0.75, 224.91 ms", REFUSED_1824)
        read_once, clock, n = _lat_reader([0.75, 224.91, 0.6, 0.6])
        lat, ones = bl.run_readback_probe(read_once, 4, 5.0, clock=clock, confirm=2)
        self.assertEqual(n[0], 4)
        self.assertEqual(bl.readback_verdict(lat, 5.0, ones, confirm=2), "")

    def test_the_sick_class_is_still_refused_after_two_reads(self):
        # root-port hairpin (barlink-switch-1023): every read 33-85 ms
        read_once, clock, n = _lat_reader([33.4, 50.5, 67.1, 84.6])
        lat, ones = bl.run_readback_probe(read_once, 4, 5.0, clock=clock, confirm=2)
        self.assertEqual(n[0], 2)                       # bounded: two stalls, no more
        why = bl.readback_verdict(lat, 5.0, ones, confirm=2)
        self.assertIn("readback stalls", why)
        self.assertIn("2 of 2 read(s) stalled", why)

    def test_a_stall_with_all_ones_refuses_at_once(self):
        read_once, clock, n = _lat_reader([40.0, 0.05], data=b"\xff" * 16)
        lat, ones = bl.run_readback_probe(read_once, 4, 5.0, clock=clock, confirm=2)
        self.assertEqual(n[0], 1)
        self.assertIn("all-ones", bl.readback_verdict(lat, 5.0, ones, confirm=2))

    def test_a_probe_of_one_read_still_decides_on_it(self):
        self.assertIn("readback stalls", bl.readback_verdict([12.0], 5.0, 0, confirm=2))
        self.assertEqual(bl.readback_verdict([0.1], 5.0, 0, confirm=2), "")

    def test_two_stalls_among_fast_reads_refuse(self):
        why = bl.readback_verdict([7.0, 0.1, 9.0, 0.1], 5.0, 0, confirm=2)
        self.assertIn("2 of 4 read(s) stalled", why)

    def test_knob(self):
        self.assertEqual(bl.readback_confirm({}), 2)
        self.assertEqual(bl.READBACK_CONFIRM_DEFAULT, 2)
        self.assertEqual(bl.readback_confirm({bl.ENV_READBACK_CONFIRM: "1"}), 1)
        self.assertEqual(bl.readback_confirm({bl.ENV_READBACK_CONFIRM: "0"}), 1)
        self.assertEqual(bl.readback_confirm({bl.ENV_READBACK_CONFIRM: "99"}), 8)
        self.assertEqual(bl.readback_confirm({bl.ENV_READBACK_CONFIRM: "x"}), 2)

    def test_probe_on_the_mapping_keeps_a_lane_with_one_slow_read(self):
        class Cu:
            def __init__(self):
                self.reads, self.freed = 0, []

            def malloc(self, n):
                return 0x1000

            def free(self, p):
                self.freed.append(p)

            def memcpy(self, dst, src, n):
                if src == 0xBEEF:
                    self.reads += 1
                    if self.reads == 1:
                        time.sleep(0.012)       # one cold read, 12 ms
                else:
                    ctypes.memmove(dst, b"\x00" * 16, n)

        cu = Cu()
        lanes = bl.Bar1Lanes("nonce", "P", 0, 0, [(0, 1)], log=lambda m: None, root="/tmp")
        with mock.patch.dict("os.environ", {bl.ENV_WRITE_ORDER: "1"}):
            why, note = lanes._probe_readback(cu, 0xBEEF)
        self.assertEqual(why, "")
        self.assertEqual(cu.reads, bl.READBACK_PROBE_N_DEFAULT)
        self.assertIn("stalls=1 confirm=2", note)


class TestMultiRegionPlan(CustomTestCase):
    def test_repro_1723_d_lane_takes_both_regions(self):
        self.assertIn("borrowed=dcp:0", DST_D)
        wins = _windows(WINDOWS_D)
        # the single-region plan of the boot: 36 MiB, 9 MiB slots
        one = bl.borrow_plan("D", "p1", PAIRS, [], wins, 4, dst_lanes=["p1"])
        self.assertEqual((one[0], one[2] >> 20, one[3] >> 20), ("dcp:0", 36, 9))
        plan = bl.borrow_plan_regions("D", "p1", PAIRS, [], wins, 4, dst_lanes=["p1"], env={})
        self.assertEqual([(n, o, z >> 20, s >> 20) for n, o, z, s in plan],
                         [("dcp:0", 0, 30, 15), ("tp:0", 0, 30, 15)])
        self.assertEqual(sum(z for _n, _o, z, _s in plan) // plan[0][3], 4)   # ring stays 4

    def test_p_lane_is_byte_identical(self):
        self.assertIn("slot=23 MiB", DST_P)
        wins = _windows(WINDOWS_P)
        one = bl.borrow_plan("P", "p1", PAIRS, [], wins, 4, dst_lanes=["p1"])
        plan = bl.borrow_plan_regions("P", "p1", PAIRS, [], wins, 4, dst_lanes=["p1"], env={})
        self.assertEqual(plan, [one])
        self.assertEqual((one[2] >> 20, one[3] >> 20), (92, 23))

    def test_switch_off_is_the_old_plan(self):
        wins = _windows(WINDOWS_D)
        one = bl.borrow_plan("D", "p1", PAIRS, [], wins, 4, dst_lanes=["p1"])
        off = bl.borrow_plan_regions("D", "p1", PAIRS, [], wins, 4, dst_lanes=["p1"],
                                     env={bl.ENV_MULTI_REGION: "0"})
        self.assertEqual(off, [one])
        self.assertTrue(bl.multi_region_on({}))

    def test_two_receiving_lanes_keep_one_region_each(self):
        # the 18.09. three-card form: two lanes into one 3080 process
        wins = _windows(WINDOWS_D)
        pairs = [(0, 1), (2, 1)]
        for lk in ("p0", "p1"):
            one = bl.borrow_plan("D", lk, pairs, [], wins, 4, dst_lanes=["p0", "p1"])
            got = bl.borrow_plan_regions("D", lk, pairs, [], wins, 4, dst_lanes=["p0", "p1"], env={})
            self.assertEqual(got, [one])

    def test_a_spare_region_that_does_not_grow_the_slot_is_left(self):
        wins = {"dcp:0": 64 * MIB, "tp:0": 4 * MIB}
        one = bl.borrow_plan("D", "p1", PAIRS, [], wins, 4, dst_lanes=["p1"])
        self.assertEqual(bl.borrow_plan_regions("D", "p1", PAIRS, [], wins, 4,
                                                dst_lanes=["p1"], env={}), [one])

    def test_refusal_text_passes_through(self):
        got = bl.borrow_plan_regions("D", "p1", PAIRS, [], {"world:0": 15 * MIB}, 4,
                                     dst_lanes=["p1"], env={})
        self.assertIsInstance(got, str)

    def test_slots_helper(self):
        slot, parts = bl.multi_region_slots([("tp:0", 0, 31 * MIB), ("dcp:0", 0, 39 * MIB)], 4)
        self.assertEqual(slot, 15 * MIB)
        self.assertEqual(parts, [("dcp:0", 0, 30 * MIB), ("tp:0", 0, 30 * MIB)])
        self.assertIsNone(bl.multi_region_slots([], 4))
        self.assertIsNone(bl.multi_region_slots([("a", 0, 3 * MIB)], 4))

    def test_slot_addr(self):
        self.assertEqual(bl.slot_addr(1000, 5, 4, 10), 1010)
        self.assertEqual(bl.slot_addr((7, 70, 700, 7000), 5, 4, 10), 70)


def _addr(buf):
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


class _Ops:
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
        return _addr(b)

    def rdma_flush(self, device):
        pass


class TestMultiRegionTransport(CustomTestCase):
    """The production ``run_bar1_units`` over a ring whose 4 slots live in two
    separate regions (2 + 2); the region ends are fenced by guard bytes that
    contiguous ``base + i * slot`` arithmetic would overwrite."""

    def _run(self, multi=True, n=11):
        bl._RB_SCRATCH.clear()
        slot, ring, guard = 4096, 4, b"\xa5" * 8192
        reg_a = bytearray(2 * slot) + bytearray(guard)     # slots 0, 1 + guard
        reg_b = bytearray(2 * slot) + bytearray(guard)     # slots 2, 3 + guard
        table = [_addr(reg_a), _addr(reg_a) + slot, _addr(reg_b), _addr(reg_b) + slot]
        root = tempfile.mkdtemp(prefix="l3x80")
        pairs = ((0, 1),)
        dep = bl.Bar1Lanes("n1", "P", 0, 0, pairs, log=lambda *_a: None, root=root)
        col = bl.Bar1Lanes("n1", "D", 1, 0, pairs, log=lambda *_a: None, root=root)
        col.recv["p0"] = SimpleNamespace(dptr=table[0], slot_bytes=slot, ring=ring,
                                         slot_addrs=table if multi else None)
        dep.peers["p0"] = SimpleNamespace(dev_ptr=table[0], slot_bytes=slot, ring=ring,
                                          slot_addrs=table if multi else None)
        srcs = [bytearray(os.urandom(slot)) for _ in range(n)]
        dsts = [bytearray(slot) for _ in range(n)]
        descs = [SimpleNamespace(kind=tp.FLAT, nbytes=slot, src_off=0, dst_off=0,
                                 param_name=f"w{i}", tag="weights_5",
                                 src_ptr=_addr(srcs[i]), dst_ptr=_addr(dsts[i]))
                 for i in range(n)]
        out, logs = {}, []

        def collect():
            out["c"] = bl.run_bar1_units(descs, _Ops(), lanes=col, lane_key="p0", role="dst",
                                         seq="17-weights_5", phase="collect", budget_s=10.0,
                                         log=logs.append)

        th = threading.Thread(target=collect)
        th.start()
        out["d"] = bl.run_bar1_units(descs, _Ops(), lanes=dep, lane_key="p0", role="src",
                                     seq="17-weights_5", phase="deposit", budget_s=10.0,
                                     log=logs.append)
        th.join(20)
        return out, srcs, dsts, reg_a, reg_b, guard, logs

    def test_bytes_land_and_no_slot_leaves_its_region(self):
        out, srcs, dsts, reg_a, reg_b, guard, logs = self._run(multi=True)
        self.assertEqual(out, {"c": "", "d": ""})
        self.assertTrue(all(bytes(s) == bytes(d) for s, d in zip(srcs, dsts)))
        self.assertEqual(bytes(reg_a[2 * 4096:]), guard)
        self.assertEqual(bytes(reg_b[2 * 4096:]), guard)
        self.assertNotEqual(bytes(reg_b[:4096]), b"\x00" * 4096)     # region B carried batches

    def test_lane_time_carries_the_wall_clock_span(self):
        _out, _s, _d, _a, _b, _g, logs = self._run(multi=True, n=5)
        lines = [x for x in logs if "PDFLIP-BAR1 lane-time" in x]
        self.assertEqual(len(lines), 2)
        for line in lines:
            f = dict(re.findall(r" (t0|t)=([\d.]+)", line))
            self.assertLessEqual(float(f["t0"]), float(f["t"]))
            self.assertLess(abs(float(f["t"]) - time.time()), 30.0)

    def test_a_table_of_the_wrong_length_is_refused_by_name(self):
        lanes = bl.Bar1Lanes("n1", "D", 1, 0, ((0, 1),), log=lambda *_a: None,
                             root=tempfile.mkdtemp(prefix="l3x80"))
        lanes.recv["p0"] = SimpleNamespace(dptr=1, slot_bytes=4096, ring=4, slot_addrs=[1, 2, 3])
        d = [SimpleNamespace(kind=tp.FLAT, nbytes=4096, src_off=0, dst_off=0, param_name="w",
                             tag="t", src_ptr=1, dst_ptr=1)]
        why = bl.run_bar1_units(d, _Ops(), lanes=lanes, lane_key="p0", role="dst", seq="1-t",
                                phase="collect", budget_s=1.0, log=lambda *_a: None)
        self.assertIn("slot table of 3 != ring 4", why)

    def test_contiguous_ring_unchanged(self):
        self.assertIsNone(bl.Bar1Lanes("n", "D", 1, 0, ((0, 1),), log=lambda *_a: None,
                                       root="/tmp").slot_table("p0", "dst"))


class TestReceiverServesEveryRegion(CustomTestCase):
    def test_open_window_exports_and_sends_each_region(self):
        src = open(bl.__file__, encoding="utf-8").read()
        body = src[src.index("    def open_window"):]
        body = body[:body.index("\n    def ", 10)]
        self.assertIn("borrow_plan_regions(", body)
        self.assertIn("extra_fds.append(", body)
        self.assertIn("[w.dmabuf_fd] + list(w.extra_fds)", body)
        self.assertIn('"regions": regions_meta', body)

    def test_connect_peer_maps_every_served_region_and_closes_every_fd(self):
        src = open(bl.__file__, encoding="utf-8").read()
        body = src[src.index("    def connect_peer"):]
        body = body[:body.index("\n    def ", 10)]
        self.assertIn('meta.get("regions")', body)
        self.assertIn("for _fd in fds:", body)
        self.assertIn("_drop_extras(cu)", body)
        self.assertIn("slot_addrs=slot_addrs", body)


class TestPreplanBeforeTurn(CustomTestCase):
    """PREPLAN-BEFORE-TURN: the collector's preplan (pointer probes, copy list,
    batch arrays) is built before it waits for the lane's turn, so it overlaps
    the previous tag's collect instead of the depositor's first-free wait."""

    def _collect(self, env):
        events = []

        class Lanes(bl.Bar1Lanes):
            def take_turn(self, lane_key, order_key, timeout_s):
                events.append(("turn-begin", time.perf_counter()))
                time.sleep(0.15)                      # the previous tag's collect
                events.append(("turn-end", time.perf_counter()))
                return True

        real = bl._collect_plan

        def spy(*a, **k):
            events.append(("plan", time.perf_counter()))
            return real(*a, **k)

        bl._RB_SCRATCH.clear()
        slot, ring, n = 4096, 4, 6
        win = bytearray(slot * ring)
        root = tempfile.mkdtemp(prefix="l3x80pp")
        pairs = ((0, 1),)
        dep = bl.Bar1Lanes("n2", "P", 0, 0, pairs, log=lambda *_a: None, root=root)
        col = Lanes("n2", "D", 1, 0, pairs, log=lambda *_a: None, root=root)
        col.recv["p0"] = SimpleNamespace(dptr=_addr(win), slot_bytes=slot, ring=ring)
        dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(win), slot_bytes=slot, ring=ring)
        srcs = [bytearray(os.urandom(slot)) for _ in range(n)]
        dsts = [bytearray(slot) for _ in range(n)]
        descs = [SimpleNamespace(kind=tp.FLAT, nbytes=slot, src_off=0, dst_off=0,
                                 param_name=f"w{i}", tag="weights_6",
                                 src_ptr=_addr(srcs[i]), dst_ptr=_addr(dsts[i]))
                 for i in range(n)]
        out, logs = {}, []
        with mock.patch.dict("os.environ", env), mock.patch.object(bl, "_collect_plan", spy):
            def collect():
                out["c"] = bl.run_bar1_units(descs, _Ops(), lanes=col, lane_key="p0", role="dst",
                                             seq="17-weights_6", phase="collect", budget_s=10.0,
                                             log=logs.append, order_key=(17, 4))

            th = threading.Thread(target=collect)
            th.start()
            out["d"] = bl.run_bar1_units(descs, _Ops(), lanes=dep, lane_key="p0", role="src",
                                         seq="17-weights_6", phase="deposit", budget_s=10.0,
                                         log=lambda *_a: None)
            th.join(20)
        self.assertEqual(out, {"c": "", "d": ""})
        self.assertTrue(all(bytes(a) == bytes(b) for a, b in zip(srcs, dsts)))
        line = [x for x in logs if "PDFLIP-BAR1 lane-time" in x][0]
        return dict(events), line

    def test_preplan_runs_before_the_turn(self):
        ev, line = self._collect({})
        self.assertLess(ev["plan"], ev["turn-begin"])
        self.assertIn("prep_at=pre-turn", line)
        turn_ms = float(re.search(r"turn_ms=(\d+)", line).group(1))
        self.assertGreaterEqual(turn_ms, 140.0)

    def test_switch_off_builds_it_after_the_turn(self):
        ev, line = self._collect({bl.ENV_PREPLAN_BEFORE_TURN: "0"})
        self.assertGreater(ev["plan"], ev["turn-end"])
        self.assertIn("prep_at=turn", line)

    def test_default_on(self):
        self.assertTrue(bl.preplan_before_turn_on({}))
        self.assertFalse(bl.preplan_before_turn_on({bl.ENV_PREPLAN_BEFORE_TURN: "0"}))
        self.assertFalse(bl.preplan_before_turn_on({bl.ENV_COLLECT_PREPLAN: "0"}))


def _pair_run(col_ops=None, dep_ops=None, env=None, n=6, lanes_cls=None, budget_s=10.0, seq="17-weights_7",
              on_send=None):
    """Depositor + collector over one contiguous 4-slot window in one process."""
    bl._RB_SCRATCH.clear()
    slot, ring = 4096, 4
    win = bytearray(slot * ring)
    root = tempfile.mkdtemp(prefix="l3x80i")
    pairs = ((0, 1),)
    dep = bl.Bar1Lanes("n3", "P", 0, 0, pairs, log=lambda *_a: None, root=root)
    col = (lanes_cls or bl.Bar1Lanes)("n3", "D", 1, 0, pairs, log=lambda *_a: None, root=root)
    col.recv["p0"] = SimpleNamespace(dptr=_addr(win), slot_bytes=slot, ring=ring)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(win), slot_bytes=slot, ring=ring)
    srcs = [bytearray(os.urandom(slot)) for _ in range(n)]
    dsts = [bytearray(slot) for _ in range(n)]
    descs = [SimpleNamespace(kind=tp.FLAT, nbytes=slot, src_off=0, dst_off=0,
                             param_name=f"w{i}", tag="weights_7",
                             src_ptr=_addr(srcs[i]), dst_ptr=_addr(dsts[i])) for i in range(n)]
    out, clog, dlog = {}, [], []

    def _hooked(orig):
        def send(self, kind, g=0, payload=None):
            if on_send is not None and on_send(kind, srcs, payload) is False:
                return None                 # the hook swallows this record
            return orig(self, kind, g, payload)
        return send

    with mock.patch.dict("os.environ", env or {}), \
            mock.patch.object(bl.FileCredits, "send", _hooked(bl.FileCredits.send)), \
            mock.patch.object(bl.SocketCredits, "send", _hooked(bl.SocketCredits.send)):
        def collect():
            out["c"] = bl.run_bar1_units(descs, col_ops or _Ops(), lanes=col, lane_key="p0",
                                         role="dst", seq=seq, phase="collect", budget_s=budget_s,
                                         log=clog.append, order_key=(17, 6))

        th = threading.Thread(target=collect)
        th.start()
        out["d"] = bl.run_bar1_units(descs, dep_ops or _Ops(), lanes=dep, lane_key="p0", role="src",
                                     seq=seq, phase="deposit", budget_s=budget_s, log=dlog.append)
        th.join(30)
    return out, srcs, dsts, clog, dlog


class TestLaneIntegrity(CustomTestCase):
    """LANE-INTEGRITY: sampled crc32 of the depositor's source against the
    collector's destination, per tag; a mismatch is a named refusal."""

    def test_clean_tag_is_verified_and_counted_per_flip(self):
        bl._INTEG.clear()
        out, srcs, dsts, clog, dlog = _pair_run(n=10)
        self.assertEqual(out, {"c": "", "d": ""})
        cl = [x for x in clog if "lane-time" in x][0]
        dl = [x for x in dlog if "lane-time" in x][0]
        self.assertIn("integrity=verified:8/8 skipped=0", cl)
        self.assertIn("integrity=sent:8", dl)
        line = bl.integrity_flip_line(17)
        self.assertIn("flip=17 tags=1 sampled=8 verified=8 skipped_no_write=0 mismatch=0", line)
        self.assertIsNone(bl.integrity_flip_line(17))          # consumed

    def test_corrupted_copy_out_is_a_named_refusal(self):
        bl._INTEG.clear()

        class Corrupt(_Ops):
            def memcpy_async(self, dst, src, nb, stream):
                ctypes.memmove(dst, src, nb)
                if nb == 4096 and not getattr(self, "_done", False):
                    self._done = True                       # first copy-out: one byte flipped
                    b = (ctypes.c_ubyte * 1).from_address(dst + 100)
                    b[0] ^= 0x5A

        out, _s, _d, clog, _dl = _pair_run(col_ops=Corrupt(), n=10)
        self.assertEqual(out["d"], "")                     # the depositor got its done
        self.assertIn("INTEGRITY MISMATCH", out["c"])
        self.assertIn("source crc32", out["c"])
        self.assertTrue(any("PDFLIP-BAR1-INTEGRITY MISMATCH" in x for x in clog))
        self.assertIn("mismatch=1", bl.integrity_flip_line(17))

    def test_off_is_named_and_not_waited_for(self):
        out, srcs, dsts, clog, dlog = _pair_run(env={bl.ENV_INTEGRITY_SAMPLES: "0"})
        self.assertEqual(out, {"c": "", "d": ""})
        self.assertTrue(all(bytes(a) == bytes(b) for a, b in zip(srcs, dsts)))
        self.assertIn("integrity=not-announced", [x for x in clog if "lane-time" in x][0])
        self.assertIn("integrity=OFF why=", [x for x in dlog if "lane-time" in x][0])

    def test_plan_is_deterministic_and_inside_the_pieces(self):
        pcs = [SimpleNamespace(kind=tp.FLAT, nbytes=10000, src_off=0, dst_off=0),
               SimpleNamespace(kind=tp.STRIDED2D, nbytes=6 * 512, rows=6, run_bytes=512,
                               spitch=4096, dpitch=1024, src_off=0, dst_off=0)]
        pcs = pcs + [SimpleNamespace(kind=tp.FLAT, nbytes=5000 + k, src_off=k, dst_off=k)
                     for k in range(8)]
        batches = [SimpleNamespace(pieces=pcs)]
        a = bl.integrity_plan(batches, "9-weights_3", 8, tp)
        self.assertEqual(a, bl.integrity_plan(batches, "9-weights_3", 8, tp))
        self.assertEqual(len(a), 8)
        self.assertEqual(a[0][:2], (0, 0))
        self.assertEqual(a[1][:2], (0, len(pcs) - 1))
        for g, i, row, off, n in a:
            pc = pcs[i]
            span = pc.nbytes if pc.kind == tp.FLAT else pc.run_bytes
            self.assertLessEqual(off + n, span)
            self.assertLess(row, getattr(pc, "rows", 1))
        self.assertEqual(bl.integrity_samples({}), 8)
        self.assertEqual(bl.integrity_samples({bl.ENV_INTEGRITY_SAMPLES: "0"}), 0)

    def test_picks_are_distinct_pieces_and_capped(self):
        """RISK 2: no piece is sampled twice; n is capped at the piece count."""
        pcs = [SimpleNamespace(kind=tp.FLAT, nbytes=9000, src_off=0, dst_off=0) for _ in range(12)]
        batches = [SimpleNamespace(pieces=pcs[:6]), SimpleNamespace(pieces=pcs[6:])]
        for seq in ("1-weights_1", "2-weights_1", "9-weights_3", "31-weights_7", "77-w"):
            a = bl.integrity_plan(batches, seq, 8, tp)
            self.assertEqual(len(a), 8)
            self.assertEqual(len({(g, i) for g, i, *_ in a}), 8, a)
            self.assertEqual({a[0][:2], a[1][:2]}, {(0, 0), (1, 5)})
            self.assertEqual(a, bl.integrity_plan(batches, seq, 8, tp))      # deterministic
        a = bl.integrity_plan(batches, "5-w", 50, tp)
        self.assertEqual(sorted((g, i) for g, i, *_ in a),
                         [(g, i) for g in (0, 1) for i in range(6)])      # all 12, once each
        one = bl.integrity_plan([SimpleNamespace(pieces=pcs[:1])], "5-w", 8, tp)
        self.assertEqual([x[:2] for x in one], [(0, 0)])

    def test_source_written_after_the_plan_is_no_false_mismatch(self):
        """RISK 1: the crcs are taken from the source as it is at copy time,
        after the plan send -- a write in the plan/credit window is not a
        corrupted copy."""
        bl._INTEG.clear()

        def on_send(kind, srcs, payload):
            if kind == "plan":
                for b in srcs:
                    for k in range(0, len(b), 97):
                        b[k] ^= 0xFF           # every source byte-ish changes after the plan
            return True

        out, srcs, dsts, clog, dlog = _pair_run(n=10, on_send=on_send)
        self.assertEqual(out, {"c": "", "d": ""})
        self.assertTrue(all(bytes(a) == bytes(b) for a, b in zip(srcs, dsts)))
        self.assertIn("integrity=verified:8/8 skipped=0", [x for x in clog if "lane-time" in x][0])
        self.assertIn("mismatch=0", bl.integrity_flip_line(17))

    def test_plan_announces_the_sum_before_any_crc_is_read(self):
        seen = {}

        class Spy(_Ops):
            def memcpy_async(self, dst, src, nb, stream):
                seen.setdefault("first_read_after_plan", seen.get("plan", False))
                ctypes.memmove(dst, src, nb)

        def on_send(kind, srcs, payload):
            if kind == "plan":
                seen["plan"] = True
                seen["announced"] = bool(payload.get("sum"))
            return True

        out, *_ = _pair_run(n=10, dep_ops=Spy(), on_send=on_send)
        self.assertEqual(out, {"c": "", "d": ""})
        self.assertTrue(seen["announced"])
        self.assertTrue(seen["first_read_after_plan"])

    def test_announced_sum_that_never_arrives_is_a_named_refusal(self):
        """The collector must not hang forever nor pass silently: after its
        (short) budget it refuses the tag by name."""
        bl._INTEG.clear()
        dropped = []

        def on_send(kind, srcs, payload):
            if kind == "sum":
                dropped.append(payload)
                return False
            return True

        t0 = time.perf_counter()
        out, _s, _d, clog, _dl = _pair_run(n=10, on_send=on_send, budget_s=1.0)
        self.assertEqual(len(dropped), 1)                 # it was announced AND sent -- but lost
        self.assertIn("the announced integrity record did not come within", out["c"])
        self.assertIn("seq=17-weights_7", out["c"])
        self.assertLess(time.perf_counter() - t0, 25)
        self.assertIsNone(bl.integrity_flip_line(17))      # never counted as verified


class TestPreplanRecheck(CustomTestCase):
    """PREPLAN-RECHECK: a destination that stops being mapped device memory
    while the collector waits for the lane's turn is refused by name."""

    def _run(self, env):
        state = {"after_turn": False}

        class Lanes(bl.Bar1Lanes):
            def take_turn(self, lane_key, order_key, timeout_s):
                time.sleep(0.05)
                state["after_turn"] = True
                return True

        class Ops(_Ops):
            def ptr_attrs(self, a):
                return (0, 0, -1) if state["after_turn"] else (0, 2, 0)

        return _pair_run(col_ops=Ops(), lanes_cls=Lanes, env=env, budget_s=2.0)

    def test_a_destination_unmapped_during_the_turn_is_refused(self):
        out, _s, _d, _c, _dl = self._run({})
        self.assertIn("is not mapped device memory", out["c"])

    def test_without_the_recheck_the_stale_plan_would_run(self):
        out, _s, _d, clog, _dl = self._run({bl.ENV_PREPLAN_RECHECK_MS: "100000"})
        self.assertEqual(out["c"], "")
        self.assertIn("prep_recheck=-", [x for x in clog if "lane-time" in x][0])

    def test_knob(self):
        self.assertEqual(bl.preplan_recheck_ms({}), 0.0)
        self.assertEqual(bl.preplan_recheck_ms({bl.ENV_PREPLAN_RECHECK_MS: "x"}), 0.0)


if __name__ == "__main__":
    unittest.main()
