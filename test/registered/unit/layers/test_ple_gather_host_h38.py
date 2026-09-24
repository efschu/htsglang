"""fnFL2 H38: PLE-GATHER-HOST, the host split of the PP0 PLE pread gather.

x144 (24.09.): the same 97841-token request took 38.53 s against 35.86 s in
x143; the whole difference sits on PP0 in the PLE pread gather of chunks 5
and 6 (3107.9 / 1668.6 ms against 1131.7 / 1164.5 ms), and PLE-GATHER-PREFILL
cannot say whether the host stole the cores, the ARC missed, or the stream
sync in front of it grew. The cases pin what the next boot's reading needs:

* the parsers read the procfs formats of this rig (incl. a comm with ') ' in
  it) and degrade to -1 instead of raising when a file is missing;
* foreign_cores = host busy minus this process, clamped at 0;
* the worker split telescopes: cpu + runq + blocked == thr_wall;
* prep + read + copy == wall of the gather (by construction);
* the gather returns the same bytes with the switch on and off, and the
  switch off emits no PLE-GATHER-HOST line (cost zero).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import logging
import os
import re
import tempfile
import threading
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import host_contention as hc
from sglang.srt.layers import prefill_timing as pt
from sglang.srt.models import qwen4_exp_ple_table as ple
from sglang.test.test_utils import CustomTestCase

_STAT = (
    "cpu  100 20 30 1000 50 5 5 10 0 0\n"
    "cpu0 50 10 15 500 25 2 2 5 0 0\n"
    "intr 12345\n"
)
_SELF = "4242 (sglang::sch) (x) R 1 2 3 4 5 6 7 8 9 10 70 30 0 0 20 0 33 0\n"
_PSI = "some avg10=0.00 avg60=0.28 avg300=0.38 total=1543408\nfull avg10=0.00 avg60=0.07 avg300=0.03 total=290955\n"
_ARC = (
    "13 1 0x01 147 39984 1 2\n"
    "name                            type data\n"
    "hits                            4    999\n"
    "demand_data_hits                4    33996294833\n"
    "demand_data_misses              4    780859373\n"
)


class TestParsers(CustomTestCase):
    def test_proc_stat_busy_and_total(self):
        busy, total = hc.parse_proc_stat(_STAT)
        # user+nice+system+irq+softirq+steal
        self.assertEqual(busy, 100 + 20 + 30 + 5 + 5 + 10)
        self.assertEqual(total, busy + 1000 + 50)

    def test_self_stat_counts_from_last_paren(self):
        # comm "(sglang::sch) (x)" holds ") (" -- utime 70 + stime 30
        self.assertEqual(hc.parse_self_stat(_SELF), 100)

    def test_psi_and_arc_and_schedstat(self):
        self.assertEqual(hc.parse_psi_some_total(_PSI), 1543408)
        self.assertEqual(hc.parse_arcstats(_ARC), (33996294833, 780859373))
        self.assertEqual(hc.parse_schedstat("112290 7000 1\n"), (112290, 7000))

    def test_missing_files_degrade_to_minus_one(self):
        self.assertEqual(hc.parse_proc_stat(None), (-1, -1))
        self.assertEqual(hc.parse_self_stat(None), -1)
        self.assertEqual(hc.parse_psi_some_total(None), -1)
        self.assertEqual(hc.parse_arcstats(None), (-1, -1))
        self.assertEqual(hc.parse_schedstat(None), (-1, -1))
        nowhere = hc.ProcPaths(*(["/nonexistent/h38"] * 6))
        s = hc.sample(nowhere)
        self.assertEqual((s.host_busy, s.self_cpu, s.psi_cpu_us, s.arc_hits), (-1, -1, -1, -1))
        d = hc.delta(s, hc.sample(nowhere))
        self.assertEqual((d.host_busy_cores, d.foreign_cores, d.arc_misses), (-1.0, -1.0, -1))
        self.assertEqual(hc.thread_sched(nowhere), (-1, -1))

    def test_real_procfs_reads(self):
        if not os.path.exists("/proc/stat"):
            self.skipTest("no procfs")
        a = hc.sample()
        b = hc.sample()
        self.assertGreaterEqual(b.host_busy, a.host_busy)
        self.assertGreater(a.host_total, 0)
        self.assertGreaterEqual(a.self_cpu, 0)
        cpu, runq = hc.thread_sched()
        self.assertGreater(cpu, 0)
        self.assertGreaterEqual(runq, 0)


class TestDelta(CustomTestCase):
    def _s(self, t, busy, own, psi_cpu=0, psi_io=0, hits=0, misses=0):
        return hc.HostSample(t, busy, busy * 2, own, psi_cpu, psi_io, hits, misses)

    def test_foreign_cores(self):
        tck = hc._CLK_TCK
        a = self._s(10.0, 1000, 100)
        b = self._s(12.0, 1000 + int(10 * tck), 100 + int(4 * tck), 3000, 500, 70, 5)
        d = hc.delta(a, b)
        self.assertAlmostEqual(d.wall_ms, 2000.0)
        self.assertAlmostEqual(d.host_busy_cores, 5.0)
        self.assertAlmostEqual(d.self_cores, 2.0)
        self.assertAlmostEqual(d.foreign_cores, 3.0)
        self.assertAlmostEqual(d.psi_cpu_ms, 3.0)
        self.assertAlmostEqual(d.psi_io_ms, 0.5)
        self.assertEqual((d.arc_hits, d.arc_misses), (70, 5))

    def test_foreign_never_negative(self):
        # /proc/stat and /proc/self/stat tick independently: own > busy happens
        tck = hc._CLK_TCK
        d = hc.delta(self._s(0.0, 0, 0), self._s(1.0, int(2 * tck), int(3 * tck)))
        self.assertEqual(d.foreign_cores, 0.0)


class TestThreadSplit(CustomTestCase):
    def test_telescopes_and_names_lead(self):
        s = hc.ThreadSplit()
        s.add(1000, 100, 800)
        s.add(1000, 100, 700)
        self.assertEqual(s.tasks, 2)
        self.assertEqual(s.cpu_ns + s.runq_ns + s.blocked_ns, s.wall_ns)
        self.assertEqual(s.lead(), "runq")

    def test_unknown_task_poisons_blocked(self):
        s = hc.ThreadSplit()
        s.add(1000, -1, -1)
        self.assertEqual(s.blocked_ns, -1)
        self.assertEqual(s.lead(), "unknown")

    def test_timed_task_runs_fn(self):
        box = []
        wall, cpu, runq = hc.timed_task(box.append, 7)
        self.assertEqual(box, [7])
        self.assertGreaterEqual(wall, 0)


class TestHostPeriod(CustomTestCase):
    """PLE-HOST-PERIOD rides log_ple_gather, so the H32 prefetch path (which
    logs through it and reads in worker processes) carries it too."""

    def setUp(self):
        pt._PERIOD.last = None

    def _lines(self, on, calls):
        with envs.SGLANG_MOE_OFFLOAD_TIMING.override(on), self.assertLogs(pt.logger, level=logging.INFO) as cm:
            pt.logger.info("sentinel")
            for _ in range(calls):
                pt.log_ple_gather(262144, 0, 1.1, 32)
        return [r.getMessage() for r in cm.records]

    def test_first_gather_opens_second_reports(self):
        lines = self._lines(True, 3)
        per = [l for l in lines if l.startswith("PLE-HOST-PERIOD ")]
        self.assertEqual(len([l for l in lines if l.startswith("PLE-GATHER-PREFILL ")]), 3)
        self.assertEqual(len(per), 2)  # a period needs two ends
        f = dict(re.findall(r"(\w+)=(-?[\d.]+)", per[0]))
        self.assertEqual(int(f["rows"]), 262144)
        self.assertGreaterEqual(float(f["period_ms"]), 0.0)
        for k in ("host_busy_cores", "self_cores", "foreign_cores", "psi_cpu_ms", "arc_misses"):
            self.assertIn(k, f)

    def test_switch_off_no_line_no_state(self):
        lines = self._lines(False, 2)
        self.assertFalse(any(l.startswith("PLE-") for l in lines))
        self.assertIsNone(pt._PERIOD.last)

    def test_period_line_carries_the_delta(self):
        tck = hc._CLK_TCK
        a = hc.HostSample(0.0, 0, 0, 0, 0, 0, 0, 0)
        b = hc.HostSample(4.5, int(9 * tck), int(20 * tck), int(4.5 * tck), 9000, 0, 100, 7)
        line = pt.format_ple_host_period(262144, hc.delta(a, b))
        self.assertIn("period_ms=4500.0", line)
        self.assertIn("foreign_cores=1.00", line)
        self.assertIn("psi_cpu_ms=9.0", line)
        self.assertIn("arc_misses=7", line)


def _table(tmpdir, rows_per_shard=64, shards=3, dim=8):
    """A real on-disk table: 3 shards in 2 files, row r = r in every column."""
    dtype = torch.bfloat16
    rb = dim * 2
    files = [os.path.join(tmpdir, "a.bin"), os.path.join(tmpdir, "b.bin")]
    shard_files = [files[0], files[0], files[1]]
    shard_offsets = [0, rows_per_shard * rb, 4096]
    blobs = {files[0]: bytearray(2 * rows_per_shard * rb), files[1]: bytearray(4096 + rows_per_shard * rb)}
    for s in range(shards):
        rows = torch.arange(s * rows_per_shard, (s + 1) * rows_per_shard, dtype=torch.float32)
        data = rows[:, None].expand(-1, dim).to(dtype).contiguous().view(torch.uint8).numpy().tobytes()
        o = shard_offsets[s]
        blobs[shard_files[s]][o : o + len(data)] = data
    for p, b in blobs.items():
        with open(p, "wb") as f:
            f.write(bytes(b))
    return ple.CheckpointMappedPleTable(
        bases=[0] * shards,
        shard_rows=rows_per_shard,
        total_rows=rows_per_shard * shards,
        dtype=dtype,
        embedding_dim=dim,
        keepalive=[],
        files=files,
        shard_files=shard_files,
        shard_offsets=shard_offsets,
    )


class TestGatherLine(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = _table(self.tmp.name)
        self.g = ple.PleCheckpointPreadGather(self.table, min_rows=1, workers=4)
        gen = torch.Generator().manual_seed(38)
        self.ids = torch.randint(0, 200, (1500,), generator=gen)  # >= 192: some out of range

    def tearDown(self):
        self.g.close()
        self.tmp.cleanup()

    def _expected(self, vocab_end):
        ok = (self.ids < vocab_end) & (self.ids < self.table.total_rows)
        return torch.where(ok, self.ids, torch.zeros_like(self.ids)).to(torch.float32)[:, None].expand(-1, 8).to(torch.bfloat16)

    def _run(self, on):
        out = torch.empty((self.ids.numel(), 8), dtype=torch.bfloat16)
        with envs.SGLANG_MOE_OFFLOAD_TIMING.override(on), self.assertLogs(pt.logger, level=logging.INFO) as cm:
            pt.logger.info("sentinel")
            self.g.gather_into(self.ids, out, vocab_start=0, vocab_end=150)
        return out, [r.getMessage() for r in cm.records]

    def test_switch_on_emits_split_that_telescopes(self):
        out, lines = self._run(True)
        self.assertTrue(torch.equal(out, self._expected(150)))
        host = [l for l in lines if l.startswith("PLE-GATHER-HOST ")]
        wall = [l for l in lines if l.startswith("PLE-GATHER-PREFILL ")]
        self.assertEqual((len(host), len(wall)), (1, 1))
        f = dict(re.findall(r"(\w+)=(-?[\w.]+)", host[0]))
        self.assertEqual(int(f["rows"]), 1500)
        # prep + read + copy is the gather's wall by construction
        self.assertAlmostEqual(
            float(f["prep_ms"]) + float(f["read_ms"]) + float(f["copy_ms"]), float(f["wall_ms"]), delta=0.3
        )
        # every worker task was timed, and its terms add up to its wall
        valid = int(((self.ids >= 0) & (self.ids < 150)).sum())
        chunk = max(256, (valid + 3) // 4)
        self.assertEqual(int(f["tasks"]), -(-valid // chunk))
        if f["lead"] != "unknown":
            self.assertAlmostEqual(
                float(f["thr_cpu_ms"]) + float(f["thr_runq_ms"]) + float(f["thr_blocked_ms"]),
                float(f["thr_wall_ms"]),
                delta=0.3,
            )
        # PLE-GATHER-PREFILL keeps its shape for every existing reader
        self.assertRegex(wall[0], r"^PLE-GATHER-PREFILL rows=1500 zero_rows=\d+ ms=[\d.]+ workers=4 ")

    def test_switch_off_same_bytes_no_host_line(self):
        out, lines = self._run(False)
        self.assertTrue(torch.equal(out, self._expected(150)))
        self.assertFalse(any(l.startswith("PLE-GATHER") for l in lines))

    def test_runq_reaches_the_line(self):
        # a contended worker: 5 ms on a core, 40 ms runnable without one
        local = threading.local()

        def fake(paths=hc.DEFAULT_PATHS):
            # per worker thread: first read before the task, second after it
            local.n = getattr(local, "n", 0) + 1
            return (10_000_000, 1_000_000) if local.n % 2 else (15_000_000, 41_000_000)

        with mock.patch.object(hc, "thread_sched", fake):
            _, lines = self._run(True)
        f = dict(re.findall(r"(\w+)=(-?[\w.]+)", [l for l in lines if l.startswith("PLE-GATHER-HOST ")][0]))
        tasks = int(f["tasks"])
        self.assertGreater(tasks, 0)
        self.assertAlmostEqual(float(f["thr_cpu_ms"]), 5.0 * tasks)
        self.assertAlmostEqual(float(f["thr_runq_ms"]), 40.0 * tasks)


if __name__ == "__main__":
    unittest.main()
