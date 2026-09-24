"""fnFL2 H43 (SGLANG_QWEN4_PLE_PREFETCH_ADMIT): the first chunk's PLE read from
the request's admission.

H32 reads chunk n+1 while chunk n computes; chunk 0 was always read inside its
own forward (x146: 4.47 s cold for the 12.6k code prompt). The PP0 scheduler
now hands each new request to the admission on intake (and the Weg-2 front
hints a request P will prefill while P is still asleep); the first chunk's
rows are read on the H32 worker processes into a third slot while the ring is
idle, and the request's first forward joins that read as H32 joins its own.

Desk only (CPU, real worker processes with an injected read delay, real
files, a fake admission clock): the admission starts the read, the first
forward joins it with ~0 wait and bytes equal to the serial gather, the
running request's ring goes first and the admission starts at its last chunk,
a read in flight is joined before any other gather submits, an abort or a
stale admission frees the slot, the switch off is H32 unchanged, group D does
not admit, the scheduler / HTTP / front hooks are wired.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import ast
import asyncio
import inspect
import logging
import re
import tempfile
import time
import types
import unittest
from array import array
from collections import namedtuple
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.models import qwen4_exp_ple_admit as adm
from sglang.srt.models import qwen4_exp_ple_prefetch as pf
from sglang.srt.models import qwen4_exp_ple_table as pt
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

Range = namedtuple("Range", "start end")
ROWS_PER_SHARD = 64
SHARDS = 4
DIM = 160
RB = DIM * 2
HEADER = 100
PF_LOGGER = "sglang.srt.models.qwen4_exp_ple_prefetch"
ADM_LOGGER = "sglang.srt.models.qwen4_exp_ple_admit"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRT = os.path.join(ROOT, "python", "sglang", "srt")


def _table(tmpdir):
    files, shard_files, shard_offsets = [], [], []
    for f in range(2):
        path = os.path.join(tmpdir, f"ple-{f}.safetensors")
        with open(path, "wb") as fh:
            fh.write(os.urandom(HEADER + 2 * ROWS_PER_SHARD * RB))
        files.append(path)
        for s in range(2):
            shard_files.append(path)
            shard_offsets.append(HEADER + s * ROWS_PER_SHARD * RB)
    return pt.CheckpointMappedPleTable(
        [0] * SHARDS, ROWS_PER_SHARD, ROWS_PER_SHARD * SHARDS, torch.bfloat16, DIM,
        keepalive=[], files=files, shard_files=shard_files, shard_offsets=shard_offsets,
    )


def _identity_hasher(tokens, lead):
    """One row per token, row id = token (the model's hasher is H32's test)."""
    return tokens[lead:].clone()


def _serial(table, ids, vocab=(0, None)):
    base = pt.PleCheckpointPreadGather(table, min_rows=16, workers=4)
    out = torch.empty((ids.numel(), DIM), dtype=torch.bfloat16)
    base.gather_into(ids, out, vocab_start=vocab[0], vocab_end=vocab[1])
    base.close()
    return out


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _chunks(records):
    out = []
    for r in records:
        m = re.search(r"PLE-PREFETCH chunk=(\d+) rows=(\d+) ready=(\w+) wait_ms=([\d.]+) "
                      r"gather_ms=([\d.]+) hit_rows=(\d+) read_rows=(\d+)", r.getMessage())
        if m:
            out.append(dict(chunk=int(m[1]), rows=int(m[2]), ready=m[3], wait_ms=float(m[4]),
                            gather_ms=float(m[5]), hit=int(m[6]), read=int(m[7])))
    return out


def _admit_lines(records):
    return [r.getMessage() for r in records if "PLE-PREFETCH admit" in r.getMessage()]


class _Req:
    """A request as the scheduler holds it (ids known from intake)."""

    def __init__(self, rid, ids):
        self.rid = rid
        self.ids = ids
        self.origin_input_ids = ids.tolist()
        self.full_untruncated_fill_ids = array("q", self.origin_input_ids)
        self.extend_range = None


class _Rig:
    """The PP0 side: the admitting gather, and batches driven like
    ``Scheduler._run_batch_forward`` + the PLE layer drive them."""

    def __init__(self, table, *, chunk=200, delay_s=0.0, procs=2, hasher=_identity_hasher):
        self.table, self.chunk = table, chunk
        self.g = adm.PleAdmitPrefetchGather(
            pt.PleCheckpointPreadGather(table, min_rows=16, workers=4), table, hasher,
            procs=procs, threads=4, delay_s=delay_s,
        )

    def batch(self, parts):
        """parts: [(req, start, end)] -> one extend forward; returns (ids, out)."""
        for req, start, end in parts:
            req.extend_range = Range(start, end)
        reqs = [p[0] for p in parts]
        pf.publish_ple_next_chunk(reqs, self.chunk)
        adm.note_ple_batch(reqs)
        ids = torch.cat([req.ids[s:e] for req, s, e in parts])
        out = torch.empty((ids.numel(), DIM), dtype=torch.bfloat16)
        self.g.gather_into(ids, out)
        return ids, out

    def chunk_of(self, req, c):
        n = len(req.origin_input_ids)
        return self.batch([(req, c * self.chunk, min((c + 1) * self.chunk, n))])

    def warm(self, tc):
        """One prefill gather in this process (vocab range + hash constants)."""
        w0 = _Req("warm", torch.randint(0, 256, (100,)))
        ids, out = self.chunk_of(w0, 0)
        tc.assertTrue(_same(out, _serial(self.table, ids)))

    def close(self):
        self.g.close()


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class TestAdmission(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = _table(self.tmp.name)
        self.gen = torch.Generator().manual_seed(11)
        self.clock = _Clock()
        self._p = mock.patch.object(adm, "_clock", self.clock)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self.tmp.cleanup()
        self.assertEqual(adm._SINKS, [])  # every rig closed its sink

    def _ids(self, n, hi=300):
        return torch.randint(0, hi, (n,), generator=self.gen)

    def test_intake_reads_the_first_chunk_and_its_forward_joins(self):
        rig = _Rig(self.table, delay_s=0.3)
        try:
            rig.warm(self)
            w = _Req("weg2-2-6", self._ids(150))
            with self.assertLogs(ADM_LOGGER, logging.INFO) as ca, \
                    self.assertLogs(PF_LOGGER, logging.INFO) as cp:
                self.assertEqual(adm.admit_ple_request(w, rig.chunk), "started")
                self.assertIsNotNone(rig.g._adm)
                self.assertFalse(rig.g._workers.ready(rig.g._adm.seq))  # reading, not waited on
                time.sleep(0.45)  # P schedules / wakes; the read finishes meanwhile
                self.clock.t += 2.5
                t = time.monotonic()
                ids, out = rig.chunk_of(w, 0)
                fwd_s = time.monotonic() - t
            self.assertTrue(_same(out, _serial(self.table, ids)))
            self.assertLess(fwd_s, 0.2)  # the forward did not pay the 0.3 s read
            line = _chunks(cp.records)[-1]
            self.assertEqual((line["chunk"], line["ready"], line["hit"], line["read"]), (0, "yes", 150, 0))
            self.assertLess(line["wait_ms"], 60.0)
            used = [l for l in _admit_lines(ca.records) if "queued_ms_before_forward" in l]
            self.assertEqual(len(used), 1)
            self.assertIn("rid=weg2-2-6 rows=150 queued_ms_before_forward=2500.0", used[0])
            self.assertIn("ready=yes source=queue dormant=0", used[0])
            self.assertIsNone(rig.g._adm)
            self.assertEqual(rig.g.stats["admit_used"], 1)
            # the admission slot is the third one; the ring stays two deep
            self.assertEqual(len(rig.g._slots), 3)
            self.assertEqual(len(rig.g._workers.slot_fds), 3)
        finally:
            rig.close()

    def test_short_forward_waits_only_the_rest_of_the_read(self):
        rig = _Rig(self.table, delay_s=0.4)
        try:
            rig.warm(self)
            w = _Req("w", self._ids(150))
            with self.assertLogs(PF_LOGGER, logging.INFO) as cp:
                adm.admit_ple_request(w, rig.chunk)
                time.sleep(0.2)
                ids, out = rig.chunk_of(w, 0)
            self.assertTrue(_same(out, _serial(self.table, ids)))
            line = _chunks(cp.records)[-1]
            self.assertEqual((line["ready"], line["hit"], line["read"]), ("no", 150, 0))
            self.assertGreater(line["wait_ms"], 100.0)
            self.assertLess(line["wait_ms"], 350.0)
        finally:
            rig.close()

    def test_multi_chunk_request_continues_on_the_ring(self):
        rig = _Rig(self.table)
        try:
            rig.warm(self)
            w = _Req("w", self._ids(700))
            adm.admit_ple_request(w, rig.chunk)
            time.sleep(0.1)
            with self.assertLogs(PF_LOGGER, logging.INFO) as cp:
                for c in range(4):
                    ids, out = rig.chunk_of(w, c)
                    self.assertTrue(_same(out, _serial(self.table, ids)), f"chunk {c}")
            lines = _chunks(cp.records)
            self.assertEqual([l["chunk"] for l in lines], [0, 1, 2, 3])
            for l in lines:
                self.assertEqual(l["read"], 0)
            # chunk 0 from the admission slot, then the H32 ring (slots 0/1)
            self.assertIsNone(rig.g._pending)
        finally:
            rig.close()

    def test_first_intake_of_a_process_is_cold_and_skipped(self):
        called = []

        def hasher(tokens, lead):
            called.append(1)
            return tokens[lead:].clone()

        hasher.ple_hash_ready = lambda: bool(called)
        hasher.ple_hash_warm = lambda: called.append(0)
        rig = _Rig(self.table, hasher=hasher)
        try:
            w = _Req("w", self._ids(150))
            with self.assertLogs(ADM_LOGGER, logging.INFO) as ca:
                self.assertEqual(adm.admit_ple_request(w, rig.chunk), "skipped:cold")
            self.assertIn("no prefill gather in this process yet", ca.output[0])
            self.assertEqual(called, [])  # never hashed (the device may be asleep)
            ids, out = rig.chunk_of(w, 0)  # served as H32 alone
            self.assertTrue(_same(out, _serial(self.table, ids)))
            self.assertEqual(called[0], 0)  # the gather warmed the constants
            w2 = _Req("w2", self._ids(150))
            self.assertEqual(adm.admit_ple_request(w2, rig.chunk), "started")
        finally:
            rig.close()

    def test_running_request_goes_first_admission_starts_at_its_last_chunk(self):
        rig = _Rig(self.table, delay_s=0.05)
        try:
            rig.warm(self)
            r = _Req("r", self._ids(600))
            w = _Req("w", self._ids(180))
            ring_slots = set()
            with self.assertLogs(ADM_LOGGER, logging.INFO) as ca, \
                    self.assertLogs(PF_LOGGER, logging.INFO) as cp:
                ids, out = rig.chunk_of(r, 0)
                self.assertTrue(_same(out, _serial(self.table, ids)))
                self.assertIsNotNone(rig.g._pending)  # r's chunk 1 is being read
                self.assertEqual(adm.admit_ple_request(w, rig.chunk), "queued")
                self.assertIsNone(rig.g._adm)
                for c in (1, 2):
                    pend = rig.g._pending
                    ring_slots.add(pend.slot)
                    time.sleep(0.1)
                    ids, out = rig.chunk_of(r, c)
                    self.assertTrue(_same(out, _serial(self.table, ids)), f"r chunk {c}")
                    if c == 1:
                        self.assertIn("w", rig.g._adm_queue)  # the ring was busy all along
                # r's last chunk queued no next chunk: w's read started behind it
                self.assertIsNone(rig.g._pending)
                self.assertEqual(rig.g._adm.rid, "w")
                self.assertEqual(ring_slots, {0, 1})
                time.sleep(0.15)
                ids, out = rig.chunk_of(w, 0)
                self.assertTrue(_same(out, _serial(self.table, ids)))
            self.assertEqual(_chunks(cp.records)[-1]["read"], 0)
            self.assertEqual(_chunks(cp.records)[-1]["hit"], 180)
            self.assertTrue(any("rid=w queued source=queue" in l for l in _admit_lines(ca.records)))
            self.assertTrue(any("rid=w rows=180 started" in l for l in _admit_lines(ca.records)))
            self.assertEqual(rig.g._workers.slot_bytes[adm.PLE_ADMIT_SLOT], max(1 << 15, 180) * RB)
        finally:
            rig.close()

    def test_waiting_admission_served_in_a_mixed_batch_is_forgotten(self):
        rig = _Rig(self.table, delay_s=0.05)
        try:
            rig.warm(self)
            r = _Req("r", self._ids(350))
            w = _Req("w", self._ids(180))
            rig.chunk_of(r, 0)
            self.assertEqual(adm.admit_ple_request(w, rig.chunk), "queued")
            time.sleep(0.1)
            with self.assertLogs(ADM_LOGGER, logging.INFO) as ca:
                # r's tail and w's head share one batch: w is served before its read began
                ids, out = rig.batch([(r, 200, 350), (w, 0, 50)])
            self.assertTrue(_same(out, _serial(self.table, ids)))
            self.assertIn("rid=w dropped reason=served_before_its_read_started", "\n".join(ca.output))
            self.assertEqual(len(rig.g._adm_queue), 0)
            self.assertIsNone(rig.g._adm)  # nothing read into the admission slot for nothing
            self.assertEqual(rig.g.stats["admit_started"], 0)
        finally:
            rig.close()

    def test_a_read_in_flight_is_joined_before_another_requests_gather(self):
        rig = _Rig(self.table, delay_s=0.3)
        try:
            rig.warm(self)
            w = _Req("w", self._ids(150))
            x = _Req("x", self._ids(170))
            self.assertEqual(adm.admit_ple_request(w, rig.chunk), "started")
            with self.assertLogs(ADM_LOGGER, logging.INFO) as ca:
                ids, out = rig.chunk_of(x, 0)  # x runs first (not FCFS): no collision
            self.assertTrue(_same(out, _serial(self.table, ids)))
            self.assertTrue(any("rid=w settled before another request's gather" in l
                                for l in _admit_lines(ca.records)))
            self.assertTrue(rig.g._adm.joined)
            with self.assertLogs(PF_LOGGER, logging.INFO) as cp:
                ids, out = rig.chunk_of(w, 0)  # w's rows are still there
            self.assertTrue(_same(out, _serial(self.table, ids)))
            line = _chunks(cp.records)[-1]
            self.assertEqual((line["ready"], line["hit"], line["read"]), ("yes", 150, 0))
        finally:
            rig.close()

    def test_abort_of_the_waiting_request_frees_the_slot(self):
        rig = _Rig(self.table, delay_s=0.3)
        try:
            rig.warm(self)
            w = _Req("weg2-9-1", self._ids(150))
            self.assertEqual(adm.admit_ple_request(w, rig.chunk), "started")
            with self.assertLogs(ADM_LOGGER, logging.INFO) as ca:
                t = time.monotonic()
                self.assertEqual(adm.drop_ple_admission("weg2-9-1"), 1)
                self.assertLess(time.monotonic() - t, 0.1)  # the abort never waits on the read
            self.assertIn("rid=weg2-9-1 dropped reason=abort", ca.output[-1])
            self.assertTrue(rig.g._adm.orphan)
            # a queued admission behind it cannot start while the orphan reads
            w2 = _Req("w2", self._ids(160))
            self.assertEqual(adm.admit_ple_request(w2, rig.chunk), "queued")
            x = _Req("x", self._ids(170))
            ids, out = rig.chunk_of(x, 0)  # joins the orphan, then w2 starts
            self.assertTrue(_same(out, _serial(self.table, ids)))
            self.assertEqual(rig.g._adm.rid, "w2")
            self.assertEqual(adm.drop_ple_admission("w2"), 1)
            # a queued one is simply forgotten
            rig.chunk_of(x, 0)
            w3 = _Req("w3", self._ids(150))
            self.assertIn(adm.admit_ple_request(w3, rig.chunk), ("started", "queued"))
            w4 = _Req("w4", self._ids(150))
            self.assertEqual(adm.admit_ple_request(w4, rig.chunk), "queued")
            self.assertEqual(adm.drop_ple_admission(None, abort_all=True), 2)
            self.assertEqual(len(rig.g._adm_queue), 0)
            ids, out = rig.chunk_of(w4, 0)  # nothing stale served
            self.assertTrue(_same(out, _serial(self.table, ids)))
        finally:
            rig.close()

    def test_cached_prefix_or_second_in_batch_drops_the_admission(self):
        rig = _Rig(self.table)
        try:
            rig.warm(self)
            w = _Req("w", self._ids(190))
            adm.admit_ple_request(w, rig.chunk)
            time.sleep(0.05)
            with self.assertLogs(ADM_LOGGER, logging.INFO) as ca, \
                    self.assertLogs(PF_LOGGER, logging.INFO) as cp:
                ids, out = rig.batch([(w, 40, 190)])  # a cached prefix of 40 tokens
            self.assertTrue(_same(out, _serial(self.table, ids)))
            self.assertIn("rid=w dropped reason=cached_prefix", ca.output[-1])
            self.assertEqual(_chunks(cp.records)[-1]["read"], 150)
            v = _Req("v", self._ids(120))
            x = _Req("x", self._ids(60))
            adm.admit_ple_request(v, rig.chunk)
            time.sleep(0.05)
            with self.assertLogs(ADM_LOGGER, logging.INFO) as ca:
                ids, out = rig.batch([(x, 0, 60), (v, 0, 120)])  # v behind x in one batch
            self.assertTrue(_same(out, _serial(self.table, ids)))
            self.assertIn("rid=v dropped reason=not_first_in_batch", ca.output[-1])
            self.assertIsNone(rig.g._adm)
        finally:
            rig.close()

    def test_front_hint_then_intake_confirms_or_readmits(self):
        rig = _Rig(self.table, delay_s=0.1)
        try:
            rig.warm(self)
            toks = self._ids(150)
            with self.assertLogs(ADM_LOGGER, logging.INFO) as ca:
                self.assertEqual(adm.admit_ple_hint("h", toks.tolist(), rig.chunk, dormant=True), "started")
                self.assertEqual(adm.admit_ple_request(_Req("h", toks), rig.chunk), "confirmed")
            self.assertIn("rid=h rows=150 started source=hint dormant=1", ca.output[0])
            self.assertEqual(rig.g.stats["admit_started"], 1)
            time.sleep(0.2)
            with self.assertLogs(PF_LOGGER, logging.INFO) as cp:
                ids, out = rig.chunk_of(_Req("h", toks), 0)
            self.assertTrue(_same(out, _serial(self.table, ids)))
            self.assertEqual(_chunks(cp.records)[-1]["read"], 0)
            # a hint whose tokens differ from the real prompt is replaced
            adm.admit_ple_hint("k", self._ids(150).tolist(), rig.chunk)
            real = _Req("k", self._ids(150))
            with self.assertLogs(ADM_LOGGER, logging.INFO) as ca:
                self.assertEqual(adm.admit_ple_request(real, rig.chunk), "queued")
            self.assertIn("rid=k dropped reason=tokens_differ", "\n".join(ca.output))
            time.sleep(0.2)
            ids, out = rig.chunk_of(_Req("x", self._ids(100)), 0)  # joins the orphan, starts k
            self.assertEqual(rig.g._adm.rid, "k")
            self.assertTrue(torch.equal(rig.g._adm.tokens, real.ids))
            time.sleep(0.2)
            with self.assertLogs(PF_LOGGER, logging.INFO) as cp:
                ids, out = rig.chunk_of(real, 0)
            self.assertTrue(_same(out, _serial(self.table, ids)))
            self.assertEqual(_chunks(cp.records)[-1]["hit"], 150)
        finally:
            rig.close()

    def test_stale_admission_expires_and_queue_is_bounded(self):
        rig = _Rig(self.table)
        try:
            rig.warm(self)
            adm.admit_ple_request(_Req("old", self._ids(150)), rig.chunk)
            for i in range(adm.PLE_ADMIT_QUEUE_MAX):
                self.assertEqual(adm.admit_ple_request(_Req(f"q{i}", self._ids(150)), rig.chunk), "queued")
            self.assertEqual(adm.admit_ple_request(_Req("over", self._ids(150)), rig.chunk),
                             "skipped:queue_full")
            time.sleep(0.05)
            self.clock.t += adm.PLE_ADMIT_TTL_S + 1
            with self.assertLogs(ADM_LOGGER, logging.INFO) as ca:
                fresh = _Req("fresh", self._ids(150))
                self.assertEqual(adm.admit_ple_request(fresh, rig.chunk), "started")
            out = "\n".join(ca.output)
            self.assertIn("rid=old dropped reason=ttl", out)
            for i in range(adm.PLE_ADMIT_QUEUE_MAX):
                self.assertIn(f"rid=q{i} dropped reason=ttl", out)
        finally:
            rig.close()

    def test_short_first_chunk_is_not_admitted(self):
        rig = _Rig(self.table)
        try:
            rig.warm(self)
            self.assertEqual(adm.admit_ple_request(_Req("s", self._ids(10)), rig.chunk), "skipped:small")
            self.assertIsNone(rig.g._adm)
        finally:
            rig.close()

    def test_a_lost_worker_at_admission_falls_back_to_serial(self):
        rig = _Rig(self.table)
        try:
            rig.warm(self)
            for pid in rig.g._workers.pids():
                os.kill(pid, 9)
            time.sleep(0.2)
            with self.assertLogs(ADM_LOGGER, logging.ERROR):
                self.assertIsNone(adm.admit_ple_request(_Req("w", self._ids(150)), rig.chunk))
            self.assertTrue(rig.g._disabled)
            self.assertEqual(adm._SINKS, [])
            ids, out = rig.chunk_of(_Req("w", self._ids(150)), 0)
            self.assertTrue(_same(out, _serial(self.table, ids)))
        finally:
            rig.close()


class TestSwitch(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = _table(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_switch_off_is_h32_unchanged(self):
        base = pt.PleCheckpointPreadGather(self.table, min_rows=16, workers=2)
        try:
            with envs.SGLANG_QWEN4_PLE_PREFETCH.override(True), \
                    envs.SGLANG_QWEN4_PLE_PREFETCH_ADMIT.override(False):
                g = pf.make_ple_prefetch_gather(base, self.table, _identity_hasher)
            self.assertIs(type(g), pf.PlePrefetchGather)
            self.assertEqual(adm._SINKS, [])
            req = _Req("w", torch.randint(0, 256, (150,)))
            self.assertIsNone(adm.admit_ple_request(req, 200))  # no admitting gather here
            adm.note_ple_batch([req])
            self.assertEqual(adm._BATCH, ())
            self.assertEqual(len(g._slots), 2)
            g._disable()
            with envs.SGLANG_QWEN4_PLE_PREFETCH.override(True), \
                    envs.SGLANG_QWEN4_PLE_PREFETCH_ADMIT.override(True):
                g = pf.make_ple_prefetch_gather(base, self.table, _identity_hasher)
            self.assertIs(type(g), adm.PleAdmitPrefetchGather)
            self.assertEqual(adm._SINKS, [g])
            g._disable()
            self.assertEqual(adm._SINKS, [])
        finally:
            base.close()

    def test_group_d_does_not_admit(self):
        from sglang.srt.managers import weg2_memory_saver as ms

        with envs.SGLANG_QWEN4_PLE_PREFETCH_ADMIT.override(True):
            with mock.patch.object(ms, "weg2_group_name", lambda: "D"):
                self.assertFalse(adm.ple_admission_wanted())
            with mock.patch.object(ms, "weg2_group_name", lambda: "P"):
                self.assertTrue(adm.ple_admission_wanted())
            with mock.patch.object(ms, "weg2_group_name", lambda: ""):
                self.assertTrue(adm.ple_admission_wanted())
        with envs.SGLANG_QWEN4_PLE_PREFETCH_ADMIT.override(False):
            self.assertFalse(adm.ple_admission_wanted())

    def test_model_hasher_reports_readiness_and_warms(self):
        emb = types.SimpleNamespace(
            layer_multipliers=torch.tensor([3, 5, 7]), ngram_heads_vocab_sizes=torch.tensor([11] * 16),
            ngram_heads_offsets=torch.arange(16) * 11, heads_per_ngram=8, ngram_size=3, eos_token_id=1,
        )
        h = pf.ple_next_chunk_hasher(emb)
        self.assertFalse(h.ple_hash_ready())
        h.ple_hash_warm()
        self.assertTrue(h.ple_hash_ready())
        emb.layer_multipliers = None  # after the warm, the device is never read again
        self.assertEqual(h(torch.arange(20), 0).numel(), 20 * 16)


def _method(tree, cls, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


def _calls(fn):
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            out.append((node.lineno, name, node))
    return sorted(out, key=lambda c: c[0])


class TestWiring(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(SRT, "managers", "scheduler.py")) as f:
            cls.src = f.read()
        cls.tree = ast.parse(cls.src)

    def test_intake_admits_before_the_queue_and_the_dormant_hold(self):
        fn = _method(self.tree, "Scheduler", "_add_request_to_queue")
        calls = _calls(fn)
        admit = [l for l, n, _ in calls if n == "_ple_admit_on_intake"]
        self.assertEqual(len(admit), 1)
        appends = [l for l, n, c in calls if n == "append" and isinstance(c.func, ast.Attribute)
                   and ast.unparse(c.func.value) in ("hold", "self.waiting_queue")]
        self.assertEqual(len(appends), 2)
        self.assertTrue(all(admit[0] < l for l in appends))
        # a retracted re-queue is not admitted (its fill is not its prompt)
        guard = [n for n in ast.walk(fn) if isinstance(n, ast.If) and "is_retracted" in ast.unparse(n.test)
                 and "_ple_admit_on_intake" in ast.unparse(n)]
        self.assertTrue(guard)

    def test_delegates(self):
        for name, callee in (("_ple_admit_on_intake", "admit_ple_request"),
                             ("handle_ple_prefetch_hint", "admit_ple_hint")):
            fn = _method(self.tree, "Scheduler", name)
            self.assertIn(callee, [n for _, n, _ in _calls(fn)])
            self.assertIn("weg2_dormant", ast.unparse(fn))
        fwd = _method(self.tree, "Scheduler", "_run_batch_forward")
        names = [n for _, n, _ in _calls(fwd)]
        self.assertLess(names.index("publish_ple_next_chunk"), names.index("note_ple_batch"))
        ab = [n for _, n, _ in _calls(_method(self.tree, "Scheduler", "abort_request"))]
        self.assertLess(ab.index("drop_ple_admission"), ab.index("_abort_request_now"))
        disp = ast.unparse(_method(self.tree, "Scheduler", "init_request_dispatcher"))
        self.assertIn("(PlePrefetchHintReqInput, self.handle_ple_prefetch_hint)", disp)

    def test_delegate_passes_the_scheduler_state(self):
        fn = _method(self.tree, "Scheduler", "_ple_admit_on_intake")
        ns = {"Req": object}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<h43>", "exec"), ns)
        seen = []
        with mock.patch.object(adm, "admit_ple_request",
                               lambda req, cs, dormant=False: seen.append((req, cs, dormant))):
            me = types.SimpleNamespace(chunked_prefill_size=16384, weg2_dormant=True)
            ns["_ple_admit_on_intake"](me, "REQ")
        self.assertEqual(seen, [("REQ", 16384, True)])

    def test_hint_struct_is_an_ipc_type(self):
        from sglang.srt.managers import io_struct

        h = io_struct.PlePrefetchHintReqInput(rid="r", input_ids=[1, 2, 3])
        self.assertIn(io_struct.PlePrefetchHintReqInput, io_struct._all_types)
        self.assertEqual((h.rid, h.input_ids), ("r", [1, 2, 3]))

    def test_http_route_exists(self):
        with open(os.path.join(SRT, "entrypoints", "http_server.py")) as f:
            src = f.read()
        self.assertIn('@app.api_route("/weg2/ple_prefetch_hint", methods=["POST"])', src)
        self.assertIn("tm._dispatch_to_scheduler(hint)", src)


class TestHint(CustomTestCase):
    def test_body_and_condition(self):
        from sglang.srt.weg2 import ple_admit_hint as ph

        self.assertTrue(ph.ple_hint_wanted(awake="D"))
        self.assertTrue(ph.ple_hint_wanted(awake=None))  # mid-flip
        self.assertFalse(ph.ple_hint_wanted(awake="P"))
        self.assertFalse(ph.ple_hint_wanted(awake="D", skip_leg1=True))
        body = ph.ple_hint_body("/v1/chat/completions", {"rid": "r", "stream": True, "messages": []})
        self.assertEqual(body, {"path": "/v1/chat/completions", "payload": {"rid": "r", "messages": []}})
        self.assertIsNone(ph.ple_hint_body("/v1/chat/completions", {"messages": []}))
        self.assertIsNone(ph.ple_hint_body("/health", {"rid": "r"}))

    def test_build_tokenizes_like_the_leg1_post(self):
        from sglang.srt.weg2 import ple_admit_hint as ph

        seen = []

        class Chat:
            def _convert_to_internal_request(self, req, raw):
                seen.append((type(req).__name__, req.rid, raw))
                return types.SimpleNamespace(input_ids=[5, 6, 7], text=None), req

        class Chat2:
            def _convert_to_internal_request(self, req, raw):
                return types.SimpleNamespace(input_ids=None, text="abc"), req

        enc = lambda s: [ord(c) for c in s]
        run = asyncio.new_event_loop().run_until_complete
        body = {"path": "/v1/chat/completions",
                "payload": {"rid": "weg2-2-6", "model": "m", "max_tokens": 1,
                            "messages": [{"role": "user", "content": "hi"}]}}
        h = run(ph.build_ple_prefetch_hint(body, serving_chat=Chat(), serving_completion=None,
                                           encode=enc, raw_request="RAW"))
        self.assertEqual((h.rid, h.input_ids), ("weg2-2-6", [5, 6, 7]))
        self.assertEqual(seen, [("ChatCompletionRequest", "weg2-2-6", "RAW")])
        h = run(ph.build_ple_prefetch_hint(body, serving_chat=Chat2(), serving_completion=None, encode=enc))
        self.assertEqual(h.input_ids, [97, 98, 99])
        h = run(ph.build_ple_prefetch_hint({"path": "/generate", "payload": {"rid": "g", "text": "ab"}},
                                           serving_chat=None, serving_completion=None, encode=enc))
        self.assertEqual(h.input_ids, [97, 98])
        self.assertIsNone(run(ph.build_ple_prefetch_hint({"path": "/generate", "payload": {"text": "ab"}},
                                                         serving_chat=None, serving_completion=None, encode=enc)))

    def test_front_posts_the_hint_only_while_p_is_not_awake(self):
        from sglang.srt.weg2 import front as fr
        from sglang.srt.weg2 import ple_admit_hint as ph

        async def go(awake, env=True, skip=False):
            posted = []

            async def rpc(g, path, body, timeout):
                posted.append((g.url, path, body))
                return 200, "{}"

            me = types.SimpleNamespace(
                awake=awake, groups={"P": types.SimpleNamespace(url="http://p")}, session=object(), rpc=rpc)
            p = types.SimpleNamespace(rid="weg2-2-6", path="/v1/chat/completions", skip_leg1=skip,
                                      payload={"rid": "weg2-2-6", "messages": [], "stream": True})
            with envs.SGLANG_WEG2_PLE_ADMIT_HINT.override(env):
                fr.Front._maybe_ple_admit_hint(me, p)
            await asyncio.sleep(0.01)
            return posted

        run = asyncio.new_event_loop().run_until_complete
        posted = run(go("D"))
        self.assertEqual(posted, [("http://p", ph.HINT_PATH,
                                   {"path": "/v1/chat/completions",
                                    "payload": {"rid": "weg2-2-6", "messages": []}})])
        self.assertEqual(run(go("P")), [])
        self.assertEqual(run(go("D", env=False)), [])
        self.assertEqual(run(go("D", skip=True)), [])
        src = inspect.getsource(fr.Front.handle_generate)
        self.assertIn("self._maybe_ple_admit_hint(p)", src)
        self.assertLess(src.index("BATCH queued"), src.index("self._maybe_ple_admit_hint(p)"))


if __name__ == "__main__":
    unittest.main()
