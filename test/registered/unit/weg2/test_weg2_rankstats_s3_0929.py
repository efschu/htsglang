"""RANKSTATS §3 (DASHBOARD-AUS-IPC, 29.09.): prefill / decode / cache blocks in the
rank's rankstats file, so C1-C5, D1, D2, E1, E2 leave the log (user via 27B:
"warum sind die ganzen werte im dashboard noch aus log").

Every field is a counter the path keeps anyway, summed next to the line it
already prints; the rankstats timer thread only READS. RED on 2188e1bd98: no
``RankPrefillLog.cum``, no ``DecodeRoundLog.cum_*``, no ``prefill``/``decode``/
``cache`` blocks, no ``_988_LOADBACK_SEEN['tok']``.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.managers import schedule_policy as sp
from sglang.srt.managers.scheduler_components import decode_round_log as drl_mod
from sglang.srt.managers.scheduler_components import metrics_reporter as mr_mod
from sglang.srt.mem_cache import match_refusal_census as mrc
from sglang.srt.weg2 import rankstats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PREFILL_KEYS = {"chunks", "new_tokens", "cached_tokens", "gpu_ms", "split_ms",
                "compute_ms", "wait_ms", "bubble_ms", "last"}
DECODE_KEYS = {"rounds", "gpu_ms", "gpu_ms_by_bs", "tokens", "running",
               "accept_len_ewma", "accept_rate_ewma", "cuda_graph"}
CACHE_KEYS = {"loadback_n", "loadback_tok", "mamba_resume_n", "mamba_tok",
              "store_incomplete_n", "prefetch"}
SCHED_KEYS = {"waiting", "running", "queue_req", "running_req", "pending_tokens"}


def _mr(rpl=None, drl=None):
    return SimpleNamespace(prefill_tokens_total=0, gen_tokens_total=7,
                           spec_total_num_accept_tokens=0, spec_total_num_forward_ct=0,
                           rank_prefill_log=rpl or mr_mod.RankPrefillLog(),
                           decode_round_log=drl, accept_len_ewma=None,
                           accept_rate_ewma=None, last_cuda_graph=None,
                           last_running_reqs=None, last_pending_tokens=None)


def _sched(mr):
    return SimpleNamespace(forward_ct=0, metrics_reporter=mr, waiting_queue=[1, 2],
                           running_batch=SimpleNamespace(reqs=[1]))


class _Timer:
    def _report(self):
        pass


def _round_result(ms, wait=None):
    return (SimpleNamespace(round_ms=ms, wait_ms=wait, split_refused=None if wait is not None else "x",
                            families=None), "decode", False)


class TestSchema(CustomTestCase):
    def test_counters_carry_the_three_s3_blocks_and_sched(self):
        rec = rankstats.scheduler_counters(_sched(_mr(drl=drl_mod.DecodeRoundLog())))
        self.assertEqual(set(rec["prefill"]), PREFILL_KEYS)
        self.assertEqual(set(rec["decode"]), DECODE_KEYS)
        self.assertEqual(set(rec["cache"]), CACHE_KEYS)
        self.assertTrue(SCHED_KEYS <= set(rec["sched"]))
        self.assertEqual(rec["sched"]["queue_req"], 2)
        self.assertEqual(rec["sched"]["running_req"], 1)
        json.dumps(rec)  # the file body must serialise

    def test_file_written_by_the_timer_carries_the_blocks(self):
        d = tempfile.mkdtemp(prefix="rkst-s3-")
        s = _sched(_mr(drl=drl_mod.DecodeRoundLog()))
        rs = rankstats.RankStats(state_dir=d, group="D", tp_rank=0, pp_rank=0,
                                 read_counters=lambda: rankstats.scheduler_counters(s), period=60)
        rs.write_once()
        with open(rs.path) as f:
            body = json.load(f)
        for k in ("prefill", "decode", "cache"):
            self.assertIn(k, body)


class TestPrefillFromTheLine(CustomTestCase):
    def test_untimed_record_sums_tokens_and_chunks(self):
        rpl = mr_mod.RankPrefillLog()
        rpl.record(new_tokens=100, cached_tokens=40, timed=False)
        rpl.record(new_tokens=5, cached_tokens=0, timed=False)
        c = rankstats._prefill_block(_mr(rpl))
        self.assertEqual((c["chunks"], c["new_tokens"], c["cached_tokens"]), (2, 105, 40))

    def test_timed_flush_sums_gpu_compute_wait(self):
        rpl = mr_mod.RankPrefillLog()
        rpl.timer = _Timer()
        rpl.record(new_tokens=16384, cached_tokens=0, timed=True)
        rpl._durations.append((0.5, 0.1, None))  # 500 ms gpu, 100 ms in collectives
        rpl.flush()
        c = rankstats._prefill_block(_mr(rpl))
        self.assertEqual(c["chunks"], 1)
        self.assertEqual(c["new_tokens"], 16384)
        self.assertAlmostEqual(c["gpu_ms"], 500.0, places=1)
        self.assertAlmostEqual(c["split_ms"], 500.0, places=1)
        self.assertAlmostEqual(c["compute_ms"], 400.0, places=1)
        self.assertAlmostEqual(c["wait_ms"], 100.0, places=1)
        self.assertEqual(c["last"]["new"], 16384)

    def test_graphed_flush_counts_gpu_but_not_the_split(self):
        rpl = mr_mod.RankPrefillLog()
        rpl.timer = _Timer()
        rpl.record(new_tokens=512, cached_tokens=0, timed=True, graphed=True)
        rpl._durations.append((0.2, 0.05, None))
        rpl.flush()
        c = rpl.cum
        self.assertAlmostEqual(c["gpu_ms"], 200.0, places=1)
        self.assertEqual((c["split_ms"], c["compute_ms"], c["wait_ms"]), (0.0, 0.0, 0.0))
        self.assertIsNone(c["last"]["compute_ms"])


class TestDecodeFromTheRound(CustomTestCase):
    def test_emit_sums_rounds_and_gpu_ms_by_bs(self):
        drl = drl_mod.DecodeRoundLog(clock=None, rank=0)
        drl._emit(SimpleNamespace(round_id=1, wall=0.0, bs=2, rows=3), [_round_result(10.0, 2.0)])
        drl._emit(SimpleNamespace(round_id=2, wall=0.0, bs=2, rows=3), [_round_result(12.0)])
        drl._emit(SimpleNamespace(round_id=3, wall=0.0, bs=4, rows=6), [_round_result(20.0, 5.0)])
        d = rankstats._decode_block(_mr(drl=drl))
        self.assertEqual(d["rounds"], 3)
        self.assertAlmostEqual(d["gpu_ms"], 42.0, places=1)
        self.assertEqual(d["gpu_ms_by_bs"], {"2": [2, 22.0], "4": [1, 20.0]})
        self.assertEqual(d["tokens"], 7)

    def test_decode_line_keeps_ewma_graph_and_running(self):
        src = inspect.getsource(mr_mod.SchedulerMetricsReporter.report_decode_stats)
        for attr in ("self.accept_len_ewma", "self.accept_rate_ewma",
                     "self.last_cuda_graph", "self.last_running_reqs"):
            self.assertIn(attr, src)
        mr = _mr(drl=drl_mod.DecodeRoundLog())
        mr.accept_len_ewma, mr.accept_rate_ewma = 2.4567, 0.51
        mr.last_cuda_graph, mr.last_running_reqs = True, 4
        d = rankstats._decode_block(mr)
        self.assertEqual((d["accept_len_ewma"], d["cuda_graph"], d["running"]), (2.457, True, 4))


class TestCacheFromTheCounters(CustomTestCase):
    def setUp(self):
        self._seen = dict(sp._988_LOADBACK_SEEN)
        self._gate = dict(mrc.PREFETCH_GATE_COUNTS)

    def tearDown(self):
        sp._988_LOADBACK_SEEN.clear()
        sp._988_LOADBACK_SEEN.update(self._seen)
        mrc.PREFETCH_GATE_COUNTS.clear()
        mrc.PREFETCH_GATE_COUNTS.update(self._gate)

    def test_loadback_prefetch_store_short_are_the_paths_own_counters(self):
        base = rankstats._cache_block(SimpleNamespace())
        req = SimpleNamespace(rid="r1", mamba_loadback_anchor_adopted=True)
        sp._note_988_loadback(req, 4096)
        mrc.note_prefetch_gate("landed")
        mrc.note_prefetch_gate("deferred")  # the census key the #1068 DEFERRED line counts
        c = rankstats._cache_block(SimpleNamespace(_weg2_store_short_seen=3))
        self.assertEqual(c["loadback_n"], base["loadback_n"] + 1)
        self.assertEqual(c["loadback_tok"], (base["loadback_tok"] or 0) + 4096)
        self.assertEqual(c["mamba_resume_n"], base["mamba_resume_n"] + 1)
        self.assertEqual(c["store_incomplete_n"], 3)
        self.assertEqual(c["prefetch"]["landed"], base["prefetch"]["landed"] + 1)
        self.assertEqual(c["prefetch"]["deferred"], base["prefetch"]["deferred"] + 1)


class TestPathWritesNothing(CustomTestCase):
    def test_counter_sites_do_no_file_io_and_import_no_rankstats(self):
        for mod in (mr_mod, drl_mod):
            tree = ast.parse(inspect.getsource(mod))
            names = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            self.assertFalse(any("rankstats" in n for n in names), mod.__name__)
        for fn in (mr_mod.RankPrefillLog.flush, drl_mod.DecodeRoundLog._emit):
            src = inspect.getsource(fn)
            self.assertNotIn("open(", src)
            self.assertNotIn("os.replace", src)


if __name__ == "__main__":
    unittest.main()
