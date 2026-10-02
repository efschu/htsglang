# SPDX-License-Identifier: Apache-2.0
"""DASH-FIELDS (02.10., user decision, rigdash depth / per-rid decode view): additive server
fields only -- no behaviour change, no new sync point, no .item() / D2H in the decode path.

F1 rankstats prefill.last.ext / prefill.last_ext: [[rid, start, end], ...] of the flushed
   prefill chunk (start = the #969 EXTENT start, end = start + computed tokens; cap 8).
F2 rankstats decode.reqs: [[rid, prompt, out], ...] of the running batch, read in the
   rankstats timer thread (cap 16).
F4 request_done: p_leg1_dispatch_ts / p_leg1_end_ts.
(F3 decode.tokens_by_bs is not built: the decode round log keeps no per-round host token count.)
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import metrics_reporter as MR  # noqa: E402
from sglang.srt.weg2 import front_requests as FRQ  # noqa: E402
from sglang.srt.weg2 import rankstats as RS  # noqa: E402


def _req(rid, start=None, prefix=0, n=0, prompt=0, out=0):
    er = None if start is None else types.SimpleNamespace(start=start, end=start + n)
    return types.SimpleNamespace(rid=rid, extend_range=er, prefix_indices=list(range(prefix)),
                                 extend_input_len=n, origin_input_ids=list(range(prompt)),
                                 output_ids=list(range(out)))


def test_f1_prefill_ext_names_each_request_of_the_chunk():
    batch = types.SimpleNamespace(reqs=[_req("weg2-0-3", start=90112, n=1024),
                                        _req("weg2-1-5", prefix=18, n=7)])
    assert MR.prefill_ext(batch) == [["weg2-0-3", 90112, 91136], ["weg2-1-5", 18, 25]]
    many = types.SimpleNamespace(reqs=[_req(f"r{i}", start=0, n=1) for i in range(20)])
    assert len(MR.prefill_ext(many)) == MR.PREFILL_EXT_CAP
    assert MR.prefill_ext(types.SimpleNamespace(reqs=None)) == []


def test_f1_the_untimed_record_keeps_last_ext_and_leaves_last_alone():
    log = MR.RankPrefillLog()
    log.record(new_tokens=7, cached_tokens=18, timed=False, ext=[["weg2-1-5", 18, 25]])
    assert log.cum["last_ext"] == [["weg2-1-5", 18, 25]]
    assert log.cum["last"] is None, "the timed reading stays untouched"
    log.record(new_tokens=3, cached_tokens=0, timed=False)  # old call form: still works
    assert log.cum["chunks"] == 2


def test_f1_the_rankstats_prefill_block_carries_it():
    log = MR.RankPrefillLog()
    log.record(new_tokens=7, cached_tokens=18, timed=False, ext=[["weg2-1-5", 18, 25]])
    blk = RS._prefill_block(types.SimpleNamespace(rank_prefill_log=log))
    assert blk["last_ext"] == [["weg2-1-5", 18, 25]]


def test_f1_the_reporter_passes_the_batch_ext():
    import inspect

    src = inspect.getsource(MR.SchedulerMetricsReporter.report_prefill_stats)
    assert "ext=prefill_ext(batch)" in src


def test_f2_decode_reqs_are_read_from_the_running_batch():
    rb = types.SimpleNamespace(reqs=[_req("weg2-14-37", prompt=66942, out=516),
                                     _req("weg2-15-38", prompt=67103, out=130)])
    assert RS._decode_reqs(rb) == [["weg2-14-37", 66942, 516], ["weg2-15-38", 67103, 130]]
    rb_many = types.SimpleNamespace(reqs=[_req(f"r{i}", prompt=1, out=1) for i in range(40)])
    assert len(RS._decode_reqs(rb_many)) == RS.DECODE_REQS_CAP


def test_f2_scheduler_counters_carry_decode_reqs():
    mr = types.SimpleNamespace(rank_prefill_log=MR.RankPrefillLog(), decode_round_log=None)
    sched = types.SimpleNamespace(metrics_reporter=mr, waiting_queue=[], forward_ct=3,
                                  running_batch=types.SimpleNamespace(
                                      reqs=[_req("weg2-9-16", prompt=118671, out=214)]))
    out = RS.scheduler_counters(sched)
    assert out["decode"]["reqs"] == [["weg2-9-16", 118671, 214]]


def test_f4_request_done_carries_p_leg1_times():
    book = FRQ.RequestBook(now=100.0)
    book.arrive("weg2-0-4", 100.0, 1)
    book.leg1_dispatch("weg2-0-4", 107.25)
    book.leg1_done("weg2-0-4", 134.5, 45284, 45056, None, None)
    rec, _park = book.done("weg2-0-4", 300.0, 200, 3)
    assert rec["p_leg1_dispatch_ts"] == 107.25 and rec["p_leg1_end_ts"] == 134.5
    book.arrive("weg2-1-5", 200.0, 1)
    rec2, _ = book.done("weg2-1-5", 201.0, 200, 1)
    assert rec2["p_leg1_dispatch_ts"] is None and rec2["p_leg1_end_ts"] is None
