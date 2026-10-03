"""FLIPZEIT 02.10.: rankstats decode.last_t -- the end of a rank's last decode
round, in the time.time() base of prefill.last.t.

rigdash measures D>P from the end of the last D decode round to the first P
prefill chunk (the user's definition) and read the start from the D log line
'Decode rank batch, rank: 0' (t + gpu-ms) because the IPC had no field.
Pinned (red before): DecodeRoundLog._emit records round open + gpu-ms as
last_end_t (the same t + gpu-ms rigdash computes from the line); rankstats
publishes it as decode.last_t, rounded like prefill.last.t, None before the
first round; the schema doc names it.
"""
from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import decode_round_log as D  # noqa: E402
from sglang.srt.weg2 import rankstats  # noqa: E402


def test_emit_records_the_end_of_the_round():
    log = D.DecodeRoundLog()
    assert log.last_end_t is None
    acc = SimpleNamespace(round_id=7, wall=1790960000.250, bs=2, rows=2, depth=None)
    res = SimpleNamespace(round_ms=18.5, wait_ms=3.0, families={}, split_refused=None)
    log._emit(acc, [(res, "decode", True)])
    assert abs(log.last_end_t - (1790960000.250 + 0.0185)) < 1e-6
    assert log.last_bs == 2 and log.cum_rounds == 1


def test_rankstats_publishes_decode_last_t():
    drl = SimpleNamespace(cum_rounds=3, cum_gpu_ms=50.0, cum_by_bs={1: [3, 50.0]}, last_bs=1,
                          last_end_t=1790960000.2685)
    mr = SimpleNamespace(decode_round_log=drl, gen_tokens_total=0, last_running_reqs=1)
    d = rankstats._decode_block(mr)
    assert d["last_t"] == round(1790960000.2685, 3)
    drl.last_end_t = None
    assert rankstats._decode_block(mr)["last_t"] is None
    assert rankstats._decode_block(SimpleNamespace(decode_round_log=None, gen_tokens_total=0))["last_t"] is None


def test_schema_doc_names_the_field():
    assert "last_t (END of the last round" in rankstats.__doc__
    assert "self.last_end_t = None" in inspect.getsource(D.DecodeRoundLog.__init__)
