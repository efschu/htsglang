"""SWEEPCENSUS-1023: the D-DECODE-GAP cause ``result_host``.

Boot 1023_223241 D TP0 23:30:28: ``D-DECODE-GAP gap_ms=33696 cause=d_prefill ... d_prefill_ms=367 extends=1 ...
spans=...,result:33182,...`` -- 33.2 s of host time in process_batch_result (the S108 turn end's publish sweep at a
full arena) was named d_prefill by the extends' fallback; without an extend the same gap read ``cause=other`` (the
monitor's DECODE-LUECKEN-UNERKLAERT 72.5 s). The result span now names the gap when it holds half of the rest.
"""

from __future__ import annotations

from flliper.srt.pdflip import d_decode_gap as G

try:
    from flliper.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(__file__)
except Exception:  # noqa: BLE001 -- the registry is optional on the desk
    pass


def _gap(result_ms, ext=None, end=34.0):
    tr = G.Tracker(400, 0, 0.0)
    tr.on_batch(0.0, 3, 0, 1)
    tr.on_run_end(0.06, 3)
    if ext is not None:
        t0, t1, tok = ext
        tr.on_batch(t0, 0, tok, 1)       # a pure extend opens its window ...
        tr._close_extend(t1)             # ... and closes at t1
    tr.span("result", result_ms)
    tr.span("recv", 20.0)
    return tr.on_batch(end, 3, 0, 1)


def test_the_metal_case_with_a_short_extend_is_result_host_not_d_prefill():
    rec = _gap(33182.0, ext=(0.2, 0.567, 196), end=33.7)
    assert rec["d_prefill_ms"] < 0.5 * rec["gap_ms"]
    assert rec["cause"] == "result_host", rec["cause"]


def test_without_an_extend_it_is_result_host_not_other():
    rec = _gap(27890.0, end=28.0)
    assert rec["cause"] == "result_host"


def test_a_small_result_span_keeps_the_old_causes():
    assert _gap(50.0, end=1.0)["cause"] == "other"
    assert _gap(50.0, ext=(0.2, 0.5, 196), end=1.0)["cause"] == "d_prefill"


def test_a_long_extend_stays_d_prefill():
    rec = _gap(3000.0, ext=(0.2, 7.2, 2048), end=10.0)
    assert rec["cause"] == "d_prefill"


def test_the_census_counts_the_new_cause():
    tr = G.Tracker(400, 0, 0.0)
    tr.on_batch(0.0, 3, 0, 1)
    tr.span("result", 9000.0)
    tr.on_batch(10.0, 3, 0, 1)
    assert tr.census["result_host"][0] == 1
