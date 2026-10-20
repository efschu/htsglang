"""02.10. NF y7l (desk/nf-y7k-integ-1002 @ 99d1977a63): the D->P "Nachlauf".

Boot ...dauer10021206_99d1977a63_1002_120637, epoch 3 (and 5, 7, 9 alike):
flip done 12:10:46.605; the P log then reads

  PP0 PDFLIP-REARM-DEFER landed layers=29 rows=4872  ... fill_ms=626   (12:10:47)
  PP1 PDFLIP-REARM-DEFER landed layers=11 rows=10208 ... fill_ms=4143  (12:10:50)
  PP2 PDFLIP-REARM-DEFER landed layers=8  rows=12256 ... fill_ms=6630  (12:10:53)

and was read as "the deferred extra-row fill (2 rows/ms) gates P's first forward
on PP1/PP2". It does not. ``fill_ms`` is fill-start -> the promotion at the
rank's FIRST forward, and under FLIPCYCLE H2 (per-layer landing, default on) a
first extend promotes at the tick only the layers whose event already
completed -- by_tick=all, ticks=1 and no ``stage=rearm_wait mode=per_layer``
line means every layer had landed BEFORE that forward. PP1's first forward
starts when PP0 has computed chunk 1 (PASS-STALL pp_rank=1 proxy_recv_ms=2951,
PP0 ``Prefill rank batch ... gpu-ms: 3532``), PP2's when PP1 has (proxy_recv_ms
=5918, PP1 gpu-ms 2482): the 6.6 s are chunk 1's compute on PP0+PP1. Per-stage
chunk-1 times against the next chunks: PP1 2482 / 2484-2544 ms, PP2 1694 /
1251-1730 ms -- no excess; PP0 3532 / 2965-3007 ms: +0.53 s, and that is

  PP0 PLE-PREFETCH chunk=0 rows=262144 ready=no wait_ms=540.8

-- the front's hint (sent while D was awake) read the prompt's TAIL window
(start=63100), PP0's told named 6208, the hint was dropped ``start_moved`` and
the told's own read started 85 ms before the forward. Same at every D->P wake of
the boot (wait_ms=406 / 241). The front knew the start: ``PDFLIP DP-WAIT ...
presence_span=6208 span_known=True``.

So:
  * the landed line now says whether a forward waited (``forward_waited_layers``)
    and when the rank's first forward came (``first_forward_ms``), so fill_ms can
    no longer be read as copy time (RED on 99d1977a63: no such fields);
  * the hint carries the front's store span (``start_hint``), P floors it to its
    page and reads THAT window -- PP0's told then CONFIRMS the hint's read
    (RED on 99d1977a63: ple_hint_start knows no start_hint, the window is the
    tail).
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import asyncio
import importlib.util
import logging
import os
import pathlib
import re
import sys
import types
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.layers.moe import expert_offload as eo
from flliper.srt.models import qwen4_exp_ple_admit as adm
from flliper.srt.pdflip import ple_admit_hint as ph

_HERE = pathlib.Path(__file__).parent
_SRT = pathlib.Path(eo.__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_rearm_defer_host_p_0929_nachlauf", _HERE / "test_pdflip_rearm_defer_host_p_0929.py")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def rig(monkeypatch):
    ops, clock = base._Ops(), _Clock()
    monkeypatch.setattr(eo, "_DEFERRED_ROWS_FILL", eo.DeferredRowsFill(stream_ops=ops, clock=clock))
    yield ops, clock


def _decode():
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))


def _landed(caplog):
    lines = [r.getMessage() for r in caplog.records if "PDFLIP-REARM-DEFER landed" in r.getMessage()]
    assert len(lines) == 1, lines
    return lines[0]


def _field(line, name):
    m = re.search(r"\b%s=(\S+)" % name, line)
    assert m, (name, line)
    return m.group(1)


# ---------------------------------------------------------------- REARM-DEFER


def test_y7l_shape_rows_landed_before_the_first_forward_gated_nothing(rig, caplog):
    """PP2: fill-start at the wake, the rows done long before, the first forward
    6.63 s later (PP0+PP1 computing chunk 1): fill_ms=6630, nobody waited."""
    ops, clock = rig
    a, b = base._cache(layer_id=40, seed=1), base._cache(layer_id=41, seed=2)
    a.rearm_after_wake(defer=base.HOST)
    b.rearm_after_wake(defer=base.HOST)
    fill = eo.deferred_rows_fill()
    fill.start()
    for ev in ops.events:
        ev.done = True  # the copies finished on the side stream
    clock.t += 6.630
    with caplog.at_level(logging.INFO, logger=eo.logger.name):
        eo.deferred_rows_tick(base._extend())  # PP2's first forward
    line = _landed(caplog)
    assert _field(line, "fill_ms") == "6630"
    assert _field(line, "forward_waited_layers") == "0"
    assert _field(line, "first_forward_ms") == "6630"
    assert "not the copy time" in line
    assert not fill.pending


def test_rows_still_in_flight_at_a_decode_tick_are_counted_as_waited(rig, caplog):
    ops, clock = rig
    a, b = base._cache(layer_id=0, seed=1), base._cache(layer_id=1, seed=2)
    a.rearm_after_wake(defer=base.HOST)
    b.rearm_after_wake(defer=base.HOST)
    eo.deferred_rows_fill().start()
    ops.events[0].done = True  # layer 0 landed, layer 1 still copying
    clock.t += 0.2
    with caplog.at_level(logging.INFO, logger=eo.logger.name):
        eo.deferred_rows_tick(_decode())  # a decode waits every event
    line = _landed(caplog)
    assert _field(line, "forward_waited_layers") == "1"
    assert ops.waited == ops.events


def test_per_layer_landing_counts_only_the_layers_the_extend_waited_for(rig, caplog):
    ops, clock = rig
    a, b = base._cache(layer_id=0, seed=1), base._cache(layer_id=1, seed=2)
    a.rearm_after_wake(defer=base.HOST)
    b.rearm_after_wake(defer=base.HOST)
    fill = eo.deferred_rows_fill()
    fill.start()
    ops.events[0].done = True
    clock.t += 0.5
    eo.deferred_rows_tick(base._extend())  # promotes layer 0 (landed), keeps 1
    assert fill.pending == [b]
    clock.t += 0.1
    with caplog.at_level(logging.INFO, logger=eo.logger.name):
        assert b.land_deferred_rows() is True  # layer 1's MoE: its event still open
    line = _landed(caplog)
    assert (_field(line, "by_tick"), _field(line, "by_eager")) == ("1", "1")
    assert _field(line, "forward_waited_layers") == "1"
    assert _field(line, "first_forward_ms") == "500"
    assert _field(line, "fill_ms") == "600"


def test_a_land_before_the_fill_started_copies_on_the_forward_stream_and_counts(rig, caplog):
    ops, _clock = rig
    a = base._cache(layer_id=0, seed=1)
    a.rearm_after_wake(defer=base.HOST)
    with caplog.at_level(logging.INFO, logger=eo.logger.name):
        assert a.land_deferred_rows() is True  # not on the side stream yet
    assert _field(_landed(caplog), "forward_waited_layers") == "1"
    assert base._rows_digest(a) == base._rows_digest(base._serial(layer_id=0, seed=1))


def test_the_counters_reset_with_the_fill(rig):
    ops, clock = rig
    a = base._cache(layer_id=0)
    a.rearm_after_wake(defer=base.HOST)
    eo.deferred_rows_fill().start()
    eo.deferred_rows_tick(_decode())
    fill = eo.deferred_rows_fill()
    assert (fill.waited, fill.t_first_tick) == (0, None)


# ---------------------------------------------------------------- PLE hint window


@pytest.mark.parametrize("n,span,told", [
    (79484, 6208, 6208),     # pdflip-2-7
    (78349, 6400, 6400),     # pdflip-4-18
    (164290, 78208, 78208),  # pdflip-6-19
    (165382, 78348, 78336),  # pdflip-8-20: P's told is page-floored (page 64)
])
def test_y7l_the_hint_window_starts_at_pp0_told(n, span, told):
    assert adm.ple_hint_start(n, 16384, start_hint=span, page_size=64) == told


def test_without_a_span_the_hint_keeps_the_tail_window():
    assert adm.ple_hint_start(79484, 16384) == 63100  # y7l's dropped window
    assert adm.ple_hint_start(79484, 16384, start_hint=None, page_size=64) == 63100
    assert adm.ple_hint_start(79484, 16384, start_hint=-1, page_size=64) == 63100
    # a span covering the whole prompt names no chunk: the tail as before
    assert adm.ple_hint_start(500, 200, start_hint=500, page_size=1) == 300
    assert adm.ple_hint_start(500, 200, start_hint=520, page_size=64) == 300
    # a known empty store prefix: the first chunk from token 0
    assert adm.ple_hint_start(79484, 16384, start_hint=0, page_size=64) == 0


def test_the_front_sends_its_store_span_when_known():
    assert ph.ple_hint_span(span_known=True, store_span_est=6208) == 6208
    assert ph.ple_hint_span(span_known=True, store_span_est=0) == 0
    assert ph.ple_hint_span(span_known=False, store_span_est=6208) is None
    payload = {"rid": "pdflip-2-7", "stream": True, "messages": []}
    body = ph.ple_hint_body("/v1/chat/completions", payload, start_hint=6208)
    assert body == {"path": "/v1/chat/completions", "payload": {"rid": "pdflip-2-7", "messages": []},
                    "start_hint": 6208}
    assert "start_hint" not in ph.ple_hint_body("/v1/chat/completions", payload)


def test_p_carries_the_start_hint_into_the_scheduler():
    class Chat:
        def _convert_to_internal_request(self, req, raw):
            return types.SimpleNamespace(input_ids=[5, 6, 7], text=None), req

    body = {"path": "/v1/chat/completions", "start_hint": 6208,
            "payload": {"rid": "pdflip-2-7", "model": "m", "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}]}}
    run = asyncio.new_event_loop().run_until_complete
    h = run(ph.build_ple_prefetch_hint(body, serving_chat=Chat(), serving_completion=None,
                                       encode=lambda s: [1]))
    assert (h.rid, h.input_ids, h.start_hint) == ("pdflip-2-7", [5, 6, 7], 6208)
    body.pop("start_hint")
    h = run(ph.build_ple_prefetch_hint(body, serving_chat=Chat(), serving_completion=None,
                                       encode=lambda s: [1]))
    assert h.start_hint == -1


def _method(tree, cls, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == name:
                    return f
    raise AssertionError((cls, name))


def test_wiring_front_and_scheduler_pass_the_span():
    front = ast.parse((_SRT / "pdflip" / "front.py").read_text())
    src = ast.unparse(_method(front, "Front", "_maybe_ple_admit_hint"))
    assert "ple_hint_span(span_known=bool(getattr(p, 'span_known', False)), store_span_est=int(getattr(p, 'store_span_est', 0) or 0))" in src
    assert "ple_hint_body(p.path, p.payload, start_hint=span)" in src
    sched = ast.parse((_SRT / "managers" / "scheduler.py").read_text())
    src = ast.unparse(_method(sched, "Scheduler", "handle_ple_prefetch_hint"))
    assert "start_hint=getattr(recv_req, 'start_hint', -1)" in src
    assert "page_size=" in src


def test_admit_ple_hint_reads_the_spanned_window(monkeypatch):
    seen = []

    class Sink:
        def admit(self, rid, ids, chunk_size, *, dormant, source, start):
            seen.append((rid, source, start))
            return "started"

    monkeypatch.setattr(adm, "_SINKS", [Sink()])
    ids = list(range(79484))
    assert adm.admit_ple_hint("pdflip-2-7", ids, 16384, dormant=True, start_hint=6208, page_size=64) == "started"
    assert adm.admit_ple_hint("x", ids, 16384, dormant=True) == "started"
    assert seen == [("pdflip-2-7", "hint", 6208), ("x", "hint", 63100)]


def test_the_front_posts_the_span_with_the_hint():
    from flliper.srt.environ import envs
    from flliper.srt.pdflip import front as fr

    async def go(**span):
        posted = []

        async def rpc(g, path, body, timeout):
            posted.append(body)
            return 200, "{}"

        me = types.SimpleNamespace(
            awake="D", groups={"P": types.SimpleNamespace(url="http://p")}, session=object(), rpc=rpc)
        p = types.SimpleNamespace(rid="pdflip-2-7", path="/v1/chat/completions", skip_leg1=False,
                                  payload={"rid": "pdflip-2-7", "messages": []}, **span)
        with envs.FLLIPER_PDFLIP_PLE_ADMIT_HINT.override(True):
            fr.Front._maybe_ple_admit_hint(me, p)
        await asyncio.sleep(0.01)
        return posted

    run = asyncio.new_event_loop().run_until_complete
    assert [b.get("start_hint") for b in run(go(span_known=True, store_span_est=6208))] == [6208]
    assert [b.get("start_hint") for b in run(go(span_known=False, store_span_est=6208))] == [None]
