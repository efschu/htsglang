# SPDX-License-Identifier: Apache-2.0
"""EARLY-FLIP (02.10.): with D idle, a request whose chars/3 price is far over X
begins the D->P flip at once; drain + D quiesce run beside the exact count and
the store probe, and the flip awaits the verdict before its first sleep RPC.

Binding flip time (user 02.10.): last D token -> first P chunk; every ms before
the begin counts. N6d (..._ec4d492f58_1002_170955): arrival -> verdict 194-224
ms (X-EXACT count 100-130 + PROBE-FAST ~100), drain + quiesce 120-250 ms after
it. 01./02.10. (3832 priced arrivals): chars/3 >= 2X caught 597 of 1063 LONGs,
7 of 604 such arrivals ended SHORT (1.2 %) -- those abort: no sleep RPC went
out, D stays awake and serves the SHORT.

Switch FLLIPER_PDFLIP_EARLY_FLIP (default on), factor FLLIPER_PDFLIP_EARLY_FLIP_X_FACTOR
(2.0). Markers 'PDFLIP-EARLY-FLIP begin|go|abort'. Hermetic: the real
Front.handle_generate and Front.flip against stubbed group RPCs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import front_tokens as FT  # noqa: E402

X = 4096
MIB = 1024 * 1024


class _Req:
    def __init__(self, payload, path="/v1/messages"):
        self._p = payload
        self.path = path

    async def json(self):
        return self._p


class _Tokens:
    def __init__(self, table):
        from concurrent.futures import ThreadPoolExecutor

        self.table = table
        self.state, self.why = "ready", "fake"
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.ids_by_text = {}

    def count(self, path, payload):
        n, delay = self.table[payload["tag"]]
        time.sleep(delay)
        return FT.Count(n=n, ids=np.arange(n, dtype=np.int32) + 10, ms=delay * 1000.0,
                        reused=0, encoded=n)

    def remember(self, text, ids):
        self.ids_by_text[text] = ids

    def ids_for(self, text):
        return self.ids_by_text.get(text)


def _front(table, flush_delay=0.05):
    with envs.FLLIPER_PDFLIP_FRONT_EXACT_TOKENS.override(True):
        f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="earlyflip",
                    store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
                    weight_chunks=2, tp_prefill_max_tokens=X, flip_min_work_tokens=X)
    f.state = "serving"
    f.ftok = _Tokens(table)
    f.calls, f.d_legs, f.p_legs, f.stops = [], [], [], []

    async def rpc(g, path, body, timeout):
        f.calls.append((g.name, path, time.monotonic()))
        if path == "/flush_cache":
            await asyncio.sleep(flush_delay)
            return 200, "{}"
        await asyncio.sleep(0.001)
        tags = tuple((body or {}).get("tags", ()))
        return 200, json.dumps({"per_tag": {t: [MIB, 1.0] for t in tags},
                                "critical_path": "rank=0 card=GPU-x ms=1"})

    async def seat(rid, est, refused=None, **kw):
        return 1

    async def leg2(request, rid, payload, text, stream, pending=None, **kw):
        f.d_legs.append((rid, f.awake, f.state, time.monotonic()))
        return F.web.json_response({})

    async def leg1(p):
        f.p_legs.append(p.rid)
        p.leg1_prompt_tokens = 64

    async def solo(rid, rem):
        return True

    f.rpc = rpc
    f.leg1 = leg1
    f.leg2 = leg2
    f._acquire_short_seat = seat
    f._x_solo_admits = solo
    f._kick_controller = lambda *a, **k: None
    f.do_stop = lambda name, detail: f.stops.append((name, detail))
    return f


def _payload(tag, chars):
    return {"model": "m", "max_tokens": 2, "tag": tag,
            "messages": [{"role": "user", "content": "a" * chars}]}


BIG = _payload("big", 3 * 3 * X)        # chars/3 ~ 3X: an early candidate


async def _run(f, payload, settle=0.8, cancel_after=None):
    t = asyncio.create_task(f.handle_generate(_Req(payload)))
    if cancel_after is not None:
        await asyncio.sleep(cancel_after)
        t.cancel()
    await asyncio.sleep(settle)
    if not t.done():
        t.cancel()
    await asyncio.gather(t, return_exceptions=True)
    h = f.__dict__.get("_early_flip_open")
    if h is not None and h.task is not None:
        await asyncio.gather(h.task, return_exceptions=True)


def _verdict_t(caplog):
    for r in caplog.records:
        if r.getMessage().startswith("PDFLIP ROUTE-VERDICT"):
            return r.created
    return None


def test_red_long_the_flip_runs_beside_the_pricing_and_completes(caplog, monkeypatch):
    monkeypatch.setenv(F.LEG1_EARLY_ENV, "1")
    f = _front({"big": (3 * X, 0.15)})
    epoch0 = f.epoch
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        asyncio.run(_run(f, BIG))
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("PDFLIP-EARLY-FLIP begin") for m in msgs), msgs
    flush = [c for c in f.calls if c[0] == "D" and c[1] == "/flush_cache"]
    assert flush, f.calls
    begin_rec = [r for r in caplog.records if r.getMessage().startswith("PDFLIP-FLIP begin")][0]
    assert begin_rec.created < _verdict_t(caplog), "the flip began only after the verdict"
    assert f.awake == "P" and f.epoch == epoch0 + 1 and f.stops == []
    go = [m for m in msgs if m.startswith("PDFLIP-EARLY-FLIP go")]
    assert go and int(go[0].split("saved_ms=")[1].split()[0]) > 0, go
    assert [p.rid for p in f.queue] == [f.p_legs[0]], (f.p_legs, [p.rid for p in f.queue])
    assert f.counters["early_flip_go"] == 1 and f.counters["early_flip_abort"] == 0
    assert any("LEG1-EARLY" in m and "EARLY-FLIP go" in m for m in msgs), msgs


def test_short_aborts_d_serves_the_state_is_clean_and_p_sees_nothing(caplog, monkeypatch):
    monkeypatch.setenv(F.LEG1_EARLY_ENV, "1")
    f = _front({"big": (3000, 0.15)})           # chars/3 says 3X, the exact count 3000 <= X
    epoch0 = f.epoch
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        asyncio.run(_run(f, BIG))
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("PDFLIP-EARLY-FLIP abort") and "why=verdict-not-long" in m for m in msgs), msgs
    assert f.awake == "D" and f.state == "serving" and f.epoch == epoch0
    assert f.counters["early_flip_abort"] == 1 and f.counters["early_flip_go"] == 0
    # D served the SHORT, awake and serving at its leg 2
    assert len(f.d_legs) == 1 and f.d_legs[0][1:3] == ("D", "serving"), f.d_legs
    # nothing slept, nothing went to P: no sleep RPC, no LEG1-EARLY
    assert not [c for c in f.calls if c[1] in ("/release_memory_occupation", "/resume_memory_occupation")]
    assert not [c for c in f.calls if c[0] == "P"] and f.p_legs == []
    assert f.queue == [] or len(f.queue) == 0
    # the flip's bookkeeping is closed
    assert getattr(f, "_flip_dst", None) is None and f._flip_t0 is None and f._flip_stage == "none"
    assert not getattr(f, "_flip_open", False) and f.__dict__.get("_early_flip_open") is None
    snap = F.Front._flip_phase(f).snap
    assert snap["phase"] is None and str(snap["last"]["aborted"]).startswith("early-flip"), snap


def test_rank_agreement_the_abort_waits_for_the_group_quiesce(caplog):
    """The verdict lands while D's quiesce (/flush_cache, one GROUP verdict over
    every rank, #1268) is still in flight: the abort comes only after it
    answered, and no group ever receives a sleep RPC -- no rank of D is left
    in a different state than the others."""
    f = _front({"big": (3000, 0.02)}, flush_delay=0.3)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        asyncio.run(_run(f, BIG, settle=1.0))
    paths = [(g, p) for g, p, _ in f.calls]
    assert paths == [("D", "/flush_cache")], paths
    abort = [r for r in caplog.records if r.getMessage().startswith("PDFLIP-EARLY-FLIP abort")][0]
    flush_t = [t for g, p, t in f.calls if p == "/flush_cache"][0]
    assert abort.relativeCreated / 1000.0 >= 0  # logged
    assert f.d_legs and f.d_legs[0][3] >= flush_t + 0.3 - 0.02, "the SHORT was routed before D's quiesce answered"


def test_switch_off_no_early_begin(caplog):
    f = _front({"big": (3 * X, 0.15)})
    with envs.FLLIPER_PDFLIP_EARLY_FLIP.override(False), caplog.at_level(logging.INFO, logger="pdflip.front"):
        asyncio.run(_run(f, BIG, settle=0.4))
    assert not f.calls and f.awake == "D" and len(f.queue) == 1
    assert not [r for r in caplog.records if "PDFLIP-EARLY-FLIP" in r.getMessage()]


def test_only_d_idle_and_only_far_over_x():
    f = _front({"big": (3 * X, 0.05)})
    f.groups["D"].outstanding["decoding"] = time.time()
    asyncio.run(_run(f, BIG, settle=0.3))
    assert f.counters["early_flip_begin"] == 0 and not f.calls
    g = _front({"mid": (3 * X, 0.05)})
    asyncio.run(_run(g, _payload("mid", 3 * X + 300), settle=0.3))   # chars/3 ~ 1.0X < 2X
    assert g.counters["early_flip_begin"] == 0 and not g.calls


def test_a_client_gone_before_the_verdict_aborts():
    f = _front({"big": (3 * X, 0.4)})
    asyncio.run(_run(f, BIG, settle=0.8, cancel_after=0.1))
    assert f.counters["early_flip_abort"] == 1 and f.awake == "D" and f.state == "serving"
    assert not [c for c in f.calls if c[1] == "/release_memory_occupation"]
