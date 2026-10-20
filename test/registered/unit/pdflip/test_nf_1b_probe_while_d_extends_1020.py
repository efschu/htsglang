"""NF-1b (20.10.): a kurz arrival is not held behind D's pass end for its D-MATCH-PROBE (SOLL R2 "sofort").

SPECIMENS (front logs under /spinning/docker-acceptance/nf/evidence/):
* rc27 ``..._893d09fa3f_1020_145331`` 15:10:02,051 ``D-PROBE rid=pdflip-4-53 n=165893 device=162944 store=162944
  tier=l2_arena uncached_d=2949 uncached_p=2949 ... probe_ms=1714.8``; CLASSIFY kurz X=8671 halfX=4335
  ``t_arrive_to_decide_ms=1932``. D was extending pdflip-4-52 (D-ADMIT 15:09:59,163; D log LOOP-ITER-SLOW
  15:10:02 ``run_ms=2289.1`` -- D's scheduler thread sat in that forward, its intake answers the probe after it).
* rc26 ``..._10d681a721_1020_143107`` 14:36:00,190 ``D-PROBE rid=pdflip-0-4 ... uncached_p=210 ... probe_ms=4175.4``.
Over all front logs of 20.10.: 2410 probes, p90 1086 ms, p99 3972 ms, 549 above 500 ms (687 s); 253 of those (392 s)
had an exact store number below halbX (``/tmp/nf1b/cover.py`` in the report).

The store number is EXACT (L2/L3, the request's namespace, D's ids for a text prompt), and D covers at least the store
(``covered = max(device + host, store)``), so ``u_d <= u_p < halbX`` is kurz whatever D answers -- the EZ-S1 ladder's
KIND_P_UPPER rung, whose leg carries the halfX-1 barrier (D refuses by name above it with its exact extent).

Hermetic: no GPU, no model, no network.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import types

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _front  # noqa: E402

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import d_extend_beacon as dxb  # noqa: E402
from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip import decision_numbers as dn  # noqa: E402
from flliper.srt.pdflip import front as front_mod  # noqa: E402
from flliper.srt.pdflip.front import Front  # noqa: E402

N_4_53 = 165893
STORE_4_53 = 162944
X_RC27 = 8671          # halfX 4335


def _fx(monkeypatch, *, extending_rest=2796, store=(STORE_4_53, "l2_arena")):
    f = _front(running=[])
    f.tp_prefill_max_tokens = X_RC27
    now = time.time()
    f.groups["D"].outstanding["pdflip-4-52"] = now - 2.9          # D-ADMIT 15:09:59,163
    f._d_ext_reading = lambda: {dxb.rid_key("pdflip-4-52"): (int((now - 2.8) * 1e9), int(extending_rest))}
    calls = []

    async def _store(rid, ids, timeout_s, extra_key=None):
        calls.append(rid)
        return store

    f._store_probe_depth = _store
    monkeypatch.setattr(Front, "_send_ids_note", lambda *a, **k: None)
    monkeypatch.delenv("FLLIPER_PDFLIP_D_EXTEND_BEACON", raising=False)
    return f, calls


def _count(n=N_4_53):
    return types.SimpleNamespace(n=n, ids=np.arange(n, dtype=np.int64), ms=1.0, reused=0, encoded=0)


def _payload():
    return {"messages": [{"role": "user", "content": "x"}], "max_tokens": 64}


def test_rc27_kurz_is_decided_while_d_extends_without_waiting_for_its_pass(monkeypatch, caplog):
    """THE REGRESSION: D extends pdflip-4-52, the probe of pdflip-4-53 cannot be answered before that pass ends; the
    store's exact u_p=2949 < halfX=4335 decides kurz NOW (KIND_P_UPPER, pending=u_p, D_NOW barrier)."""
    f, calls = _fx(monkeypatch)

    async def run():
        never = asyncio.get_running_loop().create_future()   # D inside its pass: no answer
        t0 = time.monotonic()
        got = await Front._x_kurz_while_d_extends(f, "pdflip-4-53", _count(), "/v1/messages", _payload(),
                                                  N_4_53, never)
        return got, time.monotonic() - t0

    with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
        got, waited = asyncio.run(run())
    assert got is not None, "the verdict did not wait for D's pass"
    assert waited < 0.2
    assert got.class_basis.kind == dn.KIND_P_UPPER and got.class_basis.klass == dn.CLASS_KURZ
    assert got.pending == N_4_53 - STORE_4_53 == 2949
    assert got.unpriced is True and got.depth_hint == STORE_4_53 and got.n_src == "front_ids"
    assert calls == ["pdflip-4-53"], "the store was READ for this request (an exact number)"
    assert f.counters["d_probe_skipped_d_extending"] == 1 and f.counters["d_probe_unpriced"] == 0
    assert "PDFLIP D-PROBE-SKIPPED rid=pdflip-4-53" in caplog.text and "d_extending=pdflip-4-52" in caplog.text
    assert "D-PROBE-UNPRICED" not in caplog.text


def test_d_not_extending_waits_for_d_exact_answer_as_before(monkeypatch):
    """Danger direction 1: D idle or decoding (its extend ENDED, rest 0 -- NF-R6-BEACON-DONE): no early verdict, D's
    exact answer decides, the default path is unchanged."""
    f, calls = _fx(monkeypatch, extending_rest=0)

    async def run():
        never = asyncio.get_running_loop().create_future()
        return await Front._x_kurz_while_d_extends(f, "r", _count(), "/v1/messages", _payload(), N_4_53, never)

    assert asyncio.run(run()) is None
    assert calls == [], "not even the store is asked"


def test_u_p_at_or_above_half_x_is_no_class_and_waits_for_d(monkeypatch):
    """Danger direction 2: u_p >= halbX is only an upper bound -- no class without D's number."""
    f, _ = _fx(monkeypatch, store=(N_4_53 - 4335, "l3_index"))

    async def run():
        never = asyncio.get_running_loop().create_future()
        return await Front._x_kurz_while_d_extends(f, "r", _count(), "/v1/messages", _payload(), N_4_53, never)

    assert asyncio.run(run()) is None
    assert f.counters["d_probe_skipped_d_extending"] == 0


def test_an_unread_store_is_no_number(monkeypatch):
    f, _ = _fx(monkeypatch, store=(0, "unread"))

    async def run():
        never = asyncio.get_running_loop().create_future()
        return await Front._x_kurz_while_d_extends(f, "r", _count(), "/v1/messages", _payload(), N_4_53, never)

    assert asyncio.run(run()) is None


def test_a_probe_that_already_answered_decides(monkeypatch):
    f, calls = _fx(monkeypatch)

    async def run():
        done = asyncio.get_running_loop().create_future()
        done.set_result((200, "{}"))
        return await Front._x_kurz_while_d_extends(f, "r", _count(), "/v1/messages", _payload(), N_4_53, done)

    assert asyncio.run(run()) is None and calls == []


def test_unknown_namespace_or_p_awake_never_skips(monkeypatch):
    f, _ = _fx(monkeypatch)
    f.awake = "P"
    assert Front._d_extending_rids(f) == []
    f.awake = "D"
    assert Front._d_extending_rids(f) == ["pdflip-4-52"]
    # an extend D began BEFORE this post (an older leg of the rid) is not this leg's
    f.groups["D"].outstanding["pdflip-4-52"] = time.time() + 5.0
    assert Front._d_extending_rids(f) == []


def _answer(n, device, host=0):
    return json.dumps({"success": True, "ids": dmp.ids_to_b64(np.arange(n)), "device_tokens": device,
                       "host_tokens": host, "key_match_depth": device, "mm_spans": []})


def test_the_late_probe_answer_is_compared_with_the_store_bound(monkeypatch, caplog):
    """Instrument: D's late exact answer beside the bound the class stood on -- ok when u_d <= u_p, named otherwise."""
    f, _ = _fx(monkeypatch)

    async def run(device):
        fut = asyncio.get_running_loop().create_future()
        got = await Front._x_kurz_while_d_extends(f, "pdflip-4-53", _count(), "/v1/messages", _payload(), N_4_53,
                                                  fut)
        fut.set_result((200, _answer(N_4_53, device)))
        await asyncio.sleep(0)
        return got

    with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
        assert asyncio.run(run(STORE_4_53)) is not None
    assert f.counters["d_probe_late_ok"] == 1
    assert "PDFLIP D-PROBE-LATE rid=pdflip-4-53 u_d=2949 u_p=2949" in caplog.text


def test_the_early_verdict_sits_before_the_probe_is_awaited():
    """Wiring: ``_x_exact_price`` asks the early verdict for a text prompt BEFORE it awaits D's probe result."""
    import inspect

    src = inspect.getsource(Front._x_exact_price)
    i_early = src.index("Front._x_kurz_while_d_extends(")
    i_await = src.index("await Front._d_probe_result(self, rid, _probe_task)")
    assert i_early < i_await
