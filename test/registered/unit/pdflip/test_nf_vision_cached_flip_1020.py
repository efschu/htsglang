"""NF-VISION-CACHED (20.10.): a request whose context is cached is never classed on D's device rows -- no flip, no load.

User order 20.10. (verbatim): "WARUM SOLLTE IRGENDWAS FLIPPEN ODER LADEN WENN DAS BILD UND DER GANZE KONTEXT GECACHT IST".
SOLL R7 (no flip under X; a request D cannot take under X is a defect of D, no flip reason), R9 (p_only only for an image
in the UNCACHED tail), R2 (kurz to D at once), R1a (riders on a decided flip), R11 (the front decides alone).

SPECIMEN rc28 ``boot_weg2_..._297198745d_1020_174205``, front log, rid pdflip-4-226 (prompt 226683, store 226304):
  18:13:01,728 L3-INDEX-PRESENCE depth=226304 tokens=226683
  18:13:01,729 D-PROBE-SKIPPED p_uncached=379 d_extending=pdflip-4-223 (NF-1b: D extends, kurz from the store bound)
  18:13:01,730 CLASSIFY R2 kurz 379 X=10000 halfX=5000
  18:13:01,754 D-PROBE-LATE u_d=379 device=0 host=226304 store=226304 ok   <- D's exact price, LOGGED ONLY, never booked
  18:13:07,591 W50-REROUTE front_price=379 d_extent=226683 reason=x_refusal_midstream (D: D-WALL-HEAD need_rows=226683,
               payable_evictable=0 -- the backup wall; the extent is the DEVICE ROWS = the whole prompt)
  18:13:07,592 SET-VERDICT S=226683 classes=p_only COLLECT R4 ; 18:13:12,217 FLIP_P R4 -- P computed 379, flip 8.9 s.

THE ROOT, two links: (1) rc28 lacks S107 (the text RESUME-VIA-P leg was p_only by state with u = the rows); (2) with S107
alone the leg still classed on the rows (``u_src=d_rows``), because the skipped NF-1b probe left no price behind -- its
late answer was logged and dropped. FIX: the late answer is booked as the rid's exact price (``_probe_price`` + the S63
re-probe key) exactly as an answered arrival probe; without a D price the store depth the arrival READ prices the P leg
(P reads that store); without either the leg is UNPRICED (u=0: no set member, no flip reason) and a store probe prices it.
The rows are never a class.

Hermetic: no GPU, no model, no network.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
import time

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from flliper.srt.pdflip import resume_via_p as rvp  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-pdflip-unit")

RID = "pdflip-4-226"
N = 226683            # prompt tokens (L3-INDEX-PRESENCE tokens=226683)
STORE = 226304        # depth=226304
U = N - STORE         # 379
X = 10000             # SET-VERDICT X=10000 halfX=5000


@pytest.fixture
def d_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    return tmp_path


def _front():
    from test_ez_s1_review_1019 import _fresh
    f = _fresh()
    f.tag = "dkrtest"
    f.tp_prefill_max_tokens = X
    f._kick_controller = lambda why: f.__dict__.setdefault("_kicks", []).append(why)
    return f


def _answer(n, device, host=0, spans=()):
    return json.dumps({"success": True, "ids": dmp.ids_to_b64(np.arange(n)), "device_tokens": device,
                       "host_tokens": host, "key_match_depth": device, "mm_spans": [list(s) for s in spans]})


def _late(f, answer, *, n=N, store=STORE, up=U, ns_known=True):
    """The NF-1b skipped probe's answer lands after the verdict (the real ``_d_probe_late_note`` callback)."""
    async def run():
        fut = asyncio.get_running_loop().create_future()
        F.Front._d_probe_late_note(f, RID, fut, n, store, up, path="/v1/messages",
                                  payload={"messages": [{"role": "user", "content": "x"}]}, ns_known=ns_known, ns=None)
        fut.set_result((200, answer))
        await asyncio.sleep(0)
    asyncio.run(run())


def _take(f, await_reprice=False):
    async def body():
        got = f._rvp_take()
        if await_reprice:
            await f.__dict__["_rvp_reprice_tasks"][RID]
        return got
    return asyncio.run(body())


def _assert_no_flip(f, p):
    assert p.p_only is False
    assert pp.classify_arrival(p.est_uncached, X, 0.5, False) == pp.CLASS_KURZ
    st = f._set_st()
    assert F.Front._set_cands(f, [p], st) == []
    assert pp.immediate_park_trigger([p], X) is None
    with F.envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._decode_collect(time.time()) != "P"


# ---------------------------------------------------------------------------------------------------- the metal case
def test_rc28_4_226_the_late_probe_answer_is_booked_as_the_exact_price(d_env, caplog):
    f = _front()
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        _late(f, _answer(N, 0, STORE))
    xp = f._probe_price[RID]
    assert (int(xp.covered), dmp.class_uncached(xp), bool(xp.image_in_tail)) == (STORE, U, False)
    assert f.counters["d_probe_late_priced"] == 1 and f.counters["d_probe_late_ok"] == 1
    e = f._reprobe_book[RID]
    assert e.price is xp and len(e.ids) == N and e.path == "/v1/messages"
    assert "PDFLIP D-PROBE-LATE rid=pdflip-4-226 u_d=379 u_p=379" in caplog.text


def test_rc28_4_226_the_wall_refusal_then_classes_kurz_379_and_makes_no_flip(d_env, caplog):
    """End to end: late answer booked -> D-WALL-HEAD refusal of the held stream (need_rows=226683) -> u=379 kurz, no
    set member, no P trigger, no FLIP_P (before: S=226683 classes=p_only, R4 FLIP_P, P computed 379)."""
    f = _front()
    _late(f, _answer(N, 0, STORE))
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f) == 1
    p = f.queue[-1]
    assert p.resume_via_p and p.est_uncached == U
    _assert_no_flip(f, p)
    assert any(f"RESUME-VIA-P CLASS rid={RID} u=379 u_src=d_probe" in m for m in caplog.messages)
    assert "_rvp_reprice_tasks" not in f.__dict__, "a priced leg asks nothing more"


def test_streamed_output_on_top_of_the_late_price_counts_as_uncached(d_env):
    f = _front()
    _late(f, _answer(N, 0, STORE))
    rvp.write_request(RID, list(range(N + 40)), N + 40, X, rvp.REASON_WALL)
    assert _take(f) == 1
    assert f.queue[-1].est_uncached == U + 40


def test_an_unusable_late_answer_books_nothing(d_env):
    f = _front()
    _late(f, "not json")
    assert RID not in (f.__dict__.get("_probe_price") or {})
    assert f.counters.get("d_probe_late_priced", 0) == 0


def test_a_late_answer_with_an_image_in_the_uncached_tail_books_it_r9(d_env):
    """R9 unchanged: an image past the covered prefix is in the price (the class would be p_only on it)."""
    f = _front()
    _late(f, _answer(N, 0, STORE, spans=[(STORE + 10, N - 5)]))
    assert f._probe_price[RID].image_in_tail is True
    assert pp.classify_arrival(dmp.class_uncached(f._probe_price[RID]), X, 0.5, True) == pp.CLASS_P_ONLY


# ---------------------------------------------------------------------------------------------------- no D price
def test_no_d_price_the_store_depth_the_arrival_read_prices_the_leg(d_env, caplog):
    f = _front()
    f.__dict__["_za_presence"] = {RID: (STORE, N)}
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f) == 1
    p = f.queue[-1]
    assert p.est_uncached == U
    _assert_no_flip(f, p)
    assert any(f"RESUME-VIA-P CLASS rid={RID} u=379 u_src=p_store" in m for m in caplog.messages)


def test_no_number_at_all_is_unpriced_and_a_store_probe_prices_it(d_env, caplog):
    f = _front()
    asked = []

    async def _store(rid, ids, timeout_s, extra_key=None):
        asked.append((rid, int(len(ids))))
        return STORE, "l3_index"

    f._store_probe_depth = _store
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f, await_reprice=True) == 1
    p = f.queue[-1]
    assert asked == [(RID, N)]
    assert p.est_uncached == U and f.counters["rvp_text_unpriced"] == 1 and f.counters["rvp_repriced"] == 1
    assert "arrival" in f._kicks
    _assert_no_flip(f, p)
    assert any("u_src=unpriced" in m for m in caplog.messages)
    assert any(f"RESUME-VIA-P REPRICE rid={RID} u=379 u_src=p_store" in m for m in caplog.messages)
    assert not any("u_src=d_rows" in m for m in caplog.messages)


def test_unpriced_is_no_flip_reason_before_the_store_answers(d_env):
    f = _front()
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    assert _take(f) == 1
    p = f.queue[-1]
    assert p.est_uncached == 0
    _assert_no_flip(f, p)


def test_an_unread_store_leaves_it_unpriced_and_named(d_env, caplog):
    f = _front()

    async def _store(rid, ids, timeout_s, extra_key=None):
        return 0, "unread"

    f._store_probe_depth = _store
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f, await_reprice=True) == 1
    assert f.queue[-1].est_uncached == 0 and f.counters["rvp_reprice_unread"] == 1
    assert any("REPRICE rid=pdflip-4-226 tier=unread" in m for m in caplog.messages)


def test_a_store_that_holds_nothing_prices_the_whole_context_gross(d_env):
    """The reprice is exact both ways: nothing in the store -> P computes everything -> gross, P-bound."""
    f = _front()

    async def _store(rid, ids, timeout_s, extra_key=None):
        return 0, "none"

    f._store_probe_depth = _store
    f.store_probe = object()          # the probe is open: "none" is the store's exact answer
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    assert _take(f, await_reprice=True) == 1
    p = f.queue[-1]
    assert p.est_uncached == N
    assert pp.immediate_park_trigger([p], X) is p


def test_a_non_wall_x_refusal_keeps_ds_exact_extent(d_env):
    """D's X gate extent (no wall) is D's exact number after prefix matching: unchanged."""
    f = _front()
    f.__dict__["_za_presence"] = {RID: (STORE, N)}
    rvp.write_request(RID, list(range(20000)), 15077, X, "x_refusal_midstream")
    assert _take(f) == 1
    assert f.queue[-1].est_uncached == 15077


# ---------------------------------------------------------------------------------------------------- fresh (X-REQUEUE)
WALL_BODY = (b'{"error": "W50 PdFlipTpPrefillExceeded: this group may prefill at most 10000 uncached tokens itself; this '
             b'request\'s extent after prefix matching is 226683. Refused by name D-WALL-HEAD: the extent is the device '
             b'rows this request needs; the host arena takes no backup"}')


def test_fresh_wall_refusal_without_d_price_is_priced_from_the_store():
    f = _front()
    f.__dict__["_za_presence"] = {RID: (STORE, N)}
    rows = F._d_refusal_extent(WALL_BODY)
    assert rows == N
    assert F.Front._wall_extent_to_u(f, RID, WALL_BODY, rows, n_prompt=N) == U
    assert f.counters["wall_extent_to_u_store"] == 1
    # without the prompt length the rows stay, named (S107 behaviour)
    assert F.Front._wall_extent_to_u(f, RID, WALL_BODY, rows) == N


def test_the_requeue_path_hands_the_prompt_length_to_the_wall_conversion():
    src = inspect.getsource(F.Front._requeue_after_x_refusal)
    i = src.index("Front._wall_extent_to_u(self, rid, body, d_extent,")
    assert "n_prompt=" in src[i:i + 200]


def test_the_late_note_call_site_hands_the_key_through():
    src = inspect.getsource(F.Front._x_kurz_while_d_extends)
    i = src.index("Front._d_probe_late_note(")
    seg = src[i:i + 300]
    assert all(k in seg for k in ("path=path", "payload=payload", "ns_known=ns_known", "ns=ns"))


def test_a_store_probe_that_is_not_open_is_no_number(d_env, caplog):
    """(0, "none") from a store probe that is not open means "not askable": the leg stays unpriced, never gross on it."""
    f = _front()

    async def _store(rid, ids, timeout_s, extra_key=None):
        return 0, "none"

    f._store_probe_depth = _store
    f.store_probe = None
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f, await_reprice=True) == 1
    assert f.queue[-1].est_uncached == 0 and f.counters["rvp_reprice_unread"] == 1
