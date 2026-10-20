"""NF-VISION-CACHED (20.10.): a request whose context is cached is never classed on D's device rows -- no flip, no load.

User order 20.10. (verbatim): "WARUM SOLLTE IRGENDWAS FLIPPEN ODER LADEN WENN DAS BILD UND DER GANZE KONTEXT GECACHT IST".
User 19.10.: "NICHTS SCHÄTZEN, EXAKTE ZAHLEN" (the class from D's radix / L2 / L3 at the decision). SOLL R7 (no flip under
X; a request D cannot take under X is a defect of D, no flip reason), R9 (p_only only for an image in the UNCACHED tail),
R2 (kurz to D at once), R1a (riders on a decided flip), R11 (the front decides alone).

SPECIMEN rc28 ``boot_weg2_..._297198745d_1020_174205``, front log, rid pdflip-4-226 (prompt 226683, store 226304):
  18:13:01,728 L3-INDEX-PRESENCE depth=226304 tokens=226683
  18:13:01,729 D-PROBE-SKIPPED p_uncached=379 d_extending=pdflip-4-223 (NF-1b: D extends, kurz from the store bound)
  18:13:01,754 D-PROBE-LATE u_d=379 device=0 host=226304 store=226304 ok   <- D's exact price, LOGGED ONLY, never booked
  18:13:07,591 W50-REROUTE front_price=379 d_extent=226683 reason=x_refusal_midstream (D: D-WALL-HEAD need_rows=226683,
               payable_evictable=0 -- the backup wall; the extent is the DEVICE ROWS = the whole prompt)
  18:13:07,592 SET-VERDICT S=226683 classes=p_only COLLECT R4 ; 18:13:12,217 FLIP_P R4 -- P computed 379, flip 8.9 s.

FIX (73c172d901 + review 20.10. A1-A3, B5):
  * the late NF-1b answer is booked as the rid's exact D price (only when D answered for the front's n, B5);
  * a wall refusal's INTERIM class (``Front._wall_interim``): the rid's D price or the store bound u_p = n - store may
    only say kurz (below halbX of the X at that depth, the EZ-S1 ladder ``class_basis``); anything else is UNPRICED (u=0:
    no member, no flip reason) -- u_p is an upper bound (D reads store + radix) and the arrival's store depth is old (A1);
  * D's exact number NOW (``Front._rvp_reprice``): a D probe on the entry's own ids -- the leg's CONTEXT ids, output
    included -- plus the store; it becomes the entry's u and the rid's S63 key (A2); without an answer it is asked again
    while the entry waits, each failed round named with its age (A3: ``RESUME-VIA-P REPRICE-FAILED ... age_s=``);
  * a re-probe round on another key (the arrival's prompt ids) never touches the leg (A2).

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

register_cpu_ci(est_time=15, suite="stage-a-pdflip-unit")

RID = "pdflip-4-226"
N = 226683            # prompt tokens (L3-INDEX-PRESENCE tokens=226683)
STORE = 226304        # depth=226304
U = N - STORE         # 379
X = 10000             # SET-VERDICT X=10000 halfX=5000
LAG = 20000           # a store depth that lies LAG below D's tree (u_p = LAG >= X while D's u is U)


@pytest.fixture
def d_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    monkeypatch.setattr(F.Front, "RVP_REPRICE_RETRY_S", 0.0)
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


def _ans(n, device, host=0):
    return dmp.parse_answer(json.loads(_answer(n, device, host)))


def _late(f, answer, *, n=N, store=STORE, up=U, ns_known=True):
    """The NF-1b skipped probe's answer lands after the verdict (the real ``_d_probe_late_note`` callback)."""
    async def run():
        fut = asyncio.get_running_loop().create_future()
        F.Front._d_probe_late_note(f, RID, fut, n, store, up, path="/v1/messages",
                                  payload={"messages": [{"role": "user", "content": "x"}]}, ns_known=ns_known, ns=None)
        fut.set_result((200, answer))
        await asyncio.sleep(0)
    asyncio.run(run())


def _fake_d(monkeypatch, answers):
    """D's probe: ``answers`` popped per call (None = no exact number by the deadline). Returns the calls."""
    calls = []

    def start(self, rid, path, payload, timeout_s=None):
        calls.append((rid, path, payload))
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    async def result(self, rid, task):
        return answers.pop(0) if answers else None

    monkeypatch.setattr(F.Front, "_d_probe_start", start)
    monkeypatch.setattr(F.Front, "_d_probe_result", result)
    return calls


def _store(f, depth, tier="l3_index"):
    asked = []

    async def st(rid, ids, timeout_s, extra_key=None):
        asked.append((rid, int(len(ids)), extra_key))
        return depth, tier

    f._store_probe_depth = st
    f.store_probe = object()
    return asked


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


# ---------------------------------------------------------------------------------------------------- the late NF-1b answer
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


def test_a_late_answer_for_another_n_is_not_booked_b5(d_env):
    f = _front()
    _late(f, _answer(N + 7, 0, STORE))
    assert RID not in (f.__dict__.get("_probe_price") or {})
    assert f.counters["d_probe_late_unbooked"] == 1 and f.counters["d_probe_late_violation"] == 1


def test_an_unusable_late_answer_books_nothing(d_env):
    f = _front()
    _late(f, "not json")
    assert RID not in (f.__dict__.get("_probe_price") or {})
    assert f.counters.get("d_probe_late_priced", 0) == 0


def test_a_late_answer_with_an_image_in_the_uncached_tail_books_it_r9(d_env):
    f = _front()
    _late(f, _answer(N, 0, STORE, spans=[(STORE + 10, N - 5)]))
    assert f._probe_price[RID].image_in_tail is True
    assert pp.classify_arrival(dmp.class_uncached(f._probe_price[RID]), X, 0.5, True) == pp.CLASS_P_ONLY


# ---------------------------------------------------------------------------------------------------- the metal case end to end
def test_rc28_4_226_the_wall_refusal_classes_kurz_379_and_d_confirms_it_now(d_env, monkeypatch, caplog):
    """Late answer booked -> D-WALL-HEAD refusal of the held stream (need_rows=226683) -> interim kurz 379 on the price,
    then D's exact number on the leg's context ids confirms 379: no set member, no P trigger, no FLIP_P (before: S=226683
    classes=p_only, R4 FLIP_P, P computed 379)."""
    f = _front()
    _late(f, _answer(N, 0, STORE))
    calls = _fake_d(monkeypatch, [_ans(N, 0, STORE)])
    asked = _store(f, STORE)
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f, await_reprice=True) == 1
    p = f.queue[-1]
    assert p.resume_via_p and p.est_uncached == U
    _assert_no_flip(f, p)
    assert any(f"RESUME-VIA-P CLASS rid={RID} u=379 u_src=d_probe class=kurz" in m for m in caplog.messages)
    assert any(f"RESUME-VIA-P REPRICE rid={RID} site=rvp_leg u 379 -> 379 class=kurz" in m for m in caplog.messages)
    # D was asked on the leg's own ids (D's token space, no tokenization), the store on the same ids
    assert calls[0][1] == "/generate" and "input_ids_b64" in calls[0][2]
    assert asked == [(RID, N, None)]
    assert f._reprobe_book[RID].n == N and f._reprobe_book[RID].price is f._probe_price[RID]
    assert RID not in f.__dict__["_rvp_reprice_tasks"], "B5: the task book is emptied when the task ends"


def test_streamed_output_on_top_of_the_late_price_counts_as_uncached(d_env):
    f = _front()
    _late(f, _answer(N, 0, STORE))
    rvp.write_request(RID, list(range(N + 40)), N + 40, X, rvp.REASON_WALL)
    assert _take(f) == 1
    assert f.queue[-1].est_uncached == U + 40


# ---------------------------------------------------------------------------------------------------- A1: no class on a bound
def test_the_store_bound_below_halfx_is_kurz_p_upper(d_env, caplog):
    f = _front()
    f.__dict__["_za_presence"] = {RID: (STORE, N)}
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f) == 1
    p = f.queue[-1]
    assert p.est_uncached == U
    _assert_no_flip(f, p)
    assert any(f"RESUME-VIA-P CLASS rid={RID} u=379 u_src=p_upper class=kurz" in m for m in caplog.messages)


def test_a1_the_store_bound_at_or_above_halfx_is_no_class_unpriced(d_env, caplog):
    """u_p = n - arrival store depth = LAG >= X: an upper bound, no number for the class -- unpriced, no P trigger."""
    f = _front()
    f.__dict__["_za_presence"] = {RID: (N - LAG, N)}
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f) == 1
    p = f.queue[-1]
    assert p.est_uncached == 0
    _assert_no_flip(f, p)
    assert any(f"RESUME-VIA-P CLASS rid={RID} u=0 u_src=unpriced class=unpriced" in m for m in caplog.messages), \
        "B5: the line says unpriced, never kurz"


def test_a1_an_old_d_price_at_or_above_halfx_is_no_class_unpriced(d_env):
    f = _front()
    _late(f, _answer(N, 0, N - 15000), store=N - 15000, up=15000)  # D's number at ITS time: 15000 (gross), old now
    assert dmp.class_uncached(f._probe_price[RID]) == 15000
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    assert _take(f) == 1
    assert f.queue[-1].est_uncached == 0


def test_a1_d_now_prices_the_leg_below_the_store_bound(d_env, monkeypatch):
    """The store lags D's tree by LAG: u_p >= X, D's exact u = U -- the class is D's number NOW, kurz."""
    f = _front()
    _fake_d(monkeypatch, [_ans(N, 0, STORE)])
    _store(f, N - LAG)
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    assert _take(f, await_reprice=True) == 1
    p = f.queue[-1]
    assert p.est_uncached == U and int(p.est_uncached_p) == LAG
    _assert_no_flip(f, p)


def test_d_now_prices_both_ways_nothing_cached_is_gross(d_env, monkeypatch):
    """Exact both ways: D's tree and the store hold nothing -> P computes the whole context -> gross, P-bound."""
    f = _front()
    _fake_d(monkeypatch, [_ans(N, 0, 0)])
    _store(f, 0, "none")
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    assert _take(f, await_reprice=True) == 1
    p = f.queue[-1]
    assert p.est_uncached == N
    assert pp.immediate_park_trigger([p], X) is p


# ---------------------------------------------------------------------------------------------------- A3: no number -> asked again
def test_a3_no_d_answer_is_asked_again_and_named_with_its_age(d_env, monkeypatch, caplog):
    f = _front()
    calls = _fake_d(monkeypatch, [None, _ans(N, 0, STORE)])
    _store(f, STORE)
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f, await_reprice=True) == 1
    assert len(calls) == 2
    assert f.queue[-1].est_uncached == U
    assert f.counters["rvp_reprice_failed"] == 1 and f.counters["rvp_repriced"] == 1
    assert any(f"RESUME-VIA-P REPRICE-FAILED rid={RID} site=rvp_leg attempt=1 why=no exact D number" in m
               and "age_s=" in m and "next=retry_s=" in m for m in caplog.messages)


def test_an_answer_for_other_ids_is_no_number_for_the_leg(d_env, monkeypatch, caplog):
    """Identity: D's answer must be for the leg's own ids (n and tokens) -- an answer for the prompt only is no number."""
    f = _front()
    _fake_d(monkeypatch, [_ans(N, 0, STORE), _ans(N + 5, 0, STORE)])
    _store(f, STORE)
    rvp.write_request(RID, list(range(N + 5)), N + 5, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f, await_reprice=True) == 1
    assert f.queue[-1].est_uncached == U + 5 and f.counters["rvp_reprice_failed"] == 1
    assert any("why=identity: D answered n=226683 for the leg's n=226688" in m for m in caplog.messages)


def test_a3_a_store_probe_that_is_not_open_is_no_number_and_is_asked_again(d_env, monkeypatch, caplog):
    """(0, "none") from a store probe that is not open is "not askable" -- never "the store holds nothing" (that would
    price the whole context gross on nothing): the round fails, named, and the next one prices it."""
    f = _front()
    _fake_d(monkeypatch, [_ans(N, 0, 0), _ans(N, 0, 0)])
    seen = []

    async def st(rid, ids, timeout_s, extra_key=None):
        seen.append(1)
        if len(seen) <= 2:
            f.store_probe = None
            return 0, "none"
        f.store_probe = object()
        return STORE, "l3_index"

    f._store_probe_depth = st
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f, await_reprice=True) == 1
    assert f.queue[-1].est_uncached == U
    assert any("why=store unread twice" in m for m in caplog.messages)


def test_a3_probe_off_stops_named_and_the_entry_stays_unpriced(d_env, caplog):
    f = _front()                      # no session: D's probe cannot be asked
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f, await_reprice=True) == 1
    p = f.queue[-1]
    assert p.est_uncached == 0
    _assert_no_flip(f, p)
    assert any("REPRICE-FAILED rid=pdflip-4-226" in m and "next=stop" in m for m in caplog.messages)


def test_a3_the_retry_ends_when_the_entry_leaves(d_env, monkeypatch, caplog):
    f = _front()
    n_calls = []

    def start(self, rid, path, payload, timeout_s=None):
        n_calls.append(1)
        if len(n_calls) == 2:
            f.queue.clear()           # a decided flip took it (R1a) / the stream ended
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    async def result(self, rid, task):
        return None

    monkeypatch.setattr(F.Front, "_d_probe_start", start)
    monkeypatch.setattr(F.Front, "_d_probe_result", result)
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    with caplog.at_level(logging.INFO):
        assert _take(f, await_reprice=True) == 1
    assert len(n_calls) == 2 and f.counters["rvp_reprice_gone"] == 1
    assert any("RESUME-VIA-P REPRICE-END rid=pdflip-4-226" in m for m in caplog.messages)


# ---------------------------------------------------------------------------------------------------- A2: the leg's own key
def test_a2_an_arrival_key_round_never_drops_the_legs_output(d_env, monkeypatch):
    f = _front()
    _late(f, _answer(N, 0, STORE))
    arrival = f._reprobe_book[RID]
    out = 2 * X
    _fake_d(monkeypatch, [_ans(N + out, 0, STORE)])     # D NOW on the context ids: the output is not in its tree
    _store(f, STORE)
    rvp.write_request(RID, list(range(N + out)), N + out, X, rvp.REASON_WALL)
    assert _take(f, await_reprice=True) == 1
    p = f.queue[-1]
    assert p.est_uncached == U + out                    # gross: a member
    assert f._reprobe_book[RID] is not arrival and f._reprobe_book[RID].n == N + out
    # a round asked on the ARRIVAL key lands late: it touches nothing of the leg
    assert F.Front._set_reprobe_apply(f, RID, arrival, _ans(N, 0, STORE), STORE, time.monotonic(), "late") is False
    assert F.Front._set_reprobe_unpriced(f, RID, arrival, "no exact D number by the deadline", "late") == "rvp_own_key"
    assert p.est_uncached == U + out and f.counters["set_reprobe_rvp_other_key"] == 2
    assert f._reprobe_book[RID].n == N + out, "the leg's own key stays"
    # a round on the leg's OWN key applies (D's coverage grew: the output got written back)
    leg = f._reprobe_book[RID]
    F.Front._set_reprobe_apply(f, RID, leg, _ans(N + out, 0, N + out - 100), STORE, time.monotonic(), "own")
    assert p.est_uncached == 100


def test_a2_the_arrival_key_leaves_the_book_at_the_reroute(d_env):
    f = _front()
    _late(f, _answer(N, 0, STORE))
    rvp.write_request(RID, list(range(N + 5)), N + 5, X, rvp.REASON_WALL)
    assert _take(f) == 1
    assert RID not in f._reprobe_book


# ---------------------------------------------------------------------------------------------------- non-wall refusals
def test_a_non_wall_x_refusal_keeps_ds_exact_extent(d_env):
    """D's X gate extent (no wall) is D's own measurement at the refusal: unchanged, no reprice."""
    f = _front()
    f.__dict__["_za_presence"] = {RID: (STORE, N)}
    rvp.write_request(RID, list(range(20000)), 15077, X, "x_refusal_midstream")
    assert _take(f) == 1
    assert f.queue[-1].est_uncached == 15077
    assert not f.__dict__.get("_rvp_reprice_tasks")


# ---------------------------------------------------------------------------------------------------- fresh (X-REQUEUE)
WALL_BODY = (b'{"error": "W50 PdFlipTpPrefillExceeded: this group may prefill at most 10000 uncached tokens itself; this '
             b'request\'s extent after prefix matching is 226683. Refused by name D-WALL-HEAD: the extent is the device '
             b'rows this request needs; the host arena takes no backup"}')


def test_fresh_wall_refusal_classes_kurz_on_the_store_bound_below_halfx():
    f = _front()
    f.__dict__["_za_presence"] = {RID: (STORE, N)}
    rows = F._d_refusal_extent(WALL_BODY)
    assert rows == N
    assert F.Front._wall_extent_to_u(f, RID, WALL_BODY, rows, n_prompt=N) == U
    assert f._wall_fresh[RID] == "p_upper" and f.counters["wall_extent_p_upper"] == 1


def test_fresh_wall_refusal_never_classes_on_the_rows_or_the_bound():
    f = _front()
    f.__dict__["_za_presence"] = {RID: (N - LAG, N)}
    assert F.Front._wall_extent_to_u(f, RID, WALL_BODY, N, n_prompt=N) == 0
    g = _front()
    assert F.Front._wall_extent_to_u(g, RID, WALL_BODY, N) == 0      # nothing at all: unpriced, never the rows
    assert f._wall_fresh[RID] == g._wall_fresh[RID] == "unpriced"


def test_the_requeue_path_prices_a_wall_refusal_now_and_forms_no_class_on_unpriced():
    src = inspect.getsource(F.Front._requeue_after_x_refusal)
    i = src.index("Front._wall_extent_to_u(self, rid, body, d_extent,")
    assert "n_prompt=" in src[i:i + 200]
    assert "None if _wall_src == \"unpriced\" else d_extent" in src
    i_q = src.index("self.queue.append(p)\n        self._dp_mark(p, \"reroute\")")
    assert "site=\"fresh_wall\"" in src[i_q:i_q + 700] and "if _wall_src is not None:" in src[i_q:i_q + 700]


def test_the_late_note_call_site_hands_the_key_through():
    src = inspect.getsource(F.Front._x_kurz_while_d_extends)
    i = src.index("Front._d_probe_late_note(")
    seg = src[i:i + 300]
    assert all(k in seg for k in ("path=path", "payload=payload", "ns_known=ns_known", "ns=ns"))
