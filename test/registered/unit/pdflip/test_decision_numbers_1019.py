"""EZ-S1 (plan nf-entscheidungszentrale-plan-1019, step S1; SOLL R2/R3/R4/R7/R9/R11; user 19.10. "nichts schaetzen"):
the numbers a decision stands on.

(a)  repro rc16 pdflip-140-474: probe dead, store u_p=28229 >= halbX -> MEASURE_ON_D (no D_NOW, not "gross on u_p");
     the fake D refuses with its exact extent 28229 -> class gross, rule R4, DECISION-DONE measured_u=28229
(a2) u_p < halbX with D ids -> kurz (the safe direction); the same with media (no D ids) -> MEASURE
(b)  media unpriced: n_est shows up in no decision field -- not as ``exact``, not as ``carrier_est``
(c)  chars/3 set to 10**9: no class changes
(f)  one snapshot, no await between read and commit; the epoch check refuses a stale commit
(g)  MEASURE border: u = halbX-1 admitted, u = halbX refused -- on the front's own header path AND on every D path that
     resolves the limit (X gate, S59 re-read, CAPACITY-PARK, norecompute: all through ``front_x.resolve_limit``)
(h)  KV freshness: the fetch starts at the end of P->D, the refresh runs only while D is awake, a reading older than
     the bound is ABSENT (``kv=absent``), never a number
switch off (FLLIPER_PDFLIP_EXACT_CLASS=0): the rc17 classification of an unpriced arrival, byte for byte (golden)
"""
import asyncio
import inspect
import json
import logging
import os
import re
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FLLIPER_PDFLIP_D_MATCH_PROBE_DEADLINE_MS", "150")
os.environ.setdefault("FLLIPER_PDFLIP_D_MATCH_PROBE_RETRY_MS", "10")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import decision_numbers as dn  # noqa: E402
from flliper.srt.pdflip import decision_record as dr  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import front_x  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from test_pdflip_front_x_exact import X, _FakeTokens, _front  # noqa: E402

N = 31301
U_P = 28229
PAGE = 64
HALF_X = pp.half_x_of(X, 0.5)   # 2048


def _text_payload():
    return {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hello " * 20}]}


def _img_payload():
    return {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "a" * 200},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]}


class _R:
    def __init__(self, p):
        self._p, self.path = p, "/v1/messages"

    async def json(self):
        return self._p


def _mk(monkeypatch, *, store_u_p=U_P, n=N, probe_dead=True):
    f = _front(True)
    f.ftok = _FakeTokens(n)
    f.session = object()
    f._store_probe_info = {"page_size": PAGE}
    f._vision_warm = lambda *a, **k: None
    f.vision = F.VISION_MODE_TRANSIENT
    monkeypatch.setattr(F.Front, "_p_tok_ahead_maybe", staticmethod(lambda *a, **k: None))
    f.legs = []

    async def leg2(request, rid, payload, text, stream, pending=None, **kw):
        f.routed.append(("short", rid))
        f.legs.append((rid, pending))
        return F.web.json_response({})

    f.leg2 = leg2

    async def post(g, b, timeout):
        return 503, "KeyError: 'pdflip-0-1~probe'"          # rc16: every probe dead

    f._d_probe_post = post

    async def st(rid, ids, timeout_s, extra_key=None):
        return (max(0, n - store_u_p), "l3_index")

    f._store_probe_depth = st
    return f


def _drive(f, payload, wait=0.5):
    async def go():
        t = asyncio.create_task(f.handle_generate(_R(payload)))
        await asyncio.sleep(wait)
        if not t.done():
            t.cancel()
            try:
                await t
            except BaseException:  # noqa: BLE001
                pass
    asyncio.run(go())


def _decisions(caplog):
    out = []
    for r in caplog.records:
        m = r.getMessage()
        if m.startswith(dr.DECISION_PREFIX) or dr.DONE_PREFIX in m or dr.SUM_PREFIX in m:
            out.append(m)
    return out


def _parsed(caplog, tag="DECISION"):
    res = []
    for m in _decisions(caplog):
        d = dr.parse_line(m)
        if d is not None and d["tag"] == tag:
            res.append(d)
    return res


# ------------------------------------------------------------------------------------------- pure ladder (B1, A1)

def _ex(v, src="d_probe", age_ms=0.0, now=100.0):
    return dn.Exact(v, src, now - age_ms / 1000.0)


def test_ladder_fresh_u_d_decides_alone():
    now = 100.0
    b = dn.class_basis(u_d=_ex(100, now=now), u_p=_ex(9000, "p_store", now=now), n_from_d_ids=True, x_tokens=X,
                       half_x=HALF_X, now=now)
    assert (b.kind, b.u, b.klass) == (dn.KIND_EXACT, 100, dn.CLASS_KURZ)
    b = dn.class_basis(u_d=_ex(3000, now=now), u_p=None, n_from_d_ids=True, x_tokens=X, half_x=HALF_X, now=now)
    assert (b.kind, b.klass) == (dn.KIND_EXACT, dn.CLASS_HALB)
    b = dn.class_basis(u_d=_ex(X, now=now), u_p=None, n_from_d_ids=True, x_tokens=X, half_x=HALF_X, now=now)
    assert b.klass == dn.CLASS_GROSS


def test_ladder_stale_u_d_never_classifies():
    now = 100.0
    b = dn.class_basis(u_d=_ex(100, age_ms=900, now=now), u_p=None, n_from_d_ids=True, x_tokens=X, half_x=HALF_X,
                       now=now, fresh_ms=500)
    assert b.kind == dn.KIND_MEASURE and b.klass is None


def test_ladder_u_p_only_downwards():
    now = 100.0
    low = dn.class_basis(u_d=None, u_p=_ex(HALF_X - 1, "p_store", now=now), n_from_d_ids=True, x_tokens=X,
                         half_x=HALF_X, now=now)
    assert (low.kind, low.u, low.klass) == (dn.KIND_P_UPPER, HALF_X - 1, dn.CLASS_KURZ)
    for u_p in (HALF_X, HALF_X + 1, X - 1, X, 28229, 10 ** 9):          # M1: the upper bound is no class
        b = dn.class_basis(u_d=None, u_p=_ex(u_p, "p_store", now=now), n_from_d_ids=True, x_tokens=X, half_x=HALF_X,
                           now=now)
        assert b.kind == dn.KIND_MEASURE and b.klass is None and b.u is None, (u_p, b)


def test_ladder_media_without_d_ids_has_no_upper_bound():
    now = 100.0
    b = dn.class_basis(u_d=None, u_p=_ex(10, "p_store", now=now), n_from_d_ids=False, x_tokens=X, half_x=HALF_X,
                       now=now)
    assert b.kind == dn.KIND_MEASURE


def test_ladder_unknown_is_never_kurz_and_never_gross():
    b = dn.class_basis(u_d=None, u_p=None, n_from_d_ids=True, x_tokens=X, half_x=HALF_X, now=100.0)       # M2
    assert b.kind == dn.KIND_MEASURE and b.klass not in (dn.CLASS_KURZ, dn.CLASS_GROSS, dn.CLASS_HALB)


def test_the_class_arithmetic_is_the_phase_policys():
    for u in (0, 1, HALF_X - 1, HALF_X, X - 1, X, X + 1):
        assert dn.classify(u, X, HALF_X) == pp.classify_arrival(u, X, 0.5, False)


# ------------------------------------------------------------------------------------------- (g) the border

@pytest.mark.parametrize("static_x", [X, 12288])
@pytest.mark.parametrize("page", [1, 64, 128])
def test_g_measure_border_is_halfx_minus_one_on_every_d_path(static_x, page):
    hx = pp.half_x_of(X, 0.5)
    fx = dn.measure_front_x(hx, page)
    assert fx + page == hx - 1
    hdr = front_x.request_headers(fx, 0, hx, action="MEASURE_ON_D")
    assert hdr[front_x.HEADER_ACTION] == "MEASURE_ON_D" and hdr[front_x.HEADER_HALFX] == str(hx)
    # the scheduler stamps the header values on the Req; every D path resolves the limit through ``resolve_limit``
    req = SimpleNamespace(output_ids=[], **{front_x.ATTR_X: int(hdr[front_x.HEADER_X]),
                                           front_x.ATTR_PRICED: 0, front_x.ATTR_HALFX: hx})
    limit, got_fx, _p = front_x.resolve_limit(req, static_x, page)
    assert limit == hx - 1 == dn.measure_limit(hx) and got_fx == fx
    assert not (hx - 1 > limit), "u = halbX-1 is admitted (the gate refuses uncached > limit)"
    assert hx > limit, "u = halbX is refused with its exact extent"


def test_g_every_d_refusal_path_reads_the_limit_through_resolve_limit():
    """S59 re-read, CAPACITY-PARK, RESUME-VIA-P and the norecompute divert all take their bound from the one
    function the border test above exercises (a second reader of the header would be a second border)."""
    root = os.path.join(os.path.dirname(inspect.getsourcefile(F)), "..", "managers", "scheduler.py")
    src = open(root).read()
    assert src.count("_pdflip_front_x.resolve_limit(") >= 3
    nr = open(os.path.join(os.path.dirname(inspect.getsourcefile(F)), "d_norecompute.py")).read()
    assert "resolve_limit(" in nr


# ------------------------------------------------------------------------------------------- (a) 140-474 e2e

def test_a_140_474_probe_dead_u_p_over_halfx_is_measure_then_d_extent_decides(monkeypatch, caplog):
    f = _mk(monkeypatch)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, _text_payload())
    msgs = [r.getMessage() for r in caplog.records]
    assert any("D-PROBE-UNPRICED" in m and "n_src=front_ids" in m and "p_uncached=%d" % U_P in m for m in msgs)
    dec = _parsed(caplog)
    meas = [d for d in dec if d["kind"] == "measure" and d["action"] == "MEASURE_ON_D"]
    assert len(meas) == 1 and meas[0]["rule"] == "NS:MEASURE" and meas[0]["classes"] == ["unknown"]
    assert meas[0]["members"][0].u_own == -1 and meas[0]["members"][0].src == "none"
    rid = meas[0]["rid"]
    # not D_NOW (no R2 claim on a number nobody measured), and not "gross on u_p"
    assert not any("PDFLIP SET-VERDICT" in m and "verdict=D_NOW" in m for m in msgs)
    assert not any(d["kind"] == "arrival" and "gross" in d["classes"] for d in dec)
    assert f.routed and f.routed[0] == ("short", rid), "D's admission measures it (leg 2)"
    assert f.counters["measure_on_d"] == 1 and f.counters["exact_class_measure"] == 1
    # the barrier rides with the leg
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        hdr = f._front_x_headers(rid, None, False)
    assert hdr[front_x.HEADER_ACTION] == "MEASURE_ON_D"
    assert front_x.limit_for(X, int(hdr[front_x.HEADER_X]), PAGE) == HALF_X - 1
    # D refuses with its exact extent
    body = (b'{"error": "W50 PdFlipTpPrefillExceeded: this request\'s extent after prefix matching is %d. Re-routed"}'
            % U_P)

    async def run():
        t = asyncio.create_task(f._requeue_after_x_refusal(SimpleNamespace(path="/v1/messages"), rid, _text_payload(),
                                                           "t", False, None, None, body))
        for _ in range(200):
            if f.queue:
                break
            await asyncio.sleep(0.005)
        q = list(f.queue)
        t.cancel()
        try:
            await t
        except BaseException:  # noqa: BLE001
            pass
        return q

    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        q = asyncio.run(run())
    assert len(q) == 1 and q[0].est_uncached == U_P and q[0].measured_u == U_P and q[0].measure_src == "d_extent"
    assert not q[0].p_only, "never p_only on a measured size (W123 is the only R9 door here)"
    dec = _parsed(caplog)
    ret = [d for d in dec if d["kind"] == "measure" and d["action"] == "COLLECT"]
    assert len(ret) == 1 and ret[0]["rule"] == "R4" and ret[0]["classes"] == ["gross"]
    assert ret[0]["members"][0].u_own == U_P and ret[0]["members"][0].src == "d_extent"
    done = _parsed(caplog, "DECISION-DONE")
    assert [d for d in done if d["measured_u"] == U_P and d["src"] == "d_extent" and d["outcome"] == "ok"]
    assert [d for d in done if d["executor"] == "front" and d["seq"] == meas[0]["seq"]], "the barrier is DONE-reported"


def test_a2_u_p_below_halfx_with_d_ids_is_kurz_the_safe_direction(monkeypatch, caplog):
    f = _mk(monkeypatch, store_u_p=1000)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, _text_payload())
    dec = _parsed(caplog)
    arr = [d for d in dec if d["kind"] == "arrival" and d["action"] == "CLASSIFY"]
    assert len(arr) == 1 and arr[0]["classes"] == ["kurz"] and arr[0]["rule"] == "R2"
    assert arr[0]["members"][0].u_own == 1000 and arr[0]["members"][0].src == "p_store"
    assert not [d for d in dec if d["kind"] == "measure"]
    assert f.routed and f.counters["exact_class_p_upper"] == 1
    assert "X-ROUTE" in "".join(r.getMessage() for r in caplog.records)


def test_a2_media_without_d_ids_is_measure_even_when_u_p_is_small(monkeypatch, caplog):
    f = _mk(monkeypatch, store_u_p=10)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, _img_payload())
    dec = _parsed(caplog)
    assert [d for d in dec if d["kind"] == "measure" and d["action"] == "MEASURE_ON_D"]
    assert not [d for d in dec if d["kind"] == "arrival" and d["classes"] == ["kurz"]]


# ------------------------------------------------------------------------------------------- (b), (c)

def test_b_media_unpriced_n_est_is_no_exact_and_no_carrier_est(monkeypatch, caplog):
    f = _mk(monkeypatch)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, _img_payload())
    rv = [r.getMessage() for r in caplog.records if "ROUTE-VERDICT" in r.getMessage()]
    assert rv and "src=estimate" in rv[0], rv
    assert f.counters["n_est_not_exact"] == 1
    xx = f._x_unpriced("r", None, 5000, -1, True)
    assert xx.n_src == "n_est" and xx.n == 5000
    for d in _parsed(caplog):                                    # n_est is in no decision field
        assert all(m.u_own != 5000 for m in d["members"])


def test_b_switch_off_is_the_rc17_form(monkeypatch, caplog):
    with envs.FLLIPER_PDFLIP_EXACT_CLASS.override(False):
        f = _mk(monkeypatch)
        with caplog.at_level(logging.INFO, logger="pdflip.front"):
            _drive(f, _img_payload())
        rv = [r.getMessage() for r in caplog.records if "ROUTE-VERDICT" in r.getMessage()]
        assert rv and "src=exact" in rv[0], "rc17: the unpriced media n passed as exact carrier_est"
        assert f._x_unpriced("r", None, 5000, -1, True).n_src == "size_only"
        xt = f._x_unpriced("r2", SimpleNamespace(n=N), 9999, U_P, False)
        assert xt.pending == 0 and xt.measure is False and xt.n_src == "front_ids", "rc17: pending=0, no barrier"
        assert not f.__dict__.get("_measure_on_d")


def _classes_of(monkeypatch, payload, chars3, caplog):
    f = _mk(monkeypatch, store_u_p=U_P if payload is _text_payload else 1000)
    monkeypatch.setattr(F, "price_remainder", lambda text, spans, epoch=None: (chars3, chars3, False))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, payload())
    sig = []
    for d in _parsed(caplog):
        if d["kind"] in ("measure", "arrival", "set"):
            sig.append((d["kind"], d["action"], d["rule"], tuple(d["classes"]), d["X"], d["halfX"],
                        tuple((m.u_own, m.src) for m in d["members"])))
    return sig, list(f.routed and [r[0] for r in f.routed])


@pytest.mark.parametrize("payload", [_text_payload, _img_payload])
@pytest.mark.parametrize("probe_store", ["u_p_low", "u_p_high"])
def test_c_chars3_to_a_billion_changes_no_class(monkeypatch, caplog, payload, probe_store):
    s1, r1 = _classes_of(monkeypatch, payload, 5, caplog)
    s2, r2 = _classes_of(monkeypatch, payload, 10 ** 9, caplog)
    assert s1 and s1 == s2 and r1 == r2, (s1, s2)


# ------------------------------------------------------------------------------------------- (f) one snapshot

def test_f_group_numbers_is_one_read_without_an_await():
    import ast
    assert not inspect.iscoroutinefunction(dn.NumberSource.group_numbers)
    assert not inspect.iscoroutinefunction(dn.commit_if_current)
    tree = ast.parse(inspect.getsource(dn))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("group_numbers", "commit_if_current", "arrival_numbers"):
            assert not [n for n in ast.walk(node) if isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith))], node.name


def test_f_epoch_check_refuses_a_stale_commit_and_a_straight_commit_goes_through():
    front = SimpleNamespace(epoch=4, awake="D", state="serving", _set_seats_free=lambda: 2, _dc_decoding=lambda: True)
    committed = []

    async def straight():
        snap = dn.NumberSource(front).group_numbers()
        return dn.commit_if_current(front, snap, lambda: committed.append("straight"))

    async def awaiting():
        snap = dn.NumberSource(front).group_numbers()
        await asyncio.sleep(0)          # another task flips the group while this one waits
        return dn.commit_if_current(front, snap, lambda: committed.append("stale"))

    async def flipper():
        front.epoch += 1                # the group flips while ``awaiting`` is suspended at its sleep

    async def go():
        r1 = await straight()
        a = asyncio.create_task(awaiting())
        b = asyncio.create_task(flipper())
        await asyncio.gather(a, b)
        return r1, a.result()

    r1, r2 = asyncio.run(go())
    assert r1 is True and r2 is False and committed == ["straight"]


# ------------------------------------------------------------------------------------------- (h) KV freshness

def test_h_kv_fresh_or_absent():
    now = 1000.0
    assert dn.kv_fresh(None, now, 1000) == (None, None)
    assert dn.kv_fresh((now - 0.4, {"free": 5}), now, 1000) == ({"free": 5}, 400)
    assert dn.kv_fresh((now - 1.5, {"free": 5}), now, 1000) == (None, 1500), "older than the bound: not there"
    assert dn.kv_fresh((now - 0.1, None), now, 1000)[0] is None


def test_h_refresh_runs_only_while_d_is_awake_and_never_twice():
    now = 1000.0
    assert dn.kv_refresh_due(None, now, 250, True, False) is True
    assert dn.kv_refresh_due(None, now, 250, False, False) is False                 # D asleep
    assert dn.kv_refresh_due((now - 1.0, {}), now, 250, True, True) is False        # one in flight
    assert dn.kv_refresh_due((now - 0.1, {}), now, 250, True, False) is False       # young enough
    assert dn.kv_refresh_due((now - 0.3, {}), now, 250, True, False) is True
    assert dn.kv_refresh_due((now - 9.0, {}), now, 0, True, False) is False         # switch: 0 = off


class _Resp:
    status = 200

    async def json(self):
        return {"internal_states": [{"pdflip_kv": {"free_tokens": 99999, "decode_clip": 4096}}]}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    def __init__(self):
        self.gets = 0

    def get(self, url, timeout=None):
        self.gets += 1
        return _Resp()


def test_h_fetch_starts_at_the_end_of_p_to_d_and_the_refresh_only_while_d_is_awake():
    f = _front(True)
    f.session = _Session()

    async def go():
        assert Front_flip_end(f, "P") is False                    # into P: no read
        assert Front_flip_end(f, "D") is True                     # into D: the read starts NOW, in the background
        assert Front_flip_end(f, "D") is False                    # one in flight
        await asyncio.sleep(0.05)
        assert f.session.gets == 1 and f._asr_st()["kv"] is not None
        f.awake, f.state = "P", "serving"
        f._asr_st()["kv"] = (time.time() - 5.0, {})                # old reading, D asleep
        F.Front._kv_refresh_tick(f, time.time())
        await asyncio.sleep(0.05)
        assert f.session.gets == 1, "no refresh while D sleeps"
        f.awake = "D"
        F.Front._kv_refresh_tick(f, time.time())
        await asyncio.sleep(0.05)
        assert f.session.gets == 2, "refresh while D is awake and the reading is old"
        F.Front._kv_refresh_tick(f, time.time())                   # young now
        await asyncio.sleep(0.05)
        assert f.session.gets == 2

    asyncio.run(go())


def Front_flip_end(f, dst):
    return F.Front._kv_flip_end(f, dst)


def test_h_flip_end_kick_sits_right_after_t_awake_in_flip():
    src = inspect.getsource(F.Front.flip)
    i = src.index("self.t_awake = time.time()")
    assert "Front._kv_flip_end(self, dst)" in src[i:i + 400]


def test_h_a_reading_older_than_the_bound_is_absent_never_a_number():
    f = _front(True)
    f.session = _Session()
    f.t_awake = time.time() - 60.0                                  # the reading is of THIS D phase
    st = f._asr_st()
    st["kv"] = (time.time() - 3.0, {"free_tokens": 12345})         # 3 s old: older than FLLIPER_PDFLIP_KV_FRESH_MS

    async def go():
        # the refresh in flight is slow: the budget runs out and the last reading would decide (rc17)
        async def slow(gen):
            await asyncio.sleep(5)
        f._arrival_seat_kv_fetch = slow
        return await f._arrival_seat_kv_reading()

    got = asyncio.run(asyncio.wait_for(go(), 10))
    assert got is None and f.counters["arrival_seat_kv_absent"] == 1
    assert f._asr_st().get("kv_absent_ms", 0) >= 2900
    with envs.FLLIPER_PDFLIP_EXACT_CLASS.override(False):           # rc17: the old reading decides
        f2 = _front(True)
        f2.session = _Session()
        f2.t_awake = time.time() - 60.0
        f2._asr_st()["kv"] = (time.time() - 3.0, {"free_tokens": 12345})
        f2._arrival_seat_kv_fetch = slow_fetch
        got2 = asyncio.run(asyncio.wait_for(f2._arrival_seat_kv_reading(), 10))
        assert got2 == {"free_tokens": 12345}


async def slow_fetch(gen):
    await asyncio.sleep(5)


# ------------------------------------------------------------------------------------------- price barrier / early flip

def test_price_barrier_reads_no_chars3_price(monkeypatch):
    f = _front(True)
    monkeypatch.setattr(F.Front, "PRICE_BARRIER_DECIDES", True)
    async def go():
        return F.Front._pb_register(f, "r", 10 ** 9)          # a chars/3 price far over X
    assert asyncio.run(go()) is None and not f.__dict__.get("_pb_inflight")
    with envs.FLLIPER_PDFLIP_EXACT_CLASS.override(False):
        async def go2():
            return F.Front._pb_register(f, "r2", 10 ** 9)
        assert asyncio.run(go2()) is not None, "switch off: the rc17 barrier"


def test_early_flip_only_on_an_exact_number():
    src = inspect.getsource(F.Front.handle_generate)
    i = src.index("_ef = Front._early_flip_maybe(self, rid, remainder)")
    assert 'not getattr(_xx, "unpriced", False)' in src[i - 200:i]
