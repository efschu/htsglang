"""S56 D-MATCH-PROBE (NF rc15 19.10.): the arrival class from D's EXACT numbers (pdflip/d_match_probe.py).

User 19.10.: "warum wird irgendwas geschätzt? alle zahlen liegen doch schwarz auf weiß vor zu jedem zeitpunkt?" --
binding: uncached = prompt - the exact hit at the source at the decision (D's tree incl. Mamba anchor + host tier,
the shared L2/L3 store), in D's token space. SOLL R11: the front decides alone, on those numbers. R9: an image in
the uncached tail is P-only.

THE PFLICHTTEST (fixture ``fixtures/s56_rc15_class_sources_1019.json``, extracted from boot
..._489909d138_1019_071429 front + D log: per request the front's logged price, the realised compute on P
(PDFLIP-USAGE-TRUE p_computed = P prompt - P cached) or D (D's X-GATE uncached, #969 EXTENT), D's match census,
D's W123 refusals):
  * BEFORE (the front's ledger price): 80 of 233 requests miss the realised compute by more than one page
    (74 over, 6 under) -- among them pdflip-6-11 (3513 vs 977), 10-17 (23549 vs 543), 14-74 (338 vs 5718).
  * AFTER (``exact_price`` on the source numbers in D's token space, P's number where P computes, an image tail
    P-only): 0 of 233 miss.
The fixture's source numbers are the readings at EXECUTION; the probe asks the same sources at the DECISION -- the
time between the two is the residual the next boot measures (D-PROBE line vs PDFLIP-USAGE-TRUE).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

PAGE = 64
FIX = os.path.join(os.path.dirname(__file__), "fixtures", "s56_rc15_class_sources_1019.json")


def _ans(n, device, host=0, spans=()):
    return dmp.ProbeAnswer(ids=np.arange(n, dtype=np.int32), device_tokens=device, host_tokens=host,
                           key_match_depth=device + host, mm_spans=tuple(spans))


# ----------------------------------------------------------------------------------------- the price, case by case

def test_rc15_pdflip_6_11_is_kurz_from_d_numbers():
    """Front 3513 (halb at X=6216) -- D: prompt 99217, match 98240 (prompt + output of pdflip-4-9) -> 977, kurz (R2)."""
    p = dmp.exact_price(_ans(99217, 98240, spans=[(1000 + 220 * i, 1220 + 220 * i) for i in range(8)]), 93952)
    assert (p.uncached, p.covered, p.image_in_tail, p.src) == (977, 98240, False, "d_tree")
    assert pp.classify_arrival(dmp.class_uncached(p), 6216, 0.5, False) == pp.CLASS_KURZ


def test_rc15_pdflip_14_74_image_in_the_tail_is_p_only_at_p_s_number():
    """Front 338 (kurz) -- D: prompt 85590, match 83712, 4 NEW images past it (W123), store 79872: P-only (R9), and
    the class number is what P computed: 85590 - 79872 = 5718."""
    p = dmp.exact_price(_ans(85590, 83712, spans=[(84000, 84882), (84900, 85400)]), 79872)
    assert p.image_in_tail and p.uncached == 1878 and p.uncached_p == 5718
    assert dmp.class_uncached(p) == 5718


def test_rc15_pdflip_10_17_store_covers_its_images():
    """Front 23549 (gross, the L3 probe stopped at its first image) -- the store with D's ids: 79104 of 79647 -> 543."""
    p = dmp.exact_price(_ans(79647, 0, spans=[(30000, 31000), (54400, 55300)]), 79104)
    assert (p.uncached, p.src, p.image_in_tail) == (543, "store", False)
    assert pp.classify_arrival(dmp.class_uncached(p), 6216, 0.5, False) == pp.CLASS_KURZ


def test_host_tier_counts_like_device_and_never_past_the_prompt():
    p = dmp.exact_price(_ans(5000, 2048, host=2048), 0)
    assert p.covered == 4096 and p.uncached == 904
    p = dmp.exact_price(_ans(5000, 4096, host=4096), 10 ** 9)
    assert p.covered == 5000 and p.uncached == 0


def test_nested_tool_result_images_are_counted():
    payload = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [
            {"type": "text", "text": "shot"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"}}]}]}]}
    assert dmp.nested_image_parts(payload) == 2
    assert dmp.nested_image_parts({"messages": [{"role": "user", "content": "x"}]}) == 0
    assert dmp.nested_image_parts(None) == 0


def test_answer_roundtrip_and_incomplete_answers():
    ids = np.array([5, 1 << 29, 7], dtype=np.int64)
    body = {"success": True, "ids": dmp.ids_to_b64(ids), "device_tokens": 2, "host_tokens": 0,
            "key_match_depth": 2, "mm_spans": [[1, 2]], "tokenize_ms": 3.0, "handle_ms": 0.4, "total_ms": 9.0}
    a = dmp.parse_answer(json.loads(json.dumps(body)))
    assert a is not None and a.ids.tolist() == [5, 1 << 29, 7] and a.mm_spans == ((1, 2),) and a.n == 3
    assert dmp.parse_answer({"success": False, "message": "x"}) is None
    assert dmp.parse_answer({"success": True}) is None
    assert dmp.seq_spans_ok([(0, 3)], 3) and not dmp.seq_spans_ok([(2, 9)], 3)


def test_match_numbers_reads_the_admission_prefix():
    class _MR:
        device_indices = list(range(4096))
        host_hit_length = 2048
        key_match_depth = 7000
    assert dmp.match_numbers(_MR()) == (4096, 2048, 7000)


# ------------------------------------------------------------------------------------------------- THE PFLICHTTEST

def _realised(r):
    """(where, new tokens computed) -- P: P prompt - P cached; D: D's own X-GATE uncached (#969 EXTENT)."""
    if r["p_computed"] > 0:
        return "P", int(r["p_computed"])
    if "d_uncached" in r:
        return "D", int(r["d_uncached"])
    return None, None


def _source_answer(r, where):
    """The sources' answer for the request, in D's token space (the group's prompt length)."""
    if where == "P":
        n = int(r["p_prompt"])
        spans = [(n - 2, n - 1)] if "w123_covered" in r else []   # D refused W123: an image in its tail
        return _ans(n, 0, spans=spans), int(r["p_cached"])
    n = int(r.get("d_accepted", 0)) + int(r["d_uncached"]) if "d_accepted" in r else int(r["d_uncached"]) + 0
    return _ans(n, n - int(r["d_uncached"])), 0


def test_pflicht_rc15_every_classification_number_equals_the_realised_compute():
    rows = json.load(open(FIX))
    assert len(rows) == 233
    before_miss, after_miss, n = [], [], 0
    for r in rows:
        where, real = _realised(r)
        if where is None:
            continue
        n += 1
        if abs(int(r["front_pending"]) - real) > PAGE:
            before_miss.append(r["rid"])
        ans, store = _source_answer(r, where)
        price = dmp.exact_price(ans, store)
        used = price.uncached_p if where == "P" else dmp.class_uncached(price)
        if abs(used - real) > PAGE:
            after_miss.append((r["rid"], used, real))
    assert n == 233
    assert len(before_miss) == 80, "the front ledger's misses in rc15 (evidence; 74 over, 6 under)"
    assert after_miss == [], after_miss


def test_pflicht_the_underestimates_vanish():
    """The ledger's UNDER-estimates (P computed far more than priced): with the sources, the number that rides to P
    is P's own."""
    rows = {r["rid"]: r for r in json.load(open(FIX))}
    for rid in ("pdflip-14-66", "pdflip-14-74", "pdflip-14-76", "pdflip-15-77", "pdflip-26-106"):
        r = rows[rid]
        where, real = _realised(r)
        assert where == "P" and int(r["front_pending"]) < real - PAGE, rid      # the old under-estimate
        ans, store = _source_answer(r, where)
        assert dmp.exact_price(ans, store).uncached_p == real, rid


# ------------------------------------------------------------------------------------------ the front, end to end

def test_front_prices_from_the_probe_and_routes_an_image_tail_to_p(caplog, monkeypatch):
    """``_x_exact_price`` asks D (stubbed rpc), asks the store with D's ids, and returns D's exact class number; the
    ledger's figure only stays in the line. An image in the tail returns ``image_in_tail`` (p_only, R9)."""
    from test_pdflip_front_x_exact import _FakeTokens, _front, _payload

    f = _front(True)
    f.ftok = _FakeTokens(97465)                        # the FRONT's (text-only) count: 97465 like rc15 6-11
    f.session = object()
    n = 99217
    body = {"success": True, "ids": dmp.ids_to_b64(np.arange(n) + 10), "device_tokens": 98240, "host_tokens": 0,
            "key_match_depth": 98240, "mm_spans": [[1000, 1220]], "tokenize_ms": 30.0, "handle_ms": 1.0,
            "total_ms": 40.0}
    calls = []

    async def post(g, b, timeout):
        calls.append((dmp.PROBE_PATH, b["path"]))
        return 200, json.dumps(body)

    f._d_probe_post = post

    async def store(rid, ids, timeout_s, extra_key=None):
        assert int(ids.size) == n, "the store is asked with D's ids"
        return 93952, "l3_index"

    f._store_probe_depth = store
    payload = _payload(100)
    payload["messages"].append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]})
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        xx = asyncio.run(f._x_exact_price("pdflip-6-11", "/v1/messages", payload, "t", 3513, 97465))
    assert calls == [(dmp.PROBE_PATH, "/v1/messages")]
    assert xx.pending == 977 and xx.n == n and xx.src == "d_probe_d_tree" and xx.image_in_tail is False
    assert xx.mm and xx.mm_cached, "every image inside the covered prefix"
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("PDFLIP D-PROBE ")][0]
    assert "device=98240" in line and "uncached_d=977" in line and "front_n=97465" in line
    # the tail: a new image past D's match
    body["mm_spans"] = [[98300, 98900]]
    xx = asyncio.run(f._x_exact_price("pdflip-x", "/v1/messages", payload, "t2", 3513, 97465))
    assert xx.image_in_tail is True and xx.pending == n - 93952 and xx.mm_cached is False   # P's number (R9)


def test_front_names_a_failed_probe_asks_again_and_never_keeps_its_own_count(caplog, monkeypatch):
    """S56r A1 (replaces S56's "the front's own count stands"): a failed probe is ASKED AGAIN; no exact number by the
    deadline -> UNPRICED: pending 0 only as the route to D's own admission, never the front's 3000."""
    from flliper.srt.environ import envs
    from test_pdflip_front_x_exact import _FakeTokens, _front, _payload

    monkeypatch.setenv("FLLIPER_PDFLIP_D_MATCH_PROBE_DEADLINE_MS", "300")
    monkeypatch.setenv("FLLIPER_PDFLIP_D_MATCH_PROBE_RETRY_MS", "20")
    assert envs.FLLIPER_PDFLIP_D_MATCH_PROBE_DEADLINE_MS.get() == 300
    f = _front(True)
    f.ftok = _FakeTokens(3000)
    f.session = object()
    calls = []

    async def post(g, b, timeout):
        calls.append(timeout)
        return 503, "down"

    f._d_probe_post = post

    async def store(rid, ids, timeout_s, extra_key=None):
        return 1000, "l3_index"

    f._store_probe_depth = store
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        xx = asyncio.run(f._x_exact_price("r", "/v1/messages", _payload(100), "t", 1000, 3000))
    assert len(calls) >= 3, "asked again inside the deadline"
    # EZ-S1 (plan A1): u_p = 2000 < halbX = 2048 is the ONE number an upper bound may classify: kurz, the safe direction
    # (pending = u_p). With FLLIPER_PDFLIP_EXACT_CLASS off it is the rc17 form (pending 0, D's admission measures);
    # u_p >= halbX is MEASURE_ON_D under both forms (test_decision_numbers_1019 / test_measure_on_d_1019).
    assert xx.unpriced is True and xx.pending == 2000 and xx.src == "d_probe_unpriced" and xx.known is False
    assert xx.class_basis.kind == "p_upper" and xx.measure is False
    assert xx.uncached_p == 2000, "P's exact number: the text prompt's ids are D's ids, n - store"
    assert f.counters["d_probe_failed"] == len(calls) and f.counters["d_probe_unpriced"] == 1
    msgs = [r.getMessage() for r in caplog.records]
    assert any("PDFLIP D-PROBE-FAILED rid=r attempt=1 why=status=503" in m and "next=retry" in m for m in msgs)
    assert any("PDFLIP D-PROBE-UNPRICED rid=r" in m for m in msgs)
    assert not any("X-EXACT-PRICE" in m and "rid=r " in m for m in msgs), "no ledger price is formed"


def test_front_a_payload_d_cannot_tokenize_gives_up_at_once(monkeypatch):
    """S56r A1: what no attempt can change (D's tokenizer rejects the payload: retry=False) ends at once."""
    from test_pdflip_front_x_exact import _FakeTokens, _front, _payload

    f = _front(True)
    f.ftok = _FakeTokens(3000)
    f.session = object()
    calls = []

    async def post(g, b, timeout):
        calls.append(1)
        return 200, json.dumps({"success": False, "message": "ValueError: bad", "retry": False})

    f._d_probe_post = post

    async def store(rid, ids, timeout_s, extra_key=None):
        return 0, "none"

    f._store_probe_depth = store
    xx = asyncio.run(f._x_exact_price("r", "/v1/messages", _payload(100), "t", 1000, 3000))
    assert calls == [1] and xx.unpriced is True and xx.uncached_p == 3000


def test_front_a_probe_that_recovers_prices_exactly(monkeypatch):
    """S56r A1: the second attempt answers -> the exact price, D-PROBE-RECOVERED, nothing decided before it."""
    from test_pdflip_front_x_exact import _FakeTokens, _front, _payload

    monkeypatch.setenv("FLLIPER_PDFLIP_D_MATCH_PROBE_RETRY_MS", "1")
    f = _front(True)
    f.ftok = _FakeTokens(3000)
    f.session = object()
    n = 3000
    body = {"success": True, "ids": dmp.ids_to_b64(np.arange(n) + 10), "device_tokens": 2944, "host_tokens": 0,
            "key_match_depth": 2944, "mm_spans": []}
    seq = [(0, "ClientConnectorError"), (200, json.dumps(body))]

    async def post(g, b, timeout):
        code, text = seq.pop(0)
        if code == 0:
            raise ConnectionError(text)
        return code, text

    f._d_probe_post = post

    async def store(rid, ids, timeout_s, extra_key=None):
        return 0, "none"

    f._store_probe_depth = store
    xx = asyncio.run(f._x_exact_price("r", "/v1/messages", _payload(100), "t", 1000, 3000))
    assert not getattr(xx, "unpriced", False) and xx.pending == 56 and f.counters["d_probe_recovered"] == 1


def test_front_an_unread_store_is_not_a_zero(monkeypatch):
    """S56r A1: a store probe that FAILED is not "the store holds nothing": asked once more, then UNPRICED."""
    from test_pdflip_front_x_exact import _FakeTokens, _front, _payload

    f = _front(True)
    f.ftok = _FakeTokens(3000)
    f.session = object()
    body = {"success": True, "ids": dmp.ids_to_b64(np.arange(3000) + 10), "device_tokens": 0, "host_tokens": 0,
            "key_match_depth": 0, "mm_spans": []}

    async def post(g, b, timeout):
        return 200, json.dumps(body)

    f._d_probe_post = post
    asked = []

    async def store(rid, ids, timeout_s, extra_key=None):
        asked.append(1)
        return 0, "unread"

    f._store_probe_depth = store
    xx = asyncio.run(f._x_exact_price("r", "/v1/messages", _payload(100), "t", 1000, 3000))
    assert len(asked) == 2 and xx.unpriced is True


def test_probe_switch_default_on():
    from flliper.srt.environ import envs

    assert envs.FLLIPER_PDFLIP_D_MATCH_PROBE.default is True


# --------------------------------------------------------------------------------- the scheduler half (every rank)

def _fake_sched(tree, **kw):
    import types

    base = dict(
        tree_cache=tree,
        model_config=types.SimpleNamespace(vocab_size=1000),
        pad_input_ids_func=lambda ids, mm: [(-7 if 2 <= i <= 5 else t) for i, t in enumerate(ids)],
        _get_multimodal_inputs=lambda mm: types.SimpleNamespace(
            mm_items=[types.SimpleNamespace(offsets=[(2, 5)])], padded_input_ids=None),
        _try_apply_padded_mm_input_ids=lambda recv, req, mm: False,
    )
    base.update(kw)
    fake = types.SimpleNamespace(**base)
    from flliper.srt.managers import scheduler as S

    for _m in ("_pdflip_match_probe_one", "_pdflip_match_probe_pad", "_pdflip_match_probe_local",
               "_pdflip_match_probe_group_min"):   # 27B NACHT 21.10.: the probe's halves (D-PROBE GROUP-MIN)
        setattr(fake, _m, (lambda _n: lambda *a: getattr(S.Scheduler, _n)(fake, *a))(_m))
    return fake


def test_scheduler_handler_reads_the_admission_key_read_only(monkeypatch):
    """S56r A2: the handler pads image items like the intake and READS the admission's usable prefix with
    ``peek_admission_match`` (whole PADDED ids, limit n-1) -- ``match_prefix`` (which splits, re-stamps, moves LRUs)
    is never called."""
    from array import array

    from flliper.srt.managers import scheduler as S
    from flliper.srt.managers.io_struct import PdFlipMatchProbeReqInput, wrap_as_pickle

    seen = {}

    class _Tree:
        def match_prefix(self, params):
            raise AssertionError("the probe must not call match_prefix (it mutates the tree)")

    def peek(tc, ids, extra_key, limit):
        seen["args"] = (list(ids), extra_key, limit)
        return 96, 32, 200

    monkeypatch.setattr(dmp, "peek_admission_match", peek)
    fake = _fake_sched(_Tree())
    ids = array("q", range(10, 10 + 200))
    req = PdFlipMatchProbeReqInput(rid="pdflip-6-11~probe", input_ids=ids,
                                   mm_inputs=wrap_as_pickle({"x": 1}), extra_key=None, tokenize_ms=12.0)
    out = S.Scheduler.handle_pdflip_match_probe(fake, req)
    assert out.success, out.message
    got_ids, ek, limit = seen["args"]
    assert limit == 199 and got_ids[2:6] == [-7, -7, -7, -7], "the PADDED ids are the key"
    assert (out.device_tokens, out.host_tokens, out.key_match_depth) == (96, 32, 200)
    assert out.mm_spans == [[2, 6]] and list(out.input_ids)[3] == -7 and out.tokenize_ms == 12.0


def test_scheduler_handler_answers_a_coalesced_batch_in_order(monkeypatch):
    """S56r A3: the probes the tokenizer coalesced are answered in the same round, in order."""
    from array import array

    from flliper.srt.managers import scheduler as S
    from flliper.srt.managers.io_struct import PdFlipMatchProbeReqInput, unwrap_from_pickle, wrap_as_pickle

    monkeypatch.setattr(dmp, "peek_admission_match", lambda tc, ids, ek, limit: (len(ids) // 2, 0, len(ids)))
    fake = _fake_sched(object())
    req = PdFlipMatchProbeReqInput(rid="a", input_ids=array("q", range(10)), batch=wrap_as_pickle([
        {"rid": "b", "input_ids": array("q", range(20)), "mm_inputs": None, "extra_key": None, "tokenize_ms": 1.0},
        {"rid": "c", "input_ids": array("q", range(40)), "mm_inputs": None, "extra_key": None, "tokenize_ms": 2.0}]))
    out = S.Scheduler.handle_pdflip_match_probe(fake, req)
    more = unwrap_from_pickle(out.batch)
    assert (out.rid, out.device_tokens) == ("a", 5)
    assert [(d["rid"], d["device_tokens"], d["success"]) for d in more] == [("b", 10, True), ("c", 20, True)]


def test_scheduler_handler_answers_the_group_min_on_an_asymmetric_tier(monkeypatch):
    """S56r (R11) + 27B NACHT 21.10. D-PROBE GROUP-MIN: on an asymmetric host tier without Form A the admission takes
    the RU MIN -- TP0's walk alone is not the group's number. Every rank walks and the group reduces by MIN; a single
    rank (no TP group) answers its own walk, which IS the group's. Never ``group_min_needed`` any more (27B INT8 metal
    21.10.: 22 of 22 probes refused, every arrival through the measuring barrier)."""
    from array import array

    from flliper.srt.managers import scheduler as S
    from flliper.srt.managers import tp_match_floor as T
    from flliper.srt.managers.io_struct import PdFlipMatchProbeReqInput

    class _Tree:
        def _hicache_prefetch_symmetric(self):
            return True

    monkeypatch.setattr(T, "form_a_follow_active", lambda: False)
    monkeypatch.setattr(dmp, "peek_admission_match", lambda tc, ids, ek, limit: (6, 2, 9))
    out = S.Scheduler.handle_pdflip_match_probe(
        _fake_sched(_Tree()), PdFlipMatchProbeReqInput(rid="r", input_ids=array("q", range(10))))
    assert out.success is True, out.message
    assert (out.device_tokens, out.host_tokens, out.key_match_depth) == (6, 2, 9)


def test_scheduler_handler_never_raises(monkeypatch):
    from array import array

    from flliper.srt.managers import scheduler as S
    from flliper.srt.managers.io_struct import PdFlipMatchProbeReqInput

    def boom(*a):
        raise RuntimeError("boom")

    monkeypatch.setattr(dmp, "peek_admission_match", boom)
    fake = _fake_sched(object())
    out = S.Scheduler.handle_pdflip_match_probe(fake, PdFlipMatchProbeReqInput(rid="r~probe", input_ids=None))
    assert out.success is True and out.device_tokens == 0      # an empty prompt walks nothing
    out = S.Scheduler.handle_pdflip_match_probe(
        fake, PdFlipMatchProbeReqInput(rid="r~probe", input_ids=array("q", [1, 2, 3])))
    assert out.success is False and "boom" in out.message


def test_ipc_roundtrip_of_the_probe_structs():
    from array import array

    from flliper.srt.managers import io_struct as IO

    a = IO.PdFlipMatchProbeReqInput(rid="r", input_ids=array("q", [1, 2]), mm_inputs=None, extra_key="k")
    b = IO.msgpack_decode(IO.msgpack_encode(a))
    assert isinstance(b, IO.PdFlipMatchProbeReqInput) and list(b.input_ids) == [1, 2] and b.extra_key == "k"
    o = IO.PdFlipMatchProbeReqOutput(rid="r", success=True, input_ids=array("q", [3]), device_tokens=1,
                                     mm_spans=[[0, 1]])
    o2 = IO.msgpack_decode(IO.msgpack_encode(o))
    assert o2.device_tokens == 1 and list(o2.input_ids) == [3] and o2.mm_spans == [[0, 1]]
    # S56r A3: the coalesced batch rides the same structs (msgpack wire: a PickleWrapper; pickle IPC: the object)
    import pickle

    a.batch = IO.PickleWrapper(pickle.dumps([{"rid": "b", "input_ids": array("q", [5]), "mm_inputs": None,
                                              "extra_key": None, "tokenize_ms": 0.0}]))
    b = IO.msgpack_decode(IO.msgpack_encode(a))
    assert pickle.loads(b.batch.data)[0]["rid"] == "b"
    a.batch = IO.wrap_as_pickle([{"rid": "c"}])
    assert IO.unwrap_from_pickle(pickle.loads(pickle.dumps(a)).batch)[0]["rid"] == "c"


def test_a_text_prompt_asks_d_with_the_front_ids_and_media_goes_whole(caplog):
    """Text: the front's ids ARE D's ids (rc14+rc15 X-EXACT-TOKENS 137/137) -- the probe follows the count and asks
    only D's tree (``/generate`` with the ids). Media anywhere (also a tool_result image): D tokenizes the payload."""
    from test_pdflip_front_x_exact import _FakeTokens, _front, _payload

    f = _front(True)
    f.ftok = _FakeTokens(4492)
    f.session = object()
    sent = []
    n = 4492

    async def post(g, b, timeout):
        sent.append(b)
        return 200, json.dumps({"success": True, "ids": dmp.ids_to_b64(np.arange(n) + 10), "device_tokens": 0,
                                "host_tokens": 0, "key_match_depth": 0, "mm_spans": []})

    f._d_probe_post = post
    xx = asyncio.run(f._x_exact_price("pdflip-6-12", "/v1/messages", _payload(100), "t", 4492, 4492))
    assert sent[0]["path"] == "/generate" and "input_ids_b64" in sent[0]["payload"]
    assert dmp.ids_from_b64(sent[0]["payload"]["input_ids_b64"]).tolist() == (np.arange(n) + 10).tolist()
    assert xx.pending == 4492
    sent.clear()
    img = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]}]}
    asyncio.run(f._x_exact_price("r2", "/v1/messages", img, "t2", 10, 10))
    assert sent[0]["path"] == "/v1/messages" and sent[0]["payload"] is img
    assert dmp.payload_has_any_media(img) and not dmp.payload_has_any_media(_payload(10))


def test_text_prompt_probe_asks_ds_tree_with_the_front_ids(caplog):
    """A text prompt (no media anywhere): the front count IS D's count (rc14+rc15 X-EXACT-TOKENS 137/137), so the
    probe follows the count with those ids (/generate, input_ids_b64) -- D tokenizes nothing a second time."""
    from test_pdflip_front_x_exact import _FakeTokens, _front, _payload

    f = _front(True)
    f.ftok = _FakeTokens(4492)
    f.session = object()
    sent = {}

    async def post(g, b, timeout):
        sent.update(b)
        ids = dmp.ids_from_b64(b["payload"]["input_ids_b64"])
        return 200, json.dumps({"success": True, "ids": dmp.ids_to_b64(ids), "device_tokens": 0, "host_tokens": 0,
                                "key_match_depth": 41, "mm_spans": []})

    f._d_probe_post = post

    async def store(rid, ids, timeout_s, extra_key=None):
        return 0, "none"

    f._store_probe_depth = store
    xx = asyncio.run(f._x_exact_price("pdflip-6-12", "/v1/messages", _payload(100), "t", 5616, 5635))
    assert sent["path"] == "/generate" and sent["payload"]["rid"] == "pdflip-6-12"
    assert dmp.ids_from_b64(sent["payload"]["input_ids_b64"]).tolist() == (np.arange(4492) + 10).tolist()
    assert xx.pending == 4492 and xx.src == "d_probe_none" and xx.image_in_tail is False
    assert not dmp.payload_has_any_media(_payload(100))


def test_a_w123_requeue_is_p_only_at_p_s_number():
    """rc15 pdflip-4-9: D refused W123 (covered 0) with extent 95758; P computed 14 from the store. The requeue marks
    it P-only (R9) and prices it at the arrival probe's n - store."""
    from types import SimpleNamespace

    from test_pdflip_front_x_exact import _front

    f = _front(True)
    f._probe_price = {"pdflip-4-9": dmp.exact_price(_ans(95758, 0), 95744)}
    body = (b'{"error": "W50 PdFlipTpPrefillExceeded (vision reroute): group D cannot serve this request itself; '
            b'this request\'s extent after prefix matching is 95758. Re-routed ... W123 PdFlipVisionImageNotInPrefixOnD"}')

    async def run():
        req = SimpleNamespace(path="/v1/messages")
        task = asyncio.create_task(
            f._requeue_after_x_refusal(req, "pdflip-4-9", {}, "w" * 300, False, None, None, body))
        for _ in range(200):
            if f.queue:
                break
            await asyncio.sleep(0.005)
        q = f.queue[0]
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        return q

    q = asyncio.run(run())
    assert q.p_only is True and q.est_uncached == 14 and f.counters["w123_requeue_p_only"] == 1
