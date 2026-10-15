"""FRONT-X-MIT 15.10.: the front's decision X rides with the D-direct leg; D's X gate refuses
by name above ``front_x + one page`` instead of computing silently up to the static 12288.

Built from boot ``...dauer10151242_226aad7105_1015_124245`` (h16p, report nf-core-dx-1015): D's gate knew
only the static X=12288 while the front priced with the dynamic X (1010..6733). A price too LOW
(``weg2-40-127``: priced 237, D computed 1517; ``weg2-24-60`` 1152 -> 1664; ``weg2-38-105`` 344 -> 664) was
computed on D silently. RED at the base 226aad7105: ``weg2.front_x`` does not exist, the gate has no
``_weg2_x_limit_for``, the front posts no header.

Groups
  a_*  the carrier module (headers, parse, tolerance, limit)
  b_*  D's gate, with the real numbers (admit / W31 / boundary / static ceiling / resume / switch)
  c_*  the W50 message stays parseable by the front (``_d_refusal_extent`` / ``is_x_refusal``)
  d_*  the front: decision sites record the X, the D-direct leg posts the headers, other legs none
  e_*  the plumbing is complete (request objects carry the two fields on every hop)

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no model, no GPU.
"""
from __future__ import annotations

import ast
import collections
import inspect
import logging
import os
import types
from types import MethodType, SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

import sglang.srt.managers.scheduler as sched_mod  # noqa: E402
import sglang.srt.weg2.front as front_mod  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

# Reached through a guard, never as a bare import: at the base (226aad7105) `weg2.front_x` does not exist, and a bare import
# would turn the whole file into ONE collection error instead of one red test per defect (the extraction-count trap).
try:
    from sglang.srt.weg2 import front_x  # noqa: E402
except ImportError:  # pragma: no cover - only the base commit
    front_x = None
from sglang.srt.weg2.front import Front  # noqa: E402

PAGE = 64
STATIC_X = 12288


# ----------------------------------------------------------------------------- helpers
def _req(rid="weg2-40-127", prompt=30957, cached=29440, front_x_=None, priced=None, out=()):
    """The 13:12 request: 30957 prompt tokens, D matched 29440 -> uncached 1517."""
    return SimpleNamespace(
        rid=rid,
        full_untruncated_fill_ids=list(range(prompt)),
        origin_input_ids=list(range(prompt)),
        prefix_indices=list(range(cached)),
        host_hit_length=0,
        output_ids=list(out),
        weg2_front_x=front_x_,
        weg2_front_priced=priced,
    )


def _stub(x=STATIC_X, page=PAGE):
    stub = SimpleNamespace(
        ps=SimpleNamespace(tp_size=1),
        server_args=SimpleNamespace(tp_prefill_max_tokens=x),
        page_size=page,
        tree_cache=None,
    )
    for name in ("_weg2_x_refuses", "weg2_uncached_extent", "_weg2_host_carry_tokens"):
        setattr(stub, name, MethodType(getattr(Scheduler, name), stub))
    return stub


def _verdict(caplog, stub, req):
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        refused = stub._weg2_x_refuses(req, None)
    line = [r.getMessage() for r in caplog.records if "WEG2 X-GATE rid=" in r.getMessage()]
    assert len(line) == 1, line
    return refused, line[0]


# ============================================================================ a_ module
def test_a1_headers_roundtrip():
    h = front_x.request_headers(1010, 237)
    assert h == {"x-weg2-front-x": "1010", "x-weg2-front-priced": "237"}
    assert front_x.parse_headers(h) == (1010, 237)


def test_a2_no_decision_x_means_no_header_and_absent_parses_to_none():
    assert front_x.request_headers(0, 5) == {}
    assert front_x.request_headers(None, 5) == {}
    assert front_x.parse_headers({}) == (None, None)
    assert front_x.parse_headers(None) == (None, None)


def test_a3_malformed_or_zero_x_is_absent_never_refuse_everything():
    assert front_x.parse_headers({"x-weg2-front-x": "abc"}) == (None, None)
    assert front_x.parse_headers({"x-weg2-front-x": "-5"}) == (None, None)
    assert front_x.parse_headers({"x-weg2-front-x": "0"}) == (None, None)
    assert front_x.parse_headers({"x-weg2-front-x": "1010", "x-weg2-front-priced": "zz"}) == (1010, None)


def test_a4_limit_is_front_x_plus_one_page_under_the_static_ceiling():
    assert front_x.limit_for(12288, 1010, 64) == 1074
    assert front_x.limit_for(12288, 20000, 64) == 12288, "the static X stays the hard ceiling"
    assert front_x.limit_for(12288, None, 64) == 12288
    assert front_x.limit_for(0, 1010, 64) == 0, "a gate that is off stays off"
    assert front_x.tolerance_tokens(64) == 64 and front_x.tolerance_tokens(None) == 1


def test_a5_switch_off_carries_nothing_both_ends(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_FRONT_X_CARRY", "0")
    assert front_x.request_headers(1010, 237) == {}
    assert front_x.parse_headers({"x-weg2-front-x": "1010"}) == (None, None)


def test_a6_stamp_request_copies_headers_and_leaves_others_untouched():
    obj = SimpleNamespace(weg2_front_x=None, weg2_front_priced=None)
    front_x.stamp_request(obj, SimpleNamespace(headers={"x-weg2-front-x": "1010", "x-weg2-front-priced": "237"}))
    assert (obj.weg2_front_x, obj.weg2_front_priced) == (1010, 237)
    plain = SimpleNamespace(weg2_front_x=None, weg2_front_priced=None)
    front_x.stamp_request(plain, SimpleNamespace(headers={}))
    front_x.stamp_request(plain, None)
    assert (plain.weg2_front_x, plain.weg2_front_priced) == (None, None)


def test_a7_request_front_x_only_for_a_first_admission():
    assert front_x.request_front_x(_req(front_x_=1010, priced=237)) == (1010, 237)
    assert front_x.request_front_x(_req(front_x_=1010, priced=237, out=[5])) == (None, None), \
        "a resumed request (it decoded) is bounded by the static X"
    assert front_x.request_front_x(_req()) == (None, None)


# ============================================================================ b_ D's gate
def test_b1_weg2_40_127_is_refused_by_name_not_computed_silently(caplog):
    """THE BOOT'S CASE: priced 237, decision X=1010, D realised 1517 (< static 12288).
    Base: admit (silent). Now: W31, and the line names front_x / limit."""
    req = _req(front_x_=1010, priced=237)
    stub = _stub()
    assert stub.weg2_uncached_extent(req, None) == 1517
    refused, line = _verdict(caplog, stub, req)
    assert refused is True
    assert "verdict=W31" in line and "uncached=1517" in line
    assert "X=12288" in line, "X= stays the static value (old readers parse it)"
    assert "front_x=1010" in line and "front_priced=237" in line and "limit=1074" in line


def test_b2_the_same_request_without_the_carrier_is_the_static_gate(caplog):
    refused, line = _verdict(caplog, _stub(), _req())
    assert refused is False and "verdict=admit" in line
    assert "front_x=- front_priced=- limit=12288" in line


def test_b3_boundary_is_one_page_so_rounding_never_makes_it_oscillate(caplog):
    """priced exactly at X: D matches whole pages, so it computes up to page-1 more. Admit
    through front_x + page, refuse one token later."""
    stub = _stub()
    at_limit = _req(prompt=30957, cached=30957 - 1074, front_x_=1010, priced=1010)
    over = _req(prompt=30957, cached=30957 - 1075, front_x_=1010, priced=1010)
    below = _req(prompt=30957, cached=30957 - (1010 + PAGE - 1), front_x_=1010, priced=1010)
    assert _verdict(caplog, stub, below)[0] is False
    assert _verdict(caplog, stub, at_limit)[0] is False
    assert _verdict(caplog, stub, over)[0] is True


def test_b4_front_x_above_the_static_ceiling_does_not_raise_it(caplog):
    req = _req(prompt=20000, cached=0, front_x_=50000, priced=100)
    refused, line = _verdict(caplog, _stub(), req)
    assert refused is True and "limit=12288" in line


def test_b5_a_resumed_request_is_bounded_by_the_static_x(caplog):
    req = _req(front_x_=1010, priced=237, out=[1, 2, 3])
    refused, line = _verdict(caplog, _stub(), req)
    assert refused is False and "front_x=- " in line


def test_b6_switch_off_on_d_is_the_static_gate_byte_for_byte(monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_WEG2_FRONT_X_CARRY", "0")
    refused, line = _verdict(caplog, _stub(), _req(front_x_=1010, priced=237))
    assert refused is False and "front_x=-" in line


def test_b7_the_limit_uses_the_launch_page_size_a_replicated_term():
    assert front_x.resolve_limit(_req(front_x_=1010), STATIC_X, 128)[0] == 1010 + 128
    assert front_x.resolve_limit(_req(), STATIC_X, 128) == (STATIC_X, None, None)
    assert front_x.resolve_limit(object(), STATIC_X, 128) == (STATIC_X, None, None), "a partial double is 'nothing carried'"


def test_b8_static_gate_still_refuses_above_12288_without_a_carrier(caplog):
    req = _req(prompt=40000, cached=0)
    assert _verdict(caplog, _stub(), req)[0] is True


# ============================================================================ c_ the W50 message
def _answer_stub(x=STATIC_X, page=PAGE):
    sent = []
    stub = SimpleNamespace(
        waiting_queue=[],
        server_args=SimpleNamespace(tp_prefill_max_tokens=x),
        page_size=page,
        ps=SimpleNamespace(tp_size=1),
        tree_cache=None,
        enable_hicache_storage=False,
        enable_hierarchical_cache=False,
        ipc_channels=SimpleNamespace(send_to_tokenizer=SimpleNamespace(send_output=lambda a, r: sent.append(a))),
        counters_991_x_refusal=0,
    )
    for name in ("_weg2_answer_x_refusals", "weg2_uncached_extent", "_weg2_group_min_flags"):
        if hasattr(Scheduler, name):
            setattr(stub, name, MethodType(getattr(Scheduler, name), stub))
    return stub, sent


def test_c1_the_w50_body_names_the_front_x_and_stays_parseable_by_the_front():
    stub, sent = _answer_stub()
    req = _req(front_x_=1010, priced=237)
    req.time_stats = SimpleNamespace(trace_ctx=SimpleNamespace(abort=lambda **k: None))
    req.priority = 0
    stub.waiting_queue = [req]
    stub._weg2_answer_x_refusals([req], None)
    assert len(sent) == 1
    msg = sent[0].finished_reason["message"]
    assert msg.startswith("W50 Weg2TpPrefillExceeded: this group may prefill at most 1074 uncached")
    assert "FRONT-X: the front dispatched this request on X=1010 (priced 237)" in msg
    body = msg.encode()
    assert front_mod._d_refusal_extent(body) == 1517, "the front parses D's extent from this sentence"
    assert front_mod.x_refusal_marker_in(msg) and front_mod.is_x_refusal(503, msg)


def test_c2_without_a_carrier_the_message_is_the_old_one():
    stub, sent = _answer_stub()
    req = _req()
    req.time_stats = SimpleNamespace(trace_ctx=SimpleNamespace(abort=lambda **k: None))
    req.priority = 0
    stub.waiting_queue = [req]
    stub._weg2_answer_x_refusals([req], None)
    msg = sent[0].finished_reason["message"]
    assert "at most 12288 uncached" in msg and "FRONT-X" not in msg


# ============================================================================ d_ the front
def _front_double(x_live=6733):
    f = SimpleNamespace(
        counters=collections.Counter(),
        tp_prefill_max_tokens=x_live,
        _front_price={"weg2-40-127": 237},
    )
    return f


def _pending(rid, est, d_direct=True):
    return SimpleNamespace(rid=rid, est_uncached=est, d_direct=d_direct)


def test_d1_the_last_recorded_decision_x_rides_with_a_d_direct_leg():
    f = _front_double()
    Front._note_x_decision(f, "weg2-40-127", 1010, 237)   # arrival verdict (X-ROUTE X=1010)
    Front._note_x_decision(f, "weg2-40-127", 6057, 237)   # QUEUED-SHORT later (X=6057)
    h = Front._front_x_headers(f, "weg2-40-127", None, False)
    assert h == {"x-weg2-front-x": "6057", "x-weg2-front-priced": "237"}
    assert f.counters["front_x_carried"] == 1
    assert Front._front_x_headers(f, "weg2-40-127", None, False)["x-weg2-front-x"] == "6733", \
        "a decision is consumed by the post; a re-post falls back to the X in force"


def test_d2_an_unrecorded_d_direct_rid_is_bounded_by_the_x_in_force():
    f = _front_double(x_live=6733)
    h = Front._front_x_headers(f, "weg2-40-127", None, False)
    assert h["x-weg2-front-x"] == "6733" and h["x-weg2-front-priced"] == "237"


def test_d3_a_leg_after_p_and_a_single_prefill_carry_no_x():
    f = _front_double()
    Front._note_x_decision(f, "weg2-1-1", 1010, 5)
    assert Front._front_x_headers(f, "weg2-1-1", _pending("weg2-1-1", 5, d_direct=False), False) == {}
    assert Front._front_x_headers(f, "weg2-1-1", None, True) == {}
    assert Front._front_x_headers(f, "weg2-1-1", _pending("weg2-1-1", 5, d_direct=True), False) != {}


def test_d4_switch_off_on_the_front_posts_no_header(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_FRONT_X_CARRY", "0")
    f = _front_double()
    Front._note_x_decision(f, "weg2-1-1", 1010, 5)
    assert Front._front_x_headers(f, "weg2-1-1", None, False) == {}


def test_d5_queued_short_records_the_x_of_its_own_line_on_the_request():
    from test_weg2_x_sum_price_1005 import _Fake, _p

    qs = [_p("40-127", 237, 1.0), _p("40-128", 5089, 2.0)]
    fake = _Fake(qs)
    fake.tp_prefill_max_tokens = 6733
    fake.x_split = True
    fake._d_holds_work = lambda: True
    fake._x_band_floor = lambda: 6057
    before = {k: v for k, v in vars(fake).items() if k not in ("queue", "_ready_for_d", "counters", "synced")}
    moved = Front._asr_queued_short_to_d(fake, list(qs), 10.0)
    after = {k: v for k, v in vars(fake).items() if k not in ("queue", "_ready_for_d", "counters", "synced")}
    assert [p.rid for p in moved] == ["40-127", "40-128"]
    assert (qs[0].x_decision, qs[1].x_decision) == ((6057, 237), (6057, 5089))
    assert before == after, "the hand-over writes no new attribute on the front (flip_unchanged_1503)"


def test_d5b_the_d_direct_post_await_site_passes_the_requests_decision_to_leg2():
    src = inspect.getsource(Front.handle_generate)
    assert 'if p.d_direct and getattr(p, "x_decision", None) is not None:' in src
    assert "Front._note_x_decision(self, rid, *p.x_decision)" in src
    assert src.index("Front._note_x_decision(self, rid, *p.x_decision)") < src.index("if p.d_direct:\n            # 27B idle policy (b)")


def test_d6_every_hand_over_site_records_its_x_and_the_post_carries_the_headers():
    """Structure, over the AST: each site that sets ``d_direct = True`` for a queued request
    records the decision X in the same function; leg2's D post passes the headers."""
    src = inspect.getsource(front_mod)
    tree = ast.parse(src)
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = ast.get_source_segment(src, node) or ""
    for name in ("_asr_queued_short_to_d", "_dc_hand_to_d", "_x_idle_regrant", "_d_short_drain"):
        assert "_stamp_x_decision" in funcs[name], f"{name} hands a request to D without recording its X"
    leg2 = funcs["leg2"]
    assert "_front_x_headers" in leg2 and "**_post_kw" in leg2


def test_d7_the_arrival_grant_records_the_route_verdict_x():
    src = inspect.getsource(Front.handle_generate)
    assert src.count("_note_x_decision(self, rid, x_route, remainder)") == 1, "the SHORT grant (no Pending exists)"
    assert src.count("_stamp_x_decision(p, x_route, remainder)") == 1, "the queued / kept request's arrival verdict"


# ============================================================================ e_ plumbing
def test_e1_every_hop_carries_the_two_fields():
    from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
    from sglang.srt.managers.schedule_batch import Req

    for cls in (GenerateReqInput, TokenizedGenerateReqInput):
        fields = set(getattr(cls, "__dataclass_fields__", None) or getattr(cls, "__struct_fields__", ()) or ())
        assert {"weg2_front_x", "weg2_front_priced"} <= fields, cls.__name__
    assert "self.weg2_front_x = None" in inspect.getsource(Req.__init__)
    g = GenerateReqInput(text=["a", "b"], weg2_front_x=1010, weg2_front_priced=237)
    g.normalize_batch_and_arguments()
    assert (g[1].weg2_front_x, g[1].weg2_front_priced) == (1010, 237), "the batch split keeps the carrier"


def test_e2_tokenizer_scheduler_and_serving_layers_pass_the_fields_on():
    import sglang.srt.entrypoints.openai.serving_chat as chat
    import sglang.srt.entrypoints.openai.serving_completions as comp
    import sglang.srt.managers.tokenizer_manager as tm

    assert "weg2_front_x=obj.weg2_front_x" in inspect.getsource(tm)
    assert 'req.weg2_front_x = getattr(recv_req, "weg2_front_x", None)' in inspect.getsource(sched_mod)
    for mod in (chat, comp):
        assert "weg2_front_x.stamp_request(adapted_request, raw_request)" in inspect.getsource(mod)
