"""FRONT-X-MIT 15.10.: the front's decision X rides with the D-direct leg; D's X gate refuses
by name above ``front_x + one page`` instead of computing silently up to the static 12288.

Built from boot ``...dauer10151242_226aad7105_1015_124245`` (h16p, report nf-core-dx-1015): D's gate knew
only the static X=12288 while the front priced with the dynamic X (1010..6733). A price too LOW
(``pdflip-40-127``: priced 237, D computed 1517; ``pdflip-24-60`` 1152 -> 1664; ``pdflip-38-105`` 344 -> 664) was
computed on D silently. RED at the base 226aad7105: ``pdflip.front_x`` does not exist, the gate has no
``_pdflip_x_limit_for``, the front posts no header.

Groups
  a_*  the carrier module (headers, parse, tolerance, limit)
  b_*  D's gate, with the real numbers (admit / W31 / boundary / static ceiling / resume / switch)
  c_*  the W50 message stays parseable by the front (``_d_refusal_extent`` / ``is_x_refusal``)
  d_*  the front: decision sites record the X, the D-direct leg posts the headers, other legs none
  e_*  the plumbing is complete (request objects carry the two fields on every hop); e3-e5 run the REAL hop code
  f_*  the native /generate route stamps the headers like chat/completions (the front forwards /generate to D too)
  g_*  hand-over X (D-SHORT-DRAIN, busy D) and the capacity-park vote on the effective limit, as behaviour

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

import flliper.srt.managers.scheduler as sched_mod  # noqa: E402
import flliper.srt.pdflip.front as front_mod  # noqa: E402
from flliper.srt.managers.scheduler import Scheduler  # noqa: E402

# Reached through a guard, never as a bare import: at the base (226aad7105) `pdflip.front_x` does not exist, and a bare import
# would turn the whole file into ONE collection error instead of one red test per defect (the extraction-count trap).
try:
    from flliper.srt.pdflip import front_x  # noqa: E402
except ImportError:  # pragma: no cover - only the base commit
    front_x = None
from flliper.srt.pdflip.front import Front  # noqa: E402

PAGE = 64
STATIC_X = 12288


# ----------------------------------------------------------------------------- helpers
def _req(rid="pdflip-40-127", prompt=30957, cached=29440, front_x_=None, priced=None, out=()):
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
    for name in ("_pdflip_x_refuses", "pdflip_uncached_extent", "_pdflip_host_carry_tokens"):
        setattr(stub, name, MethodType(getattr(Scheduler, name), stub))
    return stub


def _verdict(caplog, stub, req):
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        refused = stub._pdflip_x_refuses(req, None)
    line = [r.getMessage() for r in caplog.records if "PDFLIP X-GATE rid=" in r.getMessage()]
    assert len(line) == 1, line
    return refused, line[0]


# ============================================================================ a_ module
def test_a1_headers_roundtrip():
    h = front_x.request_headers(1010, 237)
    assert h == {"x-pdflip-front-x": "1010", "x-pdflip-front-priced": "237"}
    assert front_x.parse_headers(h) == (1010, 237)


def test_a2_no_decision_x_means_no_header_and_absent_parses_to_none():
    assert front_x.request_headers(0, 5) == {}
    assert front_x.request_headers(None, 5) == {}
    assert front_x.parse_headers({}) == (None, None)
    assert front_x.parse_headers(None) == (None, None)


def test_a3_malformed_or_zero_x_is_absent_never_refuse_everything():
    assert front_x.parse_headers({"x-pdflip-front-x": "abc"}) == (None, None)
    assert front_x.parse_headers({"x-pdflip-front-x": "-5"}) == (None, None)
    assert front_x.parse_headers({"x-pdflip-front-x": "0"}) == (None, None)
    assert front_x.parse_headers({"x-pdflip-front-x": "1010", "x-pdflip-front-priced": "zz"}) == (1010, None)


def test_a4_limit_is_front_x_plus_one_page_under_the_static_ceiling():
    assert front_x.limit_for(12288, 1010, 64) == 1074
    assert front_x.limit_for(12288, 20000, 64) == 12288, "the static X stays the hard ceiling"
    assert front_x.limit_for(12288, None, 64) == 12288
    assert front_x.limit_for(0, 1010, 64) == 0, "a gate that is off stays off"
    assert front_x.tolerance_tokens(64) == 64 and front_x.tolerance_tokens(None) == 1


def test_a5_switch_off_carries_nothing_both_ends(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_FRONT_X_CARRY", "0")
    assert front_x.request_headers(1010, 237) == {}
    assert front_x.parse_headers({"x-pdflip-front-x": "1010"}) == (None, None)


def test_a6_stamp_request_copies_headers_and_leaves_others_untouched():
    obj = SimpleNamespace(weg2_front_x=None, weg2_front_priced=None)
    front_x.stamp_request(obj, SimpleNamespace(headers={"x-pdflip-front-x": "1010", "x-pdflip-front-priced": "237"}))
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
def test_b1_pdflip_40_127_is_refused_by_name_not_computed_silently(caplog):
    """THE BOOT'S CASE: priced 237, decision X=1010, D realised 1517 (< static 12288).
    Base: admit (silent). Now: W31, and the line names front_x / limit."""
    req = _req(front_x_=1010, priced=237)
    stub = _stub()
    assert stub.pdflip_uncached_extent(req, None) == 1517
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
    monkeypatch.setenv("FLLIPER_PDFLIP_FRONT_X_CARRY", "0")
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
    for name in ("_pdflip_answer_x_refusals", "pdflip_uncached_extent", "_pdflip_group_min_flags"):
        if hasattr(Scheduler, name):
            setattr(stub, name, MethodType(getattr(Scheduler, name), stub))
    return stub, sent


def test_c1_the_w50_body_names_the_front_x_and_stays_parseable_by_the_front():
    stub, sent = _answer_stub()
    req = _req(front_x_=1010, priced=237)
    req.time_stats = SimpleNamespace(trace_ctx=SimpleNamespace(abort=lambda **k: None))
    req.priority = 0
    stub.waiting_queue = [req]
    stub._pdflip_answer_x_refusals([req], None)
    assert len(sent) == 1
    msg = sent[0].finished_reason["message"]
    assert msg.startswith("W50 PdFlipTpPrefillExceeded: this group may prefill at most 1074 uncached")
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
    stub._pdflip_answer_x_refusals([req], None)
    msg = sent[0].finished_reason["message"]
    assert "at most 12288 uncached" in msg and "FRONT-X" not in msg


# ============================================================================ d_ the front
def _front_double(x_live=6733):
    f = SimpleNamespace(
        counters=collections.Counter(),
        tp_prefill_max_tokens=x_live,
        _front_price={"pdflip-40-127": 237},
    )
    return f


def _pending(rid, est, d_direct=True):
    return SimpleNamespace(rid=rid, est_uncached=est, d_direct=d_direct)


def test_d1_the_last_recorded_decision_x_rides_with_a_d_direct_leg():
    f = _front_double()
    Front._note_x_decision(f, "pdflip-40-127", 1010, 237)   # arrival verdict (X-ROUTE X=1010)
    Front._note_x_decision(f, "pdflip-40-127", 6057, 237)   # QUEUED-SHORT later (X=6057)
    h = Front._front_x_headers(f, "pdflip-40-127", None, False)
    assert h == {"x-pdflip-front-x": "6057", "x-pdflip-front-priced": "237", "x-pdflip-front-halfx": "3028"}
    assert f.counters["front_x_carried"] == 1
    assert Front._front_x_headers(f, "pdflip-40-127", None, False)["x-pdflip-front-x"] == "6733", \
        "a decision is consumed by the post; a re-post falls back to the X in force"


def test_d2_an_unrecorded_d_direct_rid_is_bounded_by_the_x_in_force():
    f = _front_double(x_live=6733)
    h = Front._front_x_headers(f, "pdflip-40-127", None, False)
    assert h["x-pdflip-front-x"] == "6733" and h["x-pdflip-front-priced"] == "237"


def test_d3_a_single_prefill_carries_no_x_a_leg_after_p_carries_x_and_halfx_at_its_depth():
    """H16V review A3: every hand-over to D carries X and halbX, the one after P at the depth D continues from (the
    whole prompt); a single-prefill leg carries no X law."""
    f = _front_double()
    Front._note_x_decision(f, "pdflip-1-1", 1010, 5)
    pend = _pending("pdflip-1-1", 5, d_direct=False)
    pend.est_prompt = 9000
    h = Front._front_x_headers(f, "pdflip-1-1", pend, False)
    x = int(f.tp_prefill_max_tokens)
    assert h == {"x-pdflip-front-x": str(x), "x-pdflip-front-priced": "0", "x-pdflip-front-halfx": str(x // 2)}, h
    assert Front._front_x_headers(f, "pdflip-1-1", None, True) == {}
    assert Front._front_x_headers(f, "pdflip-1-1", _pending("pdflip-1-1", 5, d_direct=True), False) != {}


def test_d4_switch_off_on_the_front_posts_no_header(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_FRONT_X_CARRY", "0")
    f = _front_double()
    Front._note_x_decision(f, "pdflip-1-1", 1010, 5)
    assert Front._front_x_headers(f, "pdflip-1-1", None, False) == {}


def test_d5_queued_short_records_the_x_of_its_own_line_on_the_request():
    from test_pdflip_x_sum_price_1005 import _Fake, _p

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
    for name in ("_asr_queued_short_to_d", "_x_idle_regrant", "_d_short_drain"):
        assert "_stamp_x_decision" in funcs[name], f"{name} hands a request to D without recording its X"
    leg2 = funcs["leg2"]
    assert "_front_x_headers" in leg2 and "**_post_kw" in leg2


def test_d7_the_arrival_grant_records_the_route_verdict_x():
    src = inspect.getsource(Front.handle_generate)
    assert src.count("_note_x_decision(self, rid, x_route, remainder)") == 1, "the SHORT grant (no Pending exists)"
    assert src.count("_stamp_x_decision(p, x_route, remainder)") == 1, "the queued / kept request's arrival verdict"


# ============================================================================ e_ plumbing
def test_e1_every_hop_carries_the_two_fields():
    from flliper.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
    from flliper.srt.managers.schedule_batch import Req

    for cls in (GenerateReqInput, TokenizedGenerateReqInput):
        fields = set(getattr(cls, "__dataclass_fields__", None) or getattr(cls, "__struct_fields__", ()) or ())
        assert {"weg2_front_x", "weg2_front_priced"} <= fields, cls.__name__
    assert "self.weg2_front_x = None" in inspect.getsource(Req.__init__)
    g = GenerateReqInput(text=["a", "b"], weg2_front_x=1010, weg2_front_priced=237)
    g.normalize_batch_and_arguments()
    assert (g[1].weg2_front_x, g[1].weg2_front_priced) == (1010, 237), "the batch split keeps the carrier"


def test_e2_tokenizer_scheduler_and_serving_layers_pass_the_fields_on():
    import flliper.srt.entrypoints.openai.serving_chat as chat
    import flliper.srt.entrypoints.openai.serving_completions as comp
    import flliper.srt.managers.tokenizer_manager as tm

    assert "weg2_front_x=obj.weg2_front_x" in inspect.getsource(tm)
    assert 'req.weg2_front_x = getattr(recv_req, "weg2_front_x", None)' in inspect.getsource(sched_mod)
    for mod in (chat, comp):
        assert "weg2_front_x.stamp_request(adapted_request, raw_request)" in inspect.getsource(mod)


# ============================================================================ e_ plumbing as BEHAVIOUR (review dx-r2 A2)
# The source-substring checks above stay as a cheap pin, but a line AFTER the copy that sets the field to None
# (mutant R3) or a ``... if False else None`` (R4) leaves every substring in place and switches the gate off. These
# tests run the REAL hop code and read the field at the far end.
class _FixtureTokenizer:
    name_or_path = "front-x-fixture"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text] + [0]


def _tokenized(front_x_=1010, priced=237, halfx=None):
    """GenerateReqInput -> the real ``TokenizerManager._create_tokenized_object`` on a stub self."""
    from unittest.mock import MagicMock

    from flliper.srt.managers.io_struct import GenerateReqInput
    from flliper.srt.managers.tokenizer_manager import TokenizerManager
    from flliper.srt.sampling.sampling_params import SamplingParams

    obj = GenerateReqInput(text="hi", sampling_params={}, weg2_front_x=front_x_, weg2_front_priced=priced,
                           weg2_front_halfx=halfx)
    obj.normalize_batch_and_arguments()
    stub = SimpleNamespace(
        preferred_sampling_params=None,
        sampling_params_class=SamplingParams,
        tokenizer=_FixtureTokenizer(),
        model_config=SimpleNamespace(vocab_size=250000),
        server_args=SimpleNamespace(reasoning_parser=None, model_path="front-x-fixture",
                                    disaggregation_transfer_backend="none"),
        fake_bootstrap_room_counter=0,
        rid_to_state={obj.rid: SimpleNamespace(time_stats=MagicMock())},
    )
    return TokenizerManager._create_tokenized_object(stub, obj, "hi", [1, 2, 3])


def _scheduler_req(tokenized):
    """TokenizedGenerateReqInput -> the real ``Scheduler._handle_generate_request_impl`` on a partial double;
    returns the ``Req`` it queues."""
    from unittest.mock import MagicMock

    from flliper.srt.disaggregation.utils import DisaggregationMode

    queued = []
    s = MagicMock()
    s.pdflip_dormant = False
    s.session_handover_runtime = None
    s.server_args = SimpleNamespace(enable_session_radix_cache=False, disaggregation_bootstrap_port=None,
                                    max_kv_per_request=0, allow_auto_truncate=False)
    s.model_config = SimpleNamespace(vocab_size=250000, hf_eos_token_id=[2], context_len=4096)
    s.disaggregation_mode = DisaggregationMode.NULL
    s.spec_algorithm = SimpleNamespace(is_dflash_family=lambda: False)
    s._cross_schedule_mode = False
    s.metrics_reporter = SimpleNamespace(enable_metrics=False)
    s.dllm_config = None
    s.tokenizer = _FixtureTokenizer()
    s.max_req_input_len = 4096
    s.grammar_manager.process_req_with_grammar.return_value = False
    s._add_request_to_queue.side_effect = queued.append
    Scheduler._handle_generate_request_impl(s, tokenized)
    assert len(queued) == 1, queued
    return queued[0]


def test_e3_tokenizer_manager_hop_keeps_the_carrier():
    t = _tokenized(1010, 237)
    assert (t.weg2_front_x, t.weg2_front_priced) == (1010, 237)
    t0 = _tokenized(None, None)
    assert (t0.weg2_front_x, t0.weg2_front_priced) == (None, None)


@pytest.fixture
def _published_server_args():
    """``Req.is_prefill_only`` reads the published ServerArgs (speculative_algorithm): publish the two fields it reads
    for the test and restore the process-global slot after (the slot is shared with other test files)."""
    from flliper.srt.runtime_context import get_context, reset_context

    ctx = get_context()
    saved = ctx._server_args
    ctx.set_server_args(SimpleNamespace(speculative_algorithm=None, strip_thinking_cache=False))
    try:
        yield
    finally:
        if saved is None:
            reset_context()
        else:
            ctx.set_server_args(saved)


def test_e4_scheduler_hop_puts_the_carrier_on_the_req(_published_server_args):
    req = _scheduler_req(_tokenized(1010, 237))
    assert req.weg2_front_x == 1010 and req.weg2_front_priced == 237


def test_e5_the_carrier_survives_the_whole_intake_and_reaches_the_gate(caplog, _published_server_args):
    """tokenizer hop -> scheduler hop -> D's gate: the request the front priced on X=1010 is refused at
    front_x + page, the same request without the carrier is admitted (static 12288)."""
    carried = _scheduler_req(_tokenized(1010, 237))
    carried.output_ids = []
    carried.prefix_indices = []
    carried.full_untruncated_fill_ids = list(range(1500))
    carried.host_hit_length = 0
    refused, line = _verdict(caplog, _stub(), carried)
    assert refused is True and "front_x=1010" in line and "limit=1074" in line
    plain = _scheduler_req(_tokenized(None, None))
    plain.output_ids = []
    plain.prefix_indices = []
    plain.full_untruncated_fill_ids = list(range(1500))
    plain.host_hit_length = 0
    assert _verdict(caplog, _stub(), plain)[0] is False


# ============================================================================ f_ the native /generate route (review dx-r2 A1)
def _generate_route(headers, stream=False):
    """Run the real ``http_server.generate_request`` on a stub tokenizer manager; return the object the
    tokenizer manager was handed."""
    import asyncio
    import sys
    import types

    # ``http_server`` pulls in an audio helper that imports ``soundfile`` at module level; the CPU unit image may not
    # carry it. Provide an empty module for the duration of the import only (never leave a fake behind for other tests).
    _fake = None
    try:
        import soundfile  # noqa: F401
    except ImportError:
        _fake = sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))
    try:
        import flliper.srt.entrypoints.http_server as http_server
    finally:
        if _fake is not None and sys.modules.get("soundfile") is _fake:
            del sys.modules["soundfile"]
    from flliper.srt.managers.io_struct import GenerateReqInput

    seen = []

    class _TM:
        def generate_request(self, obj, request):
            seen.append(obj)

            async def _gen():
                yield {"text": "ok"}

            return _gen()

        def create_abort_task(self, obj):
            return None

    obj = GenerateReqInput(text="hi", sampling_params={}, stream=stream)
    obj.normalize_batch_and_arguments()
    old = http_server._global_state
    http_server._global_state = SimpleNamespace(tokenizer_manager=_TM())
    try:
        asyncio.run(http_server.generate_request(obj, SimpleNamespace(headers=headers)))
    finally:
        http_server._global_state = old
    assert len(seen) == 1
    return seen[0]


def test_f1_native_generate_stamps_the_front_x_headers():
    """The front forwards /generate to D (FORWARD_PATHS): a header on it must reach the request object."""
    assert "/generate" in front_mod.FORWARD_PATHS
    obj = _generate_route({"x-pdflip-front-x": "1010", "x-pdflip-front-priced": "237"})
    assert (obj.weg2_front_x, obj.weg2_front_priced) == (1010, 237)


def test_f2_native_generate_without_headers_is_untouched_and_a_bad_header_is_absent():
    obj = _generate_route({})
    assert (obj.weg2_front_x, obj.weg2_front_priced) == (None, None)
    bad = _generate_route({"x-pdflip-front-x": "abc"})
    assert (bad.weg2_front_x, bad.weg2_front_priced) == (None, None)


def test_f3_native_generate_switch_off_stamps_nothing(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_FRONT_X_CARRY", "0")
    obj = _generate_route({"x-pdflip-front-x": "1010"})
    assert obj.weg2_front_x is None


# ============================================================================ g_ hand-over X and park vote as BEHAVIOUR (A3, A4)
def test_g1_d_short_drain_on_a_busy_d_stamps_the_busy_band_floor_not_the_x_in_force():
    """Review dx-r2 A3 (mutant R7). D decodes others (x_split + _d_holds_work): the drain was decided on X_busy
    (the band floor), so each request must carry THAT X -- stamping the X in force would let D compute up to the
    idle X for a drain that was priced on the busy one. Analogue of d5 (QUEUED-SHORT)."""
    import asyncio
    import time

    f = Front("http://p", "http://d", "D", "dx-r2", "", 0, 0, {}, 45.0,
              tp_prefill_max_tokens=6733, d_short_drain_tokens=6733)
    f.x_split = True
    f._d_holds_work = lambda: True
    f._x_band_floor = lambda: 1010

    async def body():
        for i, n in enumerate((200, 300)):
            fut = asyncio.get_running_loop().create_future()
            f.queue.append(front_mod.Pending(f"s{i}", "/generate", {}, "x", time.time(), fut,
                                             est_prompt=n, est_uncached=n, d_eligible=True))
        return f._d_short_drain(time.time()), list(f._ready_for_d)

    moved, ready = asyncio.run(body())
    assert moved == 2
    assert [p.x_decision[0] for p in ready] == [1010, 1010], "busy D: the drain's X is X_busy, not the X in force (6733)"


def test_g2_d_short_drain_on_an_idle_d_stamps_the_x_in_force():
    import asyncio
    import time

    f = Front("http://p", "http://d", "D", "dx-r2", "", 0, 0, {}, 45.0,
              tp_prefill_max_tokens=6733, d_short_drain_tokens=6733)
    f.x_split = True
    f._d_holds_work = lambda: False
    f._x_band_floor = lambda: 1010
    f._x_quiet = lambda now: True

    async def body():
        fut = asyncio.get_running_loop().create_future()
        f.queue.append(front_mod.Pending("s0", "/generate", {}, "x", time.time(), fut,
                                         est_prompt=300, est_uncached=300, d_eligible=True))
        return f._d_short_drain(time.time()), list(f._ready_for_d)

    moved, ready = asyncio.run(body())
    assert moved == 1 and ready[0].x_decision[0] == 6733


def _cappark_req(front_x_):
    """A streamed request whose store read ended short: total 6000, the store holds 1000 of it (5000 short)."""
    req = _req(prompt=6000, cached=0, front_x_=front_x_, priced=237 if front_x_ else None)
    req.stream = True
    req.priority = 0
    req._pdflip_store_short_fallback = True
    req._pdflip_store_delivered = 500
    req._pdflip_store_deliverable = 1000
    req.time_stats = SimpleNamespace(trace_ctx=SimpleNamespace(abort=lambda **k: None))
    return req


def _cappark_run(monkeypatch, front_x_):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    parked, kept = [], []
    monkeypatch.setattr(sched_mod._pdflip_rvp, "park_for_capacity", lambda sched, req, d_extent, x: parked.append((req.rid, x)))
    monkeypatch.setattr(sched_mod._pdflip_rvp, "keep_on_d", lambda sched, req, d_extent, x: kept.append((req.rid, x)))
    stub, sent = _answer_stub()
    req = _cappark_req(front_x_)
    stub.waiting_queue = [req]
    stub._pdflip_answer_x_refusals([req], None)
    return parked, kept, sent


def test_g3_capacity_park_votes_on_the_effective_limit_not_the_static_x(monkeypatch):
    """Review dx-r2 A4 (mutant R5). The front priced the request on X=1010, so D's gate refuses it above 1074. Its
    store read ended 5000 tokens short of the whole context. With the STATIC X (12288) the capacity-park vote
    would say "a whole read fits" and park it -- but the whole read still misses the front limit: parked, re-read,
    re-gated, W31 again, a D-internal loop until FLLIPER_PDFLIP_RVP_CAPACITY_PARK_S. On the effective limit the read
    does not fit, so it is NOT parked: it takes the refusal path (RESUME-VIA-P for a stream, W50 by name otherwise)."""
    parked, kept, sent = _cappark_run(monkeypatch, front_x_=1010)
    assert parked == [], "a request refused at front_x + page must not capacity-park on the static X"
    # a streamed request goes to RESUME-VIA-P (keep_on_d, limit 1074), a non-stream one gets W50 by name: exactly one of the two
    assert len(kept) + len(sent) == 1, (kept, sent)
    if kept:
        assert kept[0][1] == 1074, "RESUME-VIA-P is told the effective limit, not the static X"


def test_g4_the_same_request_without_the_carrier_still_parks_for_capacity(monkeypatch):
    """Positive control for g3: without the carrier the effective limit IS the static X, the whole read fits
    (5000 <= 12288) and the #248h park happens exactly as before -- so g3 discriminates the limit, not the setup."""
    parked, kept, sent = _cappark_run(monkeypatch, front_x_=None)
    assert parked == [("pdflip-40-127", STATIC_X)] and kept == [] and sent == []
