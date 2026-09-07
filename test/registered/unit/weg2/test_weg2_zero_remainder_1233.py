"""#1233 zero-remainder (weg2/zero-remainder-0907): can-it-fail checks for the
desk-provable halves -- the front's pricing holes (record 1j findings 2-4) and
the END-OF-PREFILL ANCHOR split rule of the PrefillAdder. Hermetic: no CUDA,
no server. The store-side halves (the N-1 anchor reaching the store, the
publish sweep at /flush_cache) are boot-proven, not desk-proven.
"""

import json
import os
from types import SimpleNamespace

import pytest

from sglang.srt.weg2.front import (
    CHUNK_TOKENS,
    Front,
    Pending,
    usage_of,
    usage_of_stream_tail,
)


# ---------------------------------------------------------------- pricing
def test_usage_of_marks_a_body_without_usage_or_meta_info_as_unpriced():
    pt, ct, comp, priced = usage_of({"choices": [{"text": "x"}]})
    assert (pt, ct, comp, priced) == (0, 0, 0, False)


def test_usage_of_prices_chat_usage_and_generate_meta_info():
    body = {"usage": {"prompt_tokens": 13225, "completion_tokens": 7,
                      "prompt_tokens_details": {"cached_tokens": 13224}}}
    assert usage_of(body) == (13225, 13224, 7, True)
    gen = {"meta_info": {"prompt_tokens": 40, "cached_tokens": 39, "completion_tokens": 1}}
    assert usage_of(gen) == (40, 39, 1, True)


def test_stream_tail_prices_the_trailing_usage_chunk():
    chunks = [
        {"choices": [{"delta": {"content": "a"}}]},
        {"choices": [{"delta": {"content": "b"}}]},
        {"choices": [], "usage": {"prompt_tokens": 84027, "completion_tokens": 300,
                                  "prompt_tokens_details": {"cached_tokens": 84026}}},
    ]
    tail = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
    assert usage_of_stream_tail(tail) == (84027, 84026, 300, True)


def test_stream_tail_without_usage_is_unpriced_not_zero_zero():
    tail = b"data: " + json.dumps({"choices": [{"delta": {"content": "a"}}]}).encode() + b"\n\ndata: [DONE]\n\n"
    assert usage_of_stream_tail(tail) == (0, 0, 0, False)


def _front() -> Front:
    return Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, carrier_max_tokens=27466)


def test_short_route_never_yields_w16_even_when_mispriced():
    f = _front()
    v = f._leg2_verdict(pt=20000, ct=0, priced=True, pending=None, single_prefill=False, stream=False, rid="r")
    assert v == "short_mispriced"
    assert f.counters["short_mispriced"] == 1
    assert f.counters["W16_Weg2DoublePrefillExceeded"] == 0


def test_single_prefill_route_is_never_priced_as_double_prefill():
    f = _front()
    p = Pending("r", "/v1/chat/completions", {}, "x", 0.0, None, skip_leg1=True)
    v = f._leg2_verdict(pt=84027, ct=0, priced=True, pending=p, single_prefill=True, stream=False, rid="r")
    assert v == "single_prefill"
    assert f.counters["W16_Weg2DoublePrefillExceeded"] == 0


def test_batch_stream_over_the_bound_is_counted_under_w16_by_name():
    f = _front()
    p = Pending("r", "/v1/chat/completions", {}, "x", 0.0, None)
    v = f._leg2_verdict(pt=13225, ct=0, priced=True, pending=p, single_prefill=False, stream=True, rid="r")
    assert v == "W16"
    assert f.counters["W16_Weg2DoublePrefillExceeded"] == 1
    assert f.counters["W16_stream_served"] == 1


def test_batch_leg2_within_one_chunk_serves():
    f = _front()
    p = Pending("r", "/v1/chat/completions", {}, "x", 0.0, None)
    assert f._leg2_verdict(13225, 13225 - CHUNK_TOKENS, True, p, False, False, "r") == "serve"


# ------------------------------------------------ END-OF-PREFILL ANCHOR split
def _adder_split(armed: bool, rem_chunk_tokens, start: int, length: int, total: int):
    import importlib

    import sglang.srt.managers.schedule_policy as sp

    old = os.environ.get("SGLANG_WEG2_END_ANCHOR")
    os.environ["SGLANG_WEG2_END_ANCHOR"] = "1" if armed else "0"
    try:
        importlib.reload(sp)
        adder = SimpleNamespace(rem_chunk_tokens=rem_chunk_tokens)
        req = SimpleNamespace(full_untruncated_fill_ids=list(range(total)), rid="rid")
        return sp.PrefillAdder._weg2_end_anchor_split(adder, req, start, length)
    finally:
        if old is None:
            os.environ.pop("SGLANG_WEG2_END_ANCHOR", None)
        else:
            os.environ["SGLANG_WEG2_END_ANCHOR"] = old
        importlib.reload(sp)


def test_split_holds_the_last_token_back_when_the_extend_reaches_the_end():
    # 13225-token prompt, three 4096 chunks already done: the last chunk of
    # 937 becomes 936 + a 1-token chunk, so a boundary (anchor) lands at N-1.
    assert _adder_split(True, 4096, 12288, 937, 13225) == (936, True)


def test_split_is_identity_when_disarmed_off_chunking_or_mid_prompt():
    assert _adder_split(False, 4096, 12288, 937, 13225) == (937, False)
    assert _adder_split(True, None, 12288, 937, 13225) == (937, False)
    assert _adder_split(True, 4096, 0, 4096, 13225) == (4096, False)
    assert _adder_split(True, 4096, 13224, 1, 13225) == (1, False)


def test_reader_claim_bound_is_n_minus_one_tokens():
    # The invariant the split serves: a reader may claim at most N-1 tokens.
    from sglang.srt.managers.schedule_batch import Req

    fake = SimpleNamespace(return_logprob=False, logprob_start_len=0)
    assert Req._compute_max_prefix_len(fake, 13225) == 13224
