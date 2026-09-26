"""X-EXACT -- the front prices the PENDING tokens exactly (user 26.09. ~19:00Z).

Memory d2p-sofort-flippen-und-x-exakt-0926: X holds EXACTLY for the pending
(uncached) tokens; there is NO 1.3*X band. The front therefore needs

    pending = tokens(the prompt as D renders it) - cached_on_D (measured)

MEASURED on boot dkr27brc10bar1agent09261821 (front.log, 114 requests): chars/3
over-counted whole prompts by a median 11.5 % (3.35 real chars/token); 1 SHORT
was really above X (weg2-10-7: priced 3939, D prefilled 4780 > X=4096), 2 LONG
were really <= X (weg2-30-97 4220/3917, weg2-30-106 8114/3841) and ~6 more P
routes were LONG only by the chars/3 over-count (12-20, 14-22, 18-30, 22-40,
22-41, 24-54). 9 of 78 D-direct requests were UNDER-priced by 1.0k-4.5k because
D could resume only at a Mamba anchor depth -- a D cache fact no tokenizer
fixes; X-EXACT logs it (X-EXACT-ERR), it does not band it.

Pinned here:
  (1) accuracy on the REAL tokenizers of both profiles (27B INT8, NF INT4):
      the front's count == D's own /v1/messages/count_tokens code, for
      Claude-Code-shaped payloads built from real source text; the
      incremental (per <|im_start|> segment) encode == the whole encode;
  (2) the credit is min(D's measurement, token LCP), bound to its epoch;
  (3) the routing decision at the boundary: pending == X -> SHORT,
      X + 1 -> LONG, where chars/3 would have said LONG / SHORT;
  (4) switch off = byte-identical: nothing constructed, the route lines are
      the pre-X-EXACT strings;
  (5) the count runs off the event loop.
"""

import asyncio
import json
import logging
import os
import random
import time

import numpy as np
import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import front as F
from sglang.srt.weg2 import front_tokens as FT

MC = "/spinning/llm_stuff/club-3090/models-cache/"
TOKENIZERS = {
    # profile/format -> checkpoint (weg2/form.py PROFILES[..].formats[..].checkpoint)
    "qwen27b-int8": MC + "Qwen3.8-27B-INT8-gdncov-vocabembed",
    "nextflash-int4": MC + "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
}
#: the 27B D group's rendering args on boot dkr27brc10bar1agent09261821
#: (its server_args line): these are what /get_server_info hands the front.
D_ARGS = dict(tokenizer_mode="auto", trust_remote_code=True, chat_template=None,
              chat_template_default_kwargs='{"preserve_thinking": true}',
              reasoning_parser="qwen3", tool_call_parser="qwen3_coder",
              served_model_name="Qwen3.8-27B")
X = 4096


def _args(path):
    return dict(D_ARGS, model_path=path, tokenizer_path=path)


def _need(path):
    if not os.path.exists(os.path.join(path, "tokenizer.json")):
        pytest.skip(f"tokenizer not on this box: {path}")


_LOADED = {}


def _ft(key, segmented=True):
    k = (key, segmented)
    if k not in _LOADED:
        path = TOKENIZERS[key]
        _need(path)
        ft = FT.FrontTokens()
        ft.load(_args(path), is_multimodal=True)
        assert ft.state == "ready", ft.why
        if not segmented:
            ft._seg.enabled = False
        _LOADED[k] = ft
    return _LOADED[k]


# ---------------------------------------------------------------------------
# Claude-Code-shaped payloads from REAL text (this repository's own sources)
# ---------------------------------------------------------------------------

_SRC = os.path.join(os.path.dirname(F.__file__), "front.py")


def _real_chunks(n, size, seed):
    text = open(_SRC, encoding="utf-8").read()
    rnd = random.Random(seed)
    return [text[i:i + size] for i in (rnd.randrange(0, len(text) - size) for _ in range(n))]


def _tools():
    return [{"name": f"Tool{i}", "description": f"Tool {i}: reads, edits or runs things. " * 12,
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string", "description": "file"},
                                             "limit": {"type": "integer"},
                                             "flags": {"type": "array", "items": {"type": "string"}}},
                              "required": ["path"]}} for i in range(14)]


def _conversation(turns, seed=7):
    chunks = _real_chunks(turns * 3 + 2, 3000, seed)
    msgs = [{"role": "user", "content": "Bitte den Fehler in front.py finden -- ä ö ü ß, 中文, emoji \U0001F600."}]
    for t in range(turns):
        msgs.append({"role": "assistant", "content": [
            {"type": "thinking", "thinking": chunks[3 * t][:900], "signature": ""},
            {"type": "text", "text": chunks[3 * t + 1][:400]},
            {"type": "tool_use", "id": f"toolu_{t:03d}", "name": "Tool1",
             "input": {"path": f"/spinning/x/{t}.py", "limit": 40}}]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"toolu_{t:03d}", "content": chunks[3 * t + 2]}]})
    return {"model": "Qwen3.8-27B", "max_tokens": 4000, "stream": True,
            "system": [{"type": "text", "text": "You are Claude Code.\n" + chunks[-1]}],
            "tools": _tools(), "messages": msgs,
            "thinking": {"type": "enabled", "budget_tokens": 2000}}


def _d_count_tokens(ft_ref, payload):
    """D's own /v1/messages/count_tokens handler (entrypoints/anthropic/serving.py)."""
    from sglang.srt.entrypoints.anthropic.protocol import AnthropicCountTokensRequest

    req = AnthropicCountTokensRequest(**{k: payload[k] for k in
                                         ("model", "system", "tools", "messages", "thinking")})
    resp = asyncio.run(ft_ref._anth.handle_count_tokens(req, None))
    return json.loads(resp.body)["input_tokens"]


@pytest.mark.parametrize("key", sorted(TOKENIZERS))
def test_count_equals_d_count_tokens_on_real_text(key):
    ft, ref = _ft(key), _ft(key, segmented=False)
    for turns in (0, 3, 12):
        p = _conversation(turns, seed=turns + 1)
        p["rid"] = "weg2-1-1"  # the front injects it before pricing (#1442)
        c = ft.count("/v1/messages", p)
        assert c.n == _d_count_tokens(ref, p)
        assert c.ids.tolist() == list(ref.count("/v1/messages", p).ids)


@pytest.mark.parametrize("key", sorted(TOKENIZERS))
def test_next_turn_reuses_the_prefix_and_stays_exact(key):
    ft, ref = _ft(key), _ft(key, segmented=False)
    p = _conversation(10, seed=31)
    first = ft.count("/v1/messages", p)
    p["messages"].append({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})
    p["messages"].append({"role": "user", "content": "und weiter"})
    nxt = ft.count("/v1/messages", p)
    assert nxt.ids.tolist() == list(ref.count("/v1/messages", p).ids)
    assert nxt.reused > 0.9 * first.n  # the earlier turns come from the cache
    assert nxt.encoded < 0.1 * first.n


@pytest.mark.parametrize("key", sorted(TOKENIZERS))
def test_openai_chat_shape_equals_the_serving_code(key):
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    ft, ref = _ft(key), _ft(key, segmented=False)
    chunks = _real_chunks(4, 2500, 5)
    p = {"model": "m", "messages": [{"role": "system", "content": chunks[0]},
                                    {"role": "user", "content": chunks[1]},
                                    {"role": "assistant", "content": chunks[2]},
                                    {"role": "user", "content": chunks[3]}]}
    want = ref._chat._process_messages(ChatCompletionRequest(**p), True).prompt_ids
    assert ft.count("/v1/chat/completions", p).ids.tolist() == list(want)


@pytest.mark.parametrize("key", sorted(TOKENIZERS))
def test_segmented_encode_is_the_whole_encode_on_hostile_boundaries(key):
    ft = _ft(key)
    tok = ft._tok._tok
    seg = FT.SegmentEncoder(tok)
    assert seg.enabled, seg.why
    rnd = random.Random(3)
    pieces = ["  ", "\n", "\n\n", " \t", "é", "é", "́", "<|im_end|>", "<|im_start|>",
              "<|im_start", "|>", "abc", "def x():", "中文", "\U0001F600", "ß", "\r\n", "  \n"]
    for _ in range(300):
        s = "".join(rnd.choice(pieces) for _ in range(rnd.randrange(1, 40)))
        assert seg.encode(s) == tok.encode(s), repr(s)


def test_inline_system_in_place_renders_like_d():
    """MZ (SGLANG_ANTHROPIC_INLINE_SYSTEM_IN_PLACE) is read by the adapter at
    construction in D; the front builds the same adapter under the same env."""
    path = TOKENIZERS["qwen27b-int8"]
    _need(path)
    p = _conversation(2, seed=9)
    p["messages"].insert(1, {"role": "system", "content": "inline system reminder"})
    got = {}
    for on in (False, True):
        with envs.SGLANG_ANTHROPIC_INLINE_SYSTEM_IN_PLACE.override(on):
            ft = FT.FrontTokens()
            ft.load(_args(path), is_multimodal=True)
            ref = FT.FrontTokens()
            ref.load(_args(path), is_multimodal=True)
            ref._seg.enabled = False
            got[on] = ft.count("/v1/messages", p).n
            assert got[on] == _d_count_tokens(ref, p)


def test_cost_and_event_loop_stays_free(capsys):
    """Kosten: printed for the report; the loop's worst tick lag stays small
    while a cold ~60k-token count runs in the worker thread."""
    def rss_mib():
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
        return -1

    path = TOKENIZERS["qwen27b-int8"]
    _need(path)
    rss0 = rss_mib()
    ft = FT.FrontTokens()
    ft.load(_args(path), is_multimodal=True)
    rss1 = rss_mib()
    p = _conversation(18, seed=77)

    async def run():
        lag = 0.0
        stop = False

        async def ticker():
            nonlocal lag
            while not stop:
                t = time.perf_counter()
                await asyncio.sleep(0.001)
                lag = max(lag, time.perf_counter() - t - 0.001)

        tk = asyncio.create_task(ticker())
        await asyncio.sleep(0.01)
        loop = asyncio.get_running_loop()
        cold = await loop.run_in_executor(ft.executor, ft.count, "/v1/messages", p)
        p["messages"].append({"role": "user", "content": "weiter"})
        warm = await loop.run_in_executor(ft.executor, ft.count, "/v1/messages", p)
        stop = True
        await tk
        return cold, warm, lag

    cold, warm, lag = asyncio.run(run())
    with capsys.disabled():
        print(f"\nX-EXACT COST rss_load_mib={rss1 - rss0} load_s={ft.load_s:.1f} cold n={cold.n} {cold.ms:.0f} ms, "
              f"warm n={warm.n} {warm.ms:.0f} ms (reused {warm.reused}), loop max lag {lag * 1000:.1f} ms")
    assert warm.ms < cold.ms
    assert lag < 0.25


# ---------------------------------------------------------------------------
# (2) the credit: min(D's measurement, token LCP), bound to its epoch
# ---------------------------------------------------------------------------

def _ids(*runs):
    return np.concatenate([np.arange(a, b, dtype=np.int32) for a, b in runs])


def test_credit_is_the_measured_share_capped_by_the_token_lcp():
    ts = FT.TokenSpans(agent_span=True)
    prev = _ids((0, 10000))
    ts.record_presence(prev, cached_tokens=8000, prompt_tokens=10000, held_epoch=None)
    new = _ids((0, 9000), (50000, 53000))  # shares 9000 tokens, 3000 new
    assert ts.pending(new) == (12000 - 8000, 8000, True, "d_leg2_cached")
    # D measured more than the two texts share -> the LCP bounds it
    ts.record_presence(prev, cached_tokens=9990, prompt_tokens=10000)
    assert ts.pending(new)[:2] == (3000, 9000)


def test_held_credit_only_in_its_epoch_and_a_zero_retracts():
    ts = FT.TokenSpans(agent_span=True)
    prev = _ids((0, 10000))
    new = _ids((0, 10000), (70000, 70500))
    ts.record_presence(prev, cached_tokens=2000, prompt_tokens=10000, held_epoch=5)
    assert ts.pending(new, epoch=5) == (500, 10000, True, "d_served_epoch")
    assert ts.pending(new, epoch=6)[:2] == (8500, 2000)
    assert ts.pending(new, epoch=None)[:2] == (8500, 2000)
    ts.record_presence(prev, cached_tokens=0, prompt_tokens=10000, held_epoch=None)
    assert ts.pending(new, epoch=5) == (10500, 0, False, "none")


def test_agent_span_off_keeps_only_the_presence_witness():
    ts = FT.TokenSpans(agent_span=False)
    prev = _ids((0, 10000))
    ts.record_presence(prev, cached_tokens=2000, prompt_tokens=10000, held_epoch=5)
    ts.record_inflight(prev, 5)  # the caller only calls it with #49 on; harmless here
    assert ts.pending(_ids((0, 10000), (9e5, 9e5 + 10)), epoch=None)[:2] == (8010, 2000)


# ---------------------------------------------------------------------------
# (3)/(4) the front itself: routing at the boundary, fallback, switch off
# ---------------------------------------------------------------------------

class _Req:
    def __init__(self, payload, path="/v1/messages"):
        self._p = payload
        self.path = path

    async def json(self):
        return self._p


class _FakeTokens:
    """ready, counts every prompt as ``n`` distinct tokens (routing only)."""

    def __init__(self, n):
        from concurrent.futures import ThreadPoolExecutor

        self.state, self.why, self.n = "ready", "fake", n
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.ids_by_text = {}

    def count(self, path, payload):
        ids = np.arange(self.n, dtype=np.int32) + 10
        return FT.Count(n=self.n, ids=ids, ms=0.1, reused=0, encoded=self.n)

    def remember(self, text, ids):
        self.ids_by_text[text] = ids

    def ids_for(self, text):
        return self.ids_by_text.get(text)


def _front(exact):
    with envs.SGLANG_WEG2_FRONT_EXACT_TOKENS.override(exact):
        f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="xexact",
                    store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
                    weight_chunks=2, tp_prefill_max_tokens=X, flip_min_work_tokens=X)
    f.state = "serving"
    f.routed = []

    async def seat(rid, est, refused=None):
        return 1

    async def leg2(request, rid, payload, text, stream, pending=None, **kw):
        f.routed.append(("short", rid))
        return F.web.json_response({})

    async def solo(rid, rem):
        return True

    f._acquire_short_seat = seat
    f.leg2 = leg2
    f._x_solo_admits = solo
    f._kick_controller = lambda *a, **k: None
    return f


def _route(f, payload, path="/v1/messages"):
    async def go():
        t = asyncio.create_task(f.handle_generate(_Req(payload, path)))
        await asyncio.sleep(0.2)
        if not t.done():
            t.cancel()
            try:
                await t
            except BaseException:  # noqa: BLE001
                pass
    asyncio.run(go())
    if f.routed:
        return "short"
    return "long" if f.counters["route_long"] else "queued"


def _payload(chars):
    return {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "a" * chars}]}


@pytest.mark.parametrize("n,want", [(X, "short"), (X + 1, "long"), (1, "short")])
def test_exact_pending_decides_at_the_boundary(n, want, caplog):
    f = _front(True)
    f.ftok = _FakeTokens(n)
    # chars/3 would price ~2x the exact count: 2*X chars -> ~(2X/3 + 11) est.
    chars = 3 * X + 600 if n <= X else 3 * X - 900  # est says LONG for n<=X, SHORT for X+1
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert _route(f, _payload(chars)) == want
    msgs = [r.getMessage() for r in caplog.records]
    price = [m for m in msgs if m.startswith("WEG2 X-EXACT-PRICE")]
    assert len(price) == 1 and f"pending={n} tokens={n} " in price[0]
    verdict = [m for m in msgs if m.startswith("WEG2 ROUTE-VERDICT")][0]
    assert f"uncached={n} " in verdict and "X-EXACT:" in verdict
    assert "EXACT, front tokenizer" in [m for m in msgs if m.startswith("WEG2 X-ROUTE")][0]


def test_the_measured_d_prefix_is_subtracted(caplog):
    f = _front(True)
    f.ftok = _FakeTokens(X + 3000)
    ids = np.arange(X + 3000, dtype=np.int32) + 10
    f.tspans.record_presence(ids[:3000], cached_tokens=2999, prompt_tokens=3000)
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert _route(f, _payload(100)) == "long"  # X + 1 pending
    f2 = _front(True)
    f2.ftok = _FakeTokens(X + 3000)
    f2.tspans.record_presence(ids[:3000], cached_tokens=3000, prompt_tokens=3000)
    assert _route(f2, _payload(100)) == "short"  # exactly X pending


def test_not_ready_falls_back_to_the_estimate_by_name(caplog):
    f = _front(True)
    f.ftok.state = "loading"
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert _route(f, _payload(3 * X + 600)) == "long"  # chars/3 decides
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("WEG2 X-EXACT-FALLBACK") and "reason=tokenizer_loading" in m
               for m in msgs)
    assert f.counters["x_exact_fallback"] == 1


def test_residual_error_is_logged_not_banded(caplog):
    f = _front(True)
    f.ftok = _FakeTokens(3000)
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        _route(f, _payload(100))
        text = F.request_text(_payload(100))
        # D later reports it prefilled 4218 of 31602 (a Mamba-anchor miss, weg2-12-17 shape)
        f._x_exact_record(f.routed[0][1], text, 31602, 31602 - 4218, None, f.epoch)
    err = [r.getMessage() for r in caplog.records if r.getMessage().startswith("WEG2 X-EXACT-ERR")]
    assert len(err) == 1
    assert "pending_priced=3000 d_uncached=4218 err=-1218" in err[0] and "match=0" in err[0]
    assert f.counters["x_exact_err_n"] == 1


REF_X_ROUTE = "WEG2 X-ROUTE rid=%s est_uncached=%d X=%d (ESTIMATE, front pricing, no tokenizer)"


def test_switch_off_is_byte_identical(caplog):
    f = _front(False)
    assert f.x_exact is False and f.ftok is None and f.tspans is None
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert _route(f, _payload(3 * X + 600)) == "long"
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("X-EXACT" in m for m in msgs)
    rid = "weg2-0-1"
    est = int((3 * X + 600 + len("user:\n")) / F.CHARS_PER_TOKEN) + 1
    assert REF_X_ROUTE % (rid, est, X) in msgs
    verdict = [m for m in msgs if m.startswith("WEG2 ROUTE-VERDICT")][0]
    assert "X-EXACT" not in verdict
    assert f"uncached={est} (base for X={X}, what D must PREFILL, at CHARS_PER_TOKEN=3.0 " in verdict


def test_off_constructs_and_imports_nothing():
    import inspect

    src = inspect.getsource(F.Front.handle_generate)
    # every X-EXACT statement in the handler sits behind the switch
    assert "if self.x_exact:\n            _xx = await self._x_exact_price(" in src
    assert src.count("self._x_exact_price(") == 1
    init = inspect.getsource(F.Front.__init__)
    assert "if self.x_exact:\n            from sglang.srt.weg2.front_tokens import" in init
    if "SGLANG_WEG2_FRONT_EXACT_TOKENS" not in os.environ:
        assert envs.SGLANG_WEG2_FRONT_EXACT_TOKENS.get() is False  # default off


# ---------------------------------------------------------------------------
# (6) registry: qwen27b and nextflash off until measured; explicit wins
# ---------------------------------------------------------------------------

from sglang.srt.weg2 import form as FM  # noqa: E402
from sglang.srt.weg2 import phase_policy as PP  # noqa: E402

XE = "SGLANG_WEG2_FRONT_EXACT_TOKENS"


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


def test_the_registry_rows_are_off_until_measured():
    for prof in ("qwen27b", "nextflash"):
        assert FM.PROFILES[prof].front_exact_tokens is False
        assert FM.PROFILE_SWITCH_DEFAULTS[prof][XE] is False


@pytest.mark.parametrize("profile,explicit,row_on,want", [
    ("qwen27b", None, False, False), ("nextflash", None, False, False), (None, None, False, False),
    ("qwen27b", None, True, True),    # the operator turns the row on -> on without an env
    ("nextflash", None, True, True),
    ("qwen27b", "1", False, True), ("qwen27b", "0", True, False), (None, "1", False, True)])
def test_the_switch_follows_the_profile_row_and_an_explicit_value_wins(
        monkeypatch, profile, explicit, row_on, want):
    monkeypatch.delenv(XE, raising=False)
    monkeypatch.delenv(FM.FORM_ENV, raising=False)
    if profile is not None:
        monkeypatch.setenv(FM.FORM_ENV, _form_env(profile))
        if row_on:
            monkeypatch.setitem(FM.PROFILE_SWITCH_DEFAULTS, profile,
                                dict(FM.PROFILE_SWITCH_DEFAULTS[profile], **{XE: True}))
    if explicit is not None:
        monkeypatch.setenv(XE, explicit)
    assert envs.SGLANG_WEG2_FRONT_EXACT_TOKENS.get() is want


# ---------------------------------------------------------------------------
# (7) the queue is re-priced exactly -- what PK's needs_p reads
# ---------------------------------------------------------------------------

def _queued(f, rid, text, n, est, **kw):
    ids = np.arange(n, dtype=np.int32) + 10
    f.ftok.remember(text, ids)
    fut = asyncio.new_event_loop().create_future()
    p = F.Pending(rid, "/v1/messages", {}, text, time.time(), fut,
                  est_prompt=n, est_uncached=est, **kw)
    f.queue.append(p)
    return p, ids


def test_a_new_d_credit_reprices_the_queue_and_needs_p_follows(caplog):
    f = _front(True)
    f.ftok = _FakeTokens(1)
    p, ids = _queued(f, "weg2-1-2", "twin", X + 500, X + 500)
    assert PP.immediate_park_trigger(f.queue, X) is p  # priced over X: the park would fire
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        # its twin's first content: D now holds the first X+300 tokens (#49 in-flight)
        f.tspans.record_inflight(ids[:X + 300], f.epoch)
        assert f._x_exact_reprice_queue("inflight") == 1
    assert p.est_uncached == 200
    assert PP.immediate_park_trigger(f.queue, X) is None  # 200 pending: no park, no flip
    line = [r.getMessage() for r in caplog.records if "X-EXACT-REPRICE" in r.getMessage()][0]
    assert f"est_uncached {X + 500} -> 200 X={X} crossed=down" in line


def test_the_epoch_change_ends_the_held_credit_and_raises_the_price():
    f = _front(True)
    f.ftok = _FakeTokens(1)
    f.tspans.agent_span = True  # #49 held credit (the qwen27b row's agent_span)
    p, ids = _queued(f, "weg2-1-3", "held", X + 100, 100)
    f.tspans.record_presence(ids[:X], cached_tokens=50, prompt_tokens=X, held_epoch=f.epoch)
    assert f._x_exact_reprice_queue("x") == 0  # same epoch: 100 stays
    f.epoch += 2  # a flip to P and back
    assert f._x_exact_reprice_queue("epoch") == 1
    assert p.est_uncached == X + 100 - 50  # only the measured 50 remain credited
    assert PP.immediate_park_trigger(f.queue, X) is p


def test_measured_and_final_prices_are_never_touched():
    f = _front(True)
    f.ftok = _FakeTokens(1)
    keep = [
        _queued(f, "a", "t-a", X + 9, X + 9, leg1_done=True)[0],
        _queued(f, "b", "t-b", X + 9, X + 9, skip_leg1=True)[0],
        _queued(f, "c", "t-c", X + 9, X + 9, p_only=True)[0],
    ]
    q, _ = _queued(f, "d", "t-d", X + 9, 7777)
    q.x_requeues = 1  # D's own extent stands
    fb = F.Pending("e", "/v1/messages", {}, "fallback-priced", time.time(),
                   asyncio.new_event_loop().create_future(), est_prompt=1, est_uncached=4242)
    f.queue.append(fb)  # chars/3 fallback: no ids
    assert f._x_exact_reprice_queue("x") == 0
    assert [p.est_uncached for p in keep] == [X + 9] * 3 and q.est_uncached == 7777
    assert fb.est_uncached == 4242


def test_off_never_reprices_and_the_hooks_sit_behind_the_switch():
    import inspect

    f = _front(False)
    f.queue.append(F.Pending("z", "/v1/messages", {}, "t", time.time(),
                             asyncio.new_event_loop().create_future(), est_prompt=1,
                             est_uncached=99))
    assert f._x_exact_reprice_queue("epoch") == 0 and f.queue[0].est_uncached == 99
    src = inspect.getsource(F.Front)
    assert "if self.x_exact:\n            # X-EXACT: a held (#49) credit is bound to its epoch" in src
    assert src.count("self._x_exact_reprice_queue(") == 3


def test_the_w31_requeue_counts_the_whole_prompt_exactly():
    import inspect

    src = inspect.getsource(F.Front)
    i = src.index("_est = len(text) // int(CHARS_PER_TOKEN) + 1")
    j = src.index("p = Pending(", i)
    body = src[i:j]
    assert "if self.x_exact:" in body and "_est = int(_ids.size)" in body
