"""S67 (NF rc16, 19.10.): every D-MATCH-PROBE died of ``KeyError: '<rid>~probe'`` before D's tree was asked.

Metal (boot ..._bf9ca46f45_1019_121629, front log 12:19:52-12:21:31Z): 9/9 ``PDFLIP D-PROBE-FAILED attempt=1
why=KeyError: 'pdflip-0-2~probe' next=give_up`` -> 9 ``D-PROBE-UNPRICED``, 0 ``PDFLIP D-PROBE``. D answered every
``POST /pdflip/match_probe`` with HTTP 200 (D log ``PDFLIP D-IDS rid=pdflip-0-2~probe`` / ``D-MM-ITEM-CACHE
rid=pdflip-0-1~probe``, then ``200 OK``, no traceback: the handler turned the exception into ``{"success": false,
"message": "KeyError: ...", "retry": false}``).

ROOT: ``http_server.pdflip_match_probe`` tokenizes with ``TokenizerManager._tokenize_one_request``, whose last step
``_create_tokenized_object`` reads ``self.rid_to_state[obj.rid].time_stats`` -- a ``ReqState`` exists only for a
request ``generate_request`` admitted (``_init_req_state``). The probe is no request; its rid ``<rid>~probe`` has no
state, so the read raised AFTER the tokenization (the D-IDS / D-MM-ITEM-CACHE lines) and BEFORE the scheduler.
S56/S56r never ran this path in a test: the endpoint's tokenize half was untested, the scheduler half and the front
half were tested on hand-made answers.

FIX: ``_tokenize_one_request(obj, stateless=True)`` -- no ``ReqState`` is read, none is created (the danger direction:
a probe that REGISTERS a state under ``~probe`` is an in-flight request nobody finishes). Every real request keeps the
upstream path byte for byte (``stateless`` defaults False; a missing state stays a KeyError there).

These tests run the REAL chain: the front's probe body -> the real endpoint -> the real tokenizer manager
tokenization -> the real coalescing drain -> IPC (pickle) -> the real ``Scheduler.handle_pdflip_match_probe`` on a
REAL hybrid (FULL + MAMBA) ``UnifiedRadixCache`` -> IPC -> the endpoint's JSON -> the real front parser
(``Front._d_probe_once``). SOLL R1/R6/R11: the set sum and R6's rest are D's exact numbers only if this probe answers.
"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(__file__))

import test_pdflip_s56r_review_1019 as _real_tree  # noqa: E402  (the REAL hybrid radix fixture of S56r)

from flliper.srt.entrypoints import http_server as hs  # noqa: E402
from flliper.srt.managers import scheduler as S  # noqa: E402
from flliper.srt.managers.io_struct import GenerateReqInput, PdFlipMatchProbeReqInput  # noqa: E402
from flliper.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip.front import Front  # noqa: E402
from flliper.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=30, suite="stage-a-pdflip-unit")

A = _real_tree.A


# ------------------------------------------------------------------------------------------------ the D server half

def _tm(communicator=None):
    """A REAL ``TokenizerManager`` (its own ``_tokenize_one_request`` / ``_create_tokenized_object`` /
    ``pdflip_match_probe`` drain), only the process plumbing replaced: no tokenizer process, no ZMQ."""
    tm = object.__new__(TokenizerManager)
    tm.rid_to_state = {}
    tm.server_args = SimpleNamespace(
        disable_radix_cache=False, language_only=False, enable_return_hidden_states=False,
        enable_custom_logit_processor=False, reasoning_parser=None, model_path="dummy",
        disaggregation_transfer_backend="mooncake", limit_mm_data_per_request=None,
        encoder_transfer_backend=None)
    tm.tokenizer = None
    tm.mm_processor = None
    tm.d_mm_item_cache = None
    tm.mm_lane_refusal = None
    tm.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"]),
                                      vocab_size=1 << 20, model_path="dummy")
    tm.pdflip_prompt_ids_digest = False
    tm.context_len = 1 << 20
    tm.num_reserved_tokens = 0
    tm.allow_auto_truncate = False
    tm.validate_total_tokens = True
    tm.is_generation = True
    tm.preferred_sampling_params = None
    tm.sampling_params_class = SamplingParams
    tm.auto_create_handle_loop = lambda: None
    if communicator is not None:
        tm.pdflip_match_probe_communicator = communicator
    return tm


def _sched(cache):
    """The scheduler's own handler on a REAL tree; only what a CPU test cannot build (model, mm processor) faked."""
    fake = SimpleNamespace(tree_cache=cache, model_config=SimpleNamespace(vocab_size=1 << 20),
                           pad_input_ids_func=None, _get_multimodal_inputs=None,
                           _try_apply_padded_mm_input_ids=None)
    fake._pdflip_match_probe_one = lambda *a: S.Scheduler._pdflip_match_probe_one(fake, *a)
    return fake


def _ipc_communicator(sched, seen):
    """The FanOutCommunicator's wire: the request and the answer cross a process boundary (pickled)."""
    async def comm(req):
        req = pickle.loads(pickle.dumps(req))
        seen.append(req)
        out = S.Scheduler.handle_pdflip_match_probe(sched, req)
        return [pickle.loads(pickle.dumps(out))]
    return comm


class _RawRequest:
    def __init__(self, body, serving_chat=None):
        self._body = body
        self.app = SimpleNamespace(state=SimpleNamespace(openai_serving_chat=serving_chat, anthropic_serving=None))

    async def json(self):
        return self._body


def _post(monkeypatch, tm, body, serving_chat=None):
    monkeypatch.setattr(hs, "_global_state", SimpleNamespace(tokenizer_manager=tm))
    resp = asyncio.run(hs.pdflip_match_probe(_RawRequest(body, serving_chat)))
    assert resp.status_code == 200
    return json.loads(resp.body)


def _front_parse(rid, text):
    """The front's own reading of D's answer (``Front._d_probe_once``: answer | None, why, retry)."""
    async def run():
        fut = asyncio.get_running_loop().create_future()
        fut.set_result((200, text))
        return await Front._d_probe_once(SimpleNamespace(), rid, fut)
    return asyncio.run(run())


# ------------------------------------------------------------------------------------------------------- the tests

def test_rc16_text_probe_reaches_d_tree_and_the_front_gets_d_numbers(monkeypatch):
    """THE REPRO (red on bf9ca46f45: ``{"success": false, "message": "KeyError: 'pdflip-0-2~probe'", "retry":
    false}`` -> front ``next=give_up``). The front's TEXT fast path (``/generate`` + ``input_ids_b64``, rc16
    pdflip-0-2: ``PDFLIP D-IDS rid=pdflip-0-2~probe ... src=ids``) through the real endpoint, the real tokenization,
    the real drain, IPC and the real scheduler handler on a REAL hybrid tree: the front gets D's admission numbers."""
    fx = _real_tree._fixture()
    _real_tree._prefill(fx, A, "a")
    query = A + [7, 8, 9, 10, 11]
    before = _real_tree._snapshot(fx.cache)
    seen = []
    tm = _tm(_ipc_communicator(_sched(fx.cache), seen))
    body = {"path": "/generate", "payload": {"rid": "pdflip-0-2", "input_ids_b64": dmp.ids_to_b64(query)}}
    got = _post(monkeypatch, tm, body)
    assert got.get("success") is True, got
    # the scheduler was asked, under the probe's rid, with the front's exact ids
    assert len(seen) == 1 and seen[0].rid == "pdflip-0-2" + dmp.PROBE_RID_SUFFIX
    assert list(seen[0].input_ids) == query
    # S56r A2 / P4: the probe wrote nothing into D's tree
    assert _real_tree._snapshot(fx.cache) == before
    # the front reads D's exact admission numbers (the real walk, measured AFTER the probe on the untouched tree)
    ans, why, retry = _front_parse("pdflip-0-2", json.dumps(got))
    assert ans is not None, why
    dev, host = _real_tree._admission_numbers(fx.cache, query)
    assert (ans.n, ans.device_tokens, ans.host_tokens) == (len(query), dev, host)
    assert ans.device_tokens > 0, "the cached prefix A must be found (not a cold zero)"


def test_the_probe_registers_no_request_state(monkeypatch):
    """DANGER DIRECTION (mutant ``mutants_s67_dprobe_stateless_1019.py`` M1/M2): the obvious other "fix" -- give the
    probe a ``ReqState`` under ``<rid>~probe`` -- leaves an in-flight request nobody finishes (no detokenizer answer
    ever pops it): drain, abort sweeps and the request census count it. After any probe, answered or failed,
    ``rid_to_state`` is exactly what it was."""
    fx = _real_tree._fixture()
    _real_tree._prefill(fx, A, "a")
    seen = []
    tm = _tm(_ipc_communicator(_sched(fx.cache), seen))
    real = object()
    tm.rid_to_state = {"pdflip-0-1": real}          # a real in-flight request of the same session
    for rid in ("pdflip-0-1", "pdflip-0-3"):
        got = _post(monkeypatch, tm, {"path": "/generate",
                                      "payload": {"rid": rid, "input_ids_b64": dmp.ids_to_b64(A[:20])}})
        assert got.get("success") is True, got
    assert tm.rid_to_state == {"pdflip-0-1": real}
    assert not any(str(k).endswith(dmp.PROBE_RID_SUFFIX) for k in tm.rid_to_state)


def test_a_media_probe_tokenizes_without_state_and_reaches_the_scheduler(monkeypatch):
    """rc16 pdflip-0-1/0-4 (``D-MM-ITEM-CACHE rid=pdflip-0-1~probe ... images=5`` then 200 + KeyError): a chat
    prompt with an image goes through the same ``_create_tokenized_object`` -- the mm half (D-MM-ITEM-CACHE) is
    faked, the stateless tokenization and the hand-over to the scheduler are real."""
    from array import array

    seen = []

    async def comm(req):
        req = pickle.loads(pickle.dumps(req))
        seen.append(req)
        from flliper.srt.managers.io_struct import PdFlipMatchProbeReqOutput
        return [PdFlipMatchProbeReqOutput(rid=req.rid, success=True, input_ids=array("q", req.input_ids),
                                          device_tokens=0, host_tokens=0, key_match_depth=0, mm_spans=[[2, 6]])]

    tm = _tm(comm)
    tm.mm_processor = object()

    async def _tok(text, is_cross):
        return [1, 2, 3, 4], None

    class _MmCache:
        async def process(self, obj, prompt, compact_ids, max_req_input_len):
            return SimpleNamespace(input_ids=[1, 2, -7, -7, -7, -7, 3, 4], token_type_ids=None, mm_items=[])

    tm._tokenize_texts = _tok
    tm.tokenizer = SimpleNamespace()   # a text prompt needs one; its ids come from ``_tok``
    tm.d_mm_item_cache = _MmCache()
    tm.max_req_input_len = 1 << 20

    class _Chat:
        def _convert_to_internal_request(self, req, raw):
            return GenerateReqInput(text="<img> hi", image_data=["data:image/png;base64,AAAA"],
                                    sampling_params={"max_new_tokens": 1}), None

    monkeypatch.setattr("flliper.srt.entrypoints.openai.protocol.ChatCompletionRequest",
                        lambda **kw: SimpleNamespace(**kw))
    got = _post(monkeypatch, tm, {"path": "/v1/chat/completions",
                                  "payload": {"rid": "pdflip-0-1", "model": "m", "messages": []}},
                serving_chat=_Chat())
    assert got.get("success") is True, got
    assert len(seen) == 1 and list(seen[0].input_ids) == [1, 2, -7, -7, -7, -7, 3, 4]
    assert tm.rid_to_state == {}


def test_a_real_request_keeps_the_upstream_stateful_path():
    """Default path unchanged: a real request's tokenization still stamps its ``ReqState`` (time_stats), and a
    missing state is still a KeyError there -- ``stateless`` is the probe's opt-in, never a silent ``.get``."""
    tm = _tm()
    stamped = []
    ts = SimpleNamespace(set_tokenize_finish_time=lambda: stamped.append(1))
    tm.rid_to_state = {"pdflip-9-9": SimpleNamespace(time_stats=ts)}
    obj = GenerateReqInput(rid="pdflip-9-9", input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 1})
    obj.normalize_batch_and_arguments()
    out = asyncio.run(tm._tokenize_one_request(obj))
    assert out.time_stats is ts and stamped == [1]
    lost = GenerateReqInput(rid="pdflip-9-10", input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 1})
    lost.normalize_batch_and_arguments()
    with pytest.raises(KeyError):
        asyncio.run(tm._tokenize_one_request(lost))
    # and the stateless form reads nothing, creates nothing
    probe = GenerateReqInput(rid="pdflip-9-10~probe", input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 1})
    probe.normalize_batch_and_arguments()
    out = asyncio.run(tm._tokenize_one_request(probe, stateless=True))
    assert list(out.input_ids) == [1, 2, 3] and out.time_stats is None
    assert set(tm.rid_to_state) == {"pdflip-9-9"}


def test_the_endpoint_tokenizes_the_probe_stateless_source_pin():
    """Wiring pin: the probe endpoint is the one caller that must pass ``stateless=True`` (a refactor that drops it
    brings back the rc16 class; the behavioural tests above catch it too, this names the site)."""
    import inspect

    src = inspect.getsource(hs.pdflip_match_probe)
    assert "_tokenize_one_request(obj, stateless=True)" in src
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "rid_to_state" not in code and "_init_req_state" not in code


# ============================================================================ S67r (review S67, Auflagen A2-A4)
# A5 / A6b (estimates and the unpriced class at the front's decision) belong to the decision-centre workflow
# (coordinator 19.10.: one number source, one decision in set_verdict) -- named in the report, not fixed here.

def _media_tm(process):
    """A REAL tokenizer manager for a media probe; ``process`` stands in for D-MM-ITEM-CACHE / the HF processor."""
    from array import array

    from flliper.srt.managers.io_struct import PdFlipMatchProbeReqOutput

    async def comm(req):
        req = pickle.loads(pickle.dumps(req))
        return [PdFlipMatchProbeReqOutput(rid=req.rid, success=True, input_ids=array("q", req.input_ids),
                                          device_tokens=0, host_tokens=0, key_match_depth=0, mm_spans=[[2, 6]])]

    tm = _tm(comm)
    tm.mm_processor = object()

    async def _tok(text, is_cross):
        return [1, 2, 3, 4], None

    class _MmCache:
        async def process(self, obj, prompt, compact_ids, max_req_input_len):
            return await process()

    tm._tokenize_texts = _tok
    tm.tokenizer = SimpleNamespace()
    tm.d_mm_item_cache = _MmCache()
    tm.max_req_input_len = 1 << 20
    return tm


class _Chat:
    def _convert_to_internal_request(self, req, raw):
        return GenerateReqInput(text="<img> hi", image_data=["data:image/png;base64,AAAA"],
                                sampling_params={"max_new_tokens": 1}), None


def _post_with_ticker(monkeypatch, tm, body, serving_chat):
    """The probe POST while a 10 ms ticker measures the loop's worst wake-up lag (the loop that serves /health)."""
    import time

    monkeypatch.setattr(hs, "_global_state", SimpleNamespace(tokenizer_manager=tm))
    lag = [0.0]

    async def run():
        stop = asyncio.Event()

        async def ticker():
            while not stop.is_set():
                t0 = time.perf_counter()
                await asyncio.sleep(0.01)
                lag[0] = max(lag[0], time.perf_counter() - t0 - 0.01)

        tk = asyncio.ensure_future(ticker())
        await asyncio.sleep(0.03)
        resp = await hs.pdflip_match_probe(_RawRequest(body, serving_chat))
        stop.set()
        await tk
        return resp

    resp = asyncio.run(run())
    return json.loads(resp.body), lag[0]


@pytest.mark.parametrize("permitted", [True, False])
def test_a2_media_probe_processor_runs_off_the_health_loop(monkeypatch, permitted):
    """A2 (rc16 D log: a probe's D-MM-ITEM-CACHE miss stood the /health loop 1628.9 / 2213.5 ms). The probe's
    processor chain runs under the leg's own switch (``mm_tokenize_offload.requested()``, decided by
    ``offload_permitted()``): permitted -> on the processing thread, the loop answers; a device-resident frontend
    (not permitted) keeps it on the loop exactly as for a leg. The stand-in does what ``qwen_vl.py`` does with the
    switch (``run_offloaded`` of the chain when requested, else inline)."""
    import time

    from flliper.srt.managers import mm_tokenize_offload as mmo

    monkeypatch.setattr(mmo, "offload_permitted", lambda: permitted)
    seen = []

    async def process():
        seen.append(mmo.requested())
        if mmo.requested():
            await mmo.run_offloaded(time.sleep, 0.4)
        else:
            time.sleep(0.4)
        return SimpleNamespace(input_ids=[1, 2, -7, -7, -7, -7, 3, 4], token_type_ids=None, mm_items=[])

    monkeypatch.setattr("flliper.srt.entrypoints.openai.protocol.ChatCompletionRequest",
                        lambda **kw: SimpleNamespace(**kw))
    got, lag = _post_with_ticker(monkeypatch, _media_tm(process),
                                 {"path": "/v1/chat/completions",
                                  "payload": {"rid": "pdflip-0-1", "model": "m", "messages": []}}, _Chat())
    assert got.get("success") is True, got
    assert seen == [permitted]
    assert mmo.requested() is False, "the switch does not outlive the probe"
    if permitted:
        assert lag < 0.2, f"the loop stood {lag:.3f} s for the probe's image chain"
    else:
        assert lag >= 0.3, "not permitted: the chain stays on the loop (device frontend), as for a leg"


def test_a2_a_text_probe_never_offloads():
    """A text probe has no image chain: the section is not offloaded and opens no stats."""
    from flliper.srt.managers import mm_tokenize_offload as mmo

    with mmo.probe_section(rid="r~probe", is_mm=False) as off:
        assert off is False and mmo.requested() is False


def test_a3_stateless_true_only_in_the_probe_endpoints_pin():
    """A3 (reviewer mutant M5: ``generate_request`` with ``stateless=True`` survived every S67 test): a call with
    ``stateless=True`` exists ONLY inside the pdflip probe endpoints of http_server.py (the D-MATCH-PROBE and,
    after the S63r/S66 merge, TOK-AHEAD). Every other caller keeps the stateful upstream path."""
    import ast
    import pathlib

    allowed = {("entrypoints/http_server.py", "pdflip_match_probe"),
               ("entrypoints/http_server.py", "pdflip_tokenize_ahead")}
    root = pathlib.Path(hs.__file__).resolve().parents[1]           # python/flliper/srt
    found = []
    for path in root.rglob("*.py"):
        src = path.read_text(errors="replace")
        if "stateless" not in src:
            continue
        tree = ast.parse(src)
        stack = []

        def visit(node):
            is_fn = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_fn:
                stack.append(node.name)
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "stateless" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                        found.append((str(path.relative_to(root)), stack[-1] if stack else "<module>", node.lineno))
            for ch in ast.iter_child_nodes(node):
                visit(ch)
            if is_fn:
                stack.pop()

        visit(tree)
    # the internal hand-over _tokenize_one_request -> _create_tokenized_object passes the parameter through
    found = [f for f in found if f[:2] != ("managers/tokenizer_manager.py", "_tokenize_one_request")]
    assert found, "the probe endpoint must pass stateless=True"
    stray = [f for f in found if f[:2] not in allowed]
    assert not stray, f"stateless=True outside the probe endpoints: {stray}"


def test_a3_generate_request_stamps_the_request_time_stats():
    """A3: a REAL request through the REAL ``generate_request`` -> ``_tokenize_one_request`` ->
    ``_create_tokenized_object``: the tokenized object carries the request's own ``time_stats`` and the tokenize
    finish is stamped (M5 -- generate_request with stateless=True -- loses both)."""
    from unittest.mock import AsyncMock, Mock

    tm = _tm()
    tm.server_args.dp_size = 1
    tm._set_default_priority = Mock()
    tm._set_default_spill_class = Mock()
    tm.request_logger = Mock()
    tm.is_pause = False
    tm.is_pause_cond = asyncio.Condition()

    class _CM:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    tm.model_update_lock = SimpleNamespace(reader_lock=_CM())
    tm._validate_and_resolve_lora = AsyncMock(return_value=None)
    tm.init_mm_dispatch_order()
    stamped = []
    ts = SimpleNamespace(set_tokenize_finish_time=lambda: stamped.append(1))

    def init_state(obj, request=None):
        tm.rid_to_state[obj.rid] = SimpleNamespace(time_stats=ts, prompt_token_ids=None, pdflip_prompt_ids=None,
                                                   dispatched=False)

    tm._init_req_state = init_state
    sent = []

    class _Stop(Exception):
        pass

    def send(tok):
        sent.append(tok)
        raise _Stop()

    tm._send_one_request = send
    tm._release_req_states_on_failure = lambda rids: [tm.rid_to_state.pop(r, None) for r in rids]
    obj = GenerateReqInput(rid="pdflip-3-3", input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 1})

    async def drive():
        await tm.generate_request(obj).__anext__()

    with pytest.raises(_Stop):
        asyncio.run(drive())
    assert len(sent) == 1 and sent[0].time_stats is ts and stamped == [1]
