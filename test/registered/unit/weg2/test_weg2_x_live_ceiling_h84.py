"""H84 (boot x177 = fnFL2x177 on 09729d97c4, 25.09. 04:46-04:53Z): the live X
never left its 4096 floor, and D could not have taken a higher one anyway.

THE METAL, front log. Every re-solve read ``WEG2 X RE-SOLVE ... X=4096
r_D=78 r_P=2244..3649 flip_s=2.48..2.84``. The five r_D samples were
``uncached / leg-2 wall`` of the solo non-streamed legs 2: 683 tok / 8.77 s
(the wall carries the DECODE of the answer), 3899 / 6.61, 59 / 2.06,
1 / 4.21 (a P-routed leg 2 whose D extend was ONE tail token), 3891 / 5.59 ->
median 78 tok/s. D's own log prices the same prefills honestly (``Prefill rank
batch ... gpu-ms``): 683 tok 3.17 s, 3899 tok 5.68 s cold, 3827 tok 3.69 s warm
(1,036 tok/s); 48-64 tok take 0.87-1.23 s (the per-request base cost).

SECOND DEFECT: D refuses by W50 on its OWN ``--tp-prefill-max-tokens`` (the
start X), and the front only ever raised its own copy. An honest re-solve
(r_D 1036, r_P 3649, flip 2.6 s -> X* ~7.5k) would have routed 5-9k to D, and
D would have sent every one of them back through P.

THIRD: NF's D is bs1. A live X above the start X must not pull a burst onto D
one request after another (x177 burst 8 x 4.2k: 22.8 s batched over P, ~35 s
serially over D), so the band above the start X goes to D only as a singleton.

Hermetic, CPU, loopback sockets only: the REAL ``Front.leg2`` against a fake D
on loopback, the REAL ``Front.handle_generate`` with a recording leg 2, the REAL
re-solve, the REAL launcher argv builders, the REAL SchedulerReqTimeStats
pickle, output streamer and ``TokenizerManager._handle_batch_output``.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import json
import logging
import pickle
import textwrap
import time
from types import SimpleNamespace

import aiohttp
import msgspec
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from sglang.srt.environ import envs
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2.front import Front

START_X = 4096            # the launcher's X on x177 (source=flag), D's riegel
CEILING = 12288           # the arm's --x-ceiling-tokens
X177_R_D_WARM = 1036.0    # 3827 tok / 3.69 s, D gpu-ms
X177_R_P = 3649.0         # the first live r_P of x177
X177_FLIP_S = 2.6
MODEL = "/nonexistent/model"


# ------------------------------------------------------------------ (a) probe
async def _serve_once(pt: int, ct: int, extra: dict):
    """ONE non-streamed solo leg 2 through the REAL ``Front.leg2``: a fake D
    answers at once with ``usage`` plus ``extra`` (the OpenAI wire)."""

    async def chat(request: web.Request) -> web.Response:
        body = {"id": "x", "object": "chat.completion", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"},
             "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": 64,
                      "prompt_tokens_details": {"cached_tokens": ct}}}
        body.update(extra)
        return web.json_response(body)

    async def info(request: web.Request) -> web.Response:
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/get_server_info", info)
    d = TestServer(app)
    await d.start_server()
    f = Front("http://p", str(d.make_url("")).rstrip("/"), "D", "t", "", 0, 0, {}, 45.0,
              carrier_max_tokens=262144, tp_prefill_max_tokens=START_X)
    f.session = aiohttp.ClientSession()
    try:
        req = SimpleNamespace(path="/v1/chat/completions")
        await f.leg2(req, "weg2-0-1", {"messages": [{"role": "user", "content": "q"}]},
                     "q", False, None)
    finally:
        await f.session.close()
        await d.close()
    return f


def test_red_first_the_sample_is_ds_prefill_time_not_the_leg2_wall():
    """x177 rid weg2-0-1: 683 uncached, D prefilled them in 3.17 s. The fake
    D answers at once -- the sample must follow D's number, whatever the wall
    (the shipped probe divided by the wall: 78 tok/s on the metal)."""
    # TOLERANT ON PURPOSE (the parent tree has no such env): red there for
    # the RATE it samples, not for a missing attribute.
    floor = getattr(envs, "SGLANG_WEG2_X_RD_MIN_UNCACHED", None)
    with (floor.override(512) if floor is not None else contextlib.nullcontext()):
        f = asyncio.run(_serve_once(683, 0, {"sglext": {"weg2_prefill_s": 3.17}}))
    assert list(f._x_samples["r_d"]) == [pytest.approx(683 / 3.17)]   # ~215 tok/s
    assert "src=d_prefill_s" in f.x_flip_s_provenance()


def test_red_first_no_prefill_time_is_no_sample_never_the_wall():
    f = asyncio.run(_serve_once(3899, 0, {}))
    assert len(f._x_samples["r_d"]) == 0
    assert f.counters["r_d_skipped_no_prefill_time"] == 1


def test_red_first_a_tail_extend_of_one_token_is_no_sample():
    """x177 rid weg2-0-4: P prefilled 97,840 tokens, D extended ONE -- the
    shipped probe booked 1 tok / 4.21 s into the r_D median."""
    f = asyncio.run(_serve_once(97841, 97840, {"sglext": {"weg2_prefill_s": 0.013}}))
    assert len(f._x_samples["r_d"]) == 0
    assert f.counters["r_d_skipped_short"] == 1


def test_red_first_an_extent_below_2048_is_no_sample_by_default():
    """683 tokens is base-cost dominated (48-64 tok already take ~1 s on D)."""
    f = asyncio.run(_serve_once(683, 0, {"sglext": {"weg2_prefill_s": 3.17}}))
    assert len(f._x_samples["r_d"]) == 0
    assert f.counters["r_d_skipped_short"] == 1
    assert envs.SGLANG_WEG2_X_RD_MIN_UNCACHED.get() == 2048


def test_the_probe_boundaries():
    assert front_mod.r_d_probe(2048, 2.0, 2048) == (1024.0, "sample")
    assert front_mod.r_d_probe(2047, 2.0, 2048) == (None, "short")
    assert front_mod.r_d_probe(3899, 0.0, 2048) == (None, "no_prefill_time")
    assert front_mod.r_d_probe(3899, None, 2048) == (None, "no_prefill_time")


def test_the_front_reads_both_wires_and_nothing_else():
    rd = front_mod.d_prefill_seconds
    assert rd({"meta_info": {"prompt_tokens": 5, "weg2_prefill_s": 3.17}}) == 3.17
    assert rd({"sglext": {"weg2_prefill_s": 3.17}}) == 3.17
    assert rd({"usage": {"prompt_tokens": 5}}) is None
    assert rd({"sglext": {"weg2_prefill_s": 0.0}}) is None
    assert rd([{"meta_info": {"weg2_prefill_s": 1.0}}]) is None


def test_the_openai_wire_carries_the_field_the_front_reads():
    """D's protocol and the front's reader must agree on the name and place."""
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse, SglExt

    def wire(ext):
        return json.loads(ChatCompletionResponse(
            id="x", created=0, model="m", choices=[],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            sglext=ext).model_dump_json())

    assert front_mod.d_prefill_seconds(wire(SglExt(weg2_prefill_s=3.17))) == 3.17
    assert "sglext" not in wire(None)
    assert SglExt().model_dump() == {}, "an unset field must not reach the wire"


# ------------------------------------------------------------ (b) the ceiling
def _front(**kw) -> Front:
    return Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                 carrier_max_tokens=262144, tp_prefill_max_tokens=START_X, **kw)


def _feed_x177(f: Front, r_d: float = X177_R_D_WARM, flip_s: float = X177_FLIP_S) -> None:
    f.note_x_sample("r_p", X177_R_P)
    f.note_x_sample("flip_s", flip_s)
    f.note_x_sample("r_d", r_d)          # completes the triple: re-solve


def test_red_first_without_the_flag_the_live_x_stays_at_ds_riegel(caplog):
    """X* = 2*2.6/(1/1036 - 1/3649) ~ 7,523. D's riegel is the start X, so a
    front without --x-ceiling-tokens may not route above 4096."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    _feed_x177(f)
    assert f.counters["x_resolves"] == 1
    assert f.tp_prefill_max_tokens == START_X
    line = next(r.getMessage() for r in caplog.records if "X RE-SOLVE n=" in r.getMessage())
    assert "X*=7523" in line and "clamp=ceiling" in line and f"ceiling={START_X}" in line


def test_with_the_ceiling_the_live_x_rises_to_the_break_even():
    f = _front(x_ceiling_tokens=CEILING)
    _feed_x177(f)
    assert f.tp_prefill_max_tokens == 7523


def test_the_ceiling_wins_over_the_break_even_and_the_floor_still_holds(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front(x_ceiling_tokens=6000)
    _feed_x177(f)
    assert f.tp_prefill_max_tokens == 6000
    g = _front(x_ceiling_tokens=CEILING)
    _feed_x177(g, r_d=3000.0, flip_s=0.1)          # X* ~ 3,374 < one chunk
    assert g.tp_prefill_max_tokens == g.x_floor_tokens == 4096
    assert any("clamp=floor" in r.getMessage() for r in caplog.records)


def test_a_ceiling_below_the_start_x_is_lifted_to_it():
    """D's riegel may never sit below what the front already routes to D."""
    assert _front(x_ceiling_tokens=2048).x_ceiling_tokens == START_X
    assert L.resolve_x_ceiling(2048, START_X)[:2] == (START_X, START_X)


# ----------------------------------------------------------- (c) the launcher
def test_launcher_without_the_flag_every_argv_is_unchanged():
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
    assert ns.x_ceiling_tokens == 0
    d_x, front_c, line = L.resolve_x_ceiling(ns.x_ceiling_tokens, START_X)
    assert (d_x, front_c) == (START_X, 0) and "off" in line
    d = L.argv_d("py", MODEL, [1, 1, 1], 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], x_tokens=d_x)
    assert d[d.index("--tp-prefill-max-tokens") + 1] == str(START_X)
    fa = L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 1, 1, START_X, START_X, "D",
                          x_ceiling_tokens=front_c)
    assert fa == L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 1, 1, START_X, START_X, "D")
    assert "--x-ceiling-tokens" not in fa


def test_launcher_with_the_flag_d_gets_the_riegel_the_front_the_ceiling():
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t",
                                      "--x-ceiling-tokens", str(CEILING)])
    d_x, front_c, _ = L.resolve_x_ceiling(ns.x_ceiling_tokens, START_X)
    assert (d_x, front_c) == (CEILING, CEILING)
    d = L.argv_d("py", MODEL, [1, 1, 1], 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], x_tokens=d_x)
    assert d[d.index("--tp-prefill-max-tokens") + 1] == str(CEILING)
    fa = L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 1, 1, START_X, START_X, "D",
                          x_ceiling_tokens=front_c)
    assert fa[fa.index("--tp-prefill-max-tokens") + 1] == str(START_X)
    assert fa[fa.index("--x-ceiling-tokens") + 1] == str(CEILING)


def test_main_hands_the_riegel_to_d_at_both_launch_sites_and_never_to_p():
    """Fehler 2 was a front raised alone. Both argv_d calls (dry and real) take
    D's riegel; argv_p takes no X term at all -- group P does not change.

    --d-only is a third argv_d site with its own riegel: no P, no flip, no
    front, D prefills every uncached length itself, so x_tokens is W50
    (max_kv_per_request) there and never the flip X."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(L.main)))
    donly = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "ns.d_only"]
    assert len(donly) == 1, "the --d-only branch of main not found"
    in_donly = {id(n) for n in ast.walk(donly[0])}
    names = {"argv_d": [], "argv_p": [], "front_argv_for": []}
    donly_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
            if id(node) in in_donly:
                donly_calls.append(node)
                continue
            used = {n.id for a in list(node.args) + [k.value for k in node.keywords]
                    for n in ast.walk(a) if isinstance(n, ast.Name)}
            names[node.func.id].append(used)
    assert len(names["argv_d"]) == 2 and all("d_x_tokens" in u for u in names["argv_d"])
    # --d-only: exactly one argv_d, no argv_p, no front -- and its riegel is W50.
    assert [c.func.id for c in donly_calls] == ["argv_d"]
    x_pos = list(inspect.signature(L.argv_d).parameters).index("x_tokens")
    x_arg = donly_calls[0].args[x_pos]
    assert isinstance(x_arg, ast.Name) and x_arg.id == "max_kv_per_request"
    # ...and the front is told the same number at both sites, or D's riegel
    # rises while the live X stays pinned to the start X.
    assert len(names["front_argv_for"]) == 2
    assert all("front_x_ceiling" in u for u in names["front_argv_for"])
    assert names["argv_p"], "argv_p call sites not found"
    for used in names["argv_p"]:
        assert not used & {"x_tokens", "d_x_tokens", "front_x_ceiling"}
    assert "x_tokens" not in inspect.signature(L.argv_p).parameters


# ---------------------------------------------------------------- (d) X-SOLO
class _Req:
    def __init__(self, tokens: int):
        self.path = "/generate"
        self._payload = {"text": "x" * (3 * tokens)}   # priced tokens + 1

    async def json(self):
        return self._payload


def _band_front(x_live: int = 7500) -> Front:
    """Start X 4096 and a live X a re-solve raised to 7500 -- which the
    shipped front also did whenever its r_D allowed it (no ceiling there)."""
    f = _front()
    f.tp_prefill_max_tokens = x_live
    f.on_d = []

    async def leg2(request, rid, payload, text, stream, pending=None,
                   single_prefill=False, seat=None):
        f.on_d.append(rid)
        return "served"

    f.leg2 = leg2
    return f


async def _drive(f: Front, sizes, window_ms: int = 40, gap: bool = True,
                 settle_s: float = 0.2):
    # TOLERANT ON PURPOSE, and only here: the parent tree has no such env, and
    # these cases must be red there for the ROUTE they take, not for a
    # missing attribute.
    window = getattr(envs, "SGLANG_WEG2_X_SOLO_WINDOW_MS", None)
    with (window.override(window_ms) if window is not None else contextlib.nullcontext()):
        tasks = []
        for n in sizes:
            tasks.append(asyncio.ensure_future(f.handle_generate(_Req(n))))
            if gap:
                await asyncio.sleep(0)   # this request is inside its window now
        await asyncio.sleep(settle_s)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def test_a_singleton_in_the_band_goes_to_d_after_the_window(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _band_front()
    asyncio.run(_drive(f, [5000]))
    assert f.on_d == ["weg2-0-1"] and f.counters["route_short"] == 1
    assert any("WEG2 X-SOLO rid=weg2-0-1 uncached=5001 X_live=7500 verdict=d reason=solo"
               in r.getMessage() for r in caplog.records)


def test_red_first_a_second_arrival_in_the_window_sends_both_to_p():
    """The burst shape: 4.2k prompts arriving together. Neither may be served
    serially on bs1 D; both queue for P's batch (route LONG on the start X)."""
    f = _band_front()
    asyncio.run(_drive(f, [4250, 4300]))
    assert f.on_d == []
    assert f.counters["route_long"] == 2 and len(f.queue) == 2
    assert f.counters["x_solo_p"] == 2


def test_red_first_a_busy_d_sends_the_band_request_to_p(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _band_front()
    f.groups["D"].outstanding["weg2-0-0"] = time.time()
    asyncio.run(_drive(f, [5000]))
    assert f.on_d == [] and f.counters["route_long"] == 1
    assert any("verdict=p reason=d_outstanding=1" in r.getMessage() for r in caplog.records)


def test_outside_the_band_nothing_waits_and_nothing_changes():
    """At/below the start X: SHORT as before; above the live X: LONG as before."""
    f = _band_front()
    # A window the test never waits out: anything held by it fails here.
    asyncio.run(_drive(f, [3000, 9000], window_ms=10_000, gap=False))
    assert f.on_d == ["weg2-0-1"]
    assert f.counters["route_long"] == 1
    assert f.counters["x_solo_d"] == f.counters["x_solo_p"] == 0


# ------------------------------------------------------ (e) D's meta_info
def test_red_first_the_prefill_time_survives_the_ipc_without_metrics():
    """Forward entry at the FIRST chunk, prefill finished at the LAST: every
    chunk, no queue, no decode -- and pickled although enable_metrics is off
    (SchedulerReqTimeStats.__getstate__ shipped {} then)."""
    ts = SchedulerReqTimeStats()
    ts.set_wait_queue_entry_time(90.0)
    ts.set_forward_entry_time(100.0)      # chunk 1
    ts.set_forward_entry_time(101.5)      # chunk 2 does not move the start
    ts.set_prefill_finished_time(103.17)
    assert ts.__getstate__() == {}, "unstamped: the IPC payload is unchanged"
    ts.stamp_weg2_prefill_s()
    got = pickle.loads(pickle.dumps([ts]))[0]
    assert got.weg2_prefill_s == pytest.approx(3.17)


class _StreamReq:
    def __init__(self, finished: bool):
        self.rid, self.http_worker_ipc, self.stream = "r1", None, True
        self._finished = finished
        self.finished_reason = SimpleNamespace(to_json=lambda: {"type": "stop"}) if finished else None
        self.finished_output, self.finished_len = False, None
        self.sampling_params = SimpleNamespace(stream_interval=1, skip_special_tokens=True,
                                               spaces_between_special_tokens=True, no_stop_trim=False)
        self.output_ids = self.output_ids_through_stop = [7]
        self.send_token_offset = self.send_output_token_logprobs_offset = 0
        self.send_decode_id_offset = 0
        self.decoded_text, self.origin_input_ids = "", [1, 2]
        self.reasoning_tokens = self.cached_tokens = self.retraction_count = 0
        self.cached_tokens_device = self.cached_tokens_host = self.cached_tokens_storage = 0
        self.mm_image_tokens = self.mm_audio_tokens = self.mm_video_tokens = 0
        self.multimodal_inputs = self.customized_info = None
        self.return_hidden_states = self.return_routed_experts = self.return_indexer_topk = False
        self.time_stats = SchedulerReqTimeStats()
        self.time_stats.set_forward_entry_time(10.0)
        self.time_stats.set_prefill_finished_time(13.17)

    def finished(self):
        return self._finished

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0

    def check_match_stop_str_prefix(self):
        return False


def _streamed_time_stats(x_tokens: int, finished: bool):
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.io_struct import unwrap_from_pickle
    from sglang.srt.managers.scheduler_components.output_streamer import SchedulerOutputStreamer
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    sent = []
    s = SchedulerOutputStreamer(
        send_to_detokenizer=SimpleNamespace(send_output=sent.append), tree_cache=None,
        ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
        server_args=SimpleNamespace(tp_prefill_max_tokens=x_tokens, speculative_algorithm=None,
                                    stream_interval=1, enable_request_time_stats_logging=False),
        is_generation=True, spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL, enable_hicache_storage=lambda: False)
    s.stream_output([_StreamReq(finished)], False)
    return pickle.loads(pickle.dumps(unwrap_from_pickle(sent[0].time_stats)))[0]


def test_red_first_a_weg2_d_stamps_the_finishing_output_only():
    assert _streamed_time_stats(START_X, True).weg2_prefill_s == pytest.approx(3.17)
    assert _streamed_time_stats(START_X, False).weg2_prefill_s == 0.0, "decode outputs unchanged"
    assert _streamed_time_stats(0, True).weg2_prefill_s == 0.0, "not a Weg-2 D: nothing stamped"


def _tokenizer_meta_info(x_tokens: int) -> dict:
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()
    from sglang.srt.managers.io_struct import BatchStrOutput, wrap_as_pickle
    from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
    from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = SimpleNamespace(tp_prefill_max_tokens=x_tokens, batch_notify_size=1,
                                     weight_version="1", speculative_algorithm=None)
    tm.rid_to_state, tm.enable_metrics, tm.enable_lora = {}, False, False
    tm.incremental_streaming_output, tm.dump_requests_folder = False, ""
    tm.crash_dump_folder = ""
    obj = SimpleNamespace(rid="r1", stream=False, return_logprob=False, lora_path=None,
                          log_metrics=False)
    state = ReqState(out_list=[], finished=False, event=asyncio.Event(), obj=obj,
                     time_stats=APIServerReqTimeStats())
    tm.rid_to_state["r1"] = state
    ts = SchedulerReqTimeStats()
    ts.set_forward_entry_time(10.0)
    ts.set_prefill_finished_time(13.17)
    ts.stamp_weg2_prefill_s()
    kw = {}
    for fld in msgspec.structs.fields(BatchStrOutput):
        if fld.name == "rids":
            kw[fld.name] = ["r1"]
        elif fld.name == "finished_reasons":
            kw[fld.name] = [{"type": "stop"}]
        elif fld.name == "output_strs":
            kw[fld.name] = ["ok"]
        elif fld.name == "time_stats":
            kw[fld.name] = wrap_as_pickle([ts])
        elif fld.name in ("prompt_tokens", "completion_tokens", "reasoning_tokens",
                          "cached_tokens", "retraction_counts", "spec_verify_ct"):
            kw[fld.name] = [0]
        elif fld.default is not msgspec.NODEFAULT or fld.default_factory is not msgspec.NODEFAULT:
            continue
        else:
            kw[fld.name] = [[]]
    asyncio.run(tm._handle_batch_output(BatchStrOutput(**kw)))
    return state.out_list[-1]["meta_info"]


def test_red_first_meta_info_carries_weg2_prefill_s_on_a_weg2_d_only():
    assert _tokenizer_meta_info(START_X)["weg2_prefill_s"] == pytest.approx(3.17)
    assert "weg2_prefill_s" not in _tokenizer_meta_info(0)
