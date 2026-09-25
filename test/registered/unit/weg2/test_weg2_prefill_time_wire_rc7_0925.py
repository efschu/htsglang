# SPDX-License-Identifier: Apache-2.0
"""RC7-X: group D's prefill time on the RESPONSE wire -- the NF line's H84 D
patch (890b655dc4, applied verbatim to the 27B line so both lines carry the
same field: ``meta_info.weg2_prefill_s`` on /generate, ``sglext.weg2_prefill_s``
on a non-streamed OpenAI chat/completions answer).

These tests are the NF line's own (test_weg2_x_live_ceiling_h84.py, parts
"(a) wire" and "(e)"), ported unchanged in substance: the REAL
SchedulerReqTimeStats pickle, output streamer and
``TokenizerManager._handle_batch_output``, and the front's body reader
``d_prefill_seconds``. The 27B's second carrier (``internal_states[0]`` of
/get_server_info, for streamed legs and /v1/messages) is tested in
test_weg2_x_split_rc7_0925.py. Hermetic, CPU.
"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import msgspec
import pytest

from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.srt.weg2 import front as front_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

START_X = 4096


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
