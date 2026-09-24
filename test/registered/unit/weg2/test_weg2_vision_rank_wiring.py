"""The in-rank vision stage, WIRED into the scheduler (user design 2026-09-24) -- slice V2b.

Hermetic, CPU. Pinned:
  * the scheduler's admission never sees a held request and always gets it
    back, also when the admission raises; an unarmed scheduler admits
    exactly as before;
  * the waiting-queue abort echo carries the origin's finish reason, and the
    tokenizer's own aborts (no reason) echo unchanged;
  * the receiver injects the origin's extras on the origin only, before the
    relay; the scheduler wires the hook only when the stage is armed.
"""

import ast
import inspect
import textwrap
import types

import pytest
import torch

from sglang.srt.weg2 import vision_rank_runner as vrr


class _Item:
    def __init__(self):
        self.feature = torch.randn(4, 8)
        self.precomputed_embeddings = None
        self.image_grid_thw = torch.tensor([[1, 2, 2]])
        self.modality = "image"

    def is_image(self):
        return True


def _req(rid, items=()):
    return types.SimpleNamespace(
        rid=rid, multimodal_inputs=types.SimpleNamespace(mm_items=list(items)))


def _pass_sched(queue, *, idle=True):
    return types.SimpleNamespace(
        waiting_queue=list(queue),
        weg2_dormant=False,
        _weg2_vision_refused=set(),
        _weg2_vision_arm_refusal="",
        _weg2_vision_origin_aborts=[],
        _weg2_vision_runs=0,
        running_batch=types.SimpleNamespace(is_empty=lambda: idle),
        chunked_req=None,
        _pp_microbatches_drained=lambda: True,
        server_args=types.SimpleNamespace(model_path="/m"),
        model_config=types.SimpleNamespace(hf_config=None),
    )


@pytest.fixture
def stage_calls(monkeypatch):
    calls = []

    def fake(scheduler, reqs, **kw):
        calls.append([r.rid for r in reqs])
        return vrr.StageOutcome()

    monkeypatch.setattr(vrr, "run_rank_stage", fake)
    monkeypatch.setattr(vrr, "_rank_device", lambda: torch.device("cpu"))
    return calls


def _funnel(queue, idle):
    from sglang.srt.managers.scheduler import Scheduler

    h = _pass_sched(queue, idle=idle)
    h._weg2_vision_rank_stage = True
    h.prefill_delayer = None
    h.get_new_batch_prefill = types.MethodType(Scheduler.get_new_batch_prefill, h)
    return h


def test_the_admission_never_sees_a_held_request_and_always_gets_it_back(stage_calls):
    h = _funnel([_req("t1"), _req("i1", [_Item()])], idle=False)
    seen = []

    def raw(prefill_delayer_single_pass, running_batch):
        seen.append([r.rid for r in h.waiting_queue])
        return None, running_batch

    h._get_new_batch_prefill_raw = raw
    plan = h.get_new_batch_prefill(running_batch=h.running_batch)
    assert seen == [[]] and plan.batch_to_run is None
    assert [r.rid for r in h.waiting_queue] == ["t1", "i1"]

    def boom(**kw):
        raise RuntimeError("admission refused")

    h._get_new_batch_prefill_raw = boom
    with pytest.raises(RuntimeError):
        h.get_new_batch_prefill(running_batch=h.running_batch)
    assert [r.rid for r in h.waiting_queue] == ["t1", "i1"]


def test_an_unarmed_scheduler_admits_exactly_as_before(stage_calls):
    calls = stage_calls
    h = _funnel([_req("t1"), _req("i1", [_Item()])], idle=False)
    del h._weg2_vision_rank_stage
    seen = []
    h._get_new_batch_prefill_raw = lambda prefill_delayer_single_pass, running_batch: (
        seen.append([r.rid for r in h.waiting_queue]) or (None, running_batch))
    h.get_new_batch_prefill(running_batch=h.running_batch)
    assert seen == [["t1", "i1"]] and calls == []


def test_the_abort_echo_carries_the_origin_reason_and_only_that():
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.scheduler import Scheduler

    sent = []

    def sched():
        return types.SimpleNamespace(
            chunked_req=None,
            waiting_queue=[types.SimpleNamespace(rid="r1", mamba_pool_idx=None)],
            enable_hicache_storage=False,
            ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
                send_output=lambda out, req: sent.append(out))),
            disaggregation_mode=DisaggregationMode.NULL,
            _weg2_abort_dormant_hold=lambda recv: None,
            grammar_manager=types.SimpleNamespace(abort_requests=lambda recv: None),
            ps=types.SimpleNamespace(pp_size=1),
            running_batch=types.SimpleNamespace(reqs=[]),
            last_batch=None,
            kv_session_offload=None,
        )

    reason = {"type": "abort", "status_code": 503, "message": "W105 Weg2VisionNoRoom: x"}
    Scheduler._abort_request_now(sched(), AbortReq(rid="r1", finished_reason=reason))
    assert sent[-1].rid == "r1" and sent[-1].finished_reason == reason
    Scheduler._abort_request_now(sched(), AbortReq(rid="r1"))
    assert sent[-1].rid == "r1" and sent[-1].finished_reason is None  # the tokenizer's own


def test_the_receiver_injects_the_extras_on_the_origin_only():
    from sglang.srt.managers.scheduler_components import request_receiver as rr

    field = rr.SchedulerRequestReceiver.__dataclass_fields__["origin_extra_reqs_hook"]
    assert field.default is None
    tree = ast.parse(textwrap.dedent(inspect.getsource(rr.SchedulerRequestReceiver.recv_requests)))
    guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and "origin_extra_reqs_hook()" in ast.unparse(node)
        and "origin_extra_reqs_hook()" not in ast.unparse(node.test)
    ]
    assert len(guards) == 1 and "is_request_origin" in ast.unparse(guards[0].test)
    body = ast.unparse(tree)
    assert body.index("origin_extra_reqs_hook()") < body.index("_broadcast_reqs_across_ranks(")


def test_the_scheduler_wires_the_hook_only_when_the_stage_is_armed():
    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler.init_request_receiver)
    assert "origin_extra_reqs_hook=_vision_origin_aborts" in src
    assert "_vision_origin_aborts = None" in src
    assert 'os.environ.get("SGLANG_WEG2_VISION", "").strip() == "transient"' in src


