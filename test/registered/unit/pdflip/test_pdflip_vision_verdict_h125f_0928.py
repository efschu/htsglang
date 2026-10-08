"""H125f: the vision verdict rides the request chain (vision_verdict).

Hermetic, CPU. PP0's vision pass and a follower's admission view side by side,
on a 3-stage P group whose followers plan for themselves ('#631 ROW AUTHORITY
DISABLED' -- the NF P group).

Pinned:
  * a REFUSED image (synchronous stage) is admissible on NO stage in the pass
    that refuses it -- before the fix PP0 held it while the followers admitted
    it (the #973 split of rc12z30c 20:50:53Z, same class in the sync path);
  * a STAGED image is held on every stage for exactly one pass and released on
    every stage in the pass that dispatches PP0's verdict;
  * the verdict leaves PP0 on its next intake (the origin hook), ahead of the
    refusal aborts;
  * a text request is never held; a single stage keeps the same-pass stage;
  * the scheduler wires the follower half and dispatches the verdict.
"""

import inspect
import types

import pytest
import torch

from flliper.srt.pdflip import vision_rank_runner as vrr


class _Item:
    def __init__(self, n=4):
        self.feature = torch.randn(n, 8)
        self.precomputed_embeddings = None
        self.modality = "image"


def _req(rid, items=()):
    return types.SimpleNamespace(
        rid=rid, multimodal_inputs=types.SimpleNamespace(mm_items=list(items)))


def _queue():
    return [_req("t1"), _req("img", [_Item()]), _req("t2")]


def _pp0(queue, pp_size=3):
    s = types.SimpleNamespace(
        waiting_queue=list(queue),
        pdflip_dormant=False,
        _pdflip_vision_refused=set(),
        _pdflip_vision_arm_refusal="",
        _pdflip_vision_origin_aborts=[],
        _pdflip_vision_runs=0,
        running_batch=types.SimpleNamespace(is_empty=lambda: True),
        chunked_req=None,
        server_args=types.SimpleNamespace(model_path="/m"),
        model_config=types.SimpleNamespace(hf_config=None),
        ps=types.SimpleNamespace(pp_size=pp_size, pp_rank=0),
        pp_flip_counters=None,
    )
    try:
        from flliper.srt.pdflip import vision_verdict as vv

        vv.arm(s)
    except ImportError:  # the base: no verdict gate exists
        pass
    return s


def _follower(queue, pp_rank=1):
    """A follower's queue: the SAME request objects' twins (the chain carries
    mm_inputs to every stage, handle_generate_request builds them alike)."""
    f = types.SimpleNamespace(
        waiting_queue=[_req(r.rid, [_Item()] if r.multimodal_inputs.mm_items else [])
                       for r in queue],
        pdflip_dormant=False,
        ps=types.SimpleNamespace(pp_size=3, pp_rank=pp_rank),
    )
    try:
        from flliper.srt.pdflip import vision_verdict as vv

        f._pdflip_vision_follower_gate = vv.arm(f)
    except ImportError:
        f._pdflip_vision_follower_gate = False
    return f


def _pp0_admissible(s):
    parked = vrr.vision_rank_pass(s)
    rids = [r.rid for r in s.waiting_queue]
    vrr.vision_unpark(s, parked)
    return rids


def _follower_admissible(f):
    """What the follower's admission sees in this pass: its own gate if the
    tree has one, else the whole queue (the base: followers run no vision
    pass and, planning for themselves, admit what they hold)."""
    if not getattr(f, "_pdflip_vision_follower_gate", False):
        return [r.rid for r in f.waiting_queue]
    from flliper.srt.pdflip import vision_verdict as vv

    parked = vv.follower_pass(f)
    rids = [r.rid for r in f.waiting_queue]
    vrr.vision_unpark(f, parked)
    return rids


def _dispatch(sched, wire):
    """Every stage dispatches the identical chain list before it plans."""
    from flliper.srt.pdflip import vision_verdict as vv

    for obj in wire:
        if isinstance(obj, vv.PdFlipVisionVerdict):
            vv.absorb(sched, obj)
        else:  # an AbortReq: the request leaves every stage's queue
            sched.waiting_queue = [r for r in sched.waiting_queue if r.rid != obj.rid]


@pytest.fixture
def stage(monkeypatch):
    verdict = {"ok": True}

    def fake(scheduler, reqs, **kw):
        if verdict["ok"]:
            for r in reqs:
                for it in vrr.unstaged_items(r):
                    it.precomputed_embeddings, it.feature = torch.zeros(1, 6), None
            return vrr.StageOutcome()
        return vrr.StageOutcome(ok=False, code=vrr.W_NO_ROOM, detail="reserve: tail busy")

    monkeypatch.setattr(vrr, "run_rank_stage", fake)
    monkeypatch.setattr(vrr, "_rank_device", lambda: torch.device("cpu"))
    monkeypatch.setenv(vrr.VISION_ASYNC_ENV, "0")
    return verdict


def test_a_refused_image_is_admissible_on_no_stage_in_the_refusing_pass(stage):
    """rc12z30c class, synchronous path: PP0 refuses and holds the image until
    its abort lands; the followers never ran the vision pass and admitted it
    in the same pass -> PP1 waits for a frame PP0 does not owe -> #973."""
    stage["ok"] = False
    q = _queue()
    s, f1, f2 = _pp0(q), _follower(q, 1), _follower(q, 2)
    a0 = _pp0_admissible(s)
    assert "img" not in a0
    assert _follower_admissible(f1) == a0 == _follower_admissible(f2)


def test_a_staged_image_is_released_on_every_stage_in_the_same_pass(stage):
    q = _queue()
    s, f1, f2 = _pp0(q), _follower(q, 1), _follower(q, 2)
    # pass N: staged on PP0, held on every stage (no verdict has travelled yet)
    a0 = _pp0_admissible(s)
    assert a0 == ["t1", "t2"] == _follower_admissible(f1) == _follower_admissible(f2)
    # pass N+1: PP0's intake puts the verdict on the chain, every stage
    # dispatches it before it plans
    wire = vrr.take_origin_aborts(s)
    assert [type(o).__name__ for o in wire] == ["PdFlipVisionVerdict"] and wire[0].rid == "img"
    for sched in (s, f1, f2):
        _dispatch(sched, wire)
    a0 = _pp0_admissible(s)
    assert a0 == ["t1", "img", "t2"] == _follower_admissible(f1) == _follower_admissible(f2)
    assert vrr.take_origin_aborts(s) == []  # one verdict per request, never twice


def test_a_refusal_travels_as_the_abort_and_no_verdict(stage):
    stage["ok"] = False
    q = _queue()
    s, f1 = _pp0(q), _follower(q, 1)
    _pp0_admissible(s)
    wire = vrr.take_origin_aborts(s)
    assert [type(o).__name__ for o in wire] == ["AbortReq"] and wire[0].rid == "img"
    for sched in (s, f1):
        _dispatch(sched, wire)
    assert _pp0_admissible(s) == ["t1", "t2"] == _follower_admissible(f1)


def test_text_only_is_never_held(stage):
    q = [_req("t1"), _req("t2")]
    s, f1 = _pp0(q), _follower(q, 1)
    assert _pp0_admissible(s) == ["t1", "t2"] == _follower_admissible(f1)
    assert vrr.take_origin_aborts(s) == []


def test_a_single_stage_keeps_the_same_pass_stage(stage):
    s = _pp0(_queue(), pp_size=1)
    assert _pp0_admissible(s) == ["t1", "img", "t2"]
    assert vrr.take_origin_aborts(s) == []


def test_the_scheduler_wires_the_follower_half_and_the_dispatch():
    from flliper.srt.managers import scheduler as sm

    src = inspect.getsource(sm.Scheduler.get_new_batch_prefill)
    assert "follower_pass" in src and "_pdflip_vision_follower_gate" in src
    assert "PdFlipVisionVerdict" in inspect.getsource(sm.Scheduler.init_request_dispatcher)
