"""Vision on group D (no tower): the two V2 (dfce479f08) failures.

1. DEATH 11:20:26 (weg2-14-72, image at the front, 5837 tokens): P's tail was
   adopted as E2 (no target forward), the MTP draft extend then reached
   ``qwen4_exp_mtp._prepare_input_embeds`` with ``contains_mm_inputs()`` and
   no ``mm_input_embeds`` (only a target forward over the image produces them,
   and D has no tower) -> ``assert input_embeds is not None``, D dead.
2. W123 on every short image prompt (weg2-2-14, 1055 tokens, image 4..1027):
   the D guard priced the covered prefix at the page anchor (1024) BEFORE the
   admission consulted the adopted tail [1024, 1055) and refused an image the
   tail covers. All three tail parts had arrived (TAIL-CONSUMED removed 2+2+2
   = 6 files = 3 parts x json/pt); the READY line was never printed because
   the guard refused first.

Hermetic: the real verdict method, the real tail_adopt entry/peek and the
real MTP embed method on stubs."""
from __future__ import annotations

import ast
import os
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.weg2 import tail_adopt as ta  # noqa: E402
from sglang.srt.weg2 import tail_handoff as th  # noqa: E402
from sglang.srt.weg2 import vision_d_guard as vdg  # noqa: E402

N = 1055
PAGE = 1024
CUT = 1054


def _req(rid, image=(4, 1027), logprob=False):
    ids = list(range(100, 100 + N))
    mm = types.SimpleNamespace(mm_items=[types.SimpleNamespace(offsets=[image])])
    return types.SimpleNamespace(
        rid=rid, origin_input_ids=ids, full_untruncated_fill_ids=ids, extra_key=None,
        multimodal_inputs=mm, prefix_indices=torch.zeros(PAGE, dtype=torch.int64),
        return_logprob=logprob, return_hidden_states=False, grammar=None,
        sampling_params=types.SimpleNamespace(frequency_penalty=0.0, presence_penalty=0.0,
                                              repetition_penalty=1.0, min_new_tokens=0),
    )


def _agree(req, *, skip=True, e1=False, agreed=True):
    ids = req.origin_input_ids
    spec = th.TailSpec(rid=req.rid, n_tokens=N, page_prefix=PAGE, cut=CUT, key=th.tail_key(ids, CUT, None))
    end = types.SimpleNamespace(key=th.tail_key(ids, N, None))
    st = ta.Staged(spec=spec, headers=[types.SimpleNamespace(end=end)], verdict="ready", e1=e1)
    ta._AGREED[req.rid] = ta.Agreed(staged=st, agreed=agreed, skip=skip)


@pytest.fixture(autouse=True)
def _adopt_on(monkeypatch):
    monkeypatch.setattr(ta, "adopt_enabled", lambda: True)
    ta._AGREED.clear()
    yield
    ta._AGREED.clear()


def _sched():
    s = types.SimpleNamespace()
    s._weg2_vision_d_covered = lambda req, head_inputs=None: PAGE
    return s


def test_metal_shape_image_in_the_adopted_tail_is_admitted():
    """RED on dfce479f08/0bca87dadd: 'refuse' (W123) with the tail agreed."""
    req = _req("weg2-2-14")
    _agree(req, skip=True)
    assert Scheduler._weg2_vision_d_verdict(_sched(), req, None) == vdg.ADMIT


def test_end_only_tail_in_a_non_empty_batch_waits_instead_of_refusing():
    req = _req("weg2-2-14b")
    _agree(req, skip=True, e1=False)
    assert Scheduler._weg2_vision_d_verdict(_sched(), req, None, batch_empty=False) == vdg.DEFER


def test_e1_tail_covers_an_image_ending_before_c():
    req = _req("weg2-e1", image=(4, 1040))
    _agree(req, skip=False, e1=True)
    assert Scheduler._weg2_vision_d_verdict(_sched(), req, None, batch_empty=False) == vdg.ADMIT


def test_no_agreed_tail_is_still_refused_by_name():
    req = _req("weg2-none")
    assert Scheduler._weg2_vision_d_verdict(_sched(), req, None) == vdg.REFUSE


def test_image_ending_on_the_page_boundary_needs_no_tail():
    req = _req("weg2-edge", image=(4, PAGE - 1))
    assert Scheduler._weg2_vision_d_verdict(_sched(), req, None) == vdg.ADMIT


def test_a_tail_the_admission_cannot_take_is_refused():
    """logprob requests refuse the E2 skip; END-only parts have no E1 -> the
    target would compute the image: refused, never admitted."""
    req = _req("weg2-lp", logprob=True)
    _agree(req, skip=True, e1=False)
    assert Scheduler._weg2_vision_d_verdict(_sched(), req, None) == vdg.REFUSE


@pytest.mark.parametrize("skip,e1,batch_empty", [(True, False, True), (True, True, False),
                                                 (False, True, True), (True, False, False)])
def test_peek_agrees_with_plan_adopt(skip, e1, batch_empty):
    """The guard reads what the admission takes: peek's target start ==
    the start plan_adopt yields (skip -> N, E1 -> c, none -> page anchor)."""
    req = _req(f"weg2-peek-{skip}-{e1}-{batch_empty}")
    _agree(req, skip=skip, e1=e1)
    peek, _ = ta.peek_target_start(req, PAGE, batch_empty=batch_empty)
    taken = ta.plan_adopt(req, PAGE, batch_empty=batch_empty)
    start = PAGE if taken is None else (N if taken.skip else taken.resume_at)
    assert (PAGE if peek is None else peek) == start


# -- the draft death -------------------------------------------------------------
class _Mode:
    def is_extend(self):
        return True

    def is_draft_extend_v2(self):
        return False


def test_mtp_draft_extend_without_mm_embeds_embeds_the_ids():
    """RED on 0bca87dadd: AssertionError in _prepare_input_embeds."""
    from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

    emb = torch.nn.Embedding(2000, 8)
    stub = types.SimpleNamespace(model=types.SimpleNamespace(embed_tokens=emb))
    fb = types.SimpleNamespace(mm_input_embeds=None, forward_mode=_Mode(),
                               contains_mm_inputs=lambda: True, batch_size=1)
    ids = torch.tensor([5, 6, 7])
    out = Qwen4ExpForCausalLMMTP._prepare_input_embeds(stub, ids, fb, None)
    assert torch.equal(out, emb(ids))


def test_qwen3_5_mtp_has_no_mm_embeds_assert_left():
    """Same branch in the Qwen3.5 MTP head (27B draft family)."""
    import sglang.srt.models.qwen3_5_mtp as m

    tree = ast.parse(Path(m.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            src = ast.unparse(node.test)
            assert src != "input_embeds is not None", "assert on mm_input_embeds still present"


def test_an_image_request_is_never_resumed_via_p(monkeypatch, tmp_path):
    """ROS (7cac9f5372) holds every streamed X refusal and resumes a committed
    one through a P-only leg that carries input_ids alone -- an image request
    would reach P as its placeholder ids. It keeps the named W50 instead."""
    import types

    from sglang.srt.weg2 import resume_via_p as rvp

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    monkeypatch.delenv(rvp.ENV_OPEN_STREAM, raising=False)

    def _req(mm, out):
        return types.SimpleNamespace(rid="weg2-2-14", stream=True, output_ids=list(range(out)),
                                     multimodal_inputs=mm)

    assert rvp.eligible(_req(None, 0)) and rvp.eligible(_req(None, 5))
    assert not rvp.eligible(_req(object(), 0)), "open image stream: W50, never ids-only on P"
    assert not rvp.eligible(_req(object(), 5)), "image stream with output: W50 as well"
