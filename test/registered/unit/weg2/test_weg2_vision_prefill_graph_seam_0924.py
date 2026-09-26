"""The in-rank vision stage MEETS the P prefill graph (27B line, merge 24.09.).

The vision line (V1-V3b) was built on b1f9b553af; the boot tree it lands on
(682cae209e) carries H's full prefill CUDA graph on group P (--p-prefill-graph).
Two sides of one seam, asked the same question here with REAL objects on both
sides -- H's own test pins the eligibility rule with a ``lambda: True`` stub,
the vision tests pin the attach with fakes, and neither asks whether the
stage's OUTPUT is what the rule refuses. Hermetic, CPU. Pinned:

  * a request the stage has staged (``attach_precomputed_embeddings``: the
    embeddings on the item, the pixels dropped) still counts as an IMAGE for
    ``ForwardBatch.contains_mm_inputs`` -> the full graph refuses it BY NAME
    (``mm_inputs``) and the chunk runs eager, also on a body captured on
    mrope positions with the positions present (a multimodal P under
    ``transient`` captures that way);
  * a PP follower's copy of the same request (pixels, no embeddings -- the
    stage runs on PP0 only) is refused by the same name;
  * the text control on the same multimodal-P capture replays the graph, and
    after staging the pass sees nothing left to stage;
  * the capture precedes the stage's arming: ``init_model_worker`` captures
    (``init_all_cuda_graphs``) and runs before ``init_request_receiver``,
    which is where ``arm_rank_stage`` is called.
"""

import ast
import inspect
import textwrap
import types

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.runner import prefill_cuda_graph_runner as pcgr
from sglang.srt.planner.vision_stage_load import attach_precomputed_embeddings
from sglang.srt.weg2 import vision_rank_runner as vrr

BUCKET = 512
WIDTH = 5120  # out_hidden_size of the 27B tower, deepstack_visual_indexes = []


def _image_item():
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.randn(16, 8),
        offsets=[(2, 5)],
    )


def _extend_batch(n, mm_inputs, *, mrope=True):
    ext = torch.full((1,), n, dtype=torch.int64)
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.arange(n, dtype=torch.int64),
        req_pool_indices=torch.ones((1,), dtype=torch.int64),
        seq_lens=ext.clone(),
        out_cache_loc=torch.arange(n, dtype=torch.int64) + 100,
        seq_lens_sum=n,
        orig_seq_lens=ext.clone(),
        seq_lens_cpu=ext.clone(),
        positions=torch.arange(n, dtype=torch.int64),
        extend_num_tokens=n,
        extend_seq_lens=ext.clone(),
        extend_prefix_lens=torch.zeros((1,), dtype=torch.int64),
        extend_start_loc=torch.zeros((1,), dtype=torch.int64),
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[n],
        extend_logprob_start_lens_cpu=[n],
        global_forward_mode=ForwardMode.EXTEND,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        mm_inputs=mm_inputs,
    )
    if mrope:
        fb.mrope_positions = torch.zeros((3, n), dtype=torch.int64)
    return fb


def _full_runner():
    """The eligibility half of a full-backend runner, captured on mrope
    positions (a multimodal P: --weg2-vision transient keeps
    model_config.is_multimodal on, language_model_only only drops the tower)."""
    runner = object.__new__(pcgr.PrefillCudaGraphRunner)
    runner.capture_num_tokens = [BUCKET]
    runner.max_num_tokens = BUCKET
    runner.capture_hidden_mode = CaptureHiddenMode.FULL
    runner.prefill_backend_name = "full"
    runner._is_full_backend = True
    runner._capture_req_slots = 1
    runner.__dict__["_captured_mrope"] = True
    return runner


def test_a_staged_image_request_runs_eager_by_name():
    item = _image_item()
    mm = MultimodalInputs(mm_items=[item])
    req = types.SimpleNamespace(rid="img", multimodal_inputs=mm)
    assert vrr.unstaged_items(req) == [item]

    # THE STAGE'S OWN ATTACH (vision_rank_runner -> planner.vision_stage_load)
    rows = attach_precomputed_embeddings(
        [item], [torch.zeros(3, WIDTH, dtype=torch.bfloat16)], expected_width=WIDTH)
    assert rows == 3
    assert item.feature is None and item.precomputed_embeddings is not None
    assert item.is_image()  # the modality survives the attach
    assert vrr.unstaged_items(req) == []  # the pass does not stage it twice

    fb = _extend_batch(9, [mm])
    assert fb.contains_mm_inputs()  # the real method, no stub
    runner = _full_runner()
    assert runner._full_graph_ineligible_reason(fb) == "mm_inputs"
    assert runner.can_run_graph(fb) is False
    assert runner.__dict__["_eager_reasons"] == {"mm_inputs": 1}


def test_a_pp_follower_copy_with_pixels_only_is_refused_by_the_same_name():
    # PP1/PP2 hold the relayed request: pixels on the item, no embeddings.
    mm = MultimodalInputs(mm_items=[_image_item()])
    fb = _extend_batch(9, [mm])
    assert _full_runner()._full_graph_ineligible_reason(fb) == "mm_inputs"


def test_the_text_control_replays_the_multimodal_p_graph():
    runner = _full_runner()
    fb = _extend_batch(9, [None])
    assert not fb.contains_mm_inputs()
    assert runner._full_graph_ineligible_reason(fb) is None
    assert runner.can_run_graph(fb) is True
    # and without the positions the capture was made on: named, not replayed
    assert runner._full_graph_ineligible_reason(
        _extend_batch(9, [None], mrope=False)) == "mrope_missing"


def _calls_in(fn):
    """The call names in ``fn``'s body in SOURCE order (ast.walk alone is
    breadth-first, which would reorder calls at different depths)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    calls = sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.Call)),
        key=lambda n: (n.lineno, n.col_offset),
    )
    return [
        n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
        for n in calls
    ]


def test_the_capture_precedes_the_arming_of_the_stage():
    from sglang.srt.managers.scheduler import Scheduler

    init_calls = [
        c for c in _calls_in(Scheduler.__init__)
        if c in ("init_model_worker", "init_request_receiver")
    ]
    assert init_calls[:2] == ["init_model_worker", "init_request_receiver"]
    assert "init_all_cuda_graphs" in _calls_in(Scheduler.init_model_worker)
    assert "arm_rank_stage" in _calls_in(Scheduler.init_request_receiver)
