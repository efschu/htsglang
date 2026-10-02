# SPDX-License-Identifier: Apache-2.0
"""VISION-TENSOR-TRUTH (02.10.): the W102 SKIP verdict never truth-tests a tensor.

NF y7m (a3b06c558f) P, first request = an OpenAI-chat image request (64x64 PNG
data URI + text): ``vision_rank_pass -> skip_cached -> stage_skip_verdict``
evaluated ``len(getattr(req, "prefix_indices", None) or ())``. On the metal
``prefix_indices`` is a torch tensor; an empty one raised ``RuntimeError:
Boolean value of Tensor with no values is ambiguous`` on PP0/PP1/PP2 and the
boot died. The fakes of test_weg2_mm_xprice_1002 carried a list, so nothing
red-lit it. These tests carry REAL torch tensors.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.weg2 import vision_d_guard as G  # noqa: E402
from sglang.srt.weg2 import vision_rank_runner as V  # noqa: E402

SPAN = (70000, 71023)   # image_end = 71024


def _req(rid, prefix: torch.Tensor, host=0, offsets=(SPAN,)):
    item = types.SimpleNamespace(offsets=list(offsets), precomputed_embeddings=None,
                                 feature=torch.zeros(3, 64, 64))
    return types.SimpleNamespace(
        rid=rid, multimodal_inputs=types.SimpleNamespace(mm_items=[item]),
        prefix_indices=prefix, host_hit_length=host,
        origin_input_ids=list(range(71100)), output_ids=[])


def _sched(told=None, armed=True):
    return types.SimpleNamespace(_weg2_store_told_armed=armed, _weg2_store_told=told or {})


def _empty():
    return torch.empty((0,), dtype=torch.int64)


def _full(n):
    return torch.arange(n, dtype=torch.int64)


# ---- the crash site: stage_skip_verdict with an empty / non-empty tensor --------------

def test_empty_prefix_tensor_stages_instead_of_raising():
    """The y7m first request: no prefix at all, an empty int64 tensor."""
    r = _req("weg2-0-1", _empty())
    assert V.stage_skip_verdict(_sched({}), r) == (V.STAGE, 0, "local")
    assert V.stage_skip_verdict(_sched(armed=False), r) == (V.STAGE, 0, "local")
    assert V.stage_skip_verdict(_sched({"weg2-0-1": 0}), r) == (V.STAGE, 0, "told")


def test_non_empty_prefix_tensor_counts_its_length():
    covering = _req("weg2-1-5", _full(73088))
    assert V.stage_skip_verdict(_sched(armed=False), covering) == (V.SKIP, 73088, "local")
    assert V.stage_skip_verdict(_sched({}), covering) == (V.WAIT, 73088, "local")
    short = _req("weg2-1-6", _full(3264), host=100)
    assert V.stage_skip_verdict(_sched(armed=False), short) == (V.STAGE, 3364, "local")
    # a one-element tensor is truth-testable but must still be counted, not read as a bool
    one = _req("weg2-1-7", torch.tensor([0], dtype=torch.int64))
    assert V.stage_skip_verdict(_sched(armed=False), one) == (V.STAGE, 1, "local")


def test_host_hit_completes_a_tensor_prefix():
    r = _req("weg2-2-1", _full(70000), host=1024)
    assert V.stage_skip_verdict(_sched(armed=False), r) == (V.SKIP, 71024, "local")


# ---- the whole skip_cached path with tensor-carrying requests -------------------------

def test_skip_cached_mixed_tensor_requests():
    fresh = _req("weg2-0-1", _empty())
    told = _req("weg2-1-5", _empty())                 # P's own match empty, the told covers
    local = _req("weg2-9-9", _full(72000))            # told outstanding, own match covers
    partial = _req("weg2-9-8", _full(70464))
    s = _sched({"weg2-0-1": 3264, "weg2-1-5": 73088, "weg2-9-8": 70464})
    stage, wait = V.skip_cached(s, [fresh, told, local, partial])
    assert stage == [fresh, partial] and wait == [local]
    assert told._weg2_vision_skip and V.unstaged_items(told) == []
    assert V.unstaged_items(fresh) and V.unstaged_items(partial)
    assert s._weg2_vision_skips == 1


def test_skip_cached_without_store_told_on_tensors():
    empty = _req("weg2-0-1", _empty())
    full = _req("weg2-1-5", _full(73088))
    stage, wait = V.skip_cached(_sched(armed=False), [empty, full])
    assert stage == [empty] and wait == [] and full._weg2_vision_skip


def test_admission_belt_on_tensors():
    gone = _req("weg2-1-5", _empty())
    gone._weg2_vision_skip = True
    assert V.skip_still_covered(None, gone) is False and not gone._weg2_vision_skip
    ok = _req("weg2-1-6", _full(73088))
    ok._weg2_vision_skip = True
    assert V.skip_still_covered(None, ok) is True


def test_image_spans_and_end_with_tensor_requests():
    r = _req("weg2-0-1", _empty(), offsets=[(10, 20), SPAN])
    assert G.image_spans(r) == [(10, 20), SPAN] and G.image_end(r) == 71024
    no_mm = types.SimpleNamespace(rid="x", multimodal_inputs=None, prefix_indices=_empty())
    assert G.image_end(no_mm) == 0
    assert V.stage_skip_verdict(_sched({}), no_mm) == (V.STAGE, 0, "none")


def test_no_truth_test_on_prefix_indices_left_in_weg2():
    """Source belt for the class: no ``<prefix_indices getattr> or`` anywhere in weg2."""
    import pathlib
    import re

    root = pathlib.Path(V.__file__).parent
    pat = re.compile(r'getattr\(\s*\w+\s*,\s*"prefix_indices"\s*,\s*None\s*\)\s*or\b'
                     r'|\bprefix_indices\s+or\b|\b(?:if|not)\s+(?:\w+\.)?prefix_indices\s*[:)]')
    hits = [f"{p.name}:{i}" for p in root.rglob("*.py")
            for i, line in enumerate(p.read_text().splitlines(), 1) if pat.search(line)]
    assert hits == [], hits
