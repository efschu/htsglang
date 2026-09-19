"""fn4l 19.09.: Marlin WNA16 MoE g_idx placeholders (no actorder) are created at
load time and never come from the checkpoint -- the draft completeness check
must not count them as unloaded, but a layer WITH actorder keeps reporting them."""

from types import SimpleNamespace

import pytest
import torch


def _model(actorder):
    moe = torch.nn.Module()
    for n in ("w13_weight_packed", "w13_weight_g_idx", "w2_weight_g_idx",
              "w13_g_idx_sort_indices", "w2_g_idx_sort_indices"):
        moe.register_parameter(n, torch.nn.Parameter(torch.zeros(2), requires_grad=False))
    moe.scheme = SimpleNamespace(actorder=actorder)
    model = torch.nn.Module()
    model.experts = moe
    model.fc = torch.nn.Linear(2, 2, bias=False)
    return model


def test_g_idx_placeholders_without_actorder_are_not_unloaded():
    from sglang.srt.model_loader.weight_utils import (
        load_time_derived_param_names,
        raise_on_unloaded_draft_parameters,
    )

    model = _model(None)
    assert load_time_derived_param_names(model) == {
        "experts.w13_weight_g_idx", "experts.w2_weight_g_idx",
        "experts.w13_g_idx_sort_indices", "experts.w2_g_idx_sort_indices"}
    raise_on_unloaded_draft_parameters(
        model, {"experts.w13_weight_packed", "fc.weight"}, model_path="x")


def test_g_idx_with_actorder_still_counts_as_unloaded():
    from sglang.srt.model_loader.weight_utils import raise_on_unloaded_draft_parameters

    model = _model("group")
    with pytest.raises(ValueError, match="g_idx"):
        raise_on_unloaded_draft_parameters(
            model, {"experts.w13_weight_packed", "fc.weight"}, model_path="x")


def test_a_real_unloaded_weight_is_still_reported():
    from sglang.srt.model_loader.weight_utils import raise_on_unloaded_draft_parameters

    with pytest.raises(ValueError, match="fc.weight"):
        raise_on_unloaded_draft_parameters(
            _model(None), {"experts.w13_weight_packed"}, model_path="x")
