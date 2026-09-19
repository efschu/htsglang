"""Task #49: the NaN guard names the first non-finite layer once and counts the rest."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from types import SimpleNamespace

import torch

from sglang.srt.layers import nan_guard as ng


def test_off_costs_nothing_and_on_logs_first_hit_per_layer(monkeypatch, caplog):
    ng._reset_for_tests(False)
    assert ng.check("mlp_out", torch.tensor([float("nan")]), 3) is True
    ng._reset_for_tests(True)
    fb = SimpleNamespace(forward_mode=SimpleNamespace(name="EXTEND"), seq_lens_cpu=torch.tensor([4, 259415]))
    ok = torch.randn(4, 8)
    assert ng.check("mlp_out", ok, 3, fb) is True and ng.hits() == {}
    bad = ok.clone(); bad[1, 2] = float("inf")
    with caplog.at_level("ERROR"):
        assert ng.check("mlp_out", bad, 3, fb) is False
        assert ng.check("mlp_out", bad, 3, fb) is False
    assert ng.hits() == {("mlp_out", 3): 2}
    msgs = [r.getMessage() for r in caplog.records if "nan-guard" in r.getMessage()]
    assert len(msgs) == 1 and "layer 3" in msgs[0] and "259415" in msgs[0] and "EXTEND" in msgs[0]
    ng._reset_for_tests(None)
