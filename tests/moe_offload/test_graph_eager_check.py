"""SGLANG_GRAPH_EAGER_CHECK (19.09. diagnosis tap): hook selection and the comparison."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.model_executor import graph_eager_check as gec


class _Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.layers = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"mlp": torch.nn.Linear(4, 4)}) for _ in range(6)]
        )


def test_env_default_off_and_int(monkeypatch):
    monkeypatch.delenv(gec.ENV, raising=False)
    assert gec.steps_from_env() == 0
    monkeypatch.setenv(gec.ENV, "3")
    assert gec.steps_from_env() == 3
    monkeypatch.setenv(gec.ENV, "x")
    assert gec.steps_from_env() == 0


def test_hook_names_cover_layers_children_of_first_layers_and_embed():
    roles = [r for r, _ in gec.hook_names(_Stack())]
    assert "embed" in roles
    assert [r for r in roles if r.startswith("L") and "." not in r] == [f"L{i}" for i in range(6)]
    assert "L0.mlp" in roles and "L3.mlp" in roles and "L4.mlp" not in roles


def test_compare_names_the_first_diverging_role():
    order = ["L0", "L1", "L2"]
    e = {r: [torch.ones(2, 3)] for r in order}
    g = {"L0": [torch.ones(2, 3)], "L1": [torch.ones(2, 3) * 1.5], "L2": [torch.ones(2, 3) * 2]}
    lines, first = gec.compare(g, e, order)
    assert first == "L1" and "DIVERGES" in lines[1] and "DIVERGES" not in lines[0]
    lines, first = gec.compare(e, e, order)
    assert first is None and not any("DIVERGES" in ln for ln in lines)


def test_compare_reports_missing_and_shape_mismatch():
    lines, first = gec.compare({"L0": [torch.ones(2)]}, {"L0": [torch.ones(3)], "L1": [torch.ones(1)]}, ["L0", "L1"])
    assert first == "L0" and "shape" in lines[0] and "n/a" in lines[1]


def test_tensors_of_handles_tuple_and_logits_object():
    class O:
        next_token_logits = torch.zeros(1, 2)

    assert len(gec.tensors_of((torch.zeros(1), 5, torch.zeros(2)))) == 2
    assert len(gec.tensors_of(O())) == 1 and gec.tensors_of(3) == []


def test_state_deltas_expose_an_untouched_graph_state():
    pre = {"conv": [torch.zeros(2, 3)], "temporal": torch.zeros(2, 3), "ngram": torch.zeros(1, 2)}
    post_g = {"conv": [torch.zeros(2, 3)], "temporal": torch.zeros(2, 3), "ngram": torch.zeros(1, 2)}
    post_e = {"conv": [torch.ones(2, 3)], "temporal": torch.ones(2, 3) * 2, "ngram": torch.ones(1, 2)}
    lines = gec.state_deltas(pre, post_g, post_e)
    assert lines[0].startswith("conv: graph-vs-pre=0 eager-vs-pre=1 graph-vs-eager=1")
    assert "temporal: graph-vs-pre=0 eager-vs-pre=2" in lines[1]
    assert len(lines) == 3
