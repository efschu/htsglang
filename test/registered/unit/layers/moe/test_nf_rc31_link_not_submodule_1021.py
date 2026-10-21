"""NF-RC31-TOD (21.10.): a next-block link is a plain reference, never a registered submodule.

rc31 (boot ``boot_weg2_..._c6e3d38f55_1021_200429``) died on its FIRST flip: EM-PREFETCH (a0056194c2, default ON)
linked every MoE block to the next one with ``blk.pool_em_prefetch_next = blocks[j]``. ``nn.Module.__setattr__``
registers a Module value as a child, so layer L+1's block became a child of layer L's, and the deduplicated
``named_modules()`` / ``named_parameters()`` / ``named_buffers()`` named every linked block under the chain path
``model.layers.<first>.mlp.pool_em_prefetch_next.pool_em_prefetch_next...``. The first layer of the chain differs
per rank (P under PP holds a slice, D under TP all layers), so the weight exchange -- which pairs pieces across the
groups BY NAME -- found 304 expert pieces (8.86 GB) without a counterpart: W74 at the join prewarm, then W106 on D
(release) and W4 on P (resume), PP2 rank death, front 30032 refused.

What must hold (on real ``nn.Module`` blocks, not SimpleNamespace -- the review harness used SimpleNamespace,
which has no submodule registry, and so never saw it):

* after every link chain (EM-PREFETCH, graph pool prefetch, WP8 lookahead) every parameter / buffer / module is
  named by its OWN layer path, and the link attribute appears in no name;
* a PP slice and the full TP stack name the same layer's pieces identically (the exchange's counterpart condition);
* the link still navigates (``blk.<attr> is blocks[j]``) and is not in ``_modules``;
* ``state_dict`` / ``named_parameters(remove_duplicate=False)`` carry no chain-path duplicates.
"""

import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from flliper.srt.layers.moe import expert_offload as eo  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)

_ATTRS = ("pool_em_prefetch_next", "pool_prefetch_next", "lookahead_next")


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.pdflip_experts_w13_weight_packed = nn.Parameter(torch.zeros(4, 2), requires_grad=False)
        self.register_buffer("w13_scale", torch.zeros(4))


class _Blk(nn.Module):
    """The attributes the link functions check, set in ``__init__`` as Qwen2MoeSparseMoeBlock does."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(2, 4, bias=False)
        self.experts = _Experts()
        self.topk = SimpleNamespace(topk_config=None)
        self.lookahead_next = None
        self.pool_prefetch_next = None
        self.pool_em_prefetch_next = None


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Blk()


def _model(n: int, start: int = 0, end: int = None):
    """``model.layers`` like make_layers: the global index list, PP-missing slots outside [start, end)."""
    end = n if end is None else end
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([_Layer() if start <= i < end else nn.Identity() for i in range(n)])
    return root


def _link_all(root):
    from flliper.srt.models import qwen2_moe as q

    layers = root.model.layers
    return {
        "pool_em_prefetch_next": q.link_moe_pool_em_prefetch(layers),
        "pool_prefetch_next": q.link_moe_pool_prefetch(layers),
        "lookahead_next": q.link_moe_lookahead(layers, distance=1),
    }


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    eo.pool_em_prefetch_reset_for_tests()
    monkeypatch.setattr(eo, "_POOL_PREFETCH", True)  # the graph pool chain, default off: armed to cover it
    yield
    eo.pool_em_prefetch_reset_for_tests()


def _names(root):
    return (
        [n for n, _ in root.named_parameters()]
        + [n for n, _ in root.named_buffers()]
        + [n for n, _ in root.named_modules()]
    )


def test_every_piece_is_named_by_its_own_layer_after_linking():
    root = _model(6)
    links = _link_all(root)
    assert links == {"pool_em_prefetch_next": 5, "pool_prefetch_next": 5, "lookahead_next": 5}
    names = _names(root)
    leaked = [n for n in names if any(a in n for a in _ATTRS)]
    assert leaked == [], leaked[:3]
    params = {n for n, _ in root.named_parameters()}
    for i in range(6):
        assert f"model.layers.{i}.mlp.experts.pdflip_experts_w13_weight_packed" in params
        assert f"model.layers.{i}.mlp.gate.weight" in params


def test_pp_slice_and_tp_stack_name_the_same_layer_alike():
    """The exchange's counterpart condition: D (TP, all layers) and P rank 1 (PP slice 2..4) name layer 3 alike."""
    d = _model(6)
    p1 = _model(6, start=2, end=5)
    _link_all(d)
    _link_all(p1)
    d_names = {n for n, _ in d.named_parameters()} | {n for n, _ in d.named_buffers()}
    p_names = {n for n, _ in p1.named_parameters()} | {n for n, _ in p1.named_buffers()}
    assert p_names, "the slice holds pieces"
    assert p_names <= d_names, sorted(p_names - d_names)[:3]


def test_link_navigates_and_is_not_a_child():
    root = _model(4)
    _link_all(root)
    blocks = [layer.mlp for layer in root.model.layers]
    for i, blk in enumerate(blocks):
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        for a in _ATTRS:
            assert getattr(blk, a) is nxt
            assert a not in blk._modules
    assert blocks[-1].pool_em_prefetch_next is None


def test_no_chain_path_duplicates_in_state_dict_or_loader_view():
    root = _model(5)
    plain = _model(5)
    _link_all(root)
    assert list(root.state_dict().keys()) == list(plain.state_dict().keys())
    # the loaders build params_dict with remove_duplicate=False: no chain-path aliases may appear there either
    assert [n for n, _ in root.named_parameters(remove_duplicate=False)] == [
        n for n, _ in plain.named_parameters(remove_duplicate=False)
    ]


def test_switch_off_links_nothing():
    os.environ["FLLIPER_MOE_POOL_EM_PREFETCH"] = "0"
    try:
        eo.pool_em_prefetch_reset_for_tests()
        root = _model(3)
        from flliper.srt.models import qwen2_moe as q

        assert q.link_moe_pool_em_prefetch(root.model.layers) == 0
        assert all(layer.mlp.pool_em_prefetch_next is None for layer in root.model.layers)
    finally:
        os.environ.pop("FLLIPER_MOE_POOL_EM_PREFETCH", None)
        eo.pool_em_prefetch_reset_for_tests()
