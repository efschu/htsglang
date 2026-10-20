"""W2-APPLY: each EM wave computes ONLY its own lanes -- the upstream EP skip, taken over.

rc27 D TP0 15:36:48Z: MIXED-D T=1922 run_ms 2325, EM-FETCH em_ms 2134.6 (92 %), copy_ms 371.8;
EAGER-DEVICE-EM experts=193 waves=2 at C=105. ``run_pool_em`` ran every wave's apply over all T x K lanes
(unserved lanes on the zero row, weight 0): a full GEMM per wave, also for a wave that serves nothing.

Now a wave hands the runner its served lanes; every other lane carries expert id -1 and the apply runs inside
``filtered_lanes.active()`` -- the form ``fused_marlin_moe`` already has for EP (``moe_align_block_size`` puts -1
lanes into blocks with expert id -1, the Marlin GEMMs skip them with ``is_ep``, the second GEMM's output is zeroed).
Only a scheme that declares ``MOE_FILTERED_LANES`` (compressed-tensors WNA16 Marlin) gets -1 lanes.

What must hold (S51 CPU pool harness; its apply skips -1 lanes like the Marlin EP form):
(1) the output is bit-identical to the zero-row form and to the truth, for 1 / 2 / 3 waves;
(2) the runner sees -1 exactly for the unserved lanes, and only inside ``active()``;
(3) measured on the EM-FETCH line: ``wave_lanes`` drops from T x K per wave to the wave's served lanes --
    an empty wave 2 computes 0 lanes;
(4) no scheme / the Triton scheme keeps the zero-row form;
(5) ``fused_marlin_moe`` takes the EP form (is_ep, output zeroed) inside ``active()`` without an expert map,
    and exactly as before outside it.
"""
import importlib.util
import logging
import os
import re
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.layers.moe import em_fetch_probe as P  # noqa: E402
from flliper.srt.layers.moe import filtered_lanes as FL  # noqa: E402

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "_t_w2_em_probe", os.path.join(_HERE, "test_s78_em_fetch_probe_1019.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
S51 = T.S51
_probe = T._probe  # noqa: F401 -- autouse: fake timing events, CPU allowed


def _filtered(cache):
    cache.layer.scheme = SimpleNamespace(MOE_FILTERED_LANES=True)
    return cache


def _recording_apply(cache, seen):
    inner = S51._apply_over(cache._resident["w13"])

    def _apply(sub):
        seen.append((sub.topk_output.topk_ids.clone(), FL.is_active()))
        return inner(sub)

    return _apply


def _run(cache, routes, seed, seen=None):
    ids = torch.tensor(routes, dtype=torch.int32)
    topk = S51.StandardTopKOutput(topk_weights=S51._weights(routes, seed), topk_ids=ids, router_logits=None)
    apply_fn = _recording_apply(cache, seen) if seen is not None else S51._apply_over(cache._resident["w13"])
    return cache.run_eager_pool(S51.DispatchOutput(torch.zeros(ids.shape[0], S51.W), None, topk), apply_fn)\
        .hidden_states


@pytest.mark.parametrize("routes,scratch,seed", [
    (S51.ROUTES, S51.C, 0), (S51.WIDE, S51.C, 5), (S51.ALL, 20, 13), (S51.ALL, 12, 13), (S51.ALL, 7, 13)])
def test_the_filtered_waves_are_bit_identical_to_the_zero_row_form(monkeypatch, routes, scratch, seed):
    old = S51._offset(S51._pool_cache(monkeypatch, scratch=scratch))
    new = S51._offset(_filtered(S51._pool_cache(monkeypatch, scratch=scratch)))
    a, b = _run(old, routes, seed), _run(new, routes, seed)
    assert torch.equal(a, b)
    assert torch.equal(b[:, 0], S51._truth_off(routes, seed=seed))


def test_the_runner_sees_minus_one_exactly_for_the_unserved_lanes(monkeypatch):
    cache = _filtered(S51._pool_cache(monkeypatch, scratch=12))
    seen = []
    _run(cache, S51.ALL, 13, seen)
    assert len(seen) == 2 and all(flag for _ids, flag in seen)
    lanes = torch.tensor(S51.ALL).reshape(-1)
    valid = (lanes >= 0) & (lanes < S51.E)
    served = [(ids.reshape(-1) >= 0) for ids, _f in seen]
    assert not bool((served[0] & served[1]).any())             # each lane in one wave
    assert torch.equal(served[0] | served[1], valid)            # every valid lane in one, padding in none
    assert FL.is_active() is False                              # the context is left


def _wave_lanes(caplog):
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith(P.MARK)][-1]
    m = re.search(r"wave_served=(\S+) wave_lanes=(\S+)", line)
    assert m, line
    return [[int(x) for x in g.split("/")] for g in m.groups()]


@pytest.mark.parametrize("routes", [S51.ROUTES, S51.WIDE])
def test_measured_the_gemm_lanes_drop_to_the_served_lanes(monkeypatch, caplog, routes):
    n_lanes = len(routes) * len(routes[0])
    out = {}
    for name, mk in (("before", lambda c: c), ("after", _filtered)):
        cache = mk(S51._pool_cache(monkeypatch))
        P.reset_for_tests()
        P.on_forward(1)
        with caplog.at_level(logging.INFO, logger=P.__name__):
            _run(cache, routes, 3)
            P.on_forward(2)
        out[name] = _wave_lanes(caplog)
    (served_b, lanes_b), (served_a, lanes_a) = out["before"], out["after"]
    assert served_b == served_a
    assert lanes_b == [n_lanes, n_lanes]                       # before: every wave, every lane
    assert lanes_a == served_a                                  # after: a wave computes what it serves
    if routes is S51.ROUTES:
        assert lanes_a[1] == 0                                  # the empty wave 2 computes nothing


def test_without_the_capability_the_zero_row_form_stays(monkeypatch):
    from flliper.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
        CompressedTensorsWNA16MoE, CompressedTensorsWNA16TritonMoE)

    assert CompressedTensorsWNA16MoE.MOE_FILTERED_LANES is True
    assert CompressedTensorsWNA16TritonMoE.MOE_FILTERED_LANES is False
    for scheme in (None, SimpleNamespace(), SimpleNamespace(MOE_FILTERED_LANES=False)):
        cache = S51._pool_cache(monkeypatch)
        cache.layer.scheme = scheme
        seen = []
        _run(cache, S51.ROUTES, 0, seen)
        assert all((ids >= 0).all() and not flag for ids, flag in seen)


def test_fused_marlin_moe_takes_the_ep_form_inside_the_context():
    """The op body is a registered CUDA custom op (not drivable on a CPU desk): the decision is ``ep_form``,
    and the body wires it into BOTH GEMMs and the zeroing of the second GEMM's output -- read off the source
    of the op body, the decision itself tested directly."""
    import inspect

    from flliper.srt.layers.moe.fused_moe_triton import fused_marlin_moe as FM

    assert FL.ep_form(None) is False
    assert FL.ep_form(object()) is True
    with FL.active():
        assert FL.ep_form(None) is True
    assert FL.ep_form(None) is False
    src = inspect.getsource(FM)
    body = src[src.index("def fused_marlin_moe("):]
    body = body[: body.index("\ndef ", 10)] if "\ndef " in body[10:] else body
    assert "is_ep = ep_form(expert_map)" in body
    assert body.count("is_ep=is_ep,") == 2
    assert "if is_ep:\n        intermediate_cache3.zero_()" in body
    assert "is_ep=expert_map is not None" not in body


def test_the_49_stage_walk_never_reports_a_minus_one_lane():
    """The #49 stage walk (FLLIPER_MOE_MARLIN_C_SENTINEL=2) scans GEMM1 / ACT / GEMM2 rows for non-finite
    values. In the EP form a -1 lane's GEMM1 / ACT rows are never written: they are excluded; a live lane's
    bad row still counts; outside the EP form nothing changes."""
    import inspect

    from flliper.srt.layers.moe.fused_moe_triton import fused_marlin_moe as FM

    bad = torch.tensor([True, False, True, True])            # (token, k) rows, M=2, topk=2
    ids = torch.tensor([[3, -1], [-1, 5]])
    assert FM._live_lanes_only(bad, ids, True, 2, 2).tolist() == [True, True]   # live lanes 0 and 3
    ids2 = torch.tensor([[-1, 2], [-1, 5]])
    assert FM._live_lanes_only(bad, ids2, True, 2, 2).tolist() == [False, True]  # token 0 bad only on -1
    assert FM._live_lanes_only(bad, ids2, False, 2, 2).tolist() == [True, True]  # not EP: unchanged
    src = inspect.getsource(FM)
    for stage in ("GEMM1", "ACT", "GEMM2"):
        assert '_stage_masks["%s"] = _live_lanes_only(' % stage in src
