"""DFlash-family aux capture across PP stages (distributed/pp_aux_capture).

weg2xsn261 (17.09.2026): the first form shipped every stage's captures to the
last stage as a SEPARATE typed-channel message sent from inside the model
forward, before the stage returned its proxy. On the metal PP0 blocked in
that send (PP2 was in its ordinary proxy receive from PP1, PP1 waited for
PP0's proxy) -- the first P prefill of the DFLASH form after a flip wedged
for 240 s and the watchdog killed the boot. The hermetic channel double was
non-blocking and could not show it.

Now the captures RIDE THE PROXY: each stage adds ``aux_layer_<id>`` entries
to the dict it already hands downstream, forwards what it received, and the
last stage assembles received + own in layer-id order. These tests drive the
three stages hop by hop with plain dicts -- exactly what the proxy channel
carries -- and pin (a) the assembly order, (b) pass-through by a stage that
captures nothing, (c) duplicate refusal, (d) that the module makes NO
cross-stage send at all (the deadlock's mechanism, pinned at the source
since it cannot be executed hermetically).
"""

import os

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.distributed import pp_aux_capture as pac  # noqa: E402
from sglang.srt.distributed.pp_aux_capture import (  # noqa: E402
    AUX_KEY_PREFIX,
    PpAuxCaptureError,
    assemble_aux_on_last_stage,
    carry_aux_forward,
)


def _stages(ownership, world_size, hidden=4):
    """ownership: {stage: [layer ids]} -> (assembled list on the last stage,
    {layer id: tensor}, the proxy dict each stage sent)."""
    tensors = {}
    proxies = {}
    received = None
    out = None
    for stage in range(world_size):
        captured = {}
        for lid in ownership.get(stage, []):
            t = torch.full((3, hidden), float(lid))
            tensors[lid] = t
            captured[lid] = t
        if stage < world_size - 1:
            carry = carry_aux_forward(received=received, captured=captured, stage=stage)
            proxy = {"hidden_states": torch.zeros(3, hidden),
                     "residual": torch.zeros(3, hidden), **carry}
            proxies[stage] = proxy
            received = proxy          # the next stage's pp_proxy_tensors.tensors
        else:
            out = assemble_aux_on_last_stage(received=received, captured=captured, stage=stage)
    return out, tensors, proxies


def test_gapped_ownership_assembles_in_layer_id_order():
    """Group P's cut (39,13,12): PP0 owns captures 6/20/34, PP1 48, PP2 62."""
    out, tensors, proxies = _stages({0: [6, 20, 34], 1: [48], 2: [62]}, 3)
    assert [float(t[0, 0]) for t in out] == [6.0, 20.0, 34.0, 48.0, 62.0]
    for lid, t in zip([6, 20, 34, 48, 62], out):
        assert t is tensors[lid]
    # the carry grows hop by hop and keeps the pipeline's own keys
    assert sorted(k for k in proxies[0] if k.startswith(AUX_KEY_PREFIX)) == [
        "aux_layer_20", "aux_layer_34", "aux_layer_6"]
    assert sorted(k for k in proxies[1] if k.startswith(AUX_KEY_PREFIX)) == [
        "aux_layer_20", "aux_layer_34", "aux_layer_48", "aux_layer_6"]
    assert {"hidden_states", "residual"} <= set(proxies[1])


def test_interleaved_ownership_sorts_by_layer_not_stage():
    out, _t, _p = _stages({0: [20], 1: [6, 34], 2: [62, 48]}, 3)
    assert [float(t[0, 0]) for t in out] == [6.0, 20.0, 34.0, 48.0, 62.0]


def test_a_stage_without_captures_passes_the_carry_through():
    out, _t, proxies = _stages({0: [6, 20], 1: [], 2: [34]}, 3)
    assert [float(t[0, 0]) for t in out] == [6.0, 20.0, 34.0]
    assert sorted(k for k in proxies[1] if k.startswith(AUX_KEY_PREFIX)) == [
        "aux_layer_20", "aux_layer_6"]


def test_last_stage_alone_is_identity():
    out = assemble_aux_on_last_stage(
        received=None, captured={20: torch.ones(2, 2), 6: torch.zeros(2, 2)}, stage=0)
    assert [float(t[0, 0]) for t in out] == [0.0, 1.0]


def test_duplicate_capture_layer_is_refused_on_carry_and_on_assembly():
    with pytest.raises(PpAuxCaptureError):
        carry_aux_forward(received={"aux_layer_6": torch.ones(1)},
                          captured={6: torch.ones(1)}, stage=1)
    with pytest.raises(PpAuxCaptureError):
        assemble_aux_on_last_stage(received={"aux_layer_6": torch.ones(1)},
                                   captured={6: torch.ones(1)}, stage=2)


def test_a_malformed_carry_key_is_refused():
    with pytest.raises(PpAuxCaptureError):
        carry_aux_forward(received={"aux_layer_x": torch.ones(1)}, captured={}, stage=1)


def test_the_module_makes_no_cross_stage_send():
    """THE DEADLOCK'S MECHANISM, pinned at the source: no typed-channel send
    or receive anywhere in the capture path. The carry is the proxy dict the
    stage returns; the pipeline's own send is the only message."""
    src = open(pac.__file__).read()
    body = src.split('"""', 2)[2]          # past the module docstring
    for banned in ("send_typed_tensor_dict", "recv_typed_tensor_dict",
                   "send_tensor_dict", "recv_tensor_dict", "torch.distributed.send",
                   "dist.send", ".send("):
        assert banned not in body, f"cross-stage send reintroduced: {banned}"
    model_src = open(os.path.join(os.path.dirname(pac.__file__), "..", "models", "qwen3_5.py")).read()
    assert "exchange_captured_aux" not in model_src
    assert "**aux_carry" in model_src
