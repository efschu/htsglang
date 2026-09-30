"""#112/4: das Marlin-Workspace gehoert der KARTE, nicht dem Tensor.

fnFL2w60, TP1, nachdem P schon stand:

    marlin_make_workspace(layer.w13_weight_packed.device, 4)
    ValueError: Expected a cuda device, but got: cpu

Bei Residenz-Fraction < 1.0 baut `create_weights` JEDEN expert-major
Tensor auf dem HOST; unter der Erstboot-Adoption sind es Platzhalter.
Beides sind legitime Zustaende, in denen `.device` `cpu` sagt und der Rang
trotzdem auf einer Karte rechnet.
"""
import inspect
import types

import torch

from flliper.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16_moe as m,
)


def test_host_tensor_still_gives_a_card(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    layer = types.SimpleNamespace(
        w13_weight_packed=torch.empty(0)  # cpu -- der w60-Zustand
    )
    d = m._rang_karte(layer)
    assert d.type == "cuda", "a host tensor must not swallow the card"
    assert d.index == 0


def test_missing_tensor_still_gives_a_card(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    d = m._rang_karte(types.SimpleNamespace())
    assert d.type == "cuda" and d.index == 1


def test_card_tensor_determines_the_card():
    """Liegt das Gewicht auf einer Karte, gilt DIESE -- nicht current_device,
    sonst landet das Workspace unter Multi-Device auf der falschen."""

    class _T:
        device = torch.device("cuda", 2)

    d = m._rang_karte(types.SimpleNamespace(w13_weight_packed=_T()))
    assert d == torch.device("cuda", 2)


def test_caller_asks_the_function():
    # fnFL2x2: der Repack-Koerper steht in `_repack_to_marlin`, der aeussere
    # `process_weights_after_loading` legt ihn nur ausserhalb des Tag-Pools.
    src = inspect.getsource(m.CompressedTensorsWNA16MoE._repack_to_marlin)
    code = "\n".join(z for z in src.split("\n") if not z.lstrip().startswith("#"))
    assert "marlin_make_workspace(_rang_karte(layer), 4)" in code
    assert "marlin_make_workspace(layer.w13_weight_packed.device" not in code, (
        "the caller reads the weight's device again -- the w60 root"
    )
