"""Task #46 (fn8ag2, 20.09.): ``language_model_only`` (hf-config override)
must keep the Qwen3-VL / Qwen4-Exp vision tower from being BUILT, not only
from being loaded -- fn8ag2's census still carried ``visual`` 0.55/0.20/0.20
GiB of never-loaded parameters per rank."""

import types

from sglang.srt.models import qwen3_vl as qv


def test_language_model_only_forces_the_tower_off(monkeypatch):
    monkeypatch.setattr(qv, "get_server_args", lambda: types.SimpleNamespace(enable_multimodal=None))
    assert qv.vision_tower_forced_off() is False
    assert qv.vision_tower_forced_off(types.SimpleNamespace()) is False
    assert qv.vision_tower_forced_off(types.SimpleNamespace(language_model_only=False)) is False
    assert qv.vision_tower_forced_off(types.SimpleNamespace(language_model_only=True)) is True


def test_no_enable_multimodal_still_forces_the_tower_off(monkeypatch):
    monkeypatch.setattr(qv, "get_server_args", lambda: types.SimpleNamespace(enable_multimodal=False))
    assert qv.vision_tower_forced_off() is True
    assert qv.vision_tower_forced_off(types.SimpleNamespace(language_model_only=False)) is True


def test_the_constructor_asks_with_the_config():
    import inspect

    src = inspect.getsource(qv)
    assert "if vision_tower_forced_off(config):" in src
