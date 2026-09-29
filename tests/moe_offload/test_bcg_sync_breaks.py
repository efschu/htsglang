"""FLLIPER_BCG_SYNC_BREAKS (19.09. diagnosis switch): off by default, on by name, read once."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)


def _reset():
    bcg._SYNC_BREAKS = None


def test_default_is_off(monkeypatch):
    monkeypatch.delenv("FLLIPER_BCG_SYNC_BREAKS", raising=False)
    _reset()
    assert bcg._sync_breaks() is False


def test_on_by_name_and_read_once(monkeypatch):
    monkeypatch.setenv("FLLIPER_BCG_SYNC_BREAKS", "1")
    _reset()
    assert bcg._sync_breaks() is True
    monkeypatch.setenv("FLLIPER_BCG_SYNC_BREAKS", "0")
    assert bcg._sync_breaks() is True  # cached: the switch is a boot property
    _reset()
    assert bcg._sync_breaks() is False
