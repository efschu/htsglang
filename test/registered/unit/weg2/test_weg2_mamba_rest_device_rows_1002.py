"""DP-NACHLAUF 02.10.: the mamba staging rows load without a host wait.

N5d (0c996cf05c 1002_124821, D->P epoch 25): WEG2-START-LOADING mamba=345 ms
against 9 ms of timed sub-stages, on P's scheduler thread right before the
first prefill forward. The code path: --hicache-io-backend direct + layer_first
keeps the HOST rows on the host, P-NOSYNC (move_indices) keeps the DEVICE rows
on the card, and the staging (non-arena) rows went through
MambaPoolHost.load_to_device_per_layer -> transfer_kv_direct, whose C++ body
calls dst_indices.cpu() -- a host wait on the load stream, which start_loading
fences behind the forward in flight -- once per layer and buffer.

Pinned (red before): with device rows on the card the rows go gather -> pinned
-> non-blocking H2D -> index_copy_ (the base path is never called), the values
land identically; host rows on the host + device rows on the host, a
page-first layout or the switch off -> the base path.
"""
from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import memory_pool_host as mph  # noqa: E402
from sglang.srt.mem_cache.pool_host import arena_mamba_pool as amp  # noqa: E402


class _CardIdx(torch.Tensor):
    """A host tensor that answers is_cuda=True (the P-NOSYNC device rows)."""

    @property
    def is_cuda(self):  # noqa: D401
        return True


def _pool(L=3, rows=8):
    p = amp.ArenaMambaPoolHost.__new__(amp.ArenaMambaPoolHost)
    p.layout = "layer_first"
    p.conv_state_shapes = [(4, 2), (6, 2)]
    p.temporal_buffer = torch.arange(L * rows * 5, dtype=torch.float32).view(L, rows, 5)
    p.conv_buffer = [torch.arange(L * rows * 8, dtype=torch.float32).view(L, rows, 4, 2) + 1000,
                     torch.arange(L * rows * 12, dtype=torch.float32).view(L, rows, 6, 2) + 5000]
    dev = type("D", (), {})()
    dev.mamba_cache = type("M", (), {})()
    dev.mamba_cache.temporal = torch.zeros(L, 16, 5)
    dev.mamba_cache.conv = [torch.zeros(L, 16, 4, 2), torch.zeros(L, 16, 6, 2)]
    return p, dev


@pytest.fixture
def no_pin(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self, raising=False)


def test_switch_default_on():
    assert amp.mamba_rest_device_rows_on({})
    assert not amp.mamba_rest_device_rows_on({amp.ENV_MAMBA_REST_DEVICE_ROWS: "0"})


def test_device_rows_on_the_card_take_no_host_wait(monkeypatch, no_pin):
    monkeypatch.delenv(amp.ENV_MAMBA_REST_DEVICE_ROWS, raising=False)

    def base(*a, **k):
        raise AssertionError("the base path (transfer_kv_direct -> dst_indices.cpu()) was taken")

    monkeypatch.setattr(mph.MambaPoolHost, "load_to_device_per_layer", base)
    p, dev = _pool()
    host = torch.tensor([1, 3, 6], dtype=torch.int64)
    rows = torch.tensor([9, 2, 14], dtype=torch.int64).as_subclass(_CardIdx)
    assert p._load_rest_rows(dev, host, rows, 1, "direct") == "device"
    d = torch.tensor([9, 2, 14])
    assert torch.equal(dev.mamba_cache.temporal[1][d], p.temporal_buffer[1][host])
    assert torch.equal(dev.mamba_cache.conv[0][1][d], p.conv_buffer[0][1][host])
    assert torch.equal(dev.mamba_cache.conv[1][1][d], p.conv_buffer[1][1][host])
    assert float(dev.mamba_cache.temporal[0].abs().sum()) == 0.0   # other layers untouched


@pytest.mark.parametrize("case", ["host-rows", "page-first", "switch-off"])
def test_otherwise_the_base_path(monkeypatch, no_pin, case):
    called = []
    monkeypatch.setattr(mph.MambaPoolHost, "load_to_device_per_layer",
                        lambda self, *a, **k: called.append(a[3]))
    p, dev = _pool()
    host = torch.tensor([1, 2], dtype=torch.int64)
    rows = torch.tensor([3, 4], dtype=torch.int64)
    if case != "host-rows":
        rows = rows.as_subclass(_CardIdx)
    if case == "page-first":
        p.layout = "page_first"
    if case == "switch-off":
        monkeypatch.setenv(amp.ENV_MAMBA_REST_DEVICE_ROWS, "0")
    else:
        monkeypatch.delenv(amp.ENV_MAMBA_REST_DEVICE_ROWS, raising=False)
    assert p._load_rest_rows(dev, host, rows, 2, "direct") == "base"
    assert called == [2]
