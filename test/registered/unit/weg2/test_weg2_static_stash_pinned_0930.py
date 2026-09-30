"""WT (30.09., NF y4f): the static-state stash is ONE host block (pinned when
CUDA is there) and the wake's import copies from it asynchronously.

Measured on D TP0 (y4f dkrnf...dauer09300520): WEG2-WAKE-TAIL-SUB static_import
111.5 ms median over 32 P->D wakes -- rc12g (before 7002b171d5) 11 ms. The
pageable per-buffer stash made every buffer a synchronous pageable H2D on the
wake's critical path.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    _import_static_state,
)
from sglang.srt.weg2 import sleep_staging as ss  # noqa: E402


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("f32", torch.arange(10, dtype=torch.float32))
        self.register_buffer("bf16", torch.linspace(0, 1, 7).to(torch.bfloat16))
        self.register_buffer("mask", torch.tensor([True, False, True]))
        self.register_buffer("idx", torch.arange(5, dtype=torch.int64).view(5, 1))
        self.register_buffer("scalar", torch.tensor(3.5))
        self.register_buffer("empty", torch.empty(0, 4))
        self.inner = torch.nn.Module()
        self.inner.register_buffer("t", torch.ones(3, 4).t())  # non-contiguous source


def test_stash_is_one_block_of_views():
    m = _Model()
    stash = ss.export_static_state_host(m)
    block = stash["block"]
    assert block.dtype == torch.uint8
    for name, view in stash["buffers"]:
        assert view.untyped_storage().data_ptr() == block.untyped_storage().data_ptr(), name
        assert view.storage_offset() * view.element_size() % ss._STASH_ALIGN == 0, name


def test_stash_roundtrip_all_dtypes():
    m = _Model()
    ref = {n: b.clone() for n, b in m.named_buffers()}
    stash = ss.export_static_state_host(m)
    for _, b in m.named_buffers():
        b.zero_() if b.dtype != torch.bool else b.fill_(False)
    _import_static_state(m, stash)
    for n, b in m.named_buffers():
        assert torch.equal(b, ref[n]), n


def test_stash_is_a_copy_not_an_alias():
    m = _Model()
    stash = ss.export_static_state_host(m)
    m.f32.fill_(-1.0)
    got = dict(stash["buffers"])["f32"]
    assert torch.equal(got, torch.arange(10, dtype=torch.float32))


def test_pinned_stash_imports_non_blocking(monkeypatch):
    m = _Model()
    stash = ss.export_static_state_host(m)
    stash["pinned"] = True  # what a CUDA rank gets; the copy mode is the claim
    seen = []
    orig = torch.Tensor.copy_

    def spy(self, src, non_blocking=False):
        seen.append(non_blocking)
        return orig(self, src, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", spy)
    _import_static_state(m, stash)
    assert seen and all(seen)


def test_unpinned_stash_imports_blocking(monkeypatch):
    m = _Model()
    stash = ss.export_static_state_host(m)
    assert stash["pinned"] is False  # no CUDA in this test process
    seen = []
    orig = torch.Tensor.copy_

    def spy(self, src, non_blocking=False):
        seen.append(non_blocking)
        return orig(self, src, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", spy)
    _import_static_state(m, stash)
    assert seen and not any(seen)


def test_legacy_stash_without_pinned_key_still_imports():
    m = _Model()
    legacy = dict(buffers=[(n, b.detach().clone()) for n, b in m.named_buffers()])
    m.f32.zero_()
    _import_static_state(m, legacy)
    assert torch.equal(m.f32, torch.arange(10, dtype=torch.float32))
