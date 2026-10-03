"""Kriech-Sitz 29.09. (z30w-park): SGLANG_OPT_WEG2_DRAFT_PARK_EXACT_PIN.

z30w-park D-TP0: WEG2-DRAFT-PARK bytes=1596748576 (1522.8 MiB) while
torch.cuda host stats held 2048 MiB for it -- torch.empty(pin_memory=True)
rounds every request to the next power of two (CachingHostAllocator,
73b1a4750a). With the switch the image is pinned at its exact size through
expert_offload.pinned_exact_empty; off, the torch.empty form is unchanged.
Hermetic: the two allocators are recorded, no device needed.
"""

import pytest
import torch

import sglang.srt.layers.moe.expert_offload as eo
from sglang.srt.environ import envs
from sglang.srt.weg2 import draft_park as dp

IMAGE = 1596748576  # z30w-park D-TP0, the draft's 40 storages


class _Rec:
    def __init__(self):
        self.exact, self.pinned = [], []


@pytest.fixture
def rec(monkeypatch):
    r = _Rec()
    real_empty = torch.empty

    def fake_exact(shape, dtype):
        r.exact.append((tuple(shape), dtype))
        return real_empty(8, dtype=dtype)

    def fake_empty(*args, **kw):
        if kw.get("pin_memory"):
            r.pinned.append(args[0])
            kw = dict(kw, pin_memory=False)
            return real_empty(8, **{k: v for k, v in kw.items() if k != "pin_memory"})
        return real_empty(*args, **kw)

    monkeypatch.setattr(eo, "pinned_exact_empty", fake_exact)
    monkeypatch.setattr(dp.torch, "empty", fake_empty)
    return r


def test_switch_on_pins_the_exact_size(rec):
    with envs.SGLANG_OPT_WEG2_DRAFT_PARK_EXACT_PIN.override(True):
        dp._host_image(IMAGE, pin=True)
    assert rec.exact == [((IMAGE,), torch.uint8)]
    assert rec.pinned == []


def test_switch_off_is_the_torch_empty_form(rec):
    with envs.SGLANG_OPT_WEG2_DRAFT_PARK_EXACT_PIN.override(False):
        dp._host_image(IMAGE, pin=True)
    assert rec.exact == []
    assert rec.pinned == [IMAGE]


def test_unpinned_image_never_takes_the_exact_path(rec):
    with envs.SGLANG_OPT_WEG2_DRAFT_PARK_EXACT_PIN.override(True):
        t = dp._host_image(64, pin=False)
    assert rec.exact == [] and rec.pinned == []
    assert t.dtype == torch.uint8


def test_preallocate_goes_through_the_helper(rec):
    park = dp.DraftHostPark(pin=True)
    pop = [("a", _Sized(IMAGE // 2)), ("b", _Sized(IMAGE - IMAGE // 2))]
    with envs.SGLANG_OPT_WEG2_DRAFT_PARK_EXACT_PIN.override(True):
        nbytes, _ms = park.preallocate(pop)
    assert nbytes == IMAGE
    assert rec.exact == [((IMAGE,), torch.uint8)]


class _Sized:
    """Stands in for a uint8 view: preallocate reads only numel()."""

    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n
