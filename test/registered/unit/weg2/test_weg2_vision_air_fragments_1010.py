"""VISION-WEIGHTS (metal 10.10., window jzmxnp, INT8 flip 4096x4096): the air check passed, the encode OOMed.

THE BUG. P.log of boot dkr27bravwweightsbar1fs10100757 (line 27556): W107 ``OutOfMemoryError: Tried to
allocate 538.00 MiB ... 266.75 MiB is free ... 420.32 MiB is reserved by PyTorch but unallocated``. The weights
stage had checked the encode's work memory (1574 MiB) against the card's air = cudaMemGetInfo free (1471 MiB)
+ the allocator's idle cache (489 MiB) = 1960 MiB and let it run. The idle cache lay in pieces smaller than
538 MiB, the size of the largest single tensor (the MLP fc1 out, 65536 x 4304 x 2 B): the encode used up the
free memory with its earlier tensors and fc1 found no block -- not in the cache, not in the rest of free.

THE FIX (pinned here). The weights path returns the idle cache to the driver (``torch.cuda.empty_cache``)
BEFORE it measures, and the check counts only what the driver then reports free: cached segments that are
wholly unused become free memory a cudaMalloc of any size can take, and what stays cached sits in segments
that hold a live block -- fragments, not counted.

Hermetic, CPU: a fake card models the caching allocator (an allocation takes the best-fitting idle block or
else a new segment from free, else OOM); ``torch.cuda.empty_cache`` is patched to release its wholly unused
segments. Red on 02123ee2ef (the encode OOMs behind a passed check in both cases), green with the fix.
"""

import logging
import types

import pytest
import torch

from sglang.srt.weg2 import vision_rank_runner as vrr
from sglang.srt.weg2 import vision_victim as vv
from test_weg2_vision_rank_runner import _Item, _req, _write_model  # noqa: E402 -- the runner tests' fakes
from test_weg2_vision_victim_core import _run, _sched, _Victims  # noqa: E402

MIB = 1 << 20

#: the 27B tower of the metal (Qwen3_5ForConditionalGeneration, no deepstack): 65536 patches -> 1574 MiB
_VC = types.SimpleNamespace(hidden_size=1152, intermediate_size=4304, num_heads=16, out_hidden_size=5120,
                            spatial_merge_size=2, deepstack_visual_indexes=[], patch_size=16,
                            in_channels=3, temporal_patch_size=2)
WORK_MIB = 1574
FC1_MIB = 538  # 65536 x 4304 x 2 B: the largest single tensor of the encode
FREE_MIB = 1471


class _FragmentedCard:
    """cudaMemGetInfo free + the caching allocator's idle blocks, in MiB.

    ``whole`` are cached segments with no live block (``empty_cache`` returns them), ``split`` are idle
    pieces of segments that also hold a live block (``empty_cache`` cannot return them)."""

    def __init__(self, *, free, whole, split):
        self.free, self.whole, self.split = free, list(whole), list(split)
        self.events = []

    def air(self, device):
        self.events.append("air")
        return self.free * MIB, (sum(self.whole) + sum(self.split)) * MIB

    def empty_cache(self):
        self.events.append("empty_cache")
        self.free += sum(self.whole)
        self.whole = []

    def alloc(self, mib):
        for pool in (self.whole, self.split):
            fits = [b for b in pool if b >= mib]
            if fits:
                pool.remove(min(fits))
                return
        if self.free < mib:
            raise torch.OutOfMemoryError(
                f"Tried to allocate {mib:.2f} MiB. {self.free:.2f} MiB is free, "
                f"{sum(self.whole) + sum(self.split):.2f} MiB is reserved by PyTorch but unallocated")
        self.free -= mib


def _stage(tmp_path, monkeypatch, card):
    _write_model(tmp_path)
    monkeypatch.setattr(torch.cuda, "empty_cache", card.empty_cache)
    # NF int24d: this tree does not carry the 27B leaner tower forward (9ee0536427 VISION-WORK), so its work
    # model books 15403581440 B for the 4096x4096 image, not the metal's 1574 MiB. The test pins the AIR
    # check, not the work model: the work is booked at the metal figure.
    monkeypatch.setattr(vv, "encode_work_for", lambda vision_config, items, backend: WORK_MIB * MIB)
    it = _Item()
    it.image_grid_thw = torch.tensor([[1, 256, 256]])  # 4096x4096 at patch 16 = 65536 patches
    victims = _Victims()

    def encode(module, items):
        card.alloc(WORK_MIB - FC1_MIB)  # pixels, rope, residual, norm2: all larger than every fragment
        card.alloc(FC1_MIB)
        return vrr.encode_items(module, items)

    out = _run(_sched(), [_req("r", [it])], tmp_path, victims=victims, encode=encode,
               hf_config=types.SimpleNamespace(vision_config=_VC), air=card.air)
    assert out.planned_work_bytes == WORK_MIB * MIB
    return out, victims, it


def test_idle_cache_in_unused_segments_is_returned_before_the_check(tmp_path, monkeypatch, caplog):
    """The metal numbers, every idle piece < 538 MiB in a wholly unused segment: empty_cache makes them
    free memory before the measurement, the check sees 1960 MiB free and the encode's fc1 finds its block."""
    card = _FragmentedCard(free=FREE_MIB, whole=[300, 189], split=[])
    with caplog.at_level(logging.INFO, logger=vrr.logger.name):
        out, victims, it = _stage(tmp_path, monkeypatch, card)
    assert out.ok, out.detail
    assert card.events[:3] == ["air", "empty_cache", "air"]
    assert victims.intact() and it.precomputed_embeddings is not None
    line = next(r.getMessage() for r in caplog.records if "AIR" in r.getMessage())
    assert line.startswith("W102 Weg2VisionStage AIR ")
    for field in ("free_before_mib=1471", "idle_before_mib=489", "free_after_mib=1960", "idle_after_mib=0",
                  "work_mib=1574", "verdict=fits"):
        assert field in line, line


def test_fragments_in_live_segments_are_not_air(tmp_path, monkeypatch):
    """The same idle cache in segments that hold live blocks: empty_cache returns nothing, the honest air
    is 1471 MiB free, and the stage refuses by name (W105b, before any victim byte moved) instead of the
    OOM inside the encode."""
    card = _FragmentedCard(free=FREE_MIB, whole=[], split=[300, 189])
    out, victims, it = _stage(tmp_path, monkeypatch, card)
    assert not out.ok and out.code == vv.W_VICTIM_SHORT, out.detail
    assert "OutOfMemoryError" not in out.detail
    assert "air is 1471 MiB" in out.detail
    assert "free 1471 before" in out.detail and "1471 after empty_cache" in out.detail
    assert victims.stashed == 0 and victims.intact() and it.precomputed_embeddings is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
