"""Falsification test: SWA allocator invariants under EAGLE3-style free patterns.

Context (task #92): the Thoughtworks Gemma-4 EAGLE3 bring-up hit a SWAKVPool
double-free during tree verification in upstream sglang — their pool-to-pool
mapping used 0 (a VALID slot there) as the "unmapped" value, so freeing slot 0
was skipped and the pools desynchronized. Their fix: sentinel 0 -> -1, an
allocated mask, and a double-free guard.

This fork's SWA design is structurally different and claims the bug-class is
impossible:
  * slot 0 of BOTH sub-allocators is a reserved dummy (never handed out), so
    mapping value 0 unambiguously means "unmapped";
  * free() clears the full->swa mapping to 0, making the SWA side of a
    repeated free a no-op by construction;
  * a -1 tail entry makes translate(-1) == -1 (last_loc sentinel);
  * a caller-level double-free of FULL slots is caught by the
    available_size <= size assert in SWATokenToKVPoolAllocator.free().

These tests FALSIFY those claims if any of them is untrue. If a future change
breaks one of the guards (e.g. makes slot 0 allocatable, stops clearing the
mapping on free, or removes the post-free assert), a test here fails.

CPU-only, no GPU required:
    python -m pytest test/manual/spec/test_swa_allocator_spec_free_invariants.py -q
"""

import random

import pytest
import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

FULL_SIZE = 4096
SWA_SIZE = 1024
DEVICE = "cpu"


class _StubSWAKVPool(BaseSWAKVPool):
    """Duck-typed stand-in: the allocator only needs register_mapping and
    translate_loc_from_full_to_swa; no tensors are stored."""

    def __init__(self):
        self.full_kv_pool = None
        self.swa_kv_pool = None
        self.full_to_swa_index_mapping = None

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices]

    def get_state_buf_infos(self):
        return [], [], []

    # KVCache ABC stubs (unused by the allocator paths under test).
    def get_key_buffer(self, layer_id: int):
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int):
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int):
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError()

    def get_kv_size_bytes(self):
        return 0, 0

    def get_contiguous_buf_infos(self):
        return [], [], []


def _make_allocator() -> SWATokenToKVPoolAllocator:
    return SWATokenToKVPoolAllocator(
        size=FULL_SIZE,
        size_swa=SWA_SIZE,
        page_size=1,
        dtype=torch.bfloat16,
        device=DEVICE,
        kvcache=_StubSWAKVPool(),
        need_sort=False,
    )


def test_alloc_sets_mapping_and_free_clears_it():
    a = _make_allocator()
    full0, swa0 = a.full_available_size(), a.swa_available_size()

    idx = a.alloc(16)
    assert idx is not None and idx.numel() == 16
    mapped = a.full_to_swa_index_mapping[idx]
    assert bool((mapped > 0).all()), "freshly allocated full slots must map to real (nonzero) SWA slots"

    a.free(idx)
    assert a.full_available_size() == full0
    assert a.swa_available_size() == swa0
    assert bool((a.full_to_swa_index_mapping[idx] == 0).all()), (
        "free() must clear the full->swa mapping to 0 (the guard that makes "
        "the SWA side of a repeated free a no-op)"
    )


def test_double_free_swa_side_is_noop_and_full_side_is_caught():
    """The TW bug-class probe: free the same full indices twice.

    Expected fork behavior: the SWA side frees nothing the second time
    (mapping already cleared), and the FULL side's slot-conservation assert
    fires instead of silently corrupting the free list."""
    a = _make_allocator()
    idx = a.alloc(8)
    a.free(idx)
    swa_after_first = a.swa_available_size()

    with pytest.raises(AssertionError):
        a.free(idx)

    # Even though the second free asserted on the full side, the SWA pool must
    # not have double-freed: its available size is unchanged.
    assert a.swa_available_size() == swa_after_first, (
        "SWA sub-pool double-freed despite the cleared mapping — the TW "
        "desync bug-class is present"
    )


def test_slot_zero_is_never_allocated():
    a = _make_allocator()
    seen_full = []
    seen_swa = []
    while True:
        idx = a.alloc(64)
        if idx is None:
            break
        seen_full.append(idx)
        seen_swa.append(a.full_to_swa_index_mapping[idx])
        # keep everything allocated; drain the smaller (SWA) pool
    full_ids = torch.cat(seen_full)
    swa_ids = torch.cat(seen_swa)
    assert int(swa_ids.numel()) == SWA_SIZE, "expected to drain the SWA pool"
    assert int((full_ids == 0).sum()) == 0, "full slot 0 (dummy) was handed out"
    assert int((swa_ids == 0).sum()) == 0, "swa slot 0 (dummy) was handed out"
    # exact coverage: slots 1..SWA_SIZE each exactly once (no dup, no skip)
    assert torch.equal(
        torch.sort(swa_ids).values, torch.arange(1, SWA_SIZE + 1, dtype=swa_ids.dtype)
    )


def test_minus_one_tail_sentinel_survives_clear():
    a = _make_allocator()
    minus_one = torch.tensor([-1], dtype=torch.int64)
    assert int(a.translate_loc_from_full_to_swa(minus_one)) == -1
    a.clear()
    assert int(a.translate_loc_from_full_to_swa(minus_one)) == -1, (
        "clear() wiped the -1 tail sentinel; last_loc==-1 would now translate "
        "to swa slot 0 and corrupt the dummy slot"
    )


def test_eagle3_verify_like_alloc_free_cycles_conserve_slots():
    """Spec-decode shaped workload: per decode step, allocate a draft-token
    block, free the rejected tail (overshoot), keep the accepted head; at the
    end free everything kept. Slot conservation must be exact on both pools
    and the mapping must return to all-unmapped."""
    random.seed(1234)
    a = _make_allocator()
    full0, swa0 = a.full_available_size(), a.swa_available_size()

    kept = []
    prefill = a.alloc(256)  # prompt KV
    assert prefill is not None
    kept.append(prefill)

    num_draft_tokens = 8  # tree size per step (topk x steps + 1)
    for _ in range(60):
        block = a.alloc(num_draft_tokens)
        assert block is not None
        accept_len = random.randint(1, num_draft_tokens)  # accepted chain + bonus
        accepted, rejected = block[:accept_len], block[accept_len:]
        a.free(rejected)  # overshoot free, mirrors the post-verify free
        kept.append(accepted)

    a.free(torch.cat(kept))  # request finished

    assert a.full_available_size() == full0, "full pool leaked/over-freed slots"
    assert a.swa_available_size() == swa0, "swa pool leaked/over-freed slots"
    mapping = a.full_to_swa_index_mapping
    assert int((mapping[:-1] != 0).sum()) == 0, "stale full->swa mappings remain"
    assert int(mapping[-1]) == -1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
