"""#718 class, INDEX axis: a host index that outlived the tier that minted it.

THE SPECIMEN. W38 acceptance boot, 2026-08-26 12:54:28Z, all three ranks:

    IndexError: index 76997 is out of bounds for dimension 0 with size 30518
      pool_host/base.py:344   assert self.slot_used[indices_cpu].all()
      memory_pool_host.py:1724  return self.anchor_entry.host_pool.free(indices)
      unified_radix_cache.py:2853  mem_pool_host.free(host_indices[:unclaimed_to])

`HostPoolGroup.anchor_entry` is fixed at construction, so the anchor does not
move inside one group -- the GROUP is rebuilt onto a narrower tier at a phase
rebind. In-flight prefetch state does not move with it: `check_prefetch_progress`
holds host_indices minted against the previous, wider tier and frees them after
the rebind, against a pool whose `slot_used` is shorter.

322f33159a (#718/#847) fixed the TRANSFER path via `_entry_for_transfer` and did
not sweep the index-taking accessors one level up. That resolver cannot cover
them: it resolves by `transfer.name` through `entry_map`, and these five receive
BARE INDICES with no name. The index axis needs a range guard, not the resolver.

WHY free() DROPS AND THE OTHERS RAISE: a dropped free is a no-op on a tier that
no longer exists (nothing leaks -- the tier was torn down wholesale). An accessor
that returns or writes a page cannot drop: that shortens a result or skips a
store and the caller proceeds on data nobody wrote, which is the wrong-answer-
with-no-crash outcome `_entry_for_transfer`'s docstring names.
"""

import pytest
import torch

from sglang.srt.mem_cache.memory_pool_host import (
    StrayHostIndexError,
    _refuse_stray_host_index,
    _split_host_indices_by_binding,
)

RETIRED_TIER_INDEX = 76997  # from the specimen
LIVE_TIER_SIZE = 30518  # from the specimen


class _Pool:
    """A host pool the size of the narrowed, post-rebind tier."""

    pool_name = "kv"

    def __init__(self, size=LIVE_TIER_SIZE):
        self.size = size
        self.slot_used = torch.zeros(size, dtype=torch.bool)

    def free(self, indices):
        # The real assertion from pool_host/base.py:344 -- this is what blew up.
        assert self.slot_used[indices.cpu()].all() or len(indices) == 0
        return len(indices)


def test_specimen_reproduces_without_the_guard():
    """RED: the raw indexing path still fails exactly as W38 did."""
    pool = _Pool()
    idx = torch.tensor([10, RETIRED_TIER_INDEX])
    with pytest.raises(IndexError, match="out of bounds"):
        pool.slot_used[idx]


def test_free_drops_the_stray_and_keeps_the_live_ones():
    pool = _Pool()
    idx = torch.tensor([10, 20, RETIRED_TIER_INDEX, LIVE_TIER_SIZE - 1, 45000])
    live, n_stray = _split_host_indices_by_binding(pool, idx, "free")
    assert n_stray == 2, "both out-of-range indices must be recognised as strays"
    assert live.tolist() == [10, 20, LIVE_TIER_SIZE - 1]
    # and the surviving set is safe to index -- the whole point
    pool.slot_used[live]


def test_negative_index_is_a_stray_too():
    """A negative index silently wraps in torch -- a wrong slot, not a crash."""
    pool = _Pool()
    live, n_stray = _split_host_indices_by_binding(pool, torch.tensor([-1, 5]), "free")
    assert n_stray == 1 and live.tolist() == [5]


@pytest.mark.parametrize("indices", [torch.tensor([], dtype=torch.long), None])
def test_empty_and_none_are_untouched(indices):
    pool = _Pool()
    live, n_stray = _split_host_indices_by_binding(pool, indices, "free")
    assert n_stray == 0


def test_default_path_is_byte_identical():
    """No stray -> the SAME tensor object goes through. Backward compatibility."""
    pool = _Pool()
    idx = torch.tensor([1, 2, 3])
    live, n_stray = _split_host_indices_by_binding(pool, idx, "free")
    assert n_stray == 0
    assert live is idx, "the no-stray path must not copy or reorder"


def test_counter_arms_and_logs_once(caplog):
    """A guard that cannot be seen arming is not evidence of anything (#742)."""
    pool = _Pool()
    stray = torch.tensor([RETIRED_TIER_INDEX])
    with caplog.at_level("ERROR"):
        _split_host_indices_by_binding(pool, stray, "free")
        _split_host_indices_by_binding(pool, stray, "free")
    assert pool._stale_index_refusals == 2, "every occurrence must be counted"
    hits = [r for r in caplog.records if "HICACHE-INDEX REFUSED" in r.getMessage()]
    assert len(hits) == 1, "logged once per pool, then counted silently"


@pytest.mark.parametrize(
    "op", ["get_page_buffer_meta", "get_data_page", "set_from_flat_data_page"]
)
def test_data_accessors_refuse_loudly_by_name(op):
    """These return or write a page: dropping would be a wrong answer."""
    pool = _Pool()
    with pytest.raises(StrayHostIndexError, match="#718 class, index axis"):
        _refuse_stray_host_index(pool, RETIRED_TIER_INDEX, op)


def test_loud_refusal_is_catchable_as_indexerror():
    """Existing `except IndexError` handlers must keep working."""
    assert issubclass(StrayHostIndexError, IndexError)
    pool = _Pool()
    with pytest.raises(IndexError):
        _refuse_stray_host_index(pool, RETIRED_TIER_INDEX, "get_data_page")


def test_data_accessors_pass_live_indices_through():
    pool = _Pool()
    for idx in (0, LIVE_TIER_SIZE - 1, torch.tensor([1, 2])):
        _refuse_stray_host_index(pool, idx, "get_data_page")  # must not raise


def test_scalar_and_tensor_forms_agree():
    """get_data_page takes a scalar, get_page_buffer_meta a tensor."""
    pool = _Pool()
    with pytest.raises(StrayHostIndexError):
        _refuse_stray_host_index(pool, torch.tensor([RETIRED_TIER_INDEX]), "x")
    with pytest.raises(StrayHostIndexError):
        _refuse_stray_host_index(pool, RETIRED_TIER_INDEX, "x")
