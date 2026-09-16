"""#1426 (xsn184/185/186): under write_back the peel drains the in-flight
write-throughs and retries once before giving a leaf up -- upstream's
synchronous write-back, not a second bookkeeping of what is 'deliverable'."""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


def _peel(written_seq, ongoing):
    t = object.__new__(UnifiedRadixCache)
    calls = []
    t._is_device_leaf = lambda n: True
    t.cache_controller = types.SimpleNamespace(write_policy="write_back")
    t.ongoing_write_through = ongoing
    seq = list(written_seq)
    t.write_backup = lambda n, write_back=False: (calls.append("write"), seq.pop(0))[1]
    t.writing_check = lambda write_back=False: calls.append("drain")
    t._evict_to_host = lambda n, tr: calls.append("evict")
    node = types.SimpleNamespace(backuped=False, id=1)
    t._evict_device_leaf(node, {})
    return calls


def test_a_refused_leaf_is_retried_after_the_drain():
    assert _peel([0, 4096], ongoing={7: 1}) == ["write", "drain", "write", "drain", "evict"]


def test_a_second_zero_gives_the_leaf_up():
    assert _peel([0, 0], ongoing={7: 1}) == ["write", "drain", "write"]


def test_nothing_in_flight_means_nothing_to_drain():
    assert _peel([0], ongoing={}) == ["write"]


def test_full_evictable_size_is_the_tree_counter_again():
    from sglang.srt.mem_cache import unified_radix_cache as urc
    t = object.__new__(UnifiedRadixCache)
    t.component_evictable_size_ = {urc.BASE_COMPONENT_TYPE: 4711}
    assert t.full_evictable_size() == 4711
