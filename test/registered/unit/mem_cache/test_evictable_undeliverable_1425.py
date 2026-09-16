"""#1425 (xsn184): under write_back an un-backed device leaf the staging ring
cannot absorb is not deliverable -- the admission budget leaves it out."""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


def _tree(avail, leaves, policy="write_back"):
    t = object.__new__(UnifiedRadixCache)
    t.component_evictable_size_ = {"full": sum(n for n, _ in leaves)}
    from sglang.srt.mem_cache import unified_radix_cache as urc
    t.component_evictable_size_ = {urc.BASE_COMPONENT_TYPE: sum(n for n, _ in leaves)}
    t.evictable_device_leaves = [types.SimpleNamespace(key=[0] * n, backuped=b) for n, b in leaves]
    t.cache_controller = types.SimpleNamespace(
        write_policy=policy, mem_pool_host=types.SimpleNamespace(available_size=lambda: avail))
    return t


def test_unbacked_leaves_beyond_the_ring_are_not_counted():
    t = _tree(avail=5000, leaves=[(4096, True), (4096, False), (4096, False), (4096, False)])
    assert t.evictable_size() == 16384
    # one un-backed leaf fits the 5000 free rows, two do not
    assert t.full_evictable_size() == 16384 - 2 * 4096


def test_backed_leaves_and_write_through_are_untouched():
    t = _tree(avail=0, leaves=[(4096, True), (4096, True)])
    assert t.full_evictable_size() == 8192
    t2 = _tree(avail=0, leaves=[(4096, False)], policy="write_through")
    assert t2.full_evictable_size() == 4096
