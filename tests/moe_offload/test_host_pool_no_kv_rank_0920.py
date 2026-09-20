"""fnFL2 v12 (20.09.): a Form A expert worker holds no attention layer, so
its HiCache host pool has 0 bytes/token and the fixed-size arithmetic divided
by zero (pool_host/base.py). The rank still joins the min-reduce."""

import inspect

from sglang.srt.mem_cache.pool_host import base as b


def test_no_kv_rank_bids_the_sentinel_and_others_their_budget():
    assert b.fixed_host_pool_tokens(4, 0) == b.NO_KV_RANK_TOKENS
    assert b.fixed_host_pool_tokens(4, 1000) == int(4e9 // 1000)
    assert b.fixed_host_pool_tokens(0.5, 7616) == int(0.5e9 // 7616)


def test_constructor_sizes_through_the_helper_before_the_min_reduce():
    src = inspect.getsource(b.HostKVCache.__init__)
    assert "fixed_host_pool_tokens(host_size, self.size_per_token), host_size" in src
    assert "host_size * 1e9 // self.size_per_token" not in src


def test_mamba_anchor_pool_uses_the_same_rule():
    assert b.fixed_host_pool_slots(600 * 1024**2, 0) == b.NO_KV_RANK_TOKENS
    assert b.fixed_host_pool_slots(600 * 1024**2, 37 * 1024**2) == 16
    from sglang.srt.mem_cache import memory_pool_host as mh

    src = inspect.getsource(mh)
    assert "fixed_host_pool_slots(int(anchor_host_mib) * (1024**2), self.size_per_token)" in src
    assert "fixed_host_pool_tokens(host_size, self.size_per_token), host_size" in src
    assert "// self.size_per_token" not in src
