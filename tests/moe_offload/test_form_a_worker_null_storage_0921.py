"""fnFL2 v16 (21.09.): a Form A expert worker has no bytes in the canonical
KV page (CanonicalPageError: 'a canonical KV page cannot be 0 bytes'), but
must still run the HiCache control path -- the prefetch progress MIN reduce
spans the whole tp group. The worker rides a null storage tier that claims
every page and moves nothing."""

import inspect
import types

from sglang.srt.mem_cache import hicache_storage as hs


def test_null_backend_claims_everything_and_moves_nothing():
    b = hs.FormAWorkerNullStorage(storage_config=None)
    assert b.batch_exists(["a", "b", "c"]) == 3
    pages = [object(), object()]
    assert b.batch_get(["a", "b"], pages) == pages
    assert b.batch_set(["a"], [b""]) is True
    t = types.SimpleNamespace(name="mamba", keys=["k1", "k2"])
    r = b.batch_exists_v2(["a", "b", "c"], [t])
    assert r.kv_hit_pages == 3 and r.extra_pool_hit_pages == {"mamba": 2}
    assert b.batch_get_v2([t]) == {"mamba": [True, True]}
    assert b.batch_set_v2([t]) == {"mamba": [True, True]}
    assert b.check_disk_space() is True and b.get_stats() is None
    b.register_mem_pool_host("pool")
    b.register_mem_host_pool_v2("p2", "kv")
    assert b.mem_pool_host == "pool" and b.registered_pools == {"kv": "p2"}
    b.clear(); b.close()


def test_the_controller_routes_a_worker_to_the_null_tier():
    from sglang.srt.managers import cache_controller as cc

    src = inspect.getsource(cc.HiCacheController.attach_storage_backend)
    assert "if this_rank_is_form_a_worker():" in src
    assert "FormAWorkerNullStorage(self.storage_config)" in src
    src = inspect.getsource(cc.HiCacheController._generate_storage_config)
    assert "if this_rank_is_form_a_worker():" in src
    # the canonical window is never built on a worker
    assert src.index("if this_rank_is_form_a_worker():") < src.index("build_page_window(")
