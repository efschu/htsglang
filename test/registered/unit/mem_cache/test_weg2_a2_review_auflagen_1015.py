"""A2 review conditions (nf-chainkey-a2-review-1015) on h16o.

B-A: on the hybrid NF tree ``cache_controller.mem_pool_host`` is a
HostPoolGroup; its ``__getattr__`` did not forward ``take_lost_slots`` /
``settle_lost_rows``, so the A2 ack settle (``_a2_settle_lost``) got None and
silently settled nothing on every host rank. The first two tests are the
reviewer's probes (``/tmp/a2rev_head/.../test_a2rev_probe_hostpoolgroup.py``),
registered: the ack path runs through a REAL HostPoolGroup.

B-B: ONE P mamba anchor slot carries every PP stage's layer extents. After an
H54 take-over the stages that acked before it got 0 (a reference), the later
ones 6; A2 tombstoned the anchor by the LOCAL ack status, i.e. on the late
stages only -- the P trees disagreed (raenge-nie-uneins). The decision now
reads the SHARED slot refcount: referenced -> every stage keeps it; nobody
references it (every stage late) -> every stage tombstones it.

B-F: ``quarantine_release`` passes through the private S_REAPING, never the
public S_EVICTING (which arena_free_slots / the aged reaper take).
"""

import os
import re
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["SGLANG_HICACHE_ARENA_QUEUE_REFS"] = "1"

import torch  # noqa: E402

from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup  # noqa: E402
from sglang.srt.mem_cache.storage.file import hicache_arena as ha  # noqa: E402

from test_weg2_chainkey_a2_1015 import HALF, S, TOTAL, _arena, _late_writer, _pool, _rows  # noqa: E402


def _group(pool):
    g = object.__new__(HostPoolGroup)
    g.anchor_entry = types.SimpleNamespace(host_pool=pool)
    return g


# ------------------------------------------------------------------ B-A ---------------------------------


def test_group_forwards_the_a2_settle(tmp_path):
    _a, _b, pool_b, _slot, _r, _g = _late_writer(tmp_path, slots=1)
    g = _group(pool_b)
    assert callable(getattr(g, "complete_write", None))
    assert callable(getattr(g, "take_lost_slots", None)), "take_lost_slots not reachable through HostPoolGroup"
    assert callable(getattr(g, "settle_lost_rows", None)), "settle_lost_rows not reachable through HostPoolGroup"


def test_ack_path_settles_through_the_production_group(tmp_path):
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, UnifiedRadixCache

    _arena_a, _arena_b, pool_b, slot, _rows_b, gen = _late_writer(tmp_path, slots=1)
    hv = _rows([slot])
    pool_b._pending_mask[slot] = True
    pool_b._pending_gen[slot] = gen
    pool_b._pending_fresh[slot] = False
    pool_b.size = S   # HostPoolGroup.staging_rows evaluates host_pool.size as getattr default
    t = object.__new__(UnifiedRadixCache)
    t.cache_controller = types.SimpleNamespace(mem_pool_host=_group(pool_b), mem_pool_host_draft=None)
    t._weg2_mamba_pool = lambda: None
    cd = {BASE_COMPONENT_TYPE: types.SimpleNamespace(host_value=hv)}
    node = types.SimpleNamespace(id=9, hash_value=["X"], component_data=cd, l3_present=False)
    assert t._weg2_direct_complete(node) is True
    assert int(hv.min()) >= S + pool_b.arena_tokens, "PROD FORM: the ack left the node's rows on the lost slot"


def test_a_pool_without_the_settle_is_named(tmp_path, caplog):
    """A wiring gap is loud, never a silent no-op."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    UnifiedRadixCache._a2_settle_missing_n = 0
    t = object.__new__(UnifiedRadixCache)
    caplog.set_level("ERROR")
    t._a2_settle_lost(types.SimpleNamespace(id=7), types.SimpleNamespace(), None, torch.tensor([S]))
    assert "A2 LOST-ROWS SETTLE UNREACHABLE" in caplog.text
