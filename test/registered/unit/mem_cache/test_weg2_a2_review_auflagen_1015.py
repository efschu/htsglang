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


# ------------------------------------------------------------------ B-B ---------------------------------


def _stage(tmp_path, ext):
    a = _arena(tmp_path, 4)
    mp = _pool(a, extents=(ext,))
    mp._arena_page_tokens = 1
    mp.arena_tokens = a.slots
    return a, mp


def _tree():
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    t = object.__new__(UnifiedRadixCache)
    lru = types.SimpleNamespace(in_list=lambda n: False, remove_node=lambda n: None)
    t.host_lru_lists = {ComponentType.MAMBA: lru}
    return t


def _anchor_node(mhv):
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    cd = types.SimpleNamespace(host_value=mhv)
    return types.SimpleNamespace(id=1, component_data={ComponentType.MAMBA: cd}), cd


def _two_stage_anchor(tmp_path):
    """Two PP stages host-write their layer extents of ONE mamba anchor slot."""
    arena_a, mp_a = _stage(tmp_path, (0, HALF))
    arena_b, mp_b = _stage(tmp_path, (HALF, HALF))
    (slot, st, gen), = arena_a.claim_slots(["m"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    (slot_b, st_b, gen_b), = arena_b.claim_slots(["m"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    assert (slot_b, gen_b) == (slot, gen) and st == 0 and st_b == 1
    for mp in (mp_a, mp_b):
        mp._pending_mask[slot] = True
        mp._pending_gen[slot] = gen
    return arena_a, mp_a, mp_b, slot, gen


def test_mixed_stage_acks_decide_the_anchor_alike(tmp_path):
    """Stage A acks before the take-over (status 0, reference), stage B after (status 6, none):
    both keep the anchor -- A's reference pins the slot, it cannot be recycled."""
    arena_a, mp_a, mp_b, slot, gen = _two_stage_anchor(tmp_path)
    mhv_a, mhv_b = torch.tensor([S + slot]), torch.tensor([S + slot])
    mp_a.complete_write(mhv_a)                                      # A: 0, reference taken
    assert _arena(tmp_path, 4).quarantine_host_write([slot], [gen]) == [1]
    node_a, cd_a = _anchor_node(mhv_a)
    lost_a = _tree()._a2_mamba_lost(node_a, mp_a, mhv_a)
    mp_b.complete_write(mhv_b)                                      # B: 6, no reference
    node_b, cd_b = _anchor_node(mhv_b)
    lost_b = _tree()._a2_mamba_lost(node_b, mp_b, mhv_b)
    assert arena_a.slot_refs([slot]) == [1], "A's reference pins the slot"
    assert (lost_a, lost_b) == (False, False), f"asymmetric: A kept={not lost_a}, B kept={not lost_b}"
    assert cd_a.host_value is not None and cd_b.host_value is not None


def test_every_stage_late_tombstones_the_anchor_alike(tmp_path):
    """Both stages ack after the take-over (both 6, no reference anywhere): both tombstone."""
    arena_a, mp_a, mp_b, slot, gen = _two_stage_anchor(tmp_path)
    assert _arena(tmp_path, 4).quarantine_host_write([slot], [gen]) == [1]
    out = []
    for mp in (mp_a, mp_b):
        mhv = torch.tensor([S + slot])
        mp.complete_write(mhv)
        node, cd = _anchor_node(mhv)
        out.append((_tree()._a2_mamba_lost(node, mp, mhv), cd.host_value is None))
    assert arena_a.slot_refs([slot]) == [0]
    assert out == [(True, True), (True, True)], out


# ------------------------------------------------------------------ B-F ---------------------------------


def test_quarantine_release_passes_through_the_private_reaping_state():
    src = open(os.path.join(os.path.dirname(ha.__file__), "arena.c")).read()
    body = re.search(r"static int quarantine_release\(.*?\n}\n", src, re.S).group(0)
    assert "S_REAPING" in body and "&expect, S_EVICTING" not in body, body


def test_quarantine_release_still_frees_and_moves_the_generation(tmp_path):
    arena = _arena(tmp_path, 1)
    (s, st, g), = arena.claim_slots(["Q"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    assert arena.complete_slots([s], [g], [(0, HALF)]) == [0]
    assert arena.ref_slots([s], +1) == 1
    assert arena.quarantine_host_write([s], [g]) == [1]
    g_q = int(arena.slot_gens_np([s])[0])
    assert arena.ref_slots([s], -1) == 1
    assert arena.claim_census([s])[0]["state"] == 0
    assert int(arena.slot_gens_np([s])[0]) == g_q + 1
    assert arena.stats()["claimed"] == 0
    (s2, st2, _g2), = arena.claim_slots(["R"], [TOTAL])
    assert s2 == s and st2 == 0, "the freed slot is claimable again"
