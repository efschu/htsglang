"""Punkt 3 (user order 18.09.): a memory-pressure retract on Weg 2 group D
retains the span in the tree (evictable) instead of discarding it."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import retract_retain as rr  # noqa: E402


def test_standard_is_on_for_group_d_only():
    assert rr.retract_retains({"SGLANG_WEG2_GROUP": "D"})
    assert rr.retract_retains({"SGLANG_WEG2_GROUP": "d"})
    assert not rr.retract_retains({"SGLANG_WEG2_GROUP": "P"})
    assert not rr.retract_retains({})  # non-Weg-2 boots keep upstream's discard


def test_env_knob_turns_it_off_for_an_ab():
    assert not rr.retract_retains({"SGLANG_WEG2_GROUP": "D", rr.ENV: "0"})
    assert not rr.retract_retains({"SGLANG_WEG2_GROUP": "D", rr.ENV: "off"})
    assert rr.retract_retains({"SGLANG_WEG2_GROUP": "D", rr.ENV: "1"})


def test_group_env_mirrors_corridor_guard():
    from sglang.srt.managers.corridor_guard import GROUP_ENV
    assert rr.GROUP_ENV == GROUP_ENV


def test_release_req_retain_inserts_into_the_tree_and_skips_the_evict():
    """`release_req(retain=True)` -> `cache_finished_req(is_insert=True)`
    and no `evict_from_tree_cache`; retain=False keeps upstream's discard."""
    from sglang.srt.managers import schedule_batch as sb

    calls = []

    class _Tree:
        def cache_finished_req(self, req, is_insert=True, **kw):
            calls.append(("insert", bool(is_insert)))

        def evict(self, *a, **k):
            calls.append(("evict",))

    class _Req:
        rid = "weg2-0-1"
        req_pool_idx = 3
        kv_spill_state = "device"
        skip_radix_cache_insert = False
        origin_input_ids = [1, 2, 3]
        output_ids = [4]

        def reset_for_retract(self):
            calls.append(("reset",))

    seen = {}
    orig_rkc = sb.release_kv_cache
    orig_evict = sb.evict_from_tree_cache

    def _rkc(req, tree, is_insert=True):
        tree.cache_finished_req(req, is_insert=is_insert)

    def _evict(tree, n):
        calls.append(("evict", n))

    sb.release_kv_cache = _rkc
    sb.evict_from_tree_cache = _evict
    try:
        sa = types.SimpleNamespace(disaggregation_mode="null")
        for retain in (True, False):
            calls.clear()
            sb.release_req(req=_Req(), remaing_req_count=2, server_args=sa,
                           req_to_token_pool=None, token_to_kv_pool_allocator=None,
                           tree_cache=_Tree(), hisparse_coordinator=None, retain=retain)
            seen[retain] = list(calls)
    finally:
        sb.release_kv_cache = orig_rkc
        sb.evict_from_tree_cache = orig_evict
    assert ("insert", True) in seen[True] and not any(c[0] == "evict" for c in seen[True])
    assert ("insert", False) in seen[False] and any(c[0] == "evict" for c in seen[False])
    assert seen[True][-1] == ("reset",) and seen[False][-1] == ("reset",)


def test_retract_decode_passes_the_group_verdict_to_every_release():
    from sglang.srt.managers import schedule_batch as sb
    src = open(sb.__file__).read()
    i = src.index("    def retract_decode(")
    j = src.index("    def _get_decode_retraction_order(", i)
    body = src[i:j]
    assert "retract_retains as _weg2_rr" in body
    assert body.count("self.release_req(") == 3
    # the two RETRACT sites retain; the ABORT site (solo-OOM past max retries)
    # discards -- an aborted request has no re-admission to load back for
    assert body.count("retain=_retain)") == 2, body.count("retain=_retain)")
    k = body.index("reqs_to_abort.append(last_req)")
    assert "self.release_req(last_idx, 0, server_args)\n" in body[k:k + 200]
    assert "WEG2-RETRACT-RETAIN" in body


def test_lock_census_names_the_holders():
    """Punkt 1 instrument: the census walks the tree and names locked nodes."""
    import enum
    from sglang.srt.mem_cache import unified_radix_cache as urc
    base = urc.BASE_COMPONENT_TYPE
    comps = [base]

    class _CD:
        def __init__(self, value, lock_ref=0):
            self.value = value
            self.lock_ref = lock_ref
            self.host_lock_ref = 0

    class _Node:
        def __init__(self, nid, rows, lock_ref, backuped):
            self.id = nid
            self.children = {}
            self.component_data = {base: _CD(list(range(rows)) if rows else None, lock_ref)}
            self.backuped = backuped

    root = _Node(0, 0, 0, False)
    a = _Node(1, 4096, 0, True)      # evictable
    b = _Node(2, 95000, 1, True)     # LOCKED, backed
    c = _Node(3, 100, 2, False)      # locked, unbacked
    root.children = {"a": a, "b": b, "c": c}
    fake = types.SimpleNamespace(
        root_node=root, tree_components=comps,
        component_evictable_size_={base: 4096}, component_protected_size_={base: 95100},
        ongoing_write_through={3: None}, ongoing_load_back={},
        _collect_all_nodes=lambda: [root, a, b, c],
    )
    out = urc.UnifiedRadixCache.weg2_lock_census_str(fake, limit=2)
    assert out.startswith("lock_census nodes=4")
    assert "tracked_evictable=4096" in out and "tracked_protected=95100" in out
    assert "device_rows_in_tree=99196" in out
    assert f"{base.name}_locked_nodes=2" in out and f"{base.name}_locked_rows=95100" in out
    assert "id=2 dev=95000" in out and "wt=0" in out
    assert "id=3 dev=100" in out and "wt=1" in out
    assert "id=1 " not in out  # unlocked nodes are not holders


def test_the_stall_site_logs_the_census():
    from sglang.srt.managers import scheduler as sch
    src = open(sch.__file__).read()
    i = src.index("WEG2-INTAKE-STALL-CENSUS rid=")
    blk = src[i - 600:i + 400]
    assert "weg2_lock_census_str" in blk and "available_size()" in blk
