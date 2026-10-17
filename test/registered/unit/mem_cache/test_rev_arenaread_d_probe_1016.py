# Review probes (not for commit): READ-ROOM-D 54ea321c06 Einigkeits-Review 16.10.
import os
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache import form_a_host_shadow as R12
from sglang.srt.mem_cache.pool_host import arena_pool as AP

from test_weg2_read_room_1015 import _fill_arena_with_tree_pages, _rr, _setup
from test_weg2_read_room_d_1015 import (  # noqa: F401  (fixtures)
    _clean,
    _d_tree,
    _Pool,
    evict_mock,
    roles,
)


def _files(tmp_path, spans):
    root = tmp_path / "store"
    root.mkdir(exist_ok=True)
    for _f, sl in spans:
        for s in sl:
            (root / f"s{s}.bin").write_bytes(b"\x01")
    return str(root), {s: f"s{s}" for _f, sl in spans for s in sl}


def test_RD1_l3_present_is_group_equal_after_the_verdict(tmp_path, roles, evict_mock):
    """Ranks never disagree: after TP0's D-SPILL and the broadcast, l3_present of the spilled node must be the same
    on TP0 and the worker (the parent rule of write_backup, publish_unbacked_sweep and anchor-only read it)."""
    rr = _rr()
    me = threading.current_thread().name
    spans = [(1, list(range(0, 8)))]
    root, stem_of = _files(tmp_path, spans)
    tp0 = _d_tree(spans, None, "tp0")
    wk = _d_tree(spans, None, "worker")
    assert not tp0._t_nodes[1].l3_present and not wk._t_nodes[1].l3_present
    roles.now[me] = "host"
    assert rr.form_a_spill(tp0, _Pool(None, stem_of, root), 1) == 8
    v = R12.attach([])
    roles.now[me] = "worker"
    R12.consume(v, wk)
    assert wk._t_nodes[1].component_data[R12._base_ct()].host_value is None   # rows dropped (works)
    assert tp0._t_nodes[1].l3_present == wk._t_nodes[1].l3_present, (
        tp0._t_nodes[1].l3_present, wk._t_nodes[1].l3_present)


class _UnpairedBackend:
    """arena_secure_to_disk as the real one answers for KV pages whose QSA index is busy/missing (QS pair gate):
    nothing written, nothing on disk, lost=0, unpaired=n."""

    def arena_secure_to_disk(self, arena, cands, writer="claim_room"):
        return {"on_disk": 0, "written": 0, "lost": 0, "unpaired": len(cands)}


class _RealSecurePool:
    staging_rows = 0
    page_size = 1

    def __init__(self):
        self.arena = types.SimpleNamespace(slot_bytes=4096)
        self._backend = _UnpairedBackend()
        self._page_bytes = 4096

    def secure_rows_to_l3(self, hv):
        return AP.ArenaMHAHostPool.secure_rows_to_l3(self, hv)


def test_RD2_a_page_without_an_l3_copy_is_never_released(tmp_path, roles, evict_mock, monkeypatch):
    """D-NORECOMPUTE / store complete: the REAL secure_rows_to_l3 with an unpaired answer (QS gate: KV not written,
    lost=0). The node has no L3 copy -> it must keep its host copy and must not be marked l3_present."""
    rr = _rr()
    roles.now[threading.current_thread().name] = "host"
    monkeypatch.setattr(AP, "arena_ref_slots", lambda pool, hv: torch.as_tensor(hv, dtype=torch.int64))
    spans = [(1, list(range(0, 8)))]
    tp0 = _d_tree(spans, None, "tp0")
    got = rr.form_a_spill(tp0, _RealSecurePool(), 1)
    assert got == 0 and tp0._t_released == [] and not tp0._t_nodes[1].l3_present, (
        got, tp0._t_released, tp0._t_nodes[1].l3_present)


def test_RD3_no_rank_difference_inside_the_pass(tmp_path, roles, evict_mock):
    """R12 contract: TP0's own host decisions must not be visible to an admission of the same pass. READ-ROOM-D
    serves from check_hicache_events (before the pass's admission); until the NEXT broadcast TP0 and the worker
    differ in ``backuped`` -- the switch of _evict_device_leaf (write_backup claim vs _evict_to_host)."""
    rr = _rr()
    me = threading.current_thread().name
    spans = [(1, list(range(0, 8)))]
    root, stem_of = _files(tmp_path, spans)
    tp0 = _d_tree(spans, None, "tp0")
    wk = _d_tree(spans, None, "worker")
    roles.now[me] = "host"
    rr.form_a_spill(tp0, _Pool(None, stem_of, root), 1)
    # same pass: admission/device eviction runs here on every rank, no broadcast yet
    assert tp0._t_nodes[1].backuped == wk._t_nodes[1].backuped, (
        tp0._t_nodes[1].backuped, wk._t_nodes[1].backuped)


def test_RD4_no_over_spill_while_the_workers_still_hold(tmp_path, roles, evict_mock):
    """Sleep drain (no tp<-reqs broadcast): the read re-posts (new seq every ~1.5 s); TP0 serves every new seq,
    the header deficit does not shrink (the workers still hold every slot) -> TP0 spills again and again."""
    rr = _rr()
    me = threading.current_thread().name
    stems = [f"r{i}" for i in range(4)]
    be, arena = _setup(tmp_path, stems, slots=64)
    d_slots = _fill_arena_with_tree_pages(arena, 64, tag="dpage")
    arena.ref_slots(d_slots, +1)          # the worker's reference
    spans = [(1 + 1000 * i, d_slots[8 * i:8 * i + 8]) for i in range(8)]
    root, _ = _files(tmp_path, [])
    stem_of = {s: f"dpage{i}" for i, s in enumerate(d_slots)}
    for st in stem_of.values():
        open(os.path.join(root, st + ".bin"), "wb").write(b"\x01")
    tp0 = _d_tree(spans, arena, "tp0")
    pool = _Pool(arena, stem_of, root)
    roles.now[me] = "host"
    tp0.W3_SPILL_MIN_PAGES = 8
    got = [rr.serve(tp0, pool, arena, 8) for _ in range(4)]   # four re-posts, no broadcast in between
    assert sum(got) <= 8, (got, "TP0 released this much for a need of 8 while 0 slots came free")
