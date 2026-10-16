"""K15 x anchor0 (review K15 A6/B5): the spill victim's L3 copy must survive the L3 LRU.

A K15 spill (``_weg2_flush_spill_room`` -> ``arena_secure_to_disk(writer="flush_spill")``) frees the
victim's arena slot against its EXISTING L3 file (boot _215812: 300 of 301 FLUSH-SPILLs written=0).
Without anchor0's ``_l3_refresh_kept`` that file keeps its old recency and the L3 LRU (cap 150 GB,
EVICTION-BG 18x on D) can take it -- the victim's anchor gone from every tier. With anchor0 the copy
becomes MRU in the owner's LRU. This test runs only where both are present (the integration
desk/nf-h16s-int-1016); on K15 alone it skips, named."""
from __future__ import annotations

import importlib.util
import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402

pytestmark = [
    pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc"),
    pytest.mark.skipif(not hasattr(HiCacheFile, "_l3_refresh_kept"),
                       reason="anchor0 _l3_refresh_kept not on this tree (K15 alone)"),
]

_HERE = os.path.dirname(__file__)


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def _awake_gate():
    try:
        from sglang.srt.mem_cache import l3_write_behind as gate
    except ImportError:
        yield
        return
    gate._reset_for_tests()
    yield
    gate._reset_for_tests()


def test_flush_spill_victim_copy_becomes_mru(tmp_path, monkeypatch, _awake_gate):
    sc = _mod("_t_store_complete_x", "../managers/test_weg2_zero_anchor_store_complete_1015.py")
    qs, be, stems, twins, young = sc._parked_on_old_copies(tmp_path, monkeypatch)
    kv = be._arenas[qs.KV_TOTAL]
    cands = [c for c in kv.evict_candidates(64) if kv.slot_stem(c[0]) in set(stems)]
    sec = be.arena_secure_to_disk(kv, cands, writer="flush_spill")     # the K15 spill's own call
    assert sec["on_disk"] == qs.PAGES and sec["lost"] == 0
    assert sec.get("refreshed") == 2 * qs.PAGES
    assert set(sc._oldest(be, qs.PAGES)) == set(young), "the victim's copy is no longer the next L3 victim"
