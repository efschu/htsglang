"""QS: a KV page reaches L3 only with its QSA index (NF y4b, weg2-52-142).

Metal (y4b 0114232648, P PP0 04:05:26): ``#1028B FETCH CAP n=5: kv=1500
claimed=0 lost=1500 caps={qsa_indexer: 359}`` -> ``#1035c ZERO-ANSWER
cause=CAPPED ... by=mamba``: the store held the shared prefix's 1500 KV pages
(96000 tokens, the fork) but its QSA index for 359; no anchor below 359, P
prefilled all 97088 tokens -- while ``P-FORK-CUT TOLD fork=96000 src=store``
told every stage the store held the fork. D's own coupling lines counted
``qsa_absent`` 685 times (590 after a #248 PARK-DEMOTE, 95 after a claim
room), P 57: the coupling ran AFTER the KV write and only counted the
missing index; the L3 write-behind wrote KV with no look at it at all.

Pinned (the real C arenas, the real L3P store and probe; RED on 768cb0b257):
  1. every writer -- #248 park demote, #257 d claim room / evict clock, the
     L3 write-behind -- puts a KV page on disk only together with its QSA
     index; a page whose index is in neither tier stays out of L3 (both
     absent), counted and named ``L3-SIDECAR-COUPLE writer=...``;
  2. the P fork stands on what the fetch can claim: the KV prefix every
     ALL_PAGES pool holds too (``all_pages_uncapped``), not the raw KV prefix.
"""
from __future__ import annotations

import logging
import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.canonical_page_store import CanonicalExtentWindow  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import (  # noqa: E402
    HiCacheFile,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

KV_TOTAL = 64
Q_TOTAL = 32
SFX = "_NF_898fe1bf"
PAGES = 8
GAP = 3          # the page whose QSA index is gone (52-142: 359)


class _KVPage:
    pass


def _backend(root, l3idx_path, arena_dir, monkeypatch):
    from sglang.srt.mem_cache.storage.file.l3_index import L3Index
    from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

    be = object.__new__(HiCacheFile)
    be.file_path = str(root)
    be._known_shards = set()
    be._legacy_flat = False
    be._key_geom = {"is_mla_model": False}
    be.metadata_cache = None
    be.dcp_owner_mode = False
    be.canonical_kv_page = _KVPage()
    be._canonical_kv_extents = CanonicalExtentWindow(KV_TOTAL, ((0, KV_TOTAL),))
    be.canonical_qsa_page = CanonicalExtentWindow(Q_TOTAL, ((0, Q_TOTAL),), label="qsa")
    be.canonical_mamba_blob = None
    be.canonical_draft_page = None
    be.kv_config_suffix = SFX
    be._kv_config_suffix_is_group_wide = True
    be.config_suffix = SFX + "_0_1"
    be._config_suffix_is_group_wide = False
    be._canonical_probe_mismatch = lambda: None
    be._evictor = LRUFileEvictor(
        str(root), SFX, tp_rank=0, writes_shared_keys=False,
        path_for_stem=be._existing_path, iter_existing=be._iter_existing_files,
    )
    idx = L3Index(str(l3idx_path), cap=1 << 12)
    be._l3idx = idx
    be._l3idx_tried = True
    be._evictor.l3_index = idx
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(arena_dir))
    arena_dir.mkdir(parents=True, exist_ok=True)
    be._arenas = {
        KV_TOTAL: ShmArena(str(arena_dir / f"arena-{KV_TOTAL}.bin"), KV_TOTAL, 32),
        Q_TOTAL: ShmArena(str(arena_dir / f"arena-{Q_TOTAL}.bin"), Q_TOTAL, 32),
    }
    return be, idx


def _boot(tmp_path, monkeypatch, name):
    monkeypatch.setenv("SGLANG_WEG2_L3_PERSIST", "1")
    root = tmp_path / "store" / "l3-nextflash-identity"
    root.mkdir(parents=True, exist_ok=True)
    (root / "L3_IDENTITY.json").write_text("{}")
    be, idx = _backend(root, tmp_path / f"{name}-l3idx.bin", tmp_path / f"shm-{name}", monkeypatch)
    be._l3p_seed_index(idx)
    return be


def _hashes(tag):
    return [f"{tag}{i:02d}" + "ab" * 30 for i in range(PAGES)]


def _put(arena, stem, total, fill):
    pay = torch.full((total,), fill & 0xFF, dtype=torch.uint8)
    assert arena.write([stem], [total], [((0, total),)], [pay.data_ptr()]) == [1]


def _prefill_with_gap(be, hashes):
    """The prefix in L2: every KV page COMPLETE, the QSA index of page GAP
    gone from L2 and never written to L3 (the y4b state)."""
    for i, h in enumerate(hashes):
        _put(be._arenas[KV_TOTAL], be._get_suffixed_key(h), KV_TOTAL, 0x40 + i)
        if i != GAP:
            _put(be._arenas[Q_TOTAL], be._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}"),
                 Q_TOTAL, 0x80 + i)


def _on_disk(be, stem) -> bool:
    return bool(be._stat_stems([stem]))


def _pairs_on_disk(be, hashes):
    """(KV pages on disk, KV pages on disk WITHOUT their QSA index)."""
    kv, alone = 0, []
    for i, h in enumerate(hashes):
        if _on_disk(be, be._get_suffixed_key(h)):
            kv += 1
            if not _on_disk(be, be._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}")):
                alone.append(i)
    return kv, alone


@pytest.fixture(autouse=True)
def _awake_gate():
    try:
        from sglang.srt.mem_cache import l3_write_behind as gate
    except ImportError:
        yield
        return
    gate._reset_for_tests()
    yield
    gate._reset_for_tests()


def test_park_demote_never_puts_a_kv_page_on_disk_without_its_qsa_index(tmp_path, monkeypatch, caplog):
    """RED on 768cb0b257: the demoter copied all 8 KV pages, page 3 without
    its index (``qsa_absent=1`` after the fact). GREEN: page 3 stays out of L3,
    the marker names the writer."""
    monkeypatch.setenv("SGLANG_WEG2_L3_WRITE_BEHIND_S", "0")
    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("a")
    _prefill_with_gap(be, hs)
    with caplog.at_level(logging.INFO):
        r = be.arena_copy_to_disk(be._arenas[KV_TOTAL], [be._get_suffixed_key(h) for h in hs])
    kv, alone = _pairs_on_disk(be, hs)
    assert alone == [], f"KV on disk without its QSA index at pages {alone}"
    assert kv == PAGES - 1 and r.get("unpaired") == 1
    assert any("L3-SIDECAR-COUPLE writer=park_demote" in m and "kv_unpaired=1" in m
               for m in caplog.messages)


@pytest.mark.parametrize("writer", ["claim_room", "evict_clock"])
def test_an_evicting_writer_drops_the_unpaired_kv_page_with_its_index(tmp_path, monkeypatch, writer):
    """#257 d claim room and the arena clock: the page leaves L2; without its
    index it does not go to L3 either -- counted apart from ``lost`` (the W3
    spill releases rows on lost == 0). RED on 768cb0b257: 8 KV written."""
    monkeypatch.setenv("SGLANG_WEG2_L3_WRITE_BEHIND_S", "0")
    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("b")
    _prefill_with_gap(be, hs)
    kv = be._arenas[KV_TOTAL]
    if writer == "evict_clock":
        assert be._arena_evict_to_disk(kv, 64) == PAGES - 1
    else:
        cands = kv.evict_candidates(64)
        assert len(cands) == PAGES
        sec = be.arena_secure_to_disk(kv, cands)
        kv.free_slots([c[0] for c in cands])
        assert sec["written"] == PAGES - 1 and sec["lost"] == 0 and sec["unpaired"] == 1
    n, alone = _pairs_on_disk(be, hs)
    assert alone == [] and n == PAGES - 1


def test_write_behind_writes_the_index_first_and_never_kv_alone(tmp_path, monkeypatch):
    """One pass: the QSA arena first, then every KV page whose index is on
    disk; page 3 (index in neither tier) is held back and asked again. RED on
    768cb0b257: all 8 KV pages written, page 3 alone."""
    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("c")
    _prefill_with_gap(be, hs)
    tot = be.l3_write_behind_pass()
    kv, alone = _pairs_on_disk(be, hs)
    assert alone == [] and kv == PAGES - 1
    assert tot["written"] == (PAGES - 1) + (PAGES - 1) and tot.get("unpaired") == 1
    # budget-cut pass: a KV page never overtakes its index
    be2 = _boot(tmp_path, monkeypatch, "boot2")
    hs2 = _hashes("d")
    for i, h in enumerate(hs2):
        _put(be2._arenas[KV_TOTAL], be2._get_suffixed_key(h), KV_TOTAL, 0x10 + i)
        _put(be2._arenas[Q_TOTAL], be2._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}"), Q_TOTAL, 0x20 + i)
    be2.l3_write_behind_pass(budget_bytes=3 * Q_TOTAL)
    _kv2, alone2 = _pairs_on_disk(be2, hs2)
    assert alone2 == []


def test_the_fork_stands_on_what_the_fetch_can_claim(tmp_path, monkeypatch):
    """y4b: KV on disk for 8 pages, QSA index for the leading 3 (the state the
    base writers leave). The probe's MIN caps the claim at 3; the fork must
    say 3 too (``P-FORK-CUT TOLD fork=96000`` against ``caps={qsa: 359}``)."""
    from sglang.srt.weg2 import p_fork_cut

    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("e")
    for i, h in enumerate(hs):   # write the files directly: KV all, QSA leading GAP
        _put(be._arenas[KV_TOTAL], be._get_suffixed_key(h), KV_TOTAL, 0x40 + i)
    be._arena_copy_pages_to_disk(be._arenas[KV_TOTAL], [be._get_suffixed_key(h) for h in hs])
    for i, h in enumerate(hs[:GAP]):
        _put(be._arenas[Q_TOTAL], be._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}"), Q_TOTAL, 1)
    be._arena_copy_pages_to_disk(
        be._arenas[Q_TOTAL], [be._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}") for h in hs[:GAP]])
    res = be.batch_exists_v2(
        hs, [PoolTransfer(name=PoolName.QSA_INDEXER, keys=[], hit_policy=PoolHitPolicy.ALL_PAGES)])
    assert (res.kv_uncapped, res.kv_hit_pages) == (PAGES, GAP)
    assert res.all_pages_uncapped == GAP
    assert p_fork_cut.store_fork_pages(res.kv_uncapped, res.all_pages_uncapped) == GAP
    assert p_fork_cut.store_fork_pages(1500, None) == 1500   # a v1 probe: unchanged
