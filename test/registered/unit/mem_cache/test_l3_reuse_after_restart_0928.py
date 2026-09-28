"""L3-REUSE 0928: after a restart the first requests resume from the persistent L3.

NF boot rc12z13 (unified 7b2c6ee5ef, P PP0 09:33:33, the first request after
the boot, the store reattached with 10135 files):

    #1472 READ-TRACE n=1 asked=512 readable=399 first_missing=eb5869b2... why=no-file
    #1028B FETCH CAP n=1: kv=399 claimed=47 lost=352
        caps={mamba: 399, qsa_indexer: 47}
    Prefill ... #new-token: 16384, #cached-token: 0   (ten chunks in a row)

On disk: 8104 KV pages, 2048 QSA index pages (= 8 x 256, the eight
``ARENA-EVICT want=256`` rounds of the QSA arena), 276 mamba blobs.

Two roots, one per number:

(A) ``claimed=47``: the QSA index page lives in its OWN arena and reached L3
    only when that arena ran full. The paths that give a KV page its L3 copy
    -- the #248 park/hand-off demoter (``arena_copy_to_disk``) and the #257 d
    claim room (``arena_secure_to_disk``) -- copied the KV page alone. The
    probe takes the MINIMUM over the pools (QSA_INDEXER is ALL_PAGES).
(B) ``readable=399 of 512``: the L2 arena lives in /dev/shm under the boot
    tag; L3 saw a page only when it LEFT L2, so every page still in L2 at the
    end of a boot died with it.

Hermetic: the real C arenas on temp files (one per width, as ``_arena_for``
keeps them), the real L3P store (``LRUFileEvictor``, #1459 ``L3Index``), the
real ``HiCacheFile.batch_exists_v2`` probe of a restarted process over the
same directory with EMPTY arenas (the new boot's /dev/shm)."""
from __future__ import annotations

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

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

KV_TOTAL = 64
Q_TOTAL = 32
SFX = "_NF_898fe1bf"
PAGES = 8


class _KVPage:
    """``canonical_kv_page`` only has to exist (key shape); its spec is read
    by the window rebind, not by the probe."""


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
    """One boot of the group: the persistent store directory is shared by
    every boot, the arena directory (/dev/shm/<boot tag>) is this boot's."""
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


def _prefill(be, hashes):
    """What the prefill leaves in L2: every KV page and its QSA index page
    COMPLETE in their arenas (direct write + the #106S sidecar write)."""
    for i, h in enumerate(hashes):
        _put(be._arenas[KV_TOTAL], be._get_suffixed_key(h), KV_TOTAL, 0x40 + i)
        _put(be._arenas[Q_TOTAL], be._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}"), Q_TOTAL, 0x80 + i)


def _probe_after_restart(tmp_path, monkeypatch, hashes):
    """The next boot's first request: fresh arenas, same store, the real
    probe with the QSA sidecar as the ALL_PAGES pool the NF stack registers."""
    be2 = _boot(tmp_path, monkeypatch, "boot2")
    res = be2.batch_exists_v2(
        hashes,
        [PoolTransfer(name=PoolName.QSA_INDEXER, keys=[], hit_policy=PoolHitPolicy.ALL_PAGES)],
    )
    return be2, res


def test_a_demoted_kv_page_carries_its_qsa_index_to_l3(tmp_path, monkeypatch):
    """(A) RED on 7b2c6ee5ef: the #248 demoter's copy puts the KV pages on
    disk and leaves their QSA index in L2; after the restart the probe finds
    kv=8 and caps the claim at 0 (the rc12z13 shape: kv=399 claimed=47).
    GREEN: the claim is the whole KV prefix."""
    monkeypatch.setenv("SGLANG_WEG2_L3_WRITE_BEHIND_S", "0")
    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("a")
    _prefill(be, hs)
    r = be.arena_copy_to_disk(be._arenas[KV_TOTAL], [be._get_suffixed_key(h) for h in hs])
    assert r["written"] == PAGES
    _be2, res = _probe_after_restart(tmp_path, monkeypatch, hs)
    assert res.kv_uncapped == PAGES, "the KV pages are on disk"
    assert res.kv_hit_pages == PAGES, (
        f"KV on disk without its QSA index: claim capped at {res.kv_hit_pages} of "
        f"{res.kv_uncapped} (zero_capped={res.zero_capped_pools})"
    )


def test_a_claim_room_eviction_carries_the_qsa_index_to_l3(tmp_path, monkeypatch):
    """(A) the #257 d claim room: evicting a KV page writes it to L3 first --
    and now its QSA index too. RED on 7b2c6ee5ef: claim 0 after the restart."""
    monkeypatch.setenv("SGLANG_WEG2_L3_WRITE_BEHIND_S", "0")
    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("b")
    _prefill(be, hs)
    kv = be._arenas[KV_TOTAL]
    cands = kv.evict_candidates(64)
    assert len(cands) == PAGES
    sec = be.arena_secure_to_disk(kv, cands)
    kv.free_slots([c[0] for c in cands])
    assert sec["written"] == PAGES and sec["lost"] == 0
    _be2, res = _probe_after_restart(tmp_path, monkeypatch, hs)
    assert (res.kv_uncapped, res.kv_hit_pages) == (PAGES, PAGES)


def test_b_pages_still_in_l2_at_the_end_of_a_boot_resume_from_l3(tmp_path, monkeypatch):
    """(B) RED on 7b2c6ee5ef: nothing ever left L2, so nothing reached L3 --
    the restart finds kv=0 (the rc12z13 tail: readable=399 of asked=512,
    why=no-file). GREEN: one write-behind pass copies every COMPLETE arena
    page (KV and QSA index) to the persistent store without freeing it, and
    the restarted probe claims the whole prompt; the bytes are the ones of
    the previous boot."""
    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("c")
    _prefill(be, hs)
    passer = getattr(be, "l3_write_behind_pass", None)
    if passer is not None:
        tot = passer()
        assert tot["written"] == 2 * PAGES and tot["pending"] == 0
        # L2 keeps them (a copy, not an eviction)
        assert all(s == 2 for s in be._arenas[KV_TOTAL].find_states(
            [be._get_suffixed_key(h) for h in hs]))
        # each page once: a second pass writes nothing
        assert passer()["written"] == 0
    be2, res = _probe_after_restart(tmp_path, monkeypatch, hs)
    assert (res.kv_uncapped, res.kv_hit_pages) == (PAGES, PAGES), (
        f"restart re-prefills: kv on disk {res.kv_uncapped}, claim {res.kv_hit_pages} of {PAGES}"
    )
    q_stem = be2._get_suffixed_key(f"{hs[3]}.{PoolName.QSA_INDEXER}")
    with open(be2._existing_path(q_stem), "rb") as f:
        assert f.read() == bytes([0x83]) * Q_TOTAL


def test_b_write_behind_respects_its_budget_and_catches_up(tmp_path, monkeypatch):
    """A pass copies at most its byte budget per arena; the rest is pending
    and the next pass takes it -- never a page twice."""
    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("d")
    _prefill(be, hs)
    first = be.l3_write_behind_pass(budget_bytes=3 * KV_TOTAL)
    assert first["written"] == 3 + 6  # 3 KV pages (64 B), 6 QSA pages (32 B)
    assert first["pending"] == (PAGES - 3) + (PAGES - 6)
    second = be.l3_write_behind_pass(budget_bytes=64 * KV_TOTAL)
    assert second["written"] == (PAGES - 3) + (PAGES - 6)
    assert be.l3_write_behind_pass()["written"] == 0


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


def test_b_census_and_pin_are_one_c_call_each(tmp_path, monkeypatch):
    """No per-slot Python in the thread: the COMPLETE census (slot,
    generation, key) and the pin come from arena.c; a pin under a stale key
    (the slot was reclaimed since the census) is refused and leaves the
    refcount untouched."""
    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("e")
    _prefill(be, hs)
    kv = be._arenas[KV_TOTAL]
    slots, gens, klo, khi = kv.complete_census()
    assert len(slots) == PAGES
    ok = kv.pin_complete(slots, klo, khi)
    assert ok.all()
    assert kv.slot_refs(slots.tolist()) == [1] * PAGES
    bad = klo.copy()
    bad[0] ^= 1
    kv.unpin(slots)
    ok2 = kv.pin_complete(slots[:2], bad[:2], khi[:2])
    assert ok2.tolist() == [False, True]
    assert kv.slot_refs(slots[:2].tolist()) == [0, 1]
    kv.unpin(slots[1:2])


def test_b_quiet_during_legs_sleep_and_dormancy(tmp_path, monkeypatch):
    """The flip owns the arena and the lanes: a pass writes nothing while a
    leg runs on this rank, after the sleep leg until the wake leg returned,
    and while the scheduler is dormant (W25) -- the existing leg bracket of
    ``_weg2_group_stop_on_leg_failure`` drives the gate, no clock."""
    from sglang.srt.managers.scheduler_components.weight_updater import (
        _weg2_group_stop_on_leg_failure,
    )
    from sglang.srt.mem_cache import l3_write_behind as gate

    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("f")
    _prefill(be, hs)
    seen = {}

    class _Sched:
        weg2_dormant = False

    class _Updater:
        scheduler = _Sched()

        def _weg2_leg_failed(self, msg, exc):
            pass

        @_weg2_group_stop_on_leg_failure
        def release_memory_occupation(self, recv_req):
            seen["in_leg"] = be.l3_write_behind_pass()
            return "released"

        @_weg2_group_stop_on_leg_failure
        def resume_memory_occupation(self, recv_req):
            seen["in_wake"] = be.l3_write_behind_pass()
            return "resumed"

    u = _Updater()
    assert u.release_memory_occupation(None) == "released"
    assert seen["in_leg"]["paused"] == "leg" and seen["in_leg"]["written"] == 0
    asleep = be.l3_write_behind_pass()
    assert asleep["paused"] == "asleep" and asleep["written"] == 0
    assert u.resume_memory_occupation(None) == "resumed"
    assert seen["in_wake"]["paused"] == "leg"
    u.scheduler.weg2_dormant = True
    assert be.l3_write_behind_pass()["paused"] == "dormant"
    u.scheduler.weg2_dormant = False
    awake = be.l3_write_behind_pass()
    assert awake["paused"] is None and awake["written"] == 2 * PAGES
    assert gate.quiet_reason() is None


def test_b_no_write_evict_ping_pong(tmp_path, monkeypatch):
    """A page the L3 cap evicts after the write-behind copied it is not
    written again while it sits in the same arena slot (each slot generation
    once); a slot that is reclaimed for a new page is."""
    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("g")
    _prefill(be, hs)
    assert be.l3_write_behind_pass()["written"] == 2 * PAGES
    victim = be._get_suffixed_key(hs[0])
    os.remove(be._existing_path(victim))
    be._l3idx.remove([victim])
    assert be.l3_write_behind_pass()["written"] == 0, "rewrote a page the cap evicted"
    kv = be._arenas[KV_TOTAL]
    (slot, _st), = kv.find_slots([victim])
    kv.free_slots([slot])
    fresh = "h00" + "cd" * 30
    _put(kv, be._get_suffixed_key(fresh), KV_TOTAL, 0x11)
    r = be.l3_write_behind_pass()
    assert r["written"] == 1 and os.path.exists(be._existing_path(be._get_suffixed_key(fresh)))


def test_b_a_refused_reservation_ends_the_arena_pass(tmp_path, monkeypatch):
    """The cap/min-free refusal is not retried page by page: the arena's pass
    ends at the first refusal, the rest stays pending for the next pass."""
    be = _boot(tmp_path, monkeypatch, "boot1")
    hs = _hashes("i")
    _prefill(be, hs)
    calls = []

    def _refuse(stem, nbytes, key="", **kw):
        calls.append(stem)
        return False

    be._evictor.reserve = _refuse
    r = be.l3_write_behind_pass()
    assert r["written"] == 0 and r["refused"] == 2
    assert len(calls) == 2, "one refused reservation per arena, not one per page"
    assert all(s == 2 for s in be._arenas[KV_TOTAL].find_states(
        [be._get_suffixed_key(h) for h in hs])), "L2 untouched"
    kv = be._arenas[KV_TOTAL]
    assert kv.slot_refs(kv.complete_census()[0].tolist()) == [0] * PAGES, "pins given back"


def test_b_write_behind_is_armed_only_on_the_persistent_store_owner(tmp_path, monkeypatch):
    """The thread starts only for a backend that attached (``__init__``
    arms it) to a PERSISTENT store as its index owner, and never with the
    tick at 0; a unit-test scaffold never starts one."""
    be = _boot(tmp_path, monkeypatch, "boot1")
    assert be._l3_write_behind_start() is False, "a scaffold backend is not armed"
    be._l3wb_armed = True
    monkeypatch.setenv("SGLANG_WEG2_L3_WRITE_BEHIND_S", "0")
    assert be._l3_write_behind_start() is False
    monkeypatch.setenv("SGLANG_WEG2_L3_WRITE_BEHIND_S", "3600")
    be._evictor._is_storage_owner = False
    assert be._l3_write_behind_start() is False, "PP1/PP2 hold no index"
    be._evictor._is_storage_owner = True
    monkeypatch.setenv("SGLANG_WEG2_L3_PERSIST", "0")
    assert be._l3_write_behind_start() is False, "not a persistent store"
    monkeypatch.setenv("SGLANG_WEG2_L3_PERSIST", "1")
    try:
        assert be._l3_write_behind_start() is True
        assert be._l3_write_behind_start() is False, "one thread per backend"
    finally:
        be._l3wb_stop.set()
        be._l3wb_thread.join(timeout=5)
