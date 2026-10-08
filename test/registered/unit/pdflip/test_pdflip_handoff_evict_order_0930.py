"""#248e (Klasse I): the hand-off needed next was evicted by the L3 fills of the
same wake.

THE METAL (y3u, boot ...0930_002717, D TP0 00:35:40-47, pdflip-12-24, 245827
tokens = 3841 pages; D tail 3.88 s):

* 00:35:40 P reset: ``handoff_kept=3841/3841 handoff_rids=1 park_kept=2644/3148
  park_rids=2`` -- 3841 + 2644 = 6485 = every slot of the KV arena
  (``arena-786432.bin slots=6485``), each kept by ORDER, none by reference.
* 00:35:41 ``#248 PARK-DEMOTE rid=pdflip-12-24 role=handoff pool=kv pages=3841
  ... l3_pages=3841 absent=0`` -- all 3841 pages COMPLETE in L2, copied to L3.
* 00:35:43 the wake reads ``['pdflip-0-5', 'pdflip-12-23', 'pdflip-12-24', ...]``
  (hold order). The fills of the two parks (47 and 178 pages missing in L2)
  find the arena full and call ``HiCacheFile._arena_evict_to_disk(arena,
  max(256, n))``, whose keep list held only the pins: ``ARENA-EVICT want=256``
  on TP0 (n=4, 5), TP1 (n=1, 2), TP2 (n=1, 2) -- 6 x 256 = 1536 slots in slot
  order. No other free between the demote and the read (no ARENA-DROP, no
  claim_room; P slept since 00:35:40).
* 12-24's first page is gone: ``#1436 ARENA-GET MISS keys=1024
  first_stem=848f4b... find=(slot=-1,state=0)`` (TP2), ``(slot=3535,state=1)``
  on TP0 -- TP1's L3 fill re-claimed it into slot 3535, inside the run TP1's
  own evict had just freed (``first=[3435 ...]``, 3435 + 256 = 3691 = TP0's
  next run). ``PDFLIP-SHORT-READ HELD rid=pdflip-12-24 completed=0 of 3841
  hit=3841``, HOLD-REFETCH, #988 LOADBACK 00:35:46.

THE FIX (``handoff_pending.evict_ordered``): the clock evict of the L3 fill
keeps the hand-off / park order -- (i) unkept pages up to ``want``; (ii) kept
pages only as many as the fill lacks, the rid read LAST first, each chain from
its TAIL; the claim's stages ii/iii use the same order. The wake notes the hold
order (``park_l3.note_hold_order``).

Hermetic, CPU: real C arena (gcc), the real KV pool claim, the real
HiCacheFile fill / clock evict / L3 copy test against a temp directory.
"""
from __future__ import annotations

import logging
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from flliper.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from flliper.srt.pdflip import handoff as ho  # noqa: E402
from flliper.srt.pdflip import handoff_pending as hp  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")

P = 4           # tokens per page
SLOTS = 12      # 6485 on the metal
SB = 256
S = 5
HANDOFF = "pdflip-12-24"   # 3841 pages on the metal, 8 here
PARK = "pdflip-12-23"      # 1152 pages on the metal, 6 here (2 only in L3)
H_CHAIN = [f"h{i}" for i in range(8)]
A_CHAIN = [f"a{i}" for i in range(6)]


class _Store:
    def __init__(self, root):
        self.root = root
        self.evictor = types.SimpleNamespace(reserve=lambda *a, **k: True,
                                             commit=lambda stem: None, abort=lambda stem: None)

    def _get_suffixed_key(self, k):
        return f"{k}.sfx"

    def _path(self, stem):
        return os.path.join(self.root, stem + ".bin")

    def _stat_stems(self, stems):
        return {s: os.path.getsize(self._path(s)) for s in stems if os.path.exists(self._path(s))}


def _hicache(store):
    """The REAL clock evict (``_arena_evict_to_disk``) -- not stubbed."""
    be = object.__new__(HiCacheFile)
    be._stat_stems = store._stat_stems
    be._sharded_path = store._path
    be._existing_path = store._path
    be._ensure_shard_dir = lambda path: os.makedirs(os.path.dirname(path), exist_ok=True)
    be._evictor = store.evictor
    be._key_geom = {"is_mla_model": False}
    return be


def _kv_pool(arena, store):
    pool = object.__new__(ap.ArenaMHAHostPool)
    pool._arena_page_tokens = P
    pool.staging_rows = S
    pool.arena = arena
    pool.arena_tokens = SLOTS * P
    pool.arena_slots = SLOTS
    pool.row_slot = None
    pool._page_bytes = SB
    pool._backend = store
    pool._pending = {}
    pool._pending_mask = torch.zeros(SLOTS, dtype=torch.bool)
    pool._pending_gen = torch.zeros(SLOTS, dtype=torch.int64)
    pool._pending_fresh = torch.zeros(SLOTS, dtype=torch.bool)
    bind = getattr(hp, "bind_arena", None)  # absent on the base
    if bind is not None:
        bind(pool, arena)
    return pool


def _publish(arena, stems, fill):
    for i, stem in enumerate(stems):
        (s, st, g), = arena.claim_slots([stem], [SB])
        assert st == 0, (stem, st)
        arena.slot_view(s, SB)[:] = bytes([fill + i]) * SB
        assert arena.complete_slots([s], [g], [(0, SB)]) == [1]


def _to_l3(store, stems, fill):
    for i, stem in enumerate(stems):
        with open(store._path(stem), "wb") as f:
            f.write(bytes([fill + i]) * SB)


def _read_order(rids):
    note = getattr(hp, "note_read_order", None)  # absent on the base
    if note is not None:
        note(rids)


def _states(arena, stems):
    return [int(st) for _, st in arena.find_slots(stems)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_QUEUE_REFS", "1")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path / "arena"))
    monkeypatch.setenv("FLLIPER_PDFLIP_HANDOFF", "1")
    (tmp_path / "l3").mkdir()
    kv = ShmArena(str(tmp_path / "arena-kv.bin"), SB, SLOTS)
    store = _Store(str(tmp_path / "l3"))
    pool = _kv_pool(kv, store)
    # P publishes the hand-off (the whole chain COMPLETE in L2) and marks it
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    h = pool._stems(H_CHAIN)
    _publish(kv, h, fill=0x10)
    assert ho.write(HANDOFF, list(range(len(H_CHAIN) * P)), H_CHAIN)
    assert hp.mark(HANDOFF, len(H_CHAIN), P)
    # D: the park's head in L2, its last two pages only in L3 (P's stage-ii
    # claims took them during the hand-off's prefill)
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    a = pool._stems(A_CHAIN)
    _publish(kv, a[:4], fill=0x40)
    assert hp.mark_park(PARK, A_CHAIN, P)
    # the demoter's copies: every kept page has one in L3
    _to_l3(store, h, 0x10)
    _to_l3(store, a, 0x40)
    assert kv.stats()["complete"] == SLOTS, "the arena is full, every slot kept by order"
    yield types.SimpleNamespace(kv=kv, store=store, pool=pool, be=_hicache(store), h=h, a=a, mp=monkeypatch)
    _read_order([])
    getattr(hp, "_ARENA_POOLS", {}).clear()
    kv.close()


def test_metal_shape_the_park_fill_keeps_the_handoff_head(env, caplog):
    """RED on the base: the wake reads the park first (hold order), its fill
    needs 2 slots, the clock evict takes want=256 in slot order -- every slot,
    the hand-off's page 0 first. With #248e: exactly 2 pages go, the
    hand-off's TAIL (h7, h6); its head and the park's head stay COMPLETE."""
    _read_order([PARK, HANDOFF, "pdflip-13-25"])
    with caplog.at_level(logging.INFO):
        out = HiCacheFile.arena_fill_from_disk(env.be, env.kv, env.a[4:], SB, prefix=True)
    assert all(o is not None for o in out), f"the park's fill got its slots: {out}"
    assert _states(env.kv, env.h[:6]) == [2] * 6, "the hand-off's head stays in L2 (metal: page 0 gone)"
    assert _states(env.kv, env.h[6:]) != [2, 2], "its tail made the room"
    assert _states(env.kv, env.a) == [2] * 6, "the park is whole in L2"
    for slot, stem in zip(out, env.a[4:]):
        assert bytes(env.kv.slot_view(slot, SB)) == open(env.store._path(stem), "rb").read()
    msgs = [r.getMessage() for r in caplog.records if "#248e ORDERED-EVICT" in r.getMessage()]
    assert msgs and "site=l3fill" in msgs[-1] and f"{HANDOFF}:rank=1,pages=2,lowest_page=6" in msgs[-1], msgs


def test_the_hold_order_decides_whose_tail_goes(env):
    """The same arena, the hand-off read FIRST: the park's own tail makes the
    room, the hand-off stays whole -- the order is the hold's, not the
    role's."""
    _read_order([HANDOFF, PARK])
    out = HiCacheFile.arena_fill_from_disk(env.be, env.kv, env.a[4:], SB, prefix=True)
    assert _states(env.kv, env.h) == [2] * 8, "the hand-off read first keeps every page"
    assert _states(env.kv, env.a[:2]) == [2, 2], "the park's head stays"
    assert out  # the park's own tail (a3, a2) made the room: a swap, never the head


def test_the_claim_stage_ii_takes_the_tail_of_the_last_read(env):
    """RED on the base: a claim (D's retain, P's chunk) in the full arena --
    stage ii takes kept pages WITH an L3 copy in slot order (h0, h1: the
    hand-off's head). With #248e: in hold order, from the tail."""
    _read_order([PARK, HANDOFF])
    ids = env.pool.alloc_write(["new0", "new1"])
    assert ids is not None, "the claim found room"
    assert _states(env.kv, env.h[:6]) == [2] * 6, "the hand-off's head stays"
    assert _states(env.kv, env.a[:4]) == [2] * 4, "the park read first stays"


def test_victim_order_last_read_first_tail_first_shared_key_with_the_earlier():
    keys = np.array([10, 11, 12, 20, 21, 30, 30], dtype=np.uint64)
    rid_ix = np.array([0, 0, 0, 1, 1, 2, 0], dtype=np.int64)
    page = np.array([0, 1, 2, 0, 1, 0, 3], dtype=np.int64)
    # rid 0 = "a" (read first), 1 = "b" (read second), 2 = "x" (not read)
    order = np.argsort(keys, kind="stable")
    keep = hp.Keep(keys[order], rid_ix[order], page[order], ["a", "b", "x"], [])
    hp.note_read_order(["a", "b"])
    try:
        got = [int(keep.keys[i]) for i in hp.victim_order(keep)]
    finally:
        hp.note_read_order([])
    # "x" is not read at this wake (goes first) -- but its only key 30 is also
    # page 3 of "a", so it ranks with "a"; then "b" from its tail; "a" last,
    # tail first
    assert got == [21, 20, 30, 12, 11, 10], got


def test_no_bound_pool_is_the_pins_only_clock(tmp_path):
    """No keep order for an arena (P's hermetic arenas, the mamba blob without
    a bound pool): the clock takes ``want`` in slot order, as before.
    EVICT-KEEP: 16 slots, so the eighth-of-the-arena cap (2) is not the bound."""
    kv = ShmArena(str(tmp_path / "plain.bin"), SB, 16)
    root = tmp_path / "l3"
    root.mkdir()
    store = _Store(str(root))
    try:
        _publish(kv, ["p0", "p1", "p2", "p3"], fill=1)
        _to_l3(store, ["p0", "p1", "p2", "p3"], 1)
        be = _hicache(store)
        assert HiCacheFile._arena_evict_to_disk(be, kv, 2) == 0  # all on disk: nothing written
        assert _states(kv, ["p0", "p1", "p2", "p3"]) == [0, 0, 2, 2]
    finally:
        kv.close()


def test_the_wake_notes_the_hold_order_group_d_only(env):
    from flliper.srt.pdflip import park_l3

    hold = [types.SimpleNamespace(rid=r) for r in ("pdflip-0-5", PARK, HANDOFF)]
    park_l3.note_hold_order(hold)
    assert hp._READ_ORDER == ["pdflip-0-5", PARK, HANDOFF]
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    park_l3.note_hold_order(hold[:1])
    assert hp._READ_ORDER == ["pdflip-0-5", PARK, HANDOFF], "group P never writes the order"
