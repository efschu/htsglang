"""#1427r (z30n D, Form A token cut 0,32,32): a claim given up by the rank
that took it FIRST must not free a slot other writers joined.

Under the token cut every rank of group D claims each page: the attention
host (share 0, no extents) and the two KV owners. Whoever calls
``arena_claim`` first gets status 0 ("fresh"), the others join (status 1).
``ArenaMHAHostPool.free()`` / ``abort_write()`` of a pending write freed the
fresh claims WHOLE (``arena_free_slots``): the generation moved under the
joined owners, their merge came back 3 -- ``#1427 ARENA-COMPLETE LOST ...
(recycled under the writer)``, 75 lines on TP1/TP2 of the z30n D log, the
same slots on both, never on TP0 -- and a page the owners had ALREADY
completed was freed without a line. The D decode tail the park wrote never
became readable (store_hit stopped at P's hand-off, weg2-36-56 77184 of
77568) and the wake re-computed 3.7-6.8k tokens per request.

Hermetic: the real C arena (one mapping per rank would share the same
header; one mapping stands for all three ranks here)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.pool_host import arena_pool  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

SLOT = 4096
HALF = SLOT // 2


@pytest.fixture
def arena(tmp_path):
    a = ShmArena(str(tmp_path / "arena-kv.bin"), SLOT, 8)
    yield a
    a.close()


def _form_a_claims(arena, stem):
    """TP0 (share 0) claims first, TP1 and TP2 (the owners) join."""
    (s0, st0, g0), = arena.claim_slots([stem], [SLOT])
    (s1, st1, g1), = arena.claim_slots([stem], [SLOT])
    (s2, st2, g2), = arena.claim_slots([stem], [SLOT])
    assert (st0, st1, st2) == (0, 1, 1) and s0 == s1 == s2
    return s0, (g0, g1, g2)


def test_metal_shape_host_gives_up_its_fresh_claim_and_the_owners_merge_lost(arena):
    """RED on c963e47631: TP0 gives its pending fresh claim up (free before
    its ack) between TP1's and TP2's merge -- TP2's merge is refused (3,
    'recycled under the writer'), the page never completes."""
    s, (g0, g1, g2) = _form_a_claims(arena, "tail-page-1206")
    assert arena.complete_slots([s], [g1], [(0, HALF)]) == [0]         # TP1: its rows
    arena_pool._release_fresh(arena, [s], [g0], "free_pending")          # TP0 gives up
    assert arena.complete_slots([s], [g2], [(HALF, HALF)]) == [1], (
        "TP2's merge completes the page; before #1427r it came back 3 (LOST)")
    assert arena.find_slots(["tail-page-1206"]) == [(s, 2)], "the page is readable"


def test_metal_shape_a_page_the_owners_completed_is_not_freed_by_the_host(arena):
    """RED on c963e47631: both owners merged, the page is COMPLETE; TP0's
    late give-up of its fresh claim freed it without a line."""
    s, (g0, g1, g2) = _form_a_claims(arena, "tail-page-1207")
    assert arena.complete_slots([s], [g1], [(0, HALF)]) == [0]
    assert arena.complete_slots([s], [g2], [(HALF, HALF)]) == [1]
    arena_pool._release_fresh(arena, [s], [g0], "abort_write")
    assert arena.find_slots(["tail-page-1207"]) == [(s, 2)]


def test_a_sole_claimant_still_frees_its_claim(arena):
    """No other writer joined: the give-up frees the slot (the old behaviour
    where it was right -- no leak of CLAIMED slots in the KV arena)."""
    (s, st, g), = arena.claim_slots(["alone"], [SLOT])
    assert st == 0
    assert arena.release_claims([s], [g]) == [0]
    assert arena.find_slots(["alone"]) == [(-1, 0)]
    assert arena.stats()["claimed"] == 0
    (s2, st2, _), = arena.claim_slots(["alone"], [SLOT])
    assert st2 == 0, "the key gets a fresh slot again"


def test_a_moved_generation_is_skipped(arena):
    (s, _, g), = arena.claim_slots(["q"], [SLOT])
    arena.free_slots([s])
    assert arena.release_claims([s], [g]) == [2]


def test_the_kept_slot_is_named_once_with_its_site(arena, caplog):
    s, (g0, g1, _g2) = _form_a_claims(arena, "named")
    arena.complete_slots([s], [g1], [(0, HALF)])
    with caplog.at_level("WARNING"):
        arena_pool._release_fresh(arena, [s], [g0], "free_pending")
    lines = [r.getMessage() for r in caplog.records if "#1427r CLAIM-RELEASE" in r.getMessage()]
    assert lines and "site=free_pending" in lines[0] and "kept=1" in lines[0] and "freed=0" in lines[0]


def test_a_fake_arena_without_release_claims_frees_as_before():
    freed = []

    class Fake:
        def free_slots(self, slots):
            freed.extend(slots)

    arena_pool._release_fresh(Fake(), [3, 4], [1, 1], "abort_write")
    assert freed == [3, 4]


def test_every_fresh_give_up_site_of_the_kv_pool_goes_through_release():
    """Wiring: free(), abort_write() and both claim-refusal paths of the KV
    arena pool release fresh claims through _release_fresh -- none frees a
    fresh claim whole any more."""
    import inspect

    src = inspect.getsource(arena_pool.ArenaMHAHostPool)
    for site in ('"free_pending"', '"abort_write"', '"claim_refused"'):
        assert f"_release_fresh(" in src and site in src, site
    body = inspect.getsource(arena_pool.ArenaMHAHostPool.abort_write)
    assert "free_slots(fresh)" not in body
    body = inspect.getsource(arena_pool.ArenaMHAHostPool.free)
    assert "free_slots(fresh)" not in body
