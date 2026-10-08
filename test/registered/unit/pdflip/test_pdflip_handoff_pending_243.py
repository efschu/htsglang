"""#243 (rc12r 09271632, pdflip-12-39): a P hand-off waiting for its D seat has
no holder -- D finds it gone at admission.

THE METAL (tmp/r989/befund_243_pdflip-12-39.md).

* 16:53:40 P published the hand-off completely: END-ANCHOR ok on PP0/1/2,
  MAMBA-ARENA end_anchor=slot, TAIL-PUBLISH 76544 + 58 rows.
* 16:53:45 the flip reset released every tree reference (#1427 RESET-RELEASE
  released=3940, FULL sum=0). H81 held the six end anchors one flip, until
  P's wake at 16:54:50 ("the D phase that read them is over" -- it was not).
* 6/6 D seats were taken, so the request waited at the front until 16:57:10
  (D-ADMIT oldest_wait_s=255.3). Meanwhile the claims of other rids made room
  from the oldest unreferenced slots.
* The result at admission: ``#1028B FETCH CAP kv=1196 claimed=0 lost=1196``,
  ``anchors_in_range MAMBA: (0, -1)``, W31, a second P prefill, the client
  reset.

THE FIX (``pdflip.handoff_pending``). P marks the rid when it publishes. Every
claim that makes room passes over the marked rid's keys: the KV chain, plus
the end anchor (last two chain keys). Those go only when nothing else is
left, and then by name (HANDOFF-LOST, status file). D's group-uniform prefetch
termination, or the rid's end, removes the mark; so does the front's
``drop``. It is an ORDER, not a reference: a reference-pinned page (a park)
always wins.

Hermetic, CPU: real C arenas (arena.c, gcc) on temp files, the real KV pool
claim (``alloc_write`` -> ``_claim_np`` -> ``_evict_for_claim``), the real
mamba pool eviction (the same function), the real hand-off files.
"""
from __future__ import annotations

import ctypes
import logging
import os
import shutil
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from flliper.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from flliper.srt.pdflip import handoff as ho  # noqa: E402

try:  # absent on the base: the test then shows the loss itself
    from flliper.srt.pdflip import handoff_pending as hp
except ImportError:  # pragma: no cover - the base
    hp = None

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")

P = 4          # tokens per KV page
SLOTS = 8
SB = 256
S = 5          # staging rows
RID = "pdflip-12-39"
CHAIN = ["hA", "hB", "hC"]        # the hand-off's three pages (1196 on the metal)


class _Backend:
    def _get_suffixed_key(self, k):
        return f"{k}.sfx"

    def _log_key(self, pool, k):
        return f"{k}.mamba"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_QUEUE_REFS", "1")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path / "arena"))
    monkeypatch.setenv("FLLIPER_PDFLIP_HANDOFF", "1")
    kv = ShmArena(str(tmp_path / "arena-kv.bin"), SB, SLOTS)
    mb = ShmArena(str(tmp_path / "arena-mamba.bin"), SB, SLOTS)
    yield types.SimpleNamespace(kv=kv, mb=mb, mp=monkeypatch, tmp=tmp_path)
    kv.close()
    mb.close()


def _kv_pool(arena, page_tokens=P):
    pool = object.__new__(ap.ArenaMHAHostPool)
    pool._arena_page_tokens = page_tokens
    pool.staging_rows = S
    pool.arena = arena
    pool.arena_tokens = SLOTS * page_tokens
    pool.arena_slots = SLOTS
    pool.row_slot = None
    pool._page_bytes = SB
    pool._backend = _Backend()
    pool._pending = {}
    pool._pending_mask = torch.zeros(SLOTS, dtype=torch.bool)
    pool._pending_gen = torch.zeros(SLOTS, dtype=torch.int64)
    pool._pending_fresh = torch.zeros(SLOTS, dtype=torch.bool)
    return pool


def _mamba_pool(arena):
    pool = object.__new__(ArenaMambaPoolHost)
    pool.arena = arena
    pool._backend = _Backend()
    return pool


def _publish(arena, stems):
    out = {}
    for stem in stems:
        (s, st, g), = arena.claim_slots([stem], [SB])
        assert st == 0, (stem, st)
        assert arena.complete_slots([s], [g], [(0, SB)]) == [1]
        out[stem] = s
    return out


def _foreign_ref(arena, slot, delta=1):
    """Another process's reader reference (a park, a D tree): past this
    process's ledger."""
    c = (ctypes.c_int64 * 1)(slot)
    assert arena._lib.arena_ref_slots(arena._base, 1, c, delta) == 1


def _p_publishes(e, *, group="P"):
    """P finished the prefill: KV pages + end anchor COMPLETE in the arenas,
    the hand-off file written, the rid marked (the fix)."""
    kvp, mbp = _kv_pool(e.kv), _mamba_pool(e.mb)
    kv_slot = _publish(e.kv, kvp._stems(CHAIN))
    mb_slot = _publish(e.mb, mbp._stems(CHAIN[-1:]))
    assert ho.write(RID, list(range(len(CHAIN) * P + 3)), CHAIN)
    e.mp.setenv("FLLIPER_PDFLIP_GROUP", group)
    if hp is not None:
        hp.mark(RID, len(CHAIN), P)
    return kvp, mbp, kv_slot, mb_slot


def _h81_hold_then_wake(arena, slots):
    """H81: P's carrier reference on the end anchor across D's phase, given
    back at P's next wake."""
    arena.ref_slots(list(slots), +1)
    arena.ref_slots(list(slots), -1)


def _claimed(arena, stems):
    return sum(1 for _, st in arena.find_slots(stems) if int(st) == 2)


def test_metal_shape_seat_wait_across_p_wake_admits_with_the_whole_hand_off(env):
    """RED on e39b37d011 (rc12s). The hand-off is published, the flip reset
    and P's wake leave it unreferenced, other rids' claims fill both arenas
    while the request waits for a seat -- then D admits it. Shipped: the clock
    takes the oldest unreferenced slots, which are the hand-off's (claimed=0,
    anchor 0, the metal's lost=1196). Fixed: every other slot goes first,
    FETCH claims the whole chain and the end anchor."""
    kvp, mbp, kv_slot, mb_slot = _p_publishes(env)
    _h81_hold_then_wake(env.mb, mb_slot.values())
    # older prefix cache of other rids, unreferenced as well
    _publish(env.kv, kvp._stems([f"old{i}" for i in range(SLOTS - len(CHAIN))]))
    _publish(env.mb, mbp._stems([f"old{i}" for i in range(SLOTS - 1)]))
    # seats 6/6: D (and P after its wake) keep claiming for other rids
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "D")
    ids = kvp.alloc_write([f"other{i}" for i in range(SLOTS - len(CHAIN))])
    assert ids is not None, "the other rids' claim finds room"
    got = mbp._evict_for_claim(env.mb, SLOTS - 1)
    assert got == SLOTS - 1
    # admission: what FETCH can claim of the hand-off
    assert _claimed(env.kv, kvp._stems(CHAIN)) == len(CHAIN), "FETCH CAP claimed = the whole chain"
    assert _claimed(env.mb, mbp._stems(CHAIN[-1:])) == 1, "the end anchor is there"


def test_overflow_is_named_handoff_lost_and_a_pin_always_wins(env, caplog):
    """Nothing else left: the claim takes the kept pages -- named per rid
    (HANDOFF-LOST, first lost page), status() says lost with page_size for
    the front's re-price. A park's reference pin is never taken."""
    kvp, _, kv_slot, _ = _p_publishes(env)
    parked = _publish(env.kv, kvp._stems([f"park{i}" for i in range(SLOTS - len(CHAIN))]))
    for s in parked.values():
        _foreign_ref(env.kv, s)                       # a D park pins by reference
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "D")
    with caplog.at_level(logging.WARNING):
        ids = kvp.alloc_write(["late0", "late1"])
    assert ids is not None
    assert _claimed(env.kv, kvp._stems([f"park{i}" for i in range(SLOTS - len(CHAIN))])) == SLOTS - len(CHAIN), \
        "the park keeps every page"
    assert _claimed(env.kv, kvp._stems(CHAIN)) == len(CHAIN) - 2
    lost = [r for r in caplog.records if "HANDOFF-LOST" in r.getMessage()]
    assert len(lost) == 1 and f"rid={RID}" in lost[0].getMessage() and "pool=kv" in lost[0].getMessage()
    st = hp.status(RID)
    assert st["state"] == "lost" and st["pages"] == len(CHAIN) and st["page_size"] == P
    assert st["reason"] == "evicted"
    assert st["first_lost_page"] in (0, 1, 2)


def test_consumed_hand_off_is_evicted_like_any_page(env):
    """After D took the rid (prefetch terminated / rid ended) the order ends:
    the clock takes the hand-off's slots first again, and no loss is named."""
    kvp, _, kv_slot, _ = _p_publishes(env)
    _publish(env.kv, kvp._stems([f"old{i}" for i in range(SLOTS - len(CHAIN))]))
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "D")
    assert hp.consume(RID, "fetch") is True
    assert hp.status(RID)["state"] == "none"
    assert kvp.alloc_write([f"new{i}" for i in range(len(CHAIN))]) is not None
    assert _claimed(env.kv, kvp._stems(CHAIN)) == 0, "no order left: oldest first"


def test_three_ranks_first_consumer_ends_the_order_the_others_hold_by_reference(env):
    """Prüfpunkt 2: the mark is group-wide, the consumption group-uniform (the
    prefetch termination is a vote that passes only after EVERY rank
    resolved, and a resolve takes the rank's reference). Rank 0 removes the
    mark first; ranks 1 and 2 have not run their termination yet -- their
    pages still stand, by their own references, through a claim storm. The
    other ranks' consume is then a no-op."""
    kvp, _, kv_slot, _ = _p_publishes(env)
    _publish(env.kv, kvp._stems([f"old{i}" for i in range(SLOTS - len(CHAIN))]))
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "D")
    ranks = [_kv_pool(env.kv) for _ in range(3)]
    for r in ranks[1:]:                               # resolved on ranks 1, 2 (their read refs)
        for s in kv_slot.values():
            _foreign_ref(env.kv, s)
    assert hp.consume(RID, "fetch") is True           # rank 0 terminated first
    assert kvp.alloc_write([f"storm{i}" for i in range(SLOTS - len(CHAIN))]) is not None
    assert _claimed(env.kv, kvp._stems(CHAIN)) == len(CHAIN), "ranks 1/2 still read all pages"
    assert hp.consume(RID, "fetch") is False and hp.consume(RID, "fetch") is False


def test_p_marks_d_does_not(env):
    """Only group P marks: D's own finish writes a hand-off file too
    (``_pdflip_handoff_write`` runs on every group), it must not keep it."""
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "D")
    assert hp.mark(RID, 3, P) is False
    assert hp.status(RID)["state"] == "none"
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert hp.mark(RID, 3, P) is True
    assert hp.status(RID) == {"state": "pending", "first_lost_page": None, "pages": 3, "page_size": P,
                              "reason": None}


def test_27b_form_kv_only_page_tokens_one(env):
    """27B form: no mamba arena claim, KV pages of one token (P=1), no Form A
    workers (a Form A worker binds the plain pool: no arena, no claim, no
    order to keep). The same order holds on the KV arena."""
    kvp = _kv_pool(env.kv, page_tokens=1)
    _publish(env.kv, kvp._stems(CHAIN))
    assert ho.write(RID, list(range(len(CHAIN))), CHAIN)
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    hp.mark(RID, len(CHAIN), 1)
    _publish(env.kv, kvp._stems([f"old{i}" for i in range(SLOTS - len(CHAIN))]))
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "D")
    assert kvp.alloc_write([f"o{i}" for i in range(SLOTS - len(CHAIN))]) is not None
    assert _claimed(env.kv, kvp._stems(CHAIN)) == len(CHAIN)


def test_mamba_keeps_the_end_anchor_not_every_intermediate(env):
    """The mamba arena keeps the last two chain keys (the N-1 anchor): an
    intermediate anchor of a pending rid is prefix cache like any other."""
    mbp = _mamba_pool(env.mb)
    _publish(env.mb, mbp._stems(CHAIN))
    assert ho.write(RID, list(range(len(CHAIN) * P)), CHAIN)
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    hp.mark(RID, len(CHAIN), P)
    keep = hp.keep_for(mbp)
    assert len(keep) == 2 and sorted(keep.page.tolist()) == [1, 2]
    _publish(env.mb, mbp._stems([f"old{i}" for i in range(SLOTS - len(CHAIN))]))
    assert mbp._evict_for_claim(env.mb, SLOTS - len(CHAIN) + 1) == SLOTS - len(CHAIN) + 1
    assert _claimed(env.mb, mbp._stems(CHAIN[:1])) == 0
    assert _claimed(env.mb, mbp._stems(CHAIN[1:])) == 2


def test_census_names_kept_next_to_pinned(env):
    """Prüfpunkt 1: the HOLDERS fields show the order (handoff_kept, not a
    reference, not in sum) next to every process's reference pins."""
    kvp, _, _, _ = _p_publishes(env)
    parked = _publish(env.kv, kvp._stems(["park0", "park1"]))
    for s in parked.values():
        _foreign_ref(env.kv, s)
    line = hp.census(kvp)
    assert "handoff_kept=3/3" in line and "handoff_rids=1" in line
    assert "arena_pinned=2" in line and "arena_complete=5" in line


def test_front_seam_never_raises(env, monkeypatch):
    """status/drop: never raise, ``none`` in doubt -- no arena dir, a garbage
    rid, a torn file."""
    assert hp.drop("pdflip-nope", "served") is False
    assert hp.status(None)["state"] == "none"
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    hp.mark(RID, 3, P)
    with open(os.path.join(hp._sub(hp.PENDING), RID), "w") as f:
        f.write("{torn")
    assert hp.status(RID)["state"] == "none"
    assert hp.drop(RID, "served") is True
    monkeypatch.delenv("FLLIPER_HICACHE_ARENA_DIR")
    assert hp.status(RID)["state"] == "none" and hp.drop(RID) is False and hp.mark(RID, 1) is False


def test_tree_consumes_only_on_group_d_and_only_pdflip_rids(env, monkeypatch):
    """The tree's hook (prefetch termination, finish, abort) consumes on
    group D only; P's own prefetch of a later leg never ends the order."""
    from flliper.srt.mem_cache import unified_radix_cache as u

    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    hp.mark(RID, 3, P)
    u._pdflip_handoff_consumed(RID, "fetch")
    assert hp.status(RID)["state"] == "pending"
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "D")
    u._pdflip_handoff_consumed("plain-rid", "fetch")
    assert hp.status(RID)["state"] == "pending"
    u._pdflip_handoff_consumed(RID, "end")
    assert hp.status(RID)["state"] == "none"


def test_keep_list_follows_the_pending_dir(env):
    """The cached keep list follows the directory: a new mark is read once,
    a removed one leaves the list."""
    kvp = _kv_pool(env.kv)
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    ho.write(RID, list(range(12)), CHAIN)
    hp.mark(RID, 3, P)
    assert len(hp.keep_for(kvp)) == 3
    ho.write("pdflip-13-44", list(range(8)), ["hX", "hY"])
    time.sleep(0.01)
    hp.mark("pdflip-13-44", 2, P)
    k = hp.keep_for(kvp)
    assert len(k) == 5 and sorted(k.rids) == [RID, "pdflip-13-44"]
    assert np.all(np.diff(k.keys.astype(np.float64)) >= 0), "sorted for arena.c's bisection"
    time.sleep(0.01)
    hp.consume(RID, "fetch")
    assert hp.keep_for(kvp).rids == ["pdflip-13-44"]


def test_an_expired_mark_reads_lost_never_none(env, caplog):
    """A rid that ended where nobody reported it (the front's drop missing)
    must not stay kept forever: after FLLIPER_PDFLIP_HANDOFF_PENDING_EXPIRE_S the
    keep list drops it by name (EXPIRED). A rid that is in fact STILL waiting
    (seat waits up to 658 s measured) must then not read ``none`` -- the front
    would price it on pages nobody protects: it reads ``lost`` with
    reason=expired and first_lost_page=0, and is re-routed fresh via P."""
    kvp = _kv_pool(env.kv)
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    ho.write(RID, list(range(12)), CHAIN)
    hp.mark(RID, 3, P)
    old = time.time() - 1000
    os.utime(os.path.join(hp._sub(hp.PENDING), RID), (old, old))
    env.mp.setenv("FLLIPER_PDFLIP_HANDOFF_PENDING_EXPIRE_S", "900")
    with caplog.at_level(logging.INFO):
        assert len(hp.keep_for(kvp)) == 0
    assert any("EXPIRED" in r.getMessage() and RID in r.getMessage() for r in caplog.records)
    st = hp.status(RID)
    assert st["state"] == "lost" and st["reason"] == "expired" and st["first_lost_page"] == 0
    assert st["pages"] == 3 and st["page_size"] == P
    assert hp.drop(RID, "served") is False and hp.status(RID)["state"] == "none", \
        "the front's drop clears the loss too"


def test_a_renewed_mark_keeps_the_new_chain(env):
    """P re-prefills the same rid (RESUME-VIA-P: the context grew): the
    renewed mark re-reads the chain, the keep list follows the new pages."""
    kvp = _kv_pool(env.kv)
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    ho.write(RID, list(range(12)), CHAIN)
    hp.mark(RID, 3, P)
    assert len(hp.keep_for(kvp)) == 3
    time.sleep(0.01)
    ho.write(RID, list(range(20)), CHAIN + ["hD", "hE"])
    hp.mark(RID, 5, P)
    k = hp.keep_for(kvp)
    assert len(k) == 5 and sorted(k.page.tolist()) == [0, 1, 2, 3, 4]
