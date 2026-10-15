"""A2 VOTE-CACHE 1015 (nf-chainkey-a2-review-1015 B-C, nf-a2cost-1015).

The A2 key proof (``arena_pool._chain_keys_hold``: stems -> ``find_slots_np``) ran in EVERY scheduler
pass of the D usable vote, once per waiting rid with a host chain (~0.85..1.0 us per page, ~+1.1 ms per
100k chain and pass). The vote's proof now re-validates a REMEMBERED key->slot->generation fact against
the live arena (one strided state read + one generation call, no hashing) and falls back to the full
proof on any doubt. The load-time proof (``verify_load_chain``) stays the full ``find`` every time.

What this file pins:
  * COST (counters, no wall clock): hashed stems per pass, off vs on;
  * the value never differs from the full proof: a hit needs the same keys, the same rows, every slot
    COMPLETE now and every slot at the remembered generation;
  * generation moved / state moved / key changed / rows changed -> the full proof runs again;
  * the proof's fill is bracketed by the generations (a slot moving under the proof is not remembered);
  * a cache error is a cost, never a value (fallback to the full proof);
  * only positive proofs are remembered, bounded (LRU), switchable (default ON);
  * a foreign key is never loaded: ``verify_load_chain`` ignores the cache entirely;
  * RANK AGREEMENT: a warm rank, a cold rank and a cache-less rank vote identically over a random
    arena history (the group MIN is the same whatever each rank's cache holds).
"""

import os
import random
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["SGLANG_HICACHE_ARENA_QUEUE_REFS"] = "1"

import numpy as np
import pytest
import torch

from sglang.srt.mem_cache.pool_host import arena_pool as ap

from test_weg2_chainkey_a2_1015 import TOTAL, _arena, _node, _pool, _provable, _publish, _rows, _verify

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")


@pytest.fixture(autouse=True)
def _vote_cache_on(monkeypatch):
    monkeypatch.delenv(ap.ENV_VOTE_CACHE, raising=False)       # default ON
    for k in ap._VOTE_CACHE_N:
        ap._VOTE_CACHE_N[k] = 0.0 if k.endswith("_us") else 0


class _Count:
    """Counts what the arena is asked: stems hashed (full proofs) and cache validations."""

    def __init__(self, arena):
        self.stems = 0
        self.finds = 0
        self.valid = 0
        real_find, real_sg = arena.find_slots_np, arena.slot_state_gens_np

        def find(stems):
            self.finds += 1
            self.stems += len(stems)
            return real_find(stems)

        def sg(slots):
            self.valid += 1
            return real_sg(slots)

        arena.find_slots_np = find
        arena.slot_state_gens_np = sg
        self.real_find, self.real_sg = real_find, real_sg


def _chain(arena, keys, per_node=2, publish=True):
    slot_of = _publish(arena, keys) if publish else {}
    nodes = []
    for i in range(0, len(keys), per_node):
        ks = keys[i:i + per_node]
        nodes.append(_node(i, ks, _rows([slot_of[k] for k in ks])))
    return nodes, slot_of


# ------------------------------------------------------------------ COST: counters, not wall clock ---------


def test_repro_cost_every_pass_hashed_the_chain_before_and_hashes_it_once_now(tmp_path, monkeypatch):
    """B-C: N usable-vote passes over the same waiting chain. Cache OFF is the A2 behaviour (every pass
    hashes every page); ON hashes the chain once and re-validates the arena for the other passes."""
    keys = [f"k{i}" for i in range(24)]
    passes = 20

    arena = _arena(tmp_path, 32, "off.bin")
    nodes, _ = _chain(arena, keys, per_node=6)
    pool = _pool(arena)
    c = _Count(arena)
    monkeypatch.setenv(ap.ENV_VOTE_CACHE, "0")
    for _ in range(passes):
        assert _provable(pool, nodes) == (24, 24)
    assert c.stems == passes * 24 and c.valid == 0, "A2 as shipped: the whole chain is hashed in every pass"
    assert ap.vote_cache_stats()["hit"] == 0

    monkeypatch.delenv(ap.ENV_VOTE_CACHE)
    arena2 = _arena(tmp_path, 32, "on.bin")
    nodes2, _ = _chain(arena2, keys, per_node=6)
    pool2 = _pool(arena2)
    c2 = _Count(arena2)
    for _ in range(passes):
        assert _provable(pool2, nodes2) == (24, 24)
    st = ap.vote_cache_stats()
    assert c2.stems == 24, "the chain is hashed once"
    assert st["hit"] == passes - 1 and st["full"] == 1 and st["fill"] == 1, st
    assert st["hit_us"] >= 0 and st["full_us"] > 0


def test_hit_returns_the_value_of_the_full_proof(tmp_path):
    arena = _arena(tmp_path, 8)
    nodes, _ = _chain(arena, [f"k{i}" for i in range(6)], per_node=3)
    pool = _pool(arena)
    first = _provable(pool, nodes)
    again = _provable(pool, nodes)
    assert first == again == (6, 6)
    assert ap.vote_cache_stats()["hit"] == 1


# ------------------------------------------------------------------ the entry dies with what it proved ------


def test_generation_moved_runs_the_full_proof_again(tmp_path):
    """A slot freed and recycled for ANOTHER key is COMPLETE again, one generation later: the chain's
    page names no slot any more -> cut there. (A state-only check would call it proven.)"""
    arena = _arena(tmp_path, 4)                      # exactly the chain's pages: the freed slot is the next claim
    nodes, slot_of = _chain(arena, ["a", "b", "c", "d"], per_node=2)
    pool = _pool(arena)
    c = _Count(arena)
    assert _provable(pool, nodes) == (4, 4)
    assert _provable(pool, nodes) == (4, 4)
    assert c.stems == 4 and ap.vote_cache_stats()["hit"] == 1
    arena.free_slots([slot_of["c"]])
    (sy, st, gy), = arena.claim_slots(["Y"], [TOTAL])
    assert sy == slot_of["c"] and st == 0 and arena.complete_slots([sy], [gy], [(0, TOTAL)]) == [1]
    assert arena.claim_census([sy])[0]["state"] == 2     # COMPLETE again, another key's page
    assert _provable(pool, nodes) == (2, 4), "page 2's key names no slot: cut at the last proven page"
    stats = ap.vote_cache_stats()
    assert stats["gen_moved"] == 1 and stats["full"] == 2, stats
    assert c.stems >= 4 + 4, "the miss hashed the chain again (fast proof, then the page proof)"


def test_same_key_refilled_into_the_same_slot_is_proven_by_the_full_proof(tmp_path):
    """Generation moved but the key's page is back on the same slot (refill): the cache cannot know, so the
    FULL proof decides -- and proves it. Never a wrong cut, never a wrong hit."""
    arena = _arena(tmp_path, 2)
    nodes, slot_of = _chain(arena, ["a", "b"], per_node=2)
    pool = _pool(arena)
    assert _provable(pool, nodes) == (2, 2)
    arena.free_slots([slot_of["b"]])
    assert _publish(arena, ["b"])["b"] == slot_of["b"]
    assert _provable(pool, nodes) == (2, 2)
    assert ap.vote_cache_stats()["gen_moved"] == 1
    assert _provable(pool, nodes) == (2, 2)              # re-remembered under the new generation
    assert ap.vote_cache_stats()["hit"] == 1


def test_state_moved_runs_the_full_proof_again(tmp_path):
    """EVICTING keeps the generation and unlinks the key: only the state check sees it. (A
    generation-only check would call the page proven.)"""
    arena = _arena(tmp_path, 4)
    nodes, slot_of = _chain(arena, ["a", "b", "c", "d"], per_node=2)
    pool = _pool(arena)
    assert _provable(pool, nodes) == (4, 4)
    gens_before = arena.slot_gens_np([slot_of["c"]]).tolist()
    got = arena.evict_candidates(1, keep_stems=["a", "b", "d"])
    assert [g[0] for g in got] == [slot_of["c"]]
    assert arena.slot_gens_np([slot_of["c"]]).tolist() == gens_before, "evicting does not move the generation"
    assert arena.claim_census([slot_of["c"]])[0]["state"] == 3
    assert _provable(pool, nodes) == (2, 4)
    assert ap.vote_cache_stats()["state_moved"] == 1


def test_key_changed_in_place_runs_the_full_proof_again(tmp_path):
    arena = _arena(tmp_path, 8)
    nodes, _ = _chain(arena, ["a", "b", "c", "d"], per_node=2)
    pool = _pool(arena)
    assert _provable(pool, nodes) == (4, 4)
    nodes[1].hash_value[0] = "zz"                       # same length, same first/last key: only a full compare sees it
    assert _provable(pool, nodes) == (2, 4)
    assert ap.vote_cache_stats()["changed"] == 1


def test_rows_changed_is_a_miss_at_the_cache(tmp_path):
    arena = _arena(tmp_path, 8)
    nodes, slot_of = _chain(arena, ["a", "b"], per_node=2)
    pool = _pool(arena)
    assert _provable(pool, nodes) == (2, 2)
    vc = pool._a2_vote_cache
    keys = ["a", "b"]
    ar = torch.cat([n.host_value for n in nodes]) - 16
    assert vc.hit(arena, ar, keys) is True
    other = torch.cat([_rows([slot_of["b"], slot_of["a"]])]) - 16
    assert vc.hit(arena, other, keys) is False, "same keys on other rows is not the remembered chain"
    assert ap.vote_cache_stats()["changed"] == 1


def test_a_chain_that_does_not_hold_is_never_remembered(tmp_path):
    arena = _arena(tmp_path, 8)
    slot_of = _publish(arena, ["a", "b"])
    pool = _pool(arena)
    node = _node(1, ["a", "ghost"], _rows([slot_of["a"], slot_of["b"]]))   # 'ghost' names no slot
    assert _provable(pool, [node]) == (1, 2)
    assert _provable(pool, [node]) == (1, 2)
    assert len(pool._a2_vote_cache.d) == 0 and ap.vote_cache_stats()["hit"] == 0


# ------------------------------------------------------------------ the fill is bracketed ----------------


def test_a_slot_moving_under_the_proof_is_not_remembered(tmp_path):
    """The proof read the old binding; a slot is recycled right after the find. The generations before and
    after the proof differ -> nothing is stored, the next pass proves again (and cuts)."""
    arena = _arena(tmp_path, 4)
    nodes, slot_of = _chain(arena, ["a", "b", "c", "d"], per_node=2)
    pool = _pool(arena)
    real = arena.find_slots_np
    fired = []

    def racing_find(stems):
        out = real(stems)
        if not fired:
            fired.append(1)
            arena.free_slots([slot_of["d"]])
            (sy, st, gy), = arena.claim_slots(["Y"], [TOTAL])
            arena.complete_slots([sy], [gy], [(0, TOTAL)])
        return out

    arena.find_slots_np = racing_find
    assert _provable(pool, nodes) == (4, 4), "the proof itself saw the pre-race state"
    assert ap.vote_cache_stats()["unfilled_race"] == 1 and len(pool._a2_vote_cache.d) == 0
    assert _provable(pool, nodes) == (3, 4), "page 3's key lost its slot: the next pass cuts, no stale hit"


# ------------------------------------------------------------------ errors, switch, bound ---------------


def test_a_cache_error_falls_back_to_the_full_proof(tmp_path):
    arena = _arena(tmp_path, 8)
    nodes, _ = _chain(arena, ["a", "b", "c", "d"], per_node=2)
    pool = _pool(arena)

    def boom(slots):
        raise RuntimeError("header view gone")

    real = arena.slot_state_gens_np
    assert _provable(pool, nodes) == (4, 4)                       # warm: an entry exists
    arena.slot_state_gens_np = boom
    assert _provable(pool, nodes) == (4, 4), "the HIT path fails: the full proof answers, no exception into the vote"
    st = ap.vote_cache_stats()
    assert st["error"] == 1 and st["hit"] == 0, st                # the hit's error; the proof then runs uncached
    arena.slot_state_gens_np = real
    pool._a2_vote_cache.d.clear()
    arena.slot_state_gens_np = boom
    assert _provable(pool, nodes) == (4, 4)                       # cold: only the fill path fails
    assert _provable(pool, nodes) == (4, 4), "never an exception into the vote, never another value"
    assert ap.vote_cache_stats()["error"] == 3 and len(pool._a2_vote_cache.d) == 0   # the fill's g0 failed twice: nothing stored

    # a broken chain still cuts through the error path
    nodes[1].hash_value[1] = "zz"
    assert _provable(pool, nodes) == (3, 4)


def test_switch_off_is_the_a2_behaviour(tmp_path, monkeypatch):
    arena = _arena(tmp_path, 8)
    nodes, _ = _chain(arena, ["a", "b"], per_node=1)
    pool = _pool(arena)
    c = _Count(arena)
    for off in ("0", "false", "off"):
        monkeypatch.setenv(ap.ENV_VOTE_CACHE, off)
        assert _provable(pool, nodes) == (2, 2)
    assert c.finds == 3 and not hasattr(pool, "_a2_vote_cache")
    assert ap.vote_cache_stats()["hit"] == 0 and ap.vote_cache_stats()["full"] == 0
    monkeypatch.delenv(ap.ENV_VOTE_CACHE)
    assert _provable(pool, nodes) == (2, 2) and _provable(pool, nodes) == (2, 2)
    assert ap.vote_cache_stats()["hit"] == 1


def test_default_is_on():
    assert ap._vote_cache_enabled() is True


def test_entries_are_bounded_lru(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "VOTE_CACHE_MAX", 3)
    arena = _arena(tmp_path, 16)
    pool = _pool(arena)
    slot_of = _publish(arena, [f"k{i}" for i in range(6)])
    nodes = [[_node(i, [f"k{i}"], _rows([slot_of[f"k{i}"]]))] for i in range(6)]
    for n in nodes:
        assert _provable(pool, n) == (1, 1)
    vc = pool._a2_vote_cache
    assert len(vc.d) == 3 and ap.vote_cache_stats()["evicted"] == 3
    assert set(s[0] for s in vc.d) == {"k3", "k4", "k5"}, "the oldest entries went"


# ------------------------------------------------------------------ the load-time proof is untouched ------


def test_verify_load_chain_never_consults_the_cache(tmp_path, monkeypatch):
    """The proof right before a transfer stays the full find every time: a warm vote cache is never read,
    and a recycled slot is refused by name (a foreign key is never loaded)."""
    arena = _arena(tmp_path, 4)
    nodes, slot_of = _chain(arena, ["a", "b", "c", "d"], per_node=2)
    pool = _pool(arena)
    assert _provable(pool, nodes) == (4, 4)                      # warm
    c = _Count(arena)
    monkeypatch.setattr(ap._VoteKeyCache, "hit", lambda *a, **k: pytest.fail("the load proof read the vote cache"))
    host = torch.cat([n.host_value for n in nodes])
    for _ in range(3):
        assert ap.verify_load_chain(pool, nodes, host, rows_of=lambda n: n.host_value,
                                    stems_of=lambda h: list(h)) is host
    assert c.finds == 3 and c.stems == 12 and c.valid == 0, "every load proves the whole chain by key again"
    monkeypatch.undo()
    # the slot of 'c' is recycled for another key while the vote cache is warm
    arena.free_slots([slot_of["c"]])
    (sy, st, gy), = arena.claim_slots(["Y"], [TOTAL])
    assert arena.complete_slots([sy], [gy], [(0, TOTAL)]) == [1]
    with pytest.raises(ap.ArenaChainMismatch):
        _verify(pool, nodes)
    assert _provable(pool, nodes) == (2, 4)


# ------------------------------------------------------------------ rank agreement -----------------------


def test_warm_cold_and_cacheless_ranks_vote_identically_over_a_random_arena_history(tmp_path, monkeypatch):
    """Three ranks over ONE shared arena file: W keeps its cache warm through the whole history, U builds a
    fresh cache for every vote (cold), C runs with the cache switched off (the A2 proof). After every
    arena operation every rank votes on the same chains; the three votes must be equal -- so the group MIN
    is the same whatever any rank's cache holds. Operations: evict, free, recycle for another key, refill
    a key elsewhere, claim without completing, no-op. Chains keep their ORIGINAL rows (stale after recycle)."""
    rng = random.Random(1015)
    nslots = 10
    arena_w = _arena(tmp_path, nslots, "shared.bin")
    arena_u = _arena(tmp_path, nslots, "shared.bin")
    arena_c = _arena(tmp_path, nslots, "shared.bin")
    keys = [f"k{i}" for i in range(8)]
    slot_of = _publish(arena_w, keys)
    pool_w, pool_c = _pool(arena_w), _pool(arena_c)

    def mk(ks, per):
        return [_node(i, ks[i:i + per], _rows([slot_of[k] for k in ks[i:i + per]])) for i in range(0, len(ks), per)]

    chains = [mk(keys[0:4], 2), mk(keys[2:6], 2), mk(keys[0:8], 4), mk(keys[5:8], 3), mk(keys[1:2], 1)]
    extra = iter(f"x{i}" for i in range(10_000))
    known = list(keys)
    claimed = []

    def slot(k):
        return int(arena_w.find_slots_np([k])[0][0])

    def vote_all():
        out = []
        for ch in chains:
            w = _provable(pool_w, ch)
            monkeypatch.setenv(ap.ENV_VOTE_CACHE, "0")
            c = _provable(pool_c, ch)
            monkeypatch.delenv(ap.ENV_VOTE_CACHE)
            u = _provable(_pool(arena_u), ch)
            assert w == c == u, (w, c, u)
            out.append(w)
        return out

    def op():
        k = rng.choice(("evict", "free", "recycle", "refill", "refill", "refill", "claim", "reap", "noop", "noop", "noop"))
        who = rng.choice(known)
        if k == "evict":
            got = arena_w.evict_candidates(1, keep_stems=[x for x in known if x != who])
            if got:
                arena_w.free_evicted([g[0] for g in got])
        elif k == "free":
            s = slot(who)
            if s >= 0 and arena_w.slot_refs([s]) == [0] and s not in claimed:
                arena_w.free_slots([s])
        elif k == "recycle":
            nk = next(extra)
            (s, st, g), = arena_w.claim_slots([nk], [TOTAL])
            if st == 0 and s >= 0:
                arena_w.complete_slots([s], [g], [(0, TOTAL)])
                known.append(nk)
        elif k == "refill":
            if slot(who) < 0:
                (s, st, g), = arena_w.claim_slots([who], [TOTAL])
                if st == 0 and s >= 0:
                    arena_w.complete_slots([s], [g], [(0, TOTAL)])
        elif k == "claim":
            nk = next(extra)
            (s, st, g), = arena_w.claim_slots([nk], [TOTAL])
            if st == 0 and s >= 0:
                claimed.append(s)
        elif k == "reap":
            for s in claimed:
                arena_w.free_slots([s])
            claimed.clear()
        return k

    seen = set()
    cuts = set()
    for _ in range(400):
        seen.add(op())
        votes = vote_all()
        cuts.add(tuple(v[0] for v in votes))
    st = ap.vote_cache_stats()
    assert st["hit"] > 20, f"the history must exercise hits: {st}"
    assert st["gen_moved"] + st["state_moved"] > 5, f"and invalidations: {st}"
    assert len(cuts) > 3, "and cut the chains at different depths"
    assert {"evict", "free", "recycle", "refill", "claim", "reap"} <= seen


# ------------------------------------------------------------------ the arena helper --------------------


def test_slot_state_gens_np_values_and_out_of_range(tmp_path):
    arena = _arena(tmp_path, 3)
    slot_of = _publish(arena, ["a", "b"])
    st, gn = arena.slot_state_gens_np([slot_of["a"], slot_of["b"], 2, -1, 99])
    assert st.tolist() == [2, 2, 0, -1, -1]
    assert gn.tolist()[:3] == arena.slot_gens_np([slot_of["a"], slot_of["b"], 2]).tolist() and gn.tolist()[3:] == [-1, -1]
    st0, gn0 = arena.slot_state_gens_np([])
    assert st0.shape == (0,) and gn0.shape == (0,)
    arena.free_slots([slot_of["a"]])
    st2, gn2 = arena.slot_state_gens_np([slot_of["a"]])
    assert st2.tolist() == [0] and gn2.tolist() != gn.tolist()[:1], "free moves the generation"
    arena._mm.close()          # raises BufferError when a numpy view still pins the mapping
