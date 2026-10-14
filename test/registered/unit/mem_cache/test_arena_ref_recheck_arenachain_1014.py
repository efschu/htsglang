# SPDX-License-Identifier: Apache-2.0
"""ARENA-CHAIN (NF 14.10. 07:29:33Z, P PP0): reference, then look the key up again.

The death: ``#1424 CHAIN MISMATCH node=460 page=174 of 848 bad=256: key slot -1 (state 0)`` ->
``ArenaChainMismatch`` -> RANK-DEATH of PP0 (``..._065207.P.log`` 173055). The cause is a window in
``HiCacheController._arena_page_get``:

* ``arena_fill_from_disk`` returns COMPLETE slots with NO reference ("no reference taken yet");
* the reader references them afterwards;
* in between another rank's full-arena claim runs the clock (>= 256 slots a round), clears the slots and
  re-claims them for ITS stems;
* ``arena_ref_slots(+1)`` takes any COMPLETE *or CLAIMED* slot without a key check (arena.c: "#1427 CLAIMED slots
  take refs too") -- so the reference succeeds on a slot that belongs to another page, and the node's rows address it.

The same window exists for the pages found in the L2 (the lead batch, the ``_pre`` pages referenced ahead of the fill):
find, then the reference. The fix is the Linux page-cache rule (``folio_try_get`` then re-load the index entry): after
every reference the stems are looked up again, and a page is kept only where its stem still names THIS slot, COMPLETE.

Real objects: the C arena on a temp file, a bound ArenaMHAHostPool in a HostPoolGroup, the controller's real
``_arena_page_get``; the backend is the L3 double of rc12z. The "other rank" is the real arena, driven between the two
steps (clock = ``free_slots``, its claims = ``claim_slots`` of foreign stems until it holds the victims).

The invariant every returned page must satisfy (what ``verify_load_chain`` later enforces with its life): the stem of
page i names exactly the slot the page's rows were resolved to, COMPLETE. RED on f4afb10f28 (the page count is the full
count and the invariant is broken for the victims), GREEN with the recheck.
"""

import os
import shutil
import tempfile
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache import hicache_storage as hs
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from sglang.srt.mem_cache.storage.file import hicache_arena as ha
from sglang.test.test_utils import CustomTestCase

PAGE = 64
S = 16
SLOTS = 64
ENV = "SGLANG_WEG2_ARENA_REF_RECHECK"


class _Backend:
    """The L3 double: every stem is 'on disk'; a fill claims a fresh slot, completes it, hands it out WITHOUT a
    reference. ``after_fill(arena, stems, out)`` is the other rank acting in the window."""

    def __init__(self, after_fill=None):
        self.fill_calls = []
        self.after_fill = after_fill

    def _get_suffixed_key(self, h):
        return h

    def _suffix_for_key(self, k):
        return ("", None)

    def _arena_evict_to_disk(self, arena, want):
        return 0

    def arena_fill_from_disk(self, arena, stems, total_bytes, prefix=False):
        self.fill_calls.append(list(stems))
        out = []
        for st in stems:
            (slot, status, gen), = arena.claim_slots([st], [int(total_bytes)])
            if status == 2:
                out.append(slot)
                continue
            if status != 0:
                out.append(None)
                continue
            cs = arena.complete_slots([slot], [gen], [(0, int(total_bytes))])
            out.append(slot if cs and cs[0] in (1, 2) else None)
        if self.after_fill is not None:
            self.after_fill(arena, stems, out)
        return out


class _Op:
    probe_pins = None
    completed_tokens = 0

    def __init__(self):
        self.completed = 0

    def increment(self, n):
        self.completed += n
        return True


def _foreign_takeover(arena, victims):
    """Another rank under pressure: its clock clears ``victims`` (it takes slots in slot order, >= 256 a round) and its
    own claims re-take exactly them (CLAIMED, foreign stems). Returns the foreign stems that sit on them."""
    victims = [int(v) for v in victims]
    arena.free_slots(victims)
    taken, extra = {}, []
    for k in range(4 * SLOTS):
        (slot, status, gen), = arena.claim_slots([f"foreign{k}"], [PAGE])
        if status != 0:
            break
        if slot in victims:
            taken[slot] = f"foreign{k}"
        else:
            extra.append(slot)
        if len(taken) == len(victims):
            break
    if extra:
        arena.free_slots(extra)
    assert set(taken) == set(victims), "the foreign claims must sit on the victims' slots"
    return taken


class ArenaChainRefRecheck(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._env = os.environ.pop(ENV, None)
        fd, self.path = tempfile.mkstemp(prefix="arena-chain-recheck-", suffix=".bin")
        os.close(fd)
        os.unlink(self.path)

    def tearDown(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass
        if self._env is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = self._env
        super().tearDown()

    # -- rig -----------------------------------------------------------
    def _rank(self, backend):
        arena = ha.ShmArena(self.path, PAGE, SLOTS)
        pool = ArenaMHAHostPool.__new__(ArenaMHAHostPool)
        pool.size, pool.page_size, pool.layout, pool.device = S, 1, "layer_first", "cpu"
        pool.layer_num, pool.dtype, pool.head_num, pool.head_dim = 1, torch.bfloat16, 1, 16
        pool.pin_memory = False
        pool._arena_init_fields()
        pool.bind(arena, types.SimpleNamespace(extents=[(0, 32), (32, 32)], total_bytes=PAGE), role="kv", pin=False)
        pool._backend = backend
        group = HostPoolGroup([PoolEntry(name=PoolName.KV, host_pool=pool, device_pool=None,
                                         layer_mapping={}, is_primary_index_anchor=True)])
        cc = types.SimpleNamespace(mem_pool_host=group, storage_backend=backend, page_size=1)
        resolved = []
        real = group.resolve_rows

        def spy(host_indices, slots):
            resolved.extend(int(s) for s in slots)
            return real(host_indices, slots)

        group.resolve_rows = spy
        return arena, pool, group, cc, resolved

    def _publish(self, arena, pool, hashes, *, evict=()):
        """The pages are published into the arena; the ones in ``evict`` then live only in the L3."""
        rows = pool.alloc_write(hashes)
        self.assertIsNotNone(rows)
        pool.complete_write(rows)
        pool.free(rows)
        gone = [s for s, _st in arena.find_slots([hashes[i] for i in evict])]
        if gone:
            arena.free_slots(gone)

    def _read(self, cc, pool, hashes):
        hi = pool.alloc_read(len(hashes))
        op = _Op()
        got = HiCacheController._arena_page_get(cc, op, hashes, hi)
        return got, op

    def _assert_every_returned_page_is_proven(self, arena, hashes, resolved):
        """The invariant ``verify_load_chain`` kills a rank for: the stem names the resolved slot, COMPLETE."""
        for i, slot in enumerate(resolved):
            (fs, st), = arena.find_slots([hashes[i]])
            self.assertEqual((int(fs), int(st)), (int(slot), 2),
                             f"page {i}: its rows were resolved to slot {slot} but its stem names slot {fs} (state {st}) "
                             "-- another page's slot (the node-460 page-174 death)")

    def _refs(self, arena, slots):
        return [int(x) for x in arena.slot_refs([int(s) for s in slots])]

    # -- (1) the fill site: the incident ------------------------------------
    def test_fill_site_a_foreign_reclaim_between_fill_and_ref_cuts_the_prefix_and_leaves_no_foreign_ref(self):
        hashes = [f"p{j}" for j in range(8)]
        victims = {}

        def other_rank(arena, stems, out):
            victims.update(_foreign_takeover(arena, [out[3], out[4], out[5]]))

        backend = _Backend(after_fill=other_rank)
        arena, pool, group, cc, resolved = self._rank(backend)
        self._publish(arena, pool, hashes, evict=range(8))     # all 8 live only in the L3
        got, op = self._read(cc, pool, hashes)
        self.assertEqual(got, 3, "pages 0-2 proven; page 3 on is the foreign slots' -- the prefix ends there")
        self.assertEqual(op.completed, 3)
        self.assertEqual(len(resolved), 3)
        self._assert_every_returned_page_is_proven(arena, hashes, resolved)
        self.assertEqual(self._refs(arena, victims), [0, 0, 0], "the references taken on the foreign slots are given back")
        self.assertEqual(self._refs(arena, resolved), [1, 1, 1], "the proven pages stay referenced for the load")

    def test_fill_site_the_cut_page_is_read_again_by_the_next_call(self):
        hashes = [f"p{j}" for j in range(6)]
        state = {"first": True, "victims": {}}

        def other_rank(arena, stems, out):
            if state["first"]:
                state["first"] = False
                state["victims"].update(_foreign_takeover(arena, [out[2], out[3]]))

        backend = _Backend(after_fill=other_rank)
        arena, pool, group, cc, resolved = self._rank(backend)
        self._publish(arena, pool, hashes, evict=range(6))
        got, _op = self._read(cc, pool, hashes)
        self.assertEqual(got, 2)
        arena.free_slots(list(state["victims"]))                 # the foreign claims end
        got2, _op2 = self._read(cc, pool, hashes[2:])
        self.assertEqual(got2, 4, "the L3 gives the rest the second time: nothing was lost, only not loaded wrong")

    # -- (2) the lead batch: find, then ref ----------------------------------
    def test_lead_site_pages_found_in_l2_are_checked_after_the_reference_and_read_again(self):
        hashes = [f"p{j}" for j in range(8)]
        backend = _Backend()
        arena, pool, group, cc, resolved = self._rank(backend)
        self._publish(arena, pool, hashes)                       # all 8 COMPLETE in the L2
        slots0 = [int(s) for s, _st in arena.find_slots(hashes)]
        victims = {}
        orig_find = arena.find_slots_np
        state = {"n": 0}

        def find_then_other_rank(stems):
            res = orig_find(stems)
            state["n"] += 1
            if state["n"] == 1:                                  # right after the controller's find, before its reference
                victims.update(_foreign_takeover(arena, slots0[4:]))
            return res

        arena.find_slots_np = find_then_other_rank
        got, op = self._read(cc, pool, hashes)
        self._assert_every_returned_page_is_proven(arena, hashes, resolved)
        self.assertEqual(got, 8, "pages 4-7 were gone from the L2 and came back from the L3 (read again)")
        self.assertEqual(backend.fill_calls, [hashes[4:]])
        self.assertEqual(self._refs(arena, victims), [0, 0, 0, 0], "no reference stays on a foreign slot")

    # -- (3) the pages referenced ahead of the fill --------------------------
    def test_pre_site_pages_referenced_ahead_of_the_fill_are_checked_too(self):
        hashes = [f"p{j}" for j in range(8)]
        backend = _Backend()
        arena, pool, group, cc, resolved = self._rank(backend)
        self._publish(arena, pool, hashes, evict=[0])            # page 0 only in the L3, 1-7 in the L2 past the lead
        slots0 = [int(s) for s, _st in arena.find_slots(hashes[1:])]   # slots of pages 1..7
        victims = {}
        orig_find = arena.find_slots_np
        state = {"n": 0}

        def find_then_other_rank(stems):
            res = orig_find(stems)
            state["n"] += 1
            if state["n"] == 1:
                victims.update(_foreign_takeover(arena, slots0[3:]))     # pages 4..7
            return res

        arena.find_slots_np = find_then_other_rank
        got, op = self._read(cc, pool, hashes)
        self._assert_every_returned_page_is_proven(arena, hashes, resolved)
        self.assertEqual(got, 4, "page 0 from the L3, 1-3 proven; the prefix ends at the first foreign page")
        self.assertEqual(self._refs(arena, victims), [0, 0, 0, 0])
        self.assertEqual(backend.fill_calls, [[hashes[0]]], "only the page the prefix needs is read: nothing past the cut")

    # -- (4) the switch -------------------------------------------------------
    def test_with_the_switch_off_the_old_read_path_returns_the_foreign_slots(self):
        """The danger direction, shown: without the recheck the same window loads another page's slot."""
        os.environ[ENV] = "0"
        hashes = [f"p{j}" for j in range(8)]

        def other_rank(arena, stems, out):
            _foreign_takeover(arena, [out[3], out[4], out[5]])

        backend = _Backend(after_fill=other_rank)
        arena, pool, group, cc, resolved = self._rank(backend)
        self._publish(arena, pool, hashes, evict=range(8))
        got, _op = self._read(cc, pool, hashes)
        self.assertEqual(got, 8, "the references on the CLAIMED foreign slots all succeed (arena.c takes CLAIMED refs)")
        with self.assertRaises(AssertionError):
            self._assert_every_returned_page_is_proven(arena, hashes, resolved)

    def test_a_clean_read_is_unchanged(self):
        hashes = [f"p{j}" for j in range(8)]
        backend = _Backend()
        arena, pool, group, cc, resolved = self._rank(backend)
        self._publish(arena, pool, hashes, evict=[5, 6])
        got, op = self._read(cc, pool, hashes)
        self.assertEqual((got, op.completed), (8, 8))
        self._assert_every_returned_page_is_proven(arena, hashes, resolved)
        self.assertEqual(self._refs(arena, resolved), [1] * 8)


# ---------------------------------------------------------------------------
# l3fill_past_prefix frees by generation
# ---------------------------------------------------------------------------

PG = 4096


class _FillArena:
    """Records how the fill gives up its claims past the prefix."""

    def __init__(self):
        self.buf = np.zeros((64, PG), dtype=np.uint8)
        self.next = 0
        self.gen = {}
        self.freed_plain, self.freed_gen = [], []
        self.claimed = {}

    def claim_slots(self, stems, totals):
        out = []
        for st in stems:
            if st == "s03":
                # h18port: status 3 (too large -- no claim, nobody fills it) ends the prefix. The original double
                # answered 4 (arena full); since the ROOM-WAIT line (``_hard``: a still-full stem is awaited, not an
                # end) that no longer ends the prefix -- the test was red on fe799c4d6e as well.
                out.append((-1, 3, 0))
                continue
            s = self.next
            self.next += 1
            self.gen[s] = 100 + s
            self.claimed[s] = st
            out.append((s, 0, self.gen[s]))
        return out

    def slot_ptr(self, slot):
        return self.buf[slot].ctypes.data

    def complete_slots(self, slots, gens, extents):
        return [1] * len(slots)

    def free_slots(self, slots):
        self.freed_plain += [int(s) for s in slots]

    def free_if_gen(self, slots, gens, reason="unnamed"):
        self.freed_gen.append(([int(s) for s in slots], [int(g) for g in gens], reason))
        return [1] * len(slots)


def _run_past_prefix(tmp_path):
    be = object.__new__(hs.HiCacheFile)
    root = tmp_path / "l3"
    root.mkdir()
    stems = ["s%02d" % i for i in range(8)]
    paths = {}
    for st in stems:
        p = root / (st + ".bin")
        p.write_bytes(bytes([7]) * PG)
        paths[st] = str(p)
    be._stat_stems = lambda ss: {st: PG for st in ss}
    be._existing_path = lambda st: paths[st]
    be._arena_evict_to_disk = lambda arena, want, need=None: 0
    arena = _FillArena()
    out = hs.HiCacheFile.arena_fill_from_disk(be, arena, stems, PG, prefix=True)
    return arena, out


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_l3fill_past_prefix_gives_its_claims_up_by_generation(tmp_path):
    arena, out = _run_past_prefix(tmp_path)
    assert [o is not None for o in out] == [True, True, True] + [False] * 5, "the prefix ends at the page with no claim"
    assert arena.freed_plain == [], "past-the-prefix claims are NOT freed by slot alone (a slot another rank joined and completed went with it)"
    assert len(arena.freed_gen) == 1
    slots, gens, reason = arena.freed_gen[0]
    assert reason == "l3fill_past_prefix"
    assert slots == [3, 4, 5, 6], "the claims of pages 4-7 (page 3 got none)"
    assert gens == [103, 104, 105, 106], "each claim is given up under the generation it took"


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_mutant_the_past_prefix_claims_freed_by_slot_alone_is_what_the_test_forbids(tmp_path, monkeypatch):
    """The old path (an arena without free_if_gen frees by slot): the assertion above sees it."""
    monkeypatch.setattr(_FillArena, "free_if_gen", None)
    arena, _out = _run_past_prefix(tmp_path)
    assert arena.freed_plain == [3, 4, 5, 6] and not arena.freed_gen


# ---------------------------------------------------------------------------
# the exception names the page and the node the proof could not carry
# ---------------------------------------------------------------------------


def test_the_chain_mismatch_carries_the_page_and_the_node():
    from sglang.srt.mem_cache.pool_host import arena_pool as ap
    from test_arena_chain_rekey_1424c import SLOT_OF, _Arena, _metal_chain, _pool, _verify

    nodes = _metal_chain()
    with pytest.raises(ap.ArenaChainMismatch) as cm:
        _verify(_pool(_Arena(SLOT_OF)), nodes)
    e = cm.value
    assert e.page is not None
    assert any(e.node is n for n in nodes)
    assert 0 <= e.node_page < len(e.node.hash_value)
    assert (e.kv_tokens, e.avail_after) == (0, 0), "the load-back terms of AgreedRoomShort, so one retreat reads both"


def test_the_node_of_a_page():
    from sglang.srt.mem_cache.pool_host import arena_pool as ap

    keys = {"a": ["k0", "k1"], "b": ["k2"], "c": ["k3", "k4", "k5"]}
    nodes = list(keys)
    of = lambda n: keys[n]  # noqa: E731
    assert ap._chain_node_of_page(nodes, of, 0) == ("a", 0)
    assert ap._chain_node_of_page(nodes, of, 2) == ("b", 0)
    assert ap._chain_node_of_page(nodes, of, 5) == ("c", 2)
    assert ap._chain_node_of_page(nodes, of, 99) == ("c", 2), "past the end names the last node"
    assert ap._chain_node_of_page(nodes, of, None) == (None, None)


# ---------------------------------------------------------------------------
# mutants: each of the three reference sites must have its own test that goes red without the recheck there
# ---------------------------------------------------------------------------

import sys  # noqa: E402

from sglang.srt.managers import cache_controller as _cc  # noqa: E402


def _blind_at(site):
    """``_arena_first_unproven`` that proves every page at ONE call site (found by the caller's own locals)."""
    real = _cc._arena_first_unproven

    def mutant(arena, stems, idxs, slots):
        loc = sys._getframe(1).f_locals
        here = ("fill" if idxs is loc.get("_fill_idx") else "pre" if idxs is loc.get("_pi")
                else "lead" if idxs is loc.get("_li") else "?")
        if here == site:
            return -1, None
        return real(arena, stems, idxs, slots)

    return mutant


def _fails(name, patch):
    suite = unittest.TestSuite([ArenaChainRefRecheck(name)])
    res = unittest.TestResult()
    with patch:
        suite.run(res)
    return bool(res.failures or res.errors)


from unittest import mock  # noqa: E402

_SITE_TESTS = {
    "fill": "test_fill_site_a_foreign_reclaim_between_fill_and_ref_cuts_the_prefix_and_leaves_no_foreign_ref",
    "lead": "test_lead_site_pages_found_in_l2_are_checked_after_the_reference_and_read_again",
    "pre": "test_pre_site_pages_referenced_ahead_of_the_fill_are_checked_too",
}


@pytest.mark.parametrize("site", ["fill", "lead", "pre"])
def test_mutant_no_recheck_at_one_site_turns_that_sites_test_red(site):
    name = _SITE_TESTS[site]
    assert not _fails(name, mock.patch.object(_cc, "_arena_first_unproven", _cc._arena_first_unproven)), "the unmutated test passes"
    assert _fails(name, mock.patch.object(_cc, "_arena_first_unproven", _blind_at(site))), f"the {site} site survived without its recheck"


def test_mutant_a_cut_that_keeps_the_reference_on_the_foreign_slot_is_caught():
    """The cut without giving the references back: the refs stay on the foreign slots (the test reads them)."""
    real = ha.ShmArena.ref_slots_np

    def no_release(self, slots, delta):
        return 0 if delta < 0 else real(self, slots, delta)

    assert _fails(_SITE_TESTS["fill"], mock.patch.object(ha.ShmArena, "ref_slots_np", no_release))


if __name__ == "__main__":
    unittest.main()
