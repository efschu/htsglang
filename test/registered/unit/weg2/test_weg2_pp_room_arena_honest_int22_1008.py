# SPDX-License-Identifier: Apache-2.0
"""PR-ARENA: the room vote's walk prices a backup at the arena's FREE slots (NF int22, 08.10.).

THE DEATH. P log boot_weg2_dkrnfint4h6ablxcw3spbar1dauer10081717_6b3bd1a6df_1008_171755.P.log,
PP1 18:34:20Z (217573-217589): ``EVICT-FRONTIER-CENSUS request=16449 delivered_before=8192
reported_evictable=195968 on_frontier=8192 behind_device_child=187776`` and ``EXTEND-RELIEF
evicted=0 asked=16321`` -> ``Prefill out of memory`` (``Available full tokens: 208256
(full_available_size=12288 + full_evictable_size_=195968)``, ``EVICTION UNDER-DELIVERED: asked
for 12353 tokens, the pool received 8192``) -> RANK-DEATH, PP2/PP0 after it. PP0 had launched
the 16321-token pass (``#969N ADMIT slot=2 fwd_ct=100 bs=2 extend=16321``, 18:34:16) under the
PP room vote -- so PP1's room fact said the stage could pay it.

The tree dump after the death (217639-217722) shows why the peel could not: the two device
leaves of the two unlocked chains, [620] (behind [594]..[609], 86016 tokens) and [576] (behind
[568]/[569]..[575], 109952 tokens), are un-backed (``#1421 BACKUP-REFUSED why=arena_claim`` /
``parent_unbacked`` on 609/620/576 at 18:34:19-20) and each carries host-only children
([621]/[619], [577]..[583]) that UD-H did not clear (no ``UD-H HOST-CHILDREN-CLEARED`` on PP1 at
the death). The KV arena had no free slot (``#1427 ARENA-CLAIM REFUSED statuses=[0, 4] (4 = no
free slot)``, ``ARENA-DROP ... freed=0``).

THE WALK'S ERROR. ``pp_room_vote.estimate_payable`` pays an un-backed write_back node by a
backup "into the host room still free" -- and ``follower_fact`` read that room as
``mem_pool_host.available_size()``. On an arena-bound pool that is ``(slots - claimed) * P``
(``ArenaMHAHostPool.available_size``, #1440b: COMPLETE slots count as free there, for the
front's read gate). With 541 complete slots it says 34624 tokens of backup room while every
claim is refused; the walk backs up [620] and [576], pays both chains: payable 195968, room
200064 > 16321. The peel pays what the claim pays: nothing.

THE RULE. On an arena-bound KV host pool the walk's backup room is the arena's FREE slots
(``slots - complete - claimed``) times the slot's tokens -- a claim that needs no room-making.
An un-backed node beyond that room is payable only by the UD drop, the same test the peel
applies. Switch ``SGLANG_WEG2_ENABLE_PP_ROOM_ARENA_FREE_ROOM`` (default on); off = the
``available_size()`` reading, byte for byte. A pool without a bound arena reads as before.
"""

import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache import common  # noqa: E402
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup  # noqa: E402
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402
from sglang.srt.weg2 import pp_room_vote as PR  # noqa: E402

PAGE = 64                  # 'pages=64' per 4096-token node
SLOTS = 580                # 'ARENA-FREE ... slots_total=580'
CLAIMED = 39               # 'ARENA-FREE reason=claim_refused slots=39'
COMPLETE = SLOTS - CLAIMED  # no free slot: 'statuses=[0, 4] (4 = no free slot)'
AVAIL_PP1 = 12288 - 8192   # 'avail 12288->12288' after the 8192 the first peel paid
REPORTED = 195968          # 'full_evictable_size_=195968'
CHUNK = 16321              # 'Try to allocate 16321 tokens'


class _Arena:
    def __init__(self, slots, complete, claimed):
        self.slots, self.complete, self.claimed = slots, complete, claimed

    def stats(self):
        return {"slots": self.slots, "complete": self.complete, "claimed": self.claimed,
                "slot_bytes": 786432, "file_bytes": 0}


def _arena_group(slots=SLOTS, complete=COMPLETE, claimed=CLAIMED):
    """The NF KV host pool: a HostPoolGroup whose anchor is the arena pool. Its
    available_size() is the production one (ArenaMHAHostPool.available_size)."""
    inner = types.SimpleNamespace(arena=_Arena(slots, complete, claimed), arena_read=True,
                                  _arena_page_tokens=PAGE, free_slots=torch.empty(0))
    inner.available_size = types.MethodType(ArenaMHAHostPool.available_size, inner)
    group = HostPoolGroup.__new__(HostPoolGroup)
    object.__setattr__(group, "anchor_entry", types.SimpleNamespace(host_pool=inner))
    return group


class _CD:
    def __init__(self, n=0, lock=0, host_lock=0):
        self.value = torch.arange(n) if n else None
        self.lock_ref = lock
        self.host_lock_ref = host_lock


class _Node:
    def __init__(self, parent, *, n=0, backuped=False, evicted=False, lock=0, host_lock=0):
        self.id = id(self)
        self.parent = parent
        self.children = {}
        self.backuped = backuped
        self.evicted = evicted
        self.component_data = [_CD(n, lock, host_lock), _CD(), _CD()]
        if parent is not None:
            parent.children[self.id] = self


def _chain(parent, sizes, backed=True):
    for n in sizes:
        parent = _Node(parent, n=n, backuped=backed)
    return parent


def _host_tail(parent, k):
    """Host-only children a peel cannot clear (host-locked)."""
    for _ in range(k):
        parent = _Node(parent, backuped=True, evicted=True, host_lock=1)


def _death_tree():
    """PP1 at 18:34:20 (P.log 217639-217722), unlocked device part = 195968."""
    root = _Node(None)
    locked = root
    for n in (4096, 640, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 16384):
        locked = _Node(locked, n=n, lock=1)                    # [614]..[626] weg2-45-272
    a = _chain(root, [4096] * 19)                              # [594]..[612], backed
    a = _Node(a, n=4096, backuped=False)                       # [609] arena_claim refused
    leaf_a = _Node(a, n=4096, backuped=False)                  # [620] parent_unbacked
    _host_tail(leaf_a, 2)                                      # [621], [619]
    b = _chain(root, [3392, 81984])                            # [568], [569]
    _host_tail(b, 4)                                           # [564]..[567]
    b = _chain(b, [4096] * 5, backed=False)                    # [571]..[575]
    leaf_b = _Node(b, n=4096, backuped=False)                  # [576] arena_claim refused
    _host_tail(leaf_b, 8)                                      # [577]..[583]
    _host_tail(root, 28)                                       # [535]..[562]
    return root


def _tree(root, pool):
    return types.SimpleNamespace(
        root_node=root, ongoing_write_through={},
        cache_controller=types.SimpleNamespace(write_policy="write_back", mem_pool_host=pool),
        evictable_size=lambda: REPORTED,
    )


def _fact(tree, avail=AVAIL_PP1):
    alloc = types.SimpleNamespace(available_size=lambda: avail)
    return PR.follower_fact(tree=tree, allocator=alloc, rank=1, executed=99, local_floor=True)


class DeathTreeTest(unittest.TestCase):
    def test_fixture_matches_the_dump(self):
        def dev(n):
            v = n.component_data[0].value
            own = len(v) if (v is not None and n.component_data[0].lock_ref == 0) else 0
            return own + sum(dev(c) for c in n.children.values())

        self.assertEqual(dev(_death_tree()), REPORTED)
        # the production reading the walk used: complete slots count as free
        self.assertEqual(_arena_group().available_size(), COMPLETE * PAGE)

    def test_18_34_20_pp1_fact_pays_only_what_the_peel_pays(self):
        """RED on 6b3bd1a6df: payable=195968, room=200064 -> PP0 admits 16321.
        (The default: no switch set.)"""
        fact = _fact(_tree(_death_tree(), _arena_group()))
        self.assertEqual(fact.payable, 0)
        self.assertEqual(fact.room, AVAIL_PP1)

    def test_18_34_16_pp0_no_longer_launches_16321(self):
        fact = _fact(_tree(_death_tree(), _arena_group()))
        book = PR.RoomBook()
        book.begin_pass()
        book.absorb([fact])
        verdict = book.cap(own_room=262144)
        self.assertLess(verdict.cap, CHUNK)
        self.assertEqual(verdict.rank, 1)
        # PP0's admission reads R_m through the one budget reader
        pp0 = types.SimpleNamespace(
            token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: 200000),
            uniform_avail_floor=None, evictable_size=lambda: 60000,
        )
        setattr(pp0, common.PP_ROOM_CAP_ATTR, verdict.cap)
        self.assertLess(common.fundable_extend_tokens(pp0), CHUNK)


class ArenaRoomTest(unittest.TestCase):
    def test_free_slots_still_pay_a_backup(self):
        """Free arena slots are claim room without room-making: the walk pays the
        backups they cover. 8 un-backed nodes x 4096 (the walk spends the room
        on every un-backed node it meets, UD-droppable ones included) -> both
        leaves back up and both chains pay."""
        free = 8 * 4096 // PAGE
        pool = _arena_group(complete=COMPLETE - free)
        with envs.SGLANG_WEG2_ENABLE_PP_ROOM_ARENA_FREE_ROOM.override(True):
            self.assertEqual(PR.backup_room_tokens(pool), 8 * 4096)
            fact = _fact(_tree(_death_tree(), pool))
        self.assertEqual(fact.payable, REPORTED)

    def test_switch_off_is_the_old_reading(self):
        pool = _arena_group()
        with envs.SGLANG_WEG2_ENABLE_PP_ROOM_ARENA_FREE_ROOM.override(False):
            self.assertEqual(PR.backup_room_tokens(pool), COMPLETE * PAGE)

    def test_a_pool_without_an_arena_reads_as_before(self):
        plain = types.SimpleNamespace(available_size=lambda: 777)
        unbound = types.SimpleNamespace(available_size=lambda: 555, arena_read=True, arena=None)
        with envs.SGLANG_WEG2_ENABLE_PP_ROOM_ARENA_FREE_ROOM.override(True):
            self.assertEqual(PR.backup_room_tokens(plain), 777)
            self.assertEqual(PR.backup_room_tokens(unbound), 555)
            self.assertEqual(PR.backup_room_tokens(None), 0)

    def test_an_unreadable_arena_is_no_room(self):
        class _Broken(_Arena):
            def stats(self):
                raise OSError("unmapped")

        pool = _arena_group()
        pool.anchor_entry.host_pool.arena = _Broken(0, 0, 0)
        with envs.SGLANG_WEG2_ENABLE_PP_ROOM_ARENA_FREE_ROOM.override(True):
            self.assertEqual(PR.backup_room_tokens(pool), 0)


if __name__ == "__main__":
    unittest.main()
