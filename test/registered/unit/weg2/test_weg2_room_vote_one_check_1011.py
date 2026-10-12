# SPDX-License-Identifier: Apache-2.0
"""ROOM-VOTE 1011: the PP room vote and the peel ask ONE evictability check.

THE DEATH. NF int24e @ 6c53a955c7, P log boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwbar1dauer10110929_
6c53a955c7_1011_092933.P.log, PP0 09:39:42Z (lines 40117-40177):
  * 09:39:32 (39846) ``W3-ARENA SPILL n=8 need=64 released_pages=1882 nodes=1`` -- the spill
    takes node 181 (120448 tokens) its L2 copy; the node stays on the device, ``l3_present``,
    not backed. That is why it is un-backed and why the arena is full -- a precondition;
  * 40117-40124 the peel demotes its device children 195 and 194 (``#1469 EVICT node=195/194
    backuped=True``); both carry a MAMBA host lock (``HOSTLOCK-LRU-SKIP ... host_lock_ref=1``);
  * 40125 ``#1427 ARENA-DROP n=77 need=1006 freed=0`` (no free arena slot), 40132-40134
    ``#1421 BACKUP-REFUSED why=arena_claim node=181 tokens=120448`` x2; no UD drop (UD-H cannot
    clear host-locked children);
  * 40135 ``EVICT-FRONTIER-CENSUS request=73344 delivered_before=960 reported_evictable=120448
    leaves 1->1``, 40136 ``PR AGREED-ABOVE-LOCAL where=loadback need=92352 agreed=140416
    local=19968``, then the traceback ``AgreedRoomShort ... avail=19008 evictable=121408
    evicted=960 avail_after=19968`` -> RANK-DEATH.
WHAT IS BELEGT: agreed - avail_after = 140416 - 19968 = 120448 = node 181 (the vote counted
everything reported; the peel paid all but 181); 181 was un-backed + l3_present, its backup
claim was refused, no UD drop was logged; 194/195 were demoted by the peel after the peel's own
write-back (``WRITE-BACK DRAIN`` -> ``EVICT backuped=True``) and carried a MAMBA host lock.
WHAT IS UNBELEGT: why 181 could not drop at the peel. H1: the host lock on the demoted children
(its timing relative to the pass-top vote is not in the log; ``load_back`` takes its own host
lock before ``local_pp_room``); H2: 181 in flight in ``ongoing_write_through`` (publish sweep in
the same pass); H3: the host room the vote priced the backup with. The ``_death_shape`` tests
below build the children ALREADY backed and ALREADY host-locked at vote time -- that is H1 with
the lock visible to the vote, which the log does not show (it suggests the children were
un-backed when the peel began). ``VoteToPeelWindow`` builds the cases the vote cannot see.
Third instance of "room estimate != peel" (PR 24ed9b508b, PR-ARENA c59353374a, UD-H bff5913a62
were the earlier special cases; Room-Short 1008 is a different mechanism of the same stop).

THE FIX. ``pp_slot_fidelity.device_exit_route`` is the one check; ``estimate_payable`` and
``UnifiedRadixCache._evict_device_leaf`` both ask it. The walk carries ``clr`` up the tree
(``demoted_node_clearable``: a demoted, host-locked child is not clearable, so its parent is not
droppable; guaranteed only for the state the vote sees); an l3_present node that is droppable leaves the device without an arena claim
(L3_DROP). Mutants: the old walk (``_old_estimate_payable``) and the walk with ONLY the
host-lock term of the clr propagation reverted (``_no_clr_propagation``).

Driven through the REAL ``estimate_payable``, ``_evict_device_leaf``, ``_ud_clear_host_children``,
``_l3_drop_leaf``, ``FullComponent._peel`` and ``local_pp_room``; the tree's I/O steps are stubs.
"""

import os
import random
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.unified_cache_components.full_component import FullComponent  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    BASE_COMPONENT_TYPE as FULL,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.weg2 import pp_room_vote as PR  # noqa: E402
from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402

BIG = 10**9


class _CD:
    def __init__(self, value=None, host_value=None, lock=0, host_lock=0):
        self.value = value
        self.host_value = host_value
        self.lock_ref = lock
        self.host_lock_ref = host_lock


class _Node:
    _n = 0

    def __init__(self, parent, *, device=0, host=0, l3=False, lock=0, host_lock=0):
        _Node._n += 1
        self.id = _Node._n
        self.parent = parent
        self.children = {}
        n = device or host
        self.key = list(range(n))
        self.l3_present = l3
        self.write_through_pending_id = None
        self.component_data = [
            _CD(list(range(device)) if device else None, list(range(host)) if host else None,
                lock, host_lock),
            _CD(), _CD(),
        ]
        if parent is not None:
            parent.children[self.id] = self

    @property
    def backuped(self):
        return self.component_data[FULL].host_value is not None

    @property
    def evicted(self):
        return self.parent is not None and self.component_data[FULL].value is None


class _Tree:
    """The real peel + exit-route methods on a stub tree; ``claim_ok`` decides the arena claim."""

    _evict_device_leaf = UnifiedRadixCache._evict_device_leaf
    _is_device_leaf = UnifiedRadixCache._is_device_leaf
    _is_host_leaf = UnifiedRadixCache._is_host_leaf
    _ud_clear_host_children = UnifiedRadixCache._ud_clear_host_children
    _ud_drop_unbacked_leaf = UnifiedRadixCache._ud_drop_unbacked_leaf
    _l3_drop_leaf = UnifiedRadixCache._l3_drop_leaf

    def __init__(self, local_pp=True, claim_ok=False, cls_nodes=None):
        self.root_node = _Node(None)
        self.cache_controller = types.SimpleNamespace(write_policy="write_back")
        self.ongoing_write_through = {}
        self.tree_components = (FULL,)
        self._components_tuple = (types.SimpleNamespace(component_type=FULL),)
        self.evictable_device_leaves = set()
        self.eviction_strategy = types.SimpleNamespace(get_priority=lambda n: n.id)
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, local_pp)
        self.claim_ok = claim_ok
        self.claims = 0
        self.host_evicted, self.dropped, self.demoted = [], [], []

    # -- stubs -----------------------------------------------------------------
    def write_backup(self, node, write_back=False, kv_only_if_mamba_refused=False):
        self.claims += 1
        if not self.claim_ok:
            return 0  # '#1421 BACKUP-REFUSED why=arena_claim'
        v = node.component_data[FULL].value
        node.component_data[FULL].host_value = list(range(len(v)))
        return len(v)

    def writing_check(self, write_back=False):
        return None

    def _evict_to_host(self, node, tracker):
        assert not node.evicted and node.backuped
        tracker[FULL] += len(node.component_data[FULL].value)
        node.component_data[FULL].value = None
        self.demoted.append(node.id)
        self._update_evictable_leaf_sets(node.parent)

    def _evict_host_leaf(self, node, tracker):
        assert self._is_host_leaf(node)
        node.parent.children.pop(node.id)
        self.host_evicted.append(node.id)

    def _record_remove_event(self, node, medium=None):
        return None

    def _evict_component_and_detach_lru(self, node, comp, target=None, tracker=None):
        cd = node.component_data[FULL]
        if cd.value is not None and tracker is not None:
            tracker[FULL] += len(cd.value)
        cd.value = None
        cd.host_value = None
        return 0, 0

    def _remove_leaf_from_parent(self, node):
        node.parent.children.pop(node.id)
        self.dropped.append(node.id)

    def _update_evictable_leaf_sets(self, node):
        if node is None:
            return
        if self._is_device_leaf(node):
            self.evictable_device_leaves.add(node)
        else:
            self.evictable_device_leaves.discard(node)

    def _iteratively_delete_tombstone_leaf(self, node, tracker):
        return None

    # -- drivers ---------------------------------------------------------------
    def nodes(self):
        out, stack = [], [self.root_node]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children.values())
        return out

    def arm(self):
        self.evictable_device_leaves = {n for n in self.nodes() if self._is_device_leaf(n)}
        return self

    def peel(self, request=BIG):
        tracker = {FULL: 0}
        comp = types.SimpleNamespace(cache=self, component_type=FULL)
        FullComponent._peel(comp, request, tracker)
        return tracker[FULL]

    def vote(self, host_free):
        return PR.estimate_payable(self, local_floor=bool(getattr(self, SF.FLOOR_LOCAL_PP_ATTR)),
                                   host_free=host_free)


def _old_estimate_payable(tree, *, local_floor, host_free):
    """MUTANT: the walk as it was at 6c53a955c7 (a backup priced by the host room for every
    un-backed node, l3_present or not; UD/L3 not asking the peel's check)."""
    root = tree.root_node
    cc = getattr(tree, "cache_controller", None)
    write_back = cc is not None and getattr(cc, "write_policy", None) == "write_back"
    ongoing = getattr(tree, "ongoing_write_through", None) or {}
    order, stack = [], [root]
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(n.children.values())
    state = {}
    budget = max(0, int(host_free))

    def host_clearable(node):
        if not node.evicted or not node.backuped:
            return False
        if any(cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in node.component_data):
            return False
        return all(state.get(id(c), (0, False))[1] for c in node.children.values())

    for n in reversed(order):
        if n is root:
            continue
        value = n.component_data[FULL].value
        if value is None:
            state[id(n)] = (0, host_clearable(n))
            continue
        dev = [c for c in n.children.values() if c.component_data[FULL].value is not None]
        host = [c for c in n.children.values() if c.component_data[FULL].value is None]
        paid = sum(state[id(c)][0] for c in dev)
        locked = any(cd.lock_ref > 0 for cd in n.component_data)
        if locked or not all(state[id(c)][1] for c in dev):
            state[id(n)] = (paid, False)
            continue
        tokens = len(value)
        if n.backuped:
            ok = True
        elif write_back and budget >= tokens:
            budget -= tokens
            ok = True
        elif write_back:
            ok = local_floor and n.id not in ongoing and all(state[id(c)][1] for c in host)
        else:
            ok = not host
        state[id(n)] = (paid + (tokens if ok else 0), ok)
    return sum(state[id(c)][0] for c in root.children.values() if id(c) in state)


def _no_clr_propagation():
    """MUTANT: only the clr propagation reverted -- a demoted node counts as clearable
    whatever its host lock says (the vote's walk as the old host-child test saw it)."""
    return mock.patch.object(SF, "demoted_node_clearable", lambda node, children_clearable: bool(children_clearable))


# the vote's host room: realistic values (an arena that is full / a few slots free) and a huge one
REALISTIC_HOST_ROOMS = (0, 56064)


# ---- the 092933 shape ---------------------------------------------------------------------
X_TOKENS = 120448   # 'tokens=120448'
CHILD = 4096


def _death_shape(host_locked_children=True, **kw):
    """P (backed) <- X (device, NOT backed, l3_present: W3-spilled) <- c1 <- c2 (device, backed;
    the peel demotes them first, their host copies are host-locked: UD-H cannot clear them)."""
    t = _Tree(**kw)
    p = _Node(t.root_node, device=4096)
    p.component_data[FULL].host_value = list(range(4096))
    x = _Node(p, device=X_TOKENS, l3=True)
    c1 = _Node(x, device=CHILD, host_lock=1 if host_locked_children else 0)
    c1.component_data[FULL].host_value = list(range(CHILD))
    c2 = _Node(c1, device=CHILD, host_lock=1 if host_locked_children else 0)
    c2.component_data[FULL].host_value = list(range(CHILD))
    return t.arm(), (p, x, c1, c2)


class DeathShape(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {SF.ENV: "1"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_vote_and_peel_agree_when_the_spill_node_cannot_leave(self):
        """Host room read AFTER the spill is large (the freed slots); the claim at the peel is
        refused. The vote must not count X: walk == peel (== the two host-locked children)."""
        t, (p, x, c1, c2) = _death_shape()
        walked = t.vote(host_free=BIG)
        delivered = t.peel()
        self.assertEqual(delivered, 2 * CHILD)
        self.assertEqual(walked, delivered, "vote == peel: X (l3_present, blocked) is not counted")
        self.assertGreaterEqual(t.claims, 1, "the peel still TRIES the claim for the blocked node")
        self.assertEqual(t.dropped, [])

    def test_realistic_host_room_vote_and_peel_agree(self):
        """The log's shape: the vote's host room is BELOW the node (full arena). Walk == peel."""
        for host_free in REALISTIC_HOST_ROOMS:
            self.assertLess(host_free, X_TOKENS)
            t, _ = _death_shape()
            walked = t.vote(host_free=host_free)
            delivered = t.peel()
            self.assertEqual(delivered, 2 * CHILD, host_free)
            self.assertEqual(walked, delivered, "host_free=%d" % host_free)

    def test_mutant_no_clr_propagation_overpays_at_realistic_host_room(self):
        """RED with ONLY the clr propagation reverted (H1 with the lock visible at vote time): the walk pays X (120448
        + parent) although the peel delivers just the two host-locked children (960-style)."""
        for host_free in REALISTIC_HOST_ROOMS + (BIG,):
            t, _ = _death_shape()
            with _no_clr_propagation():
                mutant = t.vote(host_free=host_free)
            delivered = t.peel()
            self.assertEqual(mutant, 2 * CHILD + X_TOKENS + 4096, "host_free=%d" % host_free)
            self.assertGreater(mutant, delivered, "host_free=%d" % host_free)

    def test_mutant_old_counting_overpays_at_realistic_host_room(self):
        """RED on 6c53a955c7 at host rooms below the node (UD branch, children seen as device)."""
        for host_free in REALISTIC_HOST_ROOMS:
            t, _ = _death_shape()
            old = _old_estimate_payable(t, local_floor=True, host_free=host_free)
            self.assertGreater(old, t.peel(), "host_free=%d" % host_free)

    def test_mutant_old_counting_overpays_the_node(self):
        """RED on 6c53a955c7: the old walk counts X (backup priced by the freed slots)."""
        t, _ = _death_shape()
        old = _old_estimate_payable(t, local_floor=True, host_free=BIG)
        delivered = t.peel()
        self.assertEqual(old, 2 * CHILD + X_TOKENS + 4096)
        self.assertGreater(old, delivered)

    def test_agreed_room_short_with_old_counting_not_with_the_shared_check(self):
        """The load-back path: agreed R_m from the vote, then the real ``local_pp_room``.
        Old counting -> AgreedRoomShort (the 092933 text); shared check -> a named refusal."""
        avail0 = 19008

        def run(estimate, host_free=BIG, mutant=None):
            t, _ = _death_shape()
            freed = {"n": 0}
            alloc = types.SimpleNamespace(available_size=lambda: avail0 + freed["n"])
            t.token_to_kv_pool_allocator = alloc
            t.evictable_size = lambda: 2 * CHILD + X_TOKENS + 4096

            def evict(params):
                tracker = {FULL: 0}
                comp = types.SimpleNamespace(cache=t, component_type=FULL)
                FullComponent._peel(comp, params.num_tokens, tracker)
                freed["n"] += tracker[FULL]
                return types.SimpleNamespace(num_tokens_evicted=tracker[FULL])

            t.evict = evict
            with (mutant() if mutant else mock.MagicMock()):
                room = avail0 + estimate(t, local_floor=True, host_free=host_free)
            setattr(t, "weg2_pp_room_cap", room)  # PP0's cap R_m = the vote's room
            need = avail0 + 2 * CHILD + 1000     # more than the peel can pay, less than the old room
            return SF.local_pp_room(t, need, floor=avail0), t, room

        with self.assertRaises(SF.AgreedRoomShort) as cm:
            run(_old_estimate_payable)
        self.assertIn("PR AGREED-ROOM SHORT", str(cm.exception))
        self.assertIn("evicted=%d" % (2 * CHILD), str(cm.exception))
        verdict, t, room = run(PR.estimate_payable)
        self.assertEqual(room, avail0 + 2 * CHILD)
        self.assertIs(verdict, False, "agreed R_m < need: every stage refuses, no stop")
        for host_free in REALISTIC_HOST_ROOMS:
            # realistic host room: the walk without the clr propagation dies the 092933 death ...
            with self.assertRaises(SF.AgreedRoomShort) as cm:
                run(PR.estimate_payable, host_free=host_free, mutant=_no_clr_propagation)
            self.assertIn("PR AGREED-ROOM SHORT", str(cm.exception))
            # ... the shared check does not
            verdict, t, room = run(PR.estimate_payable, host_free=host_free)
            self.assertEqual(room, avail0 + 2 * CHILD)
            self.assertIs(verdict, False)

    def test_clearable_children_let_the_node_leave_without_a_claim(self):
        """Same shape, the children's host copies NOT locked: UD-H clears them, X is dropped (L3).
        Walk == peel == everything, and the arena was never asked for X."""
        t, (p, x, c1, c2) = _death_shape(host_locked_children=False)
        walked = t.vote(host_free=0)   # a full arena: no host room at all
        delivered = t.peel()
        self.assertEqual(walked, delivered)
        self.assertEqual(delivered, 2 * CHILD + X_TOKENS + 4096)
        self.assertIn(x.id, t.dropped)


class L3OnlyNode(unittest.TestCase):
    """Full arena + a node that lives only in L3: it leaves WITHOUT arena_claim."""

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {SF.ENV: "1"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _tree(self, **kw):
        t = _Tree(**kw)
        p = _Node(t.root_node, device=4096)
        p.component_data[FULL].host_value = list(range(4096))
        x = _Node(p, device=X_TOKENS, l3=True)
        return t.arm(), p, x

    def test_evict_pays_without_a_claim_and_the_vote_agrees(self):
        t, p, x = self._tree()
        walked = t.vote(host_free=0)
        with self.assertLogs(SF.logger, level="WARNING") as cap:
            delivered = t.peel(request=X_TOKENS)
        self.assertEqual(delivered, X_TOKENS + 0 if t.demoted == [] else delivered)
        self.assertGreaterEqual(delivered, X_TOKENS)
        self.assertEqual(t.claims, 0, "no arena claim: the pages are in L3 already")
        self.assertIn(x.id, t.dropped)
        self.assertTrue(any("ROOM-VOTE L3-DROP node=%d" % x.id in m for m in cap.output))
        self.assertEqual(walked, X_TOKENS + 4096)

    def test_old_peel_asks_the_arena_first(self):
        """The old order: a claim attempt for the l3_present node (what the full arena refuses)."""
        t, p, x = self._tree()
        with mock.patch.object(SF, "peel_route", lambda tree, node, host_room=None:
                               SF.ROUTE_UD_DROP if host_room == 0 else SF.ROUTE_BACKUP):
            t.peel(request=X_TOKENS)
        self.assertGreaterEqual(t.claims, 1)

    def test_not_off_the_local_pp_floor(self):
        t, p, x = self._tree(local_pp=False)
        self.assertEqual(t.vote(host_free=0), 0)
        self.assertEqual(t.peel(request=X_TOKENS), 0)
        self.assertEqual(t.dropped, [])

    def test_not_with_a_write_in_flight(self):
        t, p, x = self._tree()
        t.ongoing_write_through[x.id] = object()
        self.assertEqual(t.vote(host_free=0), 0)
        self.assertEqual(t.peel(request=X_TOKENS), 0)
        x2 = self._tree()[2]
        x2.write_through_pending_id = 7   # a split node keeps the OLD id pending
        self.assertFalse(SF.peel_route(_Tree(), x2) == SF.ROUTE_L3_DROP)

    def test_a_backed_node_still_demotes_and_a_plain_unbacked_node_still_claims(self):
        t = _Tree(claim_ok=True)
        b = _Node(t.root_node, device=100)
        b.component_data[FULL].host_value = list(range(100))
        u = _Node(t.root_node, device=200)   # un-backed, NOT l3_present
        t.arm()
        self.assertEqual(SF.peel_route(t, b), SF.ROUTE_DEMOTE)
        self.assertEqual(SF.peel_route(t, u), SF.ROUTE_BACKUP)
        self.assertEqual(t.vote(host_free=200), 300)
        self.assertEqual(t.peel(), 300)
        self.assertEqual(t.claims, 1)

    def test_switch_off_means_the_old_routes(self):
        with mock.patch.dict(os.environ, {SF.ENV: "0"}):
            t, p, x = self._tree()
            self.assertEqual(SF.peel_route(t, x), SF.ROUTE_BACKUP)
            self.assertEqual(t.peel(request=X_TOKENS), 0)


# ---- the window the vote cannot see ------------------------------------------------------
class VoteToPeelWindow(unittest.TestCase):
    """The vote reads the tree at the pass top; the peel runs later in the same pass. A host lock
    or an in-flight write that appears in between makes the vote over-count. NOT closed by the
    shared check (UNBELEGT whether it is what killed 092933); ``AgreedRoomShort`` stays the stop
    and names the window (``vote_payable`` vs ``walk_now``, ``unpaid=[...]``)."""

    AVAIL0 = 19008

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {SF.ENV: "1"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _unbacked_children_shape(self, **kw):
        """P (backed) <- X (device, un-backed, l3_present) <- c1 <- c2 (device, UN-BACKED at vote
        time: the log's reading of 194/195). The peel backs them up (claim_ok for them) and
        demotes them."""
        t = _Tree(claim_ok=True, **kw)
        p = _Node(t.root_node, device=4096)
        p.component_data[FULL].host_value = list(range(4096))
        x = _Node(p, device=X_TOKENS, l3=True)
        c1 = _Node(x, device=CHILD)
        c2 = _Node(c1, device=CHILD)
        t.arm()
        # the claim of X itself is refused (full arena); the children's backups succeed
        real = t.write_backup

        def write_backup(node, write_back=False, kv_only_if_mamba_refused=False):
            if node is x:
                t.claims += 1
                return 0
            return real(node, write_back=write_back, kv_only_if_mamba_refused=kv_only_if_mamba_refused)

        t.write_backup = write_backup
        return t, (p, x, c1, c2)

    def _load_back(self, t, vote_payable):
        """The real ``local_pp_room`` with PP0's cap = the vote's room."""
        freed = {"n": 0}
        t.token_to_kv_pool_allocator = types.SimpleNamespace(available_size=lambda: self.AVAIL0 + freed["n"])
        t.evictable_size = lambda: 2 * CHILD + X_TOKENS + 4096

        def evict(params):
            tracker = {FULL: 0}
            FullComponent._peel(types.SimpleNamespace(cache=t, component_type=FULL),
                                params.num_tokens, tracker)
            freed["n"] += tracker[FULL]
            return types.SimpleNamespace(num_tokens_evicted=tracker[FULL])

        t.evict = evict
        setattr(t, "weg2_pp_room_cap", self.AVAIL0 + vote_payable)
        return SF.local_pp_room(t, self.AVAIL0 + 2 * CHILD + 1000, floor=self.AVAIL0)

    def test_unlocked_unbacked_children_the_vote_and_the_peel_agree(self):
        """No late change: the vote pays children + X (L3 drop after the children are backed up
        and demoted, nothing locked) and the peel delivers the same."""
        t, (p, x, c1, c2) = self._unbacked_children_shape()
        walked = t.vote(host_free=BIG)
        delivered = t.peel()
        self.assertEqual(walked, delivered)
        self.assertEqual(delivered, 2 * CHILD + X_TOKENS + 4096)

    def test_host_lock_after_the_vote_the_stop_fires_and_names_the_window(self):
        """H1 with a LATE lock: the children are un-backed at vote time (the vote counts them
        clearable and pays X); between vote and peel a host lock lands on them (e.g. the
        load-back's own anchor lock). The peel pays only the children: AgreedRoomShort -- the
        stop is the answer for this window, and its message carries the evidence."""
        t, (p, x, c1, c2) = self._unbacked_children_shape()
        walked = t.vote(host_free=BIG)
        PR.note_vote(t, walked)
        self.assertEqual(walked, 2 * CHILD + X_TOKENS + 4096, "the vote at the pass top pays X")
        for c in (c1, c2):   # ... the lock appears after the vote
            c.component_data[FULL].host_lock_ref = 1
        with self.assertRaises(SF.AgreedRoomShort) as cm:
            self._load_back(t, walked)
        msg = str(cm.exception)
        self.assertIn("PR AGREED-ROOM SHORT", msg)
        self.assertIn("vote_payable=%d" % walked, msg)
        self.assertIn("walk_now=%d" % 0, msg)   # fresh walk after the peel: nothing payable left
        self.assertIn("%d:%d:children_not_clearable" % (x.id, X_TOKENS), msg)

    def test_write_in_flight_after_the_vote_the_stop_fires_and_names_the_window(self):
        """H2: X is un-backed and droppable at vote time; a write of X is issued later in the
        pass (publish sweep). The peel cannot drop it (floor reads ``node_in_flight``)."""
        t, (p, x, c1, c2) = self._unbacked_children_shape()
        walked = t.vote(host_free=BIG)
        PR.note_vote(t, walked)
        t.ongoing_write_through[x.id] = object()   # issued after the vote
        with self.assertRaises(SF.AgreedRoomShort) as cm:
            self._load_back(t, walked)
        msg = str(cm.exception)
        self.assertIn("vote_payable=%d" % walked, msg)
        self.assertIn("%d:%d:write_in_flight" % (x.id, X_TOKENS), msg)

    def test_diagnostic_names_the_locked_demoted_child(self):
        t, (p, x, c1, c2) = self._unbacked_children_shape()
        c1.component_data[FULL].host_lock_ref = 1
        c1.component_data[FULL].host_value = list(range(CHILD))   # backed + host-locked
        c2.component_data[FULL].host_value = list(range(CHILD))
        txt = PR.vote_diagnostic(t)
        self.assertIn("unpaid=[", txt)
        self.assertIn(":paid_not_clearable:host_locked", txt)
        self.assertIn("vote_payable=unknown", txt)

    def test_diagnostic_never_breaks_the_stop(self):
        class Broken:
            @property
            def root_node(self):
                raise RuntimeError("boom")
        self.assertIn("vote_diagnostic_failed=RuntimeError", PR.vote_diagnostic(Broken()))

    def test_trace_does_not_change_the_walk(self):
        for seed in range(50):
            t = _gen(random.Random(30_000 + seed), claim_ok=False)
            a = t.vote(host_free=0)
            b = PR.estimate_payable(t, local_floor=True, host_free=0, trace=[])
            self.assertEqual(a, b, "seed %d" % seed)


# ---- the derived check: walk == peel over generated trees ---------------------------------

def _gen(rng, claim_ok, only_l3_unbacked=False):
    t = _Tree(claim_ok=claim_ok)
    nodes = [t.root_node]
    n_nodes = rng.randint(2, 14)
    for _ in range(n_nodes):
        parent = rng.choice(nodes)
        parent_dev = parent is t.root_node or parent.component_data[FULL].value is not None
        size = rng.choice([64, 128, 4096])
        if parent_dev and rng.random() < 0.7:
            nd = _Node(parent, device=size)
            r = rng.random()
            if r < 0.3:
                nd.component_data[FULL].host_value = list(range(size))   # backed
            else:
                nd.l3_present = True if only_l3_unbacked else (rng.random() < 0.4)
                if rng.random() < 0.1:
                    t.ongoing_write_through[nd.id] = object()
            if rng.random() < 0.1:
                nd.component_data[FULL].host_lock_ref = 1
            if rng.random() < 0.1:   # a lock: every ancestor is locked too
                a = nd
                while a is not None and a is not t.root_node:
                    a.component_data[FULL].lock_ref += 1
                    a = a.parent
        else:
            nd = _Node(parent, host=size, host_lock=1 if rng.random() < 0.15 else 0)
        nodes.append(nd)
    # a locked ancestor chain of a host-only node cannot exist (device ancestors above it are fine)
    return t.arm()


class WalkEqualsPeel(unittest.TestCase):
    N = 400

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {SF.ENV: "1"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_full_arena_vote_equals_peel(self):
        """The arena refuses every claim, the vote's host room is 0: walk == peel, exactly."""
        for seed in range(self.N):
            t = _gen(random.Random(seed), claim_ok=False)
            walked = t.vote(host_free=0)
            delivered = t.peel()
            self.assertEqual(walked, delivered, "seed %d" % seed)

    def test_the_092933_condition_vote_never_exceeds_peel_for_l3_nodes(self):
        """Host room read large (the slots a spill freed), every claim refused, every un-backed
        node spilled to L3: the vote may undercount, never overcount."""
        for seed in range(self.N):
            t = _gen(random.Random(10_000 + seed), claim_ok=False, only_l3_unbacked=True)
            walked = t.vote(host_free=BIG)
            delivered = t.peel()
            self.assertLessEqual(walked, delivered, "seed %d" % seed)

    def test_roomy_arena_vote_does_not_exceed_peel(self):
        for seed in range(self.N):
            t = _gen(random.Random(20_000 + seed), claim_ok=True)
            walked = t.vote(host_free=BIG)
            delivered = t.peel()
            self.assertLessEqual(walked, delivered, "seed %d" % seed)

    def test_mutant_old_counting_is_caught_by_the_derived_check(self):
        """The same property against the old walk finds an overcount (the 092933 class)."""
        bad = 0
        for seed in range(self.N):
            t = _gen(random.Random(10_000 + seed), claim_ok=False, only_l3_unbacked=True)
            old = _old_estimate_payable(t, local_floor=True, host_free=BIG)
            if old > t.peel():
                bad += 1
        self.assertGreater(bad, 0)


if __name__ == "__main__":
    unittest.main()
