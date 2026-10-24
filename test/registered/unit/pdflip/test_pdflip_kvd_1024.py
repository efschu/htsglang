"""KVD-1024 (front KV class D; SOLL v2 R7, R11; follow-up of front-kvoffset-1024 section 1.D).

A displaced decode was priced by its SEAT length on both sides -- the front's AGE PLAN (``Seat.tokens``) and D's own
``kv_displace_would_fit`` (``_req_kv_tokens``). Its leaving frees only the rows nobody else holds: the shared system
prompt other running decodes keep locked stays on the device.

Metal specimen: 27B int4ar ``_1024_003732`` (/spinning/docker-acceptance/27b/evidence), 18-188 / 18-189:
  front 01:02:43,727  SEAT-AGE-PLAN head=pdflip-18-188 plan=displace deficit=38313 victims=['pdflip-18-189']
                      (need 104476 > free 66163; victim seat 54133)
  D     01:02:45      KV-DISPLACE-VERDICT need=104478 free=64346 basis=adder -> 1 youngest must leave
                      (victim seat 54190 = FINISH-TRACK tokens, reserve 401: offset 1605 -> 1204)
                      DISPLACE pages_out=34816 (the #248 keep window, NOT a freed count)
                      budget 64355.368 -> 100799.776 (+36444 = 36043 rows + 401 reserve), price 104477: 181 refusals
  the 18137 rows that stayed: the shared system prompt (19-206 / 19-208 ``cached_device=18137``)

Danger directions, one test group each:
  * Uebergabe trotz vollem KV -- a (the bound on the specimen = what the release frees), b (D's verdict: nobody
                                 leaves), c (front AGE PLAN: plan None), d (front KV park #246: no victim), e (D
                                 publishes the map), f (aux-locked child blocks), g (evictable already / not on
                                 device are not credited), h (broken chain: 0)
  * Verhungern                -- v1 (an exclusive path gives its whole seat), v2 (unreadable tree / no tree path:
                                 the seat, as before), v3 (an old D without the map: the seat), v4 (the victim's own
                                 aux lock blocks nothing), v5 (enough guaranteed release: the victim leaves / is
                                 planned), v6 (the plain radix TreeNode shape reads too)
"""
from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import schedule_policy as sp  # noqa: E402
from flliper.srt.mem_cache.evict_frontier_census import AUX_LOCKED_ATTR  # noqa: E402
from flliper.srt.mem_cache.radix_cache import TreeNode  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedTreeNode  # noqa: E402
from flliper.srt.pdflip import arrival_seat_rule as asr  # noqa: E402
from flliper.srt.pdflip import d_lifetime_reading as dlr  # noqa: E402
from flliper.srt.pdflip import d_park_runtime as dpr  # noqa: E402
from flliper.srt.pdflip.front import Front  # noqa: E402

try:  # the red run (before the fix) has no module: the tests that need it fail by name, not the whole file
    from flliper.srt.pdflip import d_leaving_rows as dlv  # noqa: E402
except ImportError:  # pragma: no cover - red run only
    dlv = None

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
COMPONENTS = (ComponentType.FULL, ComponentType.MAMBA)

HEAD, VICTIM = "pdflip-18-188", "pdflip-18-189"
SHARED = 18137              # the shared system prompt (three other running decodes hold it)
SPAN = 54080                # the victim's tree span (#248 KEEP-ROLE span=54080)
SEAT = 54190                # FINISH-TRACK tokens=54190: 110 decode tail past the tree
RESERVE = 401               # offset 1605 -> 1204
PRICE, BUDGET = 104477, 64355
FRONT_SEAT, FRONT_DEFICIT = 54133, 38313


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", "0")
    for k in ("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "FLLIPER_PDFLIP_SEAT_ORDER_AFTER_FLIP",
              "FLLIPER_PDFLIP_SHORT_GATE_BY_AGE", "FLLIPER_PDFLIP_KV_FRESH_MS", "FLLIPER_PDFLIP_EXACT_CLASS",
              "FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_SEAT_AGE_KV_DISPLACE"):
        monkeypatch.delenv(k, raising=False)


# ------------------------------------------------------------------------------------- real tree nodes, real locks
def _unode(parent, rows, on_device=True, aux=0):
    """A real ``UnifiedTreeNode``: FULL value of ``rows`` rows (None = not on the device), MAMBA lock ``aux``."""
    n = UnifiedTreeNode(COMPONENTS)
    n.parent = parent
    n.key = list(range(rows))
    n.component_data[FULL].value = list(range(rows)) if on_device else None
    if aux:
        n.component_data[MAMBA].value = [0]
        n.component_data[MAMBA].lock_ref = aux
    return n


def _root():
    r = UnifiedTreeNode(COMPONENTS)
    r.key = []
    r.component_data[FULL].value = []
    r.component_data[FULL].lock_ref = 1          # the root is pinned (unified_radix_cache.py root init)
    return r


def _lock(node, root):
    """``FullComponent.acquire_component_lock`` (unified_cache_components/full_component.py): +1 on every device-on
    node up to the root."""
    cur = node
    while cur is not root:
        if cur.component_data[FULL].value is not None:
            cur.component_data[FULL].lock_ref += 1
        cur = cur.parent


def _release(node, root):
    """``FullComponent.release_component_lock``: -1 up to the root; a node at 1 -> 0 turns evictable. Returns the rows
    that turned evictable -- what the leaving really frees in the tree."""
    freed, cur = 0, node
    while cur is not root:
        cd = cur.component_data[FULL]
        if cd.value is not None:
            if cd.lock_ref == 1:
                freed += len(cd.value)
            cd.lock_ref -= 1
        cur = cur.parent
    return freed


def _tree(root, aux_nodes=(), evict=0):
    t = SimpleNamespace(root_node=root, page_size=1, uniform_avail_floor=None,
                        evictable_size=lambda: evict, supports_mamba=lambda: False)
    setattr(t, AUX_LOCKED_ATTR, {id(n): n for n in aux_nodes})
    return t


def _req(rid, seat, last_node=None, max_new=64000):
    return SimpleNamespace(rid=rid, origin_input_ids=[0] * seat, output_ids=[], last_node=last_node,
                           prefix_indices=[], sampling_params=SimpleNamespace(max_new_tokens=max_new),
                           mamba_pool_idx=0, finished=lambda: False)


def _specimen():
    """root -> P (18137, the shared prompt: the victim + three other decodes lock it) -> V (the victim's own 35943)."""
    root = _root()
    prompt = _unode(root, SHARED)
    own = _unode(prompt, SPAN - SHARED)
    for _ in range(3):
        _lock(prompt, root)                      # the three other running decodes (running=4)
    _lock(own, root)                             # the victim
    return root, prompt, own


# ======================================================================== Uebergabe trotz vollem KV
def test_component_indices_pin_the_tree_enum():
    assert (int(ComponentType.FULL), int(ComponentType.MAMBA)) == (dlv._FULL, dlv._MAMBA)


def test_a_the_bound_on_the_specimen_is_what_the_release_frees_not_the_seat():
    root, _prompt, own = _specimen()
    tree = _tree(root)
    victim = _req(VICTIM, SEAT, own)
    got = dlv.leaving_rows(tree, victim)
    assert got == SEAT - SHARED == 36053
    # the release itself (insert of the 110-row tail aside: it is the victim's own) frees exactly that
    assert _release(own, root) + (SEAT - SPAN) == got
    # metal: 36043 rows + 401 reserve = +36444 budget; the bound is within the decode drift of 3 running decodes
    assert abs(got - 36043) <= 16 and got < SEAT


def _verdict_sched(running, waiting, tree, pool):
    s = SimpleNamespace(waiting_queue=list(waiting), tree_cache=tree,
                        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: pool))
    return s


def _view(pool):
    return {"rid": HEAD, "price": PRICE, "budget": BUDGET, "reserve": {VICTIM: RESERVE}, "pool": pool, "pass": None}


def test_b_d_verdict_nobody_leaves_when_the_guaranteed_release_does_not_cover_the_older_one():
    """D 01:02:45: ``need=104478 free=64346 -> 1 youngest must leave`` on the seat; the guaranteed release 36053 + 401
    leaves 104478 - (64355 + 36454) = 3669 short -> nobody leaves (the older one waits for room, R7: no flip)."""
    root, _prompt, own = _specimen()
    tree = _tree(root)
    sched = _verdict_sched([], [_req(HEAD, 100380)], tree, pool=65447)
    victim = _req(VICTIM, SEAT, own)
    assert dpr.kv_displace_would_fit(sched, HEAD, [victim], view=_view(65447)) is False


def _front(box=None):
    f = Front("http://p", "http://d", "D", "kvd", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=12288)
    f.awake, f.state, f.admit_d = "D", "serving", True

    async def _kv():
        return (box or {}).get("kv")

    f._arrival_seat_kv_reading = _kv
    return f


class _S:
    def __init__(self, rid, tokens):
        self.rid, self.tokens = rid, tokens


def _age_front(now):
    f = _front()
    f.__dict__.setdefault("_asr_arrive", {}).update({HEAD: now - 30.0, VICTIM: now - 10.0})
    f._d_seats_live = [_S(VICTIM, FRONT_SEAT)]
    f.groups["D"].outstanding[VICTIM] = now
    return f


def _basis(leaving=None, wall=False):
    b = {"unread": False, "wall": wall, "running": {VICTIM}, "displaceable": {VICTIM},
         "reserve_by_rid": {VICTIM: RESERVE}}
    if leaving is not None:
        b["leaving_by_rid"] = leaving
    return b


def test_c_front_age_plan_prices_the_victim_by_its_guaranteed_release_18_188():
    """front 01:02:43,727 planned ``victims=['pdflip-18-189'] deficit=38313`` on the seat 54133; the guaranteed release
    36053 + 401 = 36454 < 38313: plan None -- 18-188 waits here for room, nobody is planned on for nothing."""
    now = time.time()
    f = _age_front(now)
    got = Front._arrival_seat_age_verdict(f, HEAD, now - 30.0, True, FRONT_DEFICIT,
                                          basis=_basis({VICTIM: SEAT - SHARED}))
    assert got is None


def test_d_front_kv_park_246_takes_no_victim_whose_guaranteed_release_does_not_cover():
    now = time.time()
    f = _age_front(now)
    parked = []

    async def _park(D, victim, reason, t, what, park_cooldown=0.0):
        parked.append(victim)
        return victim

    f._arrival_seat_park = _park
    got = asyncio.run(Front._arrival_seat_kv_displace(f, HEAD, now - 30.0, FRONT_DEFICIT, now,
                                                      basis=_basis({VICTIM: SEAT - SHARED})))
    assert got is None and parked == []


def test_e_d_publishes_the_guaranteed_rows_of_each_running_decode():
    root, _prompt, own = _specimen()
    tree = _tree(root)
    victim = _req(VICTIM, SEAT, own)
    sched = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 65447),
                            tree_cache=tree, running_batch=SimpleNamespace(reqs=[victim]),
                            new_token_ratio_tracker=SimpleNamespace(current=0.0), page_size=1,
                            is_mixed_chunk=False, chunked_req=None, last_batch=None,
                            chunked_commitment_ledger=None, uniform_budget_deficit=lambda: 0,
                            kv_session_offload=None)
    got = dlr.lifetime_reading(sched)
    assert got["leaving_rows_by_rid"] == {VICTIM: SEAT - SHARED}


def test_f_an_aux_locked_child_keeps_the_victims_rows_above_it_unpeelable():
    """``evict_frontier_census.blocked_tokens``: an aux (mamba) locked node with FULL lock 0 and its FULL-unlocked
    ancestors are never peeled -- once the victim's lock is gone its own nodes above such a child are among them."""
    root = _root()
    own_a = _unode(root, 1000)
    own_b = _unode(own_a, 500)
    pinned = _unode(own_a, 64, aux=1)            # a sibling branch under own_a, pinned by a write-through anchor
    _lock(own_b, root)
    tree = _tree(root, aux_nodes=[pinned])
    victim = _req(VICTIM, 1600, own_b)
    assert dlv.leaving_rows(tree, victim) == 1600 - 1000      # own_a stays (behind the pinned child), own_b + tail go


def test_g_rows_already_evictable_or_not_on_the_device_are_not_credited():
    # the bottom evicted segment (``acquire_component_lock`` skips it): the victim's last node holds no device rows
    root = _root()
    own = _unode(root, 100)
    evicted = _unode(own, 200, on_device=False)
    _lock(evicted, root)
    assert dlv.leaving_rows(_tree(root), _req(VICTIM, 400, evicted)) == 400 - 200
    # defensive (not reachable on the unified tree, whose lock walk covers every device-on ancestor): a path node
    # nobody locks is evictable already -- it is in the reading, its leaving adds nothing
    root = _root()
    loose = _unode(root, 300)
    own = _unode(loose, 100)
    own.component_data[FULL].lock_ref = 1
    assert dlv.leaving_rows(_tree(root), _req(VICTIM, 450, own)) == 450 - 300


def test_h_a_broken_parent_chain_guarantees_nothing():
    root = _root()
    orphan = _unode(None, 400)
    orphan.component_data[FULL].lock_ref = 1
    assert dlv.leaving_rows(_tree(root), _req(VICTIM, 400, orphan)) == 0


# ======================================================================== Verhungern
def test_v1_an_exclusive_path_gives_its_whole_seat():
    root = _root()
    a = _unode(root, 4000)
    b = _unode(a, 2000)
    _lock(b, root)
    assert dlv.leaving_rows(_tree(root), _req(VICTIM, 6100, b)) == 6100


def test_v2_unreadable_tree_or_no_tree_path_keeps_the_seat():
    from unittest.mock import MagicMock

    assert dlv.leaving_rows(MagicMock(), _req(VICTIM, 5000, MagicMock())) == 5000        # a class it cannot read
    assert dlv.leaving_rows(SimpleNamespace(), _req(VICTIM, 5000, None)) == 5000         # chunk cache: no tree path
    root = _root()
    assert dlv.leaving_rows(_tree(root), _req(VICTIM, 5000, root)) == 5000               # nothing inserted yet


def test_v3_an_old_d_without_the_map_is_planned_as_before():
    now = time.time()
    f = _age_front(now)
    assert Front._arrival_seat_age_verdict(f, HEAD, now - 30.0, True, FRONT_DEFICIT, basis=_basis()) == [VICTIM]


def test_v4_the_victims_own_aux_lock_blocks_nothing():
    root = _root()
    a = _unode(root, 1000)
    b = _unode(a, 500, aux=1)                    # its own mamba lock on its last node (inc_lock_ref on last_node)
    _lock(b, root)
    tree = _tree(root, aux_nodes=[b])
    assert dlv.leaving_rows(tree, _req(VICTIM, 1500, b)) == 1500
    assert dlv.leaving_rows_by_rid(tree, [_req(VICTIM, 1500, b)]) == {VICTIM: 1500}


def test_v5_enough_guaranteed_release_the_victim_is_planned_and_leaves():
    now = time.time()
    f = _age_front(now)
    covered = FRONT_DEFICIT - RESERVE
    assert Front._arrival_seat_age_verdict(f, HEAD, now - 30.0, True, FRONT_DEFICIT,
                                           basis=_basis({VICTIM: covered})) == [VICTIM]
    # D: an exclusive victim big enough -- it leaves
    root = _root()
    own = _unode(root, PRICE)
    _lock(own, root)
    sched = _verdict_sched([], [_req(HEAD, 100380)], _tree(root), pool=65447)
    assert dpr.kv_displace_would_fit(sched, HEAD, [_req(VICTIM, PRICE, own)], view=_view(65447)) is True


def test_v6_the_plain_radix_tree_node_reads_too():
    root = TreeNode()
    root.key, root.value, root.lock_ref = [], [], 1
    shared = TreeNode()
    shared.parent, shared.key, shared.value, shared.lock_ref = root, [0] * 700, [0] * 700, 2
    own = TreeNode()
    own.parent, own.key, own.value, own.lock_ref = shared, [0] * 300, [0] * 300, 1
    assert dlv.leaving_rows(_tree(root), _req(VICTIM, 1000, own)) == 300


def test_v7_victim_release_one_price_rows_capped_by_the_seat_wall_and_reserve():
    seat = {VICTIM: 1000, "other": 500}
    assert asr.victim_release(seat, {"leaving_by_rid": {VICTIM: 400}, "reserve_by_rid": {VICTIM: 7}}) == {
        VICTIM: 407, "other": 0}                 # a rid D's map does not name: nothing guaranteed
    assert asr.victim_release(seat, {"leaving_by_rid": {VICTIM: 4000}}) == {VICTIM: 1000, "other": 0}
    assert asr.victim_release(seat, {"wall": True, "leaving_by_rid": {VICTIM: 400},
                                     "reserve_by_rid": {VICTIM: 7}})[VICTIM] == 7
    assert asr.victim_release(seat, {}) == seat   # an old D: the seat
